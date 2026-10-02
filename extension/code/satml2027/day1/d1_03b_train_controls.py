#!/usr/bin/env python3
"""Day 1 - train the registered control banks for one cell.

Caches the control-training split at this cell's sigma (clean features plus
`--train-draws` noisy draws per image), then fits two supervised controls on the
cached features only:

  * the **learned shared translation**: one vector `v`, applied as
    `t'_k = normalize(t_k + v - <v,t_k> t_k)`.  This is the control that asks
    whether a shared direction *chosen by an optimizer with labels* can move a
    certificate, and it is the extension's counterpart of the Phase-4 result.
  * the **rank-8 tangent adapter**: `t'_k = normalize(t_k + tangent(LR)_k)`, the
    per-class positive control that is expected to leave the band.

Both minimise noisy + clean cross-entropy over the cached features, which is a
few seconds of work; the expensive part is the feature cache.  The training-free
noisy class-mean control needs no fitting and is built from the same cache by
`d1_03_build_banks.py`.

Usage:
  python day1/d1_03b_train_controls.py --model-id ... --dataset-id ... --sigma 0.5 \
      --items results/satml2027/items --data-root data \
      --development results/satml2027/development --out results/satml2027/controls
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.datasets import load_dataset, recover_pixels  # noqa: E402
from common.hashing import canonical_json_hash, read_json, tensor_scientific_hash, write_json_atomic  # noqa: E402
from common.models import load_model  # noqa: E402
from common.seeds import CONTROL_TRAIN_STREAM, NOISE_BLOCK, block_plan, derive_block_seed, derive_item_seed  # noqa: E402


def cache_features(args, loaded, prototypes):
    with (args.items / f"{args.dataset_id}__control_train.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: row["item_id"])
    device = loaded.device
    dataset, item_ids, labels = load_dataset(args.dataset_id, args.data_root, transform=loaded.preprocess)
    for row in rows:
        index = int(row["dataset_index"])
        if item_ids[index] != row["item_id"] or labels[index] != int(row["label"]):
            raise RuntimeError("dataset identity differs from the registered control-train list")
    pixels = recover_pixels(dataset, [int(row["dataset_index"]) for row in rows], loaded.mean, loaded.std)
    mean = torch.tensor(loaded.mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(loaded.std, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    clean = torch.zeros((len(rows), prototypes.shape[1]), dtype=torch.float32)
    noisy = torch.zeros((len(rows), args.train_draws, prototypes.shape[1]), dtype=torch.float32)
    for position, row in enumerate(rows):
        pixel = pixels[position].to(device)
        with torch.inference_mode():
            features = loaded.model.encode_image(((pixel.unsqueeze(0) - mean) / std)).to(dtype=torch.float32)
        clean[position] = (features / torch.linalg.vector_norm(features, dim=1, keepdim=True)).cpu()
        item_seed = derive_item_seed(
            base_seed=args.noise_seed, model_id=args.model_id, dataset_id=args.dataset_id,
            sigma=args.sigma, item_id=row["item_id"], stream_role=CONTROL_TRAIN_STREAM,
        )
        collected = []
        for block_index, block_draws in block_plan(args.train_draws):
            generator = torch.Generator(device=device)
            generator.manual_seed(derive_block_seed(item_seed, block_index))
            noise = torch.randn((NOISE_BLOCK, *pixel.shape), generator=generator, dtype=torch.float32, device=device)[:block_draws]
            with torch.inference_mode():
                features = loaded.model.encode_image(((pixel.unsqueeze(0) + args.sigma * noise) - mean) / std).to(dtype=torch.float32)
            collected.append((features / torch.linalg.vector_norm(features, dim=1, keepdim=True)).cpu())
        noisy[position] = torch.cat(collected)
    return rows, clean, noisy, torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)


def fit(prototypes, clean, noisy, labels, *, mode: str, rank: int, epochs: int, lr: float, temperature: float,
        clean_weight: float, device: str, optimizer_seed: int, minibatch_seed: int,
        batch_size: int, weight_decay: float):
    base = prototypes.to(device=device, dtype=torch.float32)
    clean = clean.to(device)
    flat = noisy.reshape(-1, noisy.shape[-1]).to(device)
    flat_labels = labels.repeat_interleave(noisy.shape[1]).to(device)
    clean_labels = labels.to(device)
    torch.manual_seed(optimizer_seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(optimizer_seed)
    if mode == "shared":
        parameters = [torch.zeros(base.shape[1], device=device, requires_grad=True)]
    else:
        left = torch.zeros(base.shape[0], rank, device=device)
        right = torch.randn(rank, base.shape[1], device=device) * 0.01
        left.requires_grad_(True)
        right.requires_grad_(True)
        parameters = [left, right]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)

    def bank():
        delta = parameters[0].unsqueeze(0) if mode == "shared" else parameters[0] @ parameters[1]
        tangent = delta - (base * delta).sum(dim=1, keepdim=True) * base
        candidate = base + tangent
        return candidate / torch.linalg.vector_norm(candidate, dim=1, keepdim=True)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(minibatch_seed)
    for _ in range(epochs):
        permutation = torch.randperm(flat.shape[0], generator=generator).to(device)
        for start in range(0, flat.shape[0], batch_size):
            index = permutation[start : start + batch_size]
            current = bank()
            loss = torch.nn.functional.cross_entropy(flat[index] @ current.T / temperature, flat_labels[index])
            loss = loss + clean_weight * torch.nn.functional.cross_entropy(clean @ current.T / temperature, clean_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        final = bank()
        noisy_accuracy = float((flat @ final.T).argmax(dim=1).eq(flat_labels).double().mean())
        clean_accuracy = float((clean @ final.T).argmax(dim=1).eq(clean_labels).double().mean())
    return [parameter.detach().cpu() for parameter in parameters], {"noisy_accuracy": noisy_accuracy, "clean_accuracy": clean_accuracy}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--sigma", required=True, type=float)
    parser.add_argument("--items", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--development", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--train-draws", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--clean-weight", type=float, default=0.5)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--noise-seed", type=int, default=20260920005)
    parser.add_argument("--optimizer-seed", type=int, default=20260920006)
    parser.add_argument("--minibatch-seed", type=int, default=20260920007)
    args = parser.parse_args()

    registration = read_json(args.registration)
    frozen = registration["candidates"]["control_training"]
    observed = {
        "train_draws_per_item": args.train_draws,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "adamw_weight_decay": args.weight_decay,
        "classification_temperature": args.temperature,
        "clean_loss_weight": args.clean_weight,
        "rank": args.rank,
        "noise_base_seed": args.noise_seed,
        "optimizer_seed": args.optimizer_seed,
        "minibatch_seed": args.minibatch_seed,
    }
    if observed != frozen:
        raise RuntimeError(f"control-training parameters differ from registration: {observed} != {frozen}")

    cell_id = f"{args.model_id}__{args.dataset_id}__sigma{args.sigma:.12g}"
    cell_config = next(value for value in registration["cells"] if value["cell_id"] == cell_id)
    with (args.items / f"{args.dataset_id}__control_train.csv").open("r", encoding="utf-8", newline="") as handle:
        control_rows = sorted(csv.DictReader(handle), key=lambda row: row["item_id"])
    if canonical_json_hash([row["item_id"] for row in control_rows]) != cell_config.get("control_train_items_sha256"):
        raise RuntimeError("control-training item-list hash differs from the registered value")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    cell = f"{args.model_id}__{args.dataset_id}__sigma{args.sigma:.12g}"
    loaded = load_model(args.model_id, device=device)
    development = torch.load(args.development / f"{cell}__development.pt", map_location="cpu", weights_only=False)
    prototypes = development["prototypes"].to(torch.float32)

    rows, clean, noisy, labels = cache_features(args, loaded, prototypes)
    cache_path = args.out / f"{cell}__control_train_features.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"item_ids": [row["item_id"] for row in rows], "clean_features": clean,
                "noisy_features": noisy, "labels": labels, "sigma": args.sigma,
                "final_test_access": False}, cache_path)

    shared, shared_stats = fit(prototypes, clean, noisy, labels, mode="shared", rank=args.rank,
                               epochs=args.epochs, lr=args.learning_rate, temperature=args.temperature,
                               clean_weight=args.clean_weight, device=device, optimizer_seed=args.optimizer_seed,
                               minibatch_seed=args.minibatch_seed,
                               batch_size=args.batch_size, weight_decay=args.weight_decay)
    lowrank, lowrank_stats = fit(prototypes, clean, noisy, labels, mode="lowrank", rank=args.rank,
                                 epochs=args.epochs, lr=args.learning_rate, temperature=args.temperature,
                                 clean_weight=args.clean_weight, device=device, optimizer_seed=args.optimizer_seed,
                                 minibatch_seed=args.minibatch_seed,
                                 batch_size=args.batch_size, weight_decay=args.weight_decay)
    payload = {
        "cell_id": cell,
        "shared_delta": shared[0],
        "lowrank_left": lowrank[0],
        "lowrank_right": lowrank[1],
        "rank": args.rank,
        "registration_sha256": registration["registration_sha256"],
        "training": {"epochs": args.epochs, "batch_size": args.batch_size,
                     "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                     "temperature": args.temperature,
                     "clean_weight": args.clean_weight, "train_draws": args.train_draws,
                     "noise_base_seed": args.noise_seed, "optimizer_seed": args.optimizer_seed,
                     "minibatch_seed": args.minibatch_seed},
        "final_test_access": False,
    }
    out = args.out / f"{cell}__controls.pt"
    torch.save(payload, out)
    write_json_atomic(args.out / f"{cell}__controls.json", {
        "cell_id": cell,
        "shared_delta_norm": float(torch.linalg.vector_norm(shared[0])),
        "shared_delta_sha256": tensor_scientific_hash(shared[0]),
        "shared_train_accuracy": shared_stats,
        "lowrank_train_accuracy": lowrank_stats,
        "training": payload["training"],
        "registration_sha256": registration["registration_sha256"],
        "final_test_access": False,
    })
    print(f"{cell}: ||v||={float(torch.linalg.vector_norm(shared[0])):.4f} "
          f"shared(noisy={shared_stats['noisy_accuracy']:.3f}, clean={shared_stats['clean_accuracy']:.3f}) "
          f"lowrank(noisy={lowrank_stats['noisy_accuracy']:.3f}, clean={lowrank_stats['clean_accuracy']:.3f}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
