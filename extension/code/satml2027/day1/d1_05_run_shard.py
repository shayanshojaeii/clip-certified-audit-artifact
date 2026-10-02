#!/usr/bin/env python3
"""Day 1 - the sampling worker.  One process per GPU; one shard of one cell.

Design notes (all of them are custody- or throughput-relevant):

*   **Noise is generated on the GPU** in fixed blocks of `NOISE_BLOCK` draws,
    each block seeded from the item's stream seed.  Nothing but the final vote
    counts crosses PCIe, and the forward batch size can be tuned per GPU without
    changing a single realized draw.  `--rng-selftest` prints a digest of block 0
    for a fixed seed so the two servers can be proven to agree.
*   **Every bank sees identical features.**  The image is encoded once per draw
    and all banks are scored by one matrix product, so every paired contrast
    uses common random numbers at the draw level.
*   **float32 with TF32 disabled** by default: the decision margins near a tie
    are of order 1e-3, so reduced-precision matmuls are not used for the
    confirmation stream.  `--allow-tf32` exists only for benchmarking.
*   **Resumable**: the shard file is rewritten atomically every
    `--checkpoint-every` items; a restart skips finished items.

Example:
  CUDA_VISIBLE_DEVICES=2 python day1/d1_05_run_shard.py \
      --registration configs/satml2027/exp-20260920-019a.json \
      --cell-id openai-clip-vit-l14-quickgelu__cifar100__sigma0.5 \
      --shard-index 2 --shard-count 4 \
      --items results/satml2027/items --banks results/satml2027/banks \
      --data-root data --out results/satml2027/EXP-20260920-019A
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.hashing import canonical_json_hash, read_json, write_json_atomic  # noqa: E402
from common.banks import validate_registered_bank_payload  # noqa: E402
from common.seeds import (  # noqa: E402
    CONFIRMATION_STREAM,
    DIAGNOSTIC_STREAM,
    NOISE_BLOCK,
    SELECTION_STREAM,
    block_plan,
    derive_block_seed,
    derive_item_seed,
)

GR_CLIP_IMAGE_TRANSFORM = "gr_clip_style_center_and_renormalize"


def _configure_precision(allow_tf32: bool) -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _load_items(items_dir: Path, dataset_id: str, role: str) -> list[dict[str, str]]:
    path = items_dir / f"{dataset_id}__{role}.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: row["item_id"])
    return rows


def _shard_slice(rows: list[dict[str, str]], shard_index: int, shard_count: int) -> tuple[int, list[dict[str, str]]]:
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must lie in [0, shard_count)")
    base, extra = divmod(len(rows), shard_count)
    start = shard_index * base + min(shard_index, extra)
    size = base + (1 if shard_index < extra else 0)
    return start, rows[start : start + size]


class _Scorer:
    """Holds the stacked identity banks and the optional GR-CLIP two-sided bank."""

    def __init__(self, banks: dict[str, dict], device: str):
        import torch

        self.candidate_ids = sorted(banks)
        self.class_count = int(banks[self.candidate_ids[0]]["prototypes"].shape[0])
        identity, gr = [], []
        for position, candidate_id in enumerate(self.candidate_ids):
            entry = banks[candidate_id]
            if entry["metadata"].get("image_transform", "identity") == GR_CLIP_IMAGE_TRANSFORM:
                gr.append((position, entry))
            else:
                identity.append((position, entry))
        self.identity_positions = torch.tensor([position for position, _ in identity], device=device)
        self.identity_banks = torch.stack(
            [entry["prototypes"].to(device=device, dtype=torch.float32) for _, entry in identity]
        )
        self.gr_positions = torch.tensor([position for position, _ in gr], device=device, dtype=torch.long)
        self.gr_banks = (
            torch.stack([entry["prototypes"].to(device=device, dtype=torch.float32) for _, entry in gr])
            if gr
            else None
        )
        self.gr_image_means = (
            torch.stack([entry["image_mean"].to(device=device, dtype=torch.float32) for _, entry in gr])
            if gr
            else None
        )
        if gr and any(entry.get("image_mean") is None for _, entry in gr):
            raise ValueError("a GR-CLIP candidate is missing its calibration image mean")
        self.device = device
        self.count = len(self.candidate_ids)

    def labels(self, features):
        """Return hard labels [draws, candidates] for one batch of unit-norm features."""

        import torch

        out = torch.empty((features.shape[0], self.count), dtype=torch.long, device=self.device)
        logits = torch.einsum("bd,ckd->bck", features, self.identity_banks)
        out[:, self.identity_positions] = torch.argmax(logits, dim=2)
        if self.gr_banks is not None:
            centered = features.unsqueeze(0) - self.gr_image_means.unsqueeze(1)
            norms = torch.linalg.vector_norm(centered, dim=2, keepdim=True)
            if bool(torch.any(norms <= 1e-12)):
                raise ValueError("GR-CLIP transform produced a negligible image vector")
            gr_logits = torch.einsum("cbd,ckd->bck", centered / norms, self.gr_banks)
            out[:, self.gr_positions] = torch.argmax(gr_logits, dim=2)
        return out

    def accumulate(self, labels, counts):
        """counts [candidates, classes] += one-hot sum of labels [draws, candidates]."""

        import torch

        flat = (labels + torch.arange(self.count, device=self.device) * self.class_count).reshape(-1)
        counts += torch.bincount(flat, minlength=self.count * self.class_count).reshape(
            self.count, self.class_count
        ).to(dtype=counts.dtype)


def _raw_predictions(scorer: _Scorer, clean_features) -> np.ndarray:
    return scorer.labels(clean_features).cpu().numpy().T  # [candidates, items]


def rng_selftest(device: str, shape: tuple[int, ...], seed: int = 20260920) -> str:
    import torch

    generator = torch.Generator(device=device)
    generator.manual_seed(derive_block_seed(seed, 0))
    values = torch.randn(shape, generator=generator, dtype=torch.float32, device=device)
    payload = values.cpu().numpy().tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def validate_bank_payload(payload: dict, registration: dict, cell: dict) -> dict:
    return validate_registered_bank_payload(
        payload,
        registration_sha256=registration["registration_sha256"],
        cell_id=cell["cell_id"],
    )


def validate_resume_metadata(prior_meta: dict, registration: dict, cell: dict, scorer_meta: dict) -> None:
    if (prior_meta.get("experiment_id") != registration["experiment_id"]
            or prior_meta.get("registration_sha256") != registration["registration_sha256"]
            or prior_meta.get("cell", {}).get("cell_id") != cell["cell_id"]):
        raise RuntimeError("resume checkpoint metadata differs from the active registration/cell")
    if prior_meta.get("candidate_metadata") != scorer_meta:
        raise RuntimeError("resume checkpoint candidate metadata differs from the current bank artifact")


def main() -> int:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--cell-id", type=str)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--items", type=Path)
    parser.add_argument("--banks", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--item-role", default="evaluation")
    parser.add_argument("--blocks-per-forward", type=int, default=4, help=f"forward batch = this x {NOISE_BLOCK} draws")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--allow-tf32", action="store_true", help="benchmarking only; never for confirmation runs")
    parser.add_argument("--store-features", action="store_true",
                        help="also save normalized noisy features as float32; required by N3C")
    parser.add_argument("--rng-selftest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _configure_precision(args.allow_tf32)

    if args.rng_selftest:
        for shape in ((NOISE_BLOCK, 3, 224, 224), (NOISE_BLOCK, 3, 32, 32)):
            print(f"rng digest {shape}: {rng_selftest(device, shape)}")
        return 0

    registration = read_json(args.registration)
    recorded = registration.pop("registration_sha256")
    recomputed = canonical_json_hash(registration)
    registration["registration_sha256"] = recorded
    if recorded != recomputed:
        raise RuntimeError(f"registration hash mismatch: file says {recorded}, content hashes to {recomputed}")

    is_diagnostic = registration["experiment_id"].startswith("N3C")
    cells = {cell["cell_id"]: cell for cell in registration.get("cells", [])}
    if is_diagnostic:
        cell = next(cell for cell in registration["cells"] if cell["cell_id"] == args.cell_id)
    else:
        cell = cells[args.cell_id]
    model_id, dataset_id, sigma = cell["model_id"], cell["dataset_id"], float(cell["sigma"])

    rows = _load_items(args.items, dataset_id, args.item_role)
    if len(rows) != int(cell["item_count"]):
        raise RuntimeError(f"item count {len(rows)} differs from the registration's {cell['item_count']}")
    expected_item_hash = cell.get(f"{args.item_role}_items_sha256")
    actual_item_hash = canonical_json_hash([row["item_id"] for row in rows])
    if expected_item_hash is None or actual_item_hash != expected_item_hash:
        raise RuntimeError(
            f"registered item-list hash mismatch for {args.cell_id}: "
            f"expected {expected_item_hash}, got {actual_item_hash}"
        )
    offset, shard_rows = _shard_slice(rows, args.shard_index, args.shard_count)

    bank_file = args.banks / f"{model_id}__{dataset_id}__sigma{sigma:.12g}__banks.pt"
    payload = torch.load(bank_file, map_location="cpu", weights_only=False)
    banks = validate_bank_payload(payload, registration, cell)
    scorer_meta = {cid: banks[cid]["metadata"] for cid in sorted(banks)}

    shard_path = args.out / "cells" / args.cell_id / f"shard_{args.shard_index:03d}_of_{args.shard_count:03d}.npz"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_ids = sorted(banks)
    class_count = int(cell["class_count"])
    n_items, n_cand = len(shard_rows), len(candidate_ids)
    selection = np.zeros((n_cand, n_items, class_count), dtype=np.int32)
    confirmation = np.zeros((n_cand, n_items, class_count), dtype=np.int32)
    done = np.zeros(n_items, dtype=bool)
    seeds_selection = np.zeros(n_items, dtype=np.int64)
    seeds_confirmation = np.zeros(n_items, dtype=np.int64)
    if shard_path.is_file():
        meta_path = shard_path.with_suffix(".meta.json")
        if not meta_path.is_file():
            raise RuntimeError("resume checkpoint metadata sidecar is absent")
        prior_meta = read_json(meta_path)
        validate_resume_metadata(prior_meta, registration, cell, scorer_meta)
        previous = np.load(shard_path, allow_pickle=False)
        expected_ids = [row["item_id"] for row in shard_rows]
        resume_checks = {
            "candidate_ids": list(previous["candidate_ids"]) == candidate_ids,
            "item_ids": list(previous["item_ids"]) == expected_ids,
            "dataset_indices": np.array_equal(previous["dataset_indices"], np.array([int(row["dataset_index"]) for row in shard_rows])),
            "ground_truth": np.array_equal(previous["ground_truth"], np.array([int(row["label"]) for row in shard_rows])),
            "item_offset": int(previous["item_offset"][0]) == offset,
            "selection_shape": previous["selection_counts"].shape == selection.shape,
            "confirmation_shape": previous["confirmation_counts"].shape == confirmation.shape,
            "done_shape": previous["done"].shape == done.shape,
        }
        failed = [name for name, value in resume_checks.items() if not value]
        if failed:
            raise RuntimeError(f"resume checkpoint provenance mismatch: {failed}")
        selection, confirmation, done = previous["selection_counts"], previous["confirmation_counts"], previous["done"]
        seeds_selection = previous["selection_seeds"].copy()
        seeds_confirmation = previous["confirmation_seeds"].copy()
        if bool(np.any(done & (seeds_selection == 0))):
            raise RuntimeError("completed resume rows contain missing selection seeds")
        print(f"resuming: {int(done.sum())}/{n_items} items already finished")

    print(f"cell={args.cell_id} shard={args.shard_index}/{args.shard_count} items={n_items} "
          f"banks={n_cand} device={torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}")
    if args.dry_run:
        return 0

    from common.datasets import load_dataset, recover_pixels  # noqa: E402
    from common.models import load_model  # noqa: E402

    loaded = load_model(model_id, device=device)
    dataset, item_ids, labels = load_dataset(dataset_id, args.data_root, transform=loaded.preprocess)
    for row in shard_rows:
        index = int(row["dataset_index"])
        if item_ids[index] != row["item_id"] or labels[index] != int(row["label"]):
            raise RuntimeError(f"dataset identity differs from the registered item list at {row['item_id']}")

    pixels = recover_pixels(dataset, [int(row["dataset_index"]) for row in shard_rows], loaded.mean, loaded.std)
    mean = torch.tensor(loaded.mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(loaded.std, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    scorer = _Scorer(banks, device)
    clean = []
    with torch.inference_mode():
        for start in range(0, len(shard_rows), 64):
            batch = ((pixels[start : start + 64].to(device) - mean) / std)
            features = loaded.model.encode_image(batch).to(dtype=torch.float32)
            clean.append(features / torch.linalg.vector_norm(features, dim=1, keepdim=True))
    clean_features = torch.cat(clean)
    raw_predictions = _raw_predictions(scorer, clean_features)

    certification = registration.get("certification", {})
    selection_draws = registration["sampling"]["draws_per_item"] if is_diagnostic else int(certification["selection_draws"])
    confirmation_draws = 0 if is_diagnostic else int(certification["confirmation_draws"])
    selection_base = registration["sampling"]["base_seed"] if is_diagnostic else int(certification["selection_base_seed"])
    confirmation_base = None if is_diagnostic else int(certification["confirmation_base_seed"])
    selection_role = DIAGNOSTIC_STREAM if is_diagnostic else SELECTION_STREAM

    features_store = None
    started = time.time()
    processed = 0
    feature_path = shard_path.with_name(shard_path.stem + "_features.npz")
    if args.store_features and bool(done.any()):
        if not feature_path.is_file():
            raise RuntimeError("cannot resume an N3C shard: checkpoint exists but its feature store is absent")
        saved_features = np.load(feature_path, allow_pickle=False)
        features_store = saved_features["features"].astype(np.float32, copy=False)
        if features_store.shape[:2] != (n_items, selection_draws):
            raise RuntimeError("saved N3C feature store has the wrong shape")
        if not np.array_equal(saved_features["done"], done):
            raise RuntimeError("N3C count and feature completion states diverge")
        if list(saved_features["item_ids"]) != [row["item_id"] for row in shard_rows]:
            raise RuntimeError("N3C count and feature item identities diverge")
        if saved_features["clean_features"].shape != tuple(clean_features.cpu().numpy().shape):
            raise RuntimeError("saved N3C clean-feature store has the wrong shape")
        feature_identity = {
            "experiment_id": str(saved_features["experiment_id"][0]),
            "registration_sha256": str(saved_features["registration_sha256"][0]),
            "cell_id": str(saved_features["cell_id"][0]),
            "sigma": float(saved_features["sigma"][0]),
            "shard_index": int(saved_features["shard_index"][0]),
            "shard_count": int(saved_features["shard_count"][0]),
            "stream_role": str(saved_features["stream_role"][0]),
            "base_seed": int(saved_features["base_seed"][0]),
        }
        expected_feature_identity = {
            "experiment_id": registration["experiment_id"],
            "registration_sha256": registration["registration_sha256"],
            "cell_id": cell["cell_id"],
            "sigma": sigma,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "stream_role": selection_role,
            "base_seed": int(selection_base),
        }
        if feature_identity != expected_feature_identity:
            raise RuntimeError("saved N3C feature provenance differs from the active sampling identity")
    for position, row in enumerate(shard_rows):
        if done[position]:
            continue
        pixel = pixels[position].to(device)
        item_seed = derive_item_seed(
            base_seed=selection_base, model_id=model_id, dataset_id=dataset_id,
            sigma=sigma, item_id=row["item_id"], stream_role=selection_role,
        )
        seeds_selection[position] = item_seed
        result = _count_item(
            model=loaded.model, scorer=scorer, pixel=pixel, mean=mean, std=std,
            item_seed=item_seed, draws=selection_draws, sigma=sigma,
            blocks_per_forward=args.blocks_per_forward, device=device,
            collect_features=args.store_features,
        )
        if args.store_features:
            selection[:, position], item_features = result
            if features_store is None:
                features_store = np.zeros((n_items, selection_draws, item_features.shape[1]), dtype=np.float32)
            features_store[position] = item_features
        else:
            selection[:, position] = result
        if confirmation_draws:
            confirmation_seed = derive_item_seed(
                base_seed=confirmation_base, model_id=model_id, dataset_id=dataset_id,
                sigma=sigma, item_id=row["item_id"], stream_role=CONFIRMATION_STREAM,
            )
            if confirmation_seed == item_seed:
                raise RuntimeError("selection and confirmation seed collision")
            seeds_confirmation[position] = confirmation_seed
            confirmation[:, position] = _count_item(
                model=loaded.model, scorer=scorer, pixel=pixel, mean=mean, std=std,
                item_seed=confirmation_seed, draws=confirmation_draws, sigma=sigma,
                blocks_per_forward=args.blocks_per_forward, device=device,
            )
        done[position] = True
        processed += 1
        if processed % args.checkpoint_every == 0 or position == n_items - 1:
            _save(shard_path, candidate_ids, shard_rows, offset, selection, confirmation, done,
                  raw_predictions, seeds_selection, seeds_confirmation, cell, registration, scorer_meta, started)
            if args.store_features and features_store is not None:
                _save_npz_atomic(
                    feature_path,
                    features=features_store,
                    item_ids=np.array([row["item_id"] for row in shard_rows]),
                    ground_truth=np.array([int(row["label"]) for row in shard_rows], dtype=np.int64),
                    clean_features=clean_features.cpu().numpy().astype(np.float32),
                    done=done,
                    experiment_id=np.array([registration["experiment_id"]]),
                    registration_sha256=np.array([registration["registration_sha256"]]),
                    cell_id=np.array([cell["cell_id"]]),
                    sigma=np.array([sigma], dtype=np.float64),
                    shard_index=np.array([args.shard_index], dtype=np.int64),
                    shard_count=np.array([args.shard_count], dtype=np.int64),
                    stream_role=np.array([selection_role]),
                    base_seed=np.array([selection_base], dtype=np.int64),
                )
            rate = processed * (selection_draws + confirmation_draws) / max(time.time() - started, 1e-9)
            print(f"  {int(done.sum())}/{n_items} items  {rate:,.0f} encodes/s  "
                  f"eta {(n_items - int(done.sum())) * (selection_draws + confirmation_draws) / max(rate, 1e-9) / 60:.1f} min",
                  flush=True)

    _save(shard_path, candidate_ids, shard_rows, offset, selection, confirmation, done,
          raw_predictions, seeds_selection, seeds_confirmation, cell, registration, scorer_meta, started)
    if args.store_features and features_store is not None:
        _save_npz_atomic(feature_path, features=features_store,
                         item_ids=np.array([row["item_id"] for row in shard_rows]),
                         ground_truth=np.array([int(row["label"]) for row in shard_rows], dtype=np.int64),
                         clean_features=clean_features.cpu().numpy().astype(np.float32), done=done,
                         experiment_id=np.array([registration["experiment_id"]]),
                         registration_sha256=np.array([registration["registration_sha256"]]),
                         cell_id=np.array([cell["cell_id"]]), sigma=np.array([sigma], dtype=np.float64),
                         shard_index=np.array([args.shard_index], dtype=np.int64),
                         shard_count=np.array([args.shard_count], dtype=np.int64),
                         stream_role=np.array([selection_role]),
                         base_seed=np.array([selection_base], dtype=np.int64))
        print(f"features: {feature_path}")
    print(f"done: {shard_path}")
    return 0


def _count_item(*, model, scorer, pixel, mean, std, item_seed, draws, sigma, blocks_per_forward, device,
                collect_features: bool = False):
    """Vote counts for one item: sigma-scaled Gaussian noise, encode, score every bank."""

    import torch

    counts = torch.zeros((scorer.count, scorer.class_count), dtype=torch.int32, device=device)
    collected = [] if collect_features else None
    plan = block_plan(draws)
    for start in range(0, len(plan), blocks_per_forward):
        chunk = plan[start : start + blocks_per_forward]
        pieces = []
        for block_index, block_draws in chunk:
            generator = torch.Generator(device=device)
            generator.manual_seed(derive_block_seed(item_seed, block_index))
            noise = torch.randn((NOISE_BLOCK, *pixel.shape), generator=generator, dtype=torch.float32, device=device)
            pieces.append(noise[:block_draws])
        noise = torch.cat(pieces, dim=0)
        standardized = ((pixel.unsqueeze(0) + float(sigma) * noise) - mean) / std
        with torch.inference_mode():
            features = model.encode_image(standardized).to(dtype=torch.float32)
        features = features / torch.linalg.vector_norm(features, dim=1, keepdim=True)
        scorer.accumulate(scorer.labels(features), counts)
        if collected is not None:
            collected.append(features.to(dtype=torch.float32).cpu())
    if not bool(torch.all(counts.sum(dim=1) == draws)):
        raise RuntimeError("vote totals do not match the draw budget")
    if collected is not None:
        return counts.cpu().numpy(), torch.cat(collected).numpy()
    return counts.cpu().numpy()


def _save(path, candidate_ids, rows, offset, selection, confirmation, done, raw_predictions,
          seeds_selection, seeds_confirmation, cell, registration, scorer_meta, started):
    _save_npz_atomic(
        path,
        candidate_ids=np.array(candidate_ids),
        item_ids=np.array([row["item_id"] for row in rows]),
        dataset_indices=np.array([int(row["dataset_index"]) for row in rows], dtype=np.int64),
        ground_truth=np.array([int(row["label"]) for row in rows], dtype=np.int64),
        raw_predictions=raw_predictions,
        selection_counts=selection,
        confirmation_counts=confirmation,
        done=done,
        item_offset=np.array([offset], dtype=np.int64),
        selection_seeds=seeds_selection,
        confirmation_seeds=seeds_confirmation,
        elapsed_seconds=np.array([time.time() - started]),
    )
    write_json_atomic(
        path.with_suffix(".meta.json"),
        {
            "schema_version": "satml2027.shard_meta.v1",
            "experiment_id": registration["experiment_id"],
            "registration_sha256": registration["registration_sha256"],
            "cell": cell,
            "candidate_metadata": scorer_meta,
            "items_done": int(done.sum()),
            "items_total": int(len(done)),
            "final_test_access": False,
        },
    )


def _save_npz_atomic(path: Path, **arrays) -> None:
    """Write an NPZ without NumPy silently appending a second `.npz` suffix."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
