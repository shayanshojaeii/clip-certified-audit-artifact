#!/usr/bin/env python3
"""Day 1 - development artifacts for one (model, dataset, sigma) cell.

Produces, from the development split only:
  * the prompt-ensemble text prototypes,
  * clean development features,
  * cached noisy development features (used for the noisy-margin direction and
    for the training-free control bank),
  * the two proposal directions that the frozen EXP-017 grid translates along.

Direction builders
------------------
`--direction-builder project` (default) imports the project's own
`interventions.boundary_active` / `interventions.noisy_margin` so the extension
uses literally the same code as EXP-016.  `--direction-builder registered`
implements the same two documented objectives directly from their definitions
and is used only if the project modules cannot be imported; the choice is
recorded in the output and must be recorded in the registration.

Both objectives differentiate a *common raw translation* `t_k -> normalize(t_k + a v)`
at `a = 0`, where `d/da normalize(t + a v)|_0 = v - <v,t> t`, so the gradient of a
pairwise margin `M_{Ak} = <z, t_A - t_k>` with respect to `v` is
`g_{Ak}(z) = <z,t_k> t_k - <z,t_A> t_A`.

  clean_boundary_active     per sample: minimum-norm point of the convex hull of
                            the critical `g_{A k}` (the maximin ascent direction),
                            normalized; then one equal vote per sample.
  cohen_aligned_noisy_margin gradient of the smooth lower-tail objective
                            `-log(1 + sum_k exp(-M_{Ak}/tau))`, averaged over
                            samples and cached noise draws, then normalized.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.datasets import class_names, load_dataset, recover_pixels  # noqa: E402
from common.hashing import canonical_json_hash, read_json, sha256_file, tensor_scientific_hash, write_json_atomic  # noqa: E402
from common.models import build_text_prototypes, load_model  # noqa: E402
from common.prompts import load_prompt_config, render_prompts  # noqa: E402
from common.seeds import DEVELOPMENT_STREAM, NOISE_BLOCK, block_plan, derive_block_seed, derive_item_seed  # noqa: E402


def pair_gradient(feature: torch.Tensor, prototypes: torch.Tensor, protected: int, competitor: int) -> torch.Tensor:
    return (feature @ prototypes[competitor]) * prototypes[competitor] - (feature @ prototypes[protected]) * prototypes[protected]


def minimum_norm_convex_point(vectors: torch.Tensor, iterations: int = 5000, tolerance: float = 1e-14) -> torch.Tensor:
    """Frank-Wolfe minimum-norm point of a convex hull (Wolfe's problem)."""

    weights = torch.full((vectors.shape[0],), 1.0 / vectors.shape[0], dtype=torch.float64)
    point = weights @ vectors
    for step in range(iterations):
        index = int(torch.argmin(vectors @ point))
        direction = vectors[index] - point
        denominator = float(direction @ direction)
        if denominator <= tolerance:
            break
        gamma = float(max(0.0, min(1.0, -float(point @ direction) / denominator)))
        if gamma <= tolerance:
            break
        point = point + gamma * direction
    return point


def clean_boundary_active_direction(clean_features: torch.Tensor, prototypes: torch.Tensor, tie_tolerance: float = 1e-10) -> torch.Tensor:
    bank = prototypes.to(dtype=torch.float64)
    rows = []
    for feature in clean_features.to(dtype=torch.float64):
        scores = bank @ feature
        protected = int(torch.argmax(scores))
        margins = scores[protected] - scores
        margins[protected] = float("inf")
        critical = torch.nonzero(margins <= float(margins.min()) + tie_tolerance, as_tuple=False).flatten().tolist()
        gradients = torch.stack([pair_gradient(feature, bank, protected, competitor) for competitor in critical])
        point = minimum_norm_convex_point(gradients) if len(critical) > 1 else gradients[0]
        norm = float(torch.linalg.vector_norm(point))
        if norm <= 1e-12:
            continue  # sample has no common ascent direction; it casts no vote
        rows.append(point / norm)
    if not rows:
        raise RuntimeError("no development sample produced a clean ascent direction")
    total = torch.stack(rows).sum(dim=0)
    norm = float(torch.linalg.vector_norm(total))
    if norm <= 1e-10:
        raise RuntimeError("clean local directions cancelled")
    return (total / norm).to(dtype=torch.float32)


def noisy_margin_direction(
    noisy_features: torch.Tensor, clean_features: torch.Tensor, prototypes: torch.Tensor, temperature: float
) -> torch.Tensor:
    bank = prototypes.to(dtype=torch.float64)
    protected = torch.argmax(clean_features.to(dtype=torch.float64) @ bank.T, dim=1)
    accumulator = torch.zeros(bank.shape[1], dtype=torch.float64)
    total = 0
    for position in range(noisy_features.shape[0]):
        index = int(protected[position])
        draws = noisy_features[position].to(dtype=torch.float64)
        scores = draws @ bank.T
        margins = scores[:, index : index + 1] - scores
        exponent = torch.exp(-margins / temperature)
        exponent[:, index] = 0.0
        weights = exponent / (1.0 + exponent.sum(dim=1, keepdim=True)) / temperature
        projected = scores.unsqueeze(2) * bank.unsqueeze(0)  # [draws, classes, dim]
        contribution = (weights.unsqueeze(2) * projected).sum(dim=1) - weights.sum(dim=1, keepdim=True) * (
            scores[:, index : index + 1] * bank[index].unsqueeze(0)
        )
        accumulator += contribution.sum(dim=0)
        total += draws.shape[0]
    accumulator /= max(total, 1)
    norm = float(torch.linalg.vector_norm(accumulator))
    if norm <= 1e-12:
        raise RuntimeError("noisy-margin gradient vanished")
    return (accumulator / norm).to(dtype=torch.float32)


def project_directions(clean_features, noisy_features, prototypes, temperature, project_root: Path,
                       *, dataset_id: str, sample_ids: list[str], split_manifest: Path,
                       exclusion_manifest: Path, sigma: float):
    """Use the project's own EXP-016 builders when they are importable."""

    sys.path.insert(0, str(project_root))
    from interventions.aggregation import (  # type: ignore
        aggregate_boundary_active_directions,
        build_calibration_aggregation_provenance,
    )
    from interventions.boundary_active import compute_boundary_active_direction  # type: ignore
    from interventions.noisy_margin import compute_noisy_margin_proposal  # type: ignore

    locals_ = []
    for feature in clean_features:
        result = compute_boundary_active_direction(
            feature,
            prototypes,
            tie_tolerance=1e-10,
            solver_tolerance=1e-12,
            zero_tolerance=1e-12,
            max_iterations=100000,
        )
        if not result.common_ascent_exists:
            raise RuntimeError("EXP-016 clean-boundary common-ascent contract failed")
        locals_.append(result)
    exclusion = read_json(exclusion_manifest)
    excluded_ids: list[str] = []
    stack = [exclusion]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "explicit_consumed_item_ids" and isinstance(child, list):
                    excluded_ids.extend(str(item) for item in child)
                else:
                    stack.append(child)
        elif isinstance(value, list):
            stack.extend(value)
    excluded_ids = sorted(set(excluded_ids))
    if not excluded_ids:
        raise RuntimeError("the frozen Gate-2 exclusion manifest contains no consumed item IDs")
    provenance = build_calibration_aggregation_provenance(
        dataset_id=dataset_id,
        split_id="satml2027_extension_development",
        split_role="calibration_development",
        split_manifest_sha256=sha256_file(split_manifest),
        exclusion_manifest_sha256=sha256_file(exclusion_manifest),
        sample_ids=sample_ids,
        excluded_gate2_sample_ids=excluded_ids,
        final_test_accessed=False,
    )
    aggregate = aggregate_boundary_active_directions(
        locals_,
        provenance=provenance,
        candidate_id="satml2027_exact_phase3_boundary_active",
        cancellation_tolerance=1e-10,
    )
    if not aggregate.deployable:
        raise RuntimeError("EXP-016 clean-boundary aggregate cancels below tolerance")
    clean_direction = aggregate.direction.to(dtype=torch.float32)
    proposal = compute_noisy_margin_proposal(
        noisy_features, clean_features, prototypes,
        provenance=provenance,
        candidate_id="satml2027_exact_phase3_noisy_margin",
        temperature=temperature,
        noise_sigma=sigma,
        proposal_stream_id="satml2027_extension_development_direction",
        cohen_selection_stream_id="satml2027_extension_cohen_selection",
        cohen_confirmation_stream_id="satml2027_extension_cohen_confirmation",
        expected_draws_per_sample=noisy_features.shape[1],
        clean_prediction_tie_tolerance=1e-10,
        cancellation_tolerance=1e-10,
    )
    if not proposal.deployable:
        raise RuntimeError("EXP-016 noisy-margin direction is nondeployable")
    noisy_direction = proposal.direction.to(dtype=torch.float32)
    return clean_direction, noisy_direction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--sigma", required=True, type=float)
    parser.add_argument("--items", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--prompt-config", default=None)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--split-manifest", type=Path,
                        default=Path("artifacts/day14/phase3_splits_v1/manifest.json"))
    parser.add_argument("--exclusion-manifest", type=Path,
                        default=Path("artifacts/day14/phase3_splits_v1/exclusion_manifest.json"))
    parser.add_argument("--direction-builder", choices=["project"], default="project")
    parser.add_argument("--noisy-temperature", type=float, required=True,
                        help="frozen EXP-016 lower-tail temperature; read it from the Phase-3 protocol config")
    parser.add_argument("--development-draws", type=int, default=64)
    parser.add_argument("--base-seed", type=int, default=20260920004)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    registration = read_json(args.registration)
    frozen = registration["candidates"]["direction_construction"]
    observed = {
        "builder": "exact_phase3_interfaces",
        "development_draws_per_item": args.development_draws,
        "noise_base_seed": args.base_seed,
        "noisy_margin_temperature": args.noisy_temperature,
        "tie_tolerance": 1e-10,
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "exclusion_manifest_sha256": sha256_file(args.exclusion_manifest),
    }
    frozen_direction = {key: value for key, value in frozen.items() if key != "prompt_configs"}
    if observed != frozen_direction:
        raise RuntimeError(f"direction-construction parameters differ from registration: {observed} != {frozen_direction}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    loaded = load_model(args.model_id, device=device)
    prompt_config = load_prompt_config(args.project_root, args.dataset_id, args.prompt_config)
    registered_prompt = frozen["prompt_configs"][args.dataset_id]
    if registered_prompt.get("path"):
        if args.prompt_config != registered_prompt["path"]:
            raise RuntimeError("runtime prompt-config path differs from registration")
        if sha256_file(args.project_root / args.prompt_config) != registered_prompt["sha256"]:
            raise RuntimeError("runtime prompt-config hash differs from registration")
    elif args.prompt_config is not None:
        raise RuntimeError("built-in prompt registration forbids an external prompt config")

    with (args.items / f"{args.dataset_id}__development.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: row["item_id"])

    cell_id = f"{args.model_id}__{args.dataset_id}__sigma{args.sigma:.12g}"
    cell = next(value for value in registration["cells"] if value["cell_id"] == cell_id)
    item_hash = canonical_json_hash([row["item_id"] for row in rows])
    if item_hash != cell.get("development_items_sha256"):
        raise RuntimeError("development item-list hash differs from the registered value")

    dataset, item_ids, labels = load_dataset(args.dataset_id, args.data_root, transform=loaded.preprocess)
    for row in rows:
        index = int(row["dataset_index"])
        if item_ids[index] != row["item_id"] or labels[index] != int(row["label"]):
            raise RuntimeError("dataset identity differs from the registered development list")
    names = class_names(dataset, prompt_config)
    prototypes = build_text_prototypes(loaded, render_prompts(names, prompt_config["templates"]))

    pixels = recover_pixels(dataset, [int(row["dataset_index"]) for row in rows], loaded.mean, loaded.std)
    mean = torch.tensor(loaded.mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(loaded.std, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    clean_parts = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch):
            batch = (pixels[start : start + args.batch].to(device) - mean) / std
            features = loaded.model.encode_image(batch).to(dtype=torch.float32)
            clean_parts.append((features / torch.linalg.vector_norm(features, dim=1, keepdim=True)).cpu())
    clean_features = torch.cat(clean_parts)

    noisy = torch.zeros((len(rows), args.development_draws, prototypes.shape[1]), dtype=torch.float32)
    for position, row in enumerate(rows):
        item_seed = derive_item_seed(
            base_seed=args.base_seed, model_id=args.model_id, dataset_id=args.dataset_id,
            sigma=args.sigma, item_id=row["item_id"], stream_role=DEVELOPMENT_STREAM,
        )
        pixel = pixels[position].to(device)
        collected = []
        for block_index, block_draws in block_plan(args.development_draws):
            generator = torch.Generator(device=device)
            generator.manual_seed(derive_block_seed(item_seed, block_index))
            noise = torch.randn((NOISE_BLOCK, *pixel.shape), generator=generator, dtype=torch.float32, device=device)[:block_draws]
            standardized = ((pixel.unsqueeze(0) + args.sigma * noise) - mean) / std
            with torch.inference_mode():
                features = loaded.model.encode_image(standardized).to(dtype=torch.float32)
            collected.append((features / torch.linalg.vector_norm(features, dim=1, keepdim=True)).cpu())
        noisy[position] = torch.cat(collected)

    clean_direction, noisy_direction = project_directions(
        clean_features, noisy, prototypes, args.noisy_temperature, args.project_root,
        dataset_id=args.dataset_id, sample_ids=[row["item_id"] for row in rows],
        split_manifest=args.split_manifest, exclusion_manifest=args.exclusion_manifest,
        sigma=args.sigma,
    )

    cell = f"{args.model_id}__{args.dataset_id}__sigma{args.sigma:.12g}"
    out = args.out / f"{cell}__development.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cell_id": cell,
            "model_id": args.model_id,
            "dataset_id": args.dataset_id,
            "sigma": args.sigma,
            "prototypes": prototypes,
            "clean_features": clean_features,
            "noisy_features": noisy,
            "clean_direction": clean_direction,
            "noisy_direction": noisy_direction,
            "item_ids": [row["item_id"] for row in rows],
            "direction_builder": args.direction_builder,
            "noisy_temperature": args.noisy_temperature,
            "development_draws": args.development_draws,
            "prompt_config": prompt_config,
            "final_test_access": False,
            "registration_sha256": registration["registration_sha256"],
        },
        out,
    )
    write_json_atomic(
        args.out / f"{cell}__development.json",
        {
            "cell_id": cell,
            "prototype_sha256": tensor_scientific_hash(prototypes),
            "clean_direction_sha256": tensor_scientific_hash(clean_direction),
            "noisy_direction_sha256": tensor_scientific_hash(noisy_direction),
            "clean_features_sha256": tensor_scientific_hash(clean_features),
            "development_items": len(rows),
            "development_draws": args.development_draws,
            "direction_builder": args.direction_builder,
            "noisy_temperature": args.noisy_temperature,
            "clean_zero_shot_accuracy": float(
                np.mean((clean_features @ prototypes.T).argmax(dim=1).numpy() == np.array([int(row["label"]) for row in rows]))
            ),
            "final_test_access": False,
            "registration_sha256": registration["registration_sha256"],
        },
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
