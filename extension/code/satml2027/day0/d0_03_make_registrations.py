#!/usr/bin/env python3
"""Day 0, step 3 - emit and hash the preregistration files.

Writes three registrations, each with its canonical-JSON hash printed into a
sidecar `.sha256` file:

  N3C-20260920-V3   fresh-noise diagnostic on the EXP-017 development items
  EXP-20260920-019A sigma extension on the two EXP-017 backbones
  EXP-20260920-019B backbone extension (B/16, LAION B/32, RN50)

The hash must be committed (or timestamped) BEFORE any sampling starts; the
worker refuses to run against a registration whose hash is not recorded in its
own `registration_sha256` field.

Usage:
  python day0/d0_03_make_registrations.py --items results/satml2027/items \
      --out configs/satml2027 [--scope full|no_sigma_012|core]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.certify import min_successes_for_radius  # noqa: E402
from common.hashing import canonical_json_hash, read_json, sha256_file, write_json_atomic  # noqa: E402
from common.seeds import CONFIRMATION_STREAM, DIAGNOSTIC_STREAM, NOISE_BLOCK, SELECTION_STREAM  # noqa: E402

FROZEN_STEPS = [0.0025, 0.005, 0.01, 0.02, 0.04]
FROZEN_COEFFICIENTS = [0.25, 0.5, 1.0]
RADII = [0.0, 0.12, 0.25, 0.5]

SCOPES = {
    "full": {
        "019a": {"models": ["openai-clip-vit-b32-quickgelu", "openai-clip-vit-l14-quickgelu"],
                 "sigmas": [0.12, 0.5], "datasets": ["cifar100", "eurosat", "cifar10"],
                 "extra": [{"models": ["openai-clip-vit-b32-quickgelu", "openai-clip-vit-l14-quickgelu"],
                            "sigmas": [0.25], "datasets": ["cifar10"]}]},
        "019b": {"models": ["openai-clip-vit-b16-quickgelu", "openclip-vit-b32-laion2b", "openai-clip-rn50-quickgelu"],
                 "sigmas": [0.25, 0.5], "datasets": ["cifar100", "eurosat", "cifar10"], "extra": []},
    },
    "no_sigma_012": {
        "019a": {"models": ["openai-clip-vit-b32-quickgelu", "openai-clip-vit-l14-quickgelu"],
                 "sigmas": [0.5], "datasets": ["cifar100", "eurosat", "cifar10"],
                 "extra": [{"models": ["openai-clip-vit-b32-quickgelu", "openai-clip-vit-l14-quickgelu"],
                            "sigmas": [0.25], "datasets": ["cifar10"]}]},
        "019b": {"models": ["openai-clip-vit-b16-quickgelu", "openclip-vit-b32-laion2b", "openai-clip-rn50-quickgelu"],
                 "sigmas": [0.25, 0.5], "datasets": ["cifar100", "eurosat", "cifar10"], "extra": []},
    },
    "core": {
        "019a": {"models": ["openai-clip-vit-b32-quickgelu", "openai-clip-vit-l14-quickgelu"],
                 "sigmas": [0.5], "datasets": ["cifar100", "eurosat"],
                 "extra": [{"models": ["openai-clip-vit-b32-quickgelu"], "sigmas": [0.25], "datasets": ["cifar10"]}]},
        "019b": {"models": ["openai-clip-vit-b16-quickgelu", "openclip-vit-b32-laion2b"],
                 "sigmas": [0.25], "datasets": ["cifar100", "eurosat", "cifar10"], "extra": []},
    },
}


def expand_cells(spec: dict, item_manifest: dict) -> list[dict]:
    cells: list[dict] = []
    blocks = [{"models": spec["models"], "sigmas": spec["sigmas"], "datasets": spec["datasets"]}] + list(spec["extra"])
    for block in blocks:
        for model_id in block["models"]:
            for dataset_id in block["datasets"]:
                entry = item_manifest["datasets"].get(dataset_id)
                if entry is None:
                    raise KeyError(f"item manifest has no dataset {dataset_id}; run d0_02 first")
                for sigma in block["sigmas"]:
                    cells.append(
                        {
                            "cell_id": f"{model_id}__{dataset_id}__sigma{sigma:.12g}",
                            "model_id": model_id,
                            "dataset_id": dataset_id,
                            "data_role": ("extension_new_dataset_train_split" if dataset_id == "cifar10"
                                          else "extension_reserved_accessed"),
                            "sigma": float(sigma),
                            "item_count": int(entry["splits"]["evaluation"]["count"]),
                            "class_count": int(entry["class_count"]),
                            "evaluation_items_sha256": entry["splits"]["evaluation"]["item_ids_sha256"],
                            "development_items_sha256": entry["splits"]["development"]["item_ids_sha256"],
                            "control_train_items_sha256": entry["splits"]["control_train"]["item_ids_sha256"],
                        }
                    )
    unique = {cell["cell_id"]: cell for cell in cells}
    return [unique[key] for key in sorted(unique)]


def certification_block(sigma_values: list[float]) -> dict:
    return {
        "selection_draws": 128,
        "confirmation_draws": 4096,
        "alpha_per_example": 0.001,
        "primary_radius_rule": "r = sigma (so k_min is identical at every sigma)",
        "k_min_at_r_equals_sigma": min_successes_for_radius(4096, 0.001, 0.25, 0.25),
        "reported_radii": RADII,
        "selection_stream_role": SELECTION_STREAM,
        "confirmation_stream_role": CONFIRMATION_STREAM,
        "selection_base_seed": 20260920001,
        "confirmation_base_seed": 20260920002,
        "noise_block_draws": NOISE_BLOCK,
        "encoder_precision": "float32 (TF32 disabled); a registered precision-equivalence check is reported",
    }


def predictions_block() -> list[dict]:
    return [
        {"id": "X1", "statement": "report every frozen-grid candidate's macro standard-CA change at r=sigma relative to no correction against the prospective practical-effect region, without an equivalence claim", "type": "falsifiable"},
        {"id": "X3", "statement": "the learned shared-translation control stays inside the same band", "type": "falsifiable"},
        {"id": "X4", "statement": "the per-class control banks leave the band", "type": "falsifiable positive control"},
        {"id": "X5", "statement": "top-class share and predicted-class Gini of the zero-shot bank increase with sigma", "type": "falsifiable"},
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--scope", default="full", choices=sorted(SCOPES))
    parser.add_argument("--registered-at", default=None,
                        help="frozen UTC ISO timestamp; required for byte-identical independent regeneration")
    parser.add_argument("--exp017-results", default="results/EXP-20260906-017", type=str)
    parser.add_argument("--split-manifest", type=Path,
                        default=Path("artifacts/day14/phase3_splits_v1/manifest.json"))
    parser.add_argument("--exclusion-manifest", type=Path,
                        default=Path("artifacts/day14/phase3_splits_v1/exclusion_manifest.json"))
    args = parser.parse_args()

    item_manifest = read_json(args.items / "item_manifest.json")
    created = args.registered_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    scope = SCOPES[args.scope]
    written = []

    common_candidates = {
        "frozen_grid": {
            "builder": "common/banks.py::build_eighteen_candidate_banks (port of the EXP-017 rule)",
            "verification": "d1_03_build_banks.py --verify-against-exp017 requires identical candidate IDs and max absolute tensor error <= 2e-6; exact hashes are reported separately because torch SVD/reduction kernels may differ at final float32 bits",
            "portable_reproduction_atol": 2e-6,
            "common_unit_direction_step_grid": FROZEN_STEPS,
            "gap_coefficient_grid": FROZEN_COEFFICIENTS,
            "bank_count": 18,
        },
        "controls": [
            {"control_id": "learned_shared_translation_tangent", "form": "normalize(t_k + v - <v,t_k> t_k)", "training": "preregistered supervised noisy+clean cross-entropy control on the control_train split re-encoded at this sigma"},
            {"control_id": "learned_shared_translation_pure", "form": "normalize(t_k + v)", "training": "same v as the tangent form"},
            {"control_id": "noisy_class_mean", "form": "normalize(mean of unit-norm noisy training draws per class)", "training": "optimization-free supervised noisy class-mean control"},
            {"control_id": "lowrank_tangent_r8", "form": "normalize(t_k + tangent(LR)_k), rank 8", "training": "preregistered rank-8 tangent noisy+clean cross-entropy control; not labeled as the historical Phase-4 recipe"},
        ],
        "control_training": {
            "train_draws_per_item": 16,
            "epochs": 80,
            "batch_size": 1024,
            "learning_rate": 0.005,
            "adamw_weight_decay": 0.0001,
            "classification_temperature": 0.05,
            "clean_loss_weight": 0.5,
            "rank": 8,
            "noise_base_seed": 20260920005,
            "optimizer_seed": 20260920006,
            "minibatch_seed": 20260920007,
        },
        "direction_construction": {
            "builder": "exact_phase3_interfaces",
            "development_draws_per_item": 64,
            "noise_base_seed": 20260920004,
            "noisy_margin_temperature": 0.05,
            "tie_tolerance": 1e-10,
            "prompt_configs": {
                "cifar100": {"path": "configs/prompts/cifar100_openai_readme.json", "sha256": sha256_file(Path("configs/prompts/cifar100_openai_readme.json"))},
                "eurosat": {"path": "configs/prompts/eurosat_openai_ensemble_v1.json", "sha256": sha256_file(Path("configs/prompts/eurosat_openai_ensemble_v1.json"))},
                "cifar10": {"path": None, "identity": "registered built-in CIFAR-10 OpenAI ensemble"},
            },
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "exclusion_manifest_sha256": sha256_file(args.exclusion_manifest),
        },
    }

    for experiment_id, key, title in (
        ("EXP-20260920-019A", "019a", "sigma extension on the EXP-017 backbones"),
        ("EXP-20260920-019B", "019b", "backbone extension"),
    ):
        cells = expand_cells(scope[key], item_manifest)
        sigmas = sorted({cell["sigma"] for cell in cells})
        if experiment_id.endswith("019A"):
            primary_outcome = (
                "paired macro standard certified accuracy at r = sigma versus no_correction over the balanced "
                "12 extension cells (2 models x 3 datasets x sigma in {0.12,0.5}), equal weight per cell; "
                "shared-item class-stratified simultaneous max-absolute-deviation 95% band "
                "(100000 replicates, seed 2026091605)"
            )
            bridge_cells = (
                "the two CIFAR-10 sigma=0.25 cells are a separately reported secondary bridge and are excluded "
                "from the primary 12-cell macro"
            )
        else:
            primary_outcome = (
                "paired macro standard certified accuracy at r = sigma versus no_correction over all 18 "
                "backbone-extension cells (3 models x 3 datasets x sigma in {0.25,0.5}), equal weight per cell; "
                "shared-item class-stratified simultaneous max-absolute-deviation 95% band "
                "(100000 replicates, seed 2026091605)"
            )
            bridge_cells = None
        registration = {
            "schema_version": "satml2027.registration.v1",
            "experiment_id": experiment_id,
            "title": title,
            "registered_at": created,
            "scope": args.scope,
            "status": "preregistered_extension",
            "data_role": "per-cell; CIFAR-100/EuroSAT use extension reserve, CIFAR-10 is a newly registered train-split extension dataset",
            "final_test_access": False,
            "method_selection": False,
            "item_manifest_sha256": canonical_json_hash(item_manifest),
            "cells": cells,
            "sigmas": sigmas,
            "certification": certification_block(sigmas),
            "candidates": common_candidates,
            "primary_outcome": primary_outcome,
            "bridge_cells": bridge_cells,
            "prospective_practical_effect_region": {"absolute_ca": 0.005, "interpretation": "reporting region only; not an equivalence margin or TOST"},
            "secondary_outcomes": ["standard CA at 0.12/0.25/0.5", "certified-wrong rate", "abstention", "top-class share", "predicted-class Gini", "normalized prediction entropy", "smoothed accuracy", "clean accuracy"],
            "predictions": predictions_block(),
            "stop_rules": [
                "no candidate or control is added, removed or retuned after this file is hashed",
                "the scope ladder may only be descended (full -> no_sigma_012 -> core) and any descent is recorded with its reason and timestamp before the affected cells are run",
                "cells that do not finish are reported as not run; their absence is never used to select what is reported",
            ],
        }
        registration["registration_sha256"] = canonical_json_hash(registration)
        path = write_json_atomic(args.out / f"{experiment_id.lower()}.json", registration)
        (args.out / f"{experiment_id.lower()}.sha256").write_bytes(
            (registration["registration_sha256"] + "\n").encode("ascii")
        )
        written.append((experiment_id, len(cells), registration["registration_sha256"], path))

    n3c = {
        "schema_version": "satml2027.registration.v1",
        "experiment_id": "N3C-20260920-V3",
        "title": "prospectively preregistered fresh-noise diagnostic on newly registered extension-development inputs",
        "registered_at": created,
        "status": "preregistered_diagnostic",
        "data_role": "development_theory_diagnostic",
        "final_test_access": False,
        "method_selection": False,
        "candidates": common_candidates,
        "cells": [
            {
                "cell_id": f"{model_id}__{dataset_id}__sigma0.25",
                "model_id": model_id,
                "dataset_id": dataset_id,
                "sigma": 0.25,
                "item_count": int(item_manifest["datasets"][dataset_id]["splits"]["development"]["count"]),
                "class_count": int(item_manifest["datasets"][dataset_id]["class_count"]),
                "development_items_sha256": item_manifest["datasets"][dataset_id]["splits"]["development"]["item_ids_sha256"],
                "control_train_items_sha256": item_manifest["datasets"][dataset_id]["splits"]["control_train"]["item_ids_sha256"],
            }
            for model_id in ("openai-clip-vit-b32-quickgelu", "openai-clip-vit-l14-quickgelu")
            for dataset_id in ("cifar100", "eurosat")
        ],
        "inputs": {
            "items": "newly preregistered extension-development items drawn from the remaining Phase-3 reserve, disjoint from extension evaluation and control-training items",
            "banks": "the exact Phase-3 objective/interfaces re-estimated prospectively on the registered extension-development split with 64 draws per item at sigma=0.25, plus four controls fitted or constructed before fresh N3C sampling",
        },
        "sampling": {
            "sigma": 0.25,
            "draws_per_item": 512,
            "stream_role": DIAGNOSTIC_STREAM,
            "base_seed": 20260920003,
            "alpha_per_example": 0.001,
            "p5_permutation_seed": 20260920008,
            "noise_block_draws": NOISE_BLOCK,
            "store_normalized_features": True,
            "k_min_at_512": min_successes_for_radius(512, 0.001, 0.25, 0.25),
        },
        "predictions": [
            {"id": "P1", "statement": "every draw whose argmax changes from j to k satisfies |<z, t_j - t_k>| <= |n_k - n_j| (per-class-scale form for the tangent controls), tie tolerance 1e-6", "type": "conformance", "rule": "zero exceptions"},
            {"id": "P2", "statement": "orthogonal-complement candidates: report orthogonality residual, normalizer mismatch and P1 conformance of every float-level flip", "type": "frozen measurement", "rule": "no pass/fail threshold"},
            {"id": "P4a", "statement": "marginal Gaussian calibration of the fixed clean-anchor versus fixed clean runner-up signed noisy margin, 10 equal-mass bins", "type": "falsifiable", "rule": "mean absolute calibration error <= 0.05 on both CIFAR-100 cells"},
            {"id": "P5", "statement": "attractor geometry: Spearman rho between <t_k, mean noisy feature> and noisy vote share, with a 10000-draw label-permutation reference", "type": "falsifiable", "rule": "rho > 0 in at least 3 of 4 cells"},
            {"id": "P6", "statement": "descriptive noisy-margin anatomy including per-draw winner-runner-up margin quantiles", "type": "descriptive"},
            {"id": "P7", "statement": "flip budget versus the finite-budget threshold: max_j c'_j <= k_0 + E_F; report the partition {k_0 >= k_min}, {k_0 + E_F < k_min}, {undetermined} per candidate, cell and k_0 regime", "type": "structural", "rule": "reported; the phrase 'cannot be certified' is not used"},
            {"id": "P8", "statement": "new preregistered extension shared-translation controls, fitted on the disjoint control-training split, are frozen and hash-bound before fresh N3C noise", "type": "conformance + descriptive"},
            {"id": "P9", "statement": "for every paired draw and applicable candidate, report unchanged, useful wrong-to-correct, harmful correct-to-wrong, and wrong-to-wrong winner movement against ground truth", "type": "descriptive mechanism taxonomy"},
        ],
        "stop_rules": ["no certificate is issued", "no candidate is selected", "EXP-017 is not altered"],
    }
    n3c["registration_sha256"] = canonical_json_hash(n3c)
    path = write_json_atomic(args.out / "n3c_v3.json", n3c)
    (args.out / "n3c_v3.sha256").write_bytes((n3c["registration_sha256"] + "\n").encode("ascii"))
    written.append(("N3C-20260920-V3", len(n3c["cells"]), n3c["registration_sha256"], path))

    for experiment_id, cell_count, digest, path in written:
        print(f"{experiment_id:20s} cells={cell_count:3d} sha256={digest}  {path}")
    print("\nCommit these files (or timestamp the hashes) BEFORE launching any sampling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
