#!/usr/bin/env python3
"""Day 1 - build every prototype bank a cell will be scored against.

For one (model, dataset, sigma) cell this writes a single `*_banks.pt` holding:
  * the 18 frozen EXP-017 candidates, rebuilt by the ported rule from that
    cell's development artifacts, and
  * the registered control banks (learned shared translation in both forms, the
    training-free noisy class mean, and the rank-8 tangent adapter).

`--verify-against-exp017` is the custody proof: it rebuilds an existing EXP-017
cell from its EXP-016 inputs and compares the 18 tensor hashes with the banks
saved inside EXP-017.  If those hashes match, the rule used for the extension is
demonstrably the rule that produced the audit.

Usage (build):
  python day1/d1_03_build_banks.py --development results/satml2027/development \
      --items results/satml2027/items --cell openai-clip-vit-b32-quickgelu__cifar100__sigma0.5 \
      --registration configs/satml2027/exp-20260920-019a.json --out results/satml2027/banks

Usage (verify):
  python day1/d1_03_build_banks.py --verify-against-exp017 \
      --exp016 results/EXP-20260906-016 --exp017 results/EXP-20260906-017
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.banks import (  # noqa: E402
    GR_CLIP_IMAGE_TRANSFORM,
    operator_parameters,
    build_eighteen_candidate_banks,
    low_rank_tangent_bank,
    noisy_class_mean_bank,
    normalizer_statistics,
    pure_shared_translation,
    tangent_shared_translation,
)
from common.hashing import read_json, tensor_scientific_hash, write_json_atomic  # noqa: E402

FROZEN_STEPS = [0.0025, 0.005, 0.01, 0.02, 0.04]
FROZEN_COEFFICIENTS = [0.25, 0.5, 1.0]
PORTABLE_REPRODUCTION_ATOL = 2e-6


def verify_against_exp017(exp016: Path, exp017: Path, *, atol: float = PORTABLE_REPRODUCTION_ATOL) -> int:
    """Rebuild saved banks with a frozen cross-runtime numerical contract.

    Exact hashes are reported. CPU/GPU and torch-SVD kernels may differ at the
    last float32 bits for mean/projection constructions, so scientific custody
    requires identical candidate IDs and max absolute tensor error <= 2e-6.
    """
    cells = sorted(path.name for path in (exp016 / "cells").iterdir() if path.is_dir())
    failures, checked, exact_hash_matches, tolerance_matches = [], 0, 0, 0
    for cell_id in cells:
        source = exp016 / "cells" / cell_id
        saved_path = exp017 / "cells" / cell_id / "candidate_prototype_banks.pt"
        if not saved_path.is_file():
            continue
        prototypes = torch.load(source / "text_prototypes.pt", map_location="cpu", weights_only=True).to(torch.float32).contiguous()
        features = torch.load(source / "clean_features.pt", map_location="cpu", weights_only=True)
        directions = torch.load(source / "candidate_directions.pt", map_location="cpu", weights_only=True)
        rebuilt = build_eighteen_candidate_banks(
            prototypes=prototypes,
            development_clean_features=features,
            clean_direction=directions["clean_direction"],
            noisy_direction=directions["noisy_direction"],
            common_steps=FROZEN_STEPS,
            gap_coefficients=FROZEN_COEFFICIENTS,
        )
        saved = torch.load(saved_path, map_location="cpu", weights_only=True)
        if set(saved) != set(rebuilt):
            failures.append(f"{cell_id}: candidate id sets differ")
            continue
        for candidate_id, tensor in saved.items():
            ours = rebuilt[candidate_id]["prototypes"]
            if tensor_scientific_hash(tensor) != tensor_scientific_hash(ours):
                delta = float((tensor.to(torch.float64) - ours.to(torch.float64)).abs().max())
                if delta > atol:
                    failures.append(f"{cell_id}/{candidate_id}: max|delta|={delta:.3e} > atol={atol:.3e}")
                else:
                    tolerance_matches += 1
            else:
                exact_hash_matches += 1
        checked += 1
    print(f"verified {checked} EXP-017 cells; exact_hash={exact_hash_matches}; "
          f"portable_tolerance={tolerance_matches}; failures={len(failures)}; atol={atol:.1e}")
    for failure in failures[:20]:
        print("  " + failure)
    return 0 if not failures and checked else 1


def build_cell(args) -> int:
    development = torch.load(args.development / f"{args.cell}__development.pt", map_location="cpu", weights_only=False)
    registration = read_json(args.registration)
    if development.get("registration_sha256") != registration["registration_sha256"]:
        raise RuntimeError("development artifact registration hash mismatch")
    prototypes = development["prototypes"].to(torch.float32)
    banks = build_eighteen_candidate_banks(
        prototypes=prototypes,
        development_clean_features=development["clean_features"].to(torch.float32),
        clean_direction=development["clean_direction"],
        noisy_direction=development["noisy_direction"],
        common_steps=FROZEN_STEPS,
        gap_coefficients=FROZEN_COEFFICIENTS,
    )
    image_mean = development["clean_features"].to(torch.float32).mean(dim=0)
    for entry in banks.values():
        if entry["metadata"].get("image_transform") == GR_CLIP_IMAGE_TRANSFORM:
            entry["image_mean"] = image_mean

    controls, control_operators = {}, {}
    if not args.controls or not args.controls.is_file():
        raise FileNotFoundError("registered trained controls are required")
    if not args.control_train_features or not args.control_train_features.is_file():
        raise FileNotFoundError("registered control-training feature cache is required")
    if args.controls and args.controls.is_file():
        trained = torch.load(args.controls, map_location="cpu", weights_only=False)
        if trained.get("registration_sha256") != registration["registration_sha256"]:
            raise RuntimeError("control checkpoint registration hash mismatch")
        delta = trained["shared_delta"].to(torch.float32)
        controls["control__learned_shared_translation_tangent"] = tangent_shared_translation(prototypes, delta)
        control_operators["control__learned_shared_translation_tangent"] = operator_parameters(
            prototypes, shared_vector=delta, row_scales=(1.0 - prototypes @ delta)
        )
        controls["control__learned_shared_translation_pure"] = pure_shared_translation(prototypes, delta)
        control_operators["control__learned_shared_translation_pure"] = operator_parameters(
            prototypes, shared_vector=delta
        )
        if "lowrank_left" in trained:
            controls["control__lowrank_tangent_r8"] = low_rank_tangent_bank(
                prototypes, trained["lowrank_left"].to(torch.float32), trained["lowrank_right"].to(torch.float32)
            )
    if args.control_train_features and args.control_train_features.is_file():
        cache = torch.load(args.control_train_features, map_location="cpu", weights_only=False)
        controls["control__noisy_class_mean"] = noisy_class_mean_bank(
            cache["noisy_features"], cache["labels"], int(prototypes.shape[0])
        )
    for candidate_id, tensor in controls.items():
        banks[candidate_id] = {
            "prototypes": tensor.contiguous(),
            "metadata": {
                "candidate_id": candidate_id,
                "family": "registered_control",
                "objective_id": None,
                "step_or_coefficient": None,
                "image_transform": "identity",
                "source_prototype_sha256": tensor_scientific_hash(prototypes),
                "output_prototype_sha256": tensor_scientific_hash(tensor),
                "role": "registered_control",
                "normalizer_statistics": normalizer_statistics(prototypes, tensor),
                "operator": control_operators.get(candidate_id, {"known": False}),
            },
        }

    if len(banks) != 22:
        raise RuntimeError(f"expected exactly 18 frozen plus 4 control banks, found {len(banks)}")
    if len(controls) != 4:
        raise RuntimeError(f"expected exactly four registered controls, found {len(controls)}")

    out = args.out / f"{args.cell}__banks.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cell_id": args.cell,
            "banks": banks,
            "registration_sha256": registration["registration_sha256"],
            "frozen_steps": FROZEN_STEPS,
            "frozen_coefficients": FROZEN_COEFFICIENTS,
            "final_test_access": False,
        },
        out,
    )
    write_json_atomic(
        args.out / f"{args.cell}__banks.json",
        {
            "cell_id": args.cell,
            "registration_sha256": registration["registration_sha256"],
            "bank_count": len(banks),
            "frozen_grid_count": sum(1 for entry in banks.values() if entry["metadata"]["role"] == "frozen_exp017_grid"),
            "control_count": sum(1 for entry in banks.values() if entry["metadata"]["role"] == "registered_control"),
            "banks": {key: entry["metadata"] for key, entry in sorted(banks.items())},
            "final_test_access": False,
        },
    )
    print(f"{args.cell}: {len(banks)} banks ({len(banks) - len(controls)} frozen + {len(controls)} controls) -> {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-against-exp017", action="store_true")
    parser.add_argument("--exp016", type=Path, default=Path("results/EXP-20260906-016"))
    parser.add_argument("--exp017", type=Path, default=Path("results/EXP-20260906-017"))
    parser.add_argument("--reproduction-atol", type=float, default=PORTABLE_REPRODUCTION_ATOL)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--items", type=Path)
    parser.add_argument("--cell", type=str)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--controls", type=Path, default=None)
    parser.add_argument("--control-train-features", type=Path, default=None)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.verify_against_exp017:
        if args.reproduction_atol != PORTABLE_REPRODUCTION_ATOL:
            raise RuntimeError(f"reproduction tolerance is frozen at {PORTABLE_REPRODUCTION_ATOL}")
        return verify_against_exp017(args.exp016, args.exp017, atol=args.reproduction_atol)
    for required in ("development", "cell", "registration", "out"):
        if getattr(args, required) is None:
            parser.error(f"--{required.replace('_', '-')} is required when building")
    return build_cell(args)


if __name__ == "__main__":
    raise SystemExit(main())
