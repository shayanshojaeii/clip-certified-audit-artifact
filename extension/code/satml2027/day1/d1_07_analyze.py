#!/usr/bin/env python3
"""Day 1 - analysis: per-cell metrics, macro aggregates and preregistered contrasts.

Torch-free: it reads only the merged sufficient statistics, so a reviewer can
regenerate every number with numpy and scipy.

Outputs
  cells.csv              one row per (cell, bank)
  macro.csv              one row per bank, equal weight per cell
  contrasts.json         paired contrasts versus no_correction with the
                         shared-resample class-stratified simultaneous bands and per-cell
                         exact McNemar with Holm adjustment
  concentration.csv      hub diagnostics of the smoothed predictions

Usage:
  python day1/d1_07_analyze.py --merged results/satml2027/merged/EXP-20260920-019A \
      --registration configs/satml2027/exp-20260920-019a.json --out analysis/019a
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.certify import certify_from_counts, prediction_concentration  # noqa: E402
from common.hashing import read_json, write_json_atomic  # noqa: E402

BOOTSTRAP_SEED = 2026091605
BOOTSTRAP_REPLICATES = 100000
REFERENCE = "no_correction"


def macro_cell_mean(differences: np.ndarray, cell_ids: np.ndarray) -> float:
    return float(np.mean([differences[cell_ids == cell].mean() for cell in np.unique(cell_ids)]))


def simultaneous_bootstrap(differences, cell_ids, class_labels, item_ids=None, dataset_ids=None, *, replicates=BOOTSTRAP_REPLICATES,
                           seed=BOOTSTRAP_SEED, level=0.95):
    """Shared-item, class-stratified max-deviation band.

    One sampled (dataset, class, item) multiplicity is reused in every
    model/sigma cell containing that item and for every candidate contrast.
    """

    candidate_ids = sorted(differences)
    matrix = np.stack([differences[key].astype(np.float64) for key in candidate_ids])
    points = np.array([macro_cell_mean(row, cell_ids) for row in matrix])
    if item_ids is None or dataset_ids is None:
        item_ids = np.arange(len(cell_ids)).astype(str)
        dataset_ids = np.asarray(cell_ids).astype(str)
    identities = np.array([f"{d}\0{int(y)}\0{i}" for d, y, i in zip(dataset_ids, class_labels, item_ids)])
    unique_identities = np.unique(identities)
    positions = {identity: np.flatnonzero(identities == identity) for identity in unique_identities}
    strata = {}
    for identity in unique_identities:
        dataset, label, _ = identity.split("\0", 2)
        strata.setdefault((dataset, int(label)), []).append(identity)
    generator = np.random.default_rng(seed)
    values = np.empty((replicates, len(candidate_ids)), dtype=np.float64)
    for replicate in range(replicates):
        sampled_identities = []
        for identity_group in strata.values():
            sampled_identities.extend(generator.choice(identity_group, size=len(identity_group), replace=True))
        sampled = np.concatenate([positions[identity] for identity in sampled_identities])
        for column in range(len(candidate_ids)):
            values[replicate, column] = macro_cell_mean(matrix[column, sampled], cell_ids[sampled])
    maximum_deviation = np.max(np.abs(values - points[None, :]), axis=1)
    critical = float(np.quantile(maximum_deviation, level, method="higher"))
    intervals = {
        key: {"point_difference": float(point), "lower": float(point - critical),
              "upper": float(point + critical)}
        for key, point in zip(candidate_ids, points)
    }
    metadata = {"confidence_level": level, "replicates": replicates, "seed": seed,
                "critical_max_absolute_deviation": critical,
                "method": "shared_item_multiplicity_dataset_class_stratified_max_absolute_deviation"}
    return intervals, metadata


def exact_mcnemar(proposed: np.ndarray, comparator: np.ndarray) -> tuple[int, int, float]:
    only_proposed = int(np.sum((proposed == 1) & (comparator == 0)))
    only_comparator = int(np.sum((proposed == 0) & (comparator == 1)))
    discordant = only_proposed + only_comparator
    if discordant == 0:
        return only_proposed, only_comparator, 1.0
    smaller = min(only_proposed, only_comparator)
    lower_tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
    return only_proposed, only_comparator, min(1.0, 2.0 * lower_tail)


def holm(p_values):
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted, running = [0.0] * len(p_values), 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", required=True, type=Path)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--practical-region", type=float, default=None,
                        help="prospective descriptive practical-effect region; never an equivalence margin")
    args = parser.parse_args()

    registration = read_json(args.registration)
    certification = registration["certification"]
    alpha = float(certification["alpha_per_example"])
    if args.alpha is not None and args.alpha != alpha:
        raise RuntimeError("CLI alpha differs from the registered alpha")
    practical_region = float(registration["prospective_practical_effect_region"]["absolute_ca"])
    if args.practical_region is not None and args.practical_region != practical_region:
        raise RuntimeError("CLI practical region differs from the registered value")
    radii = list(certification["reported_radii"])
    args.out.mkdir(parents=True, exist_ok=True)

    rows, concentration_rows = [], []
    outcome_by_bank: dict[str, list[np.ndarray]] = {}
    cell_key, class_key, item_key, dataset_key = [], [], [], []
    for cell in registration["cells"]:
        path = args.merged / f"{cell['cell_id']}__merged.npz"
        if not path.is_file():
            raise FileNotFoundError(f"registered analysis cell is absent: {path}")
        data = np.load(path, allow_pickle=False)
        candidate_ids = list(data["candidate_ids"])
        ground_truth = data["ground_truth"]
        sigma = float(cell["sigma"])
        primary = format(sigma, ".12g")
        for position, candidate_id in enumerate(candidate_ids):
            summary, outcomes = certify_from_counts(
                raw_predictions=data["raw_predictions"][position],
                ground_truth=ground_truth,
                selection_counts=data["selection_counts"][position],
                confirmation_counts=data["confirmation_counts"][position],
                sigma=sigma,
                alpha=alpha,
                radii=sorted(set(radii + [sigma])),
            )
            rows.append({"cell_id": cell["cell_id"], "model_id": cell["model_id"], "dataset_id": cell["dataset_id"],
                         "sigma": sigma, "candidate_id": candidate_id,
                         "primary_estimand_cell": not (cell["dataset_id"] == "cifar10" and math.isclose(sigma, 0.25)
                                                       and registration["experiment_id"].endswith("019A")),
                         **summary})
            concentration_rows.append({"cell_id": cell["cell_id"], "sigma": sigma, "candidate_id": candidate_id,
                                       **prediction_concentration(outcomes["selected_classes"], int(cell["class_count"]))})
            is_bridge = cell["dataset_id"] == "cifar10" and math.isclose(sigma, 0.25) and registration["experiment_id"].endswith("019A")
            if not is_bridge:
                outcome_by_bank.setdefault(candidate_id, []).append(outcomes[f"standard@{primary}"].astype(np.int8))
        if not (cell["dataset_id"] == "cifar10" and math.isclose(sigma, 0.25) and registration["experiment_id"].endswith("019A")):
            cell_key.append(np.full(len(ground_truth), cell["cell_id"]))
            class_key.append(ground_truth)
            item_key.append(data["item_ids"].astype(str))
            dataset_key.append(np.full(len(ground_truth), cell["dataset_id"]))

    if not rows:
        print("no merged cells found")
        return 2
    cell_ids = np.concatenate(cell_key)
    class_labels = np.concatenate(class_key)
    item_ids = np.concatenate(item_key)
    dataset_ids = np.concatenate(dataset_key)

    with (args.out / "cells.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    with (args.out / "concentration.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in concentration_rows for key in row}))
        writer.writeheader()
        writer.writerows(concentration_rows)

    macro: dict[str, dict[str, float]] = {}
    numeric = [key for key in rows[0] if isinstance(rows[0][key], (int, float)) and key != "sigma"]
    for candidate_id in sorted({row["candidate_id"] for row in rows}):
        subset = [row for row in rows if row["candidate_id"] == candidate_id and row["primary_estimand_cell"]]
        macro[candidate_id] = {key: float(np.mean([row[key] for row in subset])) for key in numeric}
        macro[candidate_id]["cell_count"] = len(subset)
    with (args.out / "macro.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["candidate_id"] + sorted(next(iter(macro.values())))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate_id, values in sorted(macro.items()):
            writer.writerow({"candidate_id": candidate_id, **values})

    if REFERENCE not in outcome_by_bank:
        print(f"reference bank {REFERENCE} is absent; contrasts skipped")
        return 0
    reference = np.concatenate(outcome_by_bank[REFERENCE])
    contrast_inputs = {
        candidate_id: np.concatenate(parts).astype(np.float64) - reference.astype(np.float64)
        for candidate_id, parts in outcome_by_bank.items() if candidate_id != REFERENCE
    }
    simultaneous, simultaneous_meta = simultaneous_bootstrap(
        contrast_inputs, cell_ids, class_labels, item_ids=item_ids, dataset_ids=dataset_ids
    )
    contrasts = {}
    for candidate_id, parts in sorted(outcome_by_bank.items()):
        if candidate_id == REFERENCE:
            continue
        proposed = np.concatenate(parts)
        interval = simultaneous[candidate_id]
        per_cell, p_values = [], []
        for cell in np.unique(cell_ids):
            mask = cell_ids == cell
            only_p, only_c, p_value = exact_mcnemar(proposed[mask], reference[mask])
            per_cell.append({"cell_id": str(cell), "proposed_only": only_p, "comparator_only": only_c, "p_value": p_value})
            p_values.append(p_value)
        for entry, adjusted in zip(per_cell, holm(p_values)):
            entry["holm_adjusted_p_value"] = adjusted
        contrasts[candidate_id] = {
            "simultaneous_confidence_interval": interval,
            "cell_mcnemar": per_cell,
            "point_inside_prospective_practical_effect_region": bool(
                abs(interval["point_difference"]) <= practical_region
            ),
        }
    write_json_atomic(args.out / "contrasts.json", {
        "experiment_id": registration["experiment_id"],
        "registration_sha256": registration["registration_sha256"],
        "reference": REFERENCE,
        "primary_outcome": "standard certified accuracy at r = sigma",
        "simultaneous_band": simultaneous_meta,
        "prospective_practical_effect_region": {
            "lower": -practical_region,
            "upper": practical_region,
            "interpretation": "descriptive only; not an equivalence margin and not a TOST conclusion",
        },
        "contrasts": contrasts,
        "final_test_access": False,
    })

    print(f"{'candidate':56s} {'macroCA@r=sigma':>15s} {'delta':>9s} {'simultaneous 95% CI':>22s}")
    for candidate_id in sorted(contrasts):
        interval = contrasts[candidate_id]["simultaneous_confidence_interval"]
        ca = np.mean([row[f"standard_certified_accuracy@{format(row['sigma'], '.12g')}"]
                      for row in rows if row["candidate_id"] == candidate_id and row["primary_estimand_cell"]])
        print(f"{candidate_id:56s} {ca:15.4f} {interval['point_difference']:+9.4f} "
              f"[{interval['lower']:+.4f}, {interval['upper']:+.4f}]")
    print(f"\nwrote {args.out}/cells.csv, macro.csv, concentration.csv, contrasts.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
