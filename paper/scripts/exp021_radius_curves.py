#!/usr/bin/env python3
"""EXP-021B descriptive certified-accuracy-versus-radius curves (D-146), from the custody-verified vote counts.

Not a registered analysis. It recomputes the registered quantity (standard certified accuracy in the sense of Cohen
et al., computed by the registered ``certify_from_counts``) on a fine radius grid, for four banks fixed on 2026-09-28
before any EXP-021 outcome existed, and pools every cell at each noise level (item-pooled, like the registered
primary estimand). Safeguards, all fail-closed:

* it runs only under the signed analysis-stage approval (``guard_ext.verify_approval``, stage "analysis");
* it reads exactly the files the run custody manifest lists (``custody_ext.verify_files_against_custody``) and checks
  every cell's labels and item order against the registered item lists;
* it requires the registered analysis output to exist for the same approval and custody manifest, and it refuses to
  write anything unless its values equal that output at every registered radius, in every cell, for every bank.

Output: analysis/exp021/021b/descriptive_radius_curves.json (read by scripts/generate_satml2027_paper_assets.py).
Usage (repository root, after the registered analysis):
  .venv/Scripts/python scripts/exp021_radius_curves.py --results results/satml2027ext/EXP-20260921-021B \\
      --custody results/satml2027ext/EXP-20260921-021B_RUN_CUSTODY.json \\
      --bank-manifest results/satml2027ext/host_returns/EXP021_BANK_MANIFEST.json \\
      --data-preflight results/satml2027ext/host_returns/EXP021_DATA_PREFLIGHT.json
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "satml2027"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from satml2027ext import analyze_ext, items_ext  # noqa: E402
from satml2027ext._common import read_json, sha256_file, write_json_atomic  # noqa: E402
from satml2027ext.guard_ext import load_registration, verify_approval  # noqa: E402
from common.certify import certify_from_counts  # noqa: E402

SCHEMA = "satml2027ext.descriptive_radius_curves.v1"
REGISTRATION = ROOT / "configs/satml2027ext/exp-20260921-021b.json"
ANALYSIS = ROOT / "analysis/exp021/021b/analysis_ext.json"
OUT = ROOT / "analysis/exp021/021b/descriptive_radius_curves.json"
CURVE_BANKS = (  # fixed before any outcome existed (D-146); never chosen from results
    "no_correction",
    "gr_clip_style_two_sided__coefficient_1",
    "control__learned_shared_translation_tangent",
    "control__lowrank_tangent_r8",
)
GRID_STEP = 0.005


def grid_for(sigma: float, trials: int, alpha: float) -> list[float]:
    """0 to just past the largest radius the budget can certify (all confirmation draws on one class)."""

    from common.certify import clopper_pearson_lower
    from scipy.stats import norm

    top = float(sigma) * float(norm.ppf(clopper_pearson_lower(trials, trials, alpha)))
    count = int(np.floor(top / GRID_STEP)) + 2
    return [round(index * GRID_STEP, 6) for index in range(count)]


def compute(registration: Mapping[str, Any], results: Path, *, cells_rows: Sequence[Mapping[str, Any]],
            not_run: Sequence[str], verified: Mapping[str, Mapping[str, Any]] | None,
            rows_by_cell: Mapping[str, Sequence[Mapping[str, Any]]] | None) -> dict[str, Any]:
    """Curves plus the registered-value check against ``cells_rows`` (the registered cells_ext.csv rows).

    ``verified``/``rows_by_cell`` are None only in tests.
    """

    certification = analyze_ext.certification_parameters(registration)
    alpha = float(certification["alpha_per_example"])
    trials = int(certification["confirmation_draws"])
    registered_radii = [float(value) for value in certification["reported_radii"]]
    expected_ids = analyze_ext.registered_candidate_ids(registration, registration["cells"][0])
    missing = [bank for bank in CURVE_BANKS if bank not in expected_ids]
    if missing:
        raise RuntimeError(f"curve banks absent from the registered family: {missing}")
    registered_rows = {(str(row["cell_id"]), str(row["candidate_id"])): row for row in cells_rows}
    not_run = set(not_run)
    pooled: dict[str, dict[str, list[np.ndarray]]] = {}
    by_dataset: dict[str, dict[str, list[np.ndarray]]] = {}
    counts: dict[str, int] = {}
    dataset_counts: dict[str, int] = {}
    checks = 0
    for cell in registration["cells"]:
        cell_id = cell["cell_id"]
        if cell_id in not_run:
            continue
        sigma = float(cell["sigma"])
        sigma_key = format(sigma, "g")
        data = analyze_ext.merge_cell_from_shards(
            cell, results, expected_ids, certification, registration=registration,
            registered_rows=None if rows_by_cell is None else rows_by_cell[cell_id],
            shard_set=analyze_ext.cell_shard_set(results, cell, verified))
        n = len(data["item_ids"])
        counts[sigma_key] = counts.get(sigma_key, 0) + n
        dataset_key = f"{cell['dataset_id']}__sigma{sigma_key}"
        dataset_counts[dataset_key] = dataset_counts.get(dataset_key, 0) + n
        radii_keys = sorted(set(registered_radii + [sigma]))
        for bank in CURVE_BANKS:
            position = expected_ids.index(bank)
            summary, outcome = certify_from_counts(
                raw_predictions=data["raw_predictions"][position], ground_truth=data["ground_truth"],
                selection_counts=data["selection_counts"][position],
                confirmation_counts=data["confirmation_counts"][position],
                sigma=sigma, alpha=alpha, radii=radii_keys)
            certified_correct = (~outcome["abstained"]) & (outcome["selected_classes"] == data["ground_truth"])
            radii = outcome["selected_class_radii"][certified_correct]
            pooled.setdefault(sigma_key, {}).setdefault(bank, []).append(radii)
            by_dataset.setdefault(dataset_key, {}).setdefault(bank, []).append(radii)
            row = registered_rows.get((cell_id, bank))
            if row is None:
                raise RuntimeError(f"the registered output has no row for {cell_id} / {bank}")
            for value in radii_keys:
                key = f"standard_certified_accuracy@{format(value, '.12g')}"
                mine = float(np.count_nonzero(radii >= value)) / n
                if abs(mine - float(summary[key])) > 1e-12 or abs(mine - float(row[key])) > 1e-12:
                    raise RuntimeError(f"{cell_id} / {bank} at r={value}: curve {mine} differs from the registered "
                                       f"value {row[key]} (or the certification summary {summary[key]})")
                checks += 1

    def curve(parts: list[np.ndarray], total: int, grid: list[float]) -> list[float]:
        radii = np.sort(np.concatenate(parts))
        # certified accuracy at r = share of all items certified correct with radius >= r
        return [float(radii.size - np.searchsorted(radii, value, side="left")) / total * 100.0 for value in grid]

    grids = {key: grid_for(float(key), trials, alpha) for key in counts}
    output = {
        "schema_version": SCHEMA,
        "descriptive": True,
        "note": ("not a registered analysis: the registered certification function on the custody-verified vote "
                 "counts, on a fine radius grid, for banks fixed before any outcome existed; equals the registered "
                 "output at every registered radius (checked)"),
        "registration_sha256": registration["registration_sha256"],
        "banks": list(CURVE_BANKS),
        "grid_step": GRID_STEP,
        "unit": "percent of items",
        "pooled": {key: {"items": counts[key], "grid": grids[key],
                         "curves": {bank: curve(pooled[key][bank], counts[key], grids[key]) for bank in CURVE_BANKS}}
                   for key in sorted(counts)},
        "by_dataset": {key: {"items": dataset_counts[key],
                             "curves": {bank: curve(by_dataset[key][bank], dataset_counts[key],
                                                    grids[key.split("__sigma")[1]]) for bank in CURVE_BANKS}}
                       for key in sorted(dataset_counts)},
        "registered_value_checks": checks,
        "not_run_cells": sorted(not_run),
    }
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--custody", required=True, type=Path)
    parser.add_argument("--bank-manifest", required=True, type=Path)
    parser.add_argument("--data-preflight", required=True, type=Path)
    parser.add_argument("--items", type=Path, default=ROOT / "results/satml2027ext/items")
    parser.add_argument("--analysis", type=Path, default=ANALYSIS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    from satml2027ext.custody_ext import verify_files_against_custody

    registration = load_registration(REGISTRATION)
    approval = verify_approval(registration, stage="analysis", approval_path=analyze_ext.DEFAULT_APPROVAL_PATH,
                               stage_files={"data_preflight": args.data_preflight, "bank_manifest": args.bank_manifest,
                                            "run_custody_manifest": args.custody})
    custody = read_json(args.custody)
    if custody.get("registration_sha256") != registration["registration_sha256"]:
        raise SystemExit("refused: the run custody manifest names another registration")
    verified = verify_files_against_custody(custody, args.results)
    report = read_json(args.analysis)
    if (report.get("registration_sha256") != registration["registration_sha256"]
            or report.get("approval_sha256") != approval["_approval_sha256"]
            or report.get("run_custody_sha256") != sha256_file(args.custody)):
        raise SystemExit("refused: the registered analysis output was not produced under this approval and custody")
    rows = {}
    for cell in registration["cells"]:
        if cell["cell_id"] in set(report.get("not_run_cells") or []):
            continue
        listed = items_ext.load_items(args.items, cell, "evaluation", registration)
        items_ext.verify_item_list(listed, cell, "evaluation")
        rows[cell["cell_id"]] = listed
    with (args.analysis.parent / "cells_ext.csv").open(encoding="utf-8", newline="") as handle:
        cells_rows = list(csv.DictReader(handle))  # floats were written with repr, so float() is exact
    output = compute(registration, args.results, cells_rows=cells_rows,
                     not_run=list(report.get("not_run_cells") or []), verified=verified, rows_by_cell=rows)
    output["cells_csv_sha256"] = sha256_file(args.analysis.parent / "cells_ext.csv")
    output.update({"approval_sha256": approval["_approval_sha256"], "run_custody_sha256": sha256_file(args.custody),
                   "analysis_sha256": sha256_file(args.analysis)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out, output)
    print(f"wrote {args.out} ({output['registered_value_checks']} registered-value checks passed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
