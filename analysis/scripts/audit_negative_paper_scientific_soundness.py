"""Audit the scientific interpretation of the immutable EXP-017 result.

This analysis is deliberately read-only.  It distinguishes the standard Cohen
certified-accuracy event from EXP-017's stricter raw-clean-anchored event and
quantifies Monte-Carlo censoring and mutually exclusive failure causes.  It
does not resample, select a new method, or rewrite the registered experiment.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/EXP-20260906-017"
OUTPUT = ROOT / "artifacts/negative_paper/exp017_scientific_soundness_audit_v1.json"
RADII = (0.0, 0.1, 0.25, 0.5)
SIGMA = 0.25
CONFIRMATION_SAMPLES = 4096


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rate(mask: torch.Tensor) -> float:
    return float(mask.double().mean().item())


def _plugin_radius(successes: torch.Tensor) -> torch.Tensor:
    """Uncertified plug-in radii, used only to audit finite-sample censoring."""
    values: list[float] = []
    normal = NormalDist()
    for success in successes.tolist():
        probability = float(success) / CONFIRMATION_SAMPLES
        if probability <= 0.5:
            values.append(0.0)
        elif probability >= 1.0:
            values.append(math.inf)
        else:
            values.append(SIGMA * normal.inv_cdf(probability))
    return torch.tensor(values, dtype=torch.float64)


def classify_failure_causes(
    *,
    selected: torch.Tensor,
    raw: torch.Tensor,
    ground_truth: torch.Tensor,
    abstained: torch.Tensor,
    certificate_radii: torch.Tensor,
    radius: float,
) -> dict[str, dict[str, int]]:
    """Return mutually exclusive standard and anchored failure partitions."""
    selected = selected.long()
    raw = raw.long()
    ground_truth = ground_truth.long()
    abstained = abstained.bool()
    certificate_radii = certificate_radii.double()

    standard = {
        "abstained": abstained,
        "nonabstaining_wrong_smoothed_class": (~abstained)
        & selected.ne(ground_truth),
        "correct_smoothed_class_but_radius_below_target": (~abstained)
        & selected.eq(ground_truth)
        & certificate_radii.lt(radius),
        "certified_correct_at_target": (~abstained)
        & selected.eq(ground_truth)
        & certificate_radii.ge(radius),
    }

    anchored = {
        "abstained": abstained,
        "nonabstaining_anchor_mismatch": (~abstained) & selected.ne(raw),
        "anchor_preserved_but_raw_class_wrong": (~abstained)
        & selected.eq(raw)
        & raw.ne(ground_truth),
        "anchor_correct_but_radius_below_target": (~abstained)
        & selected.eq(raw)
        & raw.eq(ground_truth)
        & certificate_radii.lt(radius),
        "anchored_certified_correct_at_target": (~abstained)
        & selected.eq(raw)
        & raw.eq(ground_truth)
        & certificate_radii.ge(radius),
    }

    expected = int(selected.numel())
    for name, partition in (("standard", standard), ("anchored", anchored)):
        total = sum(int(mask.sum().item()) for mask in partition.values())
        if total != expected:
            raise RuntimeError(f"{name} failure partition is not exhaustive: {total} != {expected}")
    return {
        name: {key: int(mask.sum().item()) for key, mask in partition.items()}
        for name, partition in (("standard", standard), ("anchored", anchored))
    }


def build_report() -> dict[str, Any]:
    manifest = _load_json(RESULT / "artifact_manifest.json")
    summary = _load_json(RESULT / "summary.json")
    selection = _load_json(RESULT / "selection.json")
    if manifest["final_test_accessed"] is not False or summary["final_test_accessed"] is not False:
        raise RuntimeError("EXP-017 reports final-test access")

    cell_paths = sorted((RESULT / "cells").iterdir())
    if len(cell_paths) != 12:
        raise RuntimeError(f"expected 12 EXP-017 cells, found {len(cell_paths)}")

    candidate_ids: list[str] | None = None
    per_candidate_cells: dict[str, list[dict[str, Any]]] = {}
    source_statistics_hashes: dict[str, str] = {}

    for cell in cell_paths:
        statistics_path = cell / "cohen_sufficient_statistics.pt"
        raw = torch.load(statistics_path, map_location="cpu", weights_only=True)
        ids = list(raw["candidate_ids"])
        if candidate_ids is None:
            candidate_ids = ids
        elif ids != candidate_ids:
            raise RuntimeError(f"candidate order differs in {cell.name}")
        source_statistics_hashes[cell.name] = _sha256(statistics_path)

        truth = torch.as_tensor(raw["ground_truth"]).long()
        raw_predictions = torch.as_tensor(raw["raw_predictions"]).long()
        for position, candidate_id in enumerate(ids):
            outcome = raw["outcomes"][candidate_id]
            selected = torch.as_tensor(outcome["selected_classes"]).long()
            successes = torch.as_tensor(outcome["selected_successes"]).long()
            radii = torch.as_tensor(outcome["certificate_radii"]).double()
            abstained = torch.as_tensor(outcome["abstained"]).bool()
            anchor_match = selected.eq(raw_predictions[position])
            correct_smoothed = selected.eq(truth)
            raw_correct = raw_predictions[position].eq(truth)
            plugin_radii = _plugin_radius(successes)

            standard_ca = {
                str(radius): _rate((~abstained) & correct_smoothed & radii.ge(radius))
                for radius in RADII
            }
            anchored_ca = {
                str(radius): _rate(
                    (~abstained)
                    & anchor_match
                    & raw_correct
                    & radii.ge(radius)
                )
                for radius in RADII
            }
            censoring = {
                str(radius): {
                    "plugin_pass_rate": _rate(correct_smoothed & plugin_radii.ge(radius)),
                    "exact_lcb_pass_rate": standard_ca[str(radius)],
                    "plugin_only_count": int(
                        (
                            correct_smoothed
                            & plugin_radii.ge(radius)
                            & ~((~abstained) & radii.ge(radius))
                        ).sum().item()
                    ),
                }
                for radius in RADII
            }
            per_candidate_cells.setdefault(candidate_id, []).append(
                {
                    "cell_id": cell.name,
                    "example_count": int(truth.numel()),
                    "standard_certified_accuracy": standard_ca,
                    "anchored_certified_accuracy": anchored_ca,
                    "standard_minus_anchored": {
                        str(radius): standard_ca[str(radius)] - anchored_ca[str(radius)]
                        for radius in RADII
                    },
                    "smoothed_accuracy": _rate(correct_smoothed),
                    "raw_clean_accuracy": _rate(raw_correct),
                    "anchor_agreement": _rate(anchor_match),
                    "abstention_rate": _rate(abstained),
                    "finite_sample_censoring": censoring,
                    "failure_causes_at_0.25": classify_failure_causes(
                        selected=selected,
                        raw=raw_predictions[position],
                        ground_truth=truth,
                        abstained=abstained,
                        certificate_radii=radii,
                        radius=0.25,
                    ),
                }
            )

    assert candidate_ids is not None
    if len(candidate_ids) != 18:
        raise RuntimeError(f"expected 18 candidates, found {len(candidate_ids)}")

    aggregates: dict[str, Any] = {}
    for candidate_id in candidate_ids:
        cells = per_candidate_cells[candidate_id]
        aggregate: dict[str, Any] = {
            "standard_certified_accuracy": {},
            "anchored_certified_accuracy": {},
            "standard_minus_anchored": {},
            "finite_sample_plugin_only_examples": {},
        }
        for radius in RADII:
            key = str(radius)
            standard = sum(row["standard_certified_accuracy"][key] for row in cells) / 12
            anchored = sum(row["anchored_certified_accuracy"][key] for row in cells) / 12
            aggregate["standard_certified_accuracy"][key] = standard
            aggregate["anchored_certified_accuracy"][key] = anchored
            aggregate["standard_minus_anchored"][key] = standard - anchored
            aggregate["finite_sample_plugin_only_examples"][key] = sum(
                row["finite_sample_censoring"][key]["plugin_only_count"] for row in cells
            )
        aggregate["raw_clean_accuracy"] = sum(row["raw_clean_accuracy"] for row in cells) / 12
        aggregate["smoothed_accuracy"] = sum(row["smoothed_accuracy"] for row in cells) / 12
        aggregate["anchor_agreement"] = sum(row["anchor_agreement"] for row in cells) / 12
        aggregate["abstention_rate"] = sum(row["abstention_rate"] for row in cells) / 12
        aggregates[candidate_id] = aggregate

    standard_ranking = sorted(
        candidate_ids,
        key=lambda candidate_id: (
            aggregates[candidate_id]["standard_certified_accuracy"]["0.25"],
            aggregates[candidate_id]["smoothed_accuracy"],
            aggregates[candidate_id]["raw_clean_accuracy"],
            candidate_id,
        ),
        reverse=True,
    )
    anchored_ranking = sorted(
        candidate_ids,
        key=lambda candidate_id: (
            aggregates[candidate_id]["anchored_certified_accuracy"]["0.25"],
            aggregates[candidate_id]["raw_clean_accuracy"],
            candidate_id,
        ),
        reverse=True,
    )

    historical_candidate = str(summary["selected_proposed_candidate_id"])
    no_correction = "no_correction"
    return {
        "schema_version": "negative_paper.exp017_scientific_soundness_audit.v1",
        "status": "PRIMARY_SAVED_DATA_AUDIT_PENDING_INDEPENDENT_REVIEW",
        "experiment_id": "EXP-20260906-017",
        "source_artifact_set_sha256": manifest["artifact_set_sha256"],
        "source_summary_sha256": _sha256(RESULT / "summary.json"),
        "source_statistics_sha256": source_statistics_hashes,
        "read_only": True,
        "new_sampling": False,
        "selection_or_result_rewrite": False,
        "final_test_accessed": False,
        "scope": {
            "models": 2,
            "datasets": 2,
            "folds": 3,
            "cells": 12,
            "candidates_per_cell": 18,
            "sigma": SIGMA,
            "selection_samples_per_example": 128,
            "confirmation_samples_per_example": CONFIRMATION_SAMPLES,
            "pointwise_alpha": 0.001,
        },
        "metric_semantics": {
            "standard_cohen_certified_accuracy": "nonabstaining smoothed selected class equals ground truth and certified radius reaches the target",
            "exp017_anchored_certified_accuracy": "standard event plus selected smoothed class equals the same candidate's raw-clean prediction",
            "paper_rule": "report both; never call the anchored metric standard Cohen certified accuracy",
        },
        "candidate_aggregates_equal_cell_weight": aggregates,
        "rankings_at_radius_0.25": {
            "standard_cohen": standard_ranking,
            "anchored_exp017": anchored_ranking,
        },
        "historical_selection": {
            "candidate_id": historical_candidate,
            "frozen_protocol_metric": "anchored_exp017",
            "standard_ca_at_0.25": aggregates[historical_candidate]["standard_certified_accuracy"]["0.25"],
            "anchored_ca_at_0.25": aggregates[historical_candidate]["anchored_certified_accuracy"]["0.25"],
            "standard_gain_over_no_correction_at_0.25": aggregates[historical_candidate]["standard_certified_accuracy"]["0.25"]
            - aggregates[no_correction]["standard_certified_accuracy"]["0.25"],
            "anchored_gain_over_no_correction_at_0.25": aggregates[historical_candidate]["anchored_certified_accuracy"]["0.25"]
            - aggregates[no_correction]["anchored_certified_accuracy"]["0.25"],
        },
        "per_candidate_cells": per_candidate_cells,
        "interpretation_constraints": [
            "EXP-017 can establish failure of its preregistered efficacy gate, not equivalence or universal ineffectiveness.",
            "The historical selection remains anchored and immutable even if the standard Cohen ranking differs.",
            "Plug-in radii are uncertified sensitivity diagnostics only.",
            "All confidence statements are pointwise per certificate; no simultaneous all-example guarantee is implied.",
            "Noise collapse and direction cancellation are mechanism-consistent diagnostics, not identified causal effects.",
        ],
        "saved_selection_record_sha256": _sha256(RESULT / "selection.json"),
        "saved_selection_id": selection["selected_proposed_candidate"]["candidate_id"],
    }


def main() -> None:
    report = build_report()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
