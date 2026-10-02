"""Generate the immutable-result EXP-017 negative-result diagnostic record.

This script reads only saved EXP-017 artifacts. It does not sample, retune, or
rewrite the experiment. Exact clean-accuracy eligibility is adjudicated from
integer correct-count differences so an allowed two-point loss in a 50-example
cell cannot become 2.0000000000000018 through floating-point arithmetic.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/EXP-20260906-017"
OUTPUT = ROOT / "artifacts/gate3/phase3_exp017_negative_diagnostics_v1.json"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _tensor(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_clean_gate(
    reference_correct: list[int], candidate_correct: list[int], denominators: list[int]
) -> dict[str, Any]:
    if not (
        len(reference_correct) == len(candidate_correct) == len(denominators) == 12
    ):
        raise ValueError("the frozen gate requires exactly 12 aligned cells")
    drops = [
        Fraction(100 * (reference - candidate), denominator)
        for reference, candidate, denominator in zip(
            reference_correct, candidate_correct, denominators
        )
    ]
    macro = sum(drops, Fraction(0, 1)) / 12
    maximum = max(drops)
    return {
        "macro_drop_fraction": f"{macro.numerator}/{macro.denominator}",
        "macro_drop_percentage_points": float(macro),
        "maximum_drop_fraction": f"{maximum.numerator}/{maximum.denominator}",
        "maximum_drop_percentage_points": float(maximum),
        "eligible": macro <= 1 and maximum <= 2,
    }


def _proposal_key(value: dict[str, Any]) -> tuple[float, ...]:
    preference = {"clean_boundary_active": 0.0, "cohen_aligned_noisy_margin": 1.0}
    return (
        float(value["macro_certified_accuracy_at_0_25"]),
        float(value["macro_average_certified_radius"]),
        float(value["macro_raw_clean_accuracy"]),
        -float(value["macro_prototype_bank_frobenius_displacement"]),
        -float(value["step_or_coefficient"]),
        preference[str(value["objective_id"])],
    )


def _margins(
    per_cell: dict[str, dict[str, dict[str, Any]]], proposal: str, baseline: str
) -> dict[str, Any]:
    rows = []
    for cell_id, candidates in per_cell.items():
        model, dataset, fold_text = cell_id.split("__")
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "fold": int(fold_text.removeprefix("fold")),
                "difference": float(
                    candidates[proposal]["certified_accuracy"]["0.25"]
                    - candidates[baseline]["certified_accuracy"]["0.25"]
                ),
            }
        )
    models = sorted({row["model"] for row in rows})
    datasets = sorted({row["dataset"] for row in rows})
    model_margins = {
        model: sum(row["difference"] for row in rows if row["model"] == model)
        / sum(row["model"] == model for row in rows)
        for model in models
    }
    dataset_margins = {
        dataset: sum(row["difference"] for row in rows if row["dataset"] == dataset)
        / sum(row["dataset"] == dataset for row in rows)
        for dataset in datasets
    }
    return {
        "macro_difference": sum(row["difference"] for row in rows) / len(rows),
        "model_margins": model_margins,
        "dataset_margins": dataset_margins,
        "g3_4_all_strictly_positive": all(value > 0 for value in model_margins.values())
        and all(value > 0 for value in dataset_margins.values()),
    }


def build_report() -> dict[str, Any]:
    summary = _json(RESULT / "summary.json")
    selection = _json(RESULT / "selection.json")
    manifest = _json(RESULT / "artifact_manifest.json")
    if summary["final_test_accessed"] is not False or manifest["final_test_accessed"] is not False:
        raise RuntimeError("EXP-017 reports final-test access")

    per_cell: dict[str, dict[str, dict[str, Any]]] = {}
    correct_counts: dict[str, list[int]] = {}
    denominators: list[int] = []
    diagnostic_candidates = {
        "no_correction",
        "clean_boundary_active__step_0.0025",
        "clean_boundary_active__step_0.04",
        "gr_clip_style_two_sided__coefficient_1",
    }
    diagnostic_cells: dict[str, dict[str, Any]] = {}

    for cell in sorted((RESULT / "cells").iterdir()):
        raw = _tensor(cell / "cohen_sufficient_statistics.pt")
        metrics = _json(cell / "cell_metrics.json")
        by_id = {row["candidate_id"]: row for row in metrics["candidates"]}
        per_cell[cell.name] = by_id
        ids = list(raw["candidate_ids"])
        ground_truth = torch.as_tensor(raw["ground_truth"]).long()
        raw_predictions = torch.as_tensor(raw["raw_predictions"]).long()
        denominator = int(ground_truth.numel())
        denominators.append(denominator)
        diagnostic_cells[cell.name] = {}
        for position, candidate_id in enumerate(ids):
            correct_counts.setdefault(candidate_id, []).append(
                int(raw_predictions[position].eq(ground_truth).sum().item())
            )
            if candidate_id not in diagnostic_candidates:
                continue
            outcome = raw["outcomes"][candidate_id]
            selected = torch.as_tensor(outcome["selected_classes"]).long()
            operational_radii = torch.as_tensor(outcome["operational_radii"]).double()
            values, counts = torch.unique(selected, return_counts=True)
            top_position = int(torch.argmax(counts).item())
            diagnostic_cells[cell.name][candidate_id] = {
                "example_count": denominator,
                "raw_clean_accuracy": float(by_id[candidate_id]["raw_clean_accuracy"]),
                "smoothed_accuracy": float(by_id[candidate_id]["smoothed_accuracy"]),
                "anchor_agreement_rate": float(
                    selected.eq(raw_predictions[position]).double().mean().item()
                ),
                "operational_zero_radius_rate": float(
                    operational_radii.eq(0).double().mean().item()
                ),
                "certified_accuracy_at_0_25": float(
                    by_id[candidate_id]["certified_accuracy"]["0.25"]
                ),
                "top_smoothed_class": int(values[top_position].item()),
                "top_smoothed_class_fraction": float(
                    counts[top_position].double().item() / denominator
                ),
            }

    no_correct = correct_counts["no_correction"]
    exact_gates = {
        candidate_id: exact_clean_gate(no_correct, counts, denominators)
        for candidate_id, counts in correct_counts.items()
    }
    aggregates = selection["all_aggregates"]
    eligible_proposals = [
        dict(aggregates[candidate_id], exact_clean_gate=exact_gates[candidate_id])
        for candidate_id in aggregates
        if aggregates[candidate_id]["candidate_family"] == "proposal"
        and aggregates[candidate_id]["deployable_in_every_cell"]
        and exact_gates[candidate_id]["eligible"]
    ]
    exact_proposal = max(eligible_proposals, key=_proposal_key)
    mean_ids = sorted(
        candidate_id
        for candidate_id in aggregates
        if candidate_id.startswith("global_mean_centering__")
    )
    exact_eligible_means = [candidate_id for candidate_id in mean_ids if exact_gates[candidate_id]["eligible"]]
    forced_margins = {
        candidate_id: _margins(per_cell, exact_proposal["candidate_id"], candidate_id)
        for candidate_id in mean_ids
    }

    no_correction = aggregates["no_correction"]
    saved_proposal = aggregates[summary["selected_proposed_candidate_id"]]
    gr_clip = aggregates["gr_clip_style_two_sided__coefficient_1"]
    no_correction_cells = [
        diagnostic_cells[cell]["no_correction"] for cell in sorted(diagnostic_cells)
    ]
    return {
        "schema_version": "phase3.exp017.negative_diagnostics.v1",
        "status": "PRIMARY_POSTHOC_DIAGNOSTIC_FROM_SAVED_COUNTS",
        "experiment_id": "EXP-20260906-017",
        "source_artifact_set_sha256": manifest["artifact_set_sha256"],
        "source_summary_sha256": _sha256(RESULT / "summary.json"),
        "final_test_accessed": False,
        "no_sampling_or_result_rewrite": True,
        "independent_stage_b_conclusion": "SCIENTIFIC_NEGATIVE_VERIFIED_WITH_RECORDED_LIMITATIONS",
        "floating_boundary_adjudication": {
            "written_rule": "macro drop <= 1.0 points and every cell drop <= 2.0 points",
            "defect": "float evaluation can represent exact 2/100 as 2.0000000000000018 points",
            "saved_selected_proposal": summary["selected_proposed_candidate_id"],
            "exact_count_selected_proposal": exact_proposal["candidate_id"],
            "exact_selected_proposal_metrics": {
                key: exact_proposal[key]
                for key in (
                    "macro_certified_accuracy_at_0_25",
                    "macro_average_certified_radius",
                    "macro_raw_clean_accuracy",
                    "macro_prototype_bank_frobenius_displacement",
                )
            },
            "eligible_mean_centering_candidates": exact_eligible_means,
            "pivot_unchanged": len(exact_eligible_means) == 0,
        },
        "saved_flag_annotation": {
            "saved_exp018_may_be_prepared": bool(summary["exp018_may_be_prepared_from_this_result"]),
            "authoritative_stage_b_exp018_preparation_permitted": False,
            "reason": "production flag checked proposal existence; frozen Stage-B requires G3.1 and G3.4 against an eligible selected mean-centering baseline",
            "immutable_source_files_rewritten": False,
        },
        "macro_descriptive_comparison": {
            "no_correction_certified_accuracy_at_0_25": no_correction["macro_certified_accuracy_at_0_25"],
            "saved_proposal_certified_accuracy_at_0_25": saved_proposal["macro_certified_accuracy_at_0_25"],
            "saved_proposal_minus_no_correction": saved_proposal["macro_certified_accuracy_at_0_25"] - no_correction["macro_certified_accuracy_at_0_25"],
            "exact_proposal_certified_accuracy_at_0_25": exact_proposal["macro_certified_accuracy_at_0_25"],
            "gr_clip_certified_accuracy_at_0_25": gr_clip["macro_certified_accuracy_at_0_25"],
            "gr_clip_minus_exact_proposal": gr_clip["macro_certified_accuracy_at_0_25"] - exact_proposal["macro_certified_accuracy_at_0_25"],
            "gr_clip_clean_gate_eligible": exact_gates["gr_clip_style_two_sided__coefficient_1"]["eligible"],
        },
        "forced_exact_proposal_vs_each_mean_centering": forced_margins,
        "noise_collapse_no_correction_ranges": {
            "raw_clean_accuracy": [min(row["raw_clean_accuracy"] for row in no_correction_cells), max(row["raw_clean_accuracy"] for row in no_correction_cells)],
            "smoothed_accuracy": [min(row["smoothed_accuracy"] for row in no_correction_cells), max(row["smoothed_accuracy"] for row in no_correction_cells)],
            "anchor_agreement_rate": [min(row["anchor_agreement_rate"] for row in no_correction_cells), max(row["anchor_agreement_rate"] for row in no_correction_cells)],
            "operational_zero_radius_rate": [min(row["operational_zero_radius_rate"] for row in no_correction_cells), max(row["operational_zero_radius_rate"] for row in no_correction_cells)],
            "top_smoothed_class_fraction": [min(row["top_smoothed_class_fraction"] for row in no_correction_cells), max(row["top_smoothed_class_fraction"] for row in no_correction_cells)],
        },
        "cell_diagnostics": diagnostic_cells,
        "interpretation_boundary": {
            "established": [
                "the saved global corrections are nearly certificate-neutral in the tested frozen regime",
                "the tested noise regime has low anchor agreement and frequent operational zero radii",
                "the frozen positive efficacy path fails under both saved-float and exact-count gate readings",
            ],
            "hypotheses_requiring_new_experiments": [
                "class-specific corrections avoid cancellation",
                "image instability is the unique or dominant causal bottleneck",
                "a noise-aware correction improves Cohen certified accuracy",
                "a different sigma improves the radius-accuracy tradeoff",
            ],
        },
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
