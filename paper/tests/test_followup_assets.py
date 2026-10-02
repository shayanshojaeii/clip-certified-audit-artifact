"""EXP-021 result macros and table (D-145): the generator's reading of the registered analysis reports.

Synthetic reports shaped like the registered analysis outputs (satml2027ext/analyze_ext.py) are written under a
temporary root for the three real registrations; the test checks every macro the manuscript uses, the table, and
the refusals (wrong registration, overridden replicates, incomplete primary family).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_satml2027_paper_assets as gen  # noqa: E402

PRIMARY = ["gr_clip_style_two_sided__coefficient_1", "text_only_centering__coefficient_1",
           "image_only_centering__coefficient_1", "clean_boundary_active__step_0.16",
           "cohen_aligned_noisy_margin__step_0.16"]
MODELS = ["openai-clip-vit-b32-quickgelu", "openai-clip-vit-l14-quickgelu", "openclip-vit-b32-laion2b"]


def registrations() -> dict:
    return {key: json.loads((ROOT / rel).read_text(encoding="utf-8")) for key, rel in gen.FU_REGISTRATION_RELS.items()}


def contrast(point: float) -> dict:
    """A family contrast: item-pooled (primary estimand) and equal-cell (the audit's estimand) with intervals."""
    return {"point_points": point, "ci95_unadjusted_lower": point - 1.0, "ci95_unadjusted_upper": point + 1.0,
            "macro_point_points": point + 0.1, "macro_ci95_unadjusted_lower": point - 0.9,
            "macro_ci95_unadjusted_upper": point + 1.1, "discordance": 0.05}


def decomposition() -> dict:
    return {coefficient: {"two_sided": 1.5, "text_only": 0.25, "image_only": 1.0, "interaction_point": 0.25,
                          "interaction_ci95_unadjusted": [-0.25, 0.75]} for coefficient in ("0.25", "0.5", "1")}


def q5_block() -> dict:
    """Q5 per pooled family: every control against no correction and against two-sided centering."""
    return {control: {"point_points": 12.0 if "lowrank" in control else 0.1, "direction": "positive",
                      "minus_comparator_point": 10.5 if "lowrank" in control else -1.4,
                      "minus_comparator_ci95_unadjusted": [9.5, 11.5] if "lowrank" in control else [-2.4, -0.4]}
            for control in gen.FU_CONTROLS}


STEPS = (0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32)


def step_response(cells: list[str], items: int) -> list[dict]:
    """Q3 step response rows per cell and proposal bank; counts are over items x 4,096 confirmation draws and nested
    (changed <= in budget <= draws), and the budget is the same for both objectives, as in the registered outputs."""
    rows = []
    for cell_id in cells:
        for family in ("clean_boundary_active", "cohen_aligned_noisy_margin"):
            for index, step in enumerate(STEPS):
                draws = items * 4096
                changed = draws * (index + 1) // 100
                rows.append({"candidate_id": f"{family}__step_{step}", "cell_id": cell_id, "step": step, "items": items,
                             "flip_changed_draws": changed, "flip_loose_budget_draws": min(draws, draws * (index + 1) // 5),
                             "flip_useful_draws": changed // 2, "flip_harmful_draws": changed // 3,
                             "containment_applicable": True, "loose_applicable": True})
    return rows


def report_a(reg: dict) -> dict:
    block = {"verdict": "reproduced_smaller", "point_points": 1.2, "ci95_unadjusted_lower": 0.4, "ci95_unadjusted_upper": 2.0,
             "reference_points": 89 / 30, "fresh_minus_saved_points": 1.2 - 89 / 30, "p_direction": 0.0004,
             "p_shortfall": 0.012}
    return {"schema_version": gen.FU_ANALYSIS_SCHEMA, "registration_sha256": reg["registration_sha256"],
            "inference": {"replicates_used": 100000, "replicates_overridden": False}, "not_run_cells": [],
            "families": {"pooled__sigma0.25": {"decomposition": decomposition(), "contrasts": {}}},
            "questions": {"Q1": {"primary_equal_cell_macro": block, "secondary_item_pooled": dict(block, point_points=0.5),
                                 "registered_prediction": "reproduced",
                                 "by_dataset_descriptive": {"cifar100__sigma0.25": {"equal_cell_macro_point": 0.5},
                                                            "eurosat__sigma0.25": {"equal_cell_macro_point": 5.0}}},
                          "Q3": {"verdict": "conforms", "exceptions": [],
                                 "step_response": step_response([cell["cell_id"] for cell in reg["cells"]], 250)},
                          "Q4": {"verdict": "conforms", "changed_projected_gap_draws": 3,
                                 "tie_budget": {"max_margin_bound": 1.0236e-05}},
                          "Q5": {"pooled__sigma0.25": q5_block()}}}


DATASETS_B = ("cifar100_test", "eurosat_sealed", "imagenette")


def report_b(reg: dict) -> dict:
    """analyze's shape: the pooled family and one family per dataset at each sigma; Q7 rows for every cell."""
    families, q6, q5 = {}, {}, {}
    for sigma in ("0.12", "0.25"):
        for name in ("pooled",) + DATASETS_B:
            key = f"{name}__sigma{sigma}"
            contrasts = {candidate: contrast(0.3) for candidate in reg["candidates"]["candidate_ids"] if candidate != "no_correction"}
            contrasts["global_mean_centering__coefficient_1"] = contrast(-4.0)
            contrasts["clean_boundary_active__step_0.32"] = contrast(0.9)
            contrasts["control__lowrank_tangent_r8"] = contrast(12.0)
            families[key] = {"decomposition": decomposition(), "contrasts": contrasts}
            q6[key] = {candidate: {"direction": "unresolved", "magnitude": "practically_equivalent", "point": 0.3,
                                   "ci95_unadjusted": [-0.7, 1.3]} for candidate in PRIMARY}
            q6[key][PRIMARY[0]] = {"direction": "positive", "magnitude": "inconclusive", "point": 1.5,
                                   "ci95_unadjusted": [0.5, 2.5]}
        q5[f"pooled__sigma{sigma}"] = q5_block()
    regime = [{"cell_id": cell["cell_id"], "smoothed_accuracy": 0.5 + 0.01 * i, "primary_standard_ca": 0.3 + 0.01 * i,
               "abstention_rate": 0.1} for i, cell in enumerate(reg["cells"])]
    eight = [{"cell_id": f"{model}__imagenette__sigma0.25", "standard_ca@0.25": 0.6, "standard_ca@0.5": 0.4} for model in MODELS]
    gate = {candidate: {"macro_drop_points": -12.0 if "lowrank" in candidate else 1.5, "flagged": "mean" in candidate,
                        "flagged_cells": [reg["cells"][0]["cell_id"]] if "mean" in candidate else []}
            for candidate in list(gen.FU_CONTROLS) + PRIMARY}
    return {"schema_version": gen.FU_ANALYSIS_SCHEMA, "registration_sha256": reg["registration_sha256"],
            "inference": {"replicates_used": 100000, "replicates_overridden": False}, "not_run_cells": [],
            "families": families, "clean_gate": {"candidates": gate},
            "questions": {"Q6": q6, "Q5": q5, "Q7": {"identity_regime": regime}, "Q8": {"imagenette_identity": eight},
                          "Q3": {"verdict": "conforms", "exceptions": [],
                                 "step_response": step_response([cell["cell_id"] for cell in reg["cells"]], 1000)},
                          "Q4": {"verdict": "conforms", "changed_projected_gap_draws": 0,
                                 "tie_budget": {"max_margin_bound": 1.0e-05}}}}


def report_c(reg: dict) -> dict:
    """analyze_budget's shape: one Q9 row per (registered cell, bank)."""
    rows = []
    for index, cell in enumerate(reg["cells"]):
        rows.append({"cell_id": cell["cell_id"], "candidate_id": "no_correction", "items": 100,
                     "certified_correct_4096": 25 + index, "certified_correct_100000": 28 + index,
                     "gained_with_100000": 4, "lost_with_100000": 1, "changed": 5})
        rows.append({"cell_id": cell["cell_id"], "candidate_id": "gr_clip_style_two_sided__coefficient_1", "items": 100,
                     "certified_correct_4096": 26, "certified_correct_100000": 30,
                     "gained_with_100000": 5, "lost_with_100000": 1, "changed": 6})
    return {"schema_version": gen.FU_ANALYSIS_SCHEMA, "registration_sha256": reg["registration_sha256"],
            "questions": {"Q9": {"verdict": "descriptive", "rows": rows, "prefix_identity_exact_everywhere": True,
                                 "certificate_input_identity_everywhere": True}}}


def write_cells_csv(path: Path, report: dict, *, shift: float = 0.0) -> None:
    """The registered cells table that analyze_ext writes next to the 021B report: the identity row of every cell and
    one row per primary bank (standard CA two points above the identity, clean accuracy five points below)."""
    import csv

    fields = ["abstention_rate", "candidate_id", "cell_id", "clean_accuracy", "primary_certified_wrong_rate",
              "primary_standard_ca"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["questions"]["Q7"]["identity_regime"]:
            writer.writerow({"abstention_rate": row["abstention_rate"], "candidate_id": "no_correction",
                             "cell_id": row["cell_id"], "clean_accuracy": 0.6, "primary_certified_wrong_rate": 0.05,
                             "primary_standard_ca": row["primary_standard_ca"] + shift})
            for candidate in PRIMARY:
                writer.writerow({"abstention_rate": row["abstention_rate"], "candidate_id": candidate,
                                 "cell_id": row["cell_id"], "clean_accuracy": 0.55, "primary_certified_wrong_rate": 0.05,
                                 "primary_standard_ca": row["primary_standard_ca"] + 0.02})


def write_compute_summary(root: Path) -> None:
    path = root / gen.FU_COMPUTE_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": "satml2027ext.compute_summary.v1",
                                "main_host": {"hours": 15.0}, "extra_host": {"hours": 15.44}}), encoding="utf-8")


def run(tmp_path: Path, reports: dict, *, compute: bool = False) -> tuple[gen.Macros, dict]:
    if compute:
        write_compute_summary(tmp_path)
    for key, report in reports.items():
        path = tmp_path / gen.FU_ANALYSIS_RELS[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report), encoding="utf-8")
        if key == "B":
            write_cells_csv(path.parent / "cells_ext.csv", report)
    macros, notes, tables = gen.Macros(), {}, {}
    gen.followup_results(gen.Sources(tmp_path), macros, notes, tables, registrations())
    return macros, tables


def test_every_result_macro_and_the_table(tmp_path):
    regs = registrations()
    macros, tables = run(tmp_path, {"A": report_a(regs["A"]), "B": report_b(regs["B"]), "C": report_c(regs["C"])},
                         compute=True)
    values = macros.values
    assert not macros.missing
    assert (values["FUMainHours"], values["FUExtraHours"]) == ("15.0", "15.4")
    assert values["FUQOneVerdict"] == "reproduced but smaller"
    assert values["FUQOnePoint"] == "+1.20" and values["FUQOneSaved"] == "+2.97" and values["FUQOneDiff"] == "-1.77"
    assert values["FUQOnePDirection"] == "$p<0.001$" and values["FUQOnePShortfall"] == "$p=0.012$"
    assert values["FUQTwoAText"] == "+0.25" and values["FUQTwoAInter"] == "+0.25"
    assert values["FUQFourChangedA"] == "3" and values["FUNotRunA"] == "0"
    assert (values["FUQOneEurosat"], values["FUQOneCifar"]) == ("+5.00", "+0.50")
    assert values["FUQFourMarginA"] == "1.03\\times10^{-5}" and values["FUQFourMarginB"] == "1.00\\times10^{-5}"
    assert (values["FUQFiveClassLo"], values["FUQFiveClassHi"]) == ("+0.10", "+12.00")
    assert values["FUQSixMaxBoth"] == "+0.90" and values["FUGREurosatHighPoint"] == "+0.30"
    assert values["FUGRLowPoint"] == "+1.50" and values["FUGRLowMag"] == "inconclusive"
    assert (values["FUQSixHighN"], values["FUQSixHighEquiv"], values["FUQSixHighIncon"]) == ("5", "4", "1")
    assert values["FUQSixHighMaxPoint"] == "+0.90" and values["FUQSixHighMinPoint"] == "-4.00"
    assert values["FUQFiveLowLowrankPoint"] == "+12.00"
    low = [row["smoothed_accuracy"] for row in report_b(regs["B"])["questions"]["Q7"]["identity_regime"]
           if row["cell_id"].endswith("__sigma0.12")]
    assert len(low) == 9
    assert (values["FUQSevenLowSmoothedLo"], values["FUQSevenLowSmoothedHi"]) == (
        f"{100 * min(low):.2f}", f"{100 * max(low):.2f}")
    outcomes = tables["fig_followup_outcomes.dat"].strip().splitlines()
    assert "\\def\\FUOutcomesReady{1}" in tables["fig_followup_outcomes_axis.tex"] and len(outcomes) == 7
    correct, wrong, below, abstain = (float(value) for value in outcomes[1].split()[1:])
    assert abs(correct + wrong + below + abstain - 100.0) < 1e-6 and abs(wrong - 5.0) < 1e-9 and abs(abstain - 10.0) < 1e-9
    datasets = tables["table_followup_datasets.tex"]
    assert "tab:followup-datasets" in datasets and gen.Macros.PLACEHOLDER not in datasets and "(underpowered)" in datasets
    assert "tab:followup-design" in tables["table_followup_design.tex"]
    assert values["FUQEightLBthirtytwoHalf"] == "40.00"
    assert len(regs["C"]["cells"]) == 6  # the sums below are over the six registered 021C cells
    assert (values["FUQNineItems"], values["FUQNineIdFour"], values["FUQNineIdFull"]) == ("600", "165", "183")
    assert (values["FUQNineIdGained"], values["FUQNineIdLost"]) == ("24", "6")
    assert (values["FUQNineAllCerts"], values["FUQNineAllChanged"]) == ("1,200", "66")
    assert values["FUQNineMaxChanged"] == "36" and values["FUQNineMaxId"] and values["FUQNineIdentity"] == "yes"
    table = tables["table_followup.tex"]
    assert "tab:followup" in table and gen.Macros.PLACEHOLDER not in table
    assert "$+1.50$ [$+0.50$, $+2.50$]" not in table  # the table uses the family contrasts' intervals
    # item-pooled and equal-cell intervals, then the Holm direction and magnitude classes of the primary family
    assert ("Two-sided centering $c=1$ & $+0.30$ [$-0.70$, $+1.30$] & $+0.40$ [$-0.60$, $+1.40$] & 5.0 & pos. & inconcl."
            in table)  # the registered discordance stands next to the intervals
    assert "Projected gap $c=1$ & $+0.30$ [$-0.70$, $+1.30$] & $+0.40$ [$-0.60$, $+1.40$] & 5.0 & -- & --" in table
    assert "$d{=}5.0$" in tables["table_followup_datasets.tex"]
    assert values["FUGRLowMacroPoint"] == "+0.40" and values["FUGRHighMacroUpper"] == "+1.40"
    assert values["FUQSixLowPositiveIds"] == "two-sided centering at $c=1$" and values["FUQSixLowNegativeIds"] == "none"
    # per-backbone two-sided changes on sealed EuroSAT and the clean cost of the largest dataset-family gain (cells table)
    assert all(values[f"FUGREurosat{token}{model}"] == "+2.00" for _, token in gen.FU_SIGMA_TOKENS
               for model in gen.FU_MODEL_TOKENS.values())
    assert values["FUQSixDsMaxFamilyClean"] == "-5.00"
    assert values["FUQFiveLowrankClean"] == "+12.00" and values["FUQFiveNoisyMeanCleanFlagged"] == "1"
    assert values["FUQOnePredicted"] == "reproduced"
    # the Q3 step response on the audit items: 20% of draws in budget at the smallest step, all of them at 0.04
    assert (values["FUQThreeStepMin"], values["FUQThreeBudgetLo"], values["FUQThreeBudgetFour"]) == ("0.0025", "20.0", "100.0")
    assert (values["FUQThreeChangedLo"], values["FUQThreeChangedFour"]) == ("1.00", "5.00")
    assert values["FUQFiveALowrankPoint"] == "+12.00" and values["FUQFiveASharedDir"] == "positive"
    registered = tables["table_followup_registered.tex"]
    assert "tab:followup-registered" in registered and gen.Macros.PLACEHOLDER not in registered
    assert "Rank-8 tangent adapter & $+12.00$ & pos. & $+10.50$ [$+9.50$, $+11.50$]" in registered
    # one study label per block and panel: A (one block) and B (two noise levels) in both the Q5 and the Q2 panel
    assert registered.count("Follow-up A,") == 2 and registered.count("Follow-up B,") == 4
    steps = tables["table_followup_steps.tex"]
    assert "tab:followup-steps" in steps and gen.Macros.PLACEHOLDER not in steps
    assert "0.04 & 100.00 & 100.00 & 5.00 & 5.00 &" in steps


def test_every_registered_bank_has_a_display_name():
    names = set()
    for reg in registrations().values():
        for cell in reg["cells"]:
            names.update(cell.get("candidate_ids") or reg["candidates"]["candidate_ids"])
    assert len(names) == 36
    for candidate in names:
        assert "\\_" not in gen.fu_display(candidate) and "_" not in gen.fu_display(candidate).replace("$c=", "")
    assert gen.fu_display("text_only_centering__coefficient_1") == "the text half of two-sided centering at $c=1$"
    for unknown in ("control__other", "text_only_centering__step_1", "gr_clip_style_two_sided__coefficient_x"):
        with pytest.raises(RuntimeError, match="no display name"):
            gen.fu_display(unknown)


def test_forest_figure_data_and_placeholder(tmp_path):
    regs = registrations()
    _, tables = run(tmp_path / "ready", {"B": report_b(regs["B"])})
    axis = tables["fig_followup_forest_axis.tex"]
    assert "\\def\\FUForestReady{1}" in axis and "\\def\\FUForestSeparators{5.5,7.5}" in axis
    rows = {group: tables[f"fig_followup_forest_{group}.dat"].strip().splitlines() for group in gen.FU_FOREST_GROUPS}
    assert [len(lines) - 1 for lines in rows.values()] == [5, 2, 4]  # header + rows per group
    first = rows["primary"][1].split()  # two-sided c=1 from the pooled family contrasts: 0.3 [-0.7, 1.3]
    assert first[0] == "1" and float(first[1]) == 0.3 and float(first[2]) == 1.0 and float(first[3]) == 1.0
    _, empty = run(tmp_path / "empty", {})
    assert "\\def\\FUForestReady{0}" in empty["fig_followup_forest_axis.tex"]
    assert empty["fig_followup_forest_primary.dat"].strip() == "y lo lo_minus lo_plus hi hi_minus hi_plus"
    assert "\\def\\FUCurvesReady{0}" in empty["fig_followup_curves_axis.tex"]


def curves_payload(reg: dict, analysis_sha: str) -> dict:
    grid = [0.0, 0.005, 0.01]
    block = {"items": 21000, "grid": grid, "curves": {bank: [50.0, 49.0, 48.0] for bank, _ in gen.FU_CURVE_COLUMNS}}
    return {"schema_version": gen.FU_CURVES_SCHEMA, "descriptive": True, "registration_sha256": reg["registration_sha256"],
            "banks": [bank for bank, _ in gen.FU_CURVE_COLUMNS], "registered_value_checks": 324,
            "pooled": {"0.12": block, "0.25": block}, "analysis_sha256": analysis_sha}


def test_curves_figure_requires_the_same_analysis_output(tmp_path):
    import hashlib

    regs = registrations()
    root = tmp_path / "c"
    report = report_b(regs["B"])
    path = root / gen.FU_ANALYSIS_RELS["B"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")
    write_cells_csv(path.parent / "cells_ext.csv", report)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / gen.FU_CURVES_REL).write_text(json.dumps(curves_payload(regs["B"], digest)), encoding="utf-8")
    macros, notes, tables = gen.Macros(), {}, {}
    gen.followup_results(gen.Sources(root), macros, notes, tables, regs)
    assert "\\def\\FUCurvesReady{1}" in tables["fig_followup_curves_axis.tex"]
    assert tables["fig_followup_curves_low.dat"].splitlines()[1] == "0.000 50.0000 50.0000 50.0000 50.0000"
    (root / gen.FU_CURVES_REL).write_text(json.dumps(curves_payload(regs["B"], "0" * 64)), encoding="utf-8")
    with pytest.raises(RuntimeError, match="another EXP-021B analysis output"):
        gen.followup_results(gen.Sources(root), gen.Macros(), {}, {}, regs)
    wrong = dict(curves_payload(regs["B"], digest), banks=["no_correction"])
    (root / gen.FU_CURVES_REL).write_text(json.dumps(wrong), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not the descriptive curve output"):
        gen.followup_results(gen.Sources(root), gen.Macros(), {}, {}, regs)


def test_clip_standardization_constants_match_open_clip():
    open_clip_constants = pytest.importorskip("open_clip.constants")
    assert tuple(gen.FU_CLIP_STD) == tuple(open_clip_constants.OPENAI_DATASET_STD)
    assert (f"{1.0 / max(gen.FU_CLIP_STD):.1f}", f"{1.0 / min(gen.FU_CLIP_STD):.1f}") == ("3.6", "3.8")


def test_pending_outputs_become_placeholders(tmp_path):
    macros, tables = run(tmp_path, {})
    assert {entry["macro"] for entry in macros.missing} >= {"FUQOnePoint", "FUGRHighPoint", "FUQNineIdFull"}
    assert gen.Macros.PLACEHOLDER in tables["table_followup.tex"]


def test_refusals(tmp_path):
    regs = registrations()
    wrong = dict(report_a(regs["A"]), registration_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="another registration"):
        run(tmp_path / "w", {"A": wrong})
    overridden = report_a(regs["A"])
    overridden["inference"]["replicates_overridden"] = True
    with pytest.raises(RuntimeError, match="registered bootstrap replicates"):
        run(tmp_path / "o", {"A": overridden})
    partial = report_b(regs["B"])
    partial["questions"]["Q6"]["pooled__sigma0.12"].pop(PRIMARY[-1])
    with pytest.raises(RuntimeError, match="primary family"):
        run(tmp_path / "p", {"B": partial})
    short = report_c(regs["C"])
    dropped = regs["C"]["cells"][0]["cell_id"]
    short["questions"]["Q9"]["rows"] = [row for row in short["questions"]["Q9"]["rows"] if row["cell_id"] != dropped]
    with pytest.raises(RuntimeError, match="registered EXP-021C cells"):
        run(tmp_path / "c", {"C": short})
    unevaluable = report_a(regs["A"])
    unevaluable["questions"]["Q1"] = {"verdict": "not_evaluable", "reason": "Q1 needs all twelve 021A cells; not run: x"}
    with pytest.raises(RuntimeError, match="not_evaluable"):
        run(tmp_path / "q", {"A": unevaluable})
    unschemed = dict(report_c(regs["C"]), schema_version="other")
    with pytest.raises(RuntimeError, match="schema"):
        run(tmp_path / "s", {"C": unschemed})
    root = tmp_path / "m"
    run(root, {"B": report_b(regs["B"])})  # writes a matching cells_ext.csv; now replace it with another run's
    write_cells_csv(root / gen.FU_CELLS_CSV_REL, report_b(regs["B"]), shift=0.01)
    with pytest.raises(RuntimeError, match="not the same analysis run"):
        gen.followup_results(gen.Sources(root), gen.Macros(), {}, {}, regs)
