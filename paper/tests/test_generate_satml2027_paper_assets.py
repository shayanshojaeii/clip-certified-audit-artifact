"""Tests for scripts/generate_satml2027_paper_assets.py.

The generator is run once into a temporary directory. Invariants are then
recomputed independently from the immutable artifacts so that a regression in
the generator's arithmetic or formatting is caught without trusting its output.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_satml2027_paper_assets.py"
CONTRACT = ROOT / "paper" / "satml2027" / "MACROS_CONTRACT.md"
AUDIT = ROOT / "artifacts" / "negative_paper" / "exp017_scientific_soundness_audit_v1.json"
GATE_N2 = ROOT / "artifacts" / "negative_paper" / "gate_n2_simultaneous_analysis_v1.json"
DIAGNOSTICS = ROOT / "artifacts" / "gate3" / "phase3_exp017_negative_diagnostics_v1.json"
SUMMARY = ROOT / "results" / "EXP-20260906-017" / "summary.json"
SELECTION = ROOT / "results" / "EXP-20260906-017" / "selection.json"
CELLS = ROOT / "results" / "EXP-20260906-017" / "cells"
FARLA_REPORT = ROOT / "FARLA" / "downloaded_results" / "EXP-20260917-020-FARLA-FULL" / "analysis" / "report.json"
REGISTRATIONS = [
    ROOT / "configs" / "satml2027" / "exp-20260920-019a.json",
    ROOT / "configs" / "satml2027" / "exp-20260920-019b.json",
    ROOT / "configs" / "satml2027" / "n3c_v3.json",
]
TABLES = [
    "table_family_r025.tex",
    "table_partition.tex",
    "table_grclip_cells.tex",
    "table_farla.tex",
    "table_extension.tex",
    "table_ext_regime.tex",
    "table_ext_contrasts.tex",
    "table_n3c.tex",
]
EXT_ANALYSIS = ROOT / "results" / "satml2027_local" / "analysis"
EXT_MERGED = ROOT / "results" / "satml2027_local" / "merged"
EXT_DOWNLOAD = ROOT / "results" / "satml2027_downloads" / "EXP019_N3C_20260925T172040Z"
EXT_RECEIPT = EXT_DOWNLOAD / "server_results_root" / "scientific_run_attempt4_receipt.txt"
EXT_ITEM_MANIFEST = EXT_DOWNLOAD / "scientific_results" / "items" / "item_manifest.json"
EXT_EXPERIMENTS = {"A": "EXP-20260920-019A", "B": "EXP-20260920-019B"}
N3C_EXPERIMENT = "N3C-20260920-V3"
EXT_CONTROLS = (
    "control__learned_shared_translation_tangent",
    "control__learned_shared_translation_pure",
    "control__noisy_class_mean",
    "control__lowrank_tangent_r8",
)
GR_CLIP = "gr_clip_style_two_sided__coefficient_1"
NO_CORRECTION = "no_correction"
STANDARD = "standard_cohen_certified_accuracy"


def _load_module():
    name = "generate_satml2027_paper_assets"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module  # dataclasses resolve postponed annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _torch_available(module) -> bool:
    if importlib.util.find_spec("torch") is not None:
        return True
    return any((ROOT / rel).exists() for rel in module.TORCH_INTERPRETERS)


def _number(text: str) -> float:
    return float(text.replace(",", ""))


@pytest.fixture(scope="module")
def module():
    return _load_module()


@pytest.fixture(scope="module")
def generated(module, tmp_path_factory) -> Path:
    output = tmp_path_factory.mktemp("satml2027_generated")
    module.generate(ROOT, output)
    return output


@pytest.fixture(scope="module")
def macros(generated) -> dict[str, str]:
    text = (generated / "macros.tex").read_text(encoding="utf-8")
    found = dict(re.findall(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}$", text, flags=re.MULTILINE))
    assert found, "no macros parsed from macros.tex"
    return found


@pytest.fixture(scope="module")
def sources(generated) -> dict:
    return _json(generated / "sources.json")


def test_generator_writes_every_declared_file(generated):
    for name in ["macros.tex", "sources.json", *TABLES]:
        assert (generated / name).is_file(), name


def test_macros_define_every_contract_name(macros):
    contract_names = set(re.findall(r"`\\([A-Za-z]+)`", CONTRACT.read_text(encoding="utf-8")))
    assert len(contract_names) > 100, "contract parse found too few macro names"
    missing = sorted(contract_names - set(macros))
    assert not missing, f"macros.tex lacks contract macros: {missing}"
    for name, value in macros.items():
        assert re.fullmatch(r"[A-Za-z]+", name)
        assert value.strip(), name


def _pending_extension_prefixes() -> set[str]:
    """Macro prefixes that may legitimately be ?? while registered analysis outputs are pending."""
    allowed: set[str] = set()
    for key, experiment_id in EXT_EXPERIMENTS.items():
        if not (EXT_ANALYSIS / experiment_id / "cells.csv").is_file():
            allowed.update({f"Ext{key}", "ExtBridge", "ExtSmoothed", "ExtStdCA"})
        elif not (EXT_ANALYSIS / experiment_id / "contrasts.json").is_file():
            allowed.add(f"Ext{key}")
    if not (EXT_ANALYSIS / N3C_EXPERIMENT / "n3c_predictions.json").is_file():
        allowed.add("NthreeCP")
    # EXP-021 (D-145): result macros may be ?? only while that part's registered analysis output is absent;
    # the design macros (FURegHash, FUCells, FUImages, ...) come from the registrations and are never pending.
    followup = {"A": ("FUQOne", "FUQTwoA", "FUQThreeA", "FUQThreeExceptionsA", "FUQFourA", "FUQFourChangedA", "FUNotRunA",
                      "FUQFourMarginA"),
                "B": ("FUGR", "FUQSix", "FUQTwoLow", "FUQTwoHigh", "FUQFive", "FUQSeven", "FUQEight", "FUQThreeB",
                      "FUQThreeExceptionsB", "FUQFourB", "FUQFourChangedB", "FUNotRunB", "FUQFourMarginB"),
                "C": ("FUQNine",)}
    for key, prefixes in followup.items():
        if not (ROOT / "analysis" / "exp021" / f"021{key.lower()}" / "analysis_ext.json").is_file():
            allowed.update(prefixes)
    if not (ROOT / "analysis" / "exp021" / "compute_summary.json").is_file():  # D-150: written after both hosts
        allowed.update(("FUMainHours", "FUExtraHours"))
    return allowed


def test_missing_ledger_is_consistent(module, macros, sources):
    placeholder = module.Macros.PLACEHOLDER
    ledger = {entry["macro"] for entry in sources["MISSING"]}
    placeholders = {name for name, value in macros.items() if value == placeholder}
    assert ledger == placeholders
    if _torch_available(module):
        allowed = _pending_extension_prefixes()
        unexpected = [name for name in ledger if not any(name.startswith(prefix) for prefix in allowed)]
        assert not unexpected, unexpected
    for entry in sources["MISSING"]:
        assert entry["reason"]


def test_no_correction_standard_ca_is_the_equal_cell_mean(macros):
    audit = _json(AUDIT)
    records = audit["per_candidate_cells"][NO_CORRECTION]
    assert len(records) == 12
    mean = sum(record["standard_certified_accuracy"]["0.25"] for record in records) / len(records)
    stored = audit["candidate_aggregates_equal_cell_weight"][NO_CORRECTION]["standard_certified_accuracy"]["0.25"]
    assert math.isclose(mean, stored, abs_tol=1e-9)
    assert macros["NoCorrStdCA"] == f"{100 * mean:.2f}"
    anchored = sum(record["anchored_certified_accuracy"]["0.25"] for record in records) / len(records)
    assert macros["NoCorrAnchCA"] == f"{100 * anchored:.2f}"


def test_gr_delta_equals_difference_of_macros(macros):
    delta = _number(macros["GRStdDelta"])
    assert abs(delta - (_number(macros["GRStdCA"]) - _number(macros["NoCorrStdCA"]))) <= 0.01 + 1e-9
    anchored = _number(macros["GRAnchDelta"])
    assert abs(anchored - (_number(macros["GRAnchCA"]) - _number(macros["NoCorrAnchCA"]))) <= 0.01 + 1e-9


def test_band_endpoints_are_point_plus_minus_common_critical_value(macros):
    gate = _json(GATE_N2)
    critical = gate["common_critical_value"]
    rows = {(row["candidate_id"], row["metric"], row["radius"]): row for row in gate["contrasts"]}
    assert len(rows) == 136
    assert macros["BandHalfWidth"] == f"{100 * critical:.2f}"
    diagnostics = _json(DIAGNOSTICS)["floating_boundary_adjudication"]
    checks = {
        "GRStd": (GR_CLIP, STANDARD, 0.25),
        "GRAnch": (GR_CLIP, "raw_clean_anchored_certified_accuracy", 0.25),
        "SavedStd": (diagnostics["saved_selected_proposal"], STANDARD, 0.25),
        "ExactStd": (diagnostics["exact_count_selected_proposal"], STANDARD, 0.25),
        "GMCStd": ("global_mean_centering__coefficient_1", STANDARD, 0.25),
    }
    for prefix, key in checks.items():
        row = rows[key]
        for suffix, field in (("Delta", "point_estimate"), ("Lower", "simultaneous_lower"), ("Upper", "simultaneous_upper")):
            expected = round(100 * row[field], 2)
            assert abs(_number(macros[f"{prefix}{suffix}"]) - expected) < 1e-9, f"{prefix}{suffix}"
        assert abs(_number(macros[f"{prefix}Lower"]) - (_number(macros[f"{prefix}Delta"]) - _number(macros["BandHalfWidth"]))) <= 0.011
        assert abs(_number(macros[f"{prefix}Upper"]) - (_number(macros[f"{prefix}Delta"]) + _number(macros["BandHalfWidth"]))) <= 0.011
    zero = rows[("global_mean_centering__coefficient_1", STANDARD, 0.0)]
    assert abs(_number(macros["GMCStdDeltaRZero"]) - round(100 * zero["point_estimate"], 2)) < 1e-9
    assert abs(_number(macros["GMCStdLowerRZero"]) - round(100 * zero["simultaneous_lower"], 2)) < 1e-9
    assert abs(_number(macros["GMCStdUpperRZero"]) - round(100 * zero["simultaneous_upper"], 2)) < 1e-9
    excluded = [row for row in gate["contrasts"] if row["simultaneous_lower"] > 0 or row["simultaneous_upper"] < 0]
    assert macros["BandExclZero"] == f"{len(excluded):,}"
    assert macros["BandExclZeroGR"] == f"{sum(row['candidate_id'] == GR_CLIP for row in excluded):,}"
    # The manuscript's scoped wording: only GR-CLIP excludes zero at radius 0.25, its exclusions are
    # positive, and the other exclusions are negative standard mean-centering contrasts at radius 0.
    other = [row for row in excluded if row["candidate_id"] != GR_CLIP]
    assert macros["BandExclZeroOther"] == f"{len(other):,}"
    assert other and all(
        row["simultaneous_upper"] < 0
        and float(row["radius"]) == 0.0
        and row["metric"] == STANDARD
        and row["candidate_id"].startswith("global_mean_centering__")
        for row in other
    )
    assert all(row["simultaneous_lower"] > 0 for row in excluded if row["candidate_id"] == GR_CLIP)
    assert all(row["candidate_id"] == GR_CLIP for row in excluded if float(row["radius"]) == 0.25)
    sens = critical * math.sqrt(5 / 4)
    assert macros["BandHalfWidthSens"] == f"{100 * sens:.2f}"
    gr = rows[(GR_CLIP, STANDARD, 0.25)]
    assert abs(_number(macros["GRStdLowerSens"]) - round(100 * (gr["point_estimate"] - sens), 2)) < 1e-9


def test_chowers_max_abs_delta_is_zero(macros):
    gate = _json(GATE_N2)
    chowers = [row for row in gate["contrasts"] if row["candidate_id"].startswith("chowers_exact_projected_gap__")]
    assert len(chowers) == 24
    assert max(abs(row["point_estimate"]) for row in chowers) == 0.0
    assert macros["ChowersMaxAbsDelta"] == "0.00"


def test_sources_list_sha256_for_every_artifact_read(sources):
    artifacts = sources["artifacts"]
    required = [
        AUDIT,
        GATE_N2,
        DIAGNOSTICS,
        SUMMARY,
        SELECTION,
        FARLA_REPORT,
        ROOT / "configs" / "clean_pipeline_v1.json",
        ROOT / "configs" / "clean_pipeline_vit_l14_v1.json",
        EXT_RECEIPT,
        EXT_ITEM_MANIFEST,
        *[EXT_MERGED / experiment / "merge_report.json" for experiment in (*EXT_EXPERIMENTS.values(), N3C_EXPERIMENT)],
        *[
            EXT_ANALYSIS / experiment / name
            for experiment in EXT_EXPERIMENTS.values()
            for name in ("cells.csv", "macro.csv", "concentration.csv", "contrasts.json")
            if (EXT_ANALYSIS / experiment / name).is_file()
        ],
        *([EXT_ANALYSIS / N3C_EXPERIMENT / "n3c_predictions.json"] if (EXT_ANALYSIS / N3C_EXPERIMENT / "n3c_predictions.json").is_file() else []),
        *REGISTRATIONS,
        *[path.with_suffix(".sha256") for path in REGISTRATIONS],
        *[CELLS / cell.name / "cell_metrics.json" for cell in sorted(CELLS.iterdir()) if cell.is_dir()],
    ]
    for path in required:
        rel = path.relative_to(ROOT).as_posix()
        assert rel in artifacts, rel
    assert len([rel for rel in artifacts if rel.endswith("cell_metrics.json")]) == 12
    for rel, digest in artifacts.items():
        assert re.fullmatch(r"[0-9a-f]{64}", digest), rel
        assert _sha256(ROOT / rel) == digest, rel
    if sources["MISSING"] == []:
        pt_files = [rel for rel in artifacts if rel.endswith("cohen_sufficient_statistics.pt")]
        assert len(pt_files) == 12
        recorded = _json(AUDIT)["source_statistics_sha256"]
        for rel in pt_files:
            assert artifacts[rel] == recorded[Path(rel).parent.name]


def test_failure_partition_matches_audit_and_sums_to_positions(module, macros):
    if not _torch_available(module):
        pytest.skip("torch unavailable: partition macros are expected to be ??")
    audit = _json(AUDIT)
    records = audit["per_candidate_cells"][NO_CORRECTION]
    positions = sum(record["example_count"] for record in records)
    assert macros["ExpPositions"] == f"{positions:,}"
    totals: dict[str, int] = {}
    for record in records:
        causes = record["failure_causes_at_0.25"]
        assert causes["standard"]["abstained"] == causes["anchored"]["abstained"]
        merged = {**causes["standard"], **causes["anchored"]}  # 'abstained' is shared by both blocks
        for key, value in merged.items():
            totals[key] = totals.get(key, 0) + int(value)
    assert macros["PartAbstain"] == f"{totals['abstained']:,}"
    assert macros["PartWrongClass"] == f"{totals['nonabstaining_wrong_smoothed_class']:,}"
    assert macros["PartCorrectBelowRadius"] == f"{totals['correct_smoothed_class_but_radius_below_target']:,}"
    assert macros["PartCertifiedCorrect"] == f"{totals['certified_correct_at_target']:,}"
    assert macros["PartAnchorMismatch"] == f"{totals['nonabstaining_anchor_mismatch']:,}"
    assert macros["PartAnchoredCorrect"] == f"{totals['anchored_certified_correct_at_target']:,}"
    parts = [
        _number(macros[name]) for name in ("PartAbstain", "PartWrongClass", "PartCorrectBelowRadius", "PartCertifiedCorrect")
    ]
    assert sum(parts) == positions
    wrong_rate = _number(macros["CertifiedWrongRate"])
    assert 0.0 <= wrong_rate <= 100 * totals["nonabstaining_wrong_smoothed_class"] / positions + 1e-9
    assert macros["ExpDistinctClassifiers"] == "15"


def test_exact_count_gate_recomputed_with_fractions(macros):
    selection = _json(SELECTION)["all_aggregates"]
    clean: dict[str, dict[str, tuple[int, int]]] = {}
    for cell_dir in sorted(path for path in CELLS.iterdir() if path.is_dir()):
        metrics = _json(cell_dir / "cell_metrics.json")
        n = int(metrics["item_count"])
        for record in metrics["candidates"]:
            scaled = record["raw_clean_accuracy"] * n
            assert abs(scaled - round(scaled)) <= 1e-9
            clean.setdefault(record["candidate_id"], {})[cell_dir.name] = (int(round(scaled)), n)
    reference = clean[NO_CORRECTION]

    def drops(candidate: str) -> list[Fraction]:
        return [Fraction(100 * (reference[cell][0] - correct), n) for cell, (correct, n) in clean[candidate].items()]

    rejected = []
    for candidate, aggregate in selection.items():
        cell_drops = drops(candidate)
        exact = sum(cell_drops) / len(cell_drops) <= 1 and all(drop <= 2 for drop in cell_drops)
        if not aggregate["clean_accuracy_constraint_satisfied"] and exact:
            rejected.append(candidate)
    assert macros["FloatGateRejectedCandidates"] == f"{len(rejected):,}"
    assert rejected, "expected at least one float-gate rejection in the saved selection"
    artifact = {repr(float(selection[c]["maximum_cell_clean_accuracy_drop_percentage_points"])) for c in rejected}
    assert macros["FloatGateArtifact"] == ", ".join(sorted(artifact))
    gr_drops = drops(GR_CLIP)
    worst = max(gr_drops)
    assert macros["GRWorstCellDrop"] == f"{float(worst):.2f}"
    assert macros["GRCellsImproved"] == f"{sum(drop < 0 for drop in gr_drops):,}"
    forced = _json(DIAGNOSTICS)["forced_exact_proposal_vs_each_mean_centering"]
    assert macros["ForcedComparisonFailures"] == f"{sum(not item['g3_4_all_strictly_positive'] for item in forced.values()):,}"


def test_kish_design_effect_and_pooled_delta(macros):
    audit = _json(AUDIT)
    counts = {record["cell_id"]: int(record["example_count"]) for record in audit["per_candidate_cells"][NO_CORRECTION]}
    n = sum(counts.values())
    cells = len(counts)
    weight_sum = sum(Fraction(size, cells * size) for size in counts.values())
    weight_square_sum = sum(size * Fraction(1, cells * size) ** 2 for size in counts.values())
    deff = Fraction(n) * weight_square_sum / weight_sum**2
    assert macros["KishDesignEffect"] == f"{float(deff):.3f}"
    assert macros["EffectiveN"] == f"{float(Fraction(n) / deff):,.1f}"
    gr = {record["cell_id"]: record for record in audit["per_candidate_cells"][GR_CLIP]}
    nc = {record["cell_id"]: record for record in audit["per_candidate_cells"][NO_CORRECTION]}
    numerator = sum(
        round(gr[cell]["standard_certified_accuracy"]["0.25"] * size) - round(nc[cell]["standard_certified_accuracy"]["0.25"] * size)
        for cell, size in counts.items()
    )
    expected = round(float(Fraction(100 * numerator, n)), 2)
    assert abs(_number(macros["GRStdDeltaPooled"]) - expected) < 1e-9
    assert macros["GRStdDeltaPooled"].startswith(("+", "-"))


def test_farla_and_extension_macros(macros):
    report = _json(FARLA_REPORT)
    rows = {row["candidate_id"]: row for row in report["aggregate"]}
    assert macros["FarlaFullStdCA"] == f"{100 * rows['farla_full_r8_supervised']['macro_standard_ca_primary']:.2f}"
    assert macros["FarlaIdentityClean"] == f"{100 * rows[NO_CORRECTION]['macro_clean_accuracy']:.2f}"
    contrast = report["planned_contrasts"]["farla_vs_no_correction"]["bootstrap"]
    assert abs(_number(macros["FarlaVsNoCorrLower"]) - round(100 * contrast["lower"], 2)) < 1e-9
    assert macros["FarlaShots"] == "20"
    registrations = [_json(path) for path in REGISTRATIONS]
    assert macros["ExtCellsA"] == f"{len(registrations[0]['cells']):,}"
    assert macros["ExtCellsB"] == f"{len(registrations[1]['cells']):,}"
    assert macros["ExtCellsN"] == f"{len(registrations[2]['cells']):,}"
    for path, name in zip(REGISTRATIONS, ("ExtRegHashA", "ExtRegHashB", "ExtRegHashN")):
        sidecar = path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
        assert macros[name] == sidecar[:12]
        assert re.fullmatch(r"[0-9a-f]{12}", macros[name])
    summary = _json(SUMMARY)
    assert macros["ExpCells"] == f"{summary['cell_count']:,}"
    assert macros["ExpBanks"] == f"{summary['cell_count'] * summary['candidate_count_per_cell']:,}"
    assert macros["ExpConfDraws"] == f"{summary['cohen_confirmation_samples_per_example']:,}"


def test_tables_use_booktabs_and_list_sources(generated, macros):
    hex64 = re.compile(r"[0-9a-f]{64}")
    for name in TABLES:
        text = (generated / name).read_text(encoding="utf-8")
        assert text.startswith("% Generated by scripts/generate_satml2027_paper_assets.py"), name
        header = [line for line in text.splitlines() if line.startswith("%")]
        assert any(hex64.search(line) for line in header), name
        for token in (r"\toprule", r"\midrule", r"\bottomrule", r"\scriptsize", r"\caption{"):
            assert token in text, f"{name} lacks {token}"
        assert any(suffix in text for suffix in (".json", ".pt", ".csv")), f"{name} caption does not name a source artifact"
    family = (generated / "table_family_r025.tex").read_text(encoding="utf-8")
    assert family.count(r"\\") >= 18 + 1
    assert r"\begin{table*}[t]" in family and r"\label{tab:family}" in family
    assert r"$^{\dagger}$" in family and r"$^{\ddagger}$" in family
    for label in ("No correction", "Mean-centering (text) c=0.25", "Projected gap c=1", "GR-CLIP-style two-sided c=1", "Boundary-active step 0.0025", "Noisy-margin step 0.04"):
        assert label in family
    for column in ("Candidate", "Clean acc.", "Std.\\ CA", "Anch.\\ CA", "(points)", "95\\% simultaneous band", "Eligible (float rule)", "Eligible (exact rule)"):
        assert column in family, column
    labels = {
        "table_partition.tex": r"\label{tab:partition}",
        "table_grclip_cells.tex": r"\label{tab:grclip-cells}",
        "table_farla.tex": r"\label{tab:farla}",
        "table_extension.tex": r"\label{tab:extension}",
    }
    full_width = {"table_grclip_cells.tex", "table_extension.tex", "table_farla.tex"}  # plus the family table, checked above
    for name, label in labels.items():
        text = (generated / name).read_text(encoding="utf-8")
        assert label in text, name
        environment = r"\begin{table*}[t]" if name in full_width else r"\begin{table}[t]"
        assert environment in text, name
        assert text.index(r"\caption{") < text.index(r"\begin{tabular}"), name
    extension = (generated / "table_extension.tex").read_text(encoding="utf-8")
    for name in ("ExtRegHashA", "ExtRegHashB", "ExtRegHashN"):
        assert macros[name] in extension
    grclip = (generated / "table_grclip_cells.tex").read_text(encoding="utf-8")
    assert grclip.count(r" \\") == 12 + 1
    summary = _json(SUMMARY)
    assert macros["ExpElapsedHours"] == f"{summary['elapsed_seconds'] / 3600:.1f}"
    assert macros["SelectedComparator"] == summary["selected_primary_comparator_id"].replace("_", r"\_")


def test_grclip_table_columns_average_to_the_equal_cell_macros(generated, macros):
    text = (generated / "table_grclip_cells.tex").read_text(encoding="utf-8")
    assert r"\begin{table*}[t]" in text and r"\label{tab:grclip-cells}" in text
    header = text.split(r"\toprule", 1)[1].split(r"\midrule", 1)[0]
    for column in ("No corr.\\ Std CA", "GR-CLIP Std CA", "No corr.\\ Anch.\\ CA", "GR-CLIP Anch.\\ CA"):
        assert column in header, column
    body = text.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    rows = [line.strip() for line in body.splitlines() if line.strip().endswith(r"\\")]
    assert len(rows) == 12
    table = [[cell.strip().strip("$") for cell in row[:-2].split(" & ")] for row in rows]
    assert all(len(row) == 10 for row in table)

    def column_mean(index: int) -> str:
        return f"{sum(float(row[index]) for row in table) / len(table):.2f}"

    assert column_mean(6) == macros["NoCorrStdCA"]
    assert column_mean(7) == macros["GRStdCA"]
    assert column_mean(8) == macros["NoCorrAnchCA"]
    assert column_mean(9) == macros["GRAnchCA"]
    # The per-cell values must be the audit's standard field, not cell_metrics.json certified_accuracy.
    audit = _json(AUDIT)
    standard = sorted(f"{100 * r['standard_certified_accuracy']['0.25']:.2f}" for r in audit["per_candidate_cells"][NO_CORRECTION])
    assert sorted(row[6] for row in table) == standard
    anchored = sorted(f"{100 * r['anchored_certified_accuracy']['0.25']:.2f}" for r in audit["per_candidate_cells"][GR_CLIP])
    assert sorted(row[9] for row in table) == anchored


def test_dataset_split_and_standard_minus_anchored_macros(macros):
    audit = _json(AUDIT)
    nc = {r["cell_id"]: r for r in audit["per_candidate_cells"][NO_CORRECTION]}
    gr = {r["cell_id"]: r for r in audit["per_candidate_cells"][GR_CLIP]}
    for dataset, name in (("cifar100", "GRStdDeltaCifar"), ("eurosat", "GRStdDeltaEurosat")):
        ids = [c for c in nc if f"__{dataset}__" in c]
        assert len(ids) == 6
        delta = sum(gr[c]["standard_certified_accuracy"]["0.25"] - nc[c]["standard_certified_accuracy"]["0.25"] for c in ids) / len(ids)
        assert abs(_number(macros[name]) - round(100 * delta, 2)) < 1e-9, name
        assert macros[name][0] in "+-", name
    cifar = [c for c in nc if "__cifar100__" in c]
    negative = sum(
        round(gr[c]["standard_certified_accuracy"]["0.25"] * nc[c]["example_count"])
        < round(nc[c]["standard_certified_accuracy"]["0.25"] * nc[c]["example_count"])
        for c in cifar
    )
    assert macros["GRStdCifarCellsNegative"] == f"{negative:,}"
    gaps = [100 * v["standard_minus_anchored"]["0.25"] for v in audit["candidate_aggregates_equal_cell_weight"].values()]
    assert len(gaps) == 18 and min(gaps) >= 0
    assert abs(_number(macros["StdAnchGapMin"]) - min(gaps)) < 0.0051
    assert abs(_number(macros["StdAnchGapMax"]) - max(gaps)) < 0.0051


def test_gmc_anchored_zero_radius_farla_additions_and_radius_per_coordinate(generated, macros):
    gate = _json(GATE_N2)
    row = next(
        r
        for r in gate["contrasts"]
        if r["candidate_id"] == "global_mean_centering__coefficient_1"
        and r["metric"] == "raw_clean_anchored_certified_accuracy"
        and r["radius"] == 0.0
    )
    for suffix, field in (("Delta", "point_estimate"), ("Lower", "simultaneous_lower"), ("Upper", "simultaneous_upper")):
        assert abs(_number(macros[f"GMCAnch{suffix}RZero"]) - round(100 * row[field], 2)) < 1e-9, suffix
    report = _json(FARLA_REPORT)
    rows = {r["candidate_id"]: r for r in report["aggregate"]}
    assert macros["FarlaIdentityCW"] == _half_up_pct(rows[NO_CORRECTION]["macro_certified_wrong_primary"])
    assert macros["FarlaFullCW"] == _half_up_pct(rows["farla_full_r8_supervised"]["macro_certified_wrong_primary"])
    assert macros["FarlaTailCW"] == _half_up_pct(rows["lowrank_tail_r8_supervised"]["macro_certified_wrong_primary"])
    pseudo = rows["farla_full_r8_pseudo"]
    assert macros["FarlaPseudoStdCA"] == _half_up_pct(pseudo["macro_standard_ca_primary"])
    assert macros["FarlaPseudoClean"] == _half_up_pct(pseudo["macro_clean_accuracy"])
    assert macros["FarlaPseudoCleanDrop"] == _half_up(pseudo["macro_clean_drop_percentage_points"])
    paired = report["paired_comparisons"]["farla_full_r8_pseudo"]
    assert paired["comparator"] == NO_CORRECTION and paired["proposed"] == "farla_full_r8_pseudo"
    for suffix, field in (("Delta", "point_difference"), ("Lower", "lower"), ("Upper", "upper")):
        assert abs(_number(macros[f"FarlaPseudoVsNoCorr{suffix}"]) - round(100 * paired["bootstrap"][field], 2)) < 1e-9
        assert macros[f"FarlaPseudoVsNoCorr{suffix}"][0] in "+-"
    farla_table = (generated / "table_farla.tex").read_text(encoding="utf-8")
    assert "Cert.-wrong" in farla_table
    assert "pseudo-labels (no ground truth)" in farla_table
    for label in ("Shared translation", "Low-rank r=8 (noisy CE)", "Tail-only r=8", "Full objective r=4", "Full objective r=8 &",
                  "Full objective, direct"):
        assert label in farla_table, label
    visible = "\n".join(line for line in farla_table.splitlines() if not line.lstrip().startswith("%"))
    assert "FARLA" not in visible  # the internal method name never reaches the reader (review 2026-09-28)
    planned = _json(FARLA_REPORT)["planned_contrasts"]
    assert farla_table.count(r" \\") == 2 + 8 + len(planned)  # two headers, eight candidates, every planned contrast
    pipeline = _json(ROOT / "configs" / "clean_pipeline_v1.json")["preprocessing"]
    value = 0.25 / math.sqrt(len(pipeline["mean"]) * pipeline["crop_size"] ** 2)
    mantissa, exponent = f"{value:.1e}".split("e")
    assert macros["ExpRadiusPerCoordinate"] == f"{mantissa}\\times 10^{{{int(exponent)}}}"


def test_certified_wrong_counts_and_changed_positions_recomputed(module, macros):
    if not _torch_available(module):
        pytest.skip("torch unavailable: per-example macros are expected to be ??")
    positions = int(_number(macros["ExpPositions"]))
    certified_wrong = int(_number(macros["PartCertifiedWrong"]))
    assert macros["CertifiedWrongRate"] == f"{100 * certified_wrong / positions:.2f}"
    assert certified_wrong <= int(_number(macros["PartWrongClass"]))
    expected_mismatch = int(_number(macros["PartCertifiedCorrect"]) - _number(macros["PartAnchoredCorrect"]))
    assert macros["CertifiedCorrectAnchorMismatch"] == f"{expected_mismatch:,}"
    torch = pytest.importorskip("torch")
    changed: dict[str, int] = {}
    shares: list[float] = []
    cells_without = 0
    total_certified_wrong = 0
    for cell_dir in sorted(path for path in CELLS.iterdir() if path.is_dir()):
        raw = torch.load(cell_dir / "cohen_sufficient_statistics.pt", map_location="cpu", weights_only=True)
        truth = torch.as_tensor(raw["ground_truth"]).long()

        def event(candidate: str):
            outcome = raw["outcomes"][candidate]
            return (
                (~torch.as_tensor(outcome["abstained"]).bool())
                & torch.as_tensor(outcome["selected_classes"]).long().eq(truth)
                & torch.as_tensor(outcome["certificate_radii"]).double().ge(0.25)
            )

        reference = event(NO_CORRECTION)
        for candidate in raw["candidate_ids"]:
            changed[candidate] = changed.get(candidate, 0) + int(event(candidate).ne(reference).sum())
        outcome = raw["outcomes"][NO_CORRECTION]
        selected = torch.as_tensor(outcome["selected_classes"]).long()
        wrong = (
            (~torch.as_tensor(outcome["abstained"]).bool())
            & selected.ne(truth)
            & torch.as_tensor(outcome["certificate_radii"]).double().ge(0.25)
        )
        n_wrong = int(wrong.sum())
        total_certified_wrong += n_wrong
        if n_wrong == 0:
            cells_without += 1
        else:
            shares.append(int(torch.bincount(selected[wrong]).max()) / n_wrong)
    assert total_certified_wrong == certified_wrong
    assert macros["CWCellsWithNone"] == f"{cells_without:,}"
    assert macros["CWTopClassShareLo"] == f"{100 * min(shares):.1f}"
    assert macros["CWTopClassShareHi"] == f"{100 * max(shares):.1f}"
    small = [
        c for c in changed
        if c.split("__")[0] in ("clean_boundary_active", "cohen_aligned_noisy_margin") and float(c.rsplit("_", 1)[1]) <= 0.01
    ]
    assert len(small) == 6
    assert macros["ChangedPositionsSmallSteps"] == f"{max(changed[c] for c in small):,}"
    assert macros["ChangedPositionsExactStep"] == f"{changed['clean_boundary_active__step_0.04']:,}"
    assert macros["ChangedPositionsGR"] == f"{changed[GR_CLIP]:,}"
    proposals = [c for c in changed if c.split("__")[0] in ("clean_boundary_active", "cohen_aligned_noisy_margin")]
    assert len(proposals) == 10
    assert macros["ChangedPositionsProposalsMax"] == f"{max(changed[c] for c in proposals):,}"


def _ext_rows(experiment_id: str) -> dict[str, dict[str, dict]]:
    import csv

    rows: dict[str, dict[str, dict]] = {}
    with (EXT_ANALYSIS / experiment_id / "cells.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.setdefault(row["candidate_id"], {})[row["cell_id"]] = row
    return rows


def _ext_concentration(experiment_id: str) -> dict[str, dict]:
    import csv

    with (EXT_ANALYSIS / experiment_id / "concentration.csv").open("r", encoding="utf-8", newline="") as handle:
        return {row["cell_id"]: row for row in csv.DictReader(handle) if row["candidate_id"] == NO_CORRECTION}


def _ext_std(row: dict) -> float:
    return float(row[f"standard_certified_accuracy@{format(float(row['sigma']), '.12g')}"])


def _ext_delta(rows: dict, candidate: str, field=None) -> float:
    identity = rows[NO_CORRECTION]
    primary = [cell for cell, row in identity.items() if row["primary_estimand_cell"] == "True"]
    extract = field or _ext_std
    values = [extract(rows[candidate][cell]) - extract(identity[cell]) for cell in primary]
    return 100 * sum(values) / len(values)


def test_extension_custody_macros(module, macros):
    receipt = dict(line.split("=", 1) for line in EXT_RECEIPT.read_text(encoding="utf-8").splitlines() if "=" in line)
    assert macros["ExtAReceiptStatus"] == receipt["status"]
    assert macros["ExtResultFiles"] == f"{int(receipt['result_file_count']):,}"
    assert macros["ExtResultManifest"] == receipt["result_manifest_sha256"][:12]
    merged = 0
    for experiment in (*EXT_EXPERIMENTS.values(), N3C_EXPERIMENT):
        report = _json(EXT_MERGED / experiment / "merge_report.json")
        assert report["complete"] is True and report["missing"] == []
        merged += len(report["cells"])
    assert macros["ExtCellsMerged"] == f"{merged:,}"
    manifest = _json(EXT_ITEM_MANIFEST)
    per_class = {spec["splits"]["control_train"]["per_class"] for spec in manifest["datasets"].values()}
    assert len(per_class) == 1
    assert macros["ExtControlLabelsPerClass"] == f"{next(iter(per_class)):,}"


def test_extension_contrast_macros_recomputed(module, generated, macros):
    placeholder = module.Macros.PLACEHOLDER
    contrast_table = (generated / "table_ext_contrasts.tex").read_text(encoding="utf-8")
    assert r"\begin{table*}[t]" in contrast_table and r"\label{tab:ext-contrasts}" in contrast_table
    assert f"{macros['ExtControlLabelsPerClass']} labels per class" in contrast_table
    for key, experiment_id in EXT_EXPERIMENTS.items():
        prefix = f"Ext{key}"
        if not (EXT_ANALYSIS / experiment_id / "cells.csv").is_file():
            assert macros[f"{prefix}GRDelta"] == placeholder
            continue
        rows = _ext_rows(experiment_id)
        grid = sorted(c for c in rows if c != NO_CORRECTION and not c.startswith("control__"))
        assert len(grid) == 17 and all(c in rows for c in EXT_CONTROLS)
        clean = lambda row: float(row["clean_accuracy"])  # noqa: E731
        expectations = {
            f"{prefix}GRDelta": _ext_delta(rows, GR_CLIP),
            f"{prefix}GRCleanDelta": _ext_delta(rows, GR_CLIP, clean),
            f"{prefix}LowrankDelta": _ext_delta(rows, "control__lowrank_tangent_r8"),
            f"{prefix}LowrankCleanDelta": _ext_delta(rows, "control__lowrank_tangent_r8", clean),
            f"{prefix}NoisyMeanDelta": _ext_delta(rows, "control__noisy_class_mean"),
            f"{prefix}NoisyMeanCleanDelta": _ext_delta(rows, "control__noisy_class_mean", clean),
            f"{prefix}SharedTangentDelta": _ext_delta(rows, "control__learned_shared_translation_tangent"),
            f"{prefix}SharedPureDelta": _ext_delta(rows, "control__learned_shared_translation_pure"),
            f"{prefix}GMCOneDelta": _ext_delta(rows, "global_mean_centering__coefficient_1"),
            f"{prefix}GMCOneCleanDelta": _ext_delta(rows, "global_mean_centering__coefficient_1", clean),
        }
        grid_deltas = {c: _ext_delta(rows, c) for c in grid}
        expectations[f"{prefix}MaxGridDelta"] = max(grid_deltas.values())
        expectations[f"{prefix}MinGridDelta"] = min(grid_deltas.values())
        for name, value in expectations.items():
            assert abs(_number(macros[name]) - round(value, 2)) < 1e-9, name
            assert macros[name][0] in "+-", name
        # Largest grid delta with exact integer-count arithmetic; ties are listed alphabetically.
        identity = rows[NO_CORRECTION]
        primary = [cell for cell, row in identity.items() if row["primary_estimand_cell"] == "True"]
        exact = {}
        for c in grid:
            total = Fraction(0)
            for cell in primary:
                n = int(float(identity[cell]["example_count"]))
                total += Fraction(100 * (round(_ext_std(rows[c][cell]) * n) - round(_ext_std(identity[cell]) * n)), n)
            exact[c] = total / len(primary)
        best = sorted(c for c, v in exact.items() if v == max(exact.values()))
        worst = sorted(c for c, v in exact.items() if v == min(exact.values()))
        module = _load_module()
        # readable bank names since the review of 2026-09-28 (the PDF had printed the raw candidate ids)
        assert macros[f"{prefix}MaxGridId"] == " and ".join(module.ext_text_name(c) for c in best)
        assert macros[f"{prefix}MinGridId"] == " and ".join(module.ext_text_name(c) for c in worst)
        # X1: grid banks within the registered +/-0.5-point reporting region, from the exact integer-count deltas
        assert macros[f"{prefix}InsideRegion"] == str(sum(abs(v) <= Fraction(1, 2) for v in exact.values()))
        # The contrasts table names each experiment by its study name; registry ids stay in the source comments.
        assert experiment_id not in "\n".join(line for line in contrast_table.splitlines() if not line.lstrip().startswith("%"))
        block = contrast_table.split(module.EXT_STUDY_NAMES[key], 1)[1]
        assert block.count(" & ") >= 21 * 4
        contrasts_path = EXT_ANALYSIS / experiment_id / "contrasts.json"
        if not contrasts_path.is_file():
            for stem in ("Critical", "GRLower", "GRUpper", "GridExclZero", "ControlsExclZero", "Xthree", "Xfour"):
                assert macros[f"{prefix}{stem}"] == placeholder, stem
        else:
            contrasts = _json(contrasts_path)
            critical = contrasts["simultaneous_band"]["critical_max_absolute_deviation"]
            assert macros[f"{prefix}Critical"] == f"{100 * critical:.2f}"
            intervals = {c: v["simultaneous_confidence_interval"] for c, v in contrasts["contrasts"].items()}
            for stem, candidate in (
                ("GR", GR_CLIP),
                ("Lowrank", "control__lowrank_tangent_r8"),
                ("SharedTangent", "control__learned_shared_translation_tangent"),
            ):
                interval = intervals[candidate]
                assert abs(100 * interval["point_difference"] - _ext_delta(rows, candidate)) < 1e-7
                assert abs(_number(macros[f"{prefix}{stem}Lower"]) - round(100 * (interval["point_difference"] - critical), 2)) < 1e-9
                assert abs(_number(macros[f"{prefix}{stem}Upper"]) - round(100 * (interval["point_difference"] + critical), 2)) < 1e-9
            excludes = lambda c: intervals[c]["lower"] > 0 or intervals[c]["upper"] < 0  # noqa: E731
            assert macros[f"{prefix}GridExclZero"] == f"{sum(excludes(c) for c in grid):,}"
            assert macros[f"{prefix}ControlsExclZero"] == f"{sum(excludes(c) for c in EXT_CONTROLS):,}"
            tangent = intervals["control__learned_shared_translation_tangent"]
            assert macros[f"{prefix}Xthree"] == ("confirmed" if tangent["lower"] <= 0 <= tangent["upper"] else "not confirmed")
            # addendum A2.3: at least one per-class control above zero (existential, not "every")
            positive = [intervals[c]["lower"] > 0 for c in ("control__noisy_class_mean", "control__lowrank_tangent_r8")]
            assert macros[f"{prefix}Xfour"] == ("confirmed" if any(positive) else "not confirmed")
            assert macros[f"{prefix}XfourPositiveControls"] == ("zero", "one", "two")[sum(positive)]
        shares = _ext_concentration(experiment_id)
        groups: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
        for cell_id, row in rows[NO_CORRECTION].items():
            groups.setdefault((row["model_id"], row["dataset_id"]), []).append(
                (float(row["sigma"]), float(shares[cell_id]["top_class_share"]), float(shares[cell_id]["predicted_class_gini"])))
        # addendum A1: X5 needs top-class share and predicted-class Gini both rising with sigma; descriptive wording only
        monotone = all(all(b[1] > a[1] and b[2] > a[2] for a, b in zip(sorted(v), sorted(v)[1:])) for v in groups.values())
        assert macros[f"{prefix}Xfive"] == ("supported" if monotone else "not uniformly supported")
    for stem in ("Xthree", "Xfour"):
        expected = module.prediction_summary(macros[f"ExtA{stem}"] == "confirmed", macros[f"ExtB{stem}"] == "confirmed")
        assert macros[f"Ext{stem}Summary"] == expected
        assert "confirmed and confirmed" not in expected
    expected = module.x5_summary(macros["ExtAXfive"] == "supported", macros["ExtBXfive"] == "supported")
    assert macros["ExtXfiveSummary"] == expected and "confirmed" not in expected


def test_x4_and_x5_follow_the_frozen_addendum_rules(module):
    """V11 review: X4 is existential (A2.3) and must stay so even where "every" happens to give the same verdict;
    X5 needs both statistics (A1). Counterfactual fixtures, checked against the registered evaluator d1_09."""

    noisy, lowrank = module.EXT_PER_CLASS_CONTROLS
    interval = lambda lower, upper: {"lower": lower, "upper": upper, "point_difference": (lower + upper) / 2}  # noqa: E731
    one_positive = {noisy: interval(0.01, 0.03), lowrank: interval(-0.02, 0.01)}
    none_positive = {noisy: interval(-0.01, 0.03), lowrank: interval(0.0, 0.01)}
    both_positive = {noisy: interval(0.01, 0.03), lowrank: interval(0.02, 0.04)}
    assert module.ext_x4_confirmed(one_positive) is True
    assert module.ext_x4_confirmed(none_positive) is False
    assert module.ext_x4_confirmed(both_positive) is True
    assert module.x5_group_supported([(0.25, 0.30, 0.60), (0.12, 0.20, 0.50)]) is True
    assert module.x5_group_supported([(0.12, 0.20, 0.50), (0.25, 0.30, 0.49)]) is False  # share rises, Gini falls
    assert module.x5_group_supported([(0.12, 0.20, 0.50), (0.25, 0.20, 0.60)]) is False  # share flat
    assert module.x5_summary(True, False) == "supported in Extension A but not uniformly supported in Extension B"
    sys.path.insert(0, str(ROOT / "satml2027"))
    try:
        d1_09 = importlib.import_module("day1.d1_09_sensitivity_analysis")
    except ModuleNotFoundError as error:  # pragma: no cover - depends on the host
        pytest.skip(f"registered evaluator not importable here: {error}")
    shared = {"control__learned_shared_translation_tangent": interval(-0.01, 0.01)}
    for fixture in (one_positive, none_positive, both_positive):
        registered = d1_09.evaluate_predictions({**shared, **fixture}, region=0.005, critical=0.02)["X4"]["confirmed"]
        assert registered == module.ext_x4_confirmed(fixture)


def test_extension_bridge_regime_and_table(module, generated, macros):
    placeholder = module.Macros.PLACEHOLDER
    regime = (generated / "table_ext_regime.tex").read_text(encoding="utf-8")
    assert r"\begin{table*}[t]" in regime and r"\label{tab:ext-regime}" in regime
    available = {key: exp for key, exp in EXT_EXPERIMENTS.items() if (EXT_ANALYSIS / exp / "cells.csv").is_file()}
    if "A" not in available:
        assert macros["ExtBridgeGRDeltaBthirtytwo"] == placeholder
        return
    rows_a = _ext_rows(EXT_EXPERIMENTS["A"])
    bridge = {cell: row for cell, row in rows_a[NO_CORRECTION].items() if row["primary_estimand_cell"] == "False"}
    assert len(bridge) == 2 and all(row["dataset_id"] == "cifar10" and float(row["sigma"]) == 0.25 for row in bridge.values())
    assert regime.count(r"$^{\dagger}$") == 2
    for cell, row in bridge.items():
        token = {"openai-clip-vit-b32-quickgelu": "Bthirtytwo", "openai-clip-vit-l14-quickgelu": "Lfourteen"}[row["model_id"]]
        expected = 100 * (_ext_std(rows_a[GR_CLIP][cell]) - _ext_std(row))
        assert abs(_number(macros[f"ExtBridgeGRDelta{token}"]) - round(expected, 2)) < 1e-9
    identity_rows = [row for exp in available.values() for row in _ext_rows(exp)[NO_CORRECTION].values()]
    for sigma, token in ((0.12, "Twelve"), (0.5, "Fifty")):
        at_sigma = [row for row in identity_rows if math.isclose(float(row["sigma"]), sigma)]
        cifar = [row for row in at_sigma if row["dataset_id"].startswith("cifar")]
        eurosat = [row for row in at_sigma if row["dataset_id"] == "eurosat"]
        assert cifar and eurosat
        smoothed = [float(row["smoothed_accuracy"]) for row in cifar]
        assert macros[f"ExtSmoothed{token}CifarLo"] == f"{100 * min(smoothed):.2f}"
        assert macros[f"ExtSmoothed{token}CifarHi"] == f"{100 * max(smoothed):.2f}"
        standard = [_ext_std(row) for row in cifar]
        assert macros[f"ExtStdCA{token}CifarLo"] == f"{100 * min(standard):.2f}"
        assert macros[f"ExtStdCA{token}CifarHi"] == f"{100 * max(standard):.2f}"
        eurosat_smoothed = [float(row["smoothed_accuracy"]) for row in eurosat]
        assert macros[f"ExtSmoothed{token}EurosatLo"] == f"{100 * min(eurosat_smoothed):.2f}"
        assert macros[f"ExtSmoothed{token}EurosatHi"] == f"{100 * max(eurosat_smoothed):.2f}"
    body = regime.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    data_rows = [line for line in body.splitlines() if line.strip().endswith(r"\\") and r"\multicolumn" not in line]
    assert len(data_rows) == sum(len(_ext_rows(exp)[NO_CORRECTION]) for exp in available.values())
    assert all(len(line.split(" & ")) == 10 for line in data_rows)


def test_n3c_macros_and_table(module, generated, macros):
    placeholder = module.Macros.PLACEHOLDER
    table = (generated / "table_n3c.tex").read_text(encoding="utf-8")
    assert r"\begin{table*}[t]" in table and r"\label{tab:n3c}" in table
    path = EXT_ANALYSIS / N3C_EXPERIMENT / "n3c_predictions.json"
    if not path.is_file():
        assert macros["NthreeCPone"] == placeholder
        return
    payload = _json(path)
    cells = payload["cells"]
    assert len(cells) == 4
    exceptions = sum(int(v["flips_outside_eligible_set"]) for cell in cells.values() for v in cell["P1"].values() if v.get("applicable"))
    assert macros["NthreeCPone"] == f"{exceptions:,}"
    mace = [cell["P4a"]["mean_absolute_calibration_error"] for cell_id, cell in cells.items() if "__cifar100__" in cell_id]
    assert len(mace) == 2
    verdict = "pass" if all(value <= 0.05 for value in mace) else "fail"
    assert macros["NthreeCPfourA"] == f"{max(mace):.4f} ({verdict})"
    positive = sum(cell["P5"]["spearman_rho"] > 0 for cell in cells.values())
    assert macros["NthreeCPfive"] == f"{positive} of 4 ({'pass' if positive >= 3 else 'fail'})"
    rhos = [cell["P5"]["spearman_rho"] for cell in cells.values()]
    assert (macros["NthreeCPfiveRhoLo"], macros["NthreeCPfiveRhoHi"]) == (f"{min(rhos):+.2f}", f"{max(rhos):+.2f}")
    # P2 (review of 2026-09-28): the float-level flips of the projected-gap banks and their P1 conformance
    p2 = [entry for cell in cells.values() for entry in cell["P2"].values()]
    assert macros["NthreeCPtwoFlips"] == f"{sum(int(e['p1_flips']) for e in p2):,}"
    assert macros["NthreeCPtwoExceptions"] == f"{sum(int(e['p1_exceptions']) for e in p2):,}"
    residual = max(float(e["centered_span_orthogonality_residual_max"]) for e in p2)
    mantissa, exponent = macros["NthreeCPtwoResidual"].split("\\times10^")
    bound = float(mantissa) * 10 ** int(exponent.strip("{}"))
    assert residual <= bound <= residual * 1.01  # printed as a bound, rounded up
    at_threshold = sum(cell["P6_items_already_at_threshold"] for cell in cells.values())
    assert macros["NthreeCPseven"] == f"{at_threshold:,}"
    assert macros["NthreeCPsevenPositions"] == f"{sum(cell['items'] for cell in cells.values()):,}"
    for stem, candidate in (("TwoSided", GR_CLIP), ("Lowrank", "control__lowrank_tangent_r8")):
        useful = sum(cell["P9"][candidate]["useful_wrong_to_correct"] for cell in cells.values())
        harmful = sum(cell["P9"][candidate]["harmful_correct_to_wrong"] for cell in cells.values())
        assert macros[f"NthreeCPnine{stem}Useful"] == f"{useful:,}"
        assert macros[f"NthreeCPnine{stem}Harmful"] == f"{harmful:,}"
        assert f"{useful:,} useful / {harmful:,} harmful" in macros["NthreeCPnine"]
    body = table.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    data_rows = [line for line in body.splitlines() if line.strip().endswith(r"\\") and r"\multicolumn" not in line]
    assert len(data_rows) == 4 * 4  # four banks in each of the four cells
    assert all(len(line.split(" & ")) == 8 for line in data_rows)
    assert body.count(r"\multicolumn{8}") == 4
    # Spot-check one cell x bank row against the JSON.
    first_cell = sorted(cells)[0]
    p9 = cells[first_cell]["P9"]["control__lowrank_tangent_r8"]
    expected = " & ".join(
        ["Rank-8 tangent adapter (supervised)", f"{p9['useful_wrong_to_correct']:,}", f"{p9['harmful_correct_to_wrong']:,}", f"{p9['wrong_to_wrong']:,}", f"{p9['unchanged']:,}", "n/a", "n/a", "n/a"]
    )
    assert expected + r" \\" in table


def test_generated_files_are_deterministic(module, generated, tmp_path):
    second = tmp_path / "again"
    module.generate(ROOT, second)
    for name in ["macros.tex", "sources.json", *TABLES]:
        assert (second / name).read_bytes() == (generated / name).read_bytes(), name


def _half_up(value: float, scale: int = 1) -> str:
    """The registered display rule (D-137): the stored value's exact shortest decimal, rounded half-up to 2 digits."""

    from decimal import ROUND_HALF_UP, Decimal

    return f"{(Decimal(repr(float(value))) * scale).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):.2f}"


def _half_up_pct(value: float) -> str:
    return _half_up(value, 100)


def test_display_rounding_is_half_up_on_the_exact_stored_decimal():
    """Review of reviewer bundle V4 (M-D1): ties such as the FARLA endpoint 0.01125 (1.125 points) round half-up."""

    import importlib.util

    spec = importlib.util.spec_from_file_location("paper_assets_rounding", ROOT / "scripts" / "generate_satml2027_paper_assets.py")
    module = importlib.util.module_from_spec(spec)
    import sys as _sys

    _sys.modules["paper_assets_rounding"] = module
    spec.loader.exec_module(module)
    assert module.spct(0.01125) == "+1.13" and module.spct(-0.01125) == "-1.13"
    assert module.pct(0.34175) == "34.18" and module.pct(0.10425) == "10.43" and module.spct(0.26625) == "+26.63"
    # a value that rounds to zero keeps its side (an interval bound of -0.004 must not print as +0.00; review 2026-09-28)
    assert module.spct(0.0) == "+0.00" and module.spct(-0.0) == "+0.00" and module.spts(-0.004) == "-0.00"
    assert module.spts(0.004) == "+0.00"
    assert module.pct(0.1) == "10.00" and module.pts(2.9666666666666666) == "2.97" and module.spts(-1.3833) == "-1.38"
    assert module.one_decimal_pct(0.12345) == "12.3" and module.one_decimal_pct(0.12355) == "12.4"
