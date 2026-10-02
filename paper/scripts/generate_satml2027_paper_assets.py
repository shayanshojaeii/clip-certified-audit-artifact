"""Generate every number in the SaTML 2027 manuscript from immutable artifacts.

Specification: ``paper/satml2027/MACROS_CONTRACT.md``.

Outputs (default directory ``paper/satml2027/generated/``):
  macros.tex, sources.json, table_family_r025.tex, table_partition.tex,
  table_grclip_cells.tex, table_farla.tex, table_extension.tex.

Rules implemented here:

* every emitted number is computed from the artifacts named in the contract;
  the SHA-256 of every file that is opened is recorded in ``sources.json``;
* nothing under ``results/``, ``artifacts/``, ``configs/`` or ``FARLA/`` is
  written, and no script that samples is invoked;
* a macro that cannot be derived is emitted as ``\textbf{??}`` and listed
  under ``MISSING`` in ``sources.json`` together with the reason;
* per-example outcomes are read from the twelve immutable
  ``results/EXP-20260906-017/cells/*/cohen_sufficient_statistics.pt`` files,
  the same sufficient statistics that Gate N2 consumed.  Only the decision
  fields (``candidate_ids``, ``item_ids``, ``ground_truth``,
  ``raw_predictions``, ``outcomes``) are accessed; the Monte-Carlo count
  arrays are never touched.  Reading a ``.pt`` file needs torch, which is
  imported under a guard.  When torch is unavailable the generator re-invokes
  itself as a ``--partition-worker`` under ``.venv-gate-n2``; when that is
  impossible too, the affected macros are emitted as ``??``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_REL = "scripts/generate_satml2027_paper_assets.py"
CONTRACT_REL = "paper/satml2027/MACROS_CONTRACT.md"
DEFAULT_OUTPUT_REL = "paper/satml2027/generated"
TORCH_INTERPRETERS = (
    ".venv-gate-n2/Scripts/python.exe",
    ".venv-gate-n2/Scripts/python",
    ".venv-gate-n2/bin/python",
)

# Immutable inputs, relative to the workspace root.
AUDIT_REL = "artifacts/negative_paper/exp017_scientific_soundness_audit_v1.json"
GATE_N2_REL = "artifacts/negative_paper/gate_n2_simultaneous_analysis_v1.json"
DIAGNOSTICS_REL = "artifacts/gate3/phase3_exp017_negative_diagnostics_v1.json"
SUMMARY_REL = "results/EXP-20260906-017/summary.json"
SELECTION_REL = "results/EXP-20260906-017/selection.json"
CELLS_REL = "results/EXP-20260906-017/cells"
CELL_METRICS_NAME = "cell_metrics.json"
SUFFICIENT_STATISTICS_NAME = "cohen_sufficient_statistics.pt"
FARLA_DIR_REL = "FARLA/downloaded_results/EXP-20260917-020-FARLA-FULL"
FARLA_REPORT_REL = f"{FARLA_DIR_REL}/analysis/report.json"
FARLA_CONFIG_REL = f"{FARLA_DIR_REL}/config_snapshot.json"
FARLA_SPLITS_REL = f"{FARLA_DIR_REL}/splits/manifest.json"
REGISTRATION_RELS = {
    "A": "configs/satml2027/exp-20260920-019a.json",
    "B": "configs/satml2027/exp-20260920-019b.json",
    "N": "configs/satml2027/n3c_v3.json",
}
# EXP-021 follow-up registrations (V4, registered 2026-09-26 before any sampling; D-145 brings EXP-021 into the paper).
FU_REGISTRATION_RELS = {
    "A": "configs/satml2027ext/exp-20260921-021a.json",
    "B": "configs/satml2027ext/exp-20260921-021b.json",
    "C": "configs/satml2027ext/exp-20260921-021c.json",
}
# Frozen clean pipelines of the two EXP-017 backbones; they bind the input geometry
# (RGB channels from the normalisation vector, side from crop_size).
CLEAN_PIPELINE_RELS = (
    "configs/clean_pipeline_v1.json",
    "configs/clean_pipeline_vit_l14_v1.json",
)
# Registered extension results (EXP-019A/B and N3C), unblinded under D-127 at the
# owner's instruction; the registered primary analysis outputs are primary-only.
EXT_ANALYSIS_DIR_REL = "results/satml2027_local/analysis"
EXT_MERGED_DIR_REL = "results/satml2027_local/merged"
EXT_RECEIPT_REL = (
    "results/satml2027_downloads/EXP019_N3C_20260925T172040Z/server_results_root/"
    "scientific_run_attempt4_receipt.txt"
)
EXT_ITEM_MANIFEST_REL = (
    "results/satml2027_downloads/EXP019_N3C_20260925T172040Z/scientific_results/items/item_manifest.json"
)
EXT_EXPERIMENTS = {"A": "EXP-20260920-019A", "B": "EXP-20260920-019B"}
N3C_EXPERIMENT = "N3C-20260920-V3"
# Reader-facing study names: registry identifiers stay in file paths and in the one
# identifier map of the Open Science section, never in the paper's tables or text.
EXT_STUDY_NAMES = {"A": "Extension A", "B": "Extension B"}
N3C_STUDY_NAME = "Fresh-noise diagnostic"
FU_STUDY_NAMES = {"A": "Follow-up A", "B": "Follow-up B", "C": "Follow-up C"}
EXT_CONTROL_PREFIX = "control__"
EXT_SHARED_TANGENT = "control__learned_shared_translation_tangent"
EXT_SHARED_PURE = "control__learned_shared_translation_pure"
EXT_NOISY_MEAN = "control__noisy_class_mean"
EXT_LOWRANK = "control__lowrank_tangent_r8"
EXT_CONTROL_ORDER = (EXT_SHARED_TANGENT, EXT_SHARED_PURE, EXT_NOISY_MEAN, EXT_LOWRANK)
EXT_PER_CLASS_CONTROLS = (EXT_NOISY_MEAN, EXT_LOWRANK)
EXT_MODEL_MACRO_TOKEN = {
    "openai-clip-vit-b32-quickgelu": "Bthirtytwo",
    "openai-clip-vit-l14-quickgelu": "Lfourteen",
}
EXT_SIGMA_TOKENS = ((0.12, "Twelve"), (0.5, "Fifty"))
EXT_PER_EXPERIMENT_MACROS = (
    "Critical",
    "GRDelta",
    "GRLower",
    "GRUpper",
    "GRCleanDelta",
    "LowrankDelta",
    "LowrankLower",
    "LowrankUpper",
    "LowrankCleanDelta",
    "NoisyMeanDelta",
    "NoisyMeanCleanDelta",
    "SharedTangentDelta",
    "SharedTangentLower",
    "SharedTangentUpper",
    "MaxGridDelta",
    "MaxGridId",
    "MinGridDelta",
    "MinGridId",
    "MinGridCleanDelta",
    "GMCOneDelta",
    "GMCOneCleanDelta",
    "SharedPureDelta",
    "SharedPureLower",
    "SharedPureUpper",
    "GridExclZero",
    "GridExclZeroIds",
    "InsideRegion",
    "ControlsExclZero",
    "Xthree",
    "Xfour",
    "XfourPositiveControls",
    "Xfive",
)
N3C_MACROS = (
    "NthreeCPone",
    "NthreeCPtwoResidual",
    "NthreeCPtwoMismatch",
    "NthreeCPtwoFlips",
    "NthreeCPtwoExceptions",
    "NthreeCPfiveRhoLo",
    "NthreeCPfiveRhoHi",
    "NthreeCPfourA",
    "NthreeCPfourAMax",
    "NthreeCPfourAVerdict",
    "NthreeCPfive",
    "NthreeCPfivePositive",
    "NthreeCPfiveVerdict",
    "NthreeCPseven",
    "NthreeCPsevenPositions",
    "NthreeCPsevenCandidates",
    "NthreeCPsevenUnattainableLo",
    "NthreeCPsevenUnattainableHi",
    "NthreeCPsevenUndeterminedLo",
    "NthreeCPsevenUndeterminedHi",
    "NthreeCPnine",
    "NthreeCPnineTwoSidedUseful",
    "NthreeCPnineTwoSidedHarmful",
    "NthreeCPnineLowrankUseful",
    "NthreeCPnineLowrankHarmful",
)

# Candidate and metric identifiers used by the immutable artifacts.
NO_CORRECTION = "no_correction"
GR_CLIP = "gr_clip_style_two_sided__coefficient_1"
GMC_ONE = "global_mean_centering__coefficient_1"
CHOWERS_FAMILY = "chowers_exact_projected_gap"
STANDARD_METRIC = "standard_cohen_certified_accuracy"
ANCHORED_METRIC = "raw_clean_anchored_certified_accuracy"
TARGET_RADIUS = 0.25
ZERO_RADIUS = 0.0

FARLA_IDENTITY = "no_correction"
FARLA_SHARED = "shared_translation_ce_supervised"
FARLA_LOWRANK_CE = "lowrank_ce_r8_supervised"
FARLA_TAIL = "lowrank_tail_r8_supervised"
FARLA_FULL = "farla_full_r8_supervised"
FARLA_PSEUDO = "farla_full_r8_pseudo"
FARLA_CONTRAST_NO_CORR = "farla_vs_no_correction"
FARLA_CONTRAST_TAIL = "farla_vs_lowrank_tail"

PARTITION_FIELDS_ACCESSED = (
    "candidate_ids",
    "item_ids",
    "ground_truth",
    "raw_predictions",
    "outcomes",
)
PARTITION_KEYS = (
    "abstained",
    "nonabstaining_wrong_smoothed_class",
    "correct_smoothed_class_but_radius_below_target",
    "certified_correct_at_target",
    "certified_wrong_at_target",
    "nonabstaining_anchor_mismatch",
    "anchor_preserved_but_raw_class_wrong",
    "anchor_correct_but_radius_below_target",
    "anchored_certified_correct_at_target",
)

# Presentation-only display names (never result values).
FAMILY_DISPLAY: dict[str, tuple[str, str | None]] = {
    "no_correction": ("No correction", None),
    "global_mean_centering": ("Mean-centering (text)", "c"),
    "gr_clip_style_two_sided": ("GR-CLIP-style two-sided", "c"),
    "chowers_exact_projected_gap": ("Projected gap", "c"),
    "clean_boundary_active": ("Boundary-active", "step"),
    "cohen_aligned_noisy_margin": ("Noisy-margin", "step"),
}
FAMILY_ORDER = list(FAMILY_DISPLAY)
MODEL_DISPLAY = {
    "openai-clip-vit-b32-quickgelu": "ViT-B/32",
    "openai-clip-vit-l14-quickgelu": "ViT-L/14",
    "openai-clip-vit-b16-quickgelu": "ViT-B/16",
    "openai-clip-rn50-quickgelu": "RN50",
    "openclip-vit-b32-laion2b": "ViT-B/32 (LAION-2B)",
}
DATASET_DISPLAY = {"cifar100": "CIFAR-100", "cifar10": "CIFAR-10", "eurosat": "EuroSAT"}
FARLA_DISPLAY = {
    "no_correction": "No correction",
    "shared_translation_ce_supervised": "Shared translation (noisy CE)",
    "lowrank_ce_r8_supervised": "Low-rank r=8 (noisy CE)",
    "lowrank_tail_r8_supervised": "Tail-only r=8",
    "farla_full_r4_supervised": "Full objective r=4",
    "farla_full_r8_supervised": "Full objective r=8",
    "farla_full_direct_supervised": "Full objective, direct",
    "farla_full_r8_pseudo": "Full objective r=8, pseudo-labels (no ground truth)",
}
# Extension display names: grid banks as in the family table, except the two-sided
# candidate, plus the four registered controls.
EXT_DISPLAY = {
    GR_CLIP: "Two-sided centering c=1",
    EXT_SHARED_TANGENT: "Learned shared translation (tangent)",
    EXT_SHARED_PURE: "Learned shared translation (pure)",
    EXT_NOISY_MEAN: "Noisy class mean (supervised)",
    EXT_LOWRANK: "Rank-8 tangent adapter (supervised)",
}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Display rounding (2026-09-27, D-137; review of reviewer bundle V4, M-D1): every displayed
# value is the stored value, read as its exact shortest decimal representation, rounded
# half-up at the last displayed digit. Binary-float formatting rounded exact ties such as
# 0.01125 (1.125 points) inconsistently; the FARLA rates are multiples of 1/4000 and hit
# such ties. Only FARLA values changed (eight distinct values, each by 0.01 in the last digit).


def _decimal(value: float) -> Decimal:
    return Decimal(repr(float(value)))


def _half_up(value: Decimal, digits: int) -> Decimal:
    rounded = value.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    return rounded + 0  # normalise negative zero


def pct(value: float) -> str:
    """Rate in [0,1] -> percentage points with two decimals."""
    return f"{_half_up(_decimal(value) * 100, 2):.2f}"


def spct(value: float) -> str:
    """Signed rate difference in [-1,1] -> signed percentage points."""
    return _signed(_decimal(value) * 100)


def pts(value: float) -> str:
    return f"{_half_up(_decimal(value), 2):.2f}"


def _signed(value: Decimal) -> str:
    rounded = _half_up(value, 2)
    if rounded == 0:
        # keep the side of zero: an interval bound of -0.004 must not read as +0.00 (review 2026-09-28)
        return "-0.00" if value < 0 else "+0.00"
    return f"{rounded:+.2f}"


def spts(value: float) -> str:
    return _signed(_decimal(value))


def one_decimal_pct(value: float) -> str:
    return f"{_half_up(_decimal(value) * 100, 1):.1f}"


def count(value: int) -> str:
    return f"{int(value):,}"


def count_word(value: int) -> str:
    """Small counts in words for running text (IEEE style), larger ones as digits."""

    words = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
    return words[int(value)] if 0 <= int(value) < len(words) else count(value)


def english_list(names: list[str]) -> str:
    """Names for running text: "none", "a", "a and b", "a, b, and c"."""

    if not names:
        return "none"
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + ", and " + names[-1]


def num(value: float) -> str:
    """Short numeric display for steps and coefficients (0.0025, 0.5, 1)."""
    return f"{float(value):g}"


def tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def exact_count(rate: float, n: int, *, what: str) -> int:
    """round(rate * n) with an exactness assertion (tolerance 1e-9)."""
    scaled = float(rate) * int(n)
    nearest = round(scaled)
    if abs(scaled - nearest) > 1e-9:
        raise ValueError(f"{what}: {rate} x {n} = {scaled} is not an integer count within 1e-9")
    return int(nearest)


def radius_value(mapping: dict[str, Any], radius: float) -> Any:
    """Look up a radius-keyed mapping whose keys may be '0', '0.0', '0.25'."""
    for key, value in mapping.items():
        if math.isclose(float(key), radius, abs_tol=1e-12):
            return value
    raise KeyError(f"radius {radius} not present in {sorted(mapping)}")


def candidate_family(candidate_id: str) -> str:
    return candidate_id.split("__", 1)[0]


def candidate_step(candidate_id: str) -> float | None:
    if "__" not in candidate_id:
        return None
    suffix = candidate_id.split("__", 1)[1]
    return float(suffix.rsplit("_", 1)[1])


def candidate_sort_key(candidate_id: str) -> tuple[int, float]:
    family = candidate_family(candidate_id)
    order = FAMILY_ORDER.index(family) if family in FAMILY_ORDER else len(FAMILY_ORDER)
    step = candidate_step(candidate_id)
    return (order, -1.0 if step is None else step)


def candidate_display(candidate_id: str) -> str:
    family = candidate_family(candidate_id)
    step = candidate_step(candidate_id)
    if family not in FAMILY_DISPLAY:
        return tex_escape(candidate_id)
    name, kind = FAMILY_DISPLAY[family]
    if kind is None or step is None:
        return name
    if kind == "c":
        return f"{name} c={num(step)}"
    return f"{name} step {num(step)}"


def family_display(candidate_id: str) -> tuple[str, str]:
    family = candidate_family(candidate_id)
    step = candidate_step(candidate_id)
    if family not in FAMILY_DISPLAY:
        return tex_escape(candidate_id), "--"
    name, kind = FAMILY_DISPLAY[family]
    if kind is None or step is None:
        return name, "--"
    if kind == "c":
        return name, f"c={num(step)}"
    return name, f"step {num(step)}"


def model_display(model_id: str) -> str:
    return MODEL_DISPLAY.get(model_id, tex_escape(model_id))


def dataset_display(dataset_id: str) -> str:
    return DATASET_DISPLAY.get(dataset_id, tex_escape(dataset_id))


def cell_display(model_id: str, dataset_id: str, fold: int) -> str:
    return f"{model_display(model_id)}, {dataset_display(dataset_id)}, fold {int(fold)}"


def parse_cell_id(cell_id: str) -> tuple[str, str, int]:
    model, dataset, fold_token = cell_id.rsplit("__", 2)
    if not fold_token.startswith("fold"):
        raise ValueError(f"invalid cell id: {cell_id}")
    return model, dataset, int(fold_token[len("fold"):])


def json_ready(value: Any) -> Any:
    if isinstance(value, Fraction):
        return {"fraction": f"{value.numerator}/{value.denominator}", "float": float(value)}
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


# --------------------------------------------------------------------------
# Source registry and macro store
# --------------------------------------------------------------------------


class Sources:
    """Records the SHA-256 of every immutable file that is opened."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.hashes: dict[str, str] = {}

    def register(self, rel: str) -> str:
        path = self.root / rel
        if not path.is_file():
            raise FileNotFoundError(rel)
        digest = sha256_file(path)
        self.hashes[rel] = digest
        return digest

    def read_json(self, rel: str) -> Any:
        self.register(rel)
        return json.loads((self.root / rel).read_text(encoding="utf-8"))

    def read_text(self, rel: str) -> str:
        self.register(rel)
        return (self.root / rel).read_text(encoding="utf-8")

    def read_csv(self, rel: str) -> list[dict[str, Any]]:
        self.register(rel)
        with (self.root / rel).open("r", encoding="utf-8", newline="") as handle:
            return [_parse_csv_row(row) for row in csv.DictReader(handle)]

    def ordered(self) -> list[tuple[str, str]]:
        return sorted(self.hashes.items())


def _parse_csv_row(row: dict[str, str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in row.items():
        if value in ("True", "False"):
            parsed[key] = value == "True"
            continue
        try:
            parsed[key] = float(value)
        except ValueError:
            parsed[key] = value
    return parsed


def sigma_key(sigma: float) -> str:
    """Column suffix used by the registered analysis for radius r = sigma."""
    return format(float(sigma), ".12g")


def ext_display(candidate_id: str) -> str:
    return EXT_DISPLAY.get(candidate_id) or candidate_display(candidate_id)


def ext_region_points(registration: dict[str, Any]) -> float:
    """The extension's registered prospective practical-effect region (X1), in points (a reporting region, not a margin)."""

    region = registration.get("prospective_practical_effect_region") or {}
    if "absolute_ca" not in region:
        raise RuntimeError(f"{registration.get('experiment_id')}: no prospective practical-effect region registered")
    return 100.0 * float(region["absolute_ca"])


def ext_text_name(candidate_id: str) -> str:
    """The display name for running text: lower-case initial unless it starts an acronym (GR-CLIP)."""

    label = ext_display(candidate_id)
    return label[0].lower() + label[1:] if len(label) > 1 and label[1].islower() else label


@dataclass
class ExtensionAnalysis:
    key: str
    experiment_id: str
    rows: dict[str, dict[str, dict[str, Any]]]  # candidate -> cell -> cells.csv row
    concentration: dict[str, dict[str, dict[str, Any]]]  # candidate -> cell -> row
    contrasts: dict[str, Any] | None
    contrasts_reason: str
    primary_cells: list[str]
    grid: list[str]
    controls: list[str]


def load_extension_analysis(sources: Sources, key: str, experiment_id: str) -> ExtensionAnalysis | None:
    base = f"{EXT_ANALYSIS_DIR_REL}/{experiment_id}"
    if not (sources.root / base / "cells.csv").is_file():
        return None
    rows: dict[str, dict[str, dict[str, Any]]] = {}
    for row in sources.read_csv(f"{base}/cells.csv"):
        rows.setdefault(str(row["candidate_id"]), {})[str(row["cell_id"])] = row
    concentration: dict[str, dict[str, dict[str, Any]]] = {}
    for row in sources.read_csv(f"{base}/concentration.csv"):
        concentration.setdefault(str(row["candidate_id"]), {})[str(row["cell_id"])] = row
    sources.register(f"{base}/macro.csv")
    if NO_CORRECTION not in rows:
        raise RuntimeError(f"{experiment_id}: reference bank absent from cells.csv")
    cell_ids = sorted(rows[NO_CORRECTION])
    for candidate, per_cell in rows.items():
        if sorted(per_cell) != cell_ids:
            raise RuntimeError(f"{experiment_id}: candidate {candidate} does not cover every cell")
    primary_cells = [cell_id for cell_id in cell_ids if rows[NO_CORRECTION][cell_id]["primary_estimand_cell"]]
    grid = sorted(
        (candidate for candidate in rows if candidate != NO_CORRECTION and not candidate.startswith(EXT_CONTROL_PREFIX)),
        key=candidate_sort_key,
    )
    controls = [candidate for candidate in EXT_CONTROL_ORDER if candidate in rows]
    if len(grid) != 17 or len(controls) != 4 or len(rows) != 22:
        raise RuntimeError(f"{experiment_id}: expected 17 grid banks, 4 controls and the identity bank")
    contrasts_rel = f"{base}/contrasts.json"
    if (sources.root / contrasts_rel).is_file():
        contrasts = sources.read_json(contrasts_rel)
        reason = ""
        if contrasts.get("experiment_id") != experiment_id or contrasts.get("reference") != NO_CORRECTION:
            raise RuntimeError(f"{experiment_id}: contrasts.json identity mismatch")
    else:
        contrasts = None
        reason = f"{contrasts_rel} has not been written yet (registered bootstrap still running)"
    return ExtensionAnalysis(key, experiment_id, rows, concentration, contrasts, reason, primary_cells, grid, controls)


def prediction_summary(in_a: bool, in_b: bool) -> str:
    """One phrase for the confirmatory predictions X3 and X4 over Extensions A and B (addendum A2.2, A2.3)."""

    if in_a and in_b:
        return "confirmed in both Extension A and Extension B"
    if in_a:
        return "confirmed in Extension A but not in Extension B"
    if in_b:
        return "confirmed in Extension B but not in Extension A"
    return "not confirmed in either Extension A or Extension B"


def x5_summary(in_a: bool, in_b: bool) -> str:
    """One phrase for the descriptive prediction X5: the outcome-blind addendum (A1) allows only "supported" and
    "not uniformly supported", with no inferential statement."""

    if in_a and in_b:
        return "supported in both Extension A and Extension B"
    if in_a:
        return "supported in Extension A but not uniformly supported in Extension B"
    if in_b:
        return "not uniformly supported in Extension A but supported in Extension B"
    return "not uniformly supported in either Extension A or Extension B"


def ext_x4_confirmed(intervals: dict[str, dict[str, float]]) -> bool:
    """Registered X4 (outcome-blind addendum A2.3): confirmed if AT LEAST ONE per-class control's simultaneous
    interval lies above zero. The registration's wording "the per-class control banks leave the band" was resolved to
    this existential rule before any outcome; do not strengthen it to "every" (review of V11, 2026-09-29)."""

    return any(float(intervals[candidate]["lower"]) > 0.0 for candidate in EXT_PER_CLASS_CONTROLS)


def x5_group_supported(values: list[tuple[float, float, float]]) -> bool:
    """Registered X5 for one (model, dataset): top-class share AND predicted-class Gini strictly increase between
    every pair of adjacent sigmas; ``values`` holds (sigma, top-class share, Gini) tuples in any order."""

    ordered = sorted(values)
    return all(later[1] > earlier[1] and later[2] > earlier[2] for earlier, later in zip(ordered, ordered[1:]))


class Macros:
    """Ordered macro store with a MISSING ledger."""

    PLACEHOLDER = r"\textbf{??}"

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.sections: list[tuple[str, list[str]]] = []
        self.missing: list[dict[str, str]] = []

    def section(self, title: str) -> None:
        self.sections.append((title, []))

    def set(self, name: str, value: str) -> None:
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"invalid LaTeX macro name: {name}")
        if name in self.values:
            raise ValueError(f"macro defined twice: {name}")
        if not self.sections:
            self.section("macros")
        self.values[name] = str(value)
        self.sections[-1][1].append(name)

    def miss(self, name: str, reason: str) -> None:
        self.set(name, self.PLACEHOLDER)
        self.missing.append({"macro": name, "reason": reason})

    def render(self, header_lines: list[str]) -> str:
        lines = list(header_lines)
        for title, names in self.sections:
            lines.append("")
            lines.append(f"% --- {title}")
            for name in names:
                lines.append(f"\\newcommand{{\\{name}}}{{{self.values[name]}}}")
        lines.append("")
        return "\n".join(lines)


@dataclass
class Cell:
    cell_id: str
    model_id: str
    dataset_id: str
    fold: int
    item_count: int
    candidates: dict[str, dict[str, Any]]


def load_cells(sources: Sources) -> list[Cell]:
    cells_dir = sources.root / CELLS_REL
    cell_ids = sorted(path.name for path in cells_dir.iterdir() if (path / CELL_METRICS_NAME).is_file())
    cells: list[Cell] = []
    for cell_id in cell_ids:
        metrics = sources.read_json(f"{CELLS_REL}/{cell_id}/{CELL_METRICS_NAME}")
        if metrics["cell_id"] != cell_id:
            raise RuntimeError(f"cell id mismatch in {cell_id}")
        candidates = {record["candidate_id"]: record for record in metrics["candidates"]}
        if len(candidates) != metrics["candidate_count"]:
            raise RuntimeError(f"candidate count mismatch in {cell_id}")
        model, dataset, fold = parse_cell_id(cell_id)
        if metrics["model_id"] != model or metrics["dataset_id"] != dataset or int(metrics["fold"]) != fold:
            raise RuntimeError(f"cell metadata mismatch in {cell_id}")
        cells.append(Cell(cell_id, model, dataset, fold, int(metrics["item_count"]), candidates))
    return cells


# --------------------------------------------------------------------------
# Per-example partition (torch-guarded)
# --------------------------------------------------------------------------


def compute_partition(root: Path, cell_ids: list[str], radius: float) -> dict[str, Any]:
    """Failure partition at ``radius`` from the immutable sufficient statistics.

    Only decision fields are read; ``selection_counts`` and
    ``confirmation_counts`` are never accessed.
    """
    import torch  # guarded by the caller

    per_cell: dict[str, Any] = {}
    pooled: dict[str, dict[str, int]] = {}
    changed_totals: dict[str, int] = {}
    signatures: dict[str, Any] = {}
    radii_store: dict[str, dict[str, Any]] = {}
    positions = 0
    candidate_order: list[str] | None = None
    for cell_id in sorted(cell_ids):
        path = root / CELLS_REL / cell_id / SUFFICIENT_STATISTICS_NAME
        raw = torch.load(path, map_location="cpu", weights_only=True)
        candidates = [str(value) for value in raw["candidate_ids"]]
        if candidate_order is None:
            candidate_order = candidates
        elif candidates != candidate_order:
            raise RuntimeError(f"candidate order differs in {cell_id}")
        item_ids = [str(value) for value in raw["item_ids"]]
        if len(set(item_ids)) != len(item_ids):
            raise RuntimeError(f"duplicate item ids in {cell_id}")
        truth = torch.as_tensor(raw["ground_truth"]).long()
        raw_predictions = torch.as_tensor(raw["raw_predictions"]).long()
        n = int(truth.numel())
        if raw_predictions.shape != (len(candidates), n) or len(item_ids) != n:
            raise RuntimeError(f"shape mismatch in {cell_id}")
        if set(raw["outcomes"]) != set(candidates):
            raise RuntimeError(f"outcome candidates differ from candidate_ids in {cell_id}")
        positions += n
        reference_outcome = raw["outcomes"][NO_CORRECTION]
        reference_event = (
            ~torch.as_tensor(reference_outcome["abstained"]).bool()
            & torch.as_tensor(reference_outcome["selected_classes"]).long().eq(truth)
            & torch.as_tensor(reference_outcome["certificate_radii"]).double().ge(radius)
        )
        cell_counts: dict[str, dict[str, Any]] = {}
        for index, candidate in enumerate(candidates):
            outcome = raw["outcomes"][candidate]
            selected = torch.as_tensor(outcome["selected_classes"]).long()
            radii = torch.as_tensor(outcome["certificate_radii"]).double()
            abstained = torch.as_tensor(outcome["abstained"]).bool()
            raw_prediction = raw_predictions[index]
            active = ~abstained
            correct = selected.eq(truth)
            reached = radii.ge(radius)
            anchored = selected.eq(raw_prediction)
            counts = {
                "abstained": int(abstained.sum()),
                "nonabstaining_wrong_smoothed_class": int((active & ~correct).sum()),
                "correct_smoothed_class_but_radius_below_target": int((active & correct & ~reached).sum()),
                "certified_correct_at_target": int((active & correct & reached).sum()),
                "certified_wrong_at_target": int((active & ~correct & reached).sum()),
                "nonabstaining_anchor_mismatch": int((active & ~anchored).sum()),
                "anchor_preserved_but_raw_class_wrong": int((active & anchored & ~raw_prediction.eq(truth)).sum()),
                "anchor_correct_but_radius_below_target": int((active & anchored & correct & ~reached).sum()),
                "anchored_certified_correct_at_target": int((active & anchored & correct & reached).sum()),
            }
            event = active & correct & reached
            certified_wrong = active & ~correct & reached
            top_class: int | None = None
            top_count = 0
            ties = 0
            if bool(certified_wrong.any()):
                class_counts = torch.bincount(selected[certified_wrong])
                top_count = int(class_counts.max())
                top_class = int(class_counts.argmax())
                ties = int(class_counts.eq(top_count).sum())
            counts["standard_event_changed_vs_no_correction"] = int(event.ne(reference_event).sum())
            counts["certified_wrong_top_class"] = top_class
            counts["certified_wrong_top_class_count"] = top_count
            counts["certified_wrong_top_class_ties"] = ties
            cell_counts[candidate] = counts
            totals = pooled.setdefault(candidate, {key: 0 for key in PARTITION_KEYS})
            for key in PARTITION_KEYS:
                totals[key] += counts[key]
            changed_totals[candidate] = changed_totals.get(candidate, 0) + counts["standard_event_changed_vs_no_correction"]
            signature = signatures.setdefault(candidate, hashlib.sha256())
            signature.update(cell_id.encode("utf-8"))
            signature.update(raw_prediction.to(torch.int64).numpy().tobytes())
            signature.update(selected.to(torch.int64).numpy().tobytes())
            signature.update(abstained.to(torch.uint8).numpy().tobytes())
            radii_store.setdefault(candidate, {})[cell_id] = radii
        per_cell[cell_id] = {"example_count": n, "candidates": cell_counts}

    groups: dict[str, list[str]] = {}
    for candidate, signature in signatures.items():
        groups.setdefault(signature.hexdigest(), []).append(candidate)
    signature_groups = sorted(sorted(members) for members in groups.values())
    max_gap = 0.0
    for members in signature_groups:
        for cell_id in sorted(cell_ids):
            reference = radii_store[members[0]][cell_id]
            for other in members[1:]:
                max_gap = max(max_gap, float((radii_store[other][cell_id] - reference).abs().max()))
    return {
        "radius": radius,
        "positions": positions,
        "fields_accessed": list(PARTITION_FIELDS_ACCESSED),
        "cells": per_cell,
        "pooled": pooled,
        "decision_signature_groups": signature_groups,
        "distinct_decision_classifiers": len(signature_groups),
        "max_certificate_radius_gap_within_groups": max_gap,
        "changed_standard_positions_vs_no_correction": changed_totals,
    }


def load_partition(root: Path, cell_ids: list[str], radius: float) -> tuple[dict[str, Any] | None, str]:
    """Return (partition, mode) or (None, reason) without raising."""
    try:
        import torch  # noqa: F401
    except ImportError:
        torch_available = False
    else:
        torch_available = True
    if torch_available:
        try:
            return compute_partition(root, cell_ids, radius), "in-process torch"
        except Exception as error:  # pragma: no cover - defensive
            return None, f"in-process partition failed: {type(error).__name__}: {error}"
    interpreter = next((root / rel for rel in TORCH_INTERPRETERS if (root / rel).exists()), None)
    if interpreter is None:
        return None, "torch is not importable and no .venv-gate-n2 interpreter was found"
    command = [
        str(interpreter),
        str(Path(__file__).resolve()),
        "--partition-worker",
        "--root",
        str(root),
        "--radius",
        repr(radius),
        "--cells",
        ",".join(sorted(cell_ids)),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return None, f"partition worker failed under {interpreter}: {completed.stderr.strip()[-500:]}"
    try:
        return json.loads(completed.stdout), f"subprocess worker under {interpreter.relative_to(root)}"
    except json.JSONDecodeError as error:
        return None, f"partition worker returned invalid JSON: {error}"


# --------------------------------------------------------------------------
# Assets
# --------------------------------------------------------------------------


@dataclass
class Assets:
    macros: Macros
    sources: Sources
    notes: dict[str, Any]
    tables: dict[str, str]


def _source_comment(sources: Sources, rels: list[str], description: str) -> list[str]:
    lines = [
        f"% Generated by {SCRIPT_REL}; do not edit by hand.",
        f"% {description}",
        "% Sources (SHA-256):",
    ]
    for rel in rels:
        lines.append(f"%   {rel}  {sources.hashes[rel]}")
    return lines


def _tabular(colspec: str, header: list[str], body: list[str]) -> list[str]:
    lines = [f"\\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines.extend(body)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return lines


def _table(
    caption: str,
    label: str,
    tabular_lines: list[str],
    *,
    star: bool = False,
    colsep: str | None = None,
) -> list[str]:
    environment = "table*" if star else "table"
    lines = [f"\\begin{{{environment}}}[t]", r"\centering", r"\scriptsize"]
    if colsep is not None:
        lines.append(f"\\setlength{{\\tabcolsep}}{{{colsep}}}")
    lines.extend([f"\\caption{{{caption}}}", f"\\label{{{label}}}"])
    lines.extend(tabular_lines)
    lines.append(f"\\end{{{environment}}}")
    return lines


def _math(signed: str) -> str:
    return f"${signed}$"


def _basenames(rels: list[str]) -> str:
    seen: list[str] = []
    for rel in rels:
        name = Path(rel).name
        if name not in seen:
            seen.append(name)
    return ", ".join(f"\\texttt{{{tex_escape(name)}}}" for name in seen)


def build(root: Path) -> Assets:
    sources = Sources(root)
    macros = Macros()
    notes: dict[str, Any] = {}

    summary = sources.read_json(SUMMARY_REL)
    selection = sources.read_json(SELECTION_REL)
    audit = sources.read_json(AUDIT_REL)
    gate = sources.read_json(GATE_N2_REL)
    diagnostics = sources.read_json(DIAGNOSTICS_REL)
    cells = load_cells(sources)
    cell_rels = [f"{CELLS_REL}/{cell.cell_id}/{CELL_METRICS_NAME}" for cell in cells]

    if summary["experiment_id"] != audit["experiment_id"] != gate["experiment_id"]:
        raise RuntimeError("experiment identifiers disagree across artifacts")
    if audit["source_artifact_set_sha256"] != gate["source_artifact_set_sha256"] != diagnostics["source_artifact_set_sha256"]:
        raise RuntimeError("artifact-set identity disagrees across artifacts")

    aggregates = audit["candidate_aggregates_equal_cell_weight"]
    per_candidate_cells: dict[str, dict[str, dict[str, Any]]] = {
        candidate: {record["cell_id"]: record for record in records}
        for candidate, records in audit["per_candidate_cells"].items()
    }
    candidate_ids = sorted(aggregates, key=candidate_sort_key)
    if set(candidate_ids) != set(per_candidate_cells) or len(candidate_ids) != summary["candidate_count_per_cell"]:
        raise RuntimeError("audit candidate family is incomplete")
    cell_by_id = {cell.cell_id: cell for cell in cells}
    if set(cell_by_id) != set(per_candidate_cells[NO_CORRECTION]) or len(cells) != summary["cell_count"]:
        raise RuntimeError("cell set differs between results and audit")
    example_counts = {cell.cell_id: cell.item_count for cell in cells}
    for candidate in candidate_ids:
        for cell_id, record in per_candidate_cells[candidate].items():
            if int(record["example_count"]) != example_counts[cell_id]:
                raise RuntimeError(f"example_count mismatch for {candidate}/{cell_id}")
    positions = sum(example_counts.values())

    adjudication = diagnostics["floating_boundary_adjudication"]
    saved_id = adjudication["saved_selected_proposal"]
    exact_id = adjudication["exact_count_selected_proposal"]
    if not (
        saved_id
        == audit["saved_selection_id"]
        == audit["historical_selection"]["candidate_id"]
        == summary["selected_proposed_candidate_id"]
        == selection["selected_proposed_candidate"]["candidate_id"]
    ):
        raise RuntimeError("saved selection identity disagrees across artifacts")
    notes["selections"] = {"saved": saved_id, "exact_count": exact_id}

    # ---------------------------------------------------------------- scope
    macros.section("EXP-017 scope")
    candidates_per_cell = int(summary["candidate_count_per_cell"])
    macros.set("ExpCells", count(summary["cell_count"]))
    macros.set("ExpPositions", count(positions))
    macros.set("ExpRows", count(positions * candidates_per_cell))
    macros.set("ExpSelVotes", count(positions * candidates_per_cell * int(summary["cohen_selection_samples_per_example"])))
    macros.set("ExpConfVotes", count(positions * candidates_per_cell * int(summary["cohen_confirmation_samples_per_example"])))
    macros.set("ExpBanks", count(int(summary["cell_count"]) * candidates_per_cell))
    macros.set("ExpCandidates", count(candidates_per_cell))
    macros.set("ExpOperatorBanks", count(candidates_per_cell - 1))  # every registered bank except the identity (Fig. 2 rows)
    macros.set("ExpElapsedHours", f"{float(summary['elapsed_seconds']) / 3600.0:.1f}")
    macros.set("SelectedComparator", tex_escape(str(summary["selected_primary_comparator_id"])))

    partition, partition_mode = load_partition(root, list(cell_by_id), TARGET_RADIUS)
    pt_rels = [f"{CELLS_REL}/{cell.cell_id}/{SUFFICIENT_STATISTICS_NAME}" for cell in cells]
    if partition is not None:
        for rel in pt_rels:
            digest = sources.register(rel)
            cell_id = Path(rel).parent.name
            expected = audit["source_statistics_sha256"].get(cell_id)
            if expected is not None and expected != digest:
                raise RuntimeError(f"sufficient statistics changed since the audit: {cell_id}")
        if int(partition["positions"]) != positions:
            raise RuntimeError("partition positions differ from the cell metrics")
        macros.set("ExpDistinctClassifiers", count(partition["distinct_decision_classifiers"]))
        notes["distinct_classifiers"] = {
            "definition": (
                "number of candidates with pairwise-distinct per-example decisions "
                "(raw prediction, selected smoothed class, abstention) over all 12 cells; "
                "candidates sharing every decision are one classifier"
            ),
            "identical_decision_groups": [group for group in partition["decision_signature_groups"] if len(group) > 1],
            "max_certificate_radius_gap_within_identical_groups": partition["max_certificate_radius_gap_within_groups"],
        }
    else:
        macros.miss("ExpDistinctClassifiers", f"per-example decisions unavailable: {partition_mode}")
    macros.set("ExpSelDraws", count(summary["cohen_selection_samples_per_example"]))
    macros.set("ExpConfDraws", count(summary["cohen_confirmation_samples_per_example"]))
    macros.set("ExpAlpha", num(summary["cohen_alpha_per_example"]))
    macros.set("ExpSigma", num(summary["cohen_sigma"]))
    input_sides: set[int] = set()
    input_channels: set[int] = set()
    for rel in CLEAN_PIPELINE_RELS:
        preprocessing = sources.read_json(rel)["preprocessing"]
        if preprocessing.get("convert_rgb") is not True:
            raise RuntimeError(f"{rel}: expected an RGB clean pipeline")
        input_sides.add(int(preprocessing["crop_size"]))
        input_channels.add(len(preprocessing["mean"]))
    if len(input_sides) == 1 and len(input_channels) == 1:
        side = next(iter(input_sides))
        channels = next(iter(input_channels))
        per_coordinate = TARGET_RADIUS / math.sqrt(channels * side * side)
        mantissa, exponent = f"{per_coordinate:.1e}".split("e")
        macros.set("ExpRadiusPerCoordinate", f"{mantissa}\\times 10^{{{int(exponent)}}}")
        notes["radius_per_coordinate"] = {
            "formula": "target radius / sqrt(channels x crop_size x crop_size), two significant digits, for math mode",
            "target_radius": TARGET_RADIUS,
            "channels": channels,
            "crop_size": side,
            "input_dimension": channels * side * side,
            "value": per_coordinate,
            "constant_sources": list(CLEAN_PIPELINE_RELS),
        }
    else:
        macros.miss(
            "ExpRadiusPerCoordinate",
            f"input geometry differs across clean pipelines: sides {sorted(input_sides)}, channels {sorted(input_channels)}",
        )
    items_by_dataset: dict[str, set[int]] = {}
    for cell in cells:
        items_by_dataset.setdefault(cell.dataset_id, set()).add(cell.item_count)
    for dataset, name in (("cifar100", "ExpCifarItemsPerCell"), ("eurosat", "ExpEurosatItemsPerCell")):
        values = items_by_dataset.get(dataset, set())
        if len(values) == 1:
            macros.set(name, count(next(iter(values))))
        else:
            macros.miss(name, f"dataset {dataset} has item counts {sorted(values)} in cell_metrics.json")

    # ------------------------------------------------- equal-cell aggregates
    def equal_cell(candidate: str, extract) -> float:
        values = [extract(record) for record in per_candidate_cells[candidate].values()]
        return sum(values) / len(values)

    def checked_equal_cell(candidate: str, extract, stored: float, what: str) -> float:
        value = equal_cell(candidate, extract)
        if abs(value - float(stored)) > 1e-9:
            raise RuntimeError(f"equal-cell recomputation differs from the audit for {candidate} {what}")
        return value

    def std_ca(candidate: str, radius: float = TARGET_RADIUS) -> float:
        return checked_equal_cell(
            candidate,
            lambda record: radius_value(record["standard_certified_accuracy"], radius),
            radius_value(aggregates[candidate]["standard_certified_accuracy"], radius),
            f"standard CA at {radius}",
        )

    def anch_ca(candidate: str, radius: float = TARGET_RADIUS) -> float:
        return checked_equal_cell(
            candidate,
            lambda record: radius_value(record["anchored_certified_accuracy"], radius),
            radius_value(aggregates[candidate]["anchored_certified_accuracy"], radius),
            f"anchored CA at {radius}",
        )

    def scalar(candidate: str, key: str) -> float:
        return checked_equal_cell(candidate, lambda record: record[key], aggregates[candidate][key], key)

    macros.section("Equal-cell aggregates at radius 0.25 (soundness audit)")
    macros.set("NoCorrStdCA", pct(std_ca(NO_CORRECTION)))
    macros.set("NoCorrAnchCA", pct(anch_ca(NO_CORRECTION)))
    macros.set("NoCorrClean", pct(scalar(NO_CORRECTION, "raw_clean_accuracy")))
    macros.set("NoCorrSmoothed", pct(scalar(NO_CORRECTION, "smoothed_accuracy")))
    macros.set("NoCorrAbstain", pct(scalar(NO_CORRECTION, "abstention_rate")))
    macros.set("NoCorrAgreement", pct(scalar(NO_CORRECTION, "anchor_agreement")))
    macros.set("SavedStdCA", pct(std_ca(saved_id)))
    macros.set("SavedAnchCA", pct(anch_ca(saved_id)))
    macros.set("SavedClean", pct(scalar(saved_id, "raw_clean_accuracy")))
    macros.set("ExactStdCA", pct(std_ca(exact_id)))
    macros.set("ExactAnchCA", pct(anch_ca(exact_id)))
    macros.set("ExactClean", pct(scalar(exact_id, "raw_clean_accuracy")))
    macros.set("GRStdCA", pct(std_ca(GR_CLIP)))
    macros.set("GRAnchCA", pct(anch_ca(GR_CLIP)))
    macros.set("GRClean", pct(scalar(GR_CLIP, "raw_clean_accuracy")))
    macros.set("GRAgreement", pct(scalar(GR_CLIP, "anchor_agreement")))
    macros.set("GMCStdCA", pct(std_ca(GMC_ONE)))
    macros.set("GMCAnchCA", pct(anch_ca(GMC_ONE)))
    macros.set("GMCClean", pct(scalar(GMC_ONE, "raw_clean_accuracy")))
    macros.set("GMCAgreement", pct(scalar(GMC_ONE, "anchor_agreement")))
    notes["equal_cell_macro"] = "unweighted mean over the 12 model-dataset-fold cells, recomputed from per_candidate_cells and checked against candidate_aggregates_equal_cell_weight (tolerance 1e-9)"

    # --------------------------------------------------------------- Gate N2
    critical = float(gate["common_critical_value"])
    contrasts: dict[tuple[str, str, float], dict[str, Any]] = {}
    for row in gate["contrasts"]:
        key = (row["candidate_id"], row["metric"], float(row["radius"]))
        if key in contrasts:
            raise RuntimeError(f"duplicate Gate-N2 contrast {key}")
        contrasts[key] = row
        if not (
            math.isclose(row["simultaneous_lower"], row["point_estimate"] - critical, abs_tol=1e-12)
            and math.isclose(row["simultaneous_upper"], row["point_estimate"] + critical, abs_tol=1e-12)
        ):
            raise RuntimeError(f"Gate-N2 band is not point +/- common critical value for {key}")
    if len(contrasts) != int(gate["simultaneous_hypotheses"]):
        raise RuntimeError("Gate-N2 contrast count differs from simultaneous_hypotheses")

    def contrast(candidate: str, metric: str, radius: float) -> dict[str, Any]:
        return contrasts[(candidate, metric, radius)]

    def band_macros(prefix: str, candidate: str, metric: str, radius: float, suffix: str = "") -> None:
        row = contrast(candidate, metric, radius)
        macros.set(f"{prefix}Delta{suffix}", spct(row["point_estimate"]))
        macros.set(f"{prefix}Lower{suffix}", spct(row["simultaneous_lower"]))
        macros.set(f"{prefix}Upper{suffix}", spct(row["simultaneous_upper"]))

    excluded = [
        row for row in gate["contrasts"] if row["simultaneous_lower"] > 0.0 or row["simultaneous_upper"] < 0.0
    ]
    macros.section("Gate-N2 simultaneous analysis (radius 0.25 unless noted)")
    macros.set("BandHalfWidth", pts(100.0 * critical))
    macros.set("BandReplicates", count(gate["replicates"]))
    macros.set("BandContrasts", count(len(contrasts)))
    macros.set("BandStrata", count(gate["strata"]))
    macros.set("BandExclZero", count(len(excluded)))
    macros.set("BandExclZeroGR", count(sum(row["candidate_id"] == GR_CLIP for row in excluded)))
    # The manuscript scopes uniqueness to the primary radius and names the other exclusions (V5 reviews):
    # at radius 0.25 only GR-CLIP excludes zero, every GR-CLIP exclusion is positive, and every other
    # exclusion is a negative standard-metric global-mean-centering contrast at radius 0.
    other_excluded = [row for row in excluded if row["candidate_id"] != GR_CLIP]
    if any(float(row["radius"]) == TARGET_RADIUS for row in other_excluded):
        raise RuntimeError("a bank other than GR-CLIP excludes zero at the primary radius")
    if not all(row["simultaneous_lower"] > 0.0 for row in excluded if row["candidate_id"] == GR_CLIP):
        raise RuntimeError("a GR-CLIP interval excludes zero from below")
    if not all(
        row["simultaneous_upper"] < 0.0
        and float(row["radius"]) == ZERO_RADIUS
        and row["metric"] == STANDARD_METRIC
        and candidate_family(row["candidate_id"]) == candidate_family(GMC_ONE)
        for row in other_excluded
    ):
        raise RuntimeError("an exclusion outside GR-CLIP is not a negative radius-0 standard mean-centering contrast")
    macros.set("BandExclZeroOther", count(len(other_excluded)))
    band_macros("GRStd", GR_CLIP, STANDARD_METRIC, TARGET_RADIUS)
    band_macros("GRAnch", GR_CLIP, ANCHORED_METRIC, TARGET_RADIUS)
    band_macros("SavedStd", saved_id, STANDARD_METRIC, TARGET_RADIUS)
    band_macros("ExactStd", exact_id, STANDARD_METRIC, TARGET_RADIUS)
    band_macros("GMCStd", GMC_ONE, STANDARD_METRIC, TARGET_RADIUS)
    band_macros("GMCStd", GMC_ONE, STANDARD_METRIC, ZERO_RADIUS, suffix="RZero")
    band_macros("GMCAnch", GMC_ONE, ANCHORED_METRIC, ZERO_RADIUS, suffix="RZero")
    chowers_rows = [row for row in gate["contrasts"] if candidate_family(row["candidate_id"]) == CHOWERS_FAMILY]
    if len({row["candidate_id"] for row in chowers_rows}) != 3:
        raise RuntimeError("expected three Chowers candidates in the Gate-N2 family")
    macros.set("ChowersMaxAbsDelta", pts(100.0 * max(abs(float(row["point_estimate"])) for row in chowers_rows)))
    half_sens = critical * math.sqrt(5.0 / 4.0)
    gr_std = contrast(GR_CLIP, STANDARD_METRIC, TARGET_RADIUS)
    macros.set("BandHalfWidthSens", pts(100.0 * half_sens))
    macros.set("GRStdLowerSens", spct(gr_std["point_estimate"] - half_sens))
    macros.set("GRStdUpperSens", spct(gr_std["point_estimate"] + half_sens))
    notes["gate_n2"] = {
        "common_critical_value": critical,
        "band_excluding_zero": [
            {"candidate_id": row["candidate_id"], "metric": row["metric"], "radius": row["radius"]} for row in excluded
        ],
        "finite_stratum_sensitivity": "descriptive only: half width = common critical value x sqrt(5/4)",
    }

    # ------------------------------------------------- weighting sensitivity
    def cell_counts(candidate: str, extract, what: str) -> dict[str, int]:
        return {
            cell_id: exact_count(extract(record), example_counts[cell_id], what=f"{candidate}/{cell_id} {what}")
            for cell_id, record in per_candidate_cells[candidate].items()
        }

    def std_counts(candidate: str) -> dict[str, int]:
        return cell_counts(candidate, lambda record: radius_value(record["standard_certified_accuracy"], TARGET_RADIUS), "standard CA")

    reference_std_counts = std_counts(NO_CORRECTION)

    def pooled_delta(candidate: str) -> Fraction:
        candidate_counts = std_counts(candidate)
        numerator = sum(candidate_counts[cell_id] - reference_std_counts[cell_id] for cell_id in example_counts)
        return Fraction(100 * numerator, positions)

    macros.section("Weighting sensitivity (descriptive)")
    macros.set("GRStdDeltaPooled", spts(float(pooled_delta(GR_CLIP))))
    macros.set("GMCStdDeltaPooled", spts(float(pooled_delta(GMC_ONE))))
    macros.set("SavedStdDeltaPooled", spts(float(pooled_delta(saved_id))))
    macros.set("ExactStdDeltaPooled", spts(float(pooled_delta(exact_id))))
    cell_total = len(cells)
    weight_sum = sum(Fraction(n, cell_total * n) for n in example_counts.values())
    weight_square_sum = sum(n * Fraction(1, cell_total * n) ** 2 for n in example_counts.values())
    design_effect = Fraction(positions) * weight_square_sum / (weight_sum**2)
    effective_n = Fraction(positions) / design_effect
    macros.set("KishDesignEffect", f"{float(design_effect):.3f}")
    macros.set("EffectiveN", f"{float(effective_n):,.1f}")
    notes["weighting"] = {
        "pooled_delta": "sum_c n_c (CA_cand,c - CA_ref,c) / sum_c n_c, in points, from exact per-cell counts",
        "kish_design_effect": design_effect,
        "effective_n": effective_n,
        "weights": "w_i = 1/(12 n_c) for every item in cell c",
    }

    # ------------------------------------ GR-CLIP standard-CA change by dataset
    def cell_std_rate(candidate: str, cell_id: str) -> float:
        return float(radius_value(per_candidate_cells[candidate][cell_id]["standard_certified_accuracy"], TARGET_RADIUS))

    gr_std_counts = std_counts(GR_CLIP)
    macros.section("GR-CLIP standard-CA change versus no correction by dataset (equal-cell within dataset)")
    dataset_deltas: dict[str, list[float]] = {}
    for cell in cells:
        dataset_deltas.setdefault(cell.dataset_id, []).append(
            cell_std_rate(GR_CLIP, cell.cell_id) - cell_std_rate(NO_CORRECTION, cell.cell_id)
        )
    for dataset, name in (("cifar100", "GRStdDeltaCifar"), ("eurosat", "GRStdDeltaEurosat")):
        deltas = dataset_deltas.get(dataset, [])
        if deltas:
            macros.set(name, spct(sum(deltas) / len(deltas)))
        else:
            macros.miss(name, f"no cells for dataset {dataset}")
    cifar_cells = [cell.cell_id for cell in cells if cell.dataset_id == "cifar100"]
    macros.set(
        "GRStdCifarCellsNegative",
        count(sum(gr_std_counts[cell_id] < reference_std_counts[cell_id] for cell_id in cifar_cells)),
    )
    notes["gr_by_dataset"] = {
        "definition": "unweighted mean over the cells of one dataset of (GR-CLIP standard CA - no-correction standard CA) at radius 0.25",
        "cells_per_dataset": {dataset: len(values) for dataset, values in dataset_deltas.items()},
        "cifar100_cells_with_negative_change": [
            cell_id for cell_id in cifar_cells if gr_std_counts[cell_id] < reference_std_counts[cell_id]
        ],
    }

    # -------------------------------------------- standard minus anchored gap
    macros.section("Standard minus anchored CA at radius 0.25 over the candidate family")
    gaps: dict[str, float] = {}
    for candidate in candidate_ids:
        gap = std_ca(candidate) - anch_ca(candidate)
        stored_gap = float(radius_value(aggregates[candidate]["standard_minus_anchored"], TARGET_RADIUS))
        if abs(gap - stored_gap) > 1e-9:
            raise RuntimeError(f"standard-minus-anchored gap differs from the audit for {candidate}")
        gaps[candidate] = gap
    macros.set("StdAnchGapMin", pts(100.0 * min(gaps.values())))
    macros.set("StdAnchGapMax", pts(100.0 * max(gaps.values())))
    notes["standard_minus_anchored"] = {
        "min": min(gaps, key=gaps.get),
        "max": max(gaps, key=gaps.get),
        "definition": "equal-cell standard CA minus equal-cell anchored CA at radius 0.25, checked against the audit's standard_minus_anchored",
    }

    # -------------------------------------------------- clean gate adjudication
    clean_counts: dict[str, dict[str, int]] = {}
    for candidate in candidate_ids:
        clean_counts[candidate] = {}
        for cell in cells:
            rate = float(cell.candidates[candidate]["raw_clean_accuracy"])
            audit_rate = float(per_candidate_cells[candidate][cell.cell_id]["raw_clean_accuracy"])
            if abs(rate - audit_rate) > 1e-12:
                raise RuntimeError(f"clean accuracy differs between cell_metrics and audit for {candidate}/{cell.cell_id}")
            clean_counts[candidate][cell.cell_id] = exact_count(rate, cell.item_count, what=f"{candidate}/{cell.cell_id} clean")

    def clean_drop(candidate: str, cell_id: str) -> Fraction:
        return Fraction(100 * (clean_counts[NO_CORRECTION][cell_id] - clean_counts[candidate][cell_id]), example_counts[cell_id])

    def macro_drop(candidate: str) -> Fraction:
        return sum(clean_drop(candidate, cell_id) for cell_id in example_counts) / cell_total

    def exact_eligible(candidate: str) -> bool:
        return macro_drop(candidate) <= 1 and all(clean_drop(candidate, cell_id) <= 2 for cell_id in example_counts)

    def historical_eligible(candidate: str) -> bool:
        return bool(selection["all_aggregates"][candidate]["clean_accuracy_constraint_satisfied"])

    for candidate in candidate_ids:
        stored = float(selection["all_aggregates"][candidate]["macro_clean_accuracy_drop_percentage_points"])
        if abs(stored - float(macro_drop(candidate))) > 1e-9:
            raise RuntimeError(f"macro clean drop differs from selection.json for {candidate}")
    rejected = [candidate for candidate in candidate_ids if not historical_eligible(candidate) and exact_eligible(candidate)]
    artifact_values = sorted(
        {repr(float(selection["all_aggregates"][candidate]["maximum_cell_clean_accuracy_drop_percentage_points"])) for candidate in rejected}
    )
    gr_drops = {cell_id: clean_drop(GR_CLIP, cell_id) for cell_id in example_counts}
    worst_cell = max(sorted(gr_drops), key=lambda cell_id: gr_drops[cell_id])
    worst = cell_by_id[worst_cell]
    forced = diagnostics["forced_exact_proposal_vs_each_mean_centering"]

    macros.section("Clean gate adjudication")
    macros.set("GRWorstCellDrop", pts(float(gr_drops[worst_cell])))
    macros.set("GRWorstCell", cell_display(worst.model_id, worst.dataset_id, worst.fold))
    macros.set("GRWorstCellImages", count(clean_counts[NO_CORRECTION][worst_cell] - clean_counts[GR_CLIP][worst_cell]))
    macros.set("GRCellsImproved", count(sum(clean_counts[GR_CLIP][cell_id] > clean_counts[NO_CORRECTION][cell_id] for cell_id in example_counts)))
    if len(artifact_values) == 1:
        macros.set("FloatGateArtifact", artifact_values[0])
    elif artifact_values:
        macros.set("FloatGateArtifact", ", ".join(artifact_values))
    else:
        macros.miss("FloatGateArtifact", "no candidate is rejected by the float rule and accepted by the exact-count rule")
    macros.set("FloatGateRejectedCandidates", count(len(rejected)))
    macros.set("ForcedComparisonFailures", count(sum(not item["g3_4_all_strictly_positive"] for item in forced.values())))
    notes["clean_gate"] = {
        "written_rule": adjudication["written_rule"],
        "exact_count_rule": "macro drop <= 1.0 point and every cell drop <= 2.0 points with drops computed as 100*(correct_ref - correct_cand)/n using fractions.Fraction",
        "eligibility": {
            candidate: {
                "historical_float_rule": historical_eligible(candidate),
                "exact_count_rule": exact_eligible(candidate),
                "macro_drop_points": macro_drop(candidate),
                "max_cell_drop_points": max(clean_drop(candidate, cell_id) for cell_id in example_counts),
            }
            for candidate in candidate_ids
        },
        "float_gate_rejected_candidates": rejected,
        "gr_worst_cell": {"cell_id": worst_cell, "drop_points": gr_drops[worst_cell]},
    }

    # --------------------------------------------------------- failure anatomy
    macros.section("Failure anatomy for no correction at radius 0.25 (pooled over positions)")
    partition_macros = (
        ("PartAbstain", "abstained"),
        ("PartWrongClass", "nonabstaining_wrong_smoothed_class"),
        ("PartCorrectBelowRadius", "correct_smoothed_class_but_radius_below_target"),
        ("PartCertifiedCorrect", "certified_correct_at_target"),
        ("PartAnchorMismatch", "nonabstaining_anchor_mismatch"),
        ("PartAnchoredCorrect", "anchored_certified_correct_at_target"),
    )
    if partition is not None:
        # Cross-check against the audit's per-cell failure causes (same statistics, independent code).
        for candidate in (NO_CORRECTION, GR_CLIP):
            for cell_id, record in per_candidate_cells[candidate].items():
                causes = record["failure_causes_at_0.25"]
                computed = partition["cells"][cell_id]["candidates"][candidate]
                expected = {**causes["standard"], **causes["anchored"]}
                for key, value in expected.items():
                    if int(computed[key]) != int(value):
                        raise RuntimeError(f"partition differs from the audit for {candidate}/{cell_id}/{key}")
        pooled = partition["pooled"][NO_CORRECTION]
        if sum(pooled[key] for key in ("abstained", "nonabstaining_wrong_smoothed_class", "correct_smoothed_class_but_radius_below_target", "certified_correct_at_target")) != positions:
            raise RuntimeError("standard partition does not sum to the number of positions")
        for name, key in partition_macros:
            macros.set(name, count(pooled[key]))
        macros.set("CertifiedWrongRate", pct(pooled["certified_wrong_at_target"] / positions))
        macros.set("PartCertifiedWrong", count(pooled["certified_wrong_at_target"]))
        macros.set(
            "CertifiedCorrectAnchorMismatch",
            count(pooled["certified_correct_at_target"] - pooled["anchored_certified_correct_at_target"]),
        )
        top_shares: dict[str, float] = {}
        cells_without_certified_wrong: list[str] = []
        tie_cells: list[str] = []
        for cell_id in sorted(example_counts):
            entry = partition["cells"][cell_id]["candidates"][NO_CORRECTION]
            certified_wrong_count = int(entry["certified_wrong_at_target"])
            if certified_wrong_count == 0:
                cells_without_certified_wrong.append(cell_id)
                continue
            top_shares[cell_id] = int(entry["certified_wrong_top_class_count"]) / certified_wrong_count
            if int(entry["certified_wrong_top_class_ties"]) > 1:
                tie_cells.append(cell_id)
        if top_shares:
            macros.set("CWTopClassShareLo", one_decimal_pct(min(top_shares.values())))
            macros.set("CWTopClassShareHi", one_decimal_pct(max(top_shares.values())))
        else:
            macros.miss("CWTopClassShareLo", "no cell has a certified-wrong position at radius 0.25")
            macros.miss("CWTopClassShareHi", "no cell has a certified-wrong position at radius 0.25")
        macros.set("CWCellsWithNone", count(len(cells_without_certified_wrong)))
        notes["certified_wrong_top_class"] = {
            "definition": "per cell, share of no-correction certified-wrong positions at 0.25 whose selected class is the most frequent selected class among them; cells with zero such positions excluded",
            "per_cell_share": top_shares,
            "cells_without_certified_wrong": cells_without_certified_wrong,
            "cells_with_tied_top_class": tie_cells,
        }
        notes["failure_partition"] = {
            "mode": partition_mode,
            "source_files": {rel: sources.hashes[rel] for rel in pt_rels},
            "fields_accessed": partition["fields_accessed"],
            "definitions": {
                "abstained": "abstained flag",
                "nonabstaining_wrong_smoothed_class": "not abstained and selected smoothed class != ground truth",
                "correct_smoothed_class_but_radius_below_target": "not abstained, selected == truth, certificate radius < 0.25",
                "certified_correct_at_target": "not abstained, selected == truth, certificate radius >= 0.25",
                "certified_wrong_at_target": "not abstained, selected != truth, certificate radius >= 0.25",
                "nonabstaining_anchor_mismatch": "not abstained and selected smoothed class != raw clean prediction",
                "anchored_certified_correct_at_target": "certified correct at 0.25 and selected == raw clean prediction",
            },
            "cross_check": "per-cell counts equal the audit's failure_causes_at_0.25 for no_correction and the GR-CLIP candidate",
            "pooled": {candidate: partition["pooled"][candidate] for candidate in (NO_CORRECTION, GR_CLIP)},
        }
    else:
        for name, _ in partition_macros:
            macros.miss(name, f"per-example sufficient statistics unavailable: {partition_mode}")
        for name in ("CertifiedWrongRate", "PartCertifiedWrong", "CertifiedCorrectAnchorMismatch", "CWTopClassShareLo", "CWTopClassShareHi", "CWCellsWithNone"):
            macros.miss(name, f"per-example sufficient statistics unavailable: {partition_mode}")
        notes["failure_partition"] = {"mode": partition_mode, "source_files": pt_rels}

    # ------------------------------------ changed standard-CA outcomes per candidate
    macros.section("Positions whose standard-CA outcome at radius 0.25 differs from no correction")
    small_step_candidates = [
        candidate
        for candidate in candidate_ids
        if candidate_family(candidate) in ("clean_boundary_active", "cohen_aligned_noisy_margin")
        and candidate_step(candidate) is not None
        and candidate_step(candidate) <= 0.01 + 1e-12
    ]
    if len(small_step_candidates) != 6:
        raise RuntimeError(f"expected six proposal candidates with step <= 0.01, found {small_step_candidates}")
    if partition is not None:
        changed = {candidate: int(value) for candidate, value in partition["changed_standard_positions_vs_no_correction"].items()}
        macros.set("ChangedPositionsSmallSteps", count(max(changed[candidate] for candidate in small_step_candidates)))
        macros.set("ChangedPositionsExactStep", count(changed[exact_id]))
        macros.set("ChangedPositionsGR", count(changed[GR_CLIP]))
        proposal_candidates = [
            candidate
            for candidate in candidate_ids
            if candidate_family(candidate) in ("clean_boundary_active", "cohen_aligned_noisy_margin")
        ]
        if len(proposal_candidates) != 10:
            raise RuntimeError(f"expected ten text-only proposal candidates, found {proposal_candidates}")
        macros.set("ChangedPositionsProposalsMax", count(max(changed[candidate] for candidate in proposal_candidates)))
        notes["changed_positions"] = {
            "definition": "number of positions (of all cells) whose standard-CA indicator at radius 0.25 (not abstained, selected == truth, radius >= 0.25) differs from no correction",
            "small_step_candidates": {candidate: changed[candidate] for candidate in small_step_candidates},
            "exact_step_candidate": exact_id,
            "all_candidates": changed,
        }
    else:
        for name in ("ChangedPositionsSmallSteps", "ChangedPositionsExactStep", "ChangedPositionsGR", "ChangedPositionsProposalsMax"):
            macros.miss(name, f"per-example sufficient statistics unavailable: {partition_mode}")

    # --------------------------------------------------- no-correction ranges
    ranges = diagnostics["noise_collapse_no_correction_ranges"]
    recomputed: dict[str, list[float]] = {}
    for key in ranges:
        values = [float(cell_block[NO_CORRECTION][key]) for cell_block in diagnostics["cell_diagnostics"].values()]
        recomputed[key] = [min(values), max(values)]
        if not (math.isclose(recomputed[key][0], ranges[key][0]) and math.isclose(recomputed[key][1], ranges[key][1])):
            raise RuntimeError(f"no-correction range for {key} does not match cell_diagnostics")
    macros.section("No-correction per-cell ranges (one decimal)")
    for key, stem in (
        ("raw_clean_accuracy", "RangeClean"),
        ("smoothed_accuracy", "RangeSmoothed"),
        ("anchor_agreement_rate", "RangeAgreement"),
        ("operational_zero_radius_rate", "RangeZeroRadius"),
        ("top_smoothed_class_fraction", "RangeTopClass"),
    ):
        macros.set(f"{stem}Lo", one_decimal_pct(ranges[key][0]))
        macros.set(f"{stem}Hi", one_decimal_pct(ranges[key][1]))

    # ------------------------------------------------------ candidate movement
    def displacement(candidate: str) -> float:
        values = [float(cell.candidates[candidate]["prototype_bank_frobenius_displacement"]) for cell in cells]
        value = sum(values) / len(values)
        stored = float(selection["all_aggregates"][candidate]["macro_prototype_bank_frobenius_displacement"])
        if abs(value - stored) > 1e-9:
            raise RuntimeError(f"macro displacement differs from selection.json for {candidate}")
        return value

    macros.section("Candidate movement (macro-average Frobenius displacement)")
    macros.set("DispSaved", f"{displacement(saved_id):.3f}")
    macros.set("DispExact", f"{displacement(exact_id):.3f}")
    macros.set("DispGMCOne", f"{displacement(GMC_ONE):.3f}")
    macros.set("DispGR", f"{displacement(GR_CLIP):.3f}")

    # ------------------------------------------------- finite-sample censoring
    def censored(candidate: str, radius: float) -> int:
        stored = int(radius_value(aggregates[candidate]["finite_sample_plugin_only_examples"], radius))
        summed = sum(
            int(radius_value(record["finite_sample_censoring"], radius)["plugin_only_count"])
            for record in per_candidate_cells[candidate].values()
        )
        notes.setdefault("finite_sample_censoring_checks", []).append(
            {"candidate": candidate, "radius": radius, "aggregate": stored, "sum_of_cells": summed}
        )
        return stored

    macros.section("Finite-sample censoring (plug-in-only examples)")
    macros.set("CensoredNoCorrRQuarter", count(censored(NO_CORRECTION, TARGET_RADIUS)))
    macros.set("CensoredSavedRQuarter", count(censored(saved_id, TARGET_RADIUS)))
    macros.set("CensoredNoCorrRZero", count(censored(NO_CORRECTION, ZERO_RADIUS)))

    # ------------------------------------------------------------------ FARLA
    farla = sources.read_json(FARLA_REPORT_REL)
    farla_config = sources.read_json(FARLA_CONFIG_REL)
    farla_splits = sources.read_json(FARLA_SPLITS_REL)
    farla_rows = {row["candidate_id"]: row for row in farla["aggregate"]}
    if not math.isclose(float(farla["primary_radius"]), TARGET_RADIUS):
        raise RuntimeError("FARLA primary radius is not 0.25")

    macros.section("FARLA pilot (macro standard CA at 0.25 and macro clean accuracy)")
    for name, candidate in (
        ("FarlaIdentityStdCA", FARLA_IDENTITY),
        ("FarlaSharedStdCA", FARLA_SHARED),
        ("FarlaLowrankCEStdCA", FARLA_LOWRANK_CE),
        ("FarlaTailStdCA", FARLA_TAIL),
        ("FarlaFullStdCA", FARLA_FULL),
    ):
        if candidate in farla_rows:
            macros.set(name, pct(farla_rows[candidate]["macro_standard_ca_primary"]))
        else:
            macros.miss(name, f"candidate {candidate} absent from FARLA report aggregate")
    # the label-fitted shared translation against no correction (macro difference of the two aggregates; the pilot
    # registered no bootstrap for this pair), for the results-at-a-glance table (front-page review, 2026-09-28)
    if FARLA_SHARED in farla_rows and FARLA_IDENTITY in farla_rows:
        macros.set("FarlaSharedDelta", spct(float(farla_rows[FARLA_SHARED]["macro_standard_ca_primary"])
                                            - float(farla_rows[FARLA_IDENTITY]["macro_standard_ca_primary"])))
    else:
        macros.miss("FarlaSharedDelta", "the shared translation or no correction is absent from the FARLA report aggregate")
    for name, candidate in (
        ("FarlaIdentityClean", FARLA_IDENTITY),
        ("FarlaLowrankCEClean", FARLA_LOWRANK_CE),
        ("FarlaTailClean", FARLA_TAIL),
        ("FarlaFullClean", FARLA_FULL),
    ):
        if candidate in farla_rows:
            macros.set(name, pct(farla_rows[candidate]["macro_clean_accuracy"]))
        else:
            macros.miss(name, f"candidate {candidate} absent from FARLA report aggregate")
    for prefix, contrast_id, proposed, comparator in (
        ("FarlaVsNoCorr", FARLA_CONTRAST_NO_CORR, FARLA_FULL, FARLA_IDENTITY),
        ("FarlaVsTail", FARLA_CONTRAST_TAIL, FARLA_FULL, FARLA_TAIL),
    ):
        entry = farla["planned_contrasts"].get(contrast_id)
        if entry is None or entry["proposed"] != proposed or entry["comparator"] != comparator:
            for suffix in ("Delta", "Lower", "Upper"):
                macros.miss(f"{prefix}{suffix}", f"planned contrast {contrast_id} ({proposed} vs {comparator}) absent from FARLA report")
            continue
        bootstrap = entry["bootstrap"]
        macros.set(f"{prefix}Delta", spct(bootstrap["point_difference"]))
        macros.set(f"{prefix}Lower", spct(bootstrap["lower"]))
        macros.set(f"{prefix}Upper", spct(bootstrap["upper"]))
    for name, candidate in (
        ("FarlaIdentityCW", FARLA_IDENTITY),
        ("FarlaFullCW", FARLA_FULL),
        ("FarlaTailCW", FARLA_TAIL),
    ):
        if candidate in farla_rows:
            macros.set(name, pct(farla_rows[candidate]["macro_certified_wrong_primary"]))
        else:
            macros.miss(name, f"candidate {candidate} absent from FARLA report aggregate")
    if FARLA_PSEUDO in farla_rows:
        macros.set("FarlaPseudoStdCA", pct(farla_rows[FARLA_PSEUDO]["macro_standard_ca_primary"]))
        macros.set("FarlaPseudoClean", pct(farla_rows[FARLA_PSEUDO]["macro_clean_accuracy"]))
        macros.set("FarlaPseudoCleanDrop", pts(farla_rows[FARLA_PSEUDO]["macro_clean_drop_percentage_points"]))
    else:
        for name in ("FarlaPseudoStdCA", "FarlaPseudoClean", "FarlaPseudoCleanDrop"):
            macros.miss(name, f"candidate {FARLA_PSEUDO} absent from FARLA report aggregate")
    pseudo_pair = farla.get("paired_comparisons", {}).get(FARLA_PSEUDO)
    if pseudo_pair is not None and pseudo_pair["proposed"] == FARLA_PSEUDO and pseudo_pair["comparator"] == FARLA_IDENTITY:
        macros.set("FarlaPseudoVsNoCorrDelta", spct(pseudo_pair["bootstrap"]["point_difference"]))
        macros.set("FarlaPseudoVsNoCorrLower", spct(pseudo_pair["bootstrap"]["lower"]))
        macros.set("FarlaPseudoVsNoCorrUpper", spct(pseudo_pair["bootstrap"]["upper"]))
    else:
        for name in ("FarlaPseudoVsNoCorrDelta", "FarlaPseudoVsNoCorrLower", "FarlaPseudoVsNoCorrUpper"):
            macros.miss(name, f"paired comparison {FARLA_PSEUDO} vs {FARLA_IDENTITY} absent from FARLA report")
    farla_items: dict[str, set[int]] = {}
    for row in farla["cells"]:
        farla_items.setdefault(row["dataset_id"], set()).add(int(row["example_count"]))
    split_items = {entry["dataset_id"]: int(entry["confirmation_count"]) for entry in farla_splits["datasets"]}
    for name, dataset in (("FarlaCifarConfirmItems", "cifar100"), ("FarlaEurosatConfirmItems", "eurosat")):
        values = farla_items.get(dataset, set())
        if len(values) == 1 and split_items.get(dataset) == next(iter(values)):
            macros.set(name, count(next(iter(values))))
        else:
            macros.miss(name, f"confirmation item count for {dataset} is {sorted(values)} in report.json versus {split_items.get(dataset)} in splits/manifest.json")
    shots = {int(entry["train_per_class"]) for entry in farla_config["datasets"]}
    if len(shots) == 1:
        macros.set("FarlaShots", count(next(iter(shots))))
    else:
        macros.miss("FarlaShots", f"train_per_class differs across FARLA datasets: {sorted(shots)}")
    notes["farla"] = {
        "shots_source": f"{FARLA_CONFIG_REL} datasets[].train_per_class",
        "confirmation_items_source": f"{FARLA_REPORT_REL} cells[].example_count cross-checked with {FARLA_SPLITS_REL} confirmation_count",
        "decision": farla["decision"],
    }

    # ---------------------------------------------------- registered extension
    registrations: dict[str, dict[str, Any]] = {}
    registration_hashes: dict[str, str] = {}
    for key, rel in REGISTRATION_RELS.items():
        registrations[key] = sources.read_json(rel)
        sidecar = sources.read_text(rel.replace(".json", ".sha256")).split()[0].strip().lower()
        if sidecar != registrations[key]["registration_sha256"]:
            raise RuntimeError(f"registration sidecar disagrees with the embedded registration_sha256 for {rel}")
        registration_hashes[key] = sidecar
    exp017_sigma = float(summary["cohen_sigma"])

    def primary_cells(registration: dict[str, Any]) -> list[dict[str, Any]]:
        if not registration.get("bridge_cells"):
            return list(registration["cells"])
        return [
            cell
            for cell in registration["cells"]
            if not (cell["dataset_id"] == "cifar10" and math.isclose(float(cell["sigma"]), exp017_sigma))
        ]

    def bank_total(registration: dict[str, Any]) -> int:
        candidates = registration["candidates"]
        return int(candidates["frozen_grid"]["bank_count"]) + len(candidates["controls"])

    macros.section("Registered extension (configs/satml2027)")
    macros.set("ExtCellsA", count(len(registrations["A"]["cells"])))
    macros.set("ExtPrimaryCellsA", count(len(primary_cells(registrations["A"]))))
    macros.set("ExtCellsB", count(len(registrations["B"]["cells"])))
    macros.set("ExtCellsN", count(len(registrations["N"]["cells"])))
    bank_totals = {bank_total(registration) for registration in registrations.values()}
    if len(bank_totals) == 1:
        macros.set("ExtBanks", count(next(iter(bank_totals))))
    else:
        macros.miss("ExtBanks", f"bank totals differ across registrations: {sorted(bank_totals)}")
    evaluation_cells = registrations["A"]["cells"] + registrations["B"]["cells"]
    items_cifar = {int(cell["item_count"]) for cell in evaluation_cells if cell["dataset_id"].startswith("cifar")}
    items_eurosat = {int(cell["item_count"]) for cell in evaluation_cells if cell["dataset_id"] == "eurosat"}
    for name, values in (("ExtItemsCifar", items_cifar), ("ExtItemsEurosat", items_eurosat)):
        if len(values) == 1:
            macros.set(name, count(next(iter(values))))
        else:
            macros.miss(name, f"item counts differ across evaluation cells: {sorted(values)}")
    unique_items: dict[str, int] = {}
    for cell in evaluation_cells:
        digest = cell["evaluation_items_sha256"]
        if unique_items.setdefault(digest, int(cell["item_count"])) != int(cell["item_count"]):
            raise RuntimeError("evaluation item set has inconsistent item counts across cells")
    macros.set("ExtUniqueItems", count(sum(unique_items.values())))
    macros.set("ExtRegHashA", registration_hashes["A"][:12])
    macros.set("ExtRegHashB", registration_hashes["B"][:12])
    macros.set("ExtRegHashN", registration_hashes["N"][:12])
    notes["extension"] = {
        "primary_cells_rule": "cells of EXP-019A minus the CIFAR-10 cells at the EXP-017 sigma, which the registration labels a secondary bridge",
        "bank_total": "frozen_grid.bank_count + number of registered controls",
        "unique_items": "sum of item_count over distinct evaluation_items_sha256 values across EXP-019A and EXP-019B",
        "unique_item_sets": unique_items,
    }

    # ------------------------------------ EXP-021 follow-up design (D-145; registered before sampling)
    followup: dict[str, dict[str, Any]] = {}
    for key, rel in FU_REGISTRATION_RELS.items():
        followup[key] = sources.read_json(rel)
        sidecar = sources.read_text(rel.replace(".json", ".sha256")).split()[0].strip().lower()
        if sidecar != followup[key]["registration_sha256"]:
            raise RuntimeError(f"EXP-021 registration sidecar disagrees with the embedded registration_sha256 for {rel}")

    def fu_single(values: set, what: str) -> Any:
        if len(values) != 1:
            raise RuntimeError(f"EXP-021 registrations disagree on {what}: {sorted(values)}")
        return next(iter(values))

    def fu_unique_images(registration: dict[str, Any]) -> int:
        seen: dict[str, int] = {}
        for cell in registration["cells"]:
            if seen.setdefault(cell["evaluation_items_sha256"], int(cell["item_count"])) != int(cell["item_count"]):
                raise RuntimeError("an EXP-021 evaluation item set has inconsistent item counts across cells")
        return sum(seen.values())

    def fu_items(registration: dict[str, Any], dataset: str) -> int:
        return fu_single({int(cell["item_count"]) for cell in registration["cells"] if cell["dataset_id"] == dataset},
                         f"the item count of {dataset}")

    macros.section("EXP-021 follow-up design (configs/satml2027ext registrations; outcome-independent)")
    for key in ("A", "B", "C"):
        macros.set(f"FURegHash{key}", followup[key]["registration_sha256"][:12])
        macros.set(f"FUCells{key}", count(len(followup[key]["cells"])))
    macros.set("FURegisteredDate", fu_single({reg["registered_at"][:10] for reg in followup.values()}, "the registration date"))
    macros.set("FUImagesA", count(fu_unique_images(followup["A"])))
    macros.set("FUImagesB", count(fu_unique_images(followup["B"])))
    macros.set("FUItemsCifarB", count(fu_items(followup["B"], "cifar100_test")))
    macros.set("FUItemsEurosatB", count(fu_items(followup["B"], "eurosat_sealed")))
    macros.set("FUItemsImagenetteB", count(fu_items(followup["B"], "imagenette")))
    macros.set("FUItemsC", count(fu_single({int(cell["item_count"]) for cell in followup["C"]["cells"]}, "the 021C item count")))
    sigmas_b = sorted(float(value) for value in followup["B"]["sigmas"])
    if len(sigmas_b) != 2:
        raise RuntimeError(f"EXP-021B registers {sigmas_b}, not two noise levels")
    macros.set("FUSigmaLow", num(sigmas_b[0]))
    macros.set("FUSigmaHigh", num(sigmas_b[1]))
    macros.set("FUBackbonesA", count(len({cell["model_id"] for cell in followup["A"]["cells"]})))
    macros.set("FUBackbonesB", count(len({cell["model_id"] for cell in followup["B"]["cells"]})))
    macros.set("FUBanks", count(fu_single({int(followup[key]["candidates"]["expected_bank_count"]) for key in ("A", "B")}, "the bank count")))
    macros.set("FUBanksC", count(len(followup["C"]["candidates"]["candidate_ids"])))  # Q9 scores a registered subset
    # Q8 orientation: CLIP's channel standardization (published model constant, not a result), which sets how much
    # larger a pixel-space perturbation is in the standardized coordinates that published CLIP certificates use.
    macros.set("FUClipStdLo", f"{min(FU_CLIP_STD):.2f}")
    macros.set("FUClipStdHi", f"{max(FU_CLIP_STD):.2f}")
    macros.set("FUClipScaleLo", f"{1.0 / max(FU_CLIP_STD):.1f}")
    macros.set("FUClipScaleHi", f"{1.0 / min(FU_CLIP_STD):.1f}")
    fu_ids = fu_single({tuple(followup[key]["candidates"]["candidate_ids"]) for key in ("A", "B")}, "the candidate list")
    fu_families: dict[str, list[str]] = {}
    for candidate in fu_ids:
        fu_families.setdefault(candidate_family(candidate), []).append(candidate)
    macros.set("FUControls", count(len(fu_families.get("control", []))))
    macros.set("FUCoefficients", count(fu_single({len(fu_families[name]) for name in ("global_mean_centering", "chowers_exact_projected_gap", "gr_clip_style_two_sided", "text_only_centering", "image_only_centering")}, "the coefficient count")))
    macros.set("FUSteps", count(fu_single({len(fu_families[name]) for name in ("clean_boundary_active", "cohen_aligned_noisy_margin")}, "the step count")))
    macros.set("FUStepMax", num(max(candidate_step(candidate) for candidate in fu_families["clean_boundary_active"])))
    macros.set("FUConfDraws", count(fu_single({int(followup[key]["certification"]["confirmation_draws"]) for key in ("A", "B")}, "the confirmation budget")))
    macros.set("FUBudgetDraws", count(int(followup["C"]["certification"]["confirmation_draws"])))
    macros.set("FUReplicates", count(fu_single({int(followup[key]["inference"]["uncertainty"]["replicates"]) for key in ("A", "B")}, "the bootstrap replicates")))
    macros.set("FUSESOI", num(fu_single({float(followup[key]["inference"]["practical_magnitude"]["sesoi_points"]) for key in ("A", "B")}, "the SESOI")))
    pooled_power = followup["B"]["power"]["pooled_over_datasets_per_sigma_at_discordance_0.10"]
    macros.set("FUPooledMDE", num(float(pooled_power["minimum_detectable_points"])))
    macros.set("FUPooledPowerTwo", num(float(pooled_power["power_for_2_points"])))
    power_a = followup["A"]["power"]["at_discordance_0.10"]
    macros.set("FUMDEA", num(float(power_a["minimum_detectable_points"])))
    macros.set("FUPowerTwoA", num(float(power_a["power_for_2_points"])))
    macros.set("FUEurosatMDE", num(float(followup["B"]["power"]["per_dataset_sigma_at_discordance_0.10"]["eurosat_sealed"]["minimum_detectable_points"])))
    notes["exp021_design"] = {
        "source": "configs/satml2027ext/exp-20260921-021{a,b,c}.json, each checked against its .sha256 sidecar",
        "unique_images": "sum of item_count over distinct evaluation_items_sha256 values per registration",
        "power": "registered power block of EXP-021A (at discordance 0.10) and EXP-021B (pooled over datasets per sigma and EuroSAT sealed, at discordance 0.10)",
    }

    # ------------------------------------------ registered extension results
    receipt_fields: dict[str, str] = {}
    for line in sources.read_text(EXT_RECEIPT_REL).splitlines():
        if "=" in line:
            field, value = line.split("=", 1)
            receipt_fields[field.strip()] = value.strip()
    macros.section("Registered extension receipt, merge custody and control label budget")
    if "status" in receipt_fields:
        macros.set("ExtAReceiptStatus", tex_escape(receipt_fields["status"]))
    else:
        macros.miss("ExtAReceiptStatus", "receipt lacks a status field")
    if receipt_fields.get("result_file_count", "").isdigit():
        macros.set("ExtResultFiles", count(int(receipt_fields["result_file_count"])))
    else:
        macros.miss("ExtResultFiles", "receipt lacks result_file_count")
    if re.fullmatch(r"[0-9a-f]{64}", receipt_fields.get("result_manifest_sha256", "")):
        macros.set("ExtResultManifest", receipt_fields["result_manifest_sha256"][:12])
    else:
        macros.miss("ExtResultManifest", "receipt lacks a valid result_manifest_sha256")
    merged_cells = 0
    merge_custody: dict[str, int] = {}
    for experiment_id in (*EXT_EXPERIMENTS.values(), N3C_EXPERIMENT):
        report = sources.read_json(f"{EXT_MERGED_DIR_REL}/{experiment_id}/merge_report.json")
        if report.get("complete") is not True or report.get("missing") or report.get("experiment_id") != experiment_id:
            raise RuntimeError(f"merge report for {experiment_id} is not complete")
        if report.get("final_test_access") is not False:
            raise RuntimeError(f"merge report for {experiment_id} does not declare final_test_access false")
        merge_custody[experiment_id] = len(report["cells"])
        merged_cells += len(report["cells"])
    macros.set("ExtCellsMerged", count(merged_cells))
    item_manifest = sources.read_json(EXT_ITEM_MANIFEST_REL)
    labels_per_class: dict[str, int] = {}
    control_item_hashes: dict[str, str] = {}
    for dataset_id, spec in item_manifest["datasets"].items():
        split = spec["splits"]["control_train"]
        labels_per_class[dataset_id] = int(split["per_class"])
        control_item_hashes[dataset_id] = split["item_ids_sha256"]
    for key in ("A", "B"):
        for cell in registrations[key]["cells"]:
            if control_item_hashes.get(cell["dataset_id"]) != cell["control_train_items_sha256"]:
                raise RuntimeError(f"control-train item hash mismatch for {cell['cell_id']}")
    if len(set(labels_per_class.values())) == 1:
        macros.set("ExtControlLabelsPerClass", count(next(iter(labels_per_class.values()))))
    else:
        macros.miss("ExtControlLabelsPerClass", f"control-train labels per class differ across datasets: {labels_per_class}")
    regions = {ext_region_points(registrations[key]) for key in ("A", "B")}
    if len(regions) != 1:
        raise RuntimeError(f"Extensions A and B register different practical-effect regions: {sorted(regions)}")
    macros.set("ExtRegionPoints", f"{regions.pop():.1f}")
    notes["extension_custody"] = {
        "receipt": {field: receipt_fields.get(field) for field in ("status", "result_file_count", "result_manifest_sha256")},
        "merged_cells": merge_custody,
        "control_train_labels_per_class": labels_per_class,
        "unblinding": "D-127 owner-instructed unblinding; registered primary analysis outputs are primary-only",
    }

    extension_data: dict[str, ExtensionAnalysis | None] = {
        key: load_extension_analysis(sources, key, experiment_id) for key, experiment_id in EXT_EXPERIMENTS.items()
    }

    def ext_std(row: dict[str, Any]) -> float:
        return float(row[f"standard_certified_accuracy@{sigma_key(row['sigma'])}"])

    def ext_certified_wrong(row: dict[str, Any]) -> float:
        return float(row[f"certified_wrong_rate@{sigma_key(row['sigma'])}"])

    def ext_delta(analysis: ExtensionAnalysis, candidate: str) -> float:
        identity = analysis.rows[NO_CORRECTION]
        values = [ext_std(analysis.rows[candidate][cell_id]) - ext_std(identity[cell_id]) for cell_id in analysis.primary_cells]
        return 100.0 * sum(values) / len(values)

    def ext_delta_exact(analysis: ExtensionAnalysis, candidate: str) -> Fraction:
        """Exact equal-cell delta in points from integer per-cell counts (tie-safe ordering)."""
        identity = analysis.rows[NO_CORRECTION]
        total = Fraction(0)
        for cell_id in analysis.primary_cells:
            n = int(identity[cell_id]["example_count"])
            candidate_count = exact_count(ext_std(analysis.rows[candidate][cell_id]), n, what=f"{candidate}/{cell_id} standard CA")
            reference_count = exact_count(ext_std(identity[cell_id]), n, what=f"{NO_CORRECTION}/{cell_id} standard CA")
            total += Fraction(100 * (candidate_count - reference_count), n)
        return total / len(analysis.primary_cells)

    def ext_clean_delta(analysis: ExtensionAnalysis, candidate: str) -> float:
        identity = analysis.rows[NO_CORRECTION]
        values = [
            float(analysis.rows[candidate][cell_id]["clean_accuracy"]) - float(identity[cell_id]["clean_accuracy"])
            for cell_id in analysis.primary_cells
        ]
        return 100.0 * sum(values) / len(values)

    def ext_interval(analysis: ExtensionAnalysis, candidate: str) -> dict[str, float] | None:
        if analysis.contrasts is None:
            return None
        entry = analysis.contrasts["contrasts"][candidate]["simultaneous_confidence_interval"]
        critical_value = float(analysis.contrasts["simultaneous_band"]["critical_max_absolute_deviation"])
        point = float(entry["point_difference"])
        if abs(100.0 * point - ext_delta(analysis, candidate)) > 1e-7:
            raise RuntimeError(f"{analysis.experiment_id}: contrasts.json point differs from cells.csv for {candidate}")
        if not (
            math.isclose(entry["lower"], point - critical_value, abs_tol=1e-12)
            and math.isclose(entry["upper"], point + critical_value, abs_tol=1e-12)
        ):
            raise RuntimeError(f"{analysis.experiment_id}: band is not point +/- critical value for {candidate}")
        return {"point": point, "lower": float(entry["lower"]), "upper": float(entry["upper"])}

    def excludes_zero(entry: dict[str, float]) -> bool:
        return entry["lower"] > 0.0 or entry["upper"] < 0.0

    for key, analysis in extension_data.items():
        prefix = f"Ext{key}"
        experiment_id = EXT_EXPERIMENTS[key]
        macros.section(f"Registered extension {experiment_id}: contrasts at r = sigma versus no correction")
        if analysis is None:
            for stem in EXT_PER_EXPERIMENT_MACROS:
                macros.miss(f"{prefix}{stem}", f"registered analysis outputs for {experiment_id} are not written yet")
            continue
        pending = analysis.contrasts_reason

        def set_band(stem: str, candidate: str, *, analysis: ExtensionAnalysis = analysis, prefix: str = prefix) -> None:
            macros.set(f"{prefix}{stem}Delta", spts(ext_delta(analysis, candidate)))
            entry = ext_interval(analysis, candidate)
            if entry is None:
                macros.miss(f"{prefix}{stem}Lower", analysis.contrasts_reason)
                macros.miss(f"{prefix}{stem}Upper", analysis.contrasts_reason)
            else:
                macros.set(f"{prefix}{stem}Lower", spct(entry["lower"]))
                macros.set(f"{prefix}{stem}Upper", spct(entry["upper"]))

        if analysis.contrasts is not None:
            macros.set(f"{prefix}Critical", pts(100.0 * float(analysis.contrasts["simultaneous_band"]["critical_max_absolute_deviation"])))
        else:
            macros.miss(f"{prefix}Critical", pending)
        set_band("GR", GR_CLIP)
        macros.set(f"{prefix}GRCleanDelta", spts(ext_clean_delta(analysis, GR_CLIP)))
        set_band("Lowrank", EXT_LOWRANK)
        macros.set(f"{prefix}LowrankCleanDelta", spts(ext_clean_delta(analysis, EXT_LOWRANK)))
        macros.set(f"{prefix}NoisyMeanDelta", spts(ext_delta(analysis, EXT_NOISY_MEAN)))
        macros.set(f"{prefix}NoisyMeanCleanDelta", spts(ext_clean_delta(analysis, EXT_NOISY_MEAN)))
        set_band("SharedTangent", EXT_SHARED_TANGENT)
        set_band("SharedPure", EXT_SHARED_PURE)
        # the largest text-side translation (text mean-centering c=1), named in the extension text
        macros.set(f"{prefix}GMCOneDelta", spts(ext_delta(analysis, GMC_ONE)))
        macros.set(f"{prefix}GMCOneCleanDelta", spts(ext_clean_delta(analysis, GMC_ONE)))
        grid_deltas = {candidate: ext_delta(analysis, candidate) for candidate in analysis.grid}
        exact_grid = {candidate: ext_delta_exact(analysis, candidate) for candidate in analysis.grid}
        for candidate, exact_value in exact_grid.items():
            if abs(float(exact_value) - grid_deltas[candidate]) > 1e-7:
                raise RuntimeError(f"{experiment_id}: exact and float deltas disagree for {candidate}")
        best_value = max(exact_grid.values())
        best = sorted(candidate for candidate, value in exact_grid.items() if value == best_value)
        worst_value = min(exact_grid.values())
        macros.set(f"{prefix}MaxGridDelta", spts(float(best_value)))
        # readable bank names, not raw candidate ids (the PDF printed the ids; review of 2026-09-28)
        macros.set(f"{prefix}MaxGridId", " and ".join(ext_text_name(candidate) for candidate in best))
        macros.set(f"{prefix}MinGridDelta", spts(float(worst_value)))
        worst = sorted(candidate for candidate, value in exact_grid.items() if value == worst_value)
        macros.set(f"{prefix}MinGridId", " and ".join(ext_text_name(candidate) for candidate in worst))
        # registered prediction X1: every grid bank reported against the prospective practical-effect region
        region_points = ext_region_points(registrations[key])
        macros.set(f"{prefix}InsideRegion", count(sum(abs(float(value)) <= region_points for value in exact_grid.values())))
        if len(worst) == 1:
            macros.set(f"{prefix}MinGridCleanDelta", spts(ext_clean_delta(analysis, worst[0])))
        else:
            macros.miss(f"{prefix}MinGridCleanDelta", "several grid banks share the smallest change")
        if analysis.contrasts is not None:
            grid_exclusions = [candidate for candidate in analysis.grid if excludes_zero(ext_interval(analysis, candidate))]
            control_exclusions = [candidate for candidate in analysis.controls if excludes_zero(ext_interval(analysis, candidate))]
            macros.set(f"{prefix}GridExclZero", count(len(grid_exclusions)))
            macros.set(f"{prefix}GridExclZeroIds",
                       " and ".join(ext_text_name(candidate) for candidate in grid_exclusions) if grid_exclusions else "none")
            macros.set(f"{prefix}ControlsExclZero", count(len(control_exclusions)))
            tangent = ext_interval(analysis, EXT_SHARED_TANGENT)
            pure = ext_interval(analysis, EXT_SHARED_PURE)
            x3 = tangent["lower"] <= 0.0 <= tangent["upper"]
            per_class = {candidate: ext_interval(analysis, candidate) for candidate in EXT_PER_CLASS_CONTROLS}
            x4 = ext_x4_confirmed(per_class)  # addendum A2.3: at least one per-class control above zero
            macros.set(f"{prefix}Xthree", "confirmed" if x3 else "not confirmed")
            macros.set(f"{prefix}Xfour", "confirmed" if x4 else "not confirmed")
            # both A2.3 flags are reported (descriptively, beside the registered existential verdict)
            macros.set(f"{prefix}XfourPositiveControls",
                       count_word(sum(float(interval["lower"]) > 0.0 for interval in per_class.values())))
        else:
            grid_exclusions, control_exclusions, pure = [], [], None
            for stem in ("GridExclZero", "GridExclZeroIds", "ControlsExclZero", "Xthree", "Xfour", "XfourPositiveControls"):
                macros.miss(f"{prefix}{stem}", pending)
        # X5 as registered (addendum A1): the identity bank's top-class share and predicted-class Gini both increase
        # with sigma in every model x dataset; descriptive only, so "supported" / "not uniformly supported".
        share_groups: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
        for cell_id, row in analysis.rows[NO_CORRECTION].items():
            entry = analysis.concentration[NO_CORRECTION][cell_id]
            share_groups.setdefault((str(row["model_id"]), str(row["dataset_id"])), []).append(
                (float(row["sigma"]), float(entry["top_class_share"]), float(entry["predicted_class_gini"])))
        x5_groups = {}
        for group, values in share_groups.items():
            ordered = sorted(values)
            x5_groups["/".join(group)] = {
                "sigma_to_share": {num(sigma): share for sigma, share, _ in ordered},
                "sigma_to_gini": {num(sigma): gini for sigma, _, gini in ordered},
                "strictly_increasing": x5_group_supported(values),
            }
        x5 = all(entry["strictly_increasing"] for entry in x5_groups.values())
        macros.set(f"{prefix}Xfive", "supported" if x5 else "not uniformly supported")
        notes[f"extension_{experiment_id}"] = {
            "primary_cells": analysis.primary_cells,
            "grid_banks": analysis.grid,
            "controls": analysis.controls,
            "delta_definition": "equal-cell mean over the primary cells of (candidate - no_correction) standard CA at r = sigma, points",
            "clean_delta_definition": "equal-cell mean over the primary cells of (candidate - no_correction) clean accuracy, points",
            "contrasts_available": analysis.contrasts is not None,
            "grid_deltas_points": grid_deltas,
            "grid_deltas_exact_points": exact_grid,
            "largest_grid_delta_candidates": best,
            "control_deltas_points": {candidate: ext_delta(analysis, candidate) for candidate in analysis.controls},
            "grid_band_excludes_zero": grid_exclusions,
            "control_band_excludes_zero": control_exclusions,
            "x3_rule": "the learned shared-translation (tangent) control interval contains zero",
            "x3_pure_control_interval_contains_zero": None if pure is None else bool(pure["lower"] <= 0.0 <= pure["upper"]),
            "x4_rule": "addendum A2.3: at least one per-class control (noisy class mean, rank-8 tangent adapter) has a lower band limit above zero",
            "x5_rule": "addendum A1: identity-bank top-class share and predicted-class Gini both strictly increase with sigma within every model x dataset (bridge cells included); 'supported' or 'not uniformly supported', no inferential statement",
            "x5_groups": x5_groups,
        }

    macros.section("Registered extension prediction summaries (one phrase per prediction over both experiments)")
    for stem, positive, negative, phrase in (("Xthree", "confirmed", "not confirmed", prediction_summary),
                                             ("Xfour", "confirmed", "not confirmed", prediction_summary),
                                             ("Xfive", "supported", "not uniformly supported", x5_summary)):
        verdicts = [macros.values.get(f"Ext{key}{stem}") for key in ("A", "B")]
        if any(value not in (positive, negative) for value in verdicts):
            macros.miss(f"Ext{stem}Summary", "a per-experiment verdict is missing")
            continue
        macros.set(f"Ext{stem}Summary", phrase(verdicts[0] == positive, verdicts[1] == positive))
    # the A2.3 flags beside the existential X4 verdict, as one phrase for the text
    flags = [macros.values.get(f"Ext{key}XfourPositiveControls") for key in ("A", "B")]
    if any(flag not in ("zero", "one", "two") for flag in flags):
        macros.miss("ExtXfourFlags", "a per-experiment X4 flag count is missing")
    elif flags == ["two", "two"]:
        macros.set("ExtXfourFlags", "both per-class controls do so in both experiments")
    else:
        macros.set("ExtXfourFlags", f"{flags[0]} of the two per-class controls do so in Extension A and {flags[1]} in Extension B")

    macros.section("Registered extension bridge cells (CIFAR-10 at the EXP-017 sigma) and identity regime ranges")
    analysis_a = extension_data["A"]
    for model_id, token in EXT_MODEL_MACRO_TOKEN.items():
        name = f"ExtBridgeGRDelta{token}"
        if analysis_a is None:
            macros.miss(name, f"registered analysis outputs for {EXT_EXPERIMENTS['A']} are not written yet")
            continue
        bridge = [
            cell_id
            for cell_id, row in analysis_a.rows[NO_CORRECTION].items()
            if not row["primary_estimand_cell"] and str(row["model_id"]) == model_id
        ]
        if len(bridge) != 1:
            macros.miss(name, f"expected one bridge cell for {model_id}, found {bridge}")
            continue
        cell_id = bridge[0]
        macros.set(name, spts(100.0 * (ext_std(analysis_a.rows[GR_CLIP][cell_id]) - ext_std(analysis_a.rows[NO_CORRECTION][cell_id]))))
    identity_rows = [
        row
        for analysis in extension_data.values()
        if analysis is not None
        for row in analysis.rows[NO_CORRECTION].values()
    ]
    regime_notes: dict[str, Any] = {}
    for sigma_value, token in EXT_SIGMA_TOKENS:
        at_sigma = [row for row in identity_rows if math.isclose(float(row["sigma"]), sigma_value)]
        cifar = [row for row in at_sigma if str(row["dataset_id"]).startswith("cifar")]
        eurosat = [row for row in at_sigma if str(row["dataset_id"]) == "eurosat"]
        for stem, subset, extract in (
            (f"ExtSmoothed{token}Cifar", cifar, lambda row: float(row["smoothed_accuracy"])),
            (f"ExtStdCA{token}Cifar", cifar, ext_std),
            (f"ExtSmoothed{token}Eurosat", eurosat, lambda row: float(row["smoothed_accuracy"])),
        ):
            if subset:
                values = [extract(row) for row in subset]
                macros.set(f"{stem}Lo", pct(min(values)))
                macros.set(f"{stem}Hi", pct(max(values)))
            else:
                macros.miss(f"{stem}Lo", f"no identity cells at sigma {sigma_value} are available yet")
                macros.miss(f"{stem}Hi", f"no identity cells at sigma {sigma_value} are available yet")
        regime_notes[num(sigma_value)] = {
            "cifar_cells": sorted(str(row["cell_id"]) for row in cifar),
            "eurosat_cells": sorted(str(row["cell_id"]) for row in eurosat),
        }
    notes["extension_regime_ranges"] = {
        "definition": "min and max over the identity-bank cells at the given sigma across the available registered experiments",
        "cells": regime_notes,
    }

    # ---------------------------------------------------------------- N3C
    macros.section("N3C fresh-noise diagnostic predictions")
    n3c_rel = f"{EXT_ANALYSIS_DIR_REL}/{N3C_EXPERIMENT}/n3c_predictions.json"
    n3c = sources.read_json(n3c_rel) if (root / n3c_rel).is_file() else None
    n3c_summary: dict[str, Any] = {}
    if n3c is None:
        for name in N3C_MACROS:
            macros.miss(name, f"{n3c_rel} has not been written yet")
    else:
        if n3c.get("registration_sha256") != registration_hashes["N"]:
            raise RuntimeError("N3C predictions are bound to a different registration")
        n3c_cells = n3c["cells"]
        if set(n3c_cells) != {cell["cell_id"] for cell in registrations["N"]["cells"]}:
            raise RuntimeError("N3C prediction cells differ from the registered cells")
        n3c_rules = {prediction["id"]: str(prediction.get("rule", "")) for prediction in registrations["N"]["predictions"]}
        p1_exceptions = 0
        p1_applicable: set[str] = set()
        p1_per_candidate: dict[str, int] = {}
        for cell in n3c_cells.values():
            for candidate, entry in cell["P1"].items():
                if entry.get("applicable"):
                    p1_exceptions += int(entry["flips_outside_eligible_set"])
                    p1_applicable.add(candidate)
                    p1_per_candidate[candidate] = p1_per_candidate.get(candidate, 0) + int(entry["flips_outside_eligible_set"])
        macros.set("NthreeCPone", count(p1_exceptions))
        # P2 (frozen measurement, no threshold): orthogonality residual, normalizer mismatch and P1 conformance of every
        # float-level flip of the orthogonal-complement (projected-gap) banks, over all cells (review 2026-09-28)
        p2_entries = [entry for cell in n3c_cells.values() for entry in cell["P2"].values()]
        if not p2_entries:
            raise RuntimeError("N3C P2 lists no orthogonal-complement candidate")
        macros.set("NthreeCPtwoResidual", fu_bound(max(float(e["centered_span_orthogonality_residual_max"]) for e in p2_entries)))
        macros.set("NthreeCPtwoMismatch", fu_bound(max(float(e["normalizer_mismatch_max"]) for e in p2_entries)))
        macros.set("NthreeCPtwoFlips", count(sum(int(e["p1_flips"]) for e in p2_entries)))
        macros.set("NthreeCPtwoExceptions", count(sum(int(e["p1_exceptions"]) for e in p2_entries)))
        p4_match = re.search(r"<=\s*([0-9.]+)", n3c_rules.get("P4a", ""))
        n3c_cifar_cells = sorted(cell_id for cell_id in n3c_cells if "__cifar100__" in cell_id)
        if p4_match is None or len(n3c_cifar_cells) != 2:
            for name in ("NthreeCPfourA", "NthreeCPfourAMax", "NthreeCPfourAVerdict"):
                macros.miss(name, "P4a rule threshold or the two CIFAR-100 cells could not be identified")
            p4_summary: dict[str, Any] = {}
        else:
            p4_threshold = float(p4_match.group(1))
            n3c_mace = {cell_id: float(n3c_cells[cell_id]["P4a"]["mean_absolute_calibration_error"]) for cell_id in n3c_cifar_cells}
            p4_pass = all(value <= p4_threshold for value in n3c_mace.values())
            macros.set("NthreeCPfourAMax", f"{max(n3c_mace.values()):.4f}")
            macros.set("NthreeCPfourAVerdict", "pass" if p4_pass else "fail")
            macros.set("NthreeCPfourA", f"{max(n3c_mace.values()):.4f} ({'pass' if p4_pass else 'fail'})")
            p4_summary = {"threshold": p4_threshold, "mean_absolute_calibration_error": n3c_mace, "pass": p4_pass}
        p5_match = re.search(r"at least (\d+) of (\d+)", n3c_rules.get("P5", ""))
        n3c_rhos = {cell_id: float(cell["P5"]["spearman_rho"]) for cell_id, cell in n3c_cells.items()}
        n3c_positive = sum(value > 0.0 for value in n3c_rhos.values())
        macros.set("NthreeCPfiveRhoLo", f"{min(n3c_rhos.values()):+.2f}")
        macros.set("NthreeCPfiveRhoHi", f"{max(n3c_rhos.values()):+.2f}")
        if p5_match is None or int(p5_match.group(2)) != len(n3c_cells):
            for name in ("NthreeCPfive", "NthreeCPfivePositive", "NthreeCPfiveVerdict"):
                macros.miss(name, "P5 rule could not be parsed from the registration")
            p5_summary: dict[str, Any] = {}
        else:
            p5_pass = n3c_positive >= int(p5_match.group(1))
            macros.set("NthreeCPfivePositive", count(n3c_positive))
            macros.set("NthreeCPfiveVerdict", "pass" if p5_pass else "fail")
            macros.set("NthreeCPfive", f"{n3c_positive} of {len(n3c_cells)} ({'pass' if p5_pass else 'fail'})")
            p5_summary = {"required": int(p5_match.group(1)), "spearman_rho": n3c_rhos, "positive": n3c_positive, "pass": p5_pass}
        n3c_positions = sum(int(cell["items"]) for cell in n3c_cells.values())
        n3c_at_threshold = sum(int(cell["P6_items_already_at_threshold"]) for cell in n3c_cells.values())
        p7_pooled: dict[str, dict[str, int]] = {}
        for candidate in sorted(next(iter(n3c_cells.values()))["P7"]):
            entries = [cell["P7"][candidate] for cell in n3c_cells.values()]
            if not all(entry.get("applicable") for entry in entries):
                continue
            pooled_entry = {
                field: sum(int(entry[field]) for entry in entries)
                for field in ("already_at_threshold", "threshold_unattainable_on_these_draws", "undetermined", "realized_at_threshold_after", "bound_violations")
            }
            if pooled_entry["already_at_threshold"] != n3c_at_threshold:
                raise RuntimeError(f"P7 already_at_threshold differs from P6 for {candidate}")
            if pooled_entry["already_at_threshold"] + pooled_entry["threshold_unattainable_on_these_draws"] + pooled_entry["undetermined"] != n3c_positions:
                raise RuntimeError(f"P7 partition does not sum to the item count for {candidate}")
            p7_pooled[candidate] = pooled_entry
        if p7_pooled:
            unattainable = [entry["threshold_unattainable_on_these_draws"] for entry in p7_pooled.values()]
            undetermined = [entry["undetermined"] for entry in p7_pooled.values()]
            macros.set("NthreeCPseven", count(n3c_at_threshold))
            macros.set("NthreeCPsevenPositions", count(n3c_positions))
            macros.set("NthreeCPsevenCandidates", count(len(p7_pooled)))
            macros.set("NthreeCPsevenUnattainableLo", count(min(unattainable)))
            macros.set("NthreeCPsevenUnattainableHi", count(max(unattainable)))
            macros.set("NthreeCPsevenUndeterminedLo", count(min(undetermined)))
            macros.set("NthreeCPsevenUndeterminedHi", count(max(undetermined)))
        else:
            for name in ("NthreeCPseven", "NthreeCPsevenPositions", "NthreeCPsevenCandidates", "NthreeCPsevenUnattainableLo", "NthreeCPsevenUnattainableHi", "NthreeCPsevenUndeterminedLo", "NthreeCPsevenUndeterminedHi"):
                macros.miss(name, "no candidate has an applicable P7 partition in every cell")
        p9_pooled: dict[str, dict[str, int]] = {}
        for candidate in sorted(next(iter(n3c_cells.values()))["P9"]):
            p9_pooled[candidate] = {
                field: sum(int(cell["P9"][candidate][field]) for cell in n3c_cells.values())
                for field in ("unchanged", "useful_wrong_to_correct", "harmful_correct_to_wrong", "wrong_to_wrong", "total_flips")
            }
        if GR_CLIP in p9_pooled and EXT_LOWRANK in p9_pooled:
            two_sided = p9_pooled[GR_CLIP]
            lowrank = p9_pooled[EXT_LOWRANK]
            macros.set("NthreeCPnineTwoSidedUseful", count(two_sided["useful_wrong_to_correct"]))
            macros.set("NthreeCPnineTwoSidedHarmful", count(two_sided["harmful_correct_to_wrong"]))
            macros.set("NthreeCPnineLowrankUseful", count(lowrank["useful_wrong_to_correct"]))
            macros.set("NthreeCPnineLowrankHarmful", count(lowrank["harmful_correct_to_wrong"]))
            macros.set(
                "NthreeCPnine",
                f"two-sided c=1: {count(two_sided['useful_wrong_to_correct'])} useful / {count(two_sided['harmful_correct_to_wrong'])} harmful; "
                f"rank-8 adapter: {count(lowrank['useful_wrong_to_correct'])} useful / {count(lowrank['harmful_correct_to_wrong'])} harmful",
            )
        else:
            for name in ("NthreeCPnine", "NthreeCPnineTwoSidedUseful", "NthreeCPnineTwoSidedHarmful", "NthreeCPnineLowrankUseful", "NthreeCPnineLowrankHarmful"):
                macros.miss(name, "P9 entries for the two-sided bank or the rank-8 control are absent")
        n3c_summary = {
            "k_min": n3c["k_min"],
            "draws_per_item": sorted({int(cell["draws"]) for cell in n3c_cells.values()}),
            "positions": n3c_positions,
            "p1": {"exceptions_total": p1_exceptions, "applicable_candidates": sorted(p1_applicable), "per_candidate": p1_per_candidate},
            "p4a": p4_summary,
            "p5": p5_summary,
            "p7_pooled_over_cells": p7_pooled,
            "p7_items_already_at_threshold": n3c_at_threshold,
            "p9_pooled_over_cells": p9_pooled,
        }
        notes["n3c"] = n3c_summary

    # ---------------------------------------------------------------- tables
    tables: dict[str, str] = {}

    # Family table.
    family_rows: list[str] = []
    for candidate in candidate_ids:
        name = candidate_display(candidate)
        if candidate == saved_id:
            name += r"$^{\dagger}$"
        if candidate == exact_id:
            name += r"$^{\ddagger}$"
        if candidate == NO_CORRECTION:
            delta_cell, band_cell = "reference", "--"
        else:
            row = contrast(candidate, STANDARD_METRIC, TARGET_RADIUS)
            delta_cell = _math(spct(row["point_estimate"]))
            band_cell = f"[{_math(spct(row['simultaneous_lower']))}, {_math(spct(row['simultaneous_upper']))}]"
        family_rows.append(
            " & ".join(
                [
                    name,
                    pct(scalar(candidate, "raw_clean_accuracy")),
                    pct(std_ca(candidate)),
                    pct(anch_ca(candidate)),
                    delta_cell,
                    band_cell,
                    "yes" if historical_eligible(candidate) else "no",
                    "yes" if exact_eligible(candidate) else "no",
                ]
            )
            + r" \\"
        )
    family_sources = [AUDIT_REL, GATE_N2_REL, SELECTION_REL, DIAGNOSTICS_REL, *cell_rels]
    family_caption = (
        "Complete 18-candidate family at pixel-space $\\ell_2$ radius 0.25 (equal-cell macro over "
        f"{count(len(cells))} cells, {count(positions)} positions; accuracies in percent). Standard CA is standard Cohen certified "
        "accuracy; anchored CA additionally requires the selected smoothed class to equal the raw-clean prediction. "
        f"The standard delta versus no correction carries the complete-family simultaneous 95\\% band (common half width "
        f"{macros.values['BandHalfWidth']} points, {macros.values['BandReplicates']} replicates, "
        f"{macros.values['BandContrasts']} contrasts). Float gate: historical eligibility flag as saved; exact gate: "
        "the written clean-accuracy rule recomputed with integer counts. $\\dagger$ saved selection; "
        "$\\ddagger$ exact-count selection."
    )  # sources and their hashes are in the table's source comment, not the caption (review 2026-09-28)
    tables["table_family_r025.tex"] = "\n".join(
        _source_comment(sources, family_sources, "Complete candidate family at radius 0.25.")
        + _table(
            family_caption,
            "tab:family",
            _tabular(
                "lrrrrlcc",
                [
                    "Candidate",
                    "Clean acc.",
                    "Std.\\ CA",
                    "Anch.\\ CA",
                    "$\\Delta$ std.\\ (points)",
                    "95\\% simultaneous band",
                    "Eligible (float rule)",
                    "Eligible (exact rule)",
                ],
                family_rows,
            ),
            star=True,
        )
        + [""]
    )

    # Partition table.
    partition_rows: list[str] = []
    partition_layout = (
        ("standard", "Standard metric", (
            ("Abstained", "abstained"),
            ("Nonabstaining, wrong smoothed class", "nonabstaining_wrong_smoothed_class"),
            ("\\quad of which certified wrong at 0.25", "certified_wrong_at_target"),
            ("Correct smoothed class, radius $<0.25$", "correct_smoothed_class_but_radius_below_target"),
            ("Certified correct at 0.25", "certified_correct_at_target"),
        )),
        ("anchored", "Anchored metric", (
            ("Abstained", "abstained"),
            ("Nonabstaining anchor mismatch", "nonabstaining_anchor_mismatch"),
            ("Anchor preserved, raw class wrong", "anchor_preserved_but_raw_class_wrong"),
            ("Anchor correct, radius $<0.25$", "anchor_correct_but_radius_below_target"),
            ("Anchored certified correct at 0.25", "anchored_certified_correct_at_target"),
        )),
    )
    for index, (_, title, entries) in enumerate(partition_layout):
        if index:
            partition_rows.append(r"\midrule")
        partition_rows.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{title}}}}} \\\\")
        for label, key in entries:
            if partition is None:
                cells_text = [Macros.PLACEHOLDER] * 4
            else:
                cells_text = []
                for candidate in (NO_CORRECTION, GR_CLIP):
                    value = int(partition["pooled"][candidate][key])
                    cells_text.extend([count(value), pct(value / positions)])
            partition_rows.append(" & ".join([label, *cells_text]) + r" \\")
    partition_sources = ([*pt_rels] if partition is not None else []) + [AUDIT_REL]
    partition_caption = (
        f"Failure partition at radius 0.25 pooled over {count(positions)} positions for no correction and the "
        "GR-CLIP-style two-sided candidate, under the standard metric and the anchored metric. Counts come from the "
        "audit's saved per-example sufficient statistics and reproduce its per-cell failure "
        "categories."
        + ("" if partition is not None else f" Per-example data unavailable: {tex_escape(partition_mode)}.")
    )
    tables["table_partition.tex"] = "\n".join(
        _source_comment(sources, partition_sources, "Failure partition at radius 0.25 (no correction and GR-CLIP).")
        + _table(
            partition_caption,
            "tab:partition",
            _tabular(
                "lrrrr",
                ["Outcome at $r=0.25$", "No corr.\\ $n$", "\\%", "GR-CLIP $n$", "\\%"],
                partition_rows,
            ),
            colsep="3pt",
        )
        + [""]
    )

    # GR-CLIP per-cell table.
    # Per-cell standard and anchored CA come from the audit's per_candidate_cells; the
    # cell_metrics.json "certified_accuracy" field is the historical anchored metric and
    # must never be labeled standard CA.
    grclip_rows: list[str] = []
    for cell in sorted(cells, key=lambda item: (item.model_id, item.dataset_id, item.fold)):
        nc = cell.candidates[NO_CORRECTION]
        gr = cell.candidates[GR_CLIP]
        nc_audit = per_candidate_cells[NO_CORRECTION][cell.cell_id]
        gr_audit = per_candidate_cells[GR_CLIP][cell.cell_id]
        grclip_rows.append(
            " & ".join(
                [
                    model_display(cell.model_id),
                    dataset_display(cell.dataset_id),
                    str(cell.fold),
                    pct(nc["raw_clean_accuracy"]),
                    pct(gr["raw_clean_accuracy"]),
                    _math(spts(float(gr_drops[cell.cell_id]))),
                    pct(radius_value(nc_audit["standard_certified_accuracy"], TARGET_RADIUS)),
                    pct(radius_value(gr_audit["standard_certified_accuracy"], TARGET_RADIUS)),
                    pct(radius_value(nc_audit["anchored_certified_accuracy"], TARGET_RADIUS)),
                    pct(radius_value(gr_audit["anchored_certified_accuracy"], TARGET_RADIUS)),
                ]
            )
            + r" \\"
        )
    grclip_sources = [AUDIT_REL, *cell_rels, SELECTION_REL, DIAGNOSTICS_REL]
    grclip_caption = (
        "Per-cell clean accuracy, standard certified accuracy (Std CA) and anchored certified accuracy (Anch.\\ CA) "
        "at radius 0.25 for no correction and the GR-CLIP-style two-sided candidate (coefficient 1). Drop is the "
        "clean-accuracy decrease in points computed from integer counts (negative values are improvements); the "
        f"worst cell is {macros.values['GRWorstCell']} at {macros.values['GRWorstCellDrop']} points "
        f"({macros.values['GRWorstCellImages']} images) and {macros.values['GRCellsImproved']} of "
        f"{count(len(cells))} cells improve. Column means equal the equal-cell macros "
        f"({macros.values['NoCorrStdCA']}, {macros.values['GRStdCA']}, {macros.values['NoCorrAnchCA']}, "
        f"{macros.values['GRAnchCA']})."
    )
    tables["table_grclip_cells.tex"] = "\n".join(
        _source_comment(sources, grclip_sources, "GR-CLIP versus no correction per cell.")
        + _table(
            grclip_caption,
            "tab:grclip-cells",
            _tabular(
                "lllrrrrrrr",
                [
                    "Model",
                    "Dataset",
                    "Fold",
                    "No corr.\\ clean",
                    "GR-CLIP clean",
                    "Drop (pts)",
                    "No corr.\\ Std CA",
                    "GR-CLIP Std CA",
                    "No corr.\\ Anch.\\ CA",
                    "GR-CLIP Anch.\\ CA",
                ],
                grclip_rows,
            ),
            star=True,
            colsep="3pt",
        )
        + [""]
    )

    # FARLA table.
    farla_rows_tex: list[str] = []
    for row in farla["aggregate"]:
        farla_rows_tex.append(
            " & ".join(
                [
                    FARLA_DISPLAY.get(row["candidate_id"], tex_escape(row["candidate_id"])),
                    pct(row["macro_standard_ca_primary"]),
                    pct(row["macro_anchored_ca_primary"]),
                    pct(row["macro_clean_accuracy"]),
                    pct(row["macro_certified_wrong_primary"]),
                    "pass" if row["clean_constraint_satisfied"] else "fail",
                ]
            )
            + r" \\"
        )
    contrast_rows_tex: list[str] = []
    # every planned contrast of the pilot's report, the two discussed in the text first
    farla_contrast_ids = [FARLA_CONTRAST_NO_CORR, FARLA_CONTRAST_TAIL] + sorted(
        set(farla["planned_contrasts"]) - {FARLA_CONTRAST_NO_CORR, FARLA_CONTRAST_TAIL}
    )
    for contrast_id in farla_contrast_ids:
        entry = farla["planned_contrasts"].get(contrast_id)
        if entry is None:
            contrast_rows_tex.append(f"{tex_escape(contrast_id)} & {Macros.PLACEHOLDER} & {Macros.PLACEHOLDER} & {Macros.PLACEHOLDER} \\\\")
            continue
        bootstrap = entry["bootstrap"]
        comparator = FARLA_DISPLAY.get(entry["comparator"], tex_escape(entry["comparator"]))
        if len(comparator) > 1 and comparator[1].islower():
            comparator = comparator[0].lower() + comparator[1:]  # mid-label lowercase unless an acronym
        label = f"{FARLA_DISPLAY.get(entry['proposed'], tex_escape(entry['proposed']))} vs.\\ {comparator}"
        contrast_rows_tex.append(
            " & ".join(
                [
                    label,
                    _math(spct(bootstrap["point_difference"])),
                    _math(spct(bootstrap["lower"])),
                    _math(spct(bootstrap["upper"])),
                ]
            )
            + r" \\"
        )
    farla_sources = [FARLA_REPORT_REL, FARLA_CONFIG_REL, FARLA_SPLITS_REL]
    farla_caption = (
        "Positive-control pilot: macro standard CA, anchored CA and certified-wrong rate "
        f"at radius 0.25 and macro clean accuracy over {count(farla_rows[FARLA_IDENTITY]['cell_count'])} cells "
        f"({macros.values['FarlaCifarConfirmItems']} CIFAR-100 and {macros.values['FarlaEurosatConfirmItems']} EuroSAT "
        f"confirmation items per cell, {macros.values['FarlaShots']} supervised shots per class), with all "
        f"{count_word(len(farla_contrast_ids))} planned "
        "paired contrasts and their class-stratified percentile bootstrap 95\\% intervals "
        f"({count(farla['planned_contrasts'][FARLA_CONTRAST_NO_CORR]['bootstrap']['replicates'])} replicates). "
        "Clean gate: macro drop at most 1 point and every cell drop at most 2 points."
    )
    farla_lines = _source_comment(sources, farla_sources, "Positive-control pilot candidates and planned contrasts.")
    farla_lines += [r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}", f"\\caption{{{farla_caption}}}", r"\label{tab:farla}"]
    farla_lines += _tabular(
        "lrrrrc", ["Candidate", "Std CA", "Anch.\\ CA", "Clean", "Cert.-wrong", "Clean gate"], farla_rows_tex
    )
    farla_lines += [r"\par\vspace{4pt}"]
    farla_lines += _tabular("lrrr", ["Planned contrast (Std CA)", "$\\Delta$", "95\\% lower", "95\\% upper"], contrast_rows_tex)
    farla_lines += [r"\end{table*}", ""]
    tables["table_farla.tex"] = "\n".join(farla_lines)

    # Extension table.
    extension_rows: list[str] = []
    for key, registration in registrations.items():
        cells_reg = registration["cells"]
        primary = primary_cells(registration)
        items_by_count: dict[int, list[str]] = {}
        for cell in cells_reg:
            names = items_by_count.setdefault(int(cell["item_count"]), [])
            display = dataset_display(cell["dataset_id"])
            if display not in names:
                names.append(display)
        items_text = "; ".join(f"{count(n)} ({', '.join(names)})" for n, names in sorted(items_by_count.items(), reverse=True))
        if registration.get("sigmas"):
            sigmas_text = ", ".join(num(value) for value in registration["sigmas"])
        else:
            sigmas_text = num(registration["sampling"]["sigma"])
        cells_text = count(len(cells_reg))
        if len(primary) != len(cells_reg):
            cells_text += f" ({count(len(primary))} primary)"
        candidates = registration["candidates"]
        banks_text = f"{count(bank_total(registration))} ({count(candidates['frozen_grid']['bank_count'])} + {count(len(candidates['controls']))})"
        if registration.get("primary_outcome"):
            estimand = tex_escape(str(registration["primary_outcome"]).split(";")[0].strip())
        else:
            predictions = registration.get("predictions", [])
            ids = [str(item["id"]) for item in predictions]
            estimand = tex_escape(
                f"{len(predictions)} conformance predictions ({', '.join(ids)}); {registration['stop_rules'][0]}"
                if ids
                else str(registration.get("data_role", ""))
            )
        status = f"{tex_escape(str(registration['status']).replace('_', ' '))} (\\texttt{{{registration_hashes[key][:12]}}})"
        extension_rows.append(
            " & ".join(
                [
                    EXT_STUDY_NAMES.get(key, N3C_STUDY_NAME if key == "N" else tex_escape(registration["experiment_id"])),
                    cells_text,
                    items_text,
                    sigmas_text,
                    banks_text,
                    count(len(candidates["controls"])),
                    estimand,
                    status,
                ]
            )
            + r" \\"
        )
    extension_sources = [rel for key in REGISTRATION_RELS for rel in (REGISTRATION_RELS[key], REGISTRATION_RELS[key].replace(".json", ".sha256"))]
    extension_caption = (
        "Registered extension: cells, items per cell, noise scales, prototype banks (frozen grid plus learned controls), "
        "controls, primary estimand, and registration status with the first twelve hex characters of each registration "
        "SHA-256."
    )
    tables["table_extension.tex"] = "\n".join(
        _source_comment(sources, extension_sources, "Registered extension summary.")
        + _table(
            extension_caption,
            "tab:extension",
            _tabular(
                "lp{1.3cm}p{2.2cm}p{1.2cm}p{1.1cm}rp{4.6cm}p{2.2cm}",
                ["Study", "Cells", "Items per cell", "$\\sigma$", "Banks", "Controls", "Primary estimand", "Status"],
                extension_rows,
            ),
            star=True,
            colsep="3pt",
        )
        + [""]
    )

    # Extension regime table (identity bank per registered cell).
    regime_rows: list[str] = []
    regime_sources: list[str] = []
    for index, key in enumerate(EXT_EXPERIMENTS):
        analysis = extension_data[key]
        experiment_id = EXT_EXPERIMENTS[key]
        if index:
            regime_rows.append(r"\midrule")
        regime_rows.append(f"\\multicolumn{{10}}{{l}}{{\\emph{{{EXT_STUDY_NAMES[key]}}}}} \\\\")
        if analysis is None:
            regime_rows.append(f"\\multicolumn{{10}}{{l}}{{registered analysis pending: {Macros.PLACEHOLDER}}} \\\\")
            continue
        regime_sources.extend(
            [f"{EXT_ANALYSIS_DIR_REL}/{experiment_id}/cells.csv", f"{EXT_ANALYSIS_DIR_REL}/{experiment_id}/concentration.csv"]
        )
        identity = analysis.rows[NO_CORRECTION]
        shares = analysis.concentration[NO_CORRECTION]
        for cell_id in sorted(identity, key=lambda item: (str(identity[item]["model_id"]), str(identity[item]["dataset_id"]), float(identity[item]["sigma"]))):
            row = identity[cell_id]
            sigma_text = num(row["sigma"]) + ("" if row["primary_estimand_cell"] else r"$^{\dagger}$")
            regime_rows.append(
                " & ".join(
                    [
                        model_display(str(row["model_id"])),
                        dataset_display(str(row["dataset_id"])),
                        sigma_text,
                        count(int(row["example_count"])),
                        pct(row["clean_accuracy"]),
                        pct(row["smoothed_accuracy"]),
                        pct(ext_std(row)),
                        pct(ext_certified_wrong(row)),
                        pct(row["abstention_rate"]),
                        pct(shares[cell_id]["top_class_share"]),
                    ]
                )
                + r" \\"
            )
    regime_caption = (
        "No correction in every registered cell of Extensions A and B: items, clean "
        "accuracy, smoothed accuracy, standard certified accuracy and certified-wrong rate at $r=\\sigma$, abstention, "
        "and the top-class share of the smoothed predictions. $\\dagger$ marks the two CIFAR-10 $\\sigma=0.25$ bridge "
        "cells, which are reported but excluded from the primary macro. Values are primary-verified "
        "(Appendix~\\ref{app:deviations})."
    )
    tables["table_ext_regime.tex"] = "\n".join(
        _source_comment(sources, regime_sources, "Identity-bank regime per registered extension cell.")
        + _table(
            regime_caption,
            "tab:ext-regime",
            _tabular(
                "lllrrrrrrr",
                ["Model", "Dataset", "$\\sigma$", "$n$", "Clean", "Smoothed", "Std CA ($r=\\sigma$)", "Cert.-wrong ($r=\\sigma$)", "Abstain", "Top-class share"],
                regime_rows,
            ),
            star=True,
        )
        + [""]
    )

    # Extension contrasts table.
    contrast_rows: list[str] = []
    contrast_sources: list[str] = []
    critical_texts: list[str] = []
    for index, key in enumerate(EXT_EXPERIMENTS):
        analysis = extension_data[key]
        experiment_id = EXT_EXPERIMENTS[key]
        if index:
            contrast_rows.append(r"\midrule")
        contrast_rows.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{EXT_STUDY_NAMES[key]}}}}} \\\\")
        if analysis is None:
            contrast_rows.append(f"\\multicolumn{{6}}{{l}}{{registered analysis pending: {Macros.PLACEHOLDER}}} \\\\")
            continue
        contrast_sources.append(f"{EXT_ANALYSIS_DIR_REL}/{experiment_id}/cells.csv")
        if analysis.contrasts is not None:
            contrast_sources.append(f"{EXT_ANALYSIS_DIR_REL}/{experiment_id}/contrasts.json")
            critical_texts.append(
                f"{EXT_STUDY_NAMES[key]}: {pts(100.0 * float(analysis.contrasts['simultaneous_band']['critical_max_absolute_deviation']))} points"
            )
        region_points = ext_region_points(registrations[key])
        for candidate in [*analysis.grid, *analysis.controls]:
            entry = ext_interval(analysis, candidate)
            if entry is None:
                band_text, flag_text = Macros.PLACEHOLDER, Macros.PLACEHOLDER
            else:
                band_text = f"[{_math(spct(entry['lower']))}, {_math(spct(entry['upper']))}]"
                flag_text = "yes" if excludes_zero(entry) else "no"
            # X1: grid banks against the registered reporting region; the controls are not part of X1
            region_text = ("yes" if abs(float(ext_delta_exact(analysis, candidate))) <= region_points else "no") \
                if candidate in analysis.grid else "--"
            contrast_rows.append(
                " & ".join(
                    [
                        ext_display(candidate),
                        _math(spts(ext_delta(analysis, candidate))),
                        band_text,
                        _math(spts(ext_clean_delta(analysis, candidate))),
                        flag_text,
                        region_text,
                    ]
                )
                + r" \\"
            )
    contrast_caption = (
        "Registered contrasts versus no correction in the extension: equal-cell macro change of standard certified "
        "accuracy at $r=\\sigma$ over the primary cells (points), the registered shared-item class-stratified "
        "simultaneous 95\\% band, the equal-cell clean-accuracy change (points), whether the band excludes zero, and "
        f"whether a grid bank's change lies within the registered $\\pm{macros.values['ExtRegionPoints']}$-point "
        "reporting region (prediction X1; a reporting region, not an equivalence margin). "
        "Seventeen frozen-grid banks precede the four controls; the controls are supervised with "
        f"{macros.values['ExtControlLabelsPerClass']} labels per class from the disjoint control-training split. "
        + (f"Common half widths: {'; '.join(critical_texts)}. " if critical_texts else "")
        + "Values are primary-verified (Appendix~\\ref{app:deviations})."
    )
    tables["table_ext_contrasts.tex"] = "\n".join(
        _source_comment(sources, contrast_sources, "Registered extension contrasts versus no correction.")
        + _table(
            contrast_caption,
            "tab:ext-contrasts",
            _tabular(
                "lrlrcc",
                ["Candidate", "$\\Delta$ std.\\ CA (points)", "95\\% simultaneous band", "$\\Delta$ clean (points)", "Band excludes 0",
                 f"Within $\\pm{macros.values['ExtRegionPoints']}$"],
                contrast_rows,
            ),
            star=True,
        )
        + [""]
    )

    # N3C table.
    n3c_rows: list[str] = []
    n3c_sources: list[str] = []
    if n3c is None:
        n3c_rows.append(f"\\multicolumn{{8}}{{l}}{{Diagnostic predictions pending: {Macros.PLACEHOLDER}}} \\\\")
        n3c_caption = f"{N3C_STUDY_NAME} (registered analysis pending)."
    else:
        n3c_sources.append(n3c_rel)
        n3c_cells = n3c["cells"]
        n3c_table_banks = (GMC_ONE, GR_CLIP, exact_id, EXT_LOWRANK)
        for index, cell_id in enumerate(sorted(n3c_cells)):
            cell = n3c_cells[cell_id]
            model, dataset, _ = cell_id.rsplit("__", 2)
            if index:
                n3c_rows.append(r"\midrule")
            n3c_rows.append(
                f"\\multicolumn{{8}}{{l}}{{\\emph{{{model_display(model)}, {dataset_display(dataset)}}} "
                f"({count(int(cell['items']))} items, {count(int(cell['draws']))} draws, "
                f"{count(int(cell['P6_items_already_at_threshold']))} items already at $k_{{\\min}}$)}} \\\\"
            )
            for candidate in n3c_table_banks:
                p9 = cell["P9"][candidate]
                p7 = cell["P7"].get(candidate, {})
                applicable = bool(p7.get("applicable"))
                n3c_rows.append(
                    " & ".join(
                        [
                            ext_display(candidate),
                            count(p9["useful_wrong_to_correct"]),
                            count(p9["harmful_correct_to_wrong"]),
                            count(p9["wrong_to_wrong"]),
                            count(p9["unchanged"]),
                            count(p7["already_at_threshold"]) if applicable else "n/a",
                            count(p7["threshold_unattainable_on_these_draws"]) if applicable else "n/a",
                            count(p7["undetermined"]) if applicable else "n/a",
                        ]
                    )
                    + r" \\"
                )
        n3c_caption = (
            f"{N3C_STUDY_NAME} per registered cell for four banks "
            f"($k_{{\\min}}={count(n3c['k_min'])}$ of {', '.join(count(d) for d in n3c_summary['draws_per_item'])} draws). "
            "P9: draw-level winner movements against ground truth (useful = wrong-to-correct, harmful = correct-to-wrong). "
            "P7: items whose identity-bank top count already reaches $k_{\\min}$, items for which the threshold is "
            "unattainable on these draws under the candidate's flip budget, and undetermined items; n/a where the "
            "containment operator class does not apply. No certificate is issued."
        )
    tables["table_n3c.tex"] = "\n".join(
        _source_comment(sources, n3c_sources, "Fresh-noise diagnostic predictions per cell for four banks.")
        + _table(
            n3c_caption,
            "tab:n3c",
            _tabular(
                "lrrrrrrr",
                ["Bank", "P9 useful", "P9 harmful", "P9 wrong-to-wrong", "P9 unchanged", "P7 at $k_{\\min}$", "P7 unattainable", "P7 undetermined"],
                n3c_rows,
            ),
            star=True,
        )
        + [""]
    )

    followup_results(sources, macros, notes, tables, followup)
    return Assets(macros=macros, sources=sources, notes=notes, tables=tables)


# --------------------------------------------------------------------------
# EXP-021 follow-up results (D-145): registered analysis outputs -> macros and one table
# --------------------------------------------------------------------------

FU_ANALYSIS_RELS = {
    "A": "analysis/exp021/021a/analysis_ext.json",
    "B": "analysis/exp021/021b/analysis_ext.json",
    "C": "analysis/exp021/021c/analysis_ext.json",
}
FU_ANALYSIS_SCHEMA = "satml2027ext.analysis.v3"
FU_Q1_DISPLAY = {
    "reproduced": "reproduced",
    "reproduced_smaller": "reproduced but smaller",
    "not_reproduced": "not reproduced",
    "inconclusive": "inconclusive",
    "reversed": "reversed",
}
FU_MAGNITUDE_DISPLAY = {
    "gain_at_least_sesoi": "gain of at least the SESOI",
    "loss_at_least_sesoi": "loss of at least the SESOI",
    "practically_equivalent": "practically equivalent",
    "inconclusive": "inconclusive",
}
FU_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)  # open_clip.constants.OPENAI_DATASET_STD (CLIP preprocessing)
# Forest figure of EXP-021B (D-146): rows top to bottom, grouped; every value comes from the pooled families.
FU_FOREST_ROWS = (
    ("primary", "gr_clip_style_two_sided__coefficient_1", "Two-sided centering $c{=}1$"),
    ("primary", "text_only_centering__coefficient_1", "\\quad its text half"),
    ("primary", "image_only_centering__coefficient_1", "\\quad its image half"),
    ("primary", "clean_boundary_active__step_0.16", "Boundary-active step 0.16"),
    ("primary", "cohen_aligned_noisy_margin__step_0.16", "Noisy-margin step 0.16"),
    ("shared", "global_mean_centering__coefficient_1", "Text mean-centering $c{=}1$"),
    ("shared", "chowers_exact_projected_gap__coefficient_1", "Projected gap $c{=}1$"),
    ("supervised", "control__learned_shared_translation_tangent", "Learned shared translation"),
    ("supervised", "control__lowrank_tangent_r8", "Rank-8 tangent adapter"),
    ("supervised", "control__fewshot_prompt_bank", "Few-shot prompt bank"),
    ("supervised", "control__noisy_class_mean", "Noisy class-mean bank"),
)
FU_FOREST_GROUPS = ("primary", "shared", "supervised")
# Descriptive radius curves (D-146), written by scripts/exp021_radius_curves.py after the registered analysis.
FU_CURVES_REL = "analysis/exp021/021b/descriptive_radius_curves.json"
FU_CURVES_SCHEMA = "satml2027ext.descriptive_radius_curves.v1"
FU_CURVE_COLUMNS = (("no_correction", "identity"), ("gr_clip_style_two_sided__coefficient_1", "twosided"),
                    ("control__learned_shared_translation_tangent", "sharedw"), ("control__lowrank_tangent_r8", "lowrank"))


FU_DATASET_DISPLAY = {"cifar100": "CIFAR-100 validation folds", "eurosat": "EuroSAT validation folds",
                      "cifar100_test": "CIFAR-100 test", "eurosat_sealed": "EuroSAT sealed holdout",
                      "imagenette": "Imagenette-320 validation"}


def fu_design_table(sources: Sources, followup: dict[str, dict[str, Any]]) -> str:
    """Appendix design table of EXP-021 (D-148), from the three registrations only (outcome-independent)."""

    def question_range(registration: dict[str, Any]) -> str:
        ids = sorted(int(entry["id"][1:]) for entry in registration.get("predictions", []))
        if not ids or ids != list(range(ids[0], ids[-1] + 1)):
            raise RuntimeError(f"{registration['experiment_id']}: registered questions are not a contiguous range: {ids}")
        return f"Q{ids[0]}" if len(ids) == 1 else f"Q{ids[0]}--Q{ids[-1]}"

    rows = []
    for key in ("A", "B", "C"):
        registration = followup[key]
        cells = registration["cells"]
        certification = registration["certification"]
        models = ", ".join(model_display(model) for model in sorted({cell["model_id"] for cell in cells}))
        items: dict[str, set[int]] = {}
        for cell in cells:
            items.setdefault(cell["dataset_id"], set()).add(int(cell["item_count"]))
        data = "; ".join(f"{FU_DATASET_DISPLAY[dataset]} ({', '.join(count(n) for n in sorted(counts))})"
                         for dataset, counts in sorted(items.items()))
        sigmas = ", ".join(num(value) for value in sorted({float(cell["sigma"]) for cell in cells}))
        draws = f"{count(int(certification['selection_draws']))} / {count(int(certification['confirmation_draws']))}"
        if certification.get("budget_checkpoints"):
            draws += " (prefix " + ", ".join(count(int(value)) for value in certification["budget_checkpoints"]) + ")"
        banks = count(len(registration["candidates"]["candidate_ids"]))
        if key == "A":
            power = f"MDE {num(float(registration['power']['at_discordance_0.10']['minimum_detectable_points']))}"
        elif key == "B":
            pooled = registration["power"]["pooled_over_datasets_per_sigma_at_discordance_0.10"]
            eurosat = registration["power"]["per_dataset_sigma_at_discordance_0.10"]["eurosat_sealed"]
            power = (f"pooled MDE {num(float(pooled['minimum_detectable_points']))} (power "
                     f"{num(float(pooled['power_for_2_points']))} at 2); EuroSAT MDE "
                     f"{num(float(eurosat['minimum_detectable_points']))}")
        else:
            power = "descriptive"
        rows.append(" & ".join([FU_STUDY_NAMES[key], count(len(cells)), models, data, sigmas, banks, draws,
                                num(float(certification["alpha_per_example"])), question_range(registration), power])
                    + r" \\")
    caption = ("Registered follow-up (registered before sampling; registration digests "
               "\\FURegHashA{}, \\FURegHashB{} and \\FURegHashC{}): cells, backbones, "
               "datasets with items per cell, noise levels, banks, selection and confirmation draws, the per-example "
               "error level $\\alpha$ of the one-sided Clopper--Pearson bound (confidence $1-\\alpha$), registered "
               "questions, and registered power "
               "(minimum detectable effect in points at paired discordance 0.10). Inference: class-stratified "
               "shared-item bootstrap with \\FUReplicates{} replicates, Holm within each family, smallest effect of "
               "interest \\FUSESOI{} points.")
    comment = [f"% Generated by {SCRIPT_REL}; do not edit by hand.", "% EXP-021 registered design.",
               "% Registrations (embedded registration_sha256, checked against their .sha256 sidecars):"]
    comment += [f"%   {FU_REGISTRATION_RELS[key]}  {followup[key]['registration_sha256']}" for key in ("A", "B", "C")]
    return "\n".join(
        comment
        + _table(caption, "tab:followup-design",
                 _tabular("lr" + "".join(f">{{\\raggedright\\arraybackslash}}p{{{width}}}" for width in ("2.1cm", "3.2cm"))
                          + "lr>{\\raggedright\\arraybackslash}p{1.5cm}rl>{\\raggedright\\arraybackslash}p{2.6cm}",
                          ["Study", "Cells", "Backbones", "Datasets (items per cell)", "$\\sigma$", "Banks",
                           "Draws $n_0$ / $n$", "$\\alpha$", "Questions", "Power"], rows),
                 star=True, colsep="3pt")
        + [""]
    )


FU_DATASETS_B = ("cifar100_test", "eurosat_sealed", "imagenette")
FU_CELLS_CSV_REL = "analysis/exp021/021b/cells_ext.csv"
FU_COMPUTE_REL = "analysis/exp021/compute_summary.json"


def fu_dataset_table(b: dict[str, Any] | None, registration: dict[str, Any], sources: Sources) -> str:
    """Secondary per-dataset EXP-021B families (D-148): primary contrasts per dataset and noise level."""

    power = registration["inference"]["family_power"]
    header = ["Bank", "$\\sigma$"] + [
        f"{FU_DATASET_DISPLAY[dataset]} ({power[f'{dataset}__sigma{FU_SIGMA_TOKENS[0][0]}']['label'].split(' (')[0]})"
        for dataset in FU_DATASETS_B]
    rows = []
    for candidate, label in FU_PRIMARY_LABELS.items():
        for position, (sigma_text, _) in enumerate(FU_SIGMA_TOKENS):
            cells = []
            for dataset in FU_DATASETS_B:
                if b is None:
                    cells.append(Macros.PLACEHOLDER)
                    continue
                family = f"{dataset}__sigma{sigma_text}"
                contrast = b["families"][family]["contrasts"][candidate]
                magnitude = b["questions"]["Q6"][family][candidate]["magnitude"]
                # the registration reports the observed paired discordance next to every interval
                cells.append(f"${spts(contrast['point_points'])}$ [${spts(contrast['ci95_unadjusted_lower'])}$,"
                             f"${spts(contrast['ci95_unadjusted_upper'])}$] {FU_MAGNITUDE_SHORT[magnitude]} "
                             f"$d{{=}}{fu_discordance(contrast)}$")
            rows.append(" & ".join([label if position == 0 else "", f"${sigma_text}$"] + cells) + r" \\")
        if candidate != list(FU_PRIMARY_LABELS)[-1]:
            rows.append(r"\addlinespace")
    caption = ("Secondary, dataset-specific Follow-up B families (registered, not adjudicating): change in standard "
               "certified accuracy at $r=\\sigma$ versus no correction for the five primary contrasts, pooled over the "
               "backbones of each dataset, with the unadjusted 95\\% interval, the Holm-adjusted magnitude class "
               "within the family, and the observed paired discordance $d$ (\\%, the share of positions whose "
               "certified-correct outcome differs from no correction's), reported next to every interval as "
               "registered. Headers give each family's registered power label. Adjudication uses the pooled "
               "families (Figure~\\ref{fig:followup-forest}).")
    return "\n".join(
        _source_comment(sources, [FU_ANALYSIS_RELS["B"]] if b is not None else [], "EXP-021B dataset families.")
        + _table(caption, "tab:followup-datasets", _tabular("llccc", header, rows), star=True, colsep="3pt")
        + [""]
    )


FU_DIRECTION_SHORT = {"positive": "pos.", "negative": "neg.", "unresolved": "unres."}


def _fu_pooled_family(report: dict[str, Any], sigma_text: str | None) -> str:
    """The one pooled family of a registered output (at a noise level, if given)."""

    keys = [key for key in report["families"] if key.startswith("pooled__")
            and (sigma_text is None or format(float(key.split("__sigma")[1]), "g") == sigma_text)]
    if len(keys) != 1:
        raise RuntimeError(f"expected one pooled family{'' if sigma_text is None else ' at sigma ' + sigma_text}, found {keys}")
    return keys[0]


def _fu_interval(point: float, lower: float, upper: float) -> str:
    return f"${spts(point)}$ [${spts(lower)}$, ${spts(upper)}$]"


def fu_discordance(contrast: dict[str, Any]) -> str:
    """Observed paired discordance of a registered contrast, in percent with one decimal (registration: reported
    next to every interval)."""

    value = float(contrast["discordance"])
    if not 0.0 <= value <= 1.0:
        raise RuntimeError(f"discordance {value} is not a share")
    return one_decimal_pct(value)


def _fu_bank_labels() -> dict[str, str]:
    return {candidate: label for _, candidate, label in FU_FOREST_ROWS}


def fu_followup_table(b: dict[str, Any] | None, registration: dict[str, Any], sources: Sources) -> str:
    """The pooled registered primary family and three reference rows, item-pooled (the registered primary estimand) and
    equal-cell (the audit's estimand), with the Holm-adjusted direction and magnitude classes (review of 2026-09-28)."""

    labels = {**FU_PRIMARY_LABELS, **FU_EXTRA_LABELS}
    primary_ids = list(registration["inference"]["primary_family"]["contrasts"])
    rows: list[str] = []
    for index, (sigma_text, _) in enumerate(FU_SIGMA_TOKENS):
        if index:
            rows.append(r"\midrule")
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{$\\sigma={sigma_text}$}}}} \\\\")
        for candidate, label in labels.items():
            if b is None:
                rows.append(" & ".join([label] + [Macros.PLACEHOLDER] * 5) + r" \\")
                continue
            family = _fu_pooled_family(b, sigma_text)
            row = b["families"][family]["contrasts"][candidate]
            q6 = b["questions"]["Q6"][family]
            item = _fu_interval(row["point_points"], row["ci95_unadjusted_lower"], row["ci95_unadjusted_upper"])
            equal = _fu_interval(row["macro_point_points"], row["macro_ci95_unadjusted_lower"], row["macro_ci95_unadjusted_upper"])
            if candidate in primary_ids:
                direction = FU_DIRECTION_SHORT[fu_direction(q6[candidate]["direction"])]
                magnitude = FU_MAGNITUDE_SHORT[q6[candidate]["magnitude"]]
            else:
                direction = magnitude = "--"
            rows.append(" & ".join([label, item, equal, fu_discordance(row), direction, magnitude]) + r" \\")
            if candidate == list(FU_PRIMARY_LABELS)[-1]:
                rows.append(r"\addlinespace")
    caption = ("Follow-up B (fresh items): change in standard certified accuracy at $r=\\sigma$ versus no correction, "
               "in points, pooled over datasets and backbones, with unadjusted 95\\% bootstrap intervals, item-pooled "
               "(the registered primary estimand) and equal-cell (the audit's estimand, reported alongside). For the "
               "five contrasts of the registered primary family, the Holm-adjusted direction (pos., neg., unres.\\ = "
               "unresolved) and magnitude class against the \\FUSESOI-point smallest effect of interest follow (gain, "
               "loss, equiv.\\ = practically equivalent, inconcl.\\ = inconclusive); an unadjusted interval can exclude "
               "zero while the adjusted direction is unresolved. $d$: the observed paired discordance (\\%, the share of "
               "positions whose certified-correct outcome differs from no correction's), reported next to every "
               "interval as registered. The last three rows are reference contrasts.")
    return "\n".join(
        _source_comment(sources, [FU_ANALYSIS_RELS["B"]] if b is not None else [], "Follow-up B registered contrasts.")
        + _table(caption, "tab:followup",
                 _tabular("lllrcc", ["Bank", "Item-pooled [95\\% CI]", "Equal-cell [95\\% CI]", "$d$ (\\%)", "Holm dir.",
                                    "Class"], rows),
                 star=True, colsep="4pt")
        + [""]
    )


def fu_registered_table(a: dict[str, Any] | None, b: dict[str, Any] | None, sources: Sources) -> str:
    """Registered outputs the text only summarizes: Q5 (every supervised control against no correction and against
    two-sided centering) and Q2 (the text/image decomposition at every registered coefficient), review of 2026-09-28."""

    blocks = []
    if a is not None:
        blocks.append((f"{FU_STUDY_NAMES['A']}, $\\sigma=0.25$", a, _fu_pooled_family(a, None)))
    if b is not None:
        blocks += [(f"{FU_STUDY_NAMES['B']}, $\\sigma={sigma_text}$", b, _fu_pooled_family(b, sigma_text))
                   for sigma_text, _ in FU_SIGMA_TOKENS]
    names = _fu_bank_labels()
    control_rows: list[str] = []
    decomposition_rows: list[str] = []
    for index, (label, report, family) in enumerate(blocks):
        if index:
            control_rows.append(r"\addlinespace")
            decomposition_rows.append(r"\addlinespace")
        q5 = report["questions"]["Q5"][family]
        for position, control in enumerate(FU_CONTROLS):
            entry = q5[control]
            low, high = entry["minus_comparator_ci95_unadjusted"]
            control_rows.append(" & ".join([
                label if position == 0 else "", names[control], f"${spts(entry['point_points'])}$",
                FU_DIRECTION_SHORT[fu_direction(entry["direction"])],
                _fu_interval(entry["minus_comparator_point"], low, high)]) + r" \\")
        decomposition = report["families"][family]["decomposition"]
        for position, coefficient in enumerate(sorted(decomposition, key=float)):
            entry = decomposition[coefficient]
            low, high = entry["interaction_ci95_unadjusted"]
            decomposition_rows.append(" & ".join([
                label if position == 0 else "", f"${coefficient}$", f"${spts(entry['two_sided'])}$",
                f"${spts(entry['text_only'])}$", f"${spts(entry['image_only'])}$",
                _fu_interval(entry["interaction_point"], low, high)]) + r" \\")
    if not blocks:
        control_rows = [f"\\multicolumn{{5}}{{l}}{{registered analysis pending: {Macros.PLACEHOLDER}}} \\\\"]
        decomposition_rows = [f"\\multicolumn{{6}}{{l}}{{registered analysis pending: {Macros.PLACEHOLDER}}} \\\\"]
    caption = ("Registered follow-up outputs summarized in the text, item-pooled at $r=\\sigma$ (points). Top (Q5): each "
               "supervised control against no correction, with its Holm-adjusted direction, and against two-sided "
               "centering $c=1$ with the unadjusted 95\\% interval. Bottom (Q2): two-sided centering, its text half and "
               "its image half at every registered coefficient $c$, and their interaction (two-sided minus both halves) "
               "with its unadjusted 95\\% interval.")
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
             f"\\caption{{{caption}}}", r"\label{tab:followup-registered}"]
    lines += _tabular("llrcl", ["Study", "Control", "vs.\\ no correction", "Holm dir.", "vs.\\ two-sided [95\\% CI]"],
                      control_rows)
    lines += [r"\par\vspace{4pt}"]
    lines += _tabular("llrrrl", ["Study", "$c$", "Two-sided", "Text half", "Image half", "Interaction [95\\% CI]"],
                      decomposition_rows)
    lines += [r"\end{table*}", ""]
    rels = [FU_ANALYSIS_RELS[key] for key, report in (("A", a), ("B", b)) if report is not None]
    return "\n".join(_source_comment(sources, rels, "Follow-up Q5 and Q2 registered outputs.") + lines)


def _fu_step_totals(report: dict[str, Any], draws_per_item: int) -> dict[tuple[str, float], dict[str, int]]:
    """Q3 step response pooled over cells: per (proposal family, step) the changed, in-budget, useful and harmful draws."""

    totals: dict[tuple[str, float], dict[str, int]] = {}
    for row in report["questions"]["Q3"]["step_response"]:
        key = (str(row["candidate_id"]).split("__")[0], float(row["step"]))
        entry = totals.setdefault(key, {"changed": 0, "budget": 0, "useful": 0, "harmful": 0, "draws": 0})
        entry["changed"] += int(row["flip_changed_draws"])
        entry["budget"] += int(row["flip_loose_budget_draws"])
        entry["useful"] += int(row["flip_useful_draws"])
        entry["harmful"] += int(row["flip_harmful_draws"])
        entry["draws"] += int(row["items"]) * draws_per_item
    for (family, step), entry in totals.items():
        if not entry["changed"] <= entry["budget"] <= entry["draws"]:
            raise RuntimeError(f"Q3 step response is not nested for {family} at step {step}: {entry}")
    for step in {step for _, step in totals}:
        budgets = {entry["budget"] for (family, s), entry in totals.items() if s == step}
        if len(budgets) != 1:
            raise RuntimeError(f"the loose flip budget differs between the proposal objectives at step {step}")
    return totals


FU_STEP_FAMILIES = (("clean_boundary_active", "Boundary-active"), ("cohen_aligned_noisy_margin", "Noisy-margin"))


def fu_steps_table(a: dict[str, Any] | None, b: dict[str, Any] | None, followup: dict[str, dict[str, Any]],
                   sources: Sources) -> str:
    """Q3 step response (registered to be reported whatever its shape), Follow-ups A and B, pooled over cells."""

    totals = {key: _fu_step_totals(report, int(followup[key]["certification"]["confirmation_draws"]))
              for key, report in (("A", a), ("B", b)) if report is not None}
    header = [r"\begin{tabular}{r" + "r" * 10 + "}", r"\toprule",
              r" & \multicolumn{2}{c}{In flip budget} & \multicolumn{4}{c}{Boundary-active} & "
              r"\multicolumn{4}{c}{Noisy-margin} \\",
              r"\cmidrule(lr){2-3}\cmidrule(lr){4-7}\cmidrule(lr){8-11}",
              r"Step & A & B & changed A & changed B & net A & net B & changed A & changed B & net A & net B \\",
              r"\midrule"]
    rows: list[str] = []
    steps = sorted({step for table in totals.values() for _, step in table})
    for step in steps:
        cells = [num(step)]
        for key in ("A", "B"):
            entry = totals.get(key, {}).get((FU_STEP_FAMILIES[0][0], step))
            cells.append(pct(entry["budget"] / entry["draws"]) if entry else Macros.PLACEHOLDER)
        for family, _ in FU_STEP_FAMILIES:
            for field in ("changed", "net"):
                for key in ("A", "B"):
                    entry = totals.get(key, {}).get((family, step))
                    if entry is None:
                        cells.append(Macros.PLACEHOLDER)
                    elif field == "changed":
                        cells.append(pct(entry["changed"] / entry["draws"]))
                    else:
                        cells.append(f"${spct((entry['useful'] - entry['harmful']) / entry['draws'])}$")
        rows.append(" & ".join(cells) + r" \\")
    if not steps:
        rows = [f"\\multicolumn{{11}}{{l}}{{registered analysis pending: {Macros.PLACEHOLDER}}} \\\\"]
    caption = ("Step response of the two proposal objectives (Q3, registered to be reported whatever its shape), "
               "pooled over the cells of Follow-ups A and B: the percentage of confirmation draws whose original top-1 "
               "margin is at most $2\\min\\{1,\\lVert w\\rVert\\}$, the coarse flip budget of "
               "Proposition~\\ref{prop:flip} (the same for both objectives), the percentage whose vote changes, and "
               "the net percentage of useful (wrong-to-correct) minus harmful (correct-to-wrong) changes. The coarse "
               "budget is a necessary condition: it admits far more draws than change.")
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
             f"\\caption{{{caption}}}", r"\label{tab:followup-steps}"] + header + rows + [r"\bottomrule", r"\end{tabular}",
                                                                                       r"\end{table*}", ""]
    rels = [FU_ANALYSIS_RELS[key] for key, report in (("A", a), ("B", b)) if report is not None]
    return "\n".join(_source_comment(sources, rels, "Follow-up Q3 step response.") + lines)


def fu_outcome_files(cells_rows: list[dict[str, Any]] | None, b: dict[str, Any] | None,
                     registration: dict[str, Any]) -> dict[str, str]:
    """How the uncorrected model's decisions split at r = sigma, per dataset and noise level (D-148).

    From the registered cells table (cells_ext.csv); cross-checked against the Q7 identity rows of the report.
    """

    header = "% Generated by scripts/generate_satml2027_paper_assets.py; do not edit by hand."
    order = [(dataset, sigma_text) for dataset in FU_DATASETS_B for sigma_text, _ in FU_SIGMA_TOKENS]
    labels = ",".join("{" + f"{FU_DATASET_DISPLAY[dataset].split(' ')[0]}, $\\sigma={sigma_text}$" + "}"
                      for dataset, sigma_text in order)
    lines = ["y correct wrong below abstain"]
    if cells_rows is not None and b is not None:
        items = {cell["cell_id"]: int(cell["item_count"]) for cell in registration["cells"]}
        identity = {str(row["cell_id"]): row for row in cells_rows if str(row["candidate_id"]) == "no_correction"}
        regime = {row["cell_id"]: row for row in b["questions"]["Q7"]["identity_regime"]}
        if set(identity) != set(regime):
            raise RuntimeError("cells_ext.csv and the report's Q7 rows cover different cells")
        for cell_id, row in identity.items():
            if float(row["primary_standard_ca"]) != float(regime[cell_id]["primary_standard_ca"]):
                raise RuntimeError(f"cells_ext.csv and the report disagree for {cell_id}: not the same analysis run")
        for index, (dataset, sigma_text) in enumerate(order, start=1):
            chosen = [cell_id for cell_id in identity
                      if cell_id.split("__")[1] == dataset and fu_sigma_of_cell(cell_id) == sigma_text]
            if not chosen:
                raise RuntimeError(f"no identity cell for {dataset} at sigma {sigma_text}")
            total = sum(items[cell_id] for cell_id in chosen)
            def mean(key: str) -> float:
                return sum(float(identity[cell_id][key]) * items[cell_id] for cell_id in chosen) / total
            correct, wrong, abstain = mean("primary_standard_ca"), mean("primary_certified_wrong_rate"), mean("abstention_rate")
            below = 1.0 - correct - wrong - abstain
            if below < -1e-9:
                raise RuntimeError(f"{dataset} sigma {sigma_text}: outcome shares exceed one")
            lines.append(f"{index} {100 * correct:.4f} {100 * wrong:.4f} {100 * max(below, 0.0):.4f} {100 * abstain:.4f}")
        ready = 1
    else:
        ready = 0
    return {
        "fig_followup_outcomes.dat": "\n".join(lines) + "\n",
        "fig_followup_outcomes_axis.tex": "\n".join([
            header, f"\\def\\FUOutcomesReady{{{ready}}}",
            "\\pgfplotsset{fuoutcomesaxis/.style={ytick={" + ",".join(str(i) for i in range(1, len(order) + 1))
            + "}, yticklabels={" + labels + "}}}", ""]),
    }


def fu_curves_files(curves: dict[str, Any] | None) -> dict[str, str]:
    """pgfplots data of the descriptive curves at both noise levels, or a placeholder flag."""

    header = "% Generated by scripts/generate_satml2027_paper_assets.py; do not edit by hand."
    names = " ".join(["r"] + [column for _, column in FU_CURVE_COLUMNS])
    files: dict[str, str] = {}
    limits: dict[str, float] = {}
    top = 10.0
    for sigma_text, token in FU_SIGMA_TOKENS:
        lines = [names]
        if curves is not None:
            block = curves["pooled"][sigma_text]
            grid = [float(value) for value in block["grid"]]
            columns = [[float(value) for value in block["curves"][bank]] for bank, _ in FU_CURVE_COLUMNS]
            if any(len(column) != len(grid) for column in columns):
                raise RuntimeError(f"curves at sigma {sigma_text}: a curve and its grid differ in length")
            for index, radius in enumerate(grid):
                lines.append(" ".join([f"{radius:.3f}"] + [f"{column[index]:.4f}" for column in columns]))
            limits[token] = grid[-1]
            top = max(top, max(column[0] for column in columns))
        files[f"fig_followup_curves_{token.lower()}.dat"] = "\n".join(lines) + "\n"
    ready = 1 if curves is not None else 0
    ymax = 10.0 * int(top / 10.0 + 0.999)
    files["fig_followup_curves_axis.tex"] = "\n".join([
        header,
        f"\\def\\FUCurvesReady{{{ready}}}",
        f"\\def\\FUCurvesXmaxLow{{{limits.get('Low', 0.4):.3f}}}",
        f"\\def\\FUCurvesXmaxHigh{{{limits.get('High', 0.8):.3f}}}",
        f"\\def\\FUCurvesYmax{{{ymax:.0f}}}",
        "",
    ])
    return files


def fu_forest_files(rows: dict[str, dict[str, dict[str, Any]]] | None, sesoi: float) -> dict[str, str]:
    """pgfplots data and axis style of the EXP-021B forest figure; ``rows`` maps sigma text -> candidate -> contrast.

    Without the registered output the axis file sets \\FUForestReady to 0 and the figure shows a framed placeholder;
    no value is ever drawn from anything but the registered pooled-family contrasts.
    """

    header = "% Generated by scripts/generate_satml2027_paper_assets.py; do not edit by hand."
    labels = ",".join("{" + label + "}" for _, _, label in FU_FOREST_ROWS)
    ticks = ",".join(str(index) for index in range(1, len(FU_FOREST_ROWS) + 1))
    boundaries = [index + 0.5 for index in range(1, len(FU_FOREST_ROWS))
                  if FU_FOREST_ROWS[index][0] != FU_FOREST_ROWS[index - 1][0]]
    files: dict[str, str] = {}
    if rows is None:
        low, high = -sesoi - 1.0, sesoi + 1.0
        for group in FU_FOREST_GROUPS:
            files[f"fig_followup_forest_{group}.dat"] = "y lo lo_minus lo_plus hi hi_minus hi_plus\n"
        ready = 0
    else:
        values: list[float] = [-sesoi, sesoi]
        for group in FU_FOREST_GROUPS:
            lines = ["y lo lo_minus lo_plus hi hi_minus hi_plus"]
            for index, (row_group, candidate, _) in enumerate(FU_FOREST_ROWS, start=1):
                if row_group != group:
                    continue
                cells = []
                for sigma_text, _ in FU_SIGMA_TOKENS:
                    contrast = rows[sigma_text][candidate]
                    point = float(contrast["point_points"])
                    lower, upper = float(contrast["ci95_unadjusted_lower"]), float(contrast["ci95_unadjusted_upper"])
                    if not lower <= point <= upper:
                        raise RuntimeError(f"forest: {candidate} at sigma {sigma_text} has its point outside its interval")
                    values.extend([lower, upper])
                    cells.extend([f"{point:.4f}", f"{point - lower:.4f}", f"{upper - point:.4f}"])
                lines.append(" ".join([str(index)] + cells))
            files[f"fig_followup_forest_{group}.dat"] = "\n".join(lines) + "\n"
        span = max(values) - min(values)
        low, high = min(values) - 0.06 * span, max(values) + 0.06 * span
        ready = 1
    files["fig_followup_forest_axis.tex"] = "\n".join([
        header,
        f"\\def\\FUForestReady{{{ready}}}",
        f"\\def\\FUForestXmin{{{low:.2f}}}",
        f"\\def\\FUForestXmax{{{high:.2f}}}",
        f"\\def\\FUForestRows{{{len(FU_FOREST_ROWS)}}}",
        "\\pgfplotsset{fuforestaxis/.style={xmin=" + f"{low:.2f}" + ", xmax=" + f"{high:.2f}" + ", ytick={" + ticks
        + "}, yticklabels={" + labels + "}}}",
        "\\def\\FUForestSeparators{" + ",".join(f"{value:.1f}" for value in boundaries) + "}",
        "",
    ])
    return files
FU_CONFORMANCE_DISPLAY = {"conforms": "conforms", "violated": "violated", "not_evaluable": "not evaluable"}  # Q3/Q4


def fu_conformance(verdict: Any) -> str:
    if verdict not in FU_CONFORMANCE_DISPLAY:
        raise RuntimeError(f"unknown Q3/Q4 conformance verdict {verdict!r}")
    return FU_CONFORMANCE_DISPLAY[verdict]


def fu_direction(value: Any) -> str:
    if value not in ("positive", "negative", "unresolved"):
        raise RuntimeError(f"unknown direction class {value!r}")
    return str(value)


FU_MAGNITUDE_SHORT = {"gain_at_least_sesoi": "gain", "loss_at_least_sesoi": "loss", "practically_equivalent": "equiv.",
                      "inconclusive": "inconcl."}
FU_SIGMA_TOKENS = (("0.12", "Low"), ("0.25", "High"))
FU_MODEL_TOKENS = {
    "openai-clip-vit-b32-quickgelu": "OBthirtytwo",
    "openai-clip-vit-l14-quickgelu": "OLfourteen",
    "openclip-vit-b32-laion2b": "LBthirtytwo",
}
FU_TWO_SIDED = "gr_clip_style_two_sided__coefficient_1"
FU_PRIMARY_LABELS = {
    "gr_clip_style_two_sided__coefficient_1": "Two-sided centering $c=1$",
    "text_only_centering__coefficient_1": "\\quad its text half",
    "image_only_centering__coefficient_1": "\\quad its image half",
    "clean_boundary_active__step_0.16": "Boundary-active proposal, step 0.16",
    "cohen_aligned_noisy_margin__step_0.16": "Noisy-margin proposal, step 0.16",
}
FU_EXTRA_LABELS = {
    "chowers_exact_projected_gap__coefficient_1": "Projected gap $c=1$",
    "control__lowrank_tangent_r8": "Rank-8 adapter (supervised)",
    "control__learned_shared_translation_tangent": "Learned shared translation (supervised)",
}
FU_CONTROLS = ("control__lowrank_tangent_r8", "control__learned_shared_translation_tangent",
               "control__fewshot_prompt_bank", "control__noisy_class_mean")
FU_CONTROL_TOKENS = {"control__lowrank_tangent_r8": "Lowrank", "control__learned_shared_translation_tangent": "Shared",
                     "control__fewshot_prompt_bank": "Fewshot", "control__noisy_class_mean": "NoisyMean"}


FU_FAMILY_NAMES = {  # follow-up bank families -> (TeX-ready lowercase name for mid-sentence use, parameter kind)
    "global_mean_centering": ("text mean-centering", "c"),  # the paper's names: mean-centering, two-sided centering
    "chowers_exact_projected_gap": ("projected-gap translation", "c"),
    "gr_clip_style_two_sided": ("two-sided centering", "c"),
    "text_only_centering": ("the text half of two-sided centering", "c"),
    "image_only_centering": ("the image half of two-sided centering", "c"),
    "clean_boundary_active": ("the boundary-active proposal", "step"),
    "cohen_aligned_noisy_margin": ("the noisy-margin proposal", "step"),
}
FU_FIXED_NAMES = {
    "no_correction": "no correction",
    "control__lowrank_tangent_r8": "the rank-8 tangent adapter (supervised)",
    "control__learned_shared_translation_tangent": "the learned shared translation (supervised)",
    "control__noisy_class_mean": "the noisy class-mean bank (supervised)",
    "control__fewshot_prompt_bank": "the few-shot prompt bank (supervised)",
}


def fu_display(candidate_id: str) -> str:
    """TeX-ready name of a registered follow-up bank; refuses an unknown id rather than printing it raw."""

    if candidate_id in FU_FIXED_NAMES:
        return FU_FIXED_NAMES[candidate_id]
    family, _, suffix = candidate_id.partition("__")
    kind, _, value = suffix.partition("_")
    if family in FU_FAMILY_NAMES and kind in ("coefficient", "step") and re.fullmatch(r"[0-9]+(\.[0-9]+)?", value):
        name, expected = FU_FAMILY_NAMES[family]
        if (kind == "coefficient") == (expected == "c"):
            # "at", not a comma: a comma inside a name misparses in lists of names (proofreading, 2026-09-29)
            return f"{name} at $c={value}$" if expected == "c" else f"{name} at step {value}"
    raise RuntimeError(f"no display name for the follow-up bank {candidate_id!r}")


FU_DATASET_TOKENS = {"cifar100_test": "Cifar", "eurosat_sealed": "Eurosat", "imagenette": "Imagenette"}
# running-text names of the follow-up's fresh-item datasets (the design table keeps FU_DATASET_DISPLAY)
FU_DATASET_TEXT = {"cifar100_test": "the CIFAR-100 test items", "eurosat_sealed": "the sealed EuroSAT holdout",
                   "imagenette": "Imagenette"}


def fu_bound(value: float) -> str:
    """An upper bound in scientific notation, rounded up so the printed value is still a bound ($1.03\\times10^{-5}$)."""
    import math

    value = float(value)
    if value <= 0:
        raise RuntimeError(f"a tie budget must be positive, got {value}")
    exponent = math.floor(math.log10(value))
    mantissa = math.ceil(value / 10 ** exponent * 100 - 1e-9) / 100
    if mantissa >= 10:
        mantissa, exponent = mantissa / 10, exponent + 1
    return f"{mantissa:.2f}\\times10^{{{exponent}}}"


def fu_p(value: float) -> str:
    """A complete relation in math mode ($p<0.001$ or $p=0.012$), so the text never wraps it."""
    value = float(value)
    return "$p<0.001$" if value < 0.001 else f"$p={value:.3f}$"


def fu_sigma_of_cell(cell_id: str) -> str:
    match = re.search(r"__sigma([0-9.]+)", cell_id)
    if not match:
        raise RuntimeError(f"no sigma in cell id {cell_id}")
    return format(float(match.group(1)), "g")


def followup_results(sources: Sources, macros: Macros, notes: dict[str, Any], tables: dict[str, str],
                     followup: dict[str, dict[str, Any]]) -> None:
    """EXP-021 results from the registered analysis outputs; every value is checked against its registration."""

    tables["table_followup_design.tex"] = fu_design_table(sources, followup)
    reports: dict[str, dict[str, Any] | None] = {}
    for key, rel in FU_ANALYSIS_RELS.items():
        if not (sources.root / rel).is_file():
            reports[key] = None
            continue
        report = sources.read_json(rel)
        if report.get("registration_sha256") != followup[key]["registration_sha256"]:
            raise RuntimeError(f"{rel} was produced for another registration")
        if report.get("schema_version") != FU_ANALYSIS_SCHEMA:
            raise RuntimeError(f"{rel} has schema {report.get('schema_version')}, not {FU_ANALYSIS_SCHEMA}")
        if key in ("A", "B"):
            inference = report["inference"]
            if inference.get("replicates_overridden") or int(inference["replicates_used"]) != int(
                    followup[key]["inference"]["uncertainty"]["replicates"]):
                raise RuntimeError(f"{rel} did not use the registered bootstrap replicates")
        reports[key] = report
    notes["exp021_results"] = {"sources": {key: FU_ANALYSIS_RELS[key] for key, report in reports.items() if report is not None},
                               "pending": [key for key, report in reports.items() if report is None]}

    def fill(names: list[str], values: dict[str, str] | None, reason: str) -> None:
        for name in names:
            if values is None:
                macros.miss(name, reason)
            else:
                macros.set(name, values[name])

    def pooled_key(report: dict[str, Any], sigma_text: str | None = None) -> str:
        keys = [key for key in report["families"] if key.startswith("pooled__")
                and (sigma_text is None or format(float(key.split("__sigma")[1]), "g") == sigma_text)]
        if len(keys) != 1:
            raise RuntimeError(f"expected one pooled family{'' if sigma_text is None else ' at sigma ' + sigma_text}, found {keys}")
        return keys[0]

    # EXP-021A: Q1 (fresh re-certification of the saved two-sided gain), Q2 decomposition, Q3/Q4 conformance.
    a_names = ["FUQOneVerdict", "FUQOnePoint", "FUQOneLower", "FUQOneUpper", "FUQOneSaved", "FUQOneDiff",
               "FUQOnePDirection", "FUQOnePShortfall", "FUQOnePooledVerdict", "FUQOnePooledPoint", "FUQOnePooledLower",
               "FUQOnePooledUpper", "FUQOnePooledSaved", "FUQTwoATwo", "FUQTwoAText", "FUQTwoAImage", "FUQTwoAInter",
               "FUQThreeA", "FUQThreeExceptionsA", "FUQFourA", "FUQFourChangedA", "FUNotRunA",
               "FUQOneEurosat", "FUQOneCifar", "FUQFourMarginA", "FUQOnePredicted",
               "FUQThreeBudgetLo", "FUQThreeBudgetFour", "FUQThreeChangedLo", "FUQThreeChangedFour",
               "FUQThreeUsefulFour", "FUQThreeHarmfulFour", "FUQThreeStepMin"]
    a_names += [f"FUQFiveA{FU_CONTROL_TOKENS[c]}{field}" for c in FU_CONTROLS for field in ("Point", "Dir")]
    a_values: dict[str, str] | None = None
    a = reports["A"]
    if a is not None:
        q1 = a["questions"]["Q1"]
        if "primary_equal_cell_macro" not in q1:  # analyze_ext reports Q1 as not evaluable unless all 021A cells ran
            raise RuntimeError(f"EXP-021A Q1 is {q1.get('verdict')}: {q1.get('reason')}; the follow-up text needs a decision")
        primary, pooled = q1["primary_equal_cell_macro"], q1["secondary_item_pooled"]
        decomposition = a["families"][pooled_key(a)]["decomposition"]["1"]
        a_values = {
            "FUQOneVerdict": FU_Q1_DISPLAY[primary["verdict"]],
            "FUQOnePoint": spts(primary["point_points"]),
            "FUQOneLower": spts(primary["ci95_unadjusted_lower"]),
            "FUQOneUpper": spts(primary["ci95_unadjusted_upper"]),
            "FUQOneSaved": spts(primary["reference_points"]),
            "FUQOneDiff": spts(primary["fresh_minus_saved_points"]),
            "FUQOnePDirection": fu_p(primary["p_direction"]),
            "FUQOnePShortfall": fu_p(primary["p_shortfall"]),
            "FUQOnePooledVerdict": FU_Q1_DISPLAY[pooled["verdict"]],
            "FUQOnePooledPoint": spts(pooled["point_points"]),
            "FUQOnePooledLower": spts(pooled["ci95_unadjusted_lower"]),
            "FUQOnePooledUpper": spts(pooled["ci95_unadjusted_upper"]),
            "FUQOnePooledSaved": spts(pooled["reference_points"]),
            "FUQTwoATwo": spts(decomposition["two_sided"]),
            "FUQTwoAText": spts(decomposition["text_only"]),
            "FUQTwoAImage": spts(decomposition["image_only"]),
            "FUQTwoAInter": spts(decomposition["interaction_point"]),
            "FUQThreeA": fu_conformance(a["questions"]["Q3"]["verdict"]),
            "FUQThreeExceptionsA": count(len(a["questions"]["Q3"]["exceptions"])),
            "FUQFourA": fu_conformance(a["questions"]["Q4"]["verdict"]),
            "FUQFourChangedA": count(int(a["questions"]["Q4"]["changed_projected_gap_draws"])),
            "FUNotRunA": count(len(a["not_run_cells"])),
            # descriptive per-dataset Q1 breakdown and the Q4 tie budget, both fields of the registered output
            "FUQOneEurosat": spts(q1["by_dataset_descriptive"]["eurosat__sigma0.25"]["equal_cell_macro_point"]),
            "FUQOneCifar": spts(q1["by_dataset_descriptive"]["cifar100__sigma0.25"]["equal_cell_macro_point"]),
            "FUQFourMarginA": fu_bound(a["questions"]["Q4"]["tie_budget"]["max_margin_bound"]),
            "FUQOnePredicted": FU_Q1_DISPLAY[q1["registered_prediction"]],
        }
        # Q3 step response on the audit items (Table tab:followup-steps): the flip budget is necessary but loose
        steps_a = _fu_step_totals(a, int(followup["A"]["certification"]["confirmation_draws"]))
        step_min = min(step for _, step in steps_a)
        step_four = 0.04
        if (FU_STEP_FAMILIES[0][0], step_four) not in steps_a:
            raise RuntimeError("the Q3 step response has no step 0.04")
        at = lambda family, step: steps_a[(family, step)]  # noqa: E731
        a_values["FUQThreeStepMin"] = num(step_min)
        a_values["FUQThreeBudgetLo"] = one_decimal_pct(at(FU_STEP_FAMILIES[0][0], step_min)["budget"]
                                                       / at(FU_STEP_FAMILIES[0][0], step_min)["draws"])
        a_values["FUQThreeBudgetFour"] = one_decimal_pct(at(FU_STEP_FAMILIES[0][0], step_four)["budget"]
                                                         / at(FU_STEP_FAMILIES[0][0], step_four)["draws"])
        a_values["FUQThreeChangedLo"] = pct(min(at(f, step_min)["changed"] / at(f, step_min)["draws"]
                                                for f, _ in FU_STEP_FAMILIES))
        a_values["FUQThreeChangedFour"] = pct(max(at(f, step_four)["changed"] / at(f, step_four)["draws"]
                                                  for f, _ in FU_STEP_FAMILIES))
        a_values["FUQThreeUsefulFour"] = count(at(FU_STEP_FAMILIES[0][0], step_four)["useful"])
        a_values["FUQThreeHarmfulFour"] = count(at(FU_STEP_FAMILIES[0][0], step_four)["harmful"])
        q5_a = a["questions"]["Q5"][pooled_key(a)]
        for control in FU_CONTROLS:
            a_values[f"FUQFiveA{FU_CONTROL_TOKENS[control]}Point"] = spts(q5_a[control]["point_points"])
            a_values[f"FUQFiveA{FU_CONTROL_TOKENS[control]}Dir"] = fu_direction(q5_a[control]["direction"])
    macros.section("EXP-021A results (analysis/exp021/021a/analysis_ext.json; registered analysis)")
    fill(a_names, a_values, "EXP-021A registered analysis output not present yet")

    # EXP-021B: Q6 (primary), Q2, Q5 per pooled family; Q3/Q4 conformance; Q7 regime; Q8 orientation.
    b = reports["B"]
    forest_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for sigma_text, token in FU_SIGMA_TOKENS:
        names = [f"FUGR{token}{field}" for field in ("Point", "Lower", "Upper", "Dir", "Mag")]
        names += [f"FUQSix{token}{field}" for field in ("N", "Gain", "Loss", "Equiv", "Incon", "Pos", "Neg", "Unres",
                                                         "MaxPoint", "MaxId", "MinPoint", "MinId")]
        names += [f"FUQTwo{token}{field}" for field in ("Text", "Image", "Inter")]
        names += [f"FUQFive{token}{FU_CONTROL_TOKENS[c]}{field}" for c in FU_CONTROLS for field in ("Point", "Dir")]
        names += [f"FUQSeven{token}{field}" for field in ("SmoothedLo", "SmoothedHi", "StdCALo", "StdCAHi")]
        names += [f"FUGR{token}Macro{field}" for field in ("Point", "Lower", "Upper")]
        names += [f"FUQSix{token}{sign}Ids" for sign in ("Positive", "Negative")]
        values: dict[str, str] | None = None
        if b is not None:
            key = pooled_key(b, sigma_text)
            family = b["families"][key]
            contrasts = family["contrasts"]
            forest_rows[sigma_text] = contrasts
            q6 = b["questions"]["Q6"][key]
            primary_ids = list(followup["B"]["inference"]["primary_family"]["contrasts"])
            if set(q6) != set(primary_ids):
                raise RuntimeError(f"Q6 at sigma {sigma_text} does not cover exactly the registered primary family")
            gr = q6[FU_TWO_SIDED]
            magnitudes = [q6[c]["magnitude"] for c in primary_ids]
            directions = [q6[c]["direction"] for c in primary_ids]
            shared = {c: row for c, row in contrasts.items() if not c.startswith("control__")}
            best = max(shared, key=lambda c: (shared[c]["point_points"], c))
            worst = min(shared, key=lambda c: (shared[c]["point_points"], c))
            decomposition = family["decomposition"]["1"]
            q5 = b["questions"]["Q5"][key]
            regime = [row for row in b["questions"]["Q7"]["identity_regime"] if fu_sigma_of_cell(row["cell_id"]) == sigma_text]
            if not regime:
                raise RuntimeError(f"Q7 lists no identity cell at sigma {sigma_text}")
            values = {
                f"FUGR{token}Point": spts(gr["point"]),
                f"FUGR{token}Lower": spts(gr["ci95_unadjusted"][0]),
                f"FUGR{token}Upper": spts(gr["ci95_unadjusted"][1]),
                f"FUGR{token}Dir": fu_direction(gr["direction"]),
                f"FUGR{token}Mag": FU_MAGNITUDE_DISPLAY[gr["magnitude"]],
                f"FUQSix{token}N": count(len(primary_ids)),
                f"FUQSix{token}Gain": count(magnitudes.count("gain_at_least_sesoi")),
                f"FUQSix{token}Loss": count(magnitudes.count("loss_at_least_sesoi")),
                f"FUQSix{token}Equiv": count(magnitudes.count("practically_equivalent")),
                f"FUQSix{token}Incon": count(magnitudes.count("inconclusive")),
                f"FUQSix{token}Pos": count(directions.count("positive")),
                f"FUQSix{token}Neg": count(directions.count("negative")),
                f"FUQSix{token}Unres": count(directions.count("unresolved")),
                f"FUQSix{token}MaxPoint": spts(shared[best]["point_points"]),
                f"FUQSix{token}MaxId": fu_display(best),
                f"FUQSix{token}MinPoint": spts(shared[worst]["point_points"]),
                f"FUQSix{token}MinId": fu_display(worst),
                f"FUQTwo{token}Text": spts(decomposition["text_only"]),
                f"FUQTwo{token}Image": spts(decomposition["image_only"]),
                f"FUQTwo{token}Inter": spts(decomposition["interaction_point"]),
                f"FUQSeven{token}SmoothedLo": pct(min(float(row["smoothed_accuracy"]) for row in regime)),
                f"FUQSeven{token}SmoothedHi": pct(max(float(row["smoothed_accuracy"]) for row in regime)),
                f"FUQSeven{token}StdCALo": pct(min(float(row["primary_standard_ca"]) for row in regime)),
                f"FUQSeven{token}StdCAHi": pct(max(float(row["primary_standard_ca"]) for row in regime)),
            }
            for control in FU_CONTROLS:
                values[f"FUQFive{token}{FU_CONTROL_TOKENS[control]}Point"] = spts(q5[control]["point_points"])
                values[f"FUQFive{token}{FU_CONTROL_TOKENS[control]}Dir"] = fu_direction(q5[control]["direction"])
            # the audit's estimand (equal-cell) for two-sided centering, reported beside the item-pooled primary
            values[f"FUGR{token}MacroPoint"] = spts(contrasts[FU_TWO_SIDED]["macro_point_points"])
            values[f"FUGR{token}MacroLower"] = spts(contrasts[FU_TWO_SIDED]["macro_ci95_unadjusted_lower"])
            values[f"FUGR{token}MacroUpper"] = spts(contrasts[FU_TWO_SIDED]["macro_ci95_unadjusted_upper"])
            # the primary contrasts whose Holm-adjusted direction is resolved, named for the text
            for sign in ("positive", "negative"):
                chosen = [c for c in primary_ids if q6[c]["direction"] == sign]
                values[f"FUQSix{token}{sign.capitalize()}Ids"] = english_list([fu_display(c) for c in chosen])
        macros.section(f"EXP-021B results at sigma {sigma_text} (analysis/exp021/021b/analysis_ext.json)")
        fill(names, values, "EXP-021B registered analysis output not present yet")

    b_names = ["FUQThreeB", "FUQThreeExceptionsB", "FUQFourB", "FUQFourChangedB", "FUNotRunB", "FUQFourMarginB",
               "FUQFiveClassLo", "FUQFiveClassHi", "FUQSixMaxBoth", "FUQSevenImagenetteSmoothedLo",
               "FUQSevenImagenetteSmoothedHi", "FUQSevenImagenetteStdCALo"]
    b_names += [f"FUQEight{token}{r}" for token in FU_MODEL_TOKENS.values() for r in ("Quarter", "Half")]
    b_names += [f"FUGR{name}{token}{field}" for name in FU_DATASET_TOKENS.values() for _, token in FU_SIGMA_TOKENS
                for field in ("Point", "Mag")]
    b_names += [f"FUQSixDs{label}{field}" for label in ("Max", "Min")
                for field in ("Point", "Id", "Data", "Sigma", "Mag", "CleanFlag", "CleanDrop")]
    b_names += [f"FUQFiveClass{name}{end}" for name in FU_DATASET_TOKENS.values() for end in ("Lo", "Hi")]
    b_names += ["FUQSevenWrongOverCorrect"]
    # review of 2026-09-28: per-backbone two-sided changes on sealed EuroSAT (descriptive, from the registered cells
    # table), the clean-accuracy cost of the largest dataset-family gain, the learned shared translation on EuroSAT,
    # the non-collapsed CIFAR-100 regime, and the clean-accuracy change of every supervised control
    b_names += [f"FUGREurosat{token}{model}" for _, token in FU_SIGMA_TOKENS for model in FU_MODEL_TOKENS.values()]
    b_names += ["FUQSixDsMaxFamilyClean", "FUQSevenCifarLowSmoothedLo", "FUQSevenCifarLowSmoothedHi"]
    b_names += [f"FUQFiveSharedEurosat{token}" for _, token in FU_SIGMA_TOKENS]
    b_names += [f"FUQFive{FU_CONTROL_TOKENS[c]}Clean" for c in FU_CONTROLS]
    b_names += [f"FUQFive{FU_CONTROL_TOKENS[c]}CleanFlagged" for c in FU_CONTROLS]
    b_names += [f"FUQTwo{token}Inter{end}" for _, token in FU_SIGMA_TOKENS for end in ("Lower", "Upper")]
    # reviews of V11: how the registered primary intervals sit against the +-2-point margin (descriptive, unadjusted)
    b_names += ["FUQSixMaxAbsBound", "FUQSixMaxUpper"]
    cells_rows_for_macros = (sources.read_csv(FU_CELLS_CSV_REL)
                             if b is not None and (sources.root / FU_CELLS_CSV_REL).is_file() else None)
    b_values: dict[str, str] | None = None
    if b is not None:
        b_values = {
            "FUQThreeB": fu_conformance(b["questions"]["Q3"]["verdict"]),
            "FUQThreeExceptionsB": count(len(b["questions"]["Q3"]["exceptions"])),
            "FUQFourB": fu_conformance(b["questions"]["Q4"]["verdict"]),
            "FUQFourChangedB": count(int(b["questions"]["Q4"]["changed_projected_gap_draws"])),
            "FUNotRunB": count(len(b["not_run_cells"])),
            "FUQFourMarginB": fu_bound(b["questions"]["Q4"]["tie_budget"]["max_margin_bound"]),
        }
        # range of the three class-specific supervised controls over both pooled families (Q5)
        class_specific = [float(b["questions"]["Q5"][pooled_key(b, sigma_text)][control]["point_points"])
                          for sigma_text, _ in FU_SIGMA_TOKENS
                          for control in ("control__lowrank_tangent_r8", "control__fewshot_prompt_bank",
                                          "control__noisy_class_mean")]
        b_values["FUQFiveClassLo"], b_values["FUQFiveClassHi"] = spts(min(class_specific)), spts(max(class_specific))
        # the largest point estimate of any unsupervised bank over both pooled families (Q6 families, all contrasts)
        b_values["FUQSixMaxBoth"] = spts(max(float(row["point_points"]) for sigma_text, _ in FU_SIGMA_TOKENS
                                             for candidate, row in b["families"][pooled_key(b, sigma_text)]["contrasts"].items()
                                             if not candidate.startswith("control__")))
        # the unadjusted 95% intervals of the registered primary contrasts, pooled, at both noise levels: the largest
        # absolute bound and the largest upper bound (descriptive; the registered classification stays Holm-adjusted TOST)
        primary_bounds = [tuple(float(value) for value in b["questions"]["Q6"][pooled_key(b, sigma_text)][cid]["ci95_unadjusted"])
                          for sigma_text, _ in FU_SIGMA_TOKENS
                          for cid in followup["B"]["inference"]["primary_family"]["contrasts"]]
        b_values["FUQSixMaxAbsBound"] = f"{max(max(abs(lower), abs(upper)) for lower, upper in primary_bounds):.2f}"
        b_values["FUQSixMaxUpper"] = spts(max(upper for _, upper in primary_bounds))
        # the non-collapsed cells of Q7: Imagenette at both noise levels
        imagenette = [row for row in b["questions"]["Q7"]["identity_regime"] if "__imagenette__" in row["cell_id"]]
        if not imagenette:
            raise RuntimeError("Q7 lists no Imagenette identity cell")
        b_values["FUQSevenImagenetteSmoothedLo"] = pct(min(float(row["smoothed_accuracy"]) for row in imagenette))
        b_values["FUQSevenImagenetteSmoothedHi"] = pct(max(float(row["smoothed_accuracy"]) for row in imagenette))
        b_values["FUQSevenImagenetteStdCALo"] = pct(min(float(row["primary_standard_ca"]) for row in imagenette))
        # dataset heterogeneity (review of 2026-09-28): the largest gain and loss among the registered primary contrasts in
        # the secondary dataset families, with their registered magnitude classes, and the clean-rule flag of the gain
        primary_ids = list(followup["B"]["inference"]["primary_family"]["contrasts"])
        extremes = [(float(b["questions"]["Q6"][f"{dataset}__sigma{sigma_text}"][cid]["point"]), dataset, sigma_text, cid)
                    for dataset in FU_DATASET_TOKENS for sigma_text, _ in FU_SIGMA_TOKENS for cid in primary_ids]
        for label, (point, dataset, sigma_text, cid) in (("Max", max(extremes)), ("Min", min(extremes))):
            row = b["questions"]["Q6"][f"{dataset}__sigma{sigma_text}"][cid]
            b_values[f"FUQSixDs{label}Point"] = spts(point)
            b_values[f"FUQSixDs{label}Id"] = fu_display(cid)
            b_values[f"FUQSixDs{label}Data"] = FU_DATASET_TEXT[dataset]
            b_values[f"FUQSixDs{label}Sigma"] = num(float(sigma_text))
            b_values[f"FUQSixDs{label}Mag"] = FU_MAGNITUDE_DISPLAY[row["magnitude"]]
            gate_entry = (b.get("clean_gate") or {}).get("candidates", {}).get(cid, {})
            b_values[f"FUQSixDs{label}CleanFlag"] = "flags" if gate_entry.get("flagged") else "does not flag"
            b_values[f"FUQSixDs{label}CleanDrop"] = f"{float(gate_entry.get('macro_drop_points', 0.0)):.2f}"
        # the class-specific supervised controls per dataset, over both noise levels (Q5 is pooled; these are its families)
        for dataset, name in FU_DATASET_TOKENS.items():
            points = [float(b["families"][f"{dataset}__sigma{sigma_text}"]["contrasts"][control]["point_points"])
                      for sigma_text, _ in FU_SIGMA_TOKENS
                      for control in ("control__lowrank_tangent_r8", "control__fewshot_prompt_bank", "control__noisy_class_mean")]
            b_values[f"FUQFiveClass{name}Lo"], b_values[f"FUQFiveClass{name}Hi"] = spts(min(points)), spts(max(points))
        # cells in which the uncorrected model certifies a wrong class more often than the correct one at r = sigma (Q7)
        if cells_rows_for_macros is not None:
            identity_rows = [row for row in cells_rows_for_macros if str(row["candidate_id"]) == "no_correction"]
            b_values["FUQSevenWrongOverCorrect"] = count(sum(
                float(row["primary_certified_wrong_rate"]) > float(row["primary_standard_ca"]) for row in identity_rows))
        if cells_rows_for_macros is None:
            raise RuntimeError(f"{FU_CELLS_CSV_REL} is missing next to the EXP-021B analysis output")
        by_cell = {(str(row["cell_id"]), str(row["candidate_id"])): row for row in cells_rows_for_macros}
        for sigma_text, token in FU_SIGMA_TOKENS:
            for model_id, model in FU_MODEL_TOKENS.items():
                cell_id = f"{model_id}__eurosat_sealed__sigma{sigma_text}"
                identity_row, two_sided_row = by_cell[(cell_id, "no_correction")], by_cell[(cell_id, FU_TWO_SIDED)]
                b_values[f"FUGREurosat{token}{model}"] = spts(100.0 * (float(two_sided_row["primary_standard_ca"])
                                                                       - float(identity_row["primary_standard_ca"])))
        # the largest dataset-family gain (FUQSixDsMax*): mean clean-accuracy change over that family's cells
        best_point, best_dataset, best_sigma, best_id = max(extremes)
        family_cells = [cell_id for cell_id, candidate in by_cell
                        if candidate == "no_correction" and f"__{best_dataset}__sigma{best_sigma}" in cell_id]
        if not family_cells:
            raise RuntimeError(f"no cells for the family {best_dataset} at sigma {best_sigma}")
        b_values["FUQSixDsMaxFamilyClean"] = spts(100.0 * sum(
            float(by_cell[(cell_id, best_id)]["clean_accuracy"]) - float(by_cell[(cell_id, "no_correction")]["clean_accuracy"])
            for cell_id in family_cells) / len(family_cells))
        cifar_low = [row for row in b["questions"]["Q7"]["identity_regime"]
                     if "__cifar100_test__" in row["cell_id"] and fu_sigma_of_cell(row["cell_id"]) == FU_SIGMA_TOKENS[0][0]]
        if not cifar_low:
            raise RuntimeError("Q7 lists no CIFAR-100 identity cell at the lower noise level")
        b_values["FUQSevenCifarLowSmoothedLo"] = pct(min(float(row["smoothed_accuracy"]) for row in cifar_low))
        b_values["FUQSevenCifarLowSmoothedHi"] = pct(max(float(row["smoothed_accuracy"]) for row in cifar_low))
        for sigma_text, token in FU_SIGMA_TOKENS:
            family = b["families"][f"eurosat_sealed__sigma{sigma_text}"]
            b_values[f"FUQFiveSharedEurosat{token}"] = spts(
                family["contrasts"]["control__learned_shared_translation_tangent"]["point_points"])
            decomposition = b["families"][pooled_key(b, sigma_text)]["decomposition"]["1"]
            b_values[f"FUQTwo{token}InterLower"] = spts(decomposition["interaction_ci95_unadjusted"][0])
            b_values[f"FUQTwo{token}InterUpper"] = spts(decomposition["interaction_ci95_unadjusted"][1])
        gate = b["clean_gate"]["candidates"]
        for control in FU_CONTROLS:
            b_values[f"FUQFive{FU_CONTROL_TOKENS[control]}Clean"] = spts(-float(gate[control]["macro_drop_points"]))
            b_values[f"FUQFive{FU_CONTROL_TOKENS[control]}CleanFlagged"] = count(len(gate[control]["flagged_cells"]))
        # two-sided centering in the secondary dataset families (Q6, reported, not adjudicating)
        for dataset, name in FU_DATASET_TOKENS.items():
            for sigma_text, token in FU_SIGMA_TOKENS:
                family = f"{dataset}__sigma{sigma_text}"
                b_values[f"FUGR{name}{token}Point"] = spts(b["families"][family]["contrasts"][FU_TWO_SIDED]["point_points"])
                b_values[f"FUGR{name}{token}Mag"] = FU_MAGNITUDE_DISPLAY[b["questions"]["Q6"][family][FU_TWO_SIDED]["magnitude"]]
        eight = {row["cell_id"]: row for row in b["questions"]["Q8"]["imagenette_identity"]}
        for model_id, token in FU_MODEL_TOKENS.items():
            rows = [row for cell_id, row in eight.items() if cell_id.startswith(model_id + "__")]
            if len(rows) != 1:
                raise RuntimeError(f"Q8 lists {len(rows)} Imagenette identity rows for {model_id}")
            b_values[f"FUQEight{token}Quarter"] = pct(float(rows[0]["standard_ca@0.25"]))
            b_values[f"FUQEight{token}Half"] = pct(float(rows[0]["standard_ca@0.5"]))
    macros.section("EXP-021B conformance, regime orientation and coverage")
    fill(b_names, b_values, "EXP-021B registered analysis output not present yet")

    # Paired verdicts as one phrase, so the text never reads "unresolved and unresolved" (writing review 2026-09-28).
    pair_names = ["FUGRClasses", "FUQThreeSummary", "FUQFourSummary"]
    pair_values: dict[str, str] | None = None
    if a is not None and b is not None:
        classes = []
        for sigma_text, _ in FU_SIGMA_TOKENS:
            row = b["questions"]["Q6"][pooled_key(b, sigma_text)][FU_TWO_SIDED]
            classes.append((fu_direction(row["direction"]), FU_MAGNITUDE_DISPLAY[row["magnitude"]]))
        if classes[0] == classes[1]:
            gr_classes = f"{classes[0][0]} in direction and {classes[0][1]} in magnitude at both noise levels"
        else:
            gr_classes = "; ".join(f"{direction} in direction and {magnitude} in magnitude at $\\sigma={sigma_text}$"
                                   for (direction, magnitude), (sigma_text, _) in zip(classes, FU_SIGMA_TOKENS))

        def checks(question: str) -> str:
            verdicts = [fu_conformance(report["questions"][question]["verdict"]) for report in (a, b)]
            if question == "Q3":
                exceptions = [len(report["questions"]["Q3"]["exceptions"]) for report in (a, b)]
                if verdicts[0] == verdicts[1] and exceptions == [0, 0]:
                    return f"{verdicts[0]} in both Follow-ups, with no exception"
                return (f"{verdicts[0]} in Follow-up A and {verdicts[1]} in Follow-up B, with {count(exceptions[0])} and "
                        f"{count(exceptions[1])} exceptions")
            if verdicts[0] == verdicts[1]:
                return f"{verdicts[0]} in both Follow-ups"
            return f"{verdicts[0]} in Follow-up A and {verdicts[1]} in Follow-up B"

        pair_values = {"FUGRClasses": gr_classes, "FUQThreeSummary": checks("Q3"), "FUQFourSummary": checks("Q4")}
    macros.section("EXP-021 paired verdicts as phrases")
    fill(pair_names, pair_values, "EXP-021A or EXP-021B registered analysis output not present yet")

    # EXP-021C: Q9 budget comparison (descriptive). analyze_budget reports one row per (cell, bank); the macros sum
    # the registered cells, and the most-affected bank is the one with the most changed certificates over all cells.
    c_names = ["FUQNineItems", "FUQNineIdFour", "FUQNineIdFull", "FUQNineIdGained", "FUQNineIdLost", "FUQNineAllCerts",
               "FUQNineAllChanged", "FUQNineMaxChanged", "FUQNineMaxId", "FUQNineIdentity",
               "FUQNinePrefixIdentity", "FUQNineInputIdentity"]
    c_values: dict[str, str] | None = None
    c = reports["C"]
    if c is not None:
        q9 = c["questions"]["Q9"]
        rows = q9["rows"]
        registered_cells = sorted(cell["cell_id"] for cell in followup["C"]["cells"])
        if sorted({row["cell_id"] for row in rows}) != registered_cells:
            raise RuntimeError("Q9 rows do not cover exactly the registered EXP-021C cells")
        identity = [row for row in rows if row["candidate_id"] == "no_correction"]
        if sorted(row["cell_id"] for row in identity) != registered_cells:
            raise RuntimeError(f"Q9 lists {len(identity)} identity rows; expected one per registered cell")
        changed: dict[str, int] = {}
        for row in rows:
            changed[row["candidate_id"]] = changed.get(row["candidate_id"], 0) + int(row["changed"])
        most = max(changed, key=lambda candidate: (changed[candidate], candidate))
        c_values = {
            "FUQNineItems": count(sum(int(row["items"]) for row in identity)),
            "FUQNineIdFour": count(sum(int(row["certified_correct_4096"]) for row in identity)),
            "FUQNineIdFull": count(sum(int(row["certified_correct_100000"]) for row in identity)),
            "FUQNineIdGained": count(sum(int(row["gained_with_100000"]) for row in identity)),
            "FUQNineIdLost": count(sum(int(row["lost_with_100000"]) for row in identity)),
            "FUQNineAllCerts": count(sum(int(row["items"]) for row in rows)),
            "FUQNineAllChanged": count(sum(changed.values())),
            "FUQNineMaxChanged": count(changed[most]),
            "FUQNineMaxId": fu_display(most),
            "FUQNineIdentity": "yes" if q9.get("prefix_identity_exact_everywhere") is True else "no",
            # the two registered identities, reported separately (confirmation prefix; full certificate input)
            "FUQNinePrefixIdentity": "yes" if q9.get("confirmation_prefix_identity_everywhere") is True else "no",
            "FUQNineInputIdentity": "yes" if q9.get("certificate_input_identity_everywhere") is True else "no",
        }
    macros.section("EXP-021C budget comparison (analysis/exp021/021c/analysis_ext.json)")
    fill(c_names, c_values, "EXP-021C registered analysis output not present yet")

    # Compute statement (D-150): the measured sampling windows of both hosts (scripts/exp021_compute_summary.py).
    compute_values = None
    if (sources.root / FU_COMPUTE_REL).is_file():
        compute = sources.read_json(FU_COMPUTE_REL)
        if compute.get("schema_version") != "satml2027ext.compute_summary.v1":
            raise RuntimeError(f"{FU_COMPUTE_REL} has an unexpected schema")
        compute_values = {"FUMainHours": f"{float(compute['main_host']['hours']):.1f}",
                          "FUExtraHours": f"{float(compute['extra_host']['hours']):.1f}"}
    macros.section("EXP-021 compute (analysis/exp021/compute_summary.json)")
    fill(["FUMainHours", "FUExtraHours"], compute_values, "EXP-021 compute summary not present yet")

    # Forest figure (D-146): the same pooled-family contrasts, or a placeholder flag while the output is absent.
    sesoi = float(followup["B"]["inference"]["practical_magnitude"]["sesoi_points"])
    tables.update(fu_forest_files(forest_rows if b is not None else None, sesoi))

    # Descriptive radius curves (D-146): only if they were computed from this very analysis output.
    curves = None
    if b is not None and (sources.root / FU_CURVES_REL).is_file():
        curves = sources.read_json(FU_CURVES_REL)
        expected_banks = [bank for bank, _ in FU_CURVE_COLUMNS]
        if (curves.get("schema_version") != FU_CURVES_SCHEMA or curves.get("descriptive") is not True
                or curves.get("registration_sha256") != followup["B"]["registration_sha256"]
                or curves.get("banks") != expected_banks or int(curves.get("registered_value_checks", 0)) <= 0):
            raise RuntimeError(f"{FU_CURVES_REL} is not the descriptive curve output of the EXP-021B registration")
        if curves.get("analysis_sha256") != sources.hashes.get(FU_ANALYSIS_RELS["B"]):
            raise RuntimeError(f"{FU_CURVES_REL} was computed from another EXP-021B analysis output")
    tables.update(fu_curves_files(curves))

    # Appendix: secondary dataset families and the uncorrected model's outcome composition (D-148).
    tables["table_followup_datasets.tex"] = fu_dataset_table(b, followup["B"], sources)
    cells_rows = None
    if b is not None:
        if not (sources.root / FU_CELLS_CSV_REL).is_file():
            raise RuntimeError(f"{FU_CELLS_CSV_REL} is missing next to the EXP-021B analysis output")
        cells_rows = sources.read_csv(FU_CELLS_CSV_REL)
    tables.update(fu_outcome_files(cells_rows, b, followup["B"]))

    # Tables: the pooled primary family (both estimands), the registered outputs the text summarizes, and the Q3 step
    # response (review of 2026-09-28: every registered output is reported somewhere in the paper).
    tables["table_followup.tex"] = fu_followup_table(b, followup["B"], sources)
    tables["table_followup_registered.tex"] = fu_registered_table(a, b, sources)
    tables["table_followup_steps.tex"] = fu_steps_table(a, b, followup, sources)


def write_assets(assets: Assets, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    header = [
        f"% Generated by {SCRIPT_REL}; do not edit by hand.",
        f"% Specification: {CONTRACT_REL}",
        "% Percentages are percentage points with two decimals; differences carry a sign;",
        "% counts use thousands separators; \\textbf{??} marks a macro that could not be",
        "% derived (see the MISSING section of sources.json).",
        "% Sources (SHA-256):",
    ]
    header.extend(f"%   {rel}  {digest}" for rel, digest in assets.sources.ordered())
    (output_dir / "macros.tex").write_text(assets.macros.render(header), encoding="utf-8", newline="\n")
    for name, content in assets.tables.items():
        (output_dir / name).write_text(content, encoding="utf-8", newline="\n")
    payload = {
        "schema_version": "satml2027.paper_assets.v1",
        "generator": SCRIPT_REL,
        "contract": CONTRACT_REL,
        "artifacts": dict(assets.sources.ordered()),
        "derivations": json_ready(assets.notes),
        "macros": assets.macros.values,
        "tables": sorted(assets.tables),
        "MISSING": assets.macros.missing,
    }
    (output_dir / "sources.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def generate(root: Path, output_dir: Path) -> Assets:
    assets = build(root)
    write_assets(assets, output_dir)
    return assets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--partition-worker", action="store_true", help="internal: print the per-example partition as JSON")
    parser.add_argument("--radius", type=float, default=TARGET_RADIUS)
    parser.add_argument("--cells", type=str, default="")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.partition_worker:
        cell_ids = [value for value in args.cells.split(",") if value] or sorted(
            path.name for path in (root / CELLS_REL).iterdir() if (path / SUFFICIENT_STATISTICS_NAME).is_file()
        )
        sys.stdout.write(json.dumps(compute_partition(root, cell_ids, args.radius)))
        return 0
    output_dir = (args.output_dir or (root / DEFAULT_OUTPUT_REL)).resolve()
    assets = generate(root, output_dir)
    summary = {
        "output_dir": str(output_dir),
        "macros": len(assets.macros.values),
        "missing": assets.macros.missing,
        "artifacts_hashed": len(assets.sources.hashes),
        "tables": sorted(assets.tables),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
