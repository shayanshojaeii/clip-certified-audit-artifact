#!/usr/bin/env python3
"""Day 1 - secondary sensitivity analyses of Addendum V1 (items A2, A3 and A4).

The addendum was frozen before any EXP-019 outcome existed (D-118), but the
outcomes were opened on 2026-09-25 before any approval (D-127). Its approval is
therefore a retrospective review (amendment of 2026-09-26, D-135), and every
output of this script is labelled "reviewed retrospectively after unblinding".

This script consumes exactly what the registered primary analysis
(`d1_07_analyze.py`) consumes and writes: the merged `{cell_id}__merged.npz`
files and the primary `contrasts.json`.  It recomputes the registered per-item
outcomes with the same `certify_from_counts` call, rebuilds the 21 paired
differences versus `no_correction` over the same primary cells, verifies that
its point estimates equal the ones in `contrasts.json`, and then adds three
secondary quantities that the addendum fixed before any result was inspected:

  A3  finite-stratum sensitivity: the Rao-Wu rescaled stratified bootstrap.
      Each (dataset, class) stratum's centered replicate contribution is
      multiplied by sqrt(n_s / (n_s - 1)); the registered seed and replicate
      count are reused; the generator is PCG64DXSM (Gate-N2 convention).  The
      unscaled replicate set is reported beside it as a Monte-Carlo agreement
      diagnostic for the registered critical value, together with the simple
      global inflation bound max_s sqrt(n_s / (n_s - 1)) x critical value.
  A4  the item-pooled contrast sum_c n_c delta_c / sum_c n_c beside the
      equal-cell primary for every contrast (descriptive).
  A2  the X3 / X4 evaluation from the registered band and the practical-region
      reporting flags.

The registered band written by `d1_07_analyze.py` remains primary; nothing here
changes it.  Torch-free (numpy + scipy through `common.certify`).

Fail-closed guard (see `verify_addendum_approval`): the script refuses to run
if the retired outcome-blind approval
`research/SATML2027_OUTCOME_BLIND_ANALYSIS_ADDENDUM_V1.approved.json` exists,
and otherwise unless the retrospective review
`research/SATML2027_OUTCOME_BLIND_ANALYSIS_ADDENDUM_V1.retrospective_review.json`
exists with the required status and binds the SHA-256 of the addendum, of the
retrospective amendment and of this file, records `outcome_blind_at_approval`
false, `outcomes_accessed_before_review` true and the recorded unblinding
(D-127, 2026-09-25) with a review date not before it, sets every
claim/sampling authority false, and the output does not exist yet.  Importing
the module and unit-testing its pure functions does not authorize execution on
a real result.

Usage:
  python satml2027/day1/d1_09_sensitivity_analysis.py \
      --merged results/satml2027/merged/EXP-20260920-019A \
      --registration configs/satml2027/exp-20260920-019a.json \
      --primary analysis/019a
  -> writes analysis/019a/sensitivity_v1.json
"""

from __future__ import annotations

import datetime as dt

import argparse
import math
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.certify import certify_from_counts  # noqa: E402
from common.hashing import canonical_json_hash, read_json, sha256_file, write_json_atomic  # noqa: E402
from day1 import d1_07_analyze as primary_analysis  # noqa: E402

# Retrospective amendment (2026-09-26, D-135; reviewer-bundle V3 audit, G1): the EXP-019
# outcomes were opened on 2026-09-25 before any approval (D-127), so the outcome-blind
# approval of addendum section 0.1 is retired. Only a retrospective review (schema V2,
# research/SATML2027_OUTCOME_BLIND_ANALYSIS_ADDENDUM_V1_RETROSPECTIVE_AMENDMENT_2026-09-26.md)
# authorizes this secondary sensitivity analysis. The addendum V1 file itself is unchanged.
SCHEMA_VERSION = "satml2027.addendum_v1_sensitivity.v2"
ADDENDUM_RELATIVE_PATH = Path("research") / "SATML2027_OUTCOME_BLIND_ANALYSIS_ADDENDUM_V1.md"
RETIRED_APPROVAL_RELATIVE_PATH = Path("research") / "SATML2027_OUTCOME_BLIND_ANALYSIS_ADDENDUM_V1.approved.json"
RETROSPECTIVE_AMENDMENT_RELATIVE_PATH = Path("research") / "SATML2027_OUTCOME_BLIND_ANALYSIS_ADDENDUM_V1_RETROSPECTIVE_AMENDMENT_2026-09-26.md"
APPROVAL_RELATIVE_PATH = Path("research") / "SATML2027_OUTCOME_BLIND_ANALYSIS_ADDENDUM_V1.retrospective_review.json"
REQUIRED_APPROVAL_STATUS = "INDEPENDENTLY_REVIEWED_RETROSPECTIVELY_ADDENDUM_V1"
UNBLINDING_DECISION = "D-127"
UNBLINDING_DATE = "2026-09-25"
REQUIRED_FALSE_AUTHORITY_FLAGS = (
    "new_sampling_authorized",
    "equivalence_claim_authorized",
    "confirmatory_claim_authorized",
    "causal_claim_authorized",
    "final_test_authorized",
)
OUTPUT_NAME = "sensitivity_v1.json"
GENERATOR_NAME = "PCG64DXSM"
CONFIDENCE_LEVEL = 0.95
QUANTILE_METHOD = "higher"
BAND_METHOD = "shared_item_multiplicity_dataset_class_stratified_max_absolute_deviation"
PRIMARY_OUTCOME_TEXT = "standard certified accuracy at r = sigma"
X3_CONTROL_ID = "control__learned_shared_translation_tangent"
X3_COMPANION_ID = "control__learned_shared_translation_pure"
X4_PER_CLASS_CONTROL_IDS = ("control__noisy_class_mean", "control__lowrank_tangent_r8")
POINT_AGREEMENT_ATOL = 1e-12


# --------------------------------------------------------------------------- registration


def registered_candidate_family(registration: dict) -> dict[str, Any]:
    """Enumerate the 18 frozen-grid banks and the registered controls from the registration itself."""

    grid = registration["candidates"]["frozen_grid"]
    steps = [float(value) for value in grid["common_unit_direction_step_grid"]]
    coefficients = [float(value) for value in grid["gap_coefficient_grid"]]
    frozen = ["no_correction", "gr_clip_style_two_sided__coefficient_1"]
    frozen += [f"global_mean_centering__coefficient_{value:.12g}" for value in coefficients]
    frozen += [f"chowers_exact_projected_gap__coefficient_{value:.12g}" for value in coefficients]
    frozen += [f"clean_boundary_active__step_{value:.12g}" for value in steps]
    frozen += [f"cohen_aligned_noisy_margin__step_{value:.12g}" for value in steps]
    if len(set(frozen)) != len(frozen) or len(frozen) != int(grid["bank_count"]):
        raise RuntimeError("the registered frozen grid does not enumerate the declared bank count")
    controls = [f"control__{entry['control_id']}" for entry in registration["candidates"]["controls"]]
    if len(set(controls)) != len(controls) or set(controls) & set(frozen):
        raise RuntimeError("registered control identifiers collide")
    reference = primary_analysis.REFERENCE
    if reference not in frozen:
        raise RuntimeError("the reference bank is not part of the frozen grid")
    frozen_contrasts = sorted(candidate for candidate in frozen if candidate != reference)
    return {
        "reference": reference,
        "frozen_grid_contrasts": frozen_contrasts,
        "control_contrasts": sorted(controls),
        "contrasts": sorted(frozen_contrasts + controls),
        "all_banks": sorted(frozen + controls),
    }


def is_bridge_cell(cell: dict, registration: dict) -> bool:
    """Exactly the inline rule of d1_07_analyze.py: CIFAR-10 sigma=0.25 cells of EXP-019A."""

    return bool(
        cell["dataset_id"] == "cifar10"
        and math.isclose(float(cell["sigma"]), 0.25)
        and str(registration["experiment_id"]).endswith("019A")
    )


def verify_registration_hash(registration: dict) -> str:
    """Recompute the canonical registration hash exactly as the sampling worker does."""

    recorded = registration.get("registration_sha256")
    body = {key: value for key, value in registration.items() if key != "registration_sha256"}
    recomputed = canonical_json_hash(body)
    if recorded != recomputed:
        raise RuntimeError(f"registration hash mismatch: file says {recorded}, content hashes to {recomputed}")
    return str(recorded)


# --------------------------------------------------------------------------- approval guard


def verify_addendum_approval(root: Path, script_path: Path, output_path: Path) -> dict[str, Any]:
    """Retrospective review (schema V2) of the unchanged addendum V1; the outcome-blind form is retired."""

    addendum_path = root / ADDENDUM_RELATIVE_PATH
    approval_path = root / APPROVAL_RELATIVE_PATH
    amendment_path = root / RETROSPECTIVE_AMENDMENT_RELATIVE_PATH
    if not addendum_path.is_file():
        raise RuntimeError("the outcome-blind analysis addendum V1 is absent")
    if (root / RETIRED_APPROVAL_RELATIVE_PATH).is_file():
        raise RuntimeError(
            "an outcome-blind approval of the addendum V1 is present, but that form is retired: the outcomes were "
            f"opened on {UNBLINDING_DATE} before any approval ({UNBLINDING_DECISION}); only the retrospective review applies"
        )
    if not approval_path.is_file():
        raise RuntimeError("independent retrospective approval of the analysis addendum V1 is absent")
    if not amendment_path.is_file():
        raise RuntimeError("the retrospective amendment of the analysis addendum V1 is absent")
    approval = read_json(approval_path)
    if not isinstance(approval, dict):
        raise RuntimeError("approval file is not a JSON object")
    if approval.get("status") != REQUIRED_APPROVAL_STATUS:
        raise RuntimeError("the analysis addendum V1 is not independently approved by a retrospective review")
    if approval.get("addendum_sha256") != sha256_file(addendum_path):
        raise RuntimeError("approved addendum hash mismatch")
    if approval.get("retrospective_amendment_sha256") != sha256_file(amendment_path):
        raise RuntimeError("approved retrospective-amendment hash mismatch")
    if approval.get("sensitivity_source_sha256") != sha256_file(script_path):
        raise RuntimeError("approved sensitivity-source hash mismatch")
    if approval.get("outcome_blind_at_approval") is not False:
        raise RuntimeError("a retrospective review must record outcome_blind_at_approval as false")
    if approval.get("outcomes_accessed_before_review") is not True:
        raise RuntimeError("a retrospective review must record outcomes_accessed_before_review as true")
    if approval.get("unblinding_decision") != UNBLINDING_DECISION or approval.get("unblinding_date") != UNBLINDING_DATE:
        raise RuntimeError(f"the review must name the recorded unblinding ({UNBLINDING_DECISION}, {UNBLINDING_DATE})")
    review_date = approval.get("review_date")
    try:
        parsed = dt.date.fromisoformat(str(review_date))
    except ValueError as error:
        raise RuntimeError("review_date must be an ISO date YYYY-MM-DD") from error
    if parsed < dt.date.fromisoformat(UNBLINDING_DATE):
        raise RuntimeError("a retrospective review cannot predate the recorded unblinding")
    optional_bindings = {
        "tf32_adjudication_source_sha256": script_path.with_name("d1_10_tf32_preparation_adjudication.py"),
        "primary_analysis_source_sha256": Path(primary_analysis.__file__).resolve(),
    }
    for key, path in optional_bindings.items():
        recorded = approval.get(key)
        if recorded is not None and (not path.is_file() or recorded != sha256_file(path)):
            raise RuntimeError(f"approved {key} does not match the file on disk")
    for flag in REQUIRED_FALSE_AUTHORITY_FLAGS:
        if approval.get(flag) is not False:
            raise RuntimeError(f"approval must record {flag} as false")
    if output_path.exists():
        raise RuntimeError("sensitivity output already exists; overwrite is prohibited")
    return approval


# --------------------------------------------------------------------------- primary reconstruction


def load_primary_differences(merged_dir: Path, registration: dict) -> dict[str, Any]:
    """Rebuild d1_07's per-item paired differences from the merged files it consumed."""

    certification = registration["certification"]
    alpha = float(certification["alpha_per_example"])
    radii = list(certification["reported_radii"])
    family = registered_candidate_family(registration)
    reference = family["reference"]
    expected_ids = set(family["all_banks"])
    outcomes_by_bank: dict[str, list[np.ndarray]] = {candidate: [] for candidate in family["all_banks"]}
    cells: list[dict[str, Any]] = []
    bridge_cells: list[str] = []
    cell_key, class_key, item_key, dataset_key = [], [], [], []
    for cell in registration["cells"]:
        path = merged_dir / f"{cell['cell_id']}__merged.npz"
        if not path.is_file():
            raise FileNotFoundError(f"registered analysis cell is absent: {path}")
        if is_bridge_cell(cell, registration):
            bridge_cells.append(str(cell["cell_id"]))
            continue
        data = np.load(path, allow_pickle=False)
        candidate_ids = [str(value) for value in data["candidate_ids"]]
        if set(candidate_ids) != expected_ids or len(candidate_ids) != len(expected_ids):
            raise RuntimeError(f"{cell['cell_id']}: merged bank list differs from the registered 18 + 4 family")
        ground_truth = np.asarray(data["ground_truth"])
        item_ids = np.asarray(data["item_ids"]).astype(str)
        if len(item_ids) != int(cell["item_count"]) or len(ground_truth) != len(item_ids):
            raise RuntimeError(f"{cell['cell_id']}: merged item count differs from the registration")
        if len(set(item_ids.tolist())) != len(item_ids):
            raise RuntimeError(f"{cell['cell_id']}: duplicate item identifiers")
        sigma = float(cell["sigma"])
        primary_key = format(sigma, ".12g")
        for position, candidate_id in enumerate(candidate_ids):
            _, outcomes = certify_from_counts(
                raw_predictions=data["raw_predictions"][position],
                ground_truth=ground_truth,
                selection_counts=data["selection_counts"][position],
                confirmation_counts=data["confirmation_counts"][position],
                sigma=sigma,
                alpha=alpha,
                radii=sorted(set(radii + [sigma])),
            )
            outcomes_by_bank[candidate_id].append(outcomes[f"standard@{primary_key}"].astype(np.int8))
        cells.append({
            "cell_id": str(cell["cell_id"]),
            "model_id": str(cell["model_id"]),
            "dataset_id": str(cell["dataset_id"]),
            "sigma": sigma,
            "item_count": int(len(item_ids)),
            "class_count": int(cell["class_count"]),
            "primary_estimand_cell": True,
        })
        cell_key.append(np.full(len(ground_truth), str(cell["cell_id"])))
        class_key.append(ground_truth.astype(np.int64))
        item_key.append(item_ids)
        dataset_key.append(np.full(len(ground_truth), str(cell["dataset_id"])))
    if not cells:
        raise RuntimeError("no primary-estimand cell is present")
    reference_outcomes = np.concatenate(outcomes_by_bank[reference]).astype(np.float64)
    differences = {
        candidate_id: np.concatenate(parts).astype(np.float64) - reference_outcomes
        for candidate_id, parts in outcomes_by_bank.items()
        if candidate_id != reference
    }
    return {
        "family": family,
        "cells": cells,
        "bridge_cells_excluded": bridge_cells,
        "differences": differences,
        "cell_ids": np.concatenate(cell_key),
        "class_labels": np.concatenate(class_key),
        "item_ids": np.concatenate(item_key),
        "dataset_ids": np.concatenate(dataset_key),
    }


# --------------------------------------------------------------------------- strata and bootstrap


def build_strata(differences: dict[str, np.ndarray], cell_ids: np.ndarray, class_labels: np.ndarray,
                 item_ids: np.ndarray, dataset_ids: np.ndarray, contrast_ids: list[str]) -> dict[str, Any]:
    """Aggregate each shared (dataset, class, item) identity into its per-contrast macro contribution.

    The registered statistic is the equal-cell macro mean of paired differences.
    One identity contributes d_c(j, k) / (n_k C) from every primary cell k that
    contains it, so a stratified resample with multiplicities m_j gives
    delta*_c = sum_j m_j a_{j,c}.  Strata are the (dataset, class) groups used by
    the registered band; identities inside a stratum are ordered by item id.
    """

    matrix = np.stack([np.asarray(differences[candidate], dtype=np.float64) for candidate in contrast_ids], axis=1)
    cells = sorted(str(value) for value in np.unique(cell_ids))
    cell_count = len(cells)
    cell_sizes = {cell: int(np.sum(cell_ids == cell)) for cell in cells}
    row_weights = np.array([1.0 / (cell_sizes[str(cell)] * cell_count) for cell in cell_ids], dtype=np.float64)
    rows_by_identity: dict[tuple[str, int, str], list[int]] = {}
    for row, (dataset, label, item) in enumerate(zip(dataset_ids, class_labels, item_ids)):
        rows_by_identity.setdefault((str(dataset), int(label), str(item)), []).append(row)
    members_by_stratum: dict[tuple[str, int], list[tuple[str, int, str]]] = {}
    for identity in rows_by_identity:
        members_by_stratum.setdefault((identity[0], identity[1]), []).append(identity)
    strata = []
    for key in sorted(members_by_stratum):
        members = sorted(members_by_stratum[key], key=lambda identity: identity[2])
        aggregate = np.stack([
            (row_weights[rows_by_identity[identity]][:, None] * matrix[rows_by_identity[identity]]).sum(axis=0)
            for identity in members
        ])
        multiplicities = sorted({len(rows_by_identity[identity]) for identity in members})
        strata.append({
            "dataset_id": key[0],
            "label": key[1],
            "size": len(members),
            "positions_per_identity": multiplicities,
            "aggregate": aggregate,
        })
    point = np.zeros(len(contrast_ids), dtype=np.float64)
    for stratum in strata:
        point += stratum["aggregate"].sum(axis=0)
    return {"strata": strata, "point": point, "cells": cells, "cell_sizes": cell_sizes,
            "identity_count": len(rows_by_identity)}


def rao_wu_factor(size: int) -> float:
    if size <= 1:
        return 1.0
    return math.sqrt(size / (size - 1.0))


def rao_wu_bootstrap(strata: list[dict[str, Any]], *, replicates: int, seed: int,
                     level: float = CONFIDENCE_LEVEL, chunk: int = 20000,
                     return_replicates: bool = False) -> dict[str, Any]:
    """Unscaled and Rao-Wu rescaled max-absolute-deviation bootstrap from one draw sequence.

    For each stratum, `replicates x n_s` uniform indices are drawn from PCG64DXSM
    in stratum order; multiplicity counts minus one form the centered
    contribution, which is added unscaled and multiplied by sqrt(n_s/(n_s-1)).
    The result does not depend on `chunk`.
    """

    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise ValueError("replicates must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1)")
    if not strata:
        raise ValueError("at least one stratum is required")
    width = int(strata[0]["aggregate"].shape[1])
    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    deviation_unscaled = np.zeros((replicates, width), dtype=np.float64)
    deviation_rao_wu = np.zeros((replicates, width), dtype=np.float64)
    factors = []
    singleton_strata = 0
    for stratum in strata:
        size = int(stratum["size"])
        aggregate = np.asarray(stratum["aggregate"], dtype=np.float64)
        if aggregate.shape != (size, width):
            raise ValueError("stratum aggregate has the wrong shape")
        factor = rao_wu_factor(size)
        if size <= 1:
            singleton_strata += 1
        factors.append(factor)
        draws = generator.integers(0, size, size=(replicates, size), endpoint=False)
        for start in range(0, replicates, chunk):
            block = draws[start:start + chunk]
            block_rows = block.shape[0]
            flat = (block + (np.arange(block_rows, dtype=np.int64) * size)[:, None]).ravel()
            counts = np.bincount(flat, minlength=block_rows * size).reshape(block_rows, size).astype(np.float64)
            contribution = (counts - 1.0) @ aggregate
            deviation_unscaled[start:start + block_rows] += contribution
            deviation_rao_wu[start:start + block_rows] += factor * contribution
    max_unscaled = np.max(np.abs(deviation_unscaled), axis=1)
    max_rao_wu = np.max(np.abs(deviation_rao_wu), axis=1)
    probabilities = (0.5, 0.9, 0.95, 0.99)
    result = {
        "replicates": replicates,
        "seed": seed,
        "generator": GENERATOR_NAME,
        "confidence_level": level,
        "quantile_method": QUANTILE_METHOD,
        "critical_unscaled": float(np.quantile(max_unscaled, level, method=QUANTILE_METHOD)),
        "critical_rao_wu": float(np.quantile(max_rao_wu, level, method=QUANTILE_METHOD)),
        "max_deviation_quantiles_unscaled": {
            format(p, ".12g"): float(np.quantile(max_unscaled, p, method=QUANTILE_METHOD)) for p in probabilities
        },
        "max_deviation_quantiles_rao_wu": {
            format(p, ".12g"): float(np.quantile(max_rao_wu, p, method=QUANTILE_METHOD)) for p in probabilities
        },
        "bootstrap_sd_unscaled": deviation_unscaled.std(axis=0, ddof=1) if replicates > 1 else np.zeros(width),
        "bootstrap_sd_rao_wu": deviation_rao_wu.std(axis=0, ddof=1) if replicates > 1 else np.zeros(width),
        "rao_wu_factors": factors,
        "max_rao_wu_factor": max(factors),
        "singleton_strata": singleton_strata,
    }
    if return_replicates:
        result["deviation_unscaled"] = deviation_unscaled
        result["deviation_rao_wu"] = deviation_rao_wu
    return result


# --------------------------------------------------------------------------- A2 evaluation


def interval_flags(interval: dict[str, float], *, region: float, critical: float) -> dict[str, bool]:
    lower, upper = float(interval["lower"]), float(interval["upper"])
    within_region = lower >= -region and upper <= region
    band_narrower = critical < region
    return {
        "contains_zero": lower <= 0.0 <= upper,
        "excludes_zero": lower > 0.0 or upper < 0.0,
        "excludes_zero_positive": lower > 0.0,
        "excludes_zero_negative": upper < 0.0,
        "interval_within_practical_region": within_region,
        "band_narrower_than_region": band_narrower,
        "inside_region_language_permitted": band_narrower and within_region,
    }


def evaluate_predictions(intervals: dict[str, dict[str, float]], *, region: float, critical: float) -> dict[str, Any]:
    """A2: X3 from the learned shared-translation control; X4 from the per-class controls."""

    missing = [candidate for candidate in (X3_CONTROL_ID, *X4_PER_CLASS_CONTROL_IDS) if candidate not in intervals]
    if missing:
        raise RuntimeError(f"registered control contrasts are absent: {missing}")
    x3_flags = interval_flags(intervals[X3_CONTROL_ID], region=region, critical=critical)
    companion_flags = (
        interval_flags(intervals[X3_COMPANION_ID], region=region, critical=critical)
        if X3_COMPANION_ID in intervals else None
    )
    per_class = {
        candidate: interval_flags(intervals[candidate], region=region, critical=critical)
        for candidate in X4_PER_CLASS_CONTROL_IDS
    }
    return {
        "X3": {
            "rule": "confirmed if the learned shared-translation control's simultaneous interval contains zero",
            "control_id": X3_CONTROL_ID,
            "interval": intervals[X3_CONTROL_ID],
            "confirmed": bool(x3_flags["contains_zero"]),
            "companion_control_id": X3_COMPANION_ID,
            "companion_interval": intervals.get(X3_COMPANION_ID),
            "companion_contains_zero": None if companion_flags is None else bool(companion_flags["contains_zero"]),
            "companion_role": "descriptive; the X3 verdict is defined on the tangent (trained) form only",
        },
        "X4": {
            "rule": "confirmed if at least one per-class control's simultaneous interval excludes zero on the positive side",
            "control_ids": list(X4_PER_CLASS_CONTROL_IDS),
            "intervals": {candidate: intervals[candidate] for candidate in X4_PER_CLASS_CONTROL_IDS},
            "positive_exclusions": {candidate: bool(flags["excludes_zero_positive"]) for candidate, flags in per_class.items()},
            "confirmed": bool(any(flags["excludes_zero_positive"] for flags in per_class.values())),
        },
        "practical_effect_region": {
            "half_width": region,
            "band_half_width": critical,
            "band_narrower_than_region": bool(critical < region),
            "interpretation": "reporting region only; 'inside the region' language requires a band narrower "
                              "than the region and an interval contained in it",
        },
    }


# --------------------------------------------------------------------------- main


def _environment() -> dict[str, str]:
    import scipy

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--merged", required=True, type=Path, help="directory of d1_06 merged cells")
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--primary", required=True, type=Path,
                        help="directory holding the registered d1_07 outputs (contrasts.json); output is written here")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2],
                        help="project root holding research/<addendum>.md and its approval file")
    parser.add_argument("--replicates", type=int, default=None, help="must equal the registered replicate count")
    parser.add_argument("--seed", type=int, default=None, help="must equal the registered bootstrap seed")
    parser.add_argument("--chunk", type=int, default=20000)
    args = parser.parse_args(argv)

    script_path = Path(__file__).resolve()
    root = args.root.resolve()
    output_path = args.primary / OUTPUT_NAME
    approval = verify_addendum_approval(root, script_path, output_path)

    replicates = int(primary_analysis.BOOTSTRAP_REPLICATES)
    seed = int(primary_analysis.BOOTSTRAP_SEED)
    if args.replicates is not None and args.replicates != replicates:
        raise RuntimeError("CLI replicate count differs from the registered analysis constant")
    if args.seed is not None and args.seed != seed:
        raise RuntimeError("CLI seed differs from the registered analysis constant")

    registration = read_json(args.registration)
    registration_hash = verify_registration_hash(registration)
    region = float(registration["prospective_practical_effect_region"]["absolute_ca"])
    if registration.get("final_test_access") is not False:
        raise RuntimeError("registration must record final_test_access as false")

    contrasts_path = args.primary / "contrasts.json"
    if not contrasts_path.is_file():
        raise FileNotFoundError(f"registered primary output is absent: {contrasts_path}")
    primary = read_json(contrasts_path)
    if primary.get("experiment_id") != registration["experiment_id"]:
        raise RuntimeError("contrasts.json experiment identity differs from the registration")
    if primary.get("registration_sha256") != registration_hash:
        raise RuntimeError("contrasts.json registration hash differs from the registration")
    if primary.get("reference") != primary_analysis.REFERENCE:
        raise RuntimeError("contrasts.json reference bank differs from the registered reference")
    if primary.get("primary_outcome") != PRIMARY_OUTCOME_TEXT:
        raise RuntimeError("contrasts.json primary outcome differs from the registered text")
    if primary.get("final_test_access") is not False:
        raise RuntimeError("contrasts.json must record final_test_access as false")
    band = primary.get("simultaneous_band", {})
    expected_band = {"replicates": replicates, "seed": seed, "confidence_level": CONFIDENCE_LEVEL, "method": BAND_METHOD}
    for key, expected in expected_band.items():
        if band.get(key) != expected:
            raise RuntimeError(f"contrasts.json simultaneous band field differs from the registered contract: {key}")
    registered_critical = float(band["critical_max_absolute_deviation"])
    primary_region = primary.get("prospective_practical_effect_region", {})
    if primary_region.get("lower") != -region or primary_region.get("upper") != region:
        raise RuntimeError("contrasts.json practical-effect region differs from the registration")

    reconstruction = load_primary_differences(args.merged, registration)
    family = reconstruction["family"]
    contrast_ids = family["contrasts"]
    if sorted(primary.get("contrasts", {})) != contrast_ids:
        raise RuntimeError("contrasts.json contrast family differs from the registered 21-contrast family")
    differences = reconstruction["differences"]
    cell_ids = reconstruction["cell_ids"]

    registered_points = np.array([
        float(primary["contrasts"][candidate]["simultaneous_confidence_interval"]["point_difference"])
        for candidate in contrast_ids
    ])
    recomputed_points = np.array([primary_analysis.macro_cell_mean(differences[candidate], cell_ids) for candidate in contrast_ids])
    point_gap = float(np.max(np.abs(registered_points - recomputed_points)))
    if point_gap > POINT_AGREEMENT_ATOL:
        raise RuntimeError(f"recomputed point contrasts differ from contrasts.json by {point_gap:.3e}; "
                           "the merged files are not the ones the primary analysis consumed")

    strata_info = build_strata(differences, cell_ids, reconstruction["class_labels"], reconstruction["item_ids"],
                               reconstruction["dataset_ids"], contrast_ids)
    aggregate_gap = float(np.max(np.abs(strata_info["point"] - recomputed_points)))
    if aggregate_gap > POINT_AGREEMENT_ATOL:
        raise RuntimeError("stratum aggregation does not reproduce the equal-cell macro statistic")
    bootstrap = rao_wu_bootstrap(strata_info["strata"], replicates=replicates, seed=seed, chunk=int(args.chunk))
    critical_unscaled = bootstrap["critical_unscaled"]
    critical_rao_wu = bootstrap["critical_rao_wu"]
    global_factor = float(bootstrap["max_rao_wu_factor"])

    total_rows = int(len(cell_ids))
    pooled = {candidate: float(np.sum(differences[candidate]) / total_rows) for candidate in contrast_ids}
    per_cell = {
        candidate: [
            {"cell_id": cell, "item_count": strata_info["cell_sizes"][cell],
             "cell_difference": float(differences[candidate][cell_ids == cell].mean())}
            for cell in strata_info["cells"]
        ]
        for candidate in contrast_ids
    }

    registered_intervals = {
        candidate: {
            "point_difference": float(primary["contrasts"][candidate]["simultaneous_confidence_interval"]["point_difference"]),
            "lower": float(primary["contrasts"][candidate]["simultaneous_confidence_interval"]["lower"]),
            "upper": float(primary["contrasts"][candidate]["simultaneous_confidence_interval"]["upper"]),
        }
        for candidate in contrast_ids
    }
    rao_wu_intervals = {
        candidate: {"point_difference": float(point), "lower": float(point - critical_rao_wu),
                    "upper": float(point + critical_rao_wu)}
        for candidate, point in zip(contrast_ids, recomputed_points)
    }
    bound_intervals = {
        candidate: {"point_difference": float(point), "lower": float(point - global_factor * registered_critical),
                    "upper": float(point + global_factor * registered_critical)}
        for candidate, point in zip(contrast_ids, recomputed_points)
    }

    contrasts_out = {}
    for index, candidate in enumerate(contrast_ids):
        contrasts_out[candidate] = {
            "family_role": "registered_control" if candidate in family["control_contrasts"] else "frozen_grid_candidate",
            "point_difference_equal_cell_macro": float(recomputed_points[index]),
            "registered_interval": registered_intervals[candidate],
            "registered_interval_flags": interval_flags(registered_intervals[candidate], region=region,
                                                        critical=registered_critical),
            "rao_wu_interval": rao_wu_intervals[candidate],
            "rao_wu_interval_flags": interval_flags(rao_wu_intervals[candidate], region=region, critical=critical_rao_wu),
            "global_inflation_bound_interval": bound_intervals[candidate],
            "bootstrap_sd_unscaled": float(bootstrap["bootstrap_sd_unscaled"][index]),
            "bootstrap_sd_rao_wu": float(bootstrap["bootstrap_sd_rao_wu"][index]),
            "item_pooled_point_difference": pooled[candidate],
            "item_pooled_minus_equal_cell": float(pooled[candidate] - recomputed_points[index]),
            "per_cell": per_cell[candidate],
            "weighting_note": "item-pooled value is descriptive; the equal-cell macro with the registered band is primary",
        }

    strata_summary: dict[str, dict[str, Any]] = {}
    for stratum in strata_info["strata"]:
        entry = strata_summary.setdefault(stratum["dataset_id"], {"stratum_count": 0, "sizes": set(),
                                                                   "positions_per_identity": set()})
        entry["stratum_count"] += 1
        entry["sizes"].add(int(stratum["size"]))
        entry["positions_per_identity"].update(int(v) for v in stratum["positions_per_identity"])
    for dataset, entry in strata_summary.items():
        entry["sizes"] = sorted(entry["sizes"])
        entry["positions_per_identity"] = sorted(entry["positions_per_identity"])
        entry["rao_wu_factors"] = [rao_wu_factor(size) for size in entry["sizes"]]
        entry["variance_understatement_factor_naive"] = [(size - 1) / size for size in entry["sizes"]]

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "SECONDARY_SENSITIVITY_OUTPUT; REVIEWED_RETROSPECTIVELY_AFTER_UNBLINDING (D-127); REGISTERED_BAND_REMAINS_PRIMARY",
        "approval_type": "retrospective review after unblinding (addendum V1 retrospective amendment, 2026-09-26)",
        "addendum": {"path": str(ADDENDUM_RELATIVE_PATH.as_posix()), "sha256": sha256_file(root / ADDENDUM_RELATIVE_PATH)},
        "approval": {key: approval.get(key) for key in ("status", "review_date", "reviewer", "addendum_sha256",
                                                        "retrospective_amendment_sha256", "sensitivity_source_sha256",
                                                        "tf32_adjudication_source_sha256", "primary_analysis_source_sha256",
                                                        "outcome_blind_at_approval", "outcomes_accessed_before_review",
                                                        "unblinding_decision", "unblinding_date")},
        "sensitivity_source_sha256": sha256_file(script_path),
        "primary_analysis_source_sha256": sha256_file(Path(primary_analysis.__file__).resolve()),
        "experiment_id": registration["experiment_id"],
        "registration_sha256": registration_hash,
        "registration_file_sha256": sha256_file(args.registration),
        "registration_canonical_hash_verified": True,
        "primary_contrasts_path": str(contrasts_path.as_posix()),
        "primary_contrasts_sha256": sha256_file(contrasts_path),
        "point_agreement_with_primary_max_abs": point_gap,
        "reference": family["reference"],
        "primary_outcome": PRIMARY_OUTCOME_TEXT,
        "estimand": "equal-cell macro mean over the primary cells of the paired per-cell difference in standard "
                    "certified accuracy at r = sigma versus no_correction",
        "cells": reconstruction["cells"],
        "bridge_cells_excluded": reconstruction["bridge_cells_excluded"],
        "contrast_family": {
            "count": len(contrast_ids),
            "frozen_grid_contrasts": family["frozen_grid_contrasts"],
            "control_contrasts": family["control_contrasts"],
        },
        "strata": {
            "count": len(strata_info["strata"]),
            "unique_identities": strata_info["identity_count"],
            "total_positions": total_rows,
            "by_dataset": strata_summary,
            "singleton_strata": bootstrap["singleton_strata"],
            "definition": "(dataset, class) strata of shared item identities; each identity is resampled once and its "
                          "multiplicity is reused in every primary cell that contains it",
        },
        "registered_band": {
            "critical_max_absolute_deviation": registered_critical,
            "replicates": replicates,
            "seed": seed,
            "confidence_level": CONFIDENCE_LEVEL,
            "generator": "numpy.random.default_rng (PCG64) inside d1_07_analyze.py",
            "method": BAND_METHOD,
            "primary": True,
        },
        "sensitivity_bootstrap": {
            "replicates": replicates,
            "seed": seed,
            "generator": GENERATOR_NAME,
            "confidence_level": CONFIDENCE_LEVEL,
            "quantile_method": QUANTILE_METHOD,
            "critical_unscaled": critical_unscaled,
            "critical_rao_wu": critical_rao_wu,
            "critical_rao_wu_over_unscaled": float(critical_rao_wu / critical_unscaled) if critical_unscaled > 0 else None,
            "monte_carlo_agreement_with_registered_critical": {
                "registered": registered_critical,
                "unscaled_pcg64dxsm": critical_unscaled,
                "absolute_difference": float(critical_unscaled - registered_critical),
                "relative_difference": float((critical_unscaled - registered_critical) / registered_critical)
                if registered_critical > 0 else None,
                "note": "same statistic, same seed and replicate count, different bit generator; differences are "
                        "Monte-Carlo error, not a change of method",
            },
            "global_inflation_bound": {
                "factor": global_factor,
                "definition": "max over strata of sqrt(n_s / (n_s - 1)), applied to the registered critical value",
                "critical_registered_times_factor": float(global_factor * registered_critical),
                "critical_unscaled_times_factor": float(global_factor * critical_unscaled),
            },
            "max_deviation_quantiles_unscaled": bootstrap["max_deviation_quantiles_unscaled"],
            "max_deviation_quantiles_rao_wu": bootstrap["max_deviation_quantiles_rao_wu"],
            "rao_wu_factors_by_stratum": bootstrap["rao_wu_factors"],
            "primary": False,
        },
        "contrasts": contrasts_out,
        "predictions_registered_band": evaluate_predictions(registered_intervals, region=region, critical=registered_critical),
        "predictions_rao_wu_band_secondary": evaluate_predictions(rao_wu_intervals, region=region, critical=critical_rao_wu),
        "practical_effect_region": {
            "half_width": region,
            "registered_band_narrower_than_region": bool(registered_critical < region),
            "rao_wu_band_narrower_than_region": bool(critical_rao_wu < region),
            "interpretation": "descriptive reporting region only; not an equivalence margin and not a TOST conclusion",
        },
        "multiplicity": {
            "family": f"{len(contrast_ids)} contrasts versus {family['reference']} in this experiment",
            "joint_control_across_experiments": False,
            "note": "EXP-019A and EXP-019B are separate families with separate bands; cross-experiment sentences are descriptive",
        },
        "environment": _environment(),
        "final_test_access": False,
        "new_sampling": False,
        "descriptive_only_fields": ["item_pooled_point_difference", "per_cell", "bootstrap_sd_unscaled",
                                    "bootstrap_sd_rao_wu", "predictions_rao_wu_band_secondary",
                                    "global_inflation_bound_interval"],
    }
    write_json_atomic(output_path, report)
    print(f"{'candidate':56s} {'delta':>9s} {'registered 95%':>22s} {'Rao-Wu 95%':>22s} {'pooled':>9s}")
    for candidate in contrast_ids:
        entry = contrasts_out[candidate]
        registered = entry["registered_interval"]
        rescaled = entry["rao_wu_interval"]
        print(f"{candidate:56s} {entry['point_difference_equal_cell_macro']:+9.4f} "
              f"[{registered['lower']:+.4f}, {registered['upper']:+.4f}] "
              f"[{rescaled['lower']:+.4f}, {rescaled['upper']:+.4f}] {entry['item_pooled_point_difference']:+9.4f}")
    print(f"\nregistered critical={registered_critical:.6f} unscaled(DXSM)={critical_unscaled:.6f} "
          f"Rao-Wu={critical_rao_wu:.6f} bound factor={global_factor:.6f}")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
