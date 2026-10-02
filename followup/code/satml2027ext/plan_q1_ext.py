#!/usr/bin/env python3
"""EXP-021A Q1 planning from the saved EXP-017 draws (V3; read-only; no new sampling).

021A re-certifies the twelve EXP-017 cells (same images, same banks) with fresh
Gaussian draws.  Its Q1 asks whether the post-hoc two-sided gain that the saved
EXP-017 draws gave on these images survives fresh draws.  This script derives,
from the EXP-017 files bound by the 021A registration, every number the V3
registration quotes for Q1:

1. identity checks: every 021A evaluation list equals the item ids and labels of
   its EXP-017 cells, and EXP-017 used 021A's certification parameters;
2. the saved-draw reference values, exact (``fractions.Fraction``): the
   equal-cell macro and the item-pooled paired difference in standard certified
   accuracy at r = sigma = 0.25 (two-sided c = 1 minus no_correction), and the
   per-dataset values; the outcomes are recomputed from the saved vote counts
   with ``common.certify.certify_from_counts`` and must equal the saved outcomes;
3. the registered 021A analysis applied to the saved draws (Rao-Wu shared-image
   bootstrap with the registered strata, seed and replicates): standard errors,
   the Q1 direction and shortfall p-values and the Q1 verdict (a sanity check:
   the saved draws must be classified as "reproduced");
4. two models of fresh draws on the same images (plug-in: each item's
   selected-class probability equals its saved vote share; posterior
   predictive: that probability has the Beta posterior of the saved votes).
   Certification at r = sigma succeeds when a fresh count reaches k_min.  The
   worker scores the candidate and the identity on common draws, so their
   outcomes on one image can be correlated either way.  V4 (audit E5): the
   variance of each item's difference is bounded by the sharp bound over all
   couplings with the given marginals, ``min(p + q, 2 - p - q) - (p - q)^2``
   (independence, ``p(1 - p) + q(1 - q)``, is not an upper bound: a negative
   correlation exceeds it); the simulated verdict probabilities assume an
   independent coupling and are labelled model-based;
5. normal-approximation verdict probabilities at three fixed gains (the
   reference, half of it, zero), conservatively treating the fresh estimate as
   varying by the item-bootstrap standard error;
6. for comparison, how the superseded V2 rule (Holm-adjusted two-sided
   direction test of the item-pooled value in the five-contrast family) would
   have behaved: its Holm-adjusted p on the saved draws (worst-case multiplier
   5) and its pass probability under the fresh-draw models.

EXP-017 outcomes are public (manuscript, Gate N2).  Nothing here reads 021A,
021B or 021C data.  Output: ``results/satml2027ext/planning/EXP021A_Q1_PLANNING.json``
(deterministic: no timestamps).

Portability (V4; text review of reviewer bundle V3, items 7-8; audit E6).  The
numerical routines behind some planning values (bootstrap sums, beta-binomial
tails) can differ in their last bits between runtimes.  Every float is
therefore rounded to 10 significant digits before serialization (exact values
stay exact strings), and the file carries a portability contract with three
separate tests: (1) the archived file is identified by its SHA-256; (2) a
regeneration on any runtime must agree with it under ``compare_planning``
(exact equality of every string, integer and Boolean; floats within the
declared tolerances); (3) byte identity is expected on the reference runtime
and is reported, not required, elsewhere.  ``--check FILE`` regenerates in
memory and applies (2) and (3).

Usage:
  python satml2027ext/plan_q1_ext.py --out results/satml2027ext/planning/EXP021A_Q1_PLANNING.json
  python satml2027ext/plan_q1_ext.py --check results/satml2027ext/planning/EXP021A_Q1_PLANNING.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from satml2027ext._common import PROJECT_ROOT, canonical_json_hash, read_json, sha256_file, write_json_atomic  # noqa: E402
from satml2027ext import analyze_ext  # noqa: E402
from common.certify import certify_from_counts, min_successes_for_radius  # noqa: E402

SCHEMA_VERSION = "satml2027ext.q1_planning.v2"
FLOAT_SIGNIFICANT_DIGITS = 10
PORTABLE_ABS_TOL = 1e-9
PORTABLE_REL_TOL = 1e-8
EXP017_ROOT = PROJECT_ROOT / "results/EXP-20260906-017"
EXP017_MANIFEST = EXP017_ROOT / "artifact_manifest.json"
ITEMS_DIR = PROJECT_ROOT / "results/satml2027ext/items"
MODELS = ("openai-clip-vit-b32-quickgelu", "openai-clip-vit-l14-quickgelu")
DATASETS = ("cifar100", "eurosat")
FOLDS = (0, 1, 2)
SIGMA = 0.25
RADII = (0.0, 0.1, 0.25, 0.5)
ALPHA_PER_EXAMPLE = 0.001
SELECTION_DRAWS = 128
CONFIRMATION_DRAWS = 4096
IDENTITY = "no_correction"
CONTRAST = "gr_clip_style_two_sided__coefficient_1"
STRATA = ("dataset_id", "class")
BOOTSTRAP_REPLICATES = 100000
BOOTSTRAP_SEED = 2026092101
Q1_ALPHA = 0.05
SIMULATIONS = 20000
SIMULATION_SEED = 2026092611
Z_TWO_SIDED = 1.959963984540054
Z_ONE_SIDED = 1.6448536269514722
Z_HOLM5_FIRST_STEP = 2.5758293035489004  # two-sided alpha 0.05 / 5


def normal_cdf(value: float) -> float:
    return 0.5 * math.erfc(-value / math.sqrt(2.0))


def round_floats(value: Any, digits: int = FLOAT_SIGNIFICANT_DIGITS) -> Any:
    """Every float rounded to ``digits`` significant digits (V4 portability); other values unchanged."""

    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("the planning record must not contain a non-finite number")
        return float(format(value, f".{digits}g"))
    if isinstance(value, Mapping):
        return {key: round_floats(item, digits) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_floats(item, digits) for item in value]
    raise TypeError(f"unsupported value {type(value).__name__} in the planning record")


def compare_planning(stored: Any, regenerated: Any, *, abs_tol: float = PORTABLE_ABS_TOL, rel_tol: float = PORTABLE_REL_TOL,
                     path: str = "") -> list[str]:
    """Portable agreement (test 2 of the portability contract); returns the differing paths.

    Strings, integers, Booleans and nulls must be equal; floats must satisfy
    ``|a - b| <= abs_tol + rel_tol * max(|a|, |b|)``.  ``content_sha256`` is
    derived from the rest and is not compared.
    """

    differences: list[str] = []
    if isinstance(stored, Mapping) and isinstance(regenerated, Mapping):
        keys = set(stored) | set(regenerated)
        for key in sorted(keys - {"content_sha256"} if path == "" else keys):
            if key not in stored or key not in regenerated:
                differences.append(f"{path}/{key}: present in only one record")
                continue
            differences.extend(compare_planning(stored[key], regenerated[key], abs_tol=abs_tol, rel_tol=rel_tol,
                                                path=f"{path}/{key}"))
        return differences
    if isinstance(stored, list) and isinstance(regenerated, list):
        if len(stored) != len(regenerated):
            return [f"{path}: lengths {len(stored)} and {len(regenerated)}"]
        for index, (left, right) in enumerate(zip(stored, regenerated)):
            differences.extend(compare_planning(left, right, abs_tol=abs_tol, rel_tol=rel_tol, path=f"{path}[{index}]"))
        return differences
    numeric = (int, float)
    if (isinstance(stored, float) or isinstance(regenerated, float)) and isinstance(stored, numeric) \
            and isinstance(regenerated, numeric) and not isinstance(stored, bool) and not isinstance(regenerated, bool):
        left, right = float(stored), float(regenerated)
        if not (math.isfinite(left) and math.isfinite(right)) or abs(left - right) > abs_tol + rel_tol * max(abs(left), abs(right)):
            return [f"{path}: {left!r} and {right!r}"]
        return []
    if type(stored) is not type(regenerated) or stored != regenerated:
        return [f"{path}: {stored!r} and {regenerated!r}"]
    return []


def cell_id_017(model_id: str, dataset_id: str, fold: int) -> str:
    return f"{model_id}__{dataset_id}__fold{fold}"


def cell_id_021a(model_id: str, dataset_id: str, fold: int) -> str:
    return f"{model_id}__{dataset_id}__fold{fold}__sigma{SIGMA:.12g}"


def load_exp017_cell(cell_id: str, manifest_entries: Mapping[str, str]) -> dict[str, Any]:
    import torch

    relative = f"cells/{cell_id}/cohen_sufficient_statistics.pt"
    path = EXP017_ROOT / relative
    if manifest_entries.get(relative) != sha256_file(path):
        raise RuntimeError(f"{relative} differs from the bound EXP-017 artifact manifest")
    raw = torch.load(path, map_location="cpu", weights_only=True)
    candidates = list(raw["candidate_ids"])
    out = {"item_ids": [str(value) for value in raw["item_ids"]],
           "ground_truth": np.asarray(raw["ground_truth"], dtype=np.int64), "file_sha256": sha256_file(path)}
    for candidate in (IDENTITY, CONTRAST):
        position = candidates.index(candidate)
        selection = np.asarray(raw["selection_counts"][position], dtype=np.int64)
        confirmation = np.asarray(raw["confirmation_counts"][position], dtype=np.int64)
        raw_predictions = np.asarray(raw["raw_predictions"][position], dtype=np.int64)
        if not np.all(selection.sum(axis=1) == SELECTION_DRAWS) or not np.all(confirmation.sum(axis=1) == CONFIRMATION_DRAWS):
            raise RuntimeError(f"{cell_id}: EXP-017 vote totals differ from 021A's certification budget")
        _, outcome = certify_from_counts(raw_predictions=raw_predictions, ground_truth=out["ground_truth"],
                                         selection_counts=selection, confirmation_counts=confirmation,
                                         sigma=SIGMA, alpha=ALPHA_PER_EXAMPLE, radii=list(RADII))
        saved = raw["outcomes"][candidate]
        saved_certified = ((~np.asarray(saved["abstained"], dtype=bool))
                           & (np.asarray(saved["certificate_radii"], dtype=np.float64) >= SIGMA)
                           & (np.asarray(saved["selected_classes"], dtype=np.int64) == out["ground_truth"]))
        recomputed = np.asarray(outcome[f"standard@{SIGMA:.12g}"], dtype=bool)
        if not np.array_equal(saved_certified, recomputed):
            raise RuntimeError(f"{cell_id}/{candidate}: recomputed outcomes differ from the saved EXP-017 outcomes")
        out[candidate] = {"certified": recomputed, "selection": selection, "confirmation": confirmation}
    return out


def read_list(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def certification_probability(selection: np.ndarray, confirmation: np.ndarray, truth: np.ndarray, k_min: int,
                              model: str) -> np.ndarray:
    """P(certified correct at r = sigma) per item under a model of fresh draws on the same image.

    ``plug_in``: the selected-class probability equals the saved vote share of the
    top class (selection plus confirmation draws), so a fresh confirmation count
    is Binomial(4096, share).  ``posterior_predictive``: the probability has the
    Beta(k + 1, N - k + 1) posterior of the saved top-class count k out of N
    draws, so the fresh count is beta-binomial (wider, and it regresses items
    that sat near the threshold).  Certification needs at least ``k_min``
    successes; the selection step picks the top class with probability close
    to one whenever certification is possible (share >= 0.8).
    """

    from scipy.stats import betabinom, binom

    totals = (selection + confirmation).astype(np.int64)
    draws = totals.sum(axis=1)
    top = totals.argmax(axis=1)
    k_top = totals[np.arange(len(top)), top]
    if model == "plug_in":
        probability = binom.sf(k_min - 1, CONFIRMATION_DRAWS, k_top / draws)
    elif model == "posterior_predictive":
        probability = betabinom.sf(k_min - 1, CONFIRMATION_DRAWS, k_top + 1, draws - k_top + 1)
    else:
        raise ValueError(f"unknown model {model!r}")
    probability = np.where(top == truth, probability, 0.0)
    return np.asarray(probability, dtype=np.float64)


def simulate_model(pc: np.ndarray, pi: np.ndarray, macro_weights: np.ndarray, pooled_weights: np.ndarray,
                   se: np.ndarray, references: Mapping[str, float], seed: int) -> dict[str, Any]:
    """Expected values, variance bounds and simulated normal-region Q1 verdicts for one fresh-draw model."""

    expected_difference = pc - pi
    independent_variance = pc * (1 - pc) + pi * (1 - pi)
    # V4 (audit E5): the sharp bound over every coupling of Bernoulli(p) and Bernoulli(q) outcomes
    sharp_variance = np.maximum(np.minimum(pc + pi, 2.0 - pc - pi) - (pc - pi) ** 2, 0.0)
    expected = {"macro": float((macro_weights * expected_difference).sum() * 100),
                "pooled": float((pooled_weights * expected_difference).sum() * 100)}
    independent_sd = {"macro": float(math.sqrt((macro_weights ** 2 * independent_variance).sum()) * 100),
                      "pooled": float(math.sqrt((pooled_weights ** 2 * independent_variance).sum()) * 100)}
    sharp_sd = {"macro": float(math.sqrt((macro_weights ** 2 * sharp_variance).sum()) * 100),
                "pooled": float(math.sqrt((pooled_weights ** 2 * sharp_variance).sum()) * 100)}
    random_rows = np.flatnonzero(independent_variance > 0)
    fixed_difference = np.where(independent_variance > 0, 0.0, expected_difference)
    fixed = {"macro": float((macro_weights * fixed_difference).sum() * 100),
             "pooled": float((pooled_weights * fixed_difference).sum() * 100)}
    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    names = ("macro", "pooled")
    tallies = {name: {key: 0 for key in ("reproduced", "reproduced_smaller", "not_reproduced", "inconclusive", "reversed")}
               for name in names}
    sums = {name: 0.0 for name in names}
    chunk = 2000
    for start in range(0, SIMULATIONS, chunk):
        size = min(chunk, SIMULATIONS - start)
        draw_c = generator.random((size, random_rows.size)) < pc[random_rows][None, :]
        draw_i = generator.random((size, random_rows.size)) < pi[random_rows][None, :]
        delta = draw_c.astype(np.float64) - draw_i.astype(np.float64)
        estimates = {"macro": fixed["macro"] + (delta @ macro_weights[random_rows]) * 100,
                     "pooled": fixed["pooled"] + (delta @ pooled_weights[random_rows]) * 100}
        for column, name in enumerate(names):
            sums[name] += float(estimates[name].sum())
            for value in estimates[name]:
                tallies[name][classify_normal(float(value), float(se[column]), references[name])] += 1
    return {
        "items_with_uncertain_outcome": {"rule": "certification probability in (0.001, 0.999)",
                                         "candidate": int(((pc > 0.001) & (pc < 0.999)).sum()),
                                         "identity": int(((pi > 0.001) & (pi < 0.999)).sum())},
        "expected_points": expected,
        "monte_carlo_sd_points": {
            "sharp_upper_bound_any_coupling": sharp_sd,
            "independent_coupling_model": independent_sd,
            "rule": ("per item, Var(C - I) <= min(p + q, 2 - p - q) - (p - q)^2 over every joint law with marginals p and q "
                     "(attained by the extreme coupling); items are drawn independently of each other, so the item bounds add; "
                     "the independent coupling p(1 - p) + q(1 - q) is a model value, not a bound"),
        },
        "sharp_bound_normal_verdict_probabilities": {
            name: region_probabilities(expected[name], sharp_sd[name], float(se[column]), references[name])
            for column, name in enumerate(names)
        },
        "simulated_mean_points": {name: sums[name] / SIMULATIONS for name in names},
        "verdict_probabilities": {name: {key: count / SIMULATIONS for key, count in tally.items()} for name, tally in tallies.items()},
        "verdict_probabilities_basis": "model-based: simulated with the candidate and identity outcomes coupled independently",
    }


def classify_normal(point: float, se: float, reference: float) -> str:
    if point < -Z_TWO_SIDED * se:
        return "reversed"
    positive = point > Z_TWO_SIDED * se
    shortfall = point < reference - Z_ONE_SIDED * se
    if positive:
        return "reproduced_smaller" if shortfall else "reproduced"
    return "not_reproduced" if shortfall else "inconclusive"


def normal_verdict_probabilities(delta: float, se: float, reference: float) -> dict[str, float]:
    """Exact class probabilities for X ~ N(delta, se^2) under the normal-approximation Q1 regions."""

    return region_probabilities(delta, se, se, reference)


def region_probabilities(delta: float, sd: float, se: float, reference: float) -> dict[str, float]:
    """Class probabilities for X ~ N(delta, sd^2) under the Q1 regions built from the standard error ``se``."""

    lower = -Z_TWO_SIDED * se
    upper = Z_TWO_SIDED * se
    cut = reference - Z_ONE_SIDED * se

    def mass(a: float, b: float) -> float:
        if b <= a:
            return 0.0
        if sd <= 0:
            return 1.0 if a < delta <= b else 0.0
        return normal_cdf((b - delta) / sd) - normal_cdf((a - delta) / sd)

    inf = float("inf")
    return {
        "reproduced": mass(max(upper, cut), inf),
        "reproduced_smaller": mass(upper, cut),
        "not_reproduced": mass(lower, min(upper, cut)),
        "inconclusive": mass(max(lower, cut), upper),
        "reversed": mass(-inf, lower),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    manifest_sha = sha256_file(EXP017_MANIFEST)
    if manifest_sha != args.expected_exp017_manifest_sha256:
        raise RuntimeError("EXP-017 artifact manifest differs from the hash the 021A registration binds")
    manifest_entries = {entry["path"]: entry["sha256"] for entry in read_json(EXP017_MANIFEST)["artifacts"]}
    summary = read_json(EXP017_ROOT / "summary.json")
    certification_equal = (float(summary["cohen_alpha_per_example"]) == ALPHA_PER_EXAMPLE
                           and int(summary["cohen_confirmation_samples_per_example"]) == CONFIRMATION_DRAWS
                           and int(summary["cohen_selection_samples_per_example"]) == SELECTION_DRAWS
                           and float(summary["cohen_sigma"]) == SIGMA)
    if not certification_equal:
        raise RuntimeError("EXP-017 certification parameters differ from 021A's")
    if manifest_entries.get("summary.json") not in (None, sha256_file(EXP017_ROOT / "summary.json")):
        raise RuntimeError("EXP-017 summary differs from its artifact manifest")
    k_min = min_successes_for_radius(CONFIRMATION_DRAWS, ALPHA_PER_EXAMPLE, SIGMA, SIGMA)

    position_rows: list[dict[str, Any]] = []
    differences: list[int] = []
    model_names = ("plug_in", "posterior_predictive")
    model_rows: dict[str, dict[str, list[float]]] = {name: {"candidate": [], "identity": []} for name in model_names}
    cell_of_row: list[str] = []
    identity_checks: dict[str, Any] = {}
    list_hashes: dict[str, str] = {}
    statistics_hashes: dict[str, str] = {}
    exact_cell: dict[str, Fraction] = {}
    cells = sorted((cell_id_021a(m, d, f), m, d, f) for m in MODELS for d in DATASETS for f in FOLDS)
    for cell_id, model_id, dataset_id, fold in cells:
        list_name = f"exp021a__{dataset_id}__fold{fold}__evaluation.csv"
        rows = read_list(ITEMS_DIR / list_name)
        list_hashes[list_name] = sha256_file(ITEMS_DIR / list_name)
        saved = load_exp017_cell(cell_id_017(model_id, dataset_id, fold), manifest_entries)
        statistics_hashes[cell_id_017(model_id, dataset_id, fold)] = saved["file_sha256"]
        index_of = {item_id: position for position, item_id in enumerate(saved["item_ids"])}
        registered_ids = [row["item_id"] for row in rows]
        same_set = set(registered_ids) == set(saved["item_ids"]) and len(registered_ids) == len(saved["item_ids"])
        labels_equal = same_set and all(int(row["label"]) == int(saved["ground_truth"][index_of[row["item_id"]]]) for row in rows)
        identity_checks[cell_id] = {"items_equal": same_set, "labels_equal": labels_equal,
                                    "order_equal": registered_ids == saved["item_ids"], "items": len(rows)}
        if not (same_set and labels_equal):
            raise RuntimeError(f"{cell_id}: the 021A evaluation list is not the EXP-017 item set")
        order = np.array([index_of[item_id] for item_id in registered_ids], dtype=np.int64)
        truth = saved["ground_truth"][order]
        candidate_certified = saved[CONTRAST]["certified"][order].astype(np.int64)
        identity_certified = saved[IDENTITY]["certified"][order].astype(np.int64)
        cell_difference = candidate_certified - identity_certified
        exact_cell[cell_id] = Fraction(int(cell_difference.sum()), len(rows))
        for name in model_names:
            model_rows[name]["candidate"].extend(certification_probability(
                saved[CONTRAST]["selection"][order], saved[CONTRAST]["confirmation"][order], truth, k_min, name).tolist())
            model_rows[name]["identity"].extend(certification_probability(
                saved[IDENTITY]["selection"][order], saved[IDENTITY]["confirmation"][order], truth, k_min, name).tolist())
        for position, row in enumerate(rows):
            position_rows.append({"cell_id": cell_id, "model_id": model_id, "dataset_id": dataset_id, "sigma": SIGMA,
                                  "fold": fold, "item_id": row["item_id"], "class": int(row["label"]), "index": position})
            differences.append(int(cell_difference[position]))
            cell_of_row.append(cell_id)

    cell_ids = [cell[0] for cell in cells]
    n_cells = len(cell_ids)
    sizes = {cell_id: sum(1 for value in cell_of_row if value == cell_id) for cell_id in cell_ids}
    total = len(position_rows)
    macro_exact = sum(exact_cell.values(), Fraction(0)) / n_cells
    pooled_exact = Fraction(sum(differences), total)
    by_dataset_exact = {}
    for dataset_id in DATASETS:
        members = [index for index, row in enumerate(position_rows) if row["dataset_id"] == dataset_id]
        by_dataset_exact[dataset_id] = Fraction(sum(differences[index] for index in members), len(members))

    # the registered 021A analysis estimator on the saved draws
    images = analyze_ext.build_images(position_rows, STRATA)
    diff = np.asarray(differences, dtype=np.float64)
    macro_weights = np.array([1.0 / (sizes[cell] * n_cells) for cell in cell_of_row])
    pooled_weights = np.full(total, 1.0 / total)
    aggregate = np.zeros((images["count"], 2), dtype=np.float64)
    np.add.at(aggregate, images["row_image"], np.stack([macro_weights * diff, pooled_weights * diff], axis=1))
    point = aggregate.sum(axis=0) * 100.0
    bootstrap = analyze_ext.rao_wu_bootstrap(aggregate, images["strata"], replicates=args.replicates, seed=BOOTSTRAP_SEED)
    values = point[None, :] + bootstrap["deviation"] * 100.0
    se = values.std(axis=0, ddof=1)
    references = {"macro": float(macro_exact * 100), "pooled": float(pooled_exact * 100)}
    saved_analysis = {}
    for column, name in enumerate(("macro", "pooled")):
        verdict = analyze_ext.q1_classify(float(point[column]), values[:, column], references[name], Q1_ALPHA,
                                          analyze_ext.percentile_interval(values[:, column], 0.95))
        saved_analysis[name] = {**verdict, "bootstrap_se_points": float(se[column]), "z": float(point[column] / se[column])}

    # fresh-draw models on the same images (Monte-Carlo component only)
    models = {}
    for offset, name in enumerate(model_names):
        pc = np.asarray(model_rows[name]["candidate"], dtype=np.float64)
        pi = np.asarray(model_rows[name]["identity"], dtype=np.float64)
        models[name] = simulate_model(pc, pi, macro_weights, pooled_weights, se, references, SIMULATION_SEED + offset)
    fresh_draw_models = {
        "assumption": ("items and banks fixed at EXP-017's; only the Gaussian draws are fresh. The worker scores the candidate and "
                       "the identity on common draws, so their outcomes on one image can be correlated in either direction: the "
                       "Monte-Carlo SD is bounded by the sharp any-coupling bound; the simulated verdict probabilities assume an "
                       "independent coupling and are model-based"),
        "plug_in": {"model": "selected-class probability = saved vote share (binomial)", **models["plug_in"]},
        "posterior_predictive": {"model": "selected-class probability ~ Beta(k + 1, N - k + 1) posterior of the saved votes (beta-binomial); the primary planning model", **models["posterior_predictive"]},
        "simulations": SIMULATIONS,
        "simulation_seeds": [SIMULATION_SEED, SIMULATION_SEED + 1],
        "decision_approximation": "normal regions with the saved item-bootstrap standard error: positive if x > 1.960 se; shortfall if x < reference - 1.645 se; sharp_bound_normal_verdict_probabilities put X ~ N(expected, sharp SD bound^2) in these regions",
        "limitation": "neither model represents the post-hoc selection of the contrast among the 136 EXP-017 contrasts or software differences between the EXP-017 and EXP-021 stacks; those are what Q1 measures",
    }
    scenarios = {}
    for column, name in enumerate(("macro", "pooled")):
        reference = references[name]
        scenarios[name] = {label: normal_verdict_probabilities(value, float(se[column]), reference)
                           for label, value in (("gain_equals_reference", reference), ("gain_half_reference", reference / 2.0),
                                                ("gain_zero", 0.0))}
        scenarios[name]["regions_points"] = {
            "positive_above": Z_TWO_SIDED * float(se[column]), "negative_below": -Z_TWO_SIDED * float(se[column]),
            "shortfall_below": reference - Z_ONE_SIDED * float(se[column]),
        }

    pooled_se = float(se[1])
    v2_rule = {
        "rule": "superseded V2 Q1 rule: Holm-adjusted two-sided direction test of the item-pooled value in the five-contrast family; worst case (multiplier 5) needs |z| >= 2.576",
        "saved_draws_z": saved_analysis["pooled"]["z"],
        "holm5_worst_case_adjusted_p_on_saved_draws": min(1.0, 5.0 * saved_analysis["pooled"]["p_direction"]),
        "pass_probability_normal_approximation": {
            name: 1.0 - normal_cdf((Z_HOLM5_FIRST_STEP - models[name]["expected_points"]["pooled"] / pooled_se)
                                   / (models[name]["monte_carlo_sd_points"]["sharp_upper_bound_any_coupling"]["pooled"] / pooled_se))
            for name in model_names
        },
        "pass_probability_if_the_saved_value_were_the_expectation": 1.0 - normal_cdf(
            (Z_HOLM5_FIRST_STEP - float(point[1]) / pooled_se)
            / (models["posterior_predictive"]["monte_carlo_sd_points"]["sharp_upper_bound_any_coupling"]["pooled"] / pooled_se)),
        "sd_used": "the sharp any-coupling Monte-Carlo SD bound (V4)",
        "reading": "the saved pooled value sits just above the worst-case Holm-5 threshold, so under fresh draws the V2 rule would have been close to a coin flip, and its failure branch asserted non-reproduction and a cause",
    }
    record = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "EXP-021A Q1 reference values and planning, derived from the saved EXP-017 draws on the same images",
        "contrast": f"{CONTRAST} minus {IDENTITY}; standard certified accuracy at r = sigma = {SIGMA}",
        "inputs": {
            "exp017_artifact_manifest": {"path": "results/EXP-20260906-017/artifact_manifest.json", "sha256": manifest_sha},
            "exp017_sufficient_statistics_sha256": dict(sorted(statistics_hashes.items())),
            "item_lists_sha256": dict(sorted(list_hashes.items())),
            "k_min_at_r_equals_sigma": k_min,
        },
        "identity_checks": {"certification_parameters_equal": certification_equal,
                            "outcomes_recomputed_from_counts_equal_saved": True, "cells": identity_checks},
        "reference_points": {
            "equal_cell_macro": {"exact_points": str(macro_exact * 100), "value": float(macro_exact * 100)},
            "item_pooled": {"exact_points": str(pooled_exact * 100), "value": float(pooled_exact * 100)},
            "by_dataset_item_pooled": {name: {"exact_points": str(value * 100), "value": float(value * 100)}
                                       for name, value in by_dataset_exact.items()},
            "by_cell": {cell: {"exact_points": str(value * 100), "value": float(value * 100)} for cell, value in sorted(exact_cell.items())},
            "positions": total, "unique_images": int(images["count"]),
        },
        "saved_draws_under_registered_q1_analysis": {
            "bootstrap": {"method": "Rao-Wu shared-image, strata (dataset_id, class)", "replicates": args.replicates,
                          "seed": BOOTSTRAP_SEED, "strata": len(images["strata"])},
            **saved_analysis,
        },
        "fresh_draw_models": fresh_draw_models,
        "fixed_gain_scenarios_normal_approximation": scenarios,
        "superseded_v2_rule": v2_rule,
        "reading": (
            "Q1's primary estimand is the equal-cell macro (the manuscript's headline).  On the saved draws it lies about "
            f"{saved_analysis['macro']['z']:.1f} bootstrap standard errors above zero, so the positive and shortfall regions do not "
            "overlap in an inconclusive band: every fresh-draw outcome is classified.  The item-pooled secondary estimand "
            f"(about {saved_analysis['pooled']['z']:.1f} standard errors) keeps a narrow inconclusive band.  Both fresh-draw models put the "
            "Monte-Carlo standard deviation of the macro at most "
            f"{fresh_draw_models['posterior_predictive']['monte_carlo_sd_points']['sharp_upper_bound_any_coupling']['macro']:.2f} points (sharp any-coupling bound) against an item-bootstrap "
            f"standard error of {saved_analysis['macro']['bootstrap_se_points']:.2f}: fresh draws on the same images are expected to reproduce "
            "the saved value. Q1 checks whether the historical contrast persists when the certification draws and the execution pipeline "
            "are refreshed on the same registered images and the same legacy bank tensors; it is not a calibrated decomposition of the "
            "variance and does not identify why two runs differ (the verdict regions use the item-bootstrap standard error, a conservative "
            "convention, not the fresh-draw repeat distribution). The item-level question belongs to 021B (Q6, fresh images)."
        ),
        "portability": {
            "float_rounding": f"every float rounded to {FLOAT_SIGNIFICANT_DIGITS} significant digits before serialization; exact values are exact strings",
            "tests": {
                "archived_identity": "the archived file is identified by its SHA-256 (bound by the 021A registration)",
                "portable_agreement": ("required on any runtime: satml2027ext/plan_q1_ext.py::compare_planning of a regeneration against "
                                       "the archived file (exact equality of strings, integers and Booleans; floats within the "
                                       "tolerances below); run it with --check"),
                "byte_identity": "expected on the reference runtime recorded in research/EXP021_V4_AMENDMENT_2026-09-26.md; reported, not required, elsewhere",
            },
            "abs_tol": PORTABLE_ABS_TOL,
            "rel_tol": PORTABLE_REL_TOL,
        },
    }
    record = round_floats(record)
    record["content_sha256"] = canonical_json_hash(record)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results/satml2027ext/planning/EXP021A_Q1_PLANNING.json")
    parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--expected-exp017-manifest-sha256",
                        default="a215504af683f3316f2ca70e5915ad2957ce6d0d4304313007649a636545be0c")
    parser.add_argument("--force", action="store_true", help="overwrite an existing planning file")
    parser.add_argument("--check", type=Path, default=None,
                        help="regenerate in memory and compare with this archived file (portable agreement required; byte identity reported)")
    args = parser.parse_args(argv)
    if args.check is not None:
        record = build(args)
        stored = read_json(args.check)
        differences = compare_planning(stored, record)
        regenerated_bytes = (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        byte_identical = regenerated_bytes == args.check.read_bytes()
        print(f"portable agreement: {'yes' if not differences else 'NO'} ({len(differences)} differing values)")
        for line in differences[:20]:
            print("  " + line)
        print(f"byte identity with {args.check}: {'yes' if byte_identical else 'no (expected only on the reference runtime)'}")
        print(f"archived sha256={sha256_file(args.check)}")
        return 0 if not differences else 1
    if args.out.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {args.out} (use --force to regenerate)")
    record = build(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out, record)
    print(json.dumps({key: record[key] for key in ("reference_points",)}, indent=1)[:2000])
    print(f"wrote {args.out} sha256={sha256_file(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
