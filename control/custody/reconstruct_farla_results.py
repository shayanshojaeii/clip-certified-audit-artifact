#!/usr/bin/env python3
"""Independent reconstruction of EXP-20260917-020-FARLA-FULL from its frozen outputs.

Custody rule: this script imports nothing from the FARLA package (phase4/,
analysis/, certification/, vlm_pipeline/). Every quantity below is recomputed
from the stored vote counts, raw-clean predictions, ground truth, split ledgers
and checkpoints using only numpy, scipy, torch (for .pt deserialisation) and the
standard library, then compared with analysis/report.json and each cell's
metrics.json.

What is recomputed for every candidate bank in every cell
  * number of items, abstentions, selected-class counts and histogram
  * Clopper-Pearson one-sided lower bound, alpha = 0.001, n = 4096:
        p_L = scipy.stats.beta.ppf(alpha, k, n - k + 1)      (p_L = 0 when k = 0)
  * certified radius  sigma * Phi^-1(p_L)  (0 when abstained, i.e. p_L <= 0.5)
  * standard certified accuracy at every protocol radius
        (not abstained, selected == ground truth, radius >= r)
  * anchored certified accuracy (additionally selected == raw-clean prediction)
  * certified-wrong-class rate, smoothed accuracy, raw clean accuracy
  * macro (equal-cell) aggregates and the exact-fraction clean-accuracy gate
  * planned contrasts: paired item differences, exact McNemar per cell with
    Holm adjustment, and the paired class-stratified percentile bootstrap with
    the seed and replicate count stated in report.json

Data roles
  * re-derives the Phase-4 train/selection/confirmation split from the Phase-3
    ledger using the documented rank hash, checks per-class balance, mutual
    disjointness, and that no Phase-4 item (in particular no confirmation item)
    carries a Phase-3 development/validation/excluded/sealed role.

Run from the repository root (torch is required to read the .pt files):
    .venv-gate-n2/Scripts/python FARLA/artifacts/phase4/FARLA_FULL_RESULT_CUSTODY_V1/reconstruct_farla_results.py
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import beta, norm

SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_DIR = SCRIPT_PATH.parent
FARLA_ROOT = PACKAGE_DIR.parents[2]
REPO_ROOT = PACKAGE_DIR.parents[3]
DEFAULT_RESULTS = FARLA_ROOT / "downloaded_results" / "EXP-20260917-020-FARLA-FULL"
SCHEMA_VERSION = "farla_result_custody.reconstruction.v1"
TOLERANCE = 1e-9

# Protocol constants the reconstruction insists on. They are also read from
# config_snapshot.json; a mismatch is reported as a FAIL, never silently adopted.
EXPECTED = {
    "experiment_id": "EXP-20260917-020-FARLA-FULL",
    "sigma": 0.25,
    "alpha": 0.001,
    "confirmation_draws": 4096,
    "selection_draws": 128,
    "primary_radius": 0.25,
    "radii": [0.0, 0.1, 0.25, 0.5],
    "bootstrap_replicates": 10000,
    "bootstrap_seed": 2026091605,
    "split_seed": 20260916,
    "train_per_class": 20,
    "selection_per_class": 5,
    "confirmation_per_class": 10,
    "clean_macro_limit_points": 1.0,
    "clean_cell_limit_points": 2.0,
    "minimum_successes_at_primary_radius": 3518,
}
HEADLINE_CANDIDATES = [
    "no_correction",
    "shared_translation_ce_supervised",
    "lowrank_ce_r8_supervised",
    "lowrank_tail_r8_supervised",
    "farla_full_r8_supervised",
]
# Values quoted in the custody task statement (4-decimal rounding of report.json).
TASK_STATED_HEADLINE = {
    "no_correction": {"macro_standard_ca_primary": 0.1160, "macro_clean_accuracy": 0.6088},
    "shared_translation_ce_supervised": {"macro_standard_ca_primary": 0.1175, "macro_clean_accuracy": 0.6128},
    "lowrank_ce_r8_supervised": {"macro_standard_ca_primary": 0.2710, "macro_clean_accuracy": 0.7250},
    "lowrank_tail_r8_supervised": {"macro_standard_ca_primary": 0.3650, "macro_clean_accuracy": 0.7080},
    "farla_full_r8_supervised": {"macro_standard_ca_primary": 0.3505, "macro_clean_accuracy": 0.6858},
}
REQUESTED_CONTRASTS = ["farla_vs_no_correction", "farla_vs_lowrank_tail"]
PHASE3_LEDGERS = [
    "artifacts/day14/phase3_splits_v1/assignments.csv",
    "results/EXP-20260906-017/phase3_split_assignments_snapshot.csv",
]
PHASE3_NON_RESERVED_ROLES = [
    "calibration_development",
    "calibration_validation",
    "excluded_explicit",
    "excluded_prior_access",
    "final_test_sealed",
]


# --------------------------------------------------------------------------- #
# Small utilities                                                              #
# --------------------------------------------------------------------------- #
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def radius_key(radius: float) -> str:
    """Key format used by the frozen outputs for per-radius dictionaries."""
    return format(float(radius), ".12g")


def derive_seed(base_seed: int, *parts) -> int:
    """Documented FARLA seed derivation (phase4/common.py): SHA-256 of the
    colon-joined text, first 8 bytes big-endian, masked to 63 bits."""
    text = ":".join([str(base_seed), *(str(value) for value in parts)])
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") & ((1 << 63) - 1)


def rank_hash(seed: int, dataset_id: str, item_id: str) -> str:
    """Documented Phase-4 role ranking hash (phase4/splits.py)."""
    return hashlib.sha256(f"{seed}:{dataset_id}:{item_id}".encode("utf-8")).hexdigest()


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_pt(path: Path):
    import torch  # deferred so the module imports without torch

    return torch.load(path, map_location="cpu", weights_only=True)


def to_numpy(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def max_abs_dev(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator, "float": float(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    return value


class Checks:
    """Ordered ledger of PASS / FAIL / NOT_VERIFIED findings."""

    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, check_id: str, status: str, detail: str, **extra) -> None:
        if status not in {"PASS", "FAIL", "NOT_VERIFIED"}:
            raise ValueError(status)
        self.items.append({"id": check_id, "status": status, "detail": detail, **jsonable(extra)})

    def expect(self, check_id: str, condition: bool, detail: str, **extra) -> bool:
        self.add(check_id, "PASS" if condition else "FAIL", detail, **extra)
        return bool(condition)

    def not_verified(self, check_id: str, detail: str, **extra) -> None:
        self.add(check_id, "NOT_VERIFIED", detail, **extra)

    @property
    def failures(self) -> list[dict]:
        return [item for item in self.items if item["status"] == "FAIL"]

    @property
    def unverified(self) -> list[dict]:
        return [item for item in self.items if item["status"] == "NOT_VERIFIED"]


# --------------------------------------------------------------------------- #
# Certification arithmetic                                                     #
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def clopper_pearson_lower(successes: int, trials: int, alpha: float) -> float:
    if successes == 0:
        return 0.0
    return float(beta.ppf(alpha, successes, trials - successes + 1))


def minimum_successes(trials: int, alpha: float, radius: float, sigma: float) -> int:
    """Smallest k whose one-sided lower bound reaches Phi(radius / sigma)."""
    target = float(norm.cdf(radius / sigma))
    lo, hi = 0, trials
    while lo < hi:
        mid = (lo + hi) // 2
        if clopper_pearson_lower(mid, trials, alpha) >= target:
            hi = mid
        else:
            lo = mid + 1
    if clopper_pearson_lower(lo, trials, alpha) < target:
        raise RuntimeError("target radius unattainable at this budget")
    return lo


def certify_candidate(selection_counts, confirmation_counts, raw_predictions, ground_truth, *,
                      sigma: float, alpha: float, radii: list[float]) -> dict:
    sel = np.asarray(selection_counts, dtype=np.int64)
    conf = np.asarray(confirmation_counts, dtype=np.int64)
    raw = np.asarray(raw_predictions, dtype=np.int64)
    truth = np.asarray(ground_truth, dtype=np.int64)
    n_items, class_count = sel.shape
    selected = sel.argmax(axis=1)  # first maximum on ties, same as torch.argmax
    successes = conf[np.arange(n_items), selected]
    trials_by_row = conf.sum(axis=1)
    trials = int(trials_by_row[0])
    lower = np.array([clopper_pearson_lower(int(k), trials, alpha) for k in successes], dtype=np.float64)
    abstained = lower <= 0.5
    valid = ~abstained
    radius = np.zeros(n_items, dtype=np.float64)
    if valid.any():
        radius[valid] = sigma * norm.ppf(np.clip(lower[valid], 1e-15, 1.0 - 1e-15))
    correct = selected == truth
    raw_correct = raw == truth
    anchored = selected == raw
    per_radius = {}
    for r in radii:
        reaches = valid & (radius >= float(r))
        standard = reaches & correct
        per_radius[radius_key(r)] = {
            "reaches": reaches,
            "standard": standard,
            "anchored": standard & anchored,
            "wrong_stable": reaches & ~correct,
        }
    return {
        "n_items": int(n_items),
        "class_count": int(class_count),
        "trials": trials,
        "trials_uniform": bool((trials_by_row == trials).all()),
        "selection_draws_uniform": bool((sel.sum(axis=1) == sel.sum(axis=1)[0]).all()),
        "selection_draws": int(sel.sum(axis=1)[0]),
        "selected": selected,
        "successes": successes,
        "lower": lower,
        "abstained": abstained,
        "radius": radius,
        "correct": correct,
        "raw_correct": raw_correct,
        "anchored": anchored,
        "per_radius": per_radius,
    }


def summarise_candidate(cert: dict, primary_key: str, k_min: int) -> dict:
    n = cert["n_items"]
    hist = Counter(int(c) for c in cert["selected"])
    successes = cert["successes"]
    out = {
        "example_count": n,
        "abstained_count": int(cert["abstained"].sum()),
        "abstention_rate": float(np.mean(cert["abstained"])),
        "clean_correct_count": int(cert["raw_correct"].sum()),
        "clean_accuracy": float(np.mean(cert["raw_correct"])),
        "smoothed_correct_count": int(cert["correct"].sum()),
        "smoothed_accuracy": float(np.mean(cert["correct"])),
        "selection_raw_agreement": float(np.mean(cert["anchored"])),
        "average_selected_class_radius": float(np.mean(cert["radius"])),
        "standard_certified_accuracy": {k: float(np.mean(v["standard"])) for k, v in cert["per_radius"].items()},
        "anchored_certified_accuracy": {k: float(np.mean(v["anchored"])) for k, v in cert["per_radius"].items()},
        "certified_wrong_class_rate": {k: float(np.mean(v["wrong_stable"])) for k, v in cert["per_radius"].items()},
        "standard_certified_count": {k: int(v["standard"].sum()) for k, v in cert["per_radius"].items()},
        "anchored_certified_count": {k: int(v["anchored"].sum()) for k, v in cert["per_radius"].items()},
        "certified_wrong_count": {k: int(v["wrong_stable"].sum()) for k, v in cert["per_radius"].items()},
        "minimum_successes_for_primary_radius": int(k_min),
        "items_with_successes_ge_kmin": int((successes >= k_min).sum()),
        "count_threshold_agrees_with_float_radius_test": bool(
            np.array_equal(successes >= k_min, cert["per_radius"][primary_key]["reaches"])
        ),
        "selected_successes_summary": {
            "min": int(successes.min()),
            "median": float(np.median(successes)),
            "mean": float(successes.mean()),
            "max": int(successes.max()),
        },
        "selected_class_histogram": {str(k): int(v) for k, v in sorted(hist.items())},
        "distinct_selected_classes": len(hist),
        "max_selected_class_share": float(max(hist.values()) / n),
    }
    nonabstained = cert["radius"][~cert["abstained"]]
    out["min_abs_margin_radius_minus_primary_nonabstained"] = (
        float(np.min(np.abs(nonabstained - EXPECTED["primary_radius"]))) if nonabstained.size else None
    )
    return out


# --------------------------------------------------------------------------- #
# Paired statistics                                                            #
# --------------------------------------------------------------------------- #
def exact_mcnemar(first: np.ndarray, second: np.ndarray) -> tuple[int, int, float]:
    proposed_only = int(np.sum((first == 1) & (second == 0)))
    comparator_only = int(np.sum((first == 0) & (second == 1)))
    discordant = proposed_only + comparator_only
    if discordant == 0:
        return proposed_only, comparator_only, 1.0
    smaller = min(proposed_only, comparator_only)
    lower_tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2 ** discordant)
    return proposed_only, comparator_only, min(1.0, 2.0 * lower_tail)


def holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def macro_cell_mean(differences: np.ndarray, dataset_codes: np.ndarray, code_order: list[int]) -> float:
    values = []
    for code in code_order:
        rows = differences[dataset_codes == code]
        values.extend(float(rows[:, model].mean()) for model in range(rows.shape[1]))
    return float(np.mean(values))


def paired_class_stratified_bootstrap(differences: np.ndarray, dataset_names: list[str],
                                      labels: list[int], *, replicates: int, seed: int,
                                      confidence_level: float = 0.95, progress: str | None = None) -> dict:
    """Paired percentile bootstrap over test images, stratified by dataset x class.

    Each replicate resamples image rows with replacement inside every
    (dataset, ground-truth class) stratum, applies the same rows to both methods
    and every model column, and recomputes the macro (equal-cell) difference.
    Strata are visited in (str(dataset), str(class)) order, which is the order
    documented in the frozen analysis source, so the RNG stream is consumed in
    the same order and the interval can be compared numerically.
    """
    datasets = np.asarray(dataset_names, dtype=object)
    names_sorted = sorted(set(datasets.tolist()), key=str)
    code_of = {name: i for i, name in enumerate(names_sorted)}
    codes = np.array([code_of[name] for name in datasets.tolist()], dtype=np.int64)
    code_order = [code_of[name] for name in names_sorted]
    classes = np.asarray(labels, dtype=object)
    point = macro_cell_mean(differences, codes, code_order)
    keys = sorted(set(zip(datasets.tolist(), classes.tolist())), key=lambda k: (str(k[0]), str(k[1])))
    strata = [np.flatnonzero((datasets == d) & (classes == c)) for d, c in keys]
    generator = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    started = time.time()
    for replicate in range(replicates):
        sampled = np.concatenate(
            [generator.choice(indices, size=indices.size, replace=True) for indices in strata]
        )
        values[replicate] = macro_cell_mean(differences[sampled], codes[sampled], code_order)
        if progress and (replicate + 1) % 2000 == 0:
            print(f"  {progress}: {replicate + 1}/{replicates} replicates ({time.time() - started:.0f}s)", flush=True)
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(values, [tail, 1.0 - tail], method="linear")
    return {
        "point_difference": point,
        "lower": float(lower),
        "upper": float(upper),
        "confidence_level": confidence_level,
        "replicates": replicates,
        "seed": seed,
        "strata": len(strata),
        "method": "paired_class_stratified_percentile_bootstrap (independent re-implementation)",
    }


# --------------------------------------------------------------------------- #
# Reconstruction                                                               #
# --------------------------------------------------------------------------- #
def reconstruct(results_dir: Path, repo_root: Path = REPO_ROOT, *, bootstrap_mode: str = "planned",
                check_caches: bool = True, verbose: bool = True) -> dict:
    results_dir = Path(results_dir).resolve()
    repo_root = Path(repo_root).resolve()
    farla_root = repo_root / "FARLA"
    checks = Checks()
    log = print if verbose else (lambda *a, **k: None)
    started = time.time()

    def rel(path: Path) -> str:
        try:
            return path.resolve().relative_to(repo_root).as_posix()
        except ValueError:
            return str(path)

    # ---- configuration custody ------------------------------------------- #
    recorded_sha = (results_dir / "CONFIG_SHA256").read_text(encoding="utf-8").strip()
    config_bytes = (results_dir / "config_snapshot.json").read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    snapshot_canonical = canonical_json_sha256(config)
    checks.expect(
        "config.snapshot_canonical_hash_matches_CONFIG_SHA256",
        snapshot_canonical == recorded_sha,
        f"canonical JSON SHA-256 of config_snapshot.json = {snapshot_canonical}; CONFIG_SHA256 = {recorded_sha}",
    )
    repo_config_path = farla_root / "configs" / "phase4" / "farla_full_v1.json"
    if repo_config_path.is_file():
        checks.expect(
            "config.repository_config_byte_identical_to_snapshot",
            repo_config_path.read_bytes() == config_bytes,
            f"{rel(repo_config_path)} compared byte-for-byte with config_snapshot.json",
            repository_config_file_sha256=sha256_file(repo_config_path),
            snapshot_file_sha256=hashlib.sha256(config_bytes).hexdigest(),
        )
    else:
        checks.not_verified("config.repository_config_byte_identical_to_snapshot", "repository config file absent")
    checks.expect("config.experiment_id", config.get("experiment_id") == EXPECTED["experiment_id"],
                  f"experiment_id = {config.get('experiment_id')}")
    protocol = {
        "sigma": float(config["confirmation"]["sigma"]),
        "alpha": float(config["confirmation"]["alpha_per_example"]),
        "confirmation_draws": int(config["confirmation"]["confirmation_draws"]),
        "selection_draws": int(config["confirmation"]["selection_draws"]),
        "primary_radius": float(config["confirmation"]["primary_radius"]),
        "radii": [float(r) for r in config["metrics"]["radii"]],
        "bootstrap_replicates": int(config["metrics"]["bootstrap_replicates"]),
        "bootstrap_seed": int(config["metrics"]["bootstrap_seed"]),
        "split_seed": int(config["seed"]),
        "clean_macro_limit_points": float(config["metrics"]["clean_macro_drop_percentage_points_max"]),
        "clean_cell_limit_points": float(config["metrics"]["clean_any_cell_drop_percentage_points_max"]),
    }
    for dataset in config["datasets"]:
        for role in ("train_per_class", "selection_per_class", "confirmation_per_class"):
            protocol[f"{dataset['dataset_id']}.{role}"] = int(dataset[role])
    mismatches = {
        key: (protocol[key], EXPECTED[key]) for key in (
            "sigma", "alpha", "confirmation_draws", "selection_draws", "primary_radius", "radii",
            "bootstrap_replicates", "bootstrap_seed", "split_seed", "clean_macro_limit_points",
            "clean_cell_limit_points") if protocol[key] != EXPECTED[key]
    }
    for dataset in config["datasets"]:
        for role in ("train_per_class", "selection_per_class", "confirmation_per_class"):
            key = f"{dataset['dataset_id']}.{role}"
            if protocol[key] != EXPECTED[role]:
                mismatches[key] = (protocol[key], EXPECTED[role])
    checks.expect("config.protocol_constants_as_expected", not mismatches,
                  "sigma, alpha, draws, radii, bootstrap seed/replicates, split seed, role sizes, clean-gate limits",
                  mismatches=mismatches)
    sigma, alpha = protocol["sigma"], protocol["alpha"]
    radii = protocol["radii"]
    primary_key = radius_key(protocol["primary_radius"])
    k_min = minimum_successes(protocol["confirmation_draws"], alpha, protocol["primary_radius"], sigma)
    checks.expect("certification.minimum_successes_at_primary_radius",
                  k_min == EXPECTED["minimum_successes_at_primary_radius"],
                  f"smallest k with CP_lower(k, {protocol['confirmation_draws']}, {alpha}) >= Phi({protocol['primary_radius']}/{sigma}) is {k_min}",
                  population_probability_threshold=float(norm.cdf(protocol["primary_radius"] / sigma)),
                  lower_bound_at_kmin=clopper_pearson_lower(k_min, protocol["confirmation_draws"], alpha),
                  lower_bound_at_kmin_minus_one=clopper_pearson_lower(k_min - 1, protocol["confirmation_draws"], alpha))
    candidate_ids = ["no_correction"] + [v["variant_id"] for v in config["variants"]]
    variants_by_id = {v["variant_id"]: v for v in config["variants"]}
    model_ids = list(config["models"])
    dataset_specs = {d["dataset_id"]: d for d in config["datasets"]}
    dataset_order = [d["dataset_id"] for d in config["datasets"]]
    reference = config["metrics"]["reference_candidate"]
    primary_candidate = config["metrics"]["primary_candidate"]

    # ---- split ledgers ---------------------------------------------------- #
    log("reading split ledgers ...")
    split_manifest = json.loads((results_dir / "splits" / "manifest.json").read_text(encoding="utf-8"))
    assignments_path = results_dir / "splits" / "assignments.csv"
    assignments_sha = sha256_file(assignments_path)
    checks.expect("splits.assignments_sha256_matches_manifest",
                  assignments_sha == split_manifest.get("assignments_sha256"),
                  f"splits/assignments.csv SHA-256 = {assignments_sha}",
                  manifest_value=split_manifest.get("assignments_sha256"))
    checks.expect("splits.manifest_config_sha256", split_manifest.get("config_sha256") == recorded_sha,
                  "splits/manifest.json config_sha256 equals CONFIG_SHA256")
    checks.expect("splits.manifest_final_test_not_accessed",
                  split_manifest.get("phase3_final_test_accessed") is False,
                  "splits/manifest.json declares phase3_final_test_accessed=false")
    phase4_rows = read_csv_rows(assignments_path)
    phase4_by_item: dict[tuple[str, str], dict] = {}
    duplicate_items = 0
    for row in phase4_rows:
        key = (row["dataset_id"], row["item_id"])
        if key in phase4_by_item:
            duplicate_items += 1
        phase4_by_item[key] = row
    checks.expect("splits.phase4_item_ids_unique", duplicate_items == 0,
                  f"{len(phase4_rows)} rows, {duplicate_items} duplicate (dataset, item_id) keys")
    role_counts = Counter((r["dataset_id"], r["phase4_role"]) for r in phase4_rows)
    per_class = Counter((r["dataset_id"], r["phase4_role"], int(r["label"])) for r in phase4_rows)
    balance_ok = True
    balance_detail = {}
    for dataset_id, spec in dataset_specs.items():
        for role, size_key in (("train", "train_per_class"), ("selection", "selection_per_class"),
                               ("confirmation", "confirmation_per_class")):
            expected_total = spec["class_count"] * spec[size_key]
            balance_detail[f"{dataset_id}.{role}"] = role_counts[(dataset_id, role)]
            if role_counts[(dataset_id, role)] != expected_total:
                balance_ok = False
            for label in range(spec["class_count"]):
                if per_class[(dataset_id, role, label)] != spec[size_key]:
                    balance_ok = False
    checks.expect("splits.role_counts_and_per_class_balance", balance_ok,
                  "every (dataset, role) has class_count x per_class items and every class is balanced",
                  counts=balance_detail)
    index_dupes = sum(
        1 for c in Counter((r["dataset_id"], int(r["dataset_index"])) for r in phase4_rows).values() if c > 1
    )
    checks.expect("splits.roles_disjoint_by_item_id_and_dataset_index", duplicate_items == 0 and index_dupes == 0,
                  "an item appears in exactly one Phase-4 role (by item_id and by dataset_index)",
                  duplicate_dataset_indices=index_dupes)
    checks.expect("splits.all_rows_marked_reserved_not_accessed",
                  all(r["phase3_role"] == "reserved_not_accessed" for r in phase4_rows),
                  "phase3_role column is reserved_not_accessed on every Phase-4 row")
    rank_ok = all(
        r["phase4_rank_sha256"] == rank_hash(protocol["split_seed"], r["dataset_id"], r["item_id"])
        for r in phase4_rows
    )
    checks.expect("splits.rank_hash_recomputes", rank_ok,
                  f"phase4_rank_sha256 == sha256('{protocol['split_seed']}:<dataset_id>:<item_id>') on every row")

    # Phase-3 ledger cross-check and split re-derivation
    ledgers = {}
    for relative in PHASE3_LEDGERS:
        path = repo_root / relative
        if path.is_file():
            ledgers[relative] = {"sha256": sha256_file(path), "rows": None}
        else:
            checks.not_verified(f"phase3.ledger_present:{relative}", "file absent on this machine")
    declared_source = split_manifest.get("source_phase3_assignments")
    declared_sha = split_manifest.get("source_phase3_assignments_sha256")
    if ledgers:
        for relative, info in ledgers.items():
            checks.expect(f"phase3.ledger_sha256_matches_declared_source:{relative}", info["sha256"] == declared_sha,
                          f"SHA-256 {info['sha256']} vs declared {declared_sha} (declared source: {declared_source})")
        if len(ledgers) == 2:
            shas = {info["sha256"] for info in ledgers.values()}
            checks.expect("phase3.exp017_snapshot_identical_to_day14_assignments", len(shas) == 1,
                          "the EXP-017 assignment snapshot and the day-14 Phase-3 split file are byte-identical")
    role_overlap = {}
    rederivation_ok = None
    if ledgers:
        primary_ledger = next(iter(ledgers))
        if declared_source in ledgers:
            primary_ledger = declared_source
        phase3_rows = read_csv_rows(repo_root / primary_ledger)
        ledgers[primary_ledger]["rows"] = len(phase3_rows)
        phase3_by_item = {(r["dataset_id"], r["item_id"]): r for r in phase3_rows}
        missing = 0
        label_mismatch = 0
        index_mismatch = 0
        phase3_role_of_phase4 = Counter()
        for row in phase4_rows:
            src = phase3_by_item.get((row["dataset_id"], row["item_id"]))
            if src is None:
                missing += 1
                continue
            phase3_role_of_phase4[(row["dataset_id"], row["phase4_role"], src["phase3_role"])] += 1
            if int(src["label"]) != int(row["label"]):
                label_mismatch += 1
            if int(src["dataset_index"]) != int(row["dataset_index"]):
                index_mismatch += 1
        checks.expect("phase3.every_phase4_item_found_in_ledger", missing == 0,
                      f"{missing} Phase-4 items missing from {primary_ledger}")
        checks.expect("phase3.labels_and_dataset_indices_agree", label_mismatch == 0 and index_mismatch == 0,
                      f"label mismatches={label_mismatch}, dataset_index mismatches={index_mismatch}")
        for dataset_id in dataset_order:
            for role in ("train", "selection", "confirmation"):
                entry = {"reserved_not_accessed": phase3_role_of_phase4[(dataset_id, role, "reserved_not_accessed")]}
                for p3 in PHASE3_NON_RESERVED_ROLES:
                    entry[p3] = phase3_role_of_phase4[(dataset_id, role, p3)]
                role_overlap[f"{dataset_id}.{role}"] = entry
        confirmation_overlap = sum(
            role_overlap[f"{d}.confirmation"][p3] for d in dataset_order for p3 in PHASE3_NON_RESERVED_ROLES
        )
        any_overlap = sum(
            role_overlap[f"{d}.{role}"][p3] for d in dataset_order for role in ("train", "selection", "confirmation")
            for p3 in PHASE3_NON_RESERVED_ROLES
        )
        checks.expect("phase3.no_confirmation_item_in_exp017_development_validation_excluded_or_sealed_roles",
                      confirmation_overlap == 0,
                      f"{confirmation_overlap} confirmation items carry a non-reserved Phase-3 role",
                      overlap_by_role=role_overlap)
        checks.expect("phase3.no_phase4_item_in_any_non_reserved_phase3_role", any_overlap == 0,
                      f"{any_overlap} Phase-4 items (any role) carry a non-reserved Phase-3 role")
        # Re-derive the Phase-4 allocation from the Phase-3 reserve.
        rederived = {}
        eligible = defaultdict(lambda: defaultdict(list))
        for r in phase3_rows:
            if r["phase3_role"] == "reserved_not_accessed" and r["dataset_id"] in dataset_specs:
                eligible[r["dataset_id"]][int(r["label"])].append(r)
        for dataset_id, spec in dataset_specs.items():
            t, s, c = spec["train_per_class"], spec["selection_per_class"], spec["confirmation_per_class"]
            for label in range(spec["class_count"]):
                ranked = sorted(eligible[dataset_id][label],
                                key=lambda r: (rank_hash(protocol["split_seed"], dataset_id, r["item_id"]), r["item_id"]))
                for role, start, end in (("train", 0, t), ("selection", t, t + s), ("confirmation", t + s, t + s + c)):
                    for r in ranked[start:end]:
                        rederived[(dataset_id, r["item_id"])] = (role, label, int(r["dataset_index"]))
        actual = {(r["dataset_id"], r["item_id"]): (r["phase4_role"], int(r["label"]), int(r["dataset_index"]))
                  for r in phase4_rows}
        rederivation_ok = rederived == actual
        checks.expect("splits.phase4_allocation_rederives_from_phase3_reserve", rederivation_ok,
                      "ranking reserved items per class by (rank hash, item_id) and slicing 20/5/10 reproduces splits/assignments.csv exactly",
                      rederived_items=len(rederived), ledger_items=len(actual))

    role_items = {
        (d, role): [r for r in phase4_rows if r["dataset_id"] == d and r["phase4_role"] == role]
        for d in dataset_order for role in ("train", "selection", "confirmation")
    }

    # ---- report and cells ------------------------------------------------- #
    report_path = results_dir / "analysis" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checks.expect("report.config_sha256", report.get("config_sha256") == recorded_sha,
                  "analysis/report.json config_sha256 equals CONFIG_SHA256")
    checks.expect("report.primary_radius", float(report.get("primary_radius")) == protocol["primary_radius"],
                  f"report primary_radius = {report.get('primary_radius')}")
    report_cells = {(row["cell_id"], row["candidate_id"]): row for row in report["cells"]}
    report_aggregate = {row["candidate_id"]: row for row in report["aggregate"]}

    cells_out = {}
    flags_primary: dict[str, dict[str, np.ndarray]] = {}  # cell -> candidate -> standard flags at primary radius
    cell_truth: dict[str, np.ndarray] = {}
    per_cell_rows: dict[str, dict[str, dict]] = {}
    cell_ids = [f"{m}__{d}" for m in model_ids for d in dataset_order]
    for cell_id in sorted(cell_ids):
        model_id, dataset_id = cell_id.split("__", 1)
        log(f"reconstructing cell {cell_id} ...")
        cell_dir = results_dir / "confirmation" / cell_id
        checks.expect(f"cell.{cell_id}.COMPLETE_marker", (cell_dir / "COMPLETE").is_file(), "COMPLETE marker present")
        metrics = json.loads((cell_dir / "metrics.json").read_text(encoding="utf-8"))
        stats = load_pt(cell_dir / "sufficient_statistics.pt")
        checks.expect(f"cell.{cell_id}.provenance_hashes",
                      metrics.get("config_sha256") == recorded_sha and stats.get("config_sha256") == recorded_sha
                      and stats.get("experiment_id") == EXPECTED["experiment_id"]
                      and stats.get("model_id") == model_id and stats.get("dataset_id") == dataset_id,
                      "metrics.json and sufficient_statistics.pt carry CONFIG_SHA256, experiment, model and dataset IDs")
        checks.expect(f"cell.{cell_id}.final_test_not_accessed",
                      metrics.get("phase3_final_test_accessed") is False and stats.get("phase3_final_test_accessed") is False,
                      "both files declare phase3_final_test_accessed=false")
        checks.expect(f"cell.{cell_id}.candidate_ids_match_protocol", list(stats["candidate_ids"]) == candidate_ids,
                      f"candidate order = {list(stats['candidate_ids'])}")
        item_ids = list(stats["item_ids"])
        truth = to_numpy(stats["ground_truth"]).astype(np.int64)
        expected_rows = role_items[(dataset_id, "confirmation")]
        checks.expect(f"cell.{cell_id}.item_ids_equal_confirmation_role_in_ledger_order",
                      item_ids == [r["item_id"] for r in expected_rows],
                      f"{len(item_ids)} confirmation items; ledger confirmation rows = {len(expected_rows)}")
        checks.expect(f"cell.{cell_id}.ground_truth_equals_ledger_labels",
                      np.array_equal(truth, np.array([int(r["label"]) for r in expected_rows])),
                      "ground_truth tensor equals the label column of the confirmation rows")
        # Seeds
        sel_seed_ok = all(
            int(to_numpy(stats["selection_seeds"])[i]) == derive_seed(
                int(config["confirmation"]["selection_seed"]), model_id, dataset_id, item, "phase4_confirmation_class_selection")
            for i, item in enumerate(item_ids))
        conf_seed_ok = all(
            int(to_numpy(stats["confirmation_seeds"])[i]) == derive_seed(
                int(config["confirmation"]["confirmation_seed"]), model_id, dataset_id, item, "phase4_confirmation_probability_estimation")
            for i, item in enumerate(item_ids))
        seeds_distinct = not np.any(to_numpy(stats["selection_seeds"]) == to_numpy(stats["confirmation_seeds"]))
        checks.expect(f"cell.{cell_id}.per_item_seeds_recompute_from_config_seeds",
                      sel_seed_ok and conf_seed_ok and seeds_distinct,
                      "selection/confirmation seeds equal derive_seed(config seed, model, dataset, item, purpose) and never collide")
        sel_all = to_numpy(stats["selection_counts"]).astype(np.int64)
        conf_all = to_numpy(stats["confirmation_counts"]).astype(np.int64)
        raw_all = to_numpy(stats["raw_predictions"]).astype(np.int64)
        checks.expect(f"cell.{cell_id}.vote_budgets",
                      bool((sel_all.sum(axis=2) == protocol["selection_draws"]).all())
                      and bool((conf_all.sum(axis=2) == protocol["confirmation_draws"]).all()),
                      f"every (candidate, item) row sums to {protocol['selection_draws']} selection and {protocol['confirmation_draws']} confirmation draws",
                      selection_shape=list(sel_all.shape), confirmation_shape=list(conf_all.shape))
        checks.expect(f"cell.{cell_id}.raw_predictions_shape", raw_all.shape == (len(candidate_ids), len(item_ids)),
                      f"raw_predictions shape = {list(raw_all.shape)} (per-candidate raw-clean predictions are stored)")
        banks = to_numpy(stats["candidate_banks"])
        metrics_by_candidate = {m["candidate_id"]: m for m in metrics["candidates"]}
        cell_truth[cell_id] = truth
        flags_primary[cell_id] = {}
        per_cell_rows[cell_id] = {}
        cell_out = {"model_id": model_id, "dataset_id": dataset_id, "n_items": len(item_ids),
                    "class_count": int(banks.shape[1]), "embedding_dim": int(banks.shape[2]),
                    "elapsed_seconds_reported": metrics.get("elapsed_seconds"), "candidates": {}}
        for index, candidate in enumerate(candidate_ids):
            cert = certify_candidate(sel_all[index], conf_all[index], raw_all[index], truth,
                                     sigma=sigma, alpha=alpha, radii=radii)
            summary = summarise_candidate(cert, primary_key, k_min)
            flags_primary[cell_id][candidate] = cert["per_radius"][primary_key]["standard"].astype(np.int64)
            # per-item comparison with the stored intermediate outcomes
            stored = stats["outcomes"][candidate]
            dev_lower = max_abs_dev(cert["lower"], to_numpy(stored["probability_lowers"]))
            dev_radius = max_abs_dev(cert["radius"], to_numpy(stored["selected_class_radii"]))
            same_selected = np.array_equal(cert["selected"], to_numpy(stored["selected_classes"]).astype(np.int64))
            same_successes = np.array_equal(cert["successes"], to_numpy(stored["selected_successes"]).astype(np.int64))
            same_abstained = np.array_equal(cert["abstained"], to_numpy(stored["abstained"]).astype(bool))
            flags_same = all(
                np.array_equal(cert["per_radius"][k][kind], to_numpy(stored["outcomes"][k][kind]).astype(bool))
                for k in cert["per_radius"] for kind in ("standard", "anchored", "wrong_stable")
            )
            stored_ok = (same_selected and same_successes and same_abstained and flags_same
                         and dev_lower <= TOLERANCE and dev_radius <= TOLERANCE)
            checks.expect(f"cell.{cell_id}.{candidate}.per_item_outcomes_match_stored", stored_ok,
                          "selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item",
                          max_abs_dev_lower=dev_lower, max_abs_dev_radius=dev_radius)
            # comparison with metrics.json
            m = metrics_by_candidate[candidate]
            devs = {
                "clean_accuracy": abs(summary["clean_accuracy"] - m["clean_accuracy"]),
                "smoothed_accuracy": abs(summary["smoothed_accuracy"] - m["smoothed_accuracy"]),
                "abstention_rate": abs(summary["abstention_rate"] - m["abstention_rate"]),
                "selection_raw_agreement": abs(summary["selection_raw_agreement"] - m["selection_raw_agreement"]),
                "average_selected_class_radius": abs(summary["average_selected_class_radius"] - m["average_selected_class_radius"]),
                "example_count": abs(summary["example_count"] - m["example_count"]),
            }
            for field in ("standard_certified_accuracy", "anchored_certified_accuracy", "certified_wrong_class_rate"):
                for k in summary[field]:
                    devs[f"{field}[{k}]"] = abs(summary[field][k] - m[field][k])
            metrics_dev = max(devs.values())
            checks.expect(f"cell.{cell_id}.{candidate}.metrics_json_recomputes", metrics_dev <= TOLERANCE,
                          f"maximum absolute deviation from metrics.json = {metrics_dev:.3e}", max_abs_dev=metrics_dev)
            # comparison with report.json cell row
            rrow = report_cells.get((cell_id, candidate))
            if rrow is None:
                checks.add(f"cell.{cell_id}.{candidate}.report_cell_row", "FAIL", "row absent from report.json")
                report_dev = float("inf")
            else:
                rdevs = {
                    "clean_accuracy": abs(summary["clean_accuracy"] - rrow["clean_accuracy"]),
                    "clean_correct_count": abs(summary["clean_correct_count"] - rrow["clean_correct_count"]),
                    "smoothed_accuracy": abs(summary["smoothed_accuracy"] - rrow["smoothed_accuracy"]),
                    "standard_ca_primary": abs(summary["standard_certified_accuracy"][primary_key] - rrow["standard_ca_primary"]),
                    "anchored_ca_primary": abs(summary["anchored_certified_accuracy"][primary_key] - rrow["anchored_ca_primary"]),
                    "certified_wrong_primary": abs(summary["certified_wrong_class_rate"][primary_key] - rrow["certified_wrong_primary"]),
                    "average_selected_class_radius": abs(summary["average_selected_class_radius"] - rrow["average_selected_class_radius"]),
                    "example_count": abs(summary["example_count"] - rrow["example_count"]),
                }
                report_dev = max(rdevs.values())
                checks.expect(f"cell.{cell_id}.{candidate}.report_cell_row_recomputes", report_dev <= TOLERANCE,
                              f"maximum absolute deviation from report.json cells row = {report_dev:.3e}", max_abs_dev=report_dev)
            summary["comparison"] = {
                "stored_per_item_outcomes": {"status": "PASS" if stored_ok else "FAIL",
                                             "max_abs_dev_lower_bound": dev_lower, "max_abs_dev_radius": dev_radius},
                "metrics_json": {"status": "PASS" if metrics_dev <= TOLERANCE else "FAIL", "max_abs_dev": metrics_dev},
                "report_cells": {"status": "PASS" if report_dev <= TOLERANCE else "FAIL", "max_abs_dev": report_dev},
            }
            cell_out["candidates"][candidate] = summary
            per_cell_rows[cell_id][candidate] = summary
        # bank provenance: trained checkpoints and feature caches
        for index, candidate in enumerate(candidate_ids):
            if candidate == reference:
                continue
            ckpt_path = results_dir / "trained" / cell_id / f"{candidate}.pt"
            if not ckpt_path.is_file():
                checks.not_verified(f"bank.{cell_id}.{candidate}.checkpoint", "trained checkpoint absent")
                continue
            ckpt = load_pt(ckpt_path)
            bank_same = np.array_equal(to_numpy(ckpt["candidate_prototypes"]), banks[index])
            variant_same = (ckpt.get("variant") == variants_by_id[candidate])
            summary_same = (ckpt.get("summary") == metrics["candidate_metadata"].get(candidate))
            checks.expect(f"bank.{cell_id}.{candidate}.checkpoint_bank_equals_confirmation_bank",
                          bank_same and ckpt.get("config_sha256") == recorded_sha and variant_same and summary_same,
                          "trained/<cell>/<variant>.pt candidate_prototypes equal candidate_banks[index] exactly; variant spec equals config; summary equals metrics.json metadata",
                          bank_equal=bank_same, variant_equal=variant_same, summary_equal=summary_same)
        if check_caches:
            for role in ("train", "selection"):
                cache_path = results_dir / "cache" / f"{cell_id}__{role}.pt"
                if not cache_path.is_file():
                    checks.not_verified(f"cache.{cell_id}.{role}", "feature cache absent")
                    continue
                cache = load_pt(cache_path)
                expected = role_items[(dataset_id, role)]
                ids_same = list(cache["item_ids"]) == [r["item_id"] for r in expected]
                labels_same = np.array_equal(to_numpy(cache["ground_truth"]).astype(np.int64),
                                             np.array([int(r["label"]) for r in expected]))
                idx_same = np.array_equal(to_numpy(cache["dataset_indices"]).astype(np.int64),
                                          np.array([int(r["dataset_index"]) for r in expected]))
                overlap = len(set(cache["item_ids"]) & set(item_ids))
                proto_same = np.array_equal(to_numpy(cache["prototypes"]), banks[0])
                checks.expect(f"cache.{cell_id}.{role}.items_equal_ledger_role_and_disjoint_from_confirmation",
                              ids_same and labels_same and idx_same and overlap == 0
                              and cache.get("role") == role and cache.get("config_sha256") == recorded_sha,
                              f"{role} cache holds exactly the ledger's {role} items ({len(expected)}), labels and indices agree, {overlap} overlap with confirmation items",
                              draws_per_example=int(cache.get("draws_per_example", -1)))
                checks.expect(f"cache.{cell_id}.{role}.frozen_prototypes_equal_no_correction_bank", proto_same,
                              "cache prototypes equal candidate_banks[no_correction] exactly")
                if role == "train":
                    formula = hashlib.sha256(np.ascontiguousarray(to_numpy(cache["prototypes"])).tobytes()).hexdigest()
                    if formula == cache.get("prototype_sha256"):
                        checks.expect(f"cache.{cell_id}.prototype_sha256_recomputes", True,
                                      "prototype_sha256 equals SHA-256 of the float32 row-major prototype bytes")
                    else:
                        checks.not_verified(f"cache.{cell_id}.prototype_sha256_recomputes",
                                            "prototype_sha256 formula is not documented in the outputs; the tensor equality above is the effective check")
        cells_out[cell_id] = cell_out

    # ---- aggregates --------------------------------------------------------- #
    log("aggregating ...")
    cells_sorted = sorted(cell_ids)
    aggregates = {}
    aggregate_table = []
    base_clean = {c: per_cell_rows[c][reference]["clean_correct_count"] for c in cells_sorted}
    for candidate in candidate_ids:
        rows = [per_cell_rows[c][candidate] for c in cells_sorted]
        drops = [Fraction(100 * (base_clean[c] - per_cell_rows[c][candidate]["clean_correct_count"]),
                          per_cell_rows[c][candidate]["example_count"]) for c in cells_sorted]
        macro_drop = sum(drops, Fraction()) / len(drops)
        worst_drop = max(drops)
        clean_ok = (macro_drop <= Fraction(str(protocol["clean_macro_limit_points"]))
                    and worst_drop <= Fraction(str(protocol["clean_cell_limit_points"])))
        agg = {
            "candidate_id": candidate,
            "cell_count": len(rows),
            "macro_clean_accuracy": float(np.mean([r["clean_accuracy"] for r in rows])),
            "macro_smoothed_accuracy": float(np.mean([r["smoothed_accuracy"] for r in rows])),
            "macro_standard_ca_primary": float(np.mean([r["standard_certified_accuracy"][primary_key] for r in rows])),
            "macro_anchored_ca_primary": float(np.mean([r["anchored_certified_accuracy"][primary_key] for r in rows])),
            "macro_certified_wrong_primary": float(np.mean([r["certified_wrong_class_rate"][primary_key] for r in rows])),
            "macro_average_selected_class_radius": float(np.mean([r["average_selected_class_radius"] for r in rows])),
            "macro_abstention_rate": float(np.mean([r["abstention_rate"] for r in rows])),
            "macro_clean_drop_percentage_points": float(macro_drop),
            "maximum_cell_clean_drop_percentage_points": float(worst_drop),
            "clean_constraint_satisfied": bool(clean_ok),
            "clean_drop_fractions_by_cell": {c: str(d) for c, d in zip(cells_sorted, drops)},
            "standard_ca_primary_by_cell": {c: per_cell_rows[c][candidate]["standard_certified_accuracy"][primary_key] for c in cells_sorted},
            "clean_accuracy_by_cell": {c: per_cell_rows[c][candidate]["clean_accuracy"] for c in cells_sorted},
        }
        ragg = report_aggregate.get(candidate)
        comparison = {}
        if ragg is None:
            comparison["status"] = "FAIL"
            comparison["detail"] = "candidate absent from report.json aggregate"
            dev = float("inf")
        else:
            field_devs = {}
            for field in ("macro_clean_accuracy", "macro_smoothed_accuracy", "macro_standard_ca_primary",
                          "macro_anchored_ca_primary", "macro_certified_wrong_primary",
                          "macro_average_selected_class_radius", "macro_clean_drop_percentage_points",
                          "maximum_cell_clean_drop_percentage_points", "cell_count"):
                field_devs[field] = abs(float(agg[field]) - float(ragg[field]))
            field_devs["clean_constraint_satisfied"] = 0.0 if agg["clean_constraint_satisfied"] == ragg["clean_constraint_satisfied"] else 1.0
            dev = max(field_devs.values())
            comparison = {"status": "PASS" if dev <= TOLERANCE else "FAIL", "max_abs_dev": dev,
                          "field_abs_dev": field_devs, "report_values": {k: ragg[k] for k in field_devs}}
        checks.expect(f"aggregate.{candidate}.matches_report", comparison["status"] == "PASS",
                      f"maximum absolute deviation over all aggregate fields = {dev:.3e}", max_abs_dev=dev)
        if candidate in TASK_STATED_HEADLINE:
            stated = TASK_STATED_HEADLINE[candidate]
            stated_dev = max(abs(agg[k] - v) for k, v in stated.items())
            comparison["task_stated_values"] = stated
            comparison["task_stated_max_abs_dev"] = stated_dev
            checks.expect(f"aggregate.{candidate}.matches_task_stated_headline_within_rounding", stated_dev <= 5e-5,
                          f"deviation from the 4-decimal values quoted in the custody task = {stated_dev:.2e}")
        agg["comparison_to_report"] = comparison
        aggregates[candidate] = agg
        aggregate_table.append({
            "candidate_id": candidate,
            "macro_standard_ca_primary": agg["macro_standard_ca_primary"],
            "report_macro_standard_ca_primary": None if ragg is None else ragg["macro_standard_ca_primary"],
            "macro_clean_accuracy": agg["macro_clean_accuracy"],
            "report_macro_clean_accuracy": None if ragg is None else ragg["macro_clean_accuracy"],
            "macro_anchored_ca_primary": agg["macro_anchored_ca_primary"],
            "report_macro_anchored_ca_primary": None if ragg is None else ragg["macro_anchored_ca_primary"],
            "max_abs_dev_all_fields": dev,
            "status": comparison["status"],
        })

    # ---- contrasts ------------------------------------------------------------ #
    def paired_matrix(candidate: str):
        parts, dataset_names, labels = [], [], []
        models_sorted = sorted(model_ids)
        for dataset_id in dataset_order:
            columns = [flags_primary[f"{m}__{dataset_id}"][candidate] for m in models_sorted]
            truths = [cell_truth[f"{m}__{dataset_id}"] for m in models_sorted]
            if any(not np.array_equal(truths[0], t) for t in truths[1:]):
                raise RuntimeError("paired model cells do not share ground truth")
            parts.append(np.stack(columns, axis=1))
            dataset_names.extend([dataset_id] * len(truths[0]))
            labels.extend(int(v) for v in truths[0].tolist())
        return np.concatenate(parts, axis=0).astype(np.float64), dataset_names, labels

    def contrast(proposed: str, comparator: str, run_bootstrap: bool, label: str) -> dict:
        p_matrix, names, labels = paired_matrix(proposed)
        c_matrix, _, _ = paired_matrix(comparator)
        differences = p_matrix - c_matrix
        names_arr = np.asarray(names, dtype=object)
        names_sorted = sorted(set(names), key=str)
        code_of = {n: i for i, n in enumerate(names_sorted)}
        codes = np.array([code_of[n] for n in names], dtype=np.int64)
        point = macro_cell_mean(differences, codes, [code_of[n] for n in names_sorted])
        cell_tests, raw_p = [], []
        for cell_id in cells_sorted:
            po, co, p = exact_mcnemar(flags_primary[cell_id][proposed], flags_primary[cell_id][comparator])
            raw_p.append(p)
            cell_tests.append({"cell_id": cell_id, "proposed_only": po, "comparator_only": co, "p_value": p})
        for t, adj in zip(cell_tests, holm(raw_p)):
            t["holm_adjusted_p_value"] = adj
        out = {"proposed": proposed, "comparator": comparator,
               "macro_point_difference": point,
               "macro_point_difference_from_aggregates": aggregates[proposed]["macro_standard_ca_primary"] - aggregates[comparator]["macro_standard_ca_primary"],
               "paired_items": int(differences.shape[0]), "models": sorted(model_ids),
               "cell_mcnemar": cell_tests}
        if run_bootstrap:
            out["bootstrap"] = paired_class_stratified_bootstrap(
                differences, names, labels, replicates=protocol["bootstrap_replicates"],
                seed=protocol["bootstrap_seed"], progress=label if verbose else None)
        del names_arr
        return out

    planned = {c["contrast_id"]: c for c in config["metrics"]["planned_contrasts"]}
    if bootstrap_mode == "none":
        bootstrap_ids = set()
    elif bootstrap_mode == "requested":
        bootstrap_ids = set(REQUESTED_CONTRASTS)
    elif bootstrap_mode == "planned":
        bootstrap_ids = set(planned)
    elif bootstrap_mode == "all":
        bootstrap_ids = set(planned) | {f"paired:{c}" for c in candidate_ids if c != reference}
    else:
        raise ValueError(bootstrap_mode)
    contrasts = {}
    for contrast_id, spec in planned.items():
        log(f"contrast {contrast_id} ({spec['proposed']} vs {spec['comparator']}) ...")
        result = contrast(spec["proposed"], spec["comparator"], contrast_id in bootstrap_ids, contrast_id)
        result["requested_in_custody_task"] = contrast_id in REQUESTED_CONTRASTS
        rep = report["planned_contrasts"].get(contrast_id)
        if rep is None:
            checks.add(f"contrast.{contrast_id}.present_in_report", "FAIL", "contrast absent from report.json")
        else:
            rb = rep["bootstrap"]
            result["report"] = {"bootstrap": rb, "cell_mcnemar": rep["cell_mcnemar"]}
            same_ids = rep["proposed"] == spec["proposed"] and rep["comparator"] == spec["comparator"]
            point_dev = abs(result["macro_point_difference"] - rb["point_difference"])
            checks.expect(f"contrast.{contrast_id}.point_difference_matches_report", same_ids and point_dev <= TOLERANCE,
                          f"macro paired point difference deviation = {point_dev:.3e}", max_abs_dev=point_dev)
            mc_dev = 0.0
            for mine, theirs in zip(result["cell_mcnemar"], rep["cell_mcnemar"]):
                if mine["cell_id"] != theirs["cell_id"]:
                    mc_dev = float("inf")
                    break
                mc_dev = max(mc_dev, abs(mine["proposed_only"] - theirs["proposed_only"]),
                             abs(mine["comparator_only"] - theirs["comparator_only"]),
                             abs(mine["p_value"] - theirs["p_value"]),
                             abs(mine["holm_adjusted_p_value"] - theirs["holm_adjusted_p_value"]))
            checks.expect(f"contrast.{contrast_id}.mcnemar_and_holm_match_report", mc_dev <= TOLERANCE,
                          f"discordant counts, exact p-values and Holm adjustment deviation = {mc_dev:.3e}", max_abs_dev=mc_dev)
            stated_ok = (rb.get("replicates") == protocol["bootstrap_replicates"] and rb.get("seed") == protocol["bootstrap_seed"]
                         and rb.get("method") == "paired_class_stratified_percentile_bootstrap")
            result["report_states_seed_and_replicates"] = stated_ok
            if "bootstrap" in result:
                mine = result["bootstrap"]
                b_dev = max(abs(mine["point_difference"] - rb["point_difference"]), abs(mine["lower"] - rb["lower"]),
                            abs(mine["upper"] - rb["upper"]))
                mine["max_abs_dev_from_report"] = b_dev
                mine["reproduced_exactly"] = b_dev <= TOLERANCE
                checks.expect(f"contrast.{contrast_id}.bootstrap_interval_reproduces", stated_ok and b_dev <= TOLERANCE,
                              f"re-implemented paired class-stratified percentile bootstrap (seed {protocol['bootstrap_seed']}, "
                              f"{protocol['bootstrap_replicates']} replicates) deviation = {b_dev:.3e}", max_abs_dev=b_dev)
            else:
                checks.not_verified(f"contrast.{contrast_id}.bootstrap_interval_reproduces",
                                    "bootstrap not run in this invocation (bootstrap mode excludes it); report interval carried as stated")
        contrasts[contrast_id] = result
    if bootstrap_mode == "all":
        contrasts["reference_paired_comparisons"] = {}
        for candidate in candidate_ids:
            if candidate == reference:
                continue
            log(f"paired comparison {candidate} vs {reference} ...")
            result = contrast(candidate, reference, True, f"paired:{candidate}")
            rep = report["paired_comparisons"].get(candidate)
            if rep is not None:
                rb = rep["bootstrap"]
                mine = result["bootstrap"]
                b_dev = max(abs(mine["point_difference"] - rb["point_difference"]), abs(mine["lower"] - rb["lower"]),
                            abs(mine["upper"] - rb["upper"]))
                mine["max_abs_dev_from_report"] = b_dev
                result["report"] = {"bootstrap": rb}
                checks.expect(f"paired.{candidate}.bootstrap_interval_reproduces", b_dev <= TOLERANCE,
                              f"deviation = {b_dev:.3e}", max_abs_dev=b_dev)
            contrasts["reference_paired_comparisons"][candidate] = result

    # ---- frozen decision rule ------------------------------------------------- #
    required = config["metrics"]["decision_required_contrasts"]
    decision_checks = {
        "clean_constraint": aggregates[primary_candidate]["clean_constraint_satisfied"],
        "positive_macro_gain": aggregates[primary_candidate]["macro_standard_ca_primary"] > aggregates[reference]["macro_standard_ca_primary"],
    }
    for contrast_id in required:
        c = contrasts[contrast_id]
        if "bootstrap" in c:
            decision_checks[f"paired_interval_above_zero:{contrast_id}"] = c["bootstrap"]["lower"] > 0.0
        elif "report" in c:
            decision_checks[f"paired_interval_above_zero:{contrast_id}"] = c["report"]["bootstrap"]["lower"] > 0.0
    decision = "CONTINUE_PAPER_CANDIDATE" if all(decision_checks.values()) else "DO_NOT_CLAIM_METHOD_SUCCESS"
    decision_out = {
        "recomputed_decision": decision,
        "recomputed_checks": decision_checks,
        "report_decision": report.get("decision"),
        "report_reasons": report.get("decision_reasons"),
        "interval_source": "recomputed bootstrap" if all("bootstrap" in contrasts[c] for c in required) else "report.json (bootstrap not rerun for a required contrast)",
        "note": ("The frozen rule compares the primary candidate with no_correction and the matched rank-8 noisy-CE adapter only. "
                 "It does not test the tail-only control; farla_vs_lowrank_tail is reported as a planned contrast outside the decision."),
    }
    checks.expect("decision.frozen_rule_recomputes", decision == report.get("decision"),
                  f"recomputed decision {decision}; report decision {report.get('decision')}")

    # ---- scientific boundary facts (numbers, not interpretation) ------------- #
    def macro(c, k):
        return aggregates[c][k]

    tail = contrasts.get("farla_vs_lowrank_tail", {})
    boundary = {
        "supervision": "All adapter candidates except farla_full_r8_pseudo are trained with ground-truth labels of the 20-per-class train role (supervised few-shot); no candidate is zero-shot except no_correction.",
        "tail_only_vs_full_farla": {
            "macro_standard_ca_primary": {"lowrank_tail_r8_supervised": macro("lowrank_tail_r8_supervised", "macro_standard_ca_primary"),
                                          "farla_full_r8_supervised": macro("farla_full_r8_supervised", "macro_standard_ca_primary")},
            "farla_minus_tail_macro_point": tail.get("macro_point_difference"),
            "farla_minus_tail_bootstrap": tail.get("bootstrap", tail.get("report", {}).get("bootstrap")),
            "statement": "tail-only rank-8 exceeds full FARLA rank-8 on the primary metric; the full objective is therefore not shown to be necessary",
        },
        "clean_accuracy_change_points_vs_no_correction": {
            c: -aggregates[c]["macro_clean_drop_percentage_points"] for c in candidate_ids if c != reference
        },
        "certified_wrong_class_rate_primary": {c: macro(c, "macro_certified_wrong_primary") for c in candidate_ids},
        "per_cell_standard_ca_primary": {c: aggregates[c]["standard_ca_primary_by_cell"] for c in candidate_ids},
        "eurosat_cell_size": {c: cells_out[c]["n_items"] for c in cells_sorted if c.endswith("__eurosat")},
    }

    # ---- what this reconstruction cannot verify ------------------------------- #
    not_verified = [
        "Monte-Carlo vote counts (selection_counts, confirmation_counts) cannot be regenerated here: that requires the GPU encoders, the frozen checkpoints and the source images. They are taken as stored; only their budgets (128/4096 per row), seeds and downstream arithmetic are verified.",
        "raw_predictions (per-candidate raw-clean predictions) are stored and used as given. The clean image features of the confirmation items are not stored anywhere in the result tree (only train/selection caches exist), so the clean argmax cannot be recomputed without re-encoding.",
        "Adapter training (80 epochs, losses, initialisation) is not re-run. Only the linkage checkpoint bank -> confirmation bank, the variant specification and the summary metadata are verified.",
        "screening_report.json is a diagnostic over the selection role (candidate_selection_performed=false); it is hashed in the manifest but not reconstructed.",
        "The bootstrap is an independent re-implementation of the resampling scheme documented in the frozen analysis source (FARLA/analysis/paired_bootstrap.py, SHA-256 9014b90062e3b2982916dc3fb5ebc041547d4957ab67dc75de26c915429852ed in the presampling approval). Exact numeric agreement depends on consuming the RNG stream in the same stratum order; the deviation is reported rather than assumed.",
        "Server-side execution facts (preflight.json, run.log, supervisord.log) are custody-hashed but cannot be re-executed or independently attested from this machine.",
    ]
    for item in checks.unverified:
        not_verified.append(f"{item['id']}: {item['detail']}")

    failures = checks.failures
    elapsed = time.time() - started
    devs = [c.get("max_abs_dev") for c in checks.items if isinstance(c.get("max_abs_dev"), (int, float))]
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPECTED["experiment_id"],
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "environment": {
            "python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "scipy": __import__("scipy").__version__,
            "torch": __import__("torch").__version__,
        },
        "script": {"path": rel(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH),
                   "imports_farla_code": False, "tolerance": TOLERANCE},
        "inputs": {"results_dir": rel(results_dir), "report_json_sha256": sha256_file(report_path),
                   "recorded_config_sha256": recorded_sha, "phase3_ledgers": {k: v["sha256"] for k, v in ledgers.items()},
                   "declared_source_phase3_assignments": declared_source,
                   "declared_source_phase3_assignments_sha256": declared_sha,
                   "phase4_assignments_sha256": assignments_sha},
        "protocol": {**protocol, "primary_radius_key": primary_key, "minimum_successes_at_primary_radius": k_min,
                     "candidate_ids": candidate_ids, "reference_candidate": reference, "primary_candidate": primary_candidate,
                     "bootstrap_mode": bootstrap_mode},
        "data_roles": {
            "role_counts": {f"{d}.{r}": role_counts[(d, r)] for d in dataset_order for r in ("train", "selection", "confirmation")},
            "phase3_role_of_phase4_items": role_overlap,
            "rederivation_matches_ledger": rederivation_ok,
            "description": {
                "train": "20 items per class; labels used for supervised adapter training (noisy-CE, clean-CE, tail, hubness, displacement, stable-error terms); pseudo variant uses clean zero-shot predictions as targets",
                "selection": "5 items per class; diagnostic screening only (screening_report.json, candidate_selection_performed=false); every predeclared candidate continued",
                "confirmation": "10 items per class; fresh 128-draw class selection and 4096-draw probability estimation per item; the only role entering report.json",
            },
        },
        "overall_status": "FAIL" if failures else "PASS",
        "failure_count": len(failures),
        "not_verified_count": len(checks.unverified),
        "max_abs_deviation_over_numeric_checks": max(devs) if devs else 0.0,
        "aggregate_comparison_table": aggregate_table,
        "aggregates": aggregates,
        "contrasts": contrasts,
        "decision_rule": decision_out,
        "scientific_boundary_facts": boundary,
        "cells": cells_out,
        "checks": checks.items,
        "failures": failures,
        "not_verified": not_verified,
    }


# --------------------------------------------------------------------------- #
# Rendering                                                                    #
# --------------------------------------------------------------------------- #
def format_table(rows: list[dict]) -> str:
    header = ["candidate", "std CA@0.25 recon", "report", "clean recon", "report", "anch CA recon", "report", "max|dev|", "status"]
    lines = [" | ".join(header)]
    for r in rows:
        def f(v):
            return "n/a" if v is None else f"{v:.10f}"
        lines.append(" | ".join([
            r["candidate_id"], f(r["macro_standard_ca_primary"]), f(r["report_macro_standard_ca_primary"]),
            f(r["macro_clean_accuracy"]), f(r["report_macro_clean_accuracy"]),
            f(r["macro_anchored_ca_primary"]), f(r["report_macro_anchored_ca_primary"]),
            f"{r['max_abs_dev_all_fields']:.3e}", r["status"]]))
    return "\n".join(lines)


def render_markdown(result: dict) -> str:
    p = result["protocol"]
    lines = [
        f"# Independent reconstruction report: {result['experiment_id']}",
        "",
        f"- Overall status: **{result['overall_status']}** ({result['failure_count']} FAIL, {result['not_verified_count']} NOT_VERIFIED, "
        f"{len(result['checks'])} checks)",
        f"- Generated (UTC): {result['generated_utc']}; elapsed {result['elapsed_seconds']:.0f} s",
        f"- Environment: Python {result['environment']['python']}, numpy {result['environment']['numpy']}, "
        f"scipy {result['environment']['scipy']}, torch {result['environment']['torch']}",
        f"- Script: `{result['script']['path']}` SHA-256 `{result['script']['sha256']}`; imports FARLA code: {result['script']['imports_farla_code']}",
        f"- Results directory: `{result['inputs']['results_dir']}`; CONFIG_SHA256 `{result['inputs']['recorded_config_sha256']}`",
        f"- Comparison tolerance: {result['script']['tolerance']:.0e}; maximum absolute deviation over all numeric checks: "
        f"{result['max_abs_deviation_over_numeric_checks']:.3e}",
        f"- Protocol: sigma {p['sigma']}, alpha {p['alpha']}, {p['selection_draws']} selection / {p['confirmation_draws']} confirmation draws, "
        f"primary radius {p['primary_radius']} (k_min = {p['minimum_successes_at_primary_radius']}), bootstrap {p['bootstrap_replicates']} x seed {p['bootstrap_seed']} (mode: {p['bootstrap_mode']})",
        "",
        "## Aggregate comparison with analysis/report.json (macro = equal-cell mean over 4 cells)",
        "",
        "| Candidate | Std CA@0.25 recon | report | Clean recon | report | Anchored CA recon | report | max abs dev (all fields) | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in result["aggregate_comparison_table"]:
        def f(v):
            return "n/a" if v is None else f"{v:.6f}"
        lines.append(f"| {r['candidate_id']} | {f(r['macro_standard_ca_primary'])} | {f(r['report_macro_standard_ca_primary'])} | "
                     f"{f(r['macro_clean_accuracy'])} | {f(r['report_macro_clean_accuracy'])} | {f(r['macro_anchored_ca_primary'])} | "
                     f"{f(r['report_macro_anchored_ca_primary'])} | {r['max_abs_dev_all_fields']:.3e} | {r['status']} |")
    lines += ["", "## Other macro quantities (recomputed)", "",
              "| Candidate | Smoothed acc | Abstention | Certified wrong@0.25 | Avg selected radius | Clean change vs no_correction (points) | Worst-cell clean drop (points) | Clean gate |",
              "|---|---:|---:|---:|---:|---:|---:|:---:|"]
    for c, a in result["aggregates"].items():
        lines.append(f"| {c} | {a['macro_smoothed_accuracy']:.4f} | {a['macro_abstention_rate']:.4f} | {a['macro_certified_wrong_primary']:.4f} | "
                     f"{a['macro_average_selected_class_radius']:.4f} | {-a['macro_clean_drop_percentage_points']:+.3f} | "
                     f"{a['maximum_cell_clean_drop_percentage_points']:+.2f} | {a['clean_constraint_satisfied']} |")
    lines += ["", "## Per-cell reconstruction (standard CA@0.25 / clean accuracy / abstentions / items with k >= k_min)", ""]
    cells = result["cells"]
    cell_ids = list(cells)
    lines.append("| Candidate | " + " | ".join(cell_ids) + " |")
    lines.append("|---|" + "---:|" * len(cell_ids))
    for c in result["protocol"]["candidate_ids"]:
        cols = []
        for cell in cell_ids:
            s = cells[cell]["candidates"][c]
            cols.append(f"{s['standard_certified_accuracy'][p['primary_radius_key']]:.3f} / {s['clean_accuracy']:.3f} / "
                        f"{s['abstained_count']}/{s['example_count']} / {s['items_with_successes_ge_kmin']}")
        lines.append(f"| {c} | " + " | ".join(cols) + " |")
    lines += ["", "## Planned contrasts (paired items, macro over cells)", "",
              "| Contrast | Proposed - comparator (recon) | report | Bootstrap 95% CI (recon) | report CI | Bootstrap status | Requested in task |",
              "|---|---:|---:|---:|---:|:---:|:---:|"]
    for cid, c in result["contrasts"].items():
        if cid == "reference_paired_comparisons":
            continue
        rb = c.get("report", {}).get("bootstrap", {})
        mine = c.get("bootstrap")
        rec_ci = f"[{mine['lower']:.5f}, {mine['upper']:.5f}]" if mine else "not run"
        rep_ci = f"[{rb.get('lower', float('nan')):.5f}, {rb.get('upper', float('nan')):.5f}]" if rb else "n/a"
        status = ("PASS" if mine and mine.get("reproduced_exactly") else ("FAIL" if mine else "NOT_VERIFIED"))
        lines.append(f"| {cid} ({c['proposed']} vs {c['comparator']}) | {c['macro_point_difference']:+.5f} | "
                     f"{rb.get('point_difference', float('nan')):+.5f} | {rec_ci} | {rep_ci} | {status} | {c.get('requested_in_custody_task')} |")
    lines += ["", "### Per-cell exact McNemar (proposed-only / comparator-only / p / Holm p)", ""]
    for cid, c in result["contrasts"].items():
        if cid == "reference_paired_comparisons":
            continue
        cells_txt = "; ".join(f"{t['cell_id'].split('__')[0].replace('openai-clip-', '')}/{t['cell_id'].split('__')[1]}: "
                              f"{t['proposed_only']}/{t['comparator_only']}/{t['p_value']:.2e}/{t['holm_adjusted_p_value']:.2e}"
                              for t in c["cell_mcnemar"])
        lines.append(f"- {cid}: {cells_txt}")
    d = result["decision_rule"]
    lines += ["", "## Frozen decision rule (recomputed)", "",
              f"- Recomputed decision: **{d['recomputed_decision']}**; report decision: **{d['report_decision']}** (interval source: {d['interval_source']})"]
    for k, v in d["recomputed_checks"].items():
        lines.append(f"- {k} = {v}")
    lines.append(f"- {d['note']}")
    b = result["scientific_boundary_facts"]
    t = b["tail_only_vs_full_farla"]
    lines += ["", "## Scientific boundary facts (numbers only)", "",
              f"- {b['supervision']}",
              f"- Tail-only rank-8 macro standard CA@0.25 = {t['macro_standard_ca_primary']['lowrank_tail_r8_supervised']:.4f} vs full FARLA rank-8 = "
              f"{t['macro_standard_ca_primary']['farla_full_r8_supervised']:.4f}; FARLA - tail = {t['farla_minus_tail_macro_point']:+.4f}"
              + (f", 95% CI [{t['farla_minus_tail_bootstrap']['lower']:.4f}, {t['farla_minus_tail_bootstrap']['upper']:.4f}]" if t.get("farla_minus_tail_bootstrap") else "")
              + f". {t['statement']}.",
              "- Clean-accuracy change vs no_correction (macro percentage points): "
              + ", ".join(f"{c} {v:+.2f}" for c, v in b["clean_accuracy_change_points_vs_no_correction"].items()),
              "- Certified-wrong-class rate at 0.25 (macro): " + ", ".join(f"{c} {v:.4f}" for c, v in b["certified_wrong_class_rate_primary"].items()),
              f"- EuroSAT confirmation cells hold {list(b['eurosat_cell_size'].values())[0]} items each; one item is one percentage point there."]
    lines += ["", "## Data roles", ""]
    for k, v in result["data_roles"]["role_counts"].items():
        lines.append(f"- {k}: {v} items")
    for k, v in result["data_roles"]["description"].items():
        lines.append(f"- {k}: {v}")
    lines.append(f"- Phase-4 allocation re-derived from the Phase-3 reserve matches the ledger: {result['data_roles']['rederivation_matches_ledger']}")
    if result["data_roles"]["phase3_role_of_phase4_items"]:
        lines.append("- Phase-3 role of every Phase-4 item (count by dataset.role -> phase3 role):")
        for k, v in result["data_roles"]["phase3_role_of_phase4_items"].items():
            lines.append(f"  - {k}: " + ", ".join(f"{r}={n}" for r, n in v.items()))
    lines += ["", "## Failures", ""]
    if result["failures"]:
        for f_ in result["failures"]:
            lines.append(f"- {f_['id']}: {f_['detail']}")
    else:
        lines.append("- none")
    lines += ["", "## Not verified by this reconstruction", ""]
    for item in result["not_verified"]:
        lines.append(f"- {item}")
    lines += ["", "## All checks", "", "| Check | Status | Detail |", "|---|:---:|---|"]
    for c in result["checks"]:
        detail = str(c["detail"]).replace("|", "\\|")
        lines.append(f"| `{c['id']}` | {c['status']} | {detail} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent reconstruction of the FARLA full-run results")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--out-dir", default=str(PACKAGE_DIR))
    parser.add_argument("--bootstrap", choices=["planned", "requested", "all", "none"], default="planned",
                        help="which paired bootstraps to re-run (planned = the five predeclared contrasts)")
    parser.add_argument("--skip-cache-check", action="store_true", help="do not open the feature caches (782 MB)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    result = reconstruct(Path(args.results), Path(args.repo_root), bootstrap_mode=args.bootstrap,
                         check_caches=not args.skip_cache_check, verbose=not args.quiet)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reconstruction_report.json").write_text(json.dumps(jsonable(result), indent=2) + "\n", encoding="utf-8")
    (out_dir / "reconstruction_report.md").write_text(render_markdown(result), encoding="utf-8")
    print()
    print("Aggregate comparison with analysis/report.json (tolerance 1e-9):")
    print(format_table(result["aggregate_comparison_table"]))
    print()
    for cid, c in result["contrasts"].items():
        if cid == "reference_paired_comparisons":
            continue
        mine = c.get("bootstrap")
        rb = c.get("report", {}).get("bootstrap", {})
        if mine:
            print(f"{cid}: point {c['macro_point_difference']:+.5f} (report {rb.get('point_difference', float('nan')):+.5f}); "
                  f"CI [{mine['lower']:.5f}, {mine['upper']:.5f}] (report [{rb.get('lower', float('nan')):.5f}, {rb.get('upper', float('nan')):.5f}]); "
                  f"max|dev| {mine['max_abs_dev_from_report']:.3e}; reproduced={mine['reproduced_exactly']}")
        else:
            print(f"{cid}: point {c['macro_point_difference']:+.5f} (report {rb.get('point_difference', float('nan')):+.5f}); bootstrap not run")
    print()
    print(f"decision: recomputed {result['decision_rule']['recomputed_decision']} / report {result['decision_rule']['report_decision']}")
    print(f"OVERALL: {result['overall_status']}  failures={result['failure_count']}  not_verified={result['not_verified_count']}  "
          f"max|dev|={result['max_abs_deviation_over_numeric_checks']:.3e}")
    if result["failures"]:
        for f_ in result["failures"]:
            print(f"  FAIL {f_['id']}: {f_['detail']}")
    print(f"wrote {out_dir / 'reconstruction_report.json'} and {out_dir / 'reconstruction_report.md'}")
    sys.exit(0 if result["overall_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
