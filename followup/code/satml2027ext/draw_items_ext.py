#!/usr/bin/env python3
"""Item lists and data roles for EXP-021A/B/C with a SHA-256 manifest.

Rules (all fixed here, all recorded in the manifest):

  * EXP-021A reuses the Phase-3 ``calibration_validation`` items per fold as
    evaluation and the ``calibration_development`` items of the same fold as
    development.  No draw: the lists are the assignments-file rows in file order.
  * EXP-021B evaluation lists are fresh, never-touched items:
      cifar100_test   30 per class from the official CIFAR-100 test split,
      eurosat_sealed  all 1,000 Phase-3 ``final_test_sealed`` items,
      imagenette      300 per class from the imagenette2-320 validation split.
    Development (10 per class CIFAR-100, 30 per class EuroSAT, 30 per class
    Imagenette train) and control-train (20 per class) are drawn from the
    Phase-3 ``reserved_not_accessed`` pool minus Phase-4 consumption minus the
    EXP-019 extension items (CIFAR-100, EuroSAT) or from the Imagenette training
    split, development first, then control-train, disjoint.
  * Every draw is deterministic: within a class, items are ordered by
    ``sha256("{salt}:{dataset_id}:{item_id}")`` with the salt
    ``"{experiment_id}:item_draw:v1"``; the first ``per_class`` are taken.  The
    registered order of a list is the class round-robin over those per-class
    ranks (item k is the (k // C)-th ranked item of class k mod C), so any prefix
    of the list is class-balanced.
  * EXP-021C uses the first 100 evaluation items per dataset in that order.
  * Lists whose pixels or labels are not available locally (CIFAR-100 test
    labels, the Imagenette folder) fail closed; with ``--skip-unavailable`` they
    are recorded in the manifest as ``PENDING_ITEM_LIST`` instead, which the
    registration builder refuses to finalize.

Usage:
  python satml2027ext/draw_items_ext.py --out results/satml2027ext/items \
      [--data-root data] [--skip-unavailable] [--created-at 2026-09-25T00:00:00+00:00]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _common import PROJECT_ROOT, canonical_json_hash, read_json, sha256_file, sha256_id_list, write_json_atomic  # noqa: E402
from datasets_ext import (  # noqa: E402
    CIFAR100_CLASS_COUNT,
    CIFAR100_TEST_MD5,
    IMAGENETTE_DOWNLOAD_SIZE_NOTE,
    IMAGENETTE_DOWNLOAD_URL,
    IMAGENETTE_OFFICIAL_COUNTS,
    DatasetUnavailableError,
    cifar100_test_labels,
    eurosat_sealed_items,
    imagenette_split_items,
)

SCHEMA_VERSION = "satml2027ext.item_manifest.v1"
PENDING = "PENDING_ITEM_LIST"
EXPERIMENT_IDS = {
    "021a": "EXP-20260921-021A",
    "021b": "EXP-20260921-021B",
    "021c": "EXP-20260921-021C",
}
CSV_FIELDS = ["dataset_id", "item_id", "dataset_index", "label", "source_split"]

PHASE3_FOLDS = ("0", "1", "2")
PHASE3_EVALUATION_ROLE = "calibration_validation"
PHASE3_DEVELOPMENT_ROLE = "calibration_development"
PHASE3_RESERVE_ROLE = "reserved_not_accessed"
PHASE3_SEALED_ROLE = "final_test_sealed"
PHASE3_CONSUMED_ROLES = (
    "calibration_development", "calibration_validation", "excluded_explicit", "excluded_prior_access", "final_test_sealed",
)

PLAN_021A = {"cifar100": {"class_count": 100, "per_fold": 500}, "eurosat": {"class_count": 10, "per_fold": 50}}
PLAN_021B = {
    "cifar100_test": {
        "class_count": 100,
        "evaluation": {"per_class": 30, "source": "cifar100_official_test"},
        "development": {"per_class": 10, "source": "cifar100_reserve"},
        "control_train": {"per_class": 20, "source": "cifar100_reserve"},
    },
    "eurosat_sealed": {
        "class_count": 10,
        "evaluation": {"per_class": 100, "source": "eurosat_sealed"},
        "development": {"per_class": 30, "source": "eurosat_reserve"},
        "control_train": {"per_class": 20, "source": "eurosat_reserve"},
    },
    "imagenette": {
        "class_count": 10,
        "evaluation": {"per_class": 300, "source": "imagenette_val"},
        "development": {"per_class": 30, "source": "imagenette_train"},
        "control_train": {"per_class": 20, "source": "imagenette_train"},
    },
}
ROLE_ORDER = ("evaluation", "development", "control_train")
SUBSET_021C = 100

DEFAULT_PHASE3 = Path("artifacts/day14/phase3_splits_v1/assignments.csv")
DEFAULT_PHASE4 = Path("FARLA/downloaded_results/EXP-20260917-020-FARLA-FULL/splits/assignments.csv")
DEFAULT_EXP019_ITEMS = Path("results/satml2027/items")
EXP019_DATASETS = ("cifar100", "eurosat")


# --------------------------------------------------------------------------- #
# Deterministic ranking
# --------------------------------------------------------------------------- #


def draw_salt(experiment_id: str) -> str:
    return f"{experiment_id}:item_draw:v1"


def rank_key(salt: str, dataset_id: str, item_id: str) -> str:
    return hashlib.sha256(f"{salt}:{dataset_id}:{item_id}".encode("utf-8")).hexdigest()


def stratified_draw(pool: Iterable[dict[str, str]], per_class: int, exclude: set[str], salt: str,
                    *, class_count: int) -> list[dict[str, str]]:
    """First ``per_class`` items per class by salted SHA-256 rank, class round-robin order."""

    by_class: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in pool:
        if row["item_id"] in exclude:
            continue
        by_class[int(row["label"])].append(row)
    if sorted(by_class) != list(range(class_count)):
        raise RuntimeError(f"pool classes {sorted(by_class)} differ from range({class_count})")
    ranked: list[list[dict[str, str]]] = []
    for label in range(class_count):
        candidates = sorted(by_class[label], key=lambda row: rank_key(salt, row["dataset_id"], row["item_id"]))
        if len(candidates) < per_class:
            raise RuntimeError(f"class {label} has only {len(candidates)} available items, need {per_class}")
        ranked.append(candidates[:per_class])
    return [ranked[k % class_count][k // class_count] for k in range(per_class * class_count)]


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def project_relative(path: Path | str) -> str:
    """Project-relative POSIX path, or the bare name when outside the project (no local absolute paths in artifacts)."""

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.name


def sanitize_reason(message: str, data_root: Path | str | None = None) -> str:
    """Strip local absolute prefixes (project root, data root) from an error message before recording it."""

    replacements = [(str(PROJECT_ROOT.resolve()), "<project>")]
    if data_root is not None:
        replacements.insert(0, (str(Path(data_root).resolve()), "<data-root>"))
    for prefix, token in replacements:
        message = message.replace(prefix, token).replace(prefix.replace("\\", "/"), token)
    return message


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def phase4_consumed_ids(phase4_assignments: Path | None) -> set[tuple[str, str]]:
    if phase4_assignments is None:
        return set()
    return {(row["dataset_id"], row["item_id"]) for row in load_rows(phase4_assignments)}


def exp019_item_ids(items_dir: Path | None) -> tuple[dict[str, set[str]], dict[str, object] | None]:
    """EXP-019 item ids per dataset (evaluation + development + control_train), if present locally."""

    if items_dir is None or not (items_dir / "item_manifest.json").is_file():
        return {dataset: set() for dataset in EXP019_DATASETS}, None
    manifest = read_json(items_dir / "item_manifest.json")
    ids: dict[str, set[str]] = {dataset: set() for dataset in EXP019_DATASETS}
    for dataset in EXP019_DATASETS:
        entry = manifest["datasets"].get(dataset)
        if entry is None:
            continue
        for role, split in entry["splits"].items():
            rows = load_rows(items_dir / split["csv"])
            listed = [row["item_id"] for row in rows]
            if sha256_id_list(listed) != split["item_ids_sha256"]:
                raise RuntimeError(f"EXP-019 {dataset} {role} CSV does not match its manifest hash")
            ids[dataset].update(listed)
    provenance = {
        "path": project_relative(items_dir),
        "item_manifest_sha256": canonical_json_hash(manifest),
        "experiment_id": manifest.get("experiment_id"),
        "excluded_count": {dataset: len(ids[dataset]) for dataset in EXP019_DATASETS},
    }
    return ids, provenance


def reserve_pool(phase3_rows: list[dict[str, str]], dataset_id: str, consumed: set[tuple[str, str]],
                 exp019: set[str]) -> list[dict[str, str]]:
    return [
        {"dataset_id": row["dataset_id"], "item_id": row["item_id"], "dataset_index": row["dataset_index"],
         "label": row["label"], "source_split": f"{dataset_id}_reserved_not_accessed"}
        for row in phase3_rows
        if row["dataset_id"] == dataset_id
        and row["phase3_role"] == PHASE3_RESERVE_ROLE
        and not row.get("exclusion_reason")
        and (row["dataset_id"], row["item_id"]) not in consumed
        and row["item_id"] not in exp019
    ]


def phase3_fold_rows(phase3_rows: list[dict[str, str]], dataset_id: str, role: str, fold: str) -> list[dict[str, str]]:
    return [
        {"dataset_id": row["dataset_id"], "item_id": row["item_id"], "dataset_index": row["dataset_index"],
         "label": row["label"], "source_split": f"phase3_{role}_fold{fold}"}
        for row in phase3_rows
        if row["dataset_id"] == dataset_id and row["phase3_role"] == role and row["fold"] == fold
    ]


def cifar100_test_pool(data_root: Path, *, require_official: bool) -> list[dict[str, str]]:
    labels = cifar100_test_labels(data_root, require_official=require_official)
    return [
        {"dataset_id": "cifar100", "item_id": f"cifar100-test-{index}", "dataset_index": str(index),
         "label": str(label), "source_split": "cifar100_official_test"}
        for index, label in enumerate(labels)
    ]


def imagenette_pool(data_root: Path, split: str, *, require_official: bool) -> list[dict[str, str]]:
    return [dict(row, source_split=f"imagenette_{split}")
            for row in imagenette_split_items(data_root, split, require_official=require_official)]


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #


def write_csv(path: Path, rows: list[dict[str, str]]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row[field] for field in CSV_FIELDS} for row in rows)
    item_ids = [row["item_id"] for row in rows]
    return {
        "status": "written",
        "csv": path.name,
        "csv_sha256": sha256_file(path),
        "count": len(rows),
        "item_ids_sha256": sha256_id_list(item_ids),
    }


def pending_entry(reason: str, resolution: str) -> dict[str, object]:
    return {"status": "pending", "item_ids_sha256": PENDING, "reason": reason, "resolution": resolution}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def build(args: argparse.Namespace) -> dict[str, object]:
    out = Path(args.out)
    phase3_rows = load_rows(args.phase3_assignments)
    consumed = phase4_consumed_ids(args.phase4_assignments)
    exp019_ids, exp019_provenance = exp019_item_ids(args.exp019_items)
    explicit_excluded = sorted(row["item_id"] for row in phase3_rows if row["phase3_role"] == "excluded_explicit")
    require_official = not args.allow_synthetic
    pending: list[dict[str, object]] = []

    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": args.created_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "plan": "research/EXP021_POWERED_FOLLOWUP_PLAN_2026-09-25.md Section 2.1",
        "draw_rule": "within each class ascending sha256('{salt}:{dataset_id}:{item_id}'), salt '{experiment_id}:item_draw:v1'; first per_class taken; evaluation before development before control_train, all disjoint",
        "order_rule": "class round-robin over per-class rank: item k is the (k // C)-th ranked item of class (k mod C); EXP-021A lists keep assignments-file row order",
        "synthetic_inputs_allowed": bool(args.allow_synthetic),
        "inputs": {
            "phase3_assignments": {"path": project_relative(args.phase3_assignments), "sha256": sha256_file(args.phase3_assignments)},
            "phase4_assignments": (
                {"path": project_relative(args.phase4_assignments), "sha256": sha256_file(args.phase4_assignments),
                 "consumed_count": len(consumed)}
                if args.phase4_assignments is not None else {"path": None, "applied": False}
            ),
            "exp019_items": exp019_provenance if exp019_provenance is not None else {"applied": False, "reason": "EXP-019 item manifest not found locally"},
            "cifar100_official_test": {
                "layout": "cifar-100-python/test (torchvision CIFAR100 train=False)",
                "required_test_batch_md5": CIFAR100_TEST_MD5,
                "verified_by_loader": require_official,
            },
            "imagenette_archive": (
                {"url": IMAGENETTE_DOWNLOAD_URL, "file": Path(args.imagenette_archive).name,
                 "bytes": Path(args.imagenette_archive).stat().st_size, "sha256": sha256_file(args.imagenette_archive),
                 "official_counts": dict(IMAGENETTE_OFFICIAL_COUNTS)}
                if getattr(args, "imagenette_archive", None) is not None
                else {"url": IMAGENETTE_DOWNLOAD_URL, "sha256": None, "note": "archive hash not recorded (pass --imagenette-archive)",
                      "official_counts": dict(IMAGENETTE_OFFICIAL_COUNTS)}
            ),
        },
        "exclusions": {
            "phase3_roles_never_drawn_from": list(PHASE3_CONSUMED_ROLES),
            "phase3_explicit_excluded_item_ids": explicit_excluded,
            "phase4_consumed_excluded": args.phase4_assignments is not None,
            "exp019_items_excluded": exp019_provenance is not None,
        },
        "final_test_access": {
            "eurosat_final_test_sealed_ids_listed": True,
            "cifar100_official_test_labels_read": False,
            "pixels_loaded": False,
            "consumed_once_rule": "the sealed EuroSAT holdout and the CIFAR-100 test split are consumed by EXP-021B; no later experiment may reuse them for selection",
        },
        "studies": {},
        "pending": pending,
    }

    # ---- EXP-021A: Phase-3 folds, no draw ---------------------------------- #
    exp_a = EXPERIMENT_IDS["021a"]
    study_a: dict[str, object] = {"draw_salt": None, "draw": "none (Phase-3 fold lists in assignments-file order)", "datasets": {}}
    for dataset_id, plan in PLAN_021A.items():
        folds: dict[str, object] = {}
        for fold in PHASE3_FOLDS:
            entry: dict[str, object] = {}
            for role, phase3_role in (("evaluation", PHASE3_EVALUATION_ROLE), ("development", PHASE3_DEVELOPMENT_ROLE)):
                rows = phase3_fold_rows(phase3_rows, dataset_id, phase3_role, fold)
                if len(rows) != plan["per_fold"]:
                    raise RuntimeError(f"{dataset_id} fold {fold} {phase3_role} has {len(rows)} rows, expected {plan['per_fold']}")
                info = write_csv(out / f"exp021a__{dataset_id}__fold{fold}__{role}.csv", rows)
                info["phase3_role"] = phase3_role
                info["per_class"] = plan["per_fold"] // plan["class_count"]
                entry[role] = info
            if {row["item_id"] for row in load_rows(out / entry["evaluation"]["csv"])} & {row["item_id"] for row in load_rows(out / entry["development"]["csv"])}:
                raise RuntimeError(f"{dataset_id} fold {fold}: evaluation and development overlap")
            folds[fold] = entry
        study_a["datasets"][dataset_id] = {"class_count": plan["class_count"], "data_role": "phase3_calibration_validation_same_items_fresh_noise", "folds": folds}
    manifest["studies"][exp_a] = study_a

    # ---- EXP-021B: fresh items ------------------------------------------- #
    exp_b = EXPERIMENT_IDS["021b"]
    salt_b = draw_salt(exp_b)
    study_b: dict[str, object] = {"draw_salt": salt_b, "datasets": {}}
    evaluation_rows_by_dataset: dict[str, list[dict[str, str]] | None] = {}

    def pool_for(source: str) -> list[dict[str, str]]:
        if source == "cifar100_official_test":
            rows = cifar100_test_pool(args.data_root, require_official=require_official)
            manifest["final_test_access"]["cifar100_official_test_labels_read"] = True
            return rows
        if source == "eurosat_sealed":
            return [dict(row, source_split="eurosat_final_test_sealed") for row in eurosat_sealed_items(args.phase3_assignments)]
        if source == "cifar100_reserve":
            return reserve_pool(phase3_rows, "cifar100", consumed, exp019_ids["cifar100"])
        if source == "eurosat_reserve":
            return reserve_pool(phase3_rows, "eurosat", consumed, exp019_ids["eurosat"])
        if source == "imagenette_val":
            return imagenette_pool(args.data_root, "val", require_official=require_official)
        if source == "imagenette_train":
            return imagenette_pool(args.data_root, "train", require_official=require_official)
        raise KeyError(source)

    resolutions = {
        "cifar100_official_test": "make data/cifar-100-python/test readable (currently permission denied locally) or extract the official cifar-100-python.tar.gz; the batch MD5 must equal torchvision's",
        "imagenette_val": f"owner approval to download {IMAGENETTE_DOWNLOAD_URL} ({IMAGENETTE_DOWNLOAD_SIZE_NOTE}) and unpack under data/",
        "imagenette_train": f"owner approval to download {IMAGENETTE_DOWNLOAD_URL} ({IMAGENETTE_DOWNLOAD_SIZE_NOTE}) and unpack under data/",
    }

    for dataset_key, plan in PLAN_021B.items():
        class_count = int(plan["class_count"])
        taken: set[str] = set()
        splits: dict[str, object] = {}
        pools: dict[str, list[dict[str, str]]] = {}
        for role in ROLE_ORDER:
            spec = plan[role]
            source = spec["source"]
            try:
                if source not in pools:
                    pools[source] = pool_for(source)
                pool = pools[source]
            except DatasetUnavailableError as error:
                if not args.skip_unavailable:
                    raise SystemExit(
                        f"FAIL CLOSED: cannot draw {exp_b} {dataset_key} {role} ({source}): {error}\n"
                        f"Resolution: {resolutions.get(source, 'provide the data locally')}\n"
                        "Re-run with --skip-unavailable to record the list as PENDING_ITEM_LIST instead."
                    ) from error
                reason = sanitize_reason(str(error), args.data_root)
                splits[role] = dict(pending_entry(reason, resolutions.get(source, "provide the data locally")),
                                    per_class=spec["per_class"], planned_count=spec["per_class"] * class_count, source=source)
                pending.append({"study": exp_b, "dataset": dataset_key, "role": role, "source": source, "reason": reason})
                if role == "evaluation":
                    evaluation_rows_by_dataset[dataset_key] = None
                continue
            rows = stratified_draw(pool, int(spec["per_class"]), taken, salt_b, class_count=class_count)
            taken.update(row["item_id"] for row in rows)
            info = write_csv(out / f"exp021b__{dataset_key}__{role}.csv", rows)
            info.update({"per_class": spec["per_class"], "source": source, "pool_size": len(pool)})
            splits[role] = info
            if role == "evaluation":
                evaluation_rows_by_dataset[dataset_key] = rows
        study_b["datasets"][dataset_key] = {
            "class_count": class_count,
            "data_role": "fresh_never_touched_items",
            "evaluation_source": plan["evaluation"]["source"],
            "splits": splits,
        }
    manifest["studies"][exp_b] = study_b

    # ---- EXP-021C: subset of the 021B evaluation lists -------------------- #
    exp_c = EXPERIMENT_IDS["021c"]
    study_c: dict[str, object] = {"subset_size": SUBSET_021C, "rule": f"first {SUBSET_021C} evaluation items of {exp_b} per dataset in registered order", "datasets": {}}
    for dataset_key in PLAN_021B:
        rows = evaluation_rows_by_dataset.get(dataset_key)
        if rows is None:
            study_c["datasets"][dataset_key] = dict(pending_entry(f"{exp_b} {dataset_key} evaluation list is pending", "resolve the parent list first"), planned_count=SUBSET_021C)
            pending.append({"study": exp_c, "dataset": dataset_key, "role": "subset", "source": f"{exp_b}:evaluation", "reason": "parent evaluation list pending"})
            continue
        subset = rows[:SUBSET_021C]
        info = write_csv(out / f"exp021c__{dataset_key}__subset.csv", subset)
        info["parent_item_ids_sha256"] = study_b["datasets"][dataset_key]["splits"]["evaluation"]["item_ids_sha256"]
        info["per_class"] = SUBSET_021C // int(PLAN_021B[dataset_key]["class_count"])
        study_c["datasets"][dataset_key] = info
    manifest["studies"][exp_c] = study_c

    path = write_json_atomic(out / "item_manifest_ext.json", manifest)
    manifest["_manifest_path"] = str(path)
    manifest["_manifest_sha256"] = canonical_json_hash({key: value for key, value in manifest.items() if not key.startswith("_")})
    return manifest


def summary_lines(manifest: dict[str, object]) -> list[str]:
    """One line per list: study, dataset, role, count, status, hash."""

    def fmt(study: str, dataset: str, role: str, info: dict[str, object]) -> str:
        count = info.get("count", info.get("planned_count"))
        return f"{study} {dataset:15s} {role:22s} n={count!s:>5} {info['status']:8s} {info['item_ids_sha256']}"

    lines: list[str] = []
    for study, data in manifest["studies"].items():  # type: ignore[union-attr]
        for dataset, entry in data["datasets"].items():
            if "folds" in entry:
                for fold, roles in entry["folds"].items():
                    for role, info in roles.items():
                        lines.append(fmt(study, dataset, f"fold{fold}/{role}", info))
            elif "splits" in entry:
                for role, info in entry["splits"].items():
                    lines.append(fmt(study, dataset, role, info))
            else:
                lines.append(fmt(study, dataset, "subset", entry))
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase3-assignments", type=Path, default=PROJECT_ROOT / DEFAULT_PHASE3)
    parser.add_argument("--phase4-assignments", type=Path, default=PROJECT_ROOT / DEFAULT_PHASE4,
                        help="Phase-4 splits/assignments.csv (consumed reserve items); pass '' to disable")
    parser.add_argument("--exp019-items", type=Path, default=PROJECT_ROOT / DEFAULT_EXP019_ITEMS,
                        help="EXP-019 item directory (excluded from every role when its manifest is present)")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data",
                        help="folder holding cifar-100-python/, imagenette2-320/ and eurosat/ (data_ext/ locally, see D-130)")
    parser.add_argument("--imagenette-archive", type=Path, default=None,
                        help="the downloaded imagenette2-320.tgz; its SHA-256 is recorded in the manifest so the sampling host can verify its copy")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results/satml2027ext/items")
    parser.add_argument("--skip-unavailable", action="store_true",
                        help="record lists whose data is absent as PENDING_ITEM_LIST instead of failing")
    parser.add_argument("--created-at", default=None, help="frozen UTC ISO timestamp for byte-identical regeneration")
    parser.add_argument("--allow-synthetic", action="store_true",
                        help="TESTS ONLY: skip the official-data integrity checks; recorded in the manifest")
    args = parser.parse_args(argv)
    if args.phase4_assignments is not None and str(args.phase4_assignments) in ("", "."):
        args.phase4_assignments = None
    if args.phase4_assignments is not None and not args.phase4_assignments.is_file():
        raise SystemExit(f"Phase-4 assignments file not found: {args.phase4_assignments} (pass '' to disable)")
    if args.imagenette_archive is not None and not args.imagenette_archive.is_file():
        raise SystemExit(f"Imagenette archive not found: {args.imagenette_archive}")

    manifest = build(args)
    for line in summary_lines(manifest):
        print(line)
    print(f"manifest: {manifest['_manifest_path']}")
    print(f"manifest canonical sha256: {manifest['_manifest_sha256']}")
    if manifest["pending"]:
        print(f"PENDING lists: {len(manifest['pending'])} (registrations built from this manifest stay drafts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
