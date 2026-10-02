#!/usr/bin/env python3
"""Day 0, step 2 - draw and register the EXP-019 evaluation and training items.

Rules, all preregistered and recorded in the output:

  * CIFAR-100 and EuroSAT items come from the Phase-3 `reserved_not_accessed`
    pool minus everything Phase-4 consumed, so they have never been accessed.
  * CIFAR-10 is new to the project: items come from the official *training*
    split.  The official test split is never read.
  * Within each dataset the draw is class-stratified and deterministic: items
    are ordered by sha256("{salt}:{dataset_id}:{item_id}") and the first
    `per_class` of each class are taken.  The salt is the experiment id, so the
    order is independent of the Phase-3 allocation and reproducible by anyone.
  * Development items (for direction estimation and control-bank training) are
    drawn from the same pool and are disjoint from the evaluation items.

Usage:
  python day0/d0_02_draw_items.py --phase3-assignments ... --phase4-results ... \
      --data-root data --out results/satml2027/items
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.hashing import sha256_id_list, write_json_atomic  # noqa: E402

EXPERIMENT_ID = "EXP-20260920-019"
DRAW_SALT = f"{EXPERIMENT_ID}:item_draw:v1"

PLAN = {
    "cifar100": {"evaluation_per_class": 5, "development_per_class": 10, "train_per_class": 20, "class_count": 100},
    "eurosat": {"evaluation_per_class": 15, "development_per_class": 30, "train_per_class": 20, "class_count": 10},
    "cifar10": {"evaluation_per_class": 50, "development_per_class": 100, "train_per_class": 20, "class_count": 10},
}


def rank_key(dataset_id: str, item_id: str) -> str:
    return hashlib.sha256(f"{DRAW_SALT}:{dataset_id}:{item_id}".encode("utf-8")).hexdigest()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pool_from_phase3(phase3: Path, phase4_results: Path, dataset_id: str) -> list[dict[str, str]]:
    rows = load_rows(phase3)
    role_column = "phase3_role" if "phase3_role" in rows[0] else "role"
    consumed = {
        (row["dataset_id"], row["item_id"])
        for row in load_rows(phase4_results / "splits" / "assignments.csv")
    }
    return [
        {"dataset_id": row["dataset_id"], "item_id": row["item_id"],
         "dataset_index": row["dataset_index"], "label": row["label"]}
        for row in rows
        if row["dataset_id"] == dataset_id
        and row[role_column] == "reserved_not_accessed"
        and (row["dataset_id"], row["item_id"]) not in consumed
    ]


def cifar10_labels(data_root: Path) -> list[int]:
    """Training-split labels in torchvision order, with or without torchvision.

    torchvision concatenates `data_batch_1` ... `data_batch_5` in that order, so
    reading the pickled batches directly reproduces its indices exactly.  The
    fallback lets the whole of Day 0 run on a laptop that has numpy but no torch.
    """

    try:
        from torchvision.datasets import CIFAR10

        return [int(label) for label in CIFAR10(root=data_root, train=True, download=False).targets]
    except ModuleNotFoundError:
        pass
    import pickle

    base = Path(data_root) / "cifar-10-batches-py"
    if not base.is_dir():
        raise FileNotFoundError(
            f"{base} not found. Either install torchvision and download CIFAR-10, or fetch the "
            "python archive:\n"
            "  curl -L -o data/cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz\n"
            "  tar -xzf data/cifar-10-python.tar.gz -C data"
        )
    labels: list[int] = []
    for index in range(1, 6):
        with (base / f"data_batch_{index}").open("rb") as handle:
            batch = pickle.load(handle, encoding="bytes")
        labels.extend(int(value) for value in batch[b"labels"])
    if len(labels) != 50000:
        raise RuntimeError(f"CIFAR-10 training split has {len(labels)} labels, expected 50000")
    return labels


def pool_from_cifar10(data_root: Path) -> list[dict[str, str]]:
    return [
        {"dataset_id": "cifar10", "item_id": f"cifar10-train-{index}",
         "dataset_index": str(index), "label": str(int(label))}
        for index, label in enumerate(cifar10_labels(data_root))
    ]


def stratified_draw(pool, per_class: int, exclude: set[str]) -> list[dict[str, str]]:
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pool:
        if row["item_id"] in exclude:
            continue
        by_class[row["label"]].append(row)
    drawn: list[dict[str, str]] = []
    for label in sorted(by_class, key=lambda value: int(value)):
        candidates = sorted(by_class[label], key=lambda row: rank_key(row["dataset_id"], row["item_id"]))
        if len(candidates) < per_class:
            raise RuntimeError(f"class {label} has only {len(candidates)} available items, need {per_class}")
        drawn.extend(candidates[:per_class])
    drawn.sort(key=lambda row: row["item_id"])
    return drawn


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase3-assignments", required=True, type=Path)
    parser.add_argument("--phase4-results", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--datasets", nargs="*", default=["cifar100", "eurosat", "cifar10"])
    parser.add_argument("--skip-cifar10", action="store_true", help="defer CIFAR-10 until torchvision data is present")
    args = parser.parse_args()

    manifest: dict[str, object] = {
        "schema_version": "satml2027.item_manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "draw_salt": DRAW_SALT,
        "draw_rule": "ascending sha256('{salt}:{dataset_id}:{item_id}'), class-stratified, evaluation drawn first then development then control-training, all disjoint",
        "final_test_access": False,
        "datasets": {},
    }
    for dataset_id in args.datasets:
        if dataset_id == "cifar10":
            if args.skip_cifar10:
                continue
            pool = pool_from_cifar10(args.data_root)
            source = "torchvision CIFAR10 train split (official test split never read)"
        else:
            pool = pool_from_phase3(args.phase3_assignments, args.phase4_results, dataset_id)
            source = "phase3 reserved_not_accessed minus phase4-consumed"
        plan = PLAN[dataset_id]
        taken: set[str] = set()
        splits = {}
        for role, per_class in (
            ("evaluation", plan["evaluation_per_class"]),
            ("development", plan["development_per_class"]),
            ("control_train", plan["train_per_class"]),
        ):
            rows = stratified_draw(pool, per_class, taken)
            taken.update(row["item_id"] for row in rows)
            item_ids = [row["item_id"] for row in rows]
            splits[role] = {
                "per_class": per_class,
                "count": len(rows),
                "item_ids_sha256": sha256_id_list(item_ids),
                "csv": f"{dataset_id}__{role}.csv",
            }
            out_csv = args.out / f"{dataset_id}__{role}.csv"
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            with out_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["dataset_id", "item_id", "dataset_index", "label"])
                writer.writeheader()
                writer.writerows(rows)
        manifest["datasets"][dataset_id] = {
            "class_count": plan["class_count"],
            "pool_size": len(pool),
            "pool_source": source,
            "data_role": "extension_reserved_accessed" if dataset_id != "cifar10" else "extension_new_dataset_train_split",
            "splits": splits,
        }
    path = write_json_atomic(args.out / "item_manifest.json", manifest)
    print(json.dumps(manifest["datasets"], indent=2))
    print(f"manifest: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
