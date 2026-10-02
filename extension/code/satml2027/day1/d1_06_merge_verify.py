#!/usr/bin/env python3
"""Day 1 - merge the shards of every cell and prove the merge is complete.

Shards may come from several servers (`--roots A_results B_results ...`).  For
each cell the script checks that the shards tile the registered item list exactly
once, that every item is finished, that the item-id hash matches the one in the
registration, and that every row of votes sums to the registered draw budget.
It then writes one merged `.npz` per cell plus a completeness report.

Usage:
  python day1/d1_06_merge_verify.py --registration configs/satml2027/exp-20260920-019a.json \
      --roots results/satml2027/EXP-20260920-019A serverB_pull/EXP-20260920-019A \
      --out results/satml2027/merged/EXP-20260920-019A
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.hashing import read_json, sha256_id_list, write_json_atomic  # noqa: E402
from common.banks import validate_registered_bank_payload  # noqa: E402


def merge_cell(cell: dict, roots: list[Path], registration: dict, banks_dir: Path) -> tuple[dict, dict]:
    import torch

    bank_path = banks_dir / f"{cell['cell_id']}__banks.pt"
    if not bank_path.is_file():
        raise FileNotFoundError(f"registered bank artifact is absent: {bank_path}")
    bank_payload = torch.load(bank_path, map_location="cpu", weights_only=False)
    expected_banks = validate_registered_bank_payload(
        bank_payload,
        registration_sha256=registration["registration_sha256"],
        cell_id=cell["cell_id"],
    )
    expected_metadata = {candidate_id: entry["metadata"] for candidate_id, entry in sorted(expected_banks.items())}
    expected_candidate_ids = sorted(expected_banks)
    shards = []
    for root in roots:
        shards.extend(sorted((root / "cells" / cell["cell_id"]).glob("shard_*_of_*.npz")))
    shards = [path for path in shards if not path.name.endswith("_features.npz")]
    if not shards:
        raise FileNotFoundError(f"no shards for {cell['cell_id']}")
    loaded = [np.load(path, allow_pickle=False) for path in shards]
    shard_meta = []
    for path in shards:
        meta_path = path.with_suffix(".meta.json")
        if not meta_path.is_file():
            raise RuntimeError(f"{cell['cell_id']}: missing shard metadata sidecar for {path.name}")
        meta = read_json(meta_path)
        if meta.get("experiment_id") != registration["experiment_id"]:
            raise RuntimeError(f"{cell['cell_id']}: shard experiment ID mismatch")
        if meta.get("registration_sha256") != registration["registration_sha256"]:
            raise RuntimeError(f"{cell['cell_id']}: shard registration hash mismatch")
        if meta.get("cell", {}).get("cell_id") != cell["cell_id"]:
            raise RuntimeError(f"{cell['cell_id']}: shard cell metadata mismatch")
        shard_meta.append(meta)
    counts = {int(path.name.split("_of_")[1].split(".")[0]) for path in shards}
    if len(counts) != 1:
        raise RuntimeError(f"{cell['cell_id']}: inconsistent shard_count {sorted(counts)}")
    shard_count = counts.pop()
    if len(shards) != shard_count:
        raise RuntimeError(f"{cell['cell_id']}: {len(shards)} shard files but shard_count={shard_count}")

    candidate_ids = list(loaded[0]["candidate_ids"])
    if candidate_ids != expected_candidate_ids:
        raise RuntimeError(f"{cell['cell_id']}: shard bank list differs from the frozen bank artifact")
    candidate_metadata = shard_meta[0].get("candidate_metadata")
    if set(candidate_metadata or {}) != set(candidate_ids):
        raise RuntimeError(f"{cell['cell_id']}: shard metadata bank list differs from count archive")
    if candidate_metadata != expected_metadata:
        raise RuntimeError(f"{cell['cell_id']}: shard candidate metadata differs from the frozen bank artifact")
    for data in loaded[1:]:
        if list(data["candidate_ids"]) != candidate_ids:
            raise RuntimeError(f"{cell['cell_id']}: shards disagree on the bank list")
    for meta in shard_meta[1:]:
        if meta.get("candidate_metadata") != candidate_metadata:
            raise RuntimeError(f"{cell['cell_id']}: shards disagree on candidate-bank metadata")
    order = np.argsort([int(data["item_offset"][0]) for data in loaded])
    loaded = [loaded[index] for index in order]
    expected_offset = 0
    for data in loaded:
        if int(data["item_offset"][0]) != expected_offset:
            raise RuntimeError(f"{cell['cell_id']}: shards do not tile the item list")
        if not bool(np.all(data["done"])):
            raise RuntimeError(f"{cell['cell_id']}: shard at offset {expected_offset} is incomplete "
                               f"({int(data['done'].sum())}/{len(data['done'])})")
        expected_offset += len(data["item_ids"])

    merged = {
        "candidate_ids": np.array(candidate_ids),
        "item_ids": np.concatenate([data["item_ids"] for data in loaded]),
        "ground_truth": np.concatenate([data["ground_truth"] for data in loaded]),
        "raw_predictions": np.concatenate([data["raw_predictions"] for data in loaded], axis=1),
        "selection_counts": np.concatenate([data["selection_counts"] for data in loaded], axis=1),
        "confirmation_counts": np.concatenate([data["confirmation_counts"] for data in loaded], axis=1),
        "selection_seeds": np.concatenate([data["selection_seeds"] for data in loaded]),
        "confirmation_seeds": np.concatenate([data["confirmation_seeds"] for data in loaded]),
    }
    item_count = len(merged["item_ids"])
    if item_count != int(cell["item_count"]):
        raise RuntimeError(f"{cell['cell_id']}: merged {item_count} items, registration says {cell['item_count']}")
    if len(set(merged["item_ids"].tolist())) != item_count:
        raise RuntimeError(f"{cell['cell_id']}: duplicate items after merge")

    certification = registration.get("certification")
    checks: dict[str, object] = {"shard_count": shard_count, "item_count": item_count, "bank_count": len(candidate_ids)}
    ids_hash = sha256_id_list(merged["item_ids"].tolist())
    expected_hash = cell.get("evaluation_items_sha256") or cell.get("development_items_sha256")
    checks["item_ids_sha256_ordered"] = ids_hash
    checks["item_ids_match_registration"] = ids_hash == expected_hash
    if expected_hash is None or ids_hash != expected_hash:
        raise RuntimeError(
            f"{cell['cell_id']}: ordered item-list hash mismatch: expected {expected_hash}, got {ids_hash}"
        )
    if certification:
        selection_total = int(certification["selection_draws"])
        confirmation_total = int(certification["confirmation_draws"])
        if not np.all(merged["selection_counts"].sum(axis=2) == selection_total):
            raise RuntimeError(f"{cell['cell_id']}: selection votes do not sum to {selection_total}")
        if not np.all(merged["confirmation_counts"].sum(axis=2) == confirmation_total):
            raise RuntimeError(f"{cell['cell_id']}: confirmation votes do not sum to {confirmation_total}")
        checks["selection_draws"] = selection_total
        checks["confirmation_draws"] = confirmation_total
    if len(set(merged["selection_seeds"].tolist())) != item_count:
        raise RuntimeError(f"{cell['cell_id']}: repeated selection seeds")
    return merged, checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--roots", nargs="+", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--banks", type=Path, default=Path("results/satml2027/banks"))
    parser.add_argument("--allow-missing", action="store_true", help="report missing cells instead of failing")
    args = parser.parse_args()

    registration = read_json(args.registration)
    report = {
        "schema_version": "satml2027.merge_report.v1",
        "experiment_id": registration["experiment_id"],
        "registration_sha256": registration["registration_sha256"],
        "cells": {},
        "missing": [],
        "final_test_access": False,
    }
    for cell in registration["cells"]:
        try:
            merged, checks = merge_cell(cell, args.roots, registration, args.banks)
        except FileNotFoundError as error:
            if not args.allow_missing:
                raise
            report["missing"].append({"cell_id": cell["cell_id"], "reason": str(error)})
            continue
        path = args.out / f"{cell['cell_id']}__merged.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **merged)
        report["cells"][cell["cell_id"]] = checks
        print(f"{cell['cell_id']}: {checks['item_count']} items x {checks['bank_count']} banks "
              f"from {checks['shard_count']} shards -> {path.name}")
    report["complete"] = not report["missing"]
    write_json_atomic(args.out / "merge_report.json", report)
    print(f"\n{len(report['cells'])} cells merged, {len(report['missing'])} missing")
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
