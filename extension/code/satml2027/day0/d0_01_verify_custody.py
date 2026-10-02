#!/usr/bin/env python3
"""Day 0, step 1 - close Phase-4 custody and prove the chain of hashes.

Checks, all of which must pass before any Phase-4 number is cited in the paper:

  1. the Phase-3 assignments file hashes to the value recorded in the Phase-4
     split manifest;
  2. the shipped Phase-4 config reproduces the canonical-JSON hash stored in
     every Phase-4 output file;
  3. every item Phase-4 consumed was `reserved_not_accessed` in Phase-3;
  4. no artifact anywhere claims final-test access.

It then writes the ledger entry that marks those items as consumed and copies
the config into the results directory (the Phase-4 lesson: ship the config, not
only its hash).

Usage:
  python day0/d0_01_verify_custody.py \
      --phase3-assignments artifacts/day14/phase3_splits_v1/assignments.csv \
      --phase4-results results/phase4/EXP-20260917-020-FARLA-FULL \
      --phase4-config configs/phase4/farla_full_v1.json \
      --ledger artifacts/ledger/item_roles.jsonl \
      --out results/satml2027/custody
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.hashing import canonical_json_hash, read_json, sha256_file, write_json_atomic  # noqa: E402


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"{path} is empty")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase3-assignments", required=True, type=Path)
    parser.add_argument("--phase4-results", required=True, type=Path)
    parser.add_argument("--phase4-config", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--copy-config", action="store_true", help="copy the config into the results directory")
    args = parser.parse_args()

    failures: list[str] = []
    report: dict[str, object] = {
        "schema_version": "satml2027.custody_report.v1",
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    # 1 - Phase-3 assignments hash against the Phase-4 manifest.
    p3_hash = sha256_file(args.phase3_assignments)
    manifest = read_json(args.phase4_results / "splits" / "manifest.json")
    expected = manifest.get("source_phase3_assignments_sha256")
    report["phase3_assignments_sha256"] = p3_hash
    report["phase4_manifest_expects"] = expected
    if p3_hash != expected:
        failures.append(f"phase-3 assignments hash {p3_hash} != manifest {expected}")

    # 2 - config canonical hash against every recorded value.
    config = json.loads(args.phase4_config.read_text(encoding="utf-8"))
    config_hash = canonical_json_hash(config)
    recorded = {
        "CONFIG_SHA256": (args.phase4_results / "CONFIG_SHA256").read_text(encoding="utf-8").strip(),
        "preflight.json": read_json(args.phase4_results / "preflight.json")["config_sha256"],
        "splits/manifest.json": manifest["config_sha256"],
        "screening_report.json": read_json(args.phase4_results / "screening_report.json")["config_sha256"],
    }
    for metrics_path in sorted((args.phase4_results / "confirmation").glob("*/metrics.json")):
        recorded[metrics_path.relative_to(args.phase4_results).as_posix()] = read_json(metrics_path)["config_sha256"]
    report["phase4_config_canonical_sha256"] = config_hash
    report["phase4_recorded_config_hashes"] = recorded
    mismatched = {key: value for key, value in recorded.items() if value != config_hash}
    if mismatched:
        failures.append(f"config hash mismatch in {sorted(mismatched)}")
    report["phase4_config_raw_file_sha256"] = sha256_file(args.phase4_config)

    # 3 - consumed items were reserved.
    p3_rows = load_rows(args.phase3_assignments)
    role_column = "phase3_role" if "phase3_role" in p3_rows[0] else "role"
    p3_role = {(row["dataset_id"], row["item_id"]): row[role_column] for row in p3_rows}
    p4_rows = load_rows(args.phase4_results / "splits" / "assignments.csv")
    consumed = [(row["dataset_id"], row["item_id"]) for row in p4_rows]
    if len(set(consumed)) != len(consumed):
        failures.append("phase-4 assignments contain duplicate items")
    unknown = [key for key in consumed if key not in p3_role]
    wrong_role = [key for key in consumed if p3_role.get(key) not in (None, "reserved_not_accessed")]
    if unknown:
        failures.append(f"{len(unknown)} phase-4 items are absent from the phase-3 table")
    if wrong_role:
        failures.append(f"{len(wrong_role)} phase-4 items were not reserved_not_accessed")
    report["phase4_consumed_per_dataset"] = dict(Counter(dataset for dataset, _ in consumed))
    report["phase4_consumed_per_role"] = dict(
        Counter(f"{row['dataset_id']}/{row.get('phase4_role', '?')}" for row in p4_rows)
    )

    # 4 - no final-test access anywhere.
    flags: dict[str, object] = {}
    for path in sorted(args.phase4_results.rglob("*.json")):
        try:
            payload = read_json(path)
        except Exception:
            continue
        if isinstance(payload, dict):
            for key in ("phase3_final_test_accessed", "final_test_accessed", "phase3_final_test_locked"):
                if key in payload:
                    flags[f"{path.relative_to(args.phase4_results).as_posix()}::{key}"] = payload[key]
    report["final_test_flags"] = flags
    bad = {key: value for key, value in flags.items() if key.endswith("accessed") and value is not False}
    if bad:
        failures.append(f"final-test access flag is not false: {sorted(bad)}")

    # remaining pool
    remaining = Counter(
        row["dataset_id"]
        for row in p3_rows
        if row[role_column] == "reserved_not_accessed" and (row["dataset_id"], row["item_id"]) not in set(consumed)
    )
    total_reserved = Counter(row["dataset_id"] for row in p3_rows if row[role_column] == "reserved_not_accessed")
    report["reserved_before_phase4"] = dict(total_reserved)
    report["reserved_remaining"] = dict(remaining)
    report["phase3_role_counts"] = dict(Counter(f"{row['dataset_id']}/{row[role_column]}" for row in p3_rows))

    # ledger append (idempotent: one line per experiment, keyed by experiment_id)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "experiment_id": "EXP-20260917-020-FARLA-FULL",
        "recorded_at": report["checked_at"],
        "action": "consumed_reserved_items",
        "per_dataset_counts": report["phase4_consumed_per_dataset"],
        "item_list_sha256": canonical_json_hash([f"{dataset}/{item}" for dataset, item in sorted(consumed)]),
        "new_role": "phase4_accessed",
        "final_test_accessed": False,
    }
    existing = []
    if args.ledger.is_file():
        existing = [json.loads(line) for line in args.ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not any(record.get("experiment_id") == entry["experiment_id"] for record in existing):
        with args.ledger.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        report["ledger_entry_written"] = True
    else:
        report["ledger_entry_written"] = False

    if args.copy_config:
        destination = args.phase4_results / "config_snapshot.json"
        shutil.copyfile(args.phase4_config, destination)
        report["config_snapshot"] = destination.as_posix()

    report["failures"] = failures
    report["status"] = "PASS" if not failures else "FAIL"
    path = write_json_atomic(args.out / "phase4_custody_report.json", report)
    print(json.dumps({key: report[key] for key in ("status", "reserved_remaining", "phase4_consumed_per_dataset", "failures")}, indent=2))
    print(f"report: {path}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
