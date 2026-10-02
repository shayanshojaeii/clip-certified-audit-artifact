#!/usr/bin/env python3
"""Day 0, step 0 - find every input the later steps need, and print their commands.

Run this first, from the project root.  It searches for the five inputs the Day-0
and Day-1 scripts require, verifies each one is really what it claims to be,
writes `results/satml2027/paths.json`, and prints ready-to-paste commands with
the resolved paths filled in - so no step below ever needs a guessed path.

  python satml2027/day0/d0_00_locate.py                 # from the project root
  python satml2027/day0/d0_00_locate.py --root "C:/path/to/project" --shell powershell
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.hashing import write_json_atomic  # noqa: E402

PHASE4_EXPERIMENT = "EXP-20260917-020-FARLA-FULL"


SKIP_PARTS = {"tmp", "temp", ".git", "__pycache__", ".venv", "node_modules"}


def _candidates(root: Path, patterns: list[str]) -> list[Path]:
    """Glob `patterns` under `root`, skipping scratch and pre-sampling directories.

    The skip test is applied to the path RELATIVE to the root, never to its
    absolute components: a project root that itself lives under /tmp or under a
    folder called `temp` must not cause every candidate to be discarded.
    """

    found: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            if SKIP_PARTS & set(relative.parts):
                continue
            if "PHASE3_PRESAMPLING" in relative.as_posix():
                continue
            found.append(path)
    return found


def find_phase3_assignments(root: Path) -> tuple[Path | None, str]:
    for path in _candidates(root, ["artifacts/day14/phase3_splits_v1/assignments.csv",
                                   "*/artifacts/day14/phase3_splits_v1/assignments.csv",
                                   "**/phase3_splits_v1/assignments.csv"]):
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except OSError:
            continue
        role = "phase3_role" if rows and "phase3_role" in rows[0] else "role"
        if rows and role in rows[0] and "item_id" in rows[0] and len(rows) > 50000:
            reserved = sum(1 for row in rows if row[role] == "reserved_not_accessed")
            return path, f"{len(rows)} rows, {reserved} reserved_not_accessed"
    return None, "not found"


def find_phase4_results(root: Path) -> tuple[Path | None, str]:
    for path in _candidates(root, [f"**/{PHASE4_EXPERIMENT}"]):
        if (path / "CONFIG_SHA256").is_file() and (path / "splits" / "manifest.json").is_file():
            digest = (path / "CONFIG_SHA256").read_text(encoding="utf-8").strip()
            cells = len(list((path / "confirmation").glob("*/metrics.json"))) if (path / "confirmation").is_dir() else 0
            return path, f"config {digest[:12]}..., {cells} confirmation cells"
    return None, "not found"


def find_phase4_config(root: Path, results: Path | None) -> tuple[Path | None, str]:
    expected = None
    if results is not None:
        expected = (results / "CONFIG_SHA256").read_text(encoding="utf-8").strip()
    for path in _candidates(root, ["**/configs/phase4/farla_full_v1.json", "**/farla_full_v1.json"]):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("experiment_id") == PHASE4_EXPERIMENT:
            from common.hashing import canonical_json_hash

            digest = canonical_json_hash(data)
            mark = "canonical hash MATCHES the results" if digest == expected else f"hash {digest[:12]}... (results say {str(expected)[:12]}...)"
            return path, mark
    return None, "not found"


def find_results_dir(root: Path, experiment_id: str) -> tuple[Path | None, str]:
    for path in _candidates(root, [f"results/{experiment_id}", f"**/results/{experiment_id}"]):
        cells = path / "cells"
        if cells.is_dir():
            return path, f"{len(list(cells.iterdir()))} cells"
    return None, "not found"


def find_data_root(root: Path) -> tuple[Path | None, str]:
    for path in _candidates(root, ["data", "*/data"]):
        present = [name for name, probe in (
            ("cifar100", "cifar-100-python"), ("eurosat", "eurosat/2750"),
            ("cifar10", "cifar-10-batches-py"),
        ) if (path / probe).exists()]
        if present:
            missing = {"cifar100", "eurosat", "cifar10"} - set(present)
            return path, f"has {', '.join(sorted(present))}" + (f"; MISSING {', '.join(sorted(missing))}" if missing else "")
    return None, "not found"


def find_protocol_config(root: Path) -> tuple[Path | None, str]:
    for path in _candidates(root, ["configs/phase3_calibration_protocol_v1.json", "**/phase3_calibration_protocol_v1.json"]):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        hits = [line.strip() for line in text.splitlines() if "temperature" in line.lower()]
        return path, (f"temperature fields: {' | '.join(hits[:3])}" if hits else "no 'temperature' field found - search other configs")
    return None, "not found"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--shell", choices=["powershell", "bash"], default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    shell = args.shell or ("powershell" if sys.platform.startswith("win") else "bash")
    out = args.out or (root / "results" / "satml2027")

    print(f"project root: {root}\n")
    phase3, phase3_note = find_phase3_assignments(root)
    phase4, phase4_note = find_phase4_results(root)
    config, config_note = find_phase4_config(root, phase4)
    exp016, exp016_note = find_results_dir(root, "EXP-20260906-016")
    exp017, exp017_note = find_results_dir(root, "EXP-20260906-017")
    data, data_note = find_data_root(root)
    protocol, protocol_note = find_protocol_config(root)

    entries = [
        ("phase3_assignments", phase3, phase3_note, True),
        ("phase4_results", phase4, phase4_note, True),
        ("phase4_config", config, config_note, True),
        ("exp016_results", exp016, exp016_note, True),
        ("exp017_results", exp017, exp017_note, True),
        ("data_root", data, data_note, True),
        ("phase3_protocol_config", protocol, protocol_note, False),
    ]
    paths: dict[str, str | None] = {}
    missing = []
    for key, path, note, required in entries:
        relative = None if path is None else path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
        paths[key] = relative
        status = "OK " if path is not None else ("MISSING" if required else "absent ")
        print(f"  [{status}] {key:24s} {relative or '-'}")
        print(f"            {note}")
        if path is None and required:
            missing.append(key)

    paths["project_root"] = str(root)
    write_json_atomic(out / "paths.json", {"schema_version": "satml2027.paths.v1", **paths})
    print(f"\nwrote {out / 'paths.json'}")

    if data is not None and "MISSING cifar10" in data_note:
        print("\nCIFAR-10 is absent from the data root. Fetch the python archive first:")
        if shell == "powershell":
            print(f'  Invoke-WebRequest https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -OutFile "{data}\\cifar-10-python.tar.gz"')
            print(f'  tar -xzf "{data}\\cifar-10-python.tar.gz" -C "{data}"')
        else:
            print(f'  curl -L -o "{data}/cifar-10-python.tar.gz" https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
            print(f'  tar -xzf "{data}/cifar-10-python.tar.gz" -C "{data}"')
        print(f'  # expected afterwards: {data}/cifar-10-batches-py/data_batch_1 .. _5')

    if missing:
        print("\nMISSING REQUIRED INPUTS: " + ", ".join(missing))
        return 1

    def q(value) -> str:
        text = str(value)
        return f'"{text}"' if " " in text else text

    print("\n" + "=" * 78)
    print(f"Ready-to-paste Day-0 commands ({shell}); run them from {root}")
    print("=" * 78)
    steps = [
        ("A4  custody", [
            "python", q("satml2027/day0/d0_01_verify_custody.py"),
            "--phase3-assignments", q(paths["phase3_assignments"]),
            "--phase4-results", q(paths["phase4_results"]),
            "--phase4-config", q(paths["phase4_config"]),
            "--ledger", q("artifacts/ledger/item_roles.jsonl"),
            "--out", q("results/satml2027/custody"), "--copy-config",
        ]),
        ("A5  item draw", [
            "python", q("satml2027/day0/d0_02_draw_items.py"),
            "--phase3-assignments", q(paths["phase3_assignments"]),
            "--phase4-results", q(paths["phase4_results"]),
            "--data-root", q(paths["data_root"]),
            "--out", q("results/satml2027/items"),
        ]),
        ("A6  registrations", [
            "python", q("satml2027/day0/d0_03_make_registrations.py"),
            "--items", q("results/satml2027/items"),
            "--out", q("configs/satml2027"), "--scope", "full",
        ]),
        ("A8  artifact skeleton", [
            "python", q("satml2027/day0/d0_04_init_artifact_repo.py"),
            "--out", q("artifact"), "--bundle", q("satml2027"),
        ]),
        ("B4  bank custody proof (run on a server after copying EXP-016/017)", [
            "python", q("satml2027/day1/d1_03_build_banks.py"), "--verify-against-exp017",
            "--exp016", q(paths["exp016_results"]), "--exp017", q(paths["exp017_results"]),
        ]),
    ]
    for label, parts in steps:
        print(f"\n# {label}")
        print(" ".join(parts))
    print("\nAll paths above are relative to the project root, so they work on both hosts")
    print("once the same three directories are copied across (satml2027, configs/satml2027,")
    print("results/satml2027/items).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
