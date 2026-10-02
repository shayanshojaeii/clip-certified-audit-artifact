#!/usr/bin/env python3
"""Independent CPU replay of the EXP-021 registered analysis (D-149), for the second team member's check.

For each experiment (021A, 021B, 021C) it
1. verifies the result tree against its run custody manifest (every listed file, no other file);
2. recomputes the registered analysis from the vote counts with the registered code
   (``analyze_ext.analyze`` for 021A/B with the registered item lists; ``analyze_budget`` with the 021B prefix join
   for 021C), with the registered number of bootstrap replicates;
3. compares the recomputation with the released outputs (analysis/exp021/021{a,b,c}/analysis_ext.json and the CSV
   tables): every string, integer, boolean, verdict and label must be equal, every float within --tolerance
   (matrix products may differ in the last bits across CPUs and BLAS builds).
It then regenerates the paper assets into a temporary folder and requires generated/macros.tex to equal the released
one, so every number printed in the paper is shown to come from the replayed outputs.

Before the replay it checks the bindings: the signed stage-3 approval verifies for every registration with these run
custody manifests (guard_ext.verify_approval), every manifest names an archived stage-2 (sampling) approval, and every
released output carries the stage-3 approval's and its manifest's hashes. It signs nothing. CPU only (numpy, scipy).
Exit status 1 on any difference.
Usage (repository root): .venv/Scripts/python scripts/exp021_replay_results.py [--tolerance 1e-9]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

analyze_ext = items_ext = read_json = verify_files_against_custody = load_registration = None  # set by _load_replay


def _load_replay() -> None:
    """Import the registered analysis package for a full replay only: the manuscript package runs --macros-only and
    does not carry satml2027ext (review of manuscript bundle V7, 2026-09-28)."""

    global analyze_ext, items_ext, read_json, verify_files_against_custody, load_registration
    from satml2027ext import analyze_ext, items_ext
    from satml2027ext._common import read_json
    from satml2027ext.custody_ext import verify_files_against_custody
    from satml2027ext.guard_ext import load_registration

RESULTS = ROOT / "results/satml2027ext"
ITEMS = RESULTS / "items"
REGISTRATION = {key: ROOT / f"configs/satml2027ext/exp-20260921-021{key.lower()}.json" for key in "ABC"}
ANALYSIS = {key: ROOT / f"analysis/exp021/021{key.lower()}" for key in "ABC"}
ADDED_BY_MAIN = {"approval_sha256", "run_custody_sha256", "source_sha256"}  # written by main/write_outputs only


def differences(recomputed: Any, released: Any, tolerance: float, path: str = "") -> list[str]:
    """Structural comparison; floats within ``tolerance`` (absolute, or relative for large values)."""

    found: list[str] = []
    if isinstance(released, Mapping) and isinstance(recomputed, Mapping):
        for key in sorted(set(released) | set(recomputed)):
            if key not in recomputed or key not in released:
                found.append(f"{path}/{key}: present on one side only")
                continue
            found.extend(differences(recomputed[key], released[key], tolerance, f"{path}/{key}"))
    elif isinstance(released, list) and isinstance(recomputed, list):
        if len(released) != len(recomputed):
            return [f"{path}: length {len(recomputed)} != {len(released)}"]
        for index, (a, b) in enumerate(zip(recomputed, released)):
            found.extend(differences(a, b, tolerance, f"{path}[{index}]"))
    elif isinstance(released, float) or isinstance(recomputed, float):
        if isinstance(released, bool) or isinstance(recomputed, bool):
            found.append(f"{path}: {recomputed!r} != {released!r}")
        else:
            a, b = float(recomputed), float(released)
            if not (math.isnan(a) and math.isnan(b)) and abs(a - b) > tolerance * max(1.0, abs(b)):
                found.append(f"{path}: {a!r} != {b!r}")
    elif recomputed != released:
        found.append(f"{path}: {recomputed!r} != {released!r}")
    return found


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_written(value: Any) -> str:
    """The text csv.DictWriter writes for a value (analyze_ext._write_csv writes None as an empty field)."""
    return "" if value is None else value if isinstance(value, str) else str(value)


def csv_differences(recomputed: list[dict[str, Any]], released: list[dict[str, str]], tolerance: float,
                    path: str) -> list[str]:
    """Text equality per field, except that two float texts compare within ``tolerance``."""

    if len(recomputed) != len(released):
        return [f"{path}: {len(recomputed)} rows != {len(released)}"]
    found = []
    for index, (mine, theirs) in enumerate(zip(recomputed, released)):
        if set(mine) != set(theirs):
            found.append(f"{path}[{index}]: columns differ")
            continue
        for field in sorted(mine):
            a, b = as_written(mine[field]), theirs[field]
            if a == b:
                continue
            try:
                x, y = float(a), float(b)
                floats = any(mark in a + b for mark in ".eE")
            except ValueError:
                floats = False
            if not (floats and abs(x - y) <= tolerance * max(1.0, abs(y))):
                found.append(f"{path}[{index}]/{field}: {a!r} != {b!r}")
    return found


def recompute(key: str, registrations: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    registration = registrations[key]
    experiment_id = registration["experiment_id"]
    custody = read_json(RESULTS / f"{experiment_id}_RUN_CUSTODY.json")
    verified = verify_files_against_custody(custody, RESULTS / experiment_id)
    not_run = list(custody.get("not_run_cells") or [])
    if key == "C":
        parent = registrations["B"]
        parent_custody = read_json(RESULTS / f"{parent['experiment_id']}_RUN_CUSTODY.json")
        parent_verified = verify_files_against_custody(parent_custody, RESULTS / parent["experiment_id"])
        return analyze_ext.analyze_budget(registration, results=RESULTS / experiment_id, parent_registration=parent,
                                          parent_results=RESULTS / parent["experiment_id"], verified_files=verified,
                                          parent_verified_files=parent_verified)
    rows = {}
    for cell in registration["cells"]:
        if cell["cell_id"] in not_run:
            continue
        listed = items_ext.load_items(ITEMS, cell, "evaluation", registration)
        items_ext.verify_item_list(listed, cell, "evaluation")
        rows[cell["cell_id"]] = listed
    return analyze_ext.analyze(registration, results=RESULTS / experiment_id, not_run=not_run, registered_rows=rows,
                               verified_files=verified)


def compare(key: str, report: Mapping[str, Any], out_dir: Path, tolerance: float) -> list[str]:
    released = read_json(out_dir / "analysis_ext.json")
    body = {name: value for name, value in report.items() if name != "tables"}
    released_body = {name: value for name, value in released.items() if name not in ADDED_BY_MAIN}
    found = differences(json.loads(json.dumps(body)), released_body, tolerance, f"021{key.lower()}")
    for name, rows in (report.get("tables") or {}).items():
        path = out_dir / f"{name}_ext.csv"
        if not path.is_file():
            found.append(f"021{key.lower()}: released table {path.name} is missing")
            continue
        fields = sorted({field for row in rows for field in row})  # the writer's union of columns, blank if absent
        recomputed_rows = [{field: row.get(field) for field in fields} for row in rows]
        found.extend(csv_differences(recomputed_rows, read_rows(path), tolerance, f"021{key.lower()}/{path.name}"))
    return found


def approval_checks(registrations: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """The signed stage-3 approval binds these custody manifests, and every shard ran under the archived stage 2."""

    from satml2027ext._common import sha256_file
    from satml2027ext.guard_ext import verify_approval

    problems: list[str] = []
    approval = ROOT / "research/EXP021_APPROVAL.json"
    stage_files = {"data_preflight": RESULTS / "host_returns/EXP021_DATA_PREFLIGHT.json",
                   "bank_manifest": RESULTS / "host_returns/EXP021_BANK_MANIFEST.json"}
    # Anonymous artifact: the approvals are released with the approver's name redacted, so their bytes differ from the
    # files the custody chain binds. A redaction record lists, per approval, the hash of the original file, the hash of
    # the redacted copy and the redacted fields; a copy whose hash matches the record stands for the original's hash.
    # Without a record (the authors' workspace) every hash is checked as it is.
    redaction_path = ROOT / "research/EXP021_APPROVAL_REDACTION.json"
    redaction = read_json(redaction_path) if redaction_path.is_file() else {}

    def bound_sha256(path: Path) -> str:
        digest = sha256_file(path)
        record = redaction.get(path.relative_to(ROOT).as_posix())
        return record["original_sha256"] if record and record.get("redacted_sha256") == digest else digest

    archived = [path for path in sorted((ROOT / "research/exp021_approvals").glob("*.json"))
                if read_json(path).get("sampling_authorized") is True]
    sampling = {bound_sha256(path) for path in archived}
    body = read_json(approval)
    flags = [body.get(name) for name in ("preparation_authorized", "sampling_authorized", "analysis_authorized")]
    if flags != [False, False, True]:
        problems.append(f"research/EXP021_APPROVAL.json is not a stage-3 approval (flags {flags})")
    for name, path in stage_files.items():
        if body.get(f"{name}_sha256") != sha256_file(path):
            problems.append(f"the stage-3 approval binds another {name}")
    full_guard = (ROOT / ".git").exists()  # the source-commit check needs a git checkout; a ZIP extraction has none
    for key, registration in registrations.items():
        experiment_id = registration["experiment_id"]
        files = dict(stage_files, run_custody_manifest=RESULTS / f"{experiment_id}_RUN_CUSTODY.json")
        if key == "C":
            files["parent_run_custody_manifest"] = RESULTS / f"{registrations['B']['experiment_id']}_RUN_CUSTODY.json"
        if (body.get("registrations") or {}).get(experiment_id) != registration["registration_sha256"]:
            problems.append(f"{experiment_id}: the stage-3 approval binds another registration")
        if (body.get("run_custody_manifest_sha256") or {}).get(experiment_id) != sha256_file(files["run_custody_manifest"]):
            problems.append(f"{experiment_id}: the stage-3 approval binds another run custody manifest")
        if full_guard:
            try:
                verify_approval(registration, stage="analysis", approval_path=approval, stage_files=files)
            except Exception as error:  # noqa: BLE001 - reported
                problems.append(f"{experiment_id}: the stage-3 approval does not verify: {error}")
        custody = read_json(files["run_custody_manifest"])
        if custody.get("sampling_approval_sha256") not in sampling:
            problems.append(f"{experiment_id}: its shards name approval {str(custody.get('sampling_approval_sha256'))[:12]}, "
                            "not an archived stage-2 (sampling) approval")
        released = read_json(ANALYSIS[key] / "analysis_ext.json")
        if released.get("approval_sha256") != bound_sha256(approval):
            problems.append(f"{experiment_id}: the released output was not produced under the current stage-3 approval")
        if released.get("run_custody_sha256") != sha256_file(files["run_custody_manifest"]):
            problems.append(f"{experiment_id}: the released output names another run custody manifest")
    return problems


def curves_match(registrations: Mapping[str, Mapping[str, Any]], tolerance: float) -> list[str]:
    """Recompute the descriptive radius curves from the verified 021B vote counts and compare with the released file."""

    import exp021_radius_curves as curves

    released_path = ANALYSIS["B"] / "descriptive_radius_curves.json"
    if not released_path.is_file():
        return [f"{released_path.relative_to(ROOT).as_posix()} is missing"]
    registration = registrations["B"]
    custody = read_json(RESULTS / f"{registration['experiment_id']}_RUN_CUSTODY.json")
    verified = verify_files_against_custody(custody, RESULTS / registration["experiment_id"])
    rows = {cell["cell_id"]: items_ext.load_items(ITEMS, cell, "evaluation", registration) for cell in registration["cells"]
            if cell["cell_id"] not in set(custody.get("not_run_cells") or [])}
    cells_rows = read_rows(ANALYSIS["B"] / "cells_ext.csv")
    recomputed = curves.compute(registration, RESULTS / registration["experiment_id"], cells_rows=cells_rows,
                                not_run=list(custody.get("not_run_cells") or []), verified=verified, rows_by_cell=rows)
    released = {key: value for key, value in read_json(released_path).items()
                if key not in {"approval_sha256", "run_custody_sha256", "analysis_sha256", "cells_csv_sha256"}}
    return differences(json.loads(json.dumps(recomputed)), released, tolerance, "descriptive_radius_curves")


def macros_match() -> list[str]:
    if not (ROOT / "scripts/generate_satml2027_paper_assets.py").is_file():
        print("paper macros: skipped (the paper generator is in the manuscript package, not in this one)")
        return []
    import generate_satml2027_paper_assets as gen

    released = (ROOT / "paper/satml2027/generated/macros.tex").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as temporary:
        gen.generate(ROOT, Path(temporary))
        rebuilt = (Path(temporary) / "macros.tex").read_text(encoding="utf-8")
    return [] if rebuilt == released else ["paper/satml2027/generated/macros.tex differs from the regenerated macros"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--only", choices=["A", "B", "C"], action="append", default=None)
    parser.add_argument("--macros-only", action="store_true",
                        help="only regenerate the paper assets and compare macros.tex (the manuscript package)")
    args = parser.parse_args(argv)
    if args.macros_only:
        found = macros_match()
        print(f"paper macros regenerate identically: {'PASS' if not found else 'FAIL'}")
        print("REPLAY " + ("PASS" if not found else "FAIL"))
        return 1 if found else 0
    _load_replay()
    registrations = {key: load_registration(path) for key, path in REGISTRATION.items()}
    problems = approval_checks(registrations)
    scope = "with the full guard" if (ROOT / ".git").exists() else "field by field (no git checkout: source commit not re-checked)"
    print(f"stage-3 approval and custody bindings, {scope}: {'PASS' if not problems else 'FAIL'}")
    for line in problems:
        print("   ", line)
    for key in args.only or ["A", "B", "C"]:
        report = recompute(key, registrations)
        found = compare(key, report, ANALYSIS[key], args.tolerance)
        print(f"021{key}: {'PASS' if not found else f'FAIL ({len(found)} differences)'}")
        for line in found[:20]:
            print("   ", line)
        problems.extend(found)
    if not args.only or "B" in args.only:
        found = curves_match(registrations, args.tolerance)
        print(f"descriptive radius curves recompute: {'PASS' if not found else f'FAIL ({len(found)} differences)'}")
        for line in found[:10]:
            print("   ", line)
        problems.extend(found)
    if not args.only:
        found = macros_match()
        print(f"paper macros regenerate identically: {'PASS' if not found else 'FAIL'}")
        problems.extend(found)
    print("REPLAY " + ("PASS" if not problems else "FAIL"))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
