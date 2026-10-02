"""V4.1 custody-closure tests (reviewer-bundle V5 audit, S0 "unlisted sidecar").

The audit added ``shard_-001_of_1.meta.json`` to a cell directory after custody:
the V4 custody verification still passed, and the analysis read that file first
and changed the Q4 margin bound from 0.00001 to 8.00001.  Here every unlisted
entry must be refused by the custody verification and by the analysis, and the
analysis must read exactly the custody-verified files.

  .venv/Scripts/python -m pytest satml2027ext/tests/test_custody_closure_ext.py -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from satml2027ext import analyze_ext, contract_ext, custody_ext  # noqa: E402
from satml2027ext._common import canonical_json_hash, sha256_file, write_json_atomic  # noqa: E402

CANDIDATE = "chowers_exact_projected_gap__coefficient_1"
CELL = {"cell_id": "model__dataset__sigma0.25"}


def sidecar(path: Path, row_norms: list[float]) -> None:
    write_json_atomic(path, {"candidate_metadata": {CANDIDATE: {"metadata": {"operator": {
        "row_norms": row_norms, "row_scales": [1.0 for _ in row_norms]}}}}})


def make_results(root: Path, shards: int = 1) -> Path:
    cell_dir = root / "cells" / CELL["cell_id"]
    cell_dir.mkdir(parents=True)
    for index in range(shards):
        stem = contract_ext.shard_stem(index, shards)
        (cell_dir / f"{stem}.npz").write_bytes(b"archive %d" % index)
        sidecar(cell_dir / f"{stem}.meta.json", [1.0, 1.0])
    return cell_dir


def custody_for(root: Path, shards: int = 1, not_run: tuple[str, ...] = ()) -> dict:
    cell_dir = root / "cells" / CELL["cell_id"]
    files = {path.name: sha256_file(path) for path in sorted(cell_dir.iterdir())}
    custody = {
        "schema_version": custody_ext.CUSTODY_SCHEMA, "experiment_id": "EXP-20260921-021B",
        "registration_sha256": "0" * 64, "sampling_approval_sha256": "1" * 64,
        "bank_manifest_sha256": "2" * 64, "data_preflight_sha256": "3" * 64, "not_run_cells": list(not_run),
        "cells": {CELL["cell_id"]: {"shard_count": shards, "items": 1, "device_name": "probe", "files": files}},
        "file_count": len(files),
    }
    custody["content_sha256"] = canonical_json_hash(custody)
    return custody


def test_the_audit_probe_is_refused_by_custody_and_by_the_analysis(tmp_path):
    cell_dir = make_results(tmp_path)
    custody = custody_for(tmp_path)
    verified = custody_ext.verify_files_against_custody(custody, tmp_path)
    before = analyze_ext.projected_gap_tie_budget(tmp_path, CELL, [CANDIDATE], 1e-5, shard_set=verified[CELL["cell_id"]])
    assert before[CANDIDATE]["margin_bound"] == pytest.approx(1e-5)
    sidecar(cell_dir / "shard_-001_of_1.meta.json", [1.0, 9.0])  # unlisted and lexicographically first
    with pytest.raises(RuntimeError):
        custody_ext.verify_files_against_custody(custody, tmp_path)
    with pytest.raises(RuntimeError):
        analyze_ext.projected_gap_tie_budget(tmp_path, CELL, [CANDIDATE], 1e-5)
    with pytest.raises(RuntimeError):
        analyze_ext.cell_shard_set(tmp_path, CELL, verified)


@pytest.mark.parametrize("name", [
    "shard_001_of_001.meta.json",    # well-formed name, index outside the set
    "shard_000_of_002.npz",          # another shard count
    "shard_0000_of_001.npz",         # non-canonical spelling of shard 0
    "shard_000_of_001_margins.npz",  # a margin store that custody never listed
    "shard_000_of_001.npz.tmp",      # a leftover temporary file
    "notes.txt",
])
def test_any_unlisted_file_is_refused_by_custody_and_by_the_verified_analysis(tmp_path, name):
    cell_dir = make_results(tmp_path)
    custody = custody_for(tmp_path)
    verified = custody_ext.verify_files_against_custody(custody, tmp_path)
    (cell_dir / name).write_bytes(b"x")
    with pytest.raises(RuntimeError):
        custody_ext.verify_files_against_custody(custody, tmp_path)
    with pytest.raises(RuntimeError):
        analyze_ext.cell_shard_set(tmp_path, CELL, verified)


def test_a_changed_listed_file_is_refused(tmp_path):
    cell_dir = make_results(tmp_path)
    custody = custody_for(tmp_path)
    sidecar(cell_dir / "shard_000_of_001.meta.json", [1.0, 9.0])
    with pytest.raises(RuntimeError, match="differs from the run custody manifest"):
        custody_ext.verify_files_against_custody(custody, tmp_path)


def test_a_subdirectory_or_an_unlisted_cell_directory_is_refused(tmp_path):
    cell_dir = make_results(tmp_path)
    custody = custody_for(tmp_path)
    (cell_dir / "extra").mkdir()
    with pytest.raises(RuntimeError):
        custody_ext.verify_files_against_custody(custody, tmp_path)
    (cell_dir / "extra").rmdir()
    custody_ext.verify_files_against_custody(custody, tmp_path)
    (tmp_path / "cells" / "unlisted_cell").mkdir()
    with pytest.raises(RuntimeError, match="absent from the run custody manifest"):
        custody_ext.verify_files_against_custody(custody, tmp_path)


def test_a_declared_not_run_cell_may_only_be_an_empty_directory(tmp_path):
    make_results(tmp_path)
    custody = custody_for(tmp_path, not_run=("skipped",))
    (tmp_path / "cells" / "skipped").mkdir()
    custody_ext.verify_files_against_custody(custody, tmp_path)
    (tmp_path / "cells" / "skipped" / "shard_000_of_001.npz").write_bytes(b"x")
    with pytest.raises(RuntimeError):
        custody_ext.verify_files_against_custody(custody, tmp_path)


def test_the_verified_listing_is_returned_in_index_order(tmp_path):
    make_results(tmp_path, shards=3)
    verified = custody_ext.verify_files_against_custody(custody_for(tmp_path, shards=3), tmp_path)
    listing = verified[CELL["cell_id"]]
    assert [path.name for path in listing["shards"]] == [f"shard_{index:03d}_of_003.npz" for index in range(3)]
    assert [path.name for path in listing["sidecars"]] == [f"shard_{index:03d}_of_003.meta.json" for index in range(3)]
    assert analyze_ext.cell_shard_set(tmp_path, CELL, verified)["names"] == listing["names"]


def test_sidecars_that_disagree_on_an_operator_record_are_refused(tmp_path):
    cell_dir = make_results(tmp_path, shards=2)
    sidecar(cell_dir / "shard_001_of_002.meta.json", [1.0, 9.0])
    with pytest.raises(RuntimeError, match="disagree on the operator record"):
        analyze_ext.projected_gap_tie_budget(tmp_path, CELL, [CANDIDATE], 1e-5)


@pytest.mark.parametrize("case", ["missing_sidecar", "partial_margins", "margins_not_registered", "margins_missing",
                                  "mixed_counts", "missing_archive", "empty"])
def test_shard_file_set_refuses_incomplete_or_inconsistent_sets(tmp_path, case):
    cell_dir = make_results(tmp_path, shards=2)
    margins = None
    if case == "missing_sidecar":
        (cell_dir / "shard_001_of_002.meta.json").unlink()
    elif case == "partial_margins":
        (cell_dir / "shard_000_of_002_margins.npz").write_bytes(b"m")
    elif case == "margins_not_registered":
        for index in range(2):
            (cell_dir / f"shard_{index:03d}_of_002_margins.npz").write_bytes(b"m")
        margins = False
    elif case == "margins_missing":
        margins = True
    elif case == "mixed_counts":
        (cell_dir / "shard_000_of_003.npz").write_bytes(b"x")
    elif case == "missing_archive":
        (cell_dir / "shard_001_of_002.npz").unlink()
    elif case == "empty":
        for path in list(cell_dir.iterdir()):
            path.unlink()
    with pytest.raises(contract_ext.CountContractError):
        contract_ext.shard_file_set(cell_dir, "probe", sidecars=True, margins=margins)


def test_a_file_rewritten_after_verification_is_refused_on_read(tmp_path):
    cell_dir = make_results(tmp_path)
    verified = custody_ext.verify_files_against_custody(custody_for(tmp_path), tmp_path)
    listing = analyze_ext.cell_shard_set(tmp_path, CELL, verified)
    assert analyze_ext.projected_gap_tie_budget(tmp_path, CELL, [CANDIDATE], 1e-5, shard_set=listing)
    sidecar(cell_dir / "shard_000_of_001.meta.json", [1.0, 9.0])  # same name, new bytes, after the verification
    with pytest.raises(contract_ext.CountContractError, match="custody-verified bytes"):
        analyze_ext.projected_gap_tie_budget(tmp_path, CELL, [CANDIDATE], 1e-5, shard_set=listing)
    with pytest.raises(contract_ext.CountContractError):
        contract_ext.read_verified_bytes(cell_dir / "shard_000_of_001.npz", {})


def test_a_hard_linked_result_file_is_refused(tmp_path):
    cell_dir = make_results(tmp_path)
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"archive 0")
    (cell_dir / "shard_000_of_001.npz").unlink()
    os.link(outside, cell_dir / "shard_000_of_001.npz")
    with pytest.raises(contract_ext.CountContractError, match="hard links"):
        contract_ext.shard_file_set(cell_dir, "probe", sidecars=True, margins=None)


def test_a_not_run_cell_with_an_empty_subdirectory_is_refused_at_build_and_at_verification(tmp_path):
    from satml2027ext import custody_ext as custody

    make_results(tmp_path)
    registration = {"cells": [{"cell_id": CELL["cell_id"]}, {"cell_id": "skipped"}]}
    (tmp_path / "cells" / "skipped" / "empty").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="declared not run"):
        custody.not_run_check(registration, tmp_path, ["skipped"])
    with pytest.raises(RuntimeError):
        custody.verify_files_against_custody(custody_for(tmp_path, not_run=("skipped",)), tmp_path)
