"""Integration checks on the real registrations: the V3 features (internal re-review of V2), now on V4 (D-136).

  .venv/Scripts/python -m pytest satml2027ext/tests/test_integration_v3.py -q
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from satml2027ext import analyze_ext, bank_manifest_ext, candidates_ext, guard_ext  # noqa: E402
from satml2027ext import make_registrations_ext as mr  # noqa: E402
from satml2027ext import preflight_data_ext, run_shard_ext  # noqa: E402
from satml2027ext._common import canonical_json_hash, read_json, sha256_file, write_json_atomic  # noqa: E402

CONFIGS = PROJECT / "configs" / "satml2027ext"
PATHS = {key: CONFIGS / f"exp-20260921-{key}.json" for key in ("021a", "021b", "021c")}


@pytest.fixture(scope="module")
def registrations():
    return {key: guard_ext.load_registration(path) for key, path in PATHS.items()}


def test_v4_registrations_are_v4_and_supersede_v3(registrations):
    for key, registration in registrations.items():
        assert registration["registration_version"] == "V4"
        assert registration["supersedes"]["registration_sha256"] == mr.SUPERSEDED_V3[registration["experiment_id"]]
        for entry in registration["provenance"]["reviews"] + registration["provenance"]["previous_reviews"]:
            assert entry["sha256"] == sha256_file(PROJECT / entry["path"])
        assert registration["provenance"]["amendment_sha256"] == sha256_file(PROJECT / mr.AMENDMENT_PATH)
        assert registration["certification"]["blocks_per_forward"] == 4
        assert registration["models"]["openclip-vit-b32-laion2b"]["bytes"] == 605143316
        assert "sealed_access_definition" in registration["access"]
        assert "item id only" not in json.dumps(registration)


def test_v3_q1_is_bound_to_the_planning_file(registrations):
    q1 = next(entry for entry in registrations["021a"]["predictions"] if entry["id"] == "Q1")
    planning_path = PROJECT / q1["reference_source"]["path"]
    assert sha256_file(planning_path) == q1["reference_source"]["sha256"] == registrations["021a"]["power"]["q1_planning"]["sha256"]
    planning = read_json(planning_path)
    body = {key: value for key, value in planning.items() if key != "content_sha256"}
    assert canonical_json_hash(body) == planning["content_sha256"]
    assert q1["saved_draw_reference_points"]["equal_cell_macro"]["value"] == planning["reference_points"]["equal_cell_macro"]["value"]
    assert planning["inputs"]["exp017_artifact_manifest"]["sha256"] == \
        registrations["021a"]["bank_construction"]["exp016_sources"]["exp017_artifact_manifest_sha256"]
    assert q1["tests"]["family"] in {f"pooled__sigma{cell['sigma']:.12g}" for cell in registrations["021a"]["cells"]}
    for estimand in ("equal_cell_macro", "item_pooled"):
        assert set(q1["sentences"][estimand]) == set(analyze_ext.Q1_VERDICTS)


def test_v3_family_power_labels_name_real_families(registrations):
    for key in ("021a", "021b"):
        registration = registrations[key]
        families = {name for cell in registration["cells"] for name in analyze_ext._family_names(cell)}
        assert set(registration["inference"]["family_power"]) == families


def test_v3_every_real_cell_keeps_the_theorem_and_exploratory_sets(registrations):
    for key in ("021a", "021b"):
        block = registrations[key]["candidates"]
        assert len(block["flip_theorem_candidates"]) == 25
        assert block["exploratory_containment_candidates"] == ["control__learned_shared_translation_tangent"]
        for cell in registrations[key]["cells"]:
            assert len(candidates_ext.candidate_specs(registrations[key], cell)) == 36


def test_v3_m2c_the_worker_refuses_a_bank_manifest_built_from_another_preflight(tmp_path, registrations):
    registration = registrations["021b"]
    hashes = {value["experiment_id"]: value["registration_sha256"] for value in registrations.values()}
    preflight = tmp_path / "preflight.json"
    write_json_atomic(preflight, {"schema_version": preflight_data_ext.PREFLIGHT_SCHEMA, "status": "PASS", "failures": [],
                                  "registrations": hashes, "environment": preflight_data_ext.runtime_environment(),
                                  "item_list_file_sha256": {}, "image_file_sha256": {"eurosat": {}, "imagenette": {}}})
    manifest = {"schema_version": bank_manifest_ext.MANIFEST_SCHEMA, "data_preflight_sha256": "0" * 64,
                "registrations": hashes, "cells": {}}
    manifest["content_sha256"] = canonical_json_hash(manifest)
    manifest_path = tmp_path / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    approval = guard_ext.approval_template(hashes)
    approval.update(status=guard_ext.REQUIRED_APPROVAL_STATUS, preparation_authorized=True, sampling_authorized=True,
                    data_preflight_sha256=sha256_file(preflight), bank_manifest_sha256=sha256_file(manifest_path),
                    approved_by="reviewer", approved_on="2026-10-01", source_commit="f" * 40)
    approval.pop("instructions")
    approval_path = tmp_path / "approval.json"
    write_json_atomic(approval_path, approval)
    argv = ["--registration", str(PATHS["021b"]), "--cell-id", registration["cells"][0]["cell_id"],
            "--bank-manifest", str(manifest_path), "--data-preflight", str(preflight), "--data-root", str(tmp_path),
            "--out", str(tmp_path / "out")]
    from satml2027ext import models_ext

    # the offline guard is exercised elsewhere; other tests import huggingface_hub earlier in this process
    # the workspace's EXP-021 files are not committed yet; the commit check has its own test (test_guard_ext.SourceCommitTests)
    with mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}), mock.patch.object(models_ext, "require_offline", lambda: None), \
            mock.patch.object(guard_ext, "verify_source_commit", return_value={"source_commit": "f" * 40}):
        with pytest.raises(RuntimeError, match="another data preflight"):
            run_shard_ext.main(argv, approval_path=approval_path)
    assert not (tmp_path / "out").exists()
