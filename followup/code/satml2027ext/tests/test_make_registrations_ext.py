"""Tests for the V3 registration generator (schema satml2027ext.registration.v3)."""

from __future__ import annotations

import filecmp
import json
from pathlib import Path

import pytest

from conftest import ROOT

import make_registrations_ext as mr
from _common import canonical_json_hash
from satml2027ext import candidates_ext, guard_ext

REGISTERED_AT = "2026-09-26T12:00:00+00:00"
SPLIT_MANIFEST = ROOT / "artifacts/day14/phase3_splits_v1/manifest.json"
EXCLUSION_MANIFEST = ROOT / "artifacts/day14/phase3_splits_v1/exclusion_manifest.json"
STEMS = ("exp-20260921-021a", "exp-20260921-021b", "exp-20260921-021c")


def run(items: Path, out: Path, *, allow_synthetic: bool) -> None:
    argv = ["--items", str(items), "--out", str(out), "--registered-at", REGISTERED_AT,
            "--split-manifest", str(SPLIT_MANIFEST), "--exclusion-manifest", str(EXCLUSION_MANIFEST)]
    if allow_synthetic:
        argv.append("--allow-synthetic-manifest")
    assert mr.main(argv) == 0


def load(out: Path, stem: str) -> tuple[dict, str]:
    return json.loads((out / f"{stem}.json").read_text(encoding="utf-8")), (out / f"{stem}.sha256").read_text(encoding="ascii")


def test_pending_detection_and_refusal():
    registration = {"experiment_id": "X", "cells": [{"evaluation_items_sha256": mr.PENDING}], "status": None, "registration_sha256": None}
    assert mr.pending_paths(registration) == ["/cells[0]/evaluation_items_sha256"]
    with pytest.raises(mr.PendingItemListError):
        mr.finalize_registration(registration)
    final = mr.finalize_registration({"experiment_id": "X", "cells": [], "registration_sha256": None})
    assert final["status"] == "preregistered_pending_independent_approval" and mr.verify_registration_hash(final)
    assert not mr.verify_registration_hash(dict(final, cells=[{"x": 1}]))


def test_full_manifest_finalizes_and_regenerates_byte_identically(full_build, tmp_path):
    items, _ = full_build
    run(items, tmp_path / "one", allow_synthetic=True)
    run(items, tmp_path / "two", allow_synthetic=True)
    for stem in STEMS:
        assert filecmp.cmp(tmp_path / "one" / f"{stem}.json", tmp_path / "two" / f"{stem}.json", shallow=False)
        registration, sidecar = load(tmp_path / "one", stem)
        assert registration["schema_version"] == guard_ext.REGISTRATION_SCHEMA == "satml2027ext.registration.v4"
        assert registration["registration_version"] == "V4"
        assert sidecar.strip() == registration["registration_sha256"] and mr.verify_registration_hash(registration)
        assert registration["supersedes"]["registration_sha256"] == mr.SUPERSEDED_V3[registration["experiment_id"]]
        assert registration["supersedes"]["earlier"][0]["registration_sha256"] == mr.SUPERSEDED_V2[registration["experiment_id"]]
        assert registration["supersedes"]["earlier"][1]["registration_sha256"] == mr.SUPERSEDED_V1[registration["experiment_id"]]
        assert registration["supersedes"]["reviews"] == list(mr.REVIEWS_V3)
        assert registration["certification"]["blocks_per_forward"] == 4
        assert "item id only" not in registration["certification"]["streams"]["seed_derivation"]
        assert registration["models"]["openclip-vit-b32-laion2b"]["bytes"] == 605143316
        assert [entry["path"] for entry in registration["provenance"]["reviews"]] == list(mr.REVIEWS_V3)
        assert registration["provenance"]["previous_reviews"][0]["path"] == mr.REREVIEW_PATH
        assert set(registration["models"]) == set(mr.MODEL_BINDINGS) and all(len(value["sha256"]) == 64 for value in registration["models"].values())
        assert registration["sharding"]["max_items_per_shard"] == 1000
        assert registration["data_roots"]["item_lists"]["created_at"] == "2026-09-26T00:00:00+00:00"
        assert registration["provenance"]["amendment"] == "research/EXP021_V4_AMENDMENT_2026-09-26.md"
        assert registration["provenance"]["previous_amendments"][0]["path"] == "research/EXP021_V3_AMENDMENT_2026-09-26.md"
        assert "source_commit" in registration["stages"] and "approval.v3" in registration["stages"]["approval_schema"]
        assert "contract_ext.py" in registration["certification"]["flip_counters"]["count_contract"]
        assert registration["certification"]["flip_counters"]["tolerance"] == 1e-5
        assert "final_test_access" not in registration
        candidates_ext.candidate_specs(registration)  # the real block parses (V1 finding P0-A)


def test_blocks_per_experiment(full_build, tmp_path):
    items, manifest = full_build
    run(items, tmp_path, allow_synthetic=True)
    a, _ = load(tmp_path, "exp-20260921-021a")
    b, _ = load(tmp_path, "exp-20260921-021b")
    c, _ = load(tmp_path, "exp-20260921-021c")
    assert (len(a["cells"]), len(b["cells"]), len(c["cells"])) == (12, 18, 6)
    assert a["candidates"]["expected_bank_count"] == b["candidates"]["expected_bank_count"] == 36
    assert c["candidates"]["candidate_ids"] == list(candidates_ext.BUDGET_SUBSET_021C)
    assert c["candidates"]["subset_of"]["registration_sha256"] == b["registration_sha256"]
    assert (a["sealed_evaluation_access"], b["sealed_evaluation_access"], c["sealed_evaluation_access"]) == (False, True, True)
    assert {cell["dataset_id"] for cell in b["cells"] if cell["consumes_sealed_split"]} == {"cifar100_test", "eurosat_sealed"}
    assert all(cell["evaluation_split"] == "val" for cell in b["cells"] if cell["dataset_id"] == "imagenette")
    assert a["bank_construction"]["source_mode"] == "exp016_saved_tensors"
    assert a["bank_construction"]["control_training"]["role"] == "development"
    assert b["bank_construction"]["source_mode"] == "encode_registered_roles"
    assert b["bank_construction"]["control_training"]["role"] == "control_train"
    assert b["bank_construction"]["prompt_control"]["steps"] == 300
    assert c["bank_construction"]["source_mode"] == "parent_payload"
    assert a["certification"]["margin_storage"] is not None and b["certification"]["margin_storage"] is None
    assert c["certification"]["budget_checkpoints"] == [4096] and c["certification"]["confirmation_draws"] == 100000
    assert all(cell["store_identity_margins"] for cell in a["cells"]) and not any(cell["store_identity_margins"] for cell in b["cells"])
    assert b["inference"]["primary_family"]["contrasts"] == list(candidates_ext.PRIMARY_SHARED_CONTRASTS)
    assert b["inference"]["control_family"]["contrasts"] == list(candidates_ext.CONTROL_CONTRASTS)
    assert b["inference"]["secondary_family"]["size"] == 35
    assert a["power"]["unique_images"] == 1650 and a["power"]["at_discordance_0.10"]["power_for_2_points"] == 0.47
    assert b["power"]["items_needed_for_5_contrast_family"]["0.10"] == 2993
    assert [q["id"] for q in a["predictions"]] == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    q1 = a["predictions"][0]
    assert q1["saved_draw_reference_points"]["equal_cell_macro"]["exact"] == "89/30" and q1["registered_prediction"] == "reproduced"
    assert "otherwise" not in q1 and q1["tests"]["family"] == "pooled__sigma0.25"
    # V4 (reviews of reviewer bundle V3): saved legacy tensors, portable image mean, Q1 reporting and wording
    assert a["bank_construction"]["image_mean_algorithm"] == b["bank_construction"]["image_mean_algorithm"] == candidates_ext.IMAGE_MEAN_ALGORITHM
    assert "EXP-017 candidate_metadata.json" in a["bank_construction"]["exp016_sources"]["files_per_cell"]
    assert "bit for bit" in a["bank_construction"]["legacy_banks"] and "legacy_banks" not in b["bank_construction"]
    assert "execution pipeline" in q1["question"] and "18 saved EXP-017 legacy bank tensors" in a["title"]
    assert set(q1["verdict_meaning"]) == set(q1["sentences"]["equal_cell_macro"])
    for sentence in q1["sentences"]["equal_cell_macro"].values():
        assert "{difference:+.2f}" in sentence and "{p_direction:.3f}" in sentence and "{p_shortfall:.3f}" in sentence
    assert "Monte-Carlo part" not in json.dumps(q1)
    q4 = next(entry for entry in a["predictions"] if entry["id"] == "Q4")
    assert "numerical tie with" not in q4["rule"] and "tie budget" in q4["rule"]
    assert "sharp" in q1["prediction_basis"] and "monte_carlo_sd_points" in a["power"]["q1_planning"]
    assert a["power"]["q1_planning"]["sha256"] == q1["reference_source"]["sha256"]
    assert a["bank_construction"]["prompt_control"]["shots_per_class"] == 5 and b["bank_construction"]["prompt_control"]["shots_per_class"] == 20
    assert set(b["inference"]["family_power"]) == {f"{d}__sigma{s}" for d in ("cifar100_test", "imagenette", "eurosat_sealed", "pooled") for s in ("0.12", "0.25")}
    assert b["inference"]["family_power"]["eurosat_sealed__sigma0.25"]["label"].startswith("underpowered")
    assert "complete 36-bank stack" in c["bank_construction"]["rule"]
    assert [q["id"] for q in b["predictions"]] == ["Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]
    assert [q["id"] for q in c["predictions"]] == ["Q9"]
    q3 = next(q for q in a["predictions"] if q["id"] == "Q3")
    assert "not_a_rule" in q3 and "zero containment violations" in q3["rule"]
    for cell in c["cells"]:
        parent = next(p for p in b["cells"] if p["cell_id"] == cell["parent_cell_id"])
        assert cell["development_items_sha256"] == parent["development_items_sha256"]
        assert cell["consumes_sealed_split"] == parent["consumes_sealed_split"]
    fold0 = manifest["studies"]["EXP-20260921-021A"]["datasets"]["cifar100"]["folds"]["0"]
    cell = next(cell for cell in a["cells"] if cell["cell_id"] == "openai-clip-vit-b32-quickgelu__cifar100__fold0__sigma0.25")
    assert cell["evaluation_items_sha256"] == fold0["evaluation"]["item_ids_sha256"]
    assert cell["development_item_count"] == 500


def test_real_only_manifest_yields_drafts_for_pending_lists(real_only_build, tmp_path):
    items, _ = real_only_build
    run(items, tmp_path, allow_synthetic=False)
    a, sidecar_a = load(tmp_path, "exp-20260921-021a")
    b, sidecar_b = load(tmp_path, "exp-20260921-021b")
    assert a["status"] == "preregistered_pending_independent_approval" and len(sidecar_a.strip()) == 64
    assert b["status"] == "draft" and sidecar_b.startswith("DRAFT_NOT_FINALIZED")
    assert mr.pending_paths(b)
