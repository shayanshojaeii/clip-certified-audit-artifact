"""Tests for the canonical V2 candidate family (single source of truth; V1 finding P0-A)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from satml2027ext import candidates_ext as ce  # noqa: E402


def test_canonical_family_has_36_ids_and_the_registered_count_rule():
    block = ce.canonical_candidates_block()
    assert block["expected_bank_count"] == 36 == len(block["candidate_ids"]) == len(set(block["candidate_ids"]))
    assert block["schema_version"] == ce.CANDIDATES_SCHEMA_VERSION
    assert ce.candidate_specs({"candidates": block}) and ce.registered_candidate_ids({"candidates": block}) == block["candidate_ids"]
    assert len(block["flip_theorem_candidates"]) == 25  # 3 + 3 + 3 + 8 + 8 text translations
    assert set(block["projected_gap_candidates"]) == {f"chowers_exact_projected_gap__coefficient_{v}" for v in ("0.25", "0.5", "1")}
    assert set(ce.PRIMARY_SHARED_CONTRASTS) <= set(block["candidate_ids"])
    assert set(ce.CONTROL_CONTRASTS) <= set(block["candidate_ids"])
    assert not set(ce.PRIMARY_SHARED_CONTRASTS) & set(ce.CONTROL_CONTRASTS)
    assert "text_only_centering__coefficient_1" in ce.PRIMARY_SHARED_CONTRASTS
    assert all(not contrast.startswith(ce.CONTROL_PREFIX) for contrast in ce.PRIMARY_SHARED_CONTRASTS)


def test_expansion_must_agree_with_the_explicit_id_list():
    block = ce.canonical_candidates_block()
    tampered = dict(block, candidate_ids=list(reversed(block["candidate_ids"])))
    with pytest.raises(ValueError):
        ce.candidate_specs({"candidates": tampered})
    with pytest.raises(ValueError):
        ce.candidate_specs({"candidates": dict(block, schema_version="satml2027ext.candidates.v1")})
    flat_v1 = {"bank_count": 33, "candidate_ids": block["candidate_ids"], "candidates": block["candidates"]}
    with pytest.raises(ValueError):
        ce.candidate_specs({"candidates": flat_v1})


def test_subset_block_keeps_parent_order_and_identity():
    subset = ce.subset_candidates_block(ce.BUDGET_SUBSET_021C, parent_experiment_id="P", parent_registration_sha256="0" * 64)
    specs = ce.candidate_specs({"candidates": subset})
    assert [spec["candidate_id"] for spec in specs] == list(ce.BUDGET_SUBSET_021C)
    parent_positions = {spec["candidate_id"]: spec["position"] for spec in ce.canonical_specs()}
    assert [spec["position"] for spec in specs] == [parent_positions[value] for value in ce.BUDGET_SUBSET_021C]
    assert ce.is_subset_registration({"candidates": subset}) and ce.subset_parent({"candidates": subset})["experiment_id"] == "P"
    with pytest.raises(ValueError):
        ce.subset_candidates_block(["gr_clip_style_two_sided__coefficient_1"], parent_experiment_id="P", parent_registration_sha256="0")
    with pytest.raises(ValueError):
        ce.subset_candidates_block(list(reversed(ce.BUDGET_SUBSET_021C)), parent_experiment_id="P", parent_registration_sha256="0")


def test_decomposition_triple_and_step_values():
    block = ce.canonical_candidates_block()
    assert block["decomposition"]["text_only"] == [f"text_only_centering__coefficient_{v}" for v in ("0.25", "0.5", "1")]
    assert ce.step_value("clean_boundary_active__step_0.16") == 0.16
    assert ce.step_value("global_mean_centering__coefficient_1") is None
    assert ce.flip_theorem_ids(["no_correction", "text_only_centering__coefficient_1", "image_only_centering__coefficient_1"]) == [
        "text_only_centering__coefficient_1"]
