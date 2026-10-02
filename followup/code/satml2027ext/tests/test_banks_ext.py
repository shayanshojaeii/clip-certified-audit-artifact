"""Tests for satml2027ext.banks_ext (V2: 36 banks, provenance, recomputation).

Run from the repository root:
  .venv/Scripts/python -m pytest satml2027ext/tests -q
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from satml2027ext import banks_ext  # noqa: E402
from satml2027ext.banks_ext import (  # noqa: E402
    apply_image_transform,
    build_registered_banks,
    candidate_specs,
    default_candidates_block,
    load_bank_payload,
    make_bank_payload,
    save_bank_payload,
    validate_registered_bank_payload_v2,
)
from satml2027ext._common import canonical_json_hash  # noqa: E402
from common.banks import build_eighteen_candidate_banks, normalize_rows  # noqa: E402
from common.hashing import tensor_scientific_hash  # noqa: E402
from interventions.boundary_active import apply_common_translation  # noqa: E402
from interventions.core import GRCLIPStyle, MeanCentering, ProjectedGapCorrection  # noqa: E402

SEED = 20260925
EXP016_CELL = ROOT / "results" / "EXP-20260906-016" / "cells" / "openai-clip-vit-b32-quickgelu__cifar100__fold0"
EXP017_CELL = ROOT / "results" / "EXP-20260906-017" / "cells" / "openai-clip-vit-b32-quickgelu__cifar100__fold0"
LEGACY_STEPS = [0.0025, 0.005, 0.01, 0.02, 0.04]
LEGACY_COEFFICIENTS = [0.25, 0.5, 1.0]

EXPECTED_ORDER = (
    ["no_correction"]
    + [f"global_mean_centering__coefficient_{v}" for v in ("0.25", "0.5", "1")]
    + [f"chowers_exact_projected_gap__coefficient_{v}" for v in ("0.25", "0.5", "1")]
    + [f"gr_clip_style_two_sided__coefficient_{v}" for v in ("0.25", "0.5", "1")]
    + [f"text_only_centering__coefficient_{v}" for v in ("0.25", "0.5", "1")]
    + [f"image_only_centering__coefficient_{v}" for v in ("0.25", "0.5", "1")]
    + [f"clean_boundary_active__step_{v}" for v in ("0.0025", "0.005", "0.01", "0.02", "0.04", "0.08", "0.16", "0.32")]
    + [f"cohen_aligned_noisy_margin__step_{v}" for v in ("0.0025", "0.005", "0.01", "0.02", "0.04", "0.08", "0.16", "0.32")]
    + [
        "control__learned_shared_translation_tangent",
        "control__lowrank_tangent_r8",
        "control__noisy_class_mean",
        "control__fewshot_prompt_bank",
    ]
)
MODEL = "openai-clip-vit-b32-quickgelu"


def registration() -> dict:
    return {
        "experiment_id": "EXP-20260921-021B",
        "registration_sha256": "0" * 64,
        "candidates": default_candidates_block(),
        "bank_construction": {"source_mode": "encode_registered_roles", "control_training": {"role": "control_train"},
                              "prompt_control": {"steps": 300}},
        "models": {MODEL: {"sha256": "a" * 64}},
        "prompt_configs": {"cifar100_test": {"path": "configs/prompts/cifar100_openai_readme.json", "sha256": "b" * 64}},
    }


def cell() -> dict:
    return {"cell_id": f"{MODEL}__cifar100_test__sigma0.25", "class_count": 12, "dataset_id": "cifar100_test",
            "model_id": MODEL, "development_items_sha256": "c" * 64, "control_train_items_sha256": "d" * 64}


def provenance(reg: dict, c: dict) -> dict:
    return {
        "registration_sha256": reg["registration_sha256"], "cell_id": c["cell_id"], "sealed_evaluation_access": False,
        "roles": {"development": {"items_sha256": c["development_items_sha256"]},
                  "control_train": {"items_sha256": c["control_train_items_sha256"]}},
        "recipe_sha256": canonical_json_hash(reg["bank_construction"]),
        "source_mode": reg["bank_construction"]["source_mode"],
        "checkpoint": {"sha256": "a" * 64, "model_state_sha256": "e" * 64},
        "prompt_config": {"path": "configs/prompts/cifar100_openai_readme.json", "sha256": "b" * 64},
        "data_preflight_sha256": "f" * 64,
        "code_sha256": {"satml2027ext/banks_ext.py": "1" * 64},
    }


def random_sources(*, classes=12, dim=32, items=60, draws=3, dtype=torch.float32, seed=SEED, bias=0.0):
    generator = torch.Generator().manual_seed(seed)
    prototypes = normalize_rows(torch.randn(classes, dim, generator=generator, dtype=torch.float64)).to(dtype)
    offset = bias * normalize_rows(torch.randn(1, dim, generator=generator, dtype=torch.float64))
    features = normalize_rows(torch.randn(items, dim, generator=generator, dtype=torch.float64) + offset).to(dtype)
    labels = torch.arange(items) % classes
    clean = normalize_rows(torch.randn(1, dim, generator=generator, dtype=torch.float64))[0]
    noisy = normalize_rows(torch.randn(1, dim, generator=generator, dtype=torch.float64))[0]
    noisy_features = normalize_rows(
        torch.randn(items * draws, dim, generator=generator, dtype=torch.float64)
    ).reshape(items, draws, dim).to(torch.float32)
    fewshot = normalize_rows(torch.randn(classes, dim, generator=generator, dtype=torch.float64)).to(torch.float32)
    return {
        "prototypes": prototypes,
        "development_clean_features": features,
        "development_labels": labels,
        "clean_direction": clean,
        "noisy_direction": noisy,
        "controls": {
            "shared_delta": 0.05 * torch.randn(dim, generator=generator),
            "lowrank_left": 0.1 * torch.randn(classes, 8, generator=generator),
            "lowrank_right": 0.1 * torch.randn(8, dim, generator=generator),
            "noisy_features": noisy_features,
            "labels": labels,
            "fewshot_prompt_prototypes": fewshot,
            "fewshot_context_vectors": torch.randn(4, 16, generator=generator),
        },
    }


def full_payload(sources=None):
    reg, c = registration(), cell()
    sources = sources or random_sources()
    banks = build_registered_banks(candidate_specs(reg, c), sources)
    record = provenance(reg, c)
    record.update(banks_ext.provenance_source_bindings(sources))
    return make_bank_payload(banks, registration=reg, cell=c, sources=sources, provenance=record), reg, c


# --------------------------------------------------------------------------- specification


def test_candidate_specs_expand_to_the_registered_36_in_order():
    specs = candidate_specs(registration(), cell())
    assert [spec["candidate_id"] for spec in specs] == EXPECTED_ORDER
    assert [spec["position"] for spec in specs] == list(range(36))
    legacy = {spec["candidate_id"] for spec in specs if spec["legacy_exp017"]}
    assert len(legacy) == 18 and "gr_clip_style_two_sided__coefficient_0.5" not in legacy
    assert not any(spec["legacy_exp017"] for spec in specs if spec["family"] == "text_only_centering")
    image_side = {spec["candidate_id"] for spec in specs if spec["image_transform"] is not None}
    assert image_side == {f"{family}__coefficient_{v}" for family in ("gr_clip_style_two_sided", "image_only_centering")
                          for v in ("0.25", "0.5", "1")}


def test_block_carries_the_formulas_of_the_implemented_operators():
    block = default_candidates_block()
    forms = {entry["candidate_id"]: entry["form"] for entry in block["candidates"]}
    assert forms["global_mean_centering__coefficient_1"] == "t'_k = normalize(t_k - 1 (mean_w(t) - mu_I)); z' = z"
    assert forms["chowers_exact_projected_gap__coefficient_0.5"] == "t'_k = normalize(t_k + 0.5 (I - V V^T)(mu_I - mean_w(t))); z' = z"
    assert forms["text_only_centering__coefficient_1"] == "t'_k = normalize(t_k - 1 mean(t)); z' = z"
    assert forms["gr_clip_style_two_sided__coefficient_1"] == "t'_k = normalize(t_k - 1 mean(t)); z' = normalize(z - 1 mu_I)"


# --------------------------------------------------------------------------- operator identities


@pytest.mark.parametrize("coefficient", LEGACY_COEFFICIENTS)
def test_text_mean_centering_matches_interventions_mean_centering(coefficient):
    sources = random_sources()
    specs = [s for s in candidate_specs(registration(), cell()) if s["family"] == "global_mean_centering"]
    banks = build_registered_banks(specs, sources)
    image_mean = sources["development_clean_features"].mean(dim=0)
    reference = MeanCentering(image_mean, coefficient=coefficient).apply(sources["prototypes"]).prototypes
    ours = banks[f"global_mean_centering__coefficient_{coefficient:.12g}"]["prototypes"]
    assert float((ours - reference).abs().max()) <= 1e-6


@pytest.mark.parametrize("coefficient", LEGACY_COEFFICIENTS)
def test_projected_gap_matches_interventions_chowers_operator(coefficient):
    sources = random_sources()
    specs = [s for s in candidate_specs(registration(), cell()) if s["family"] == "chowers_exact_projected_gap"]
    banks = build_registered_banks(specs, sources)
    image_mean = sources["development_clean_features"].mean(dim=0)
    reference = ProjectedGapCorrection(image_mean, coefficient=coefficient).apply(sources["prototypes"]).prototypes
    ours = banks[f"chowers_exact_projected_gap__coefficient_{coefficient:.12g}"]["prototypes"]
    assert float((ours - reference).abs().max()) <= 1e-6


@pytest.mark.parametrize("coefficient", LEGACY_COEFFICIENTS)
def test_two_sided_centring_matches_gr_clip_style_at_scaled_means(coefficient):
    sources = random_sources(bias=0.8)
    specs = [s for s in candidate_specs(registration(), cell()) if s["family"] == "gr_clip_style_two_sided"]
    banks = build_registered_banks(specs, sources)
    prototypes, features = sources["prototypes"], sources["development_clean_features"]
    image_mean, text_mean = features.mean(dim=0), prototypes.mean(dim=0)
    reference = GRCLIPStyle(coefficient * image_mean, coefficient * text_mean).apply(prototypes, features)
    entry = banks[f"gr_clip_style_two_sided__coefficient_{coefficient:.12g}"]
    assert float((entry["prototypes"] - reference.prototypes).abs().max()) <= 1e-6
    transformed = apply_image_transform(features, entry["image_transform"])
    assert float((transformed - reference.image_features).abs().max()) <= 1e-6


@pytest.mark.parametrize("coefficient", LEGACY_COEFFICIENTS)
def test_text_only_is_the_exact_text_half_of_the_two_sided_operator(coefficient):
    sources = random_sources(bias=0.8)
    specs = candidate_specs(registration(), cell())
    banks = build_registered_banks(specs, sources)
    key = f"{coefficient:.12g}"
    two_sided = banks[f"gr_clip_style_two_sided__coefficient_{key}"]
    text_only = banks[f"text_only_centering__coefficient_{key}"]
    image_only = banks[f"image_only_centering__coefficient_{key}"]
    mean_centering = banks[f"global_mean_centering__coefficient_{key}"]
    # identical prototype tensors; the text half has no image transform
    assert torch.equal(text_only["prototypes"], two_sided["prototypes"])
    assert text_only["image_transform"] is None
    direct = normalize_rows(sources["prototypes"] - coefficient * sources["prototypes"].mean(dim=0).unsqueeze(0))
    assert float((text_only["prototypes"] - direct).abs().max()) <= 1e-6
    # two-sided scores = text-only prototypes applied to image-only-transformed features
    features = sources["development_clean_features"]
    two_scores = apply_image_transform(features, two_sided["image_transform"]) @ two_sided["prototypes"].T
    composed = apply_image_transform(features, image_only["image_transform"]) @ text_only["prototypes"].T
    assert float((two_scores - composed).abs().max()) <= 1e-6
    # the historical mean-centering bank is a different text translation (V1 used it as the "text half")
    assert float((mean_centering["prototypes"] - text_only["prototypes"]).abs().max()) > 1e-3


@pytest.mark.parametrize("objective,key", [("clean_boundary_active", "clean_direction"),
                                           ("cohen_aligned_noisy_margin", "noisy_direction")])
def test_common_translations_match_apply_common_translation(objective, key):
    sources = random_sources()
    specs = [s for s in candidate_specs(registration(), cell()) if s.get("objective_id") == objective]
    banks = build_registered_banks(specs, sources)
    assert len(banks) == 8
    for spec in specs:
        reference = apply_common_translation(sources["prototypes"], sources[key].to(torch.float32), spec["value"])
        assert float((banks[spec["candidate_id"]]["prototypes"] - reference).abs().max()) <= 1e-6


def test_all_36_build_with_unit_rows_and_recorded_hashes():
    payload, reg, c = full_payload()
    assert list(payload["candidates"]) == EXPECTED_ORDER
    for candidate_id, entry in payload["candidates"].items():
        norms = torch.linalg.vector_norm(entry["prototypes"].double(), dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=2e-5, rtol=0.0)
        assert entry["metadata"]["output_prototype_sha256"] == tensor_scientific_hash(entry["prototypes"])
    assert list(validate_registered_bank_payload_v2(payload, reg, c)) == EXPECTED_ORDER
    assert "noisy_features" not in payload["sources"]["controls"]  # large caches stay outside the payload
    assert max(banks_ext.recomputation_report(payload, reg, c).values()) == 0.0


# --------------------------------------------------------------------------- legacy reproduction


@pytest.mark.skipif(not (EXP016_CELL / "text_prototypes.pt").is_file() or
                    not (EXP017_CELL / "candidate_prototype_banks.pt").is_file(),
                    reason="saved EXP-016/EXP-017 cell not present")
def test_legacy_18_rebuild_from_exp016_matches_saved_exp017_banks():
    prototypes = torch.load(EXP016_CELL / "text_prototypes.pt", map_location="cpu", weights_only=True).to(torch.float32).contiguous()
    features = torch.load(EXP016_CELL / "clean_features.pt", map_location="cpu", weights_only=True)
    directions = torch.load(EXP016_CELL / "candidate_directions.pt", map_location="cpu", weights_only=True)
    saved = torch.load(EXP017_CELL / "candidate_prototype_banks.pt", map_location="cpu", weights_only=True)

    reg = registration()
    c = {"cell_id": EXP016_CELL.name, "class_count": int(prototypes.shape[0]), "dataset_id": "cifar100"}
    legacy_specs = [spec for spec in candidate_specs(reg, c) if spec["legacy_exp017"]]
    assert {spec["candidate_id"] for spec in legacy_specs} == set(saved)
    banks = build_registered_banks(legacy_specs, {
        "prototypes": prototypes, "development_clean_features": features, "image_mean": features.mean(dim=0),
        "clean_direction": directions["clean_direction"], "noisy_direction": directions["noisy_direction"],
    })
    for candidate_id, tensor in saved.items():
        delta = float((tensor.double() - banks[candidate_id]["prototypes"].double()).abs().max())
        assert delta <= 2e-6, f"{candidate_id}: max|delta|={delta:.3e}"
    ported = build_eighteen_candidate_banks(
        prototypes=prototypes, development_clean_features=features,
        clean_direction=directions["clean_direction"], noisy_direction=directions["noisy_direction"],
        common_steps=LEGACY_STEPS, gap_coefficients=LEGACY_COEFFICIENTS,
    )
    for candidate_id, entry in ported.items():
        assert torch.equal(entry["prototypes"], banks[candidate_id]["prototypes"]), candidate_id


# --------------------------------------------------------------------------- projected gap and image-only


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_projected_gap_is_orthogonal_to_prototype_differences_and_decision_invariant(seed):
    sources = random_sources(dtype=torch.float64, seed=seed, bias=0.5)
    prototypes = sources["prototypes"]
    projected = banks_ext.projected_gap_direction(prototypes, sources["development_clean_features"].mean(dim=0))
    differences = prototypes[:, None, :] - prototypes[None, :, :]
    assert float((differences @ projected).abs().max()) <= 1e-9
    specs = [s for s in candidate_specs(registration(), cell()) if s["family"] == "chowers_exact_projected_gap"]
    banks = build_registered_banks(specs, sources)
    generator = torch.Generator().manual_seed(seed + 100)
    draws = normalize_rows(torch.randn(5000, prototypes.shape[1], generator=generator, dtype=torch.float64))
    base = (draws @ prototypes.T).argmax(dim=1)
    for spec in specs:
        entry = banks[spec["candidate_id"]]
        normalizers = entry["metadata"]["operator"]["row_norms"]
        assert max(normalizers) - min(normalizers) <= 1e-9
        assert torch.equal((draws @ entry["prototypes"].T).argmax(dim=1), base)


def test_image_only_centring_changes_some_argmax_but_not_the_prototypes():
    sources = random_sources(bias=1.5, items=400)
    specs = [s for s in candidate_specs(registration(), cell()) if s["family"] == "image_only_centering"]
    banks = build_registered_banks(specs, sources)
    prototypes = sources["prototypes"]
    generator = torch.Generator().manual_seed(7)
    offset = sources["development_clean_features"].mean(dim=0)
    draws = normalize_rows(torch.randn(2000, prototypes.shape[1], generator=generator) + 1.5 * offset)
    base = (draws @ prototypes.T).argmax(dim=1)
    for spec in specs:
        entry = banks[spec["candidate_id"]]
        assert torch.equal(entry["prototypes"], prototypes)
        transformed = apply_image_transform(draws, entry["image_transform"])
        assert int(((transformed @ prototypes.T).argmax(dim=1) != base).sum()) > 0, spec["candidate_id"]


# --------------------------------------------------------------------------- validator


def test_validator_accepts_the_built_payload_and_the_saved_roundtrip(tmp_path):
    payload, reg, c = full_payload()
    pt, js = save_bank_payload(payload, tmp_path / "cell__banks.pt")
    assert pt.is_file() and js.is_file()
    loaded = load_bank_payload(pt)
    assert list(validate_registered_bank_payload_v2(loaded, reg, c)) == EXPECTED_ORDER
    with pytest.raises(FileExistsError):
        save_bank_payload(payload, tmp_path / "cell__banks.pt")


def test_self_consistent_payload_built_from_other_sources_is_rejected():
    """P0-D: every hash agrees with the payload, but the banks were not built from the stored sources."""

    payload, reg, c = full_payload()
    forged = copy.deepcopy(payload)
    other = normalize_rows(torch.randn(1, 32, generator=torch.Generator().manual_seed(99), dtype=torch.float64))[0].to(torch.float32)
    forged["sources"]["clean_direction"] = other
    forged["source_hashes"]["clean_direction"] = tensor_scientific_hash(other)
    for entry in forged["candidates"].values():
        entry["metadata"]["source_hashes"] = copy.deepcopy(forged["source_hashes"])
    # V3 (re-review M1a): the provenance commits to the stored direction, so a forger must edit it too
    with pytest.raises(RuntimeError, match="direction_hashes.clean"):
        validate_registered_bank_payload_v2(forged, reg, c, recompute=False)
    forged["provenance"]["direction_hashes"]["clean"] = tensor_scientific_hash(other)
    validate_registered_bank_payload_v2(forged, reg, c, recompute=False)  # full self-consistency alone passes
    with pytest.raises(RuntimeError, match="not the registered operator applied to the stored sources"):
        validate_registered_bank_payload_v2(forged, reg, c)


def test_provenance_is_bound_to_the_registration():
    payload, reg, c = full_payload()
    for path, value in ((("roles", "control_train", "items_sha256"), "9" * 64),
                        (("checkpoint", "sha256"), "9" * 64),
                        (("prompt_config", "sha256"), "9" * 64),
                        (("recipe_sha256",), "9" * 64),
                        (("sealed_evaluation_access",), True)):
        broken = copy.deepcopy(payload)
        target = broken["provenance"]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(RuntimeError):
            validate_registered_bank_payload_v2(broken, reg, c)
    broken = copy.deepcopy(payload)
    del broken["provenance"]["roles"]["control_train"]
    with pytest.raises(RuntimeError, match="control_train"):
        validate_registered_bank_payload_v2(broken, reg, c)


def test_validator_rejects_missing_extra_nonunit_and_tampered_banks():
    payload, reg, c = full_payload()
    broken = copy.deepcopy(payload)
    del broken["candidates"]["clean_boundary_active__step_0.16"]
    with pytest.raises(RuntimeError, match="missing"):
        validate_registered_bank_payload_v2(broken, reg, c)
    broken = copy.deepcopy(payload)
    broken["candidates"]["control__learned_shared_translation_pure"] = copy.deepcopy(broken["candidates"]["no_correction"])
    with pytest.raises(RuntimeError, match="extra"):
        validate_registered_bank_payload_v2(broken, reg, c)
    broken = copy.deepcopy(payload)
    entry = broken["candidates"]["global_mean_centering__coefficient_0.5"]
    tensor = entry["prototypes"].clone()
    tensor[0] *= 1.01
    entry["prototypes"] = tensor
    entry["metadata"]["output_prototype_sha256"] = tensor_scientific_hash(tensor)
    with pytest.raises(RuntimeError, match="non-unit"):
        validate_registered_bank_payload_v2(broken, reg, c)
    broken = copy.deepcopy(payload)
    entry = broken["candidates"]["cohen_aligned_noisy_margin__step_0.32"]
    tensor = entry["prototypes"].clone()
    tensor[[0, 1]] = tensor[[1, 0]]
    entry["prototypes"] = tensor
    with pytest.raises(RuntimeError, match="hash mismatch"):
        validate_registered_bank_payload_v2(broken, reg, c)
    broken = copy.deepcopy(payload)
    entry["metadata"]["output_prototype_sha256"] = tensor_scientific_hash(tensor)
    broken["candidates"]["cohen_aligned_noisy_margin__step_0.32"] = entry  # self-consistent swap
    # V3 (re-review M1c): the recorded operator no longer reproduces the swapped rows
    with pytest.raises(RuntimeError, match="not the registered operator|recorded operator"):
        validate_registered_bank_payload_v2(broken, reg, c)


def test_validator_rejects_mismatched_image_mean_registration_cell_or_order():
    payload, reg, c = full_payload()
    broken = copy.deepcopy(payload)
    entry = broken["candidates"]["image_only_centering__coefficient_1"]
    entry["image_transform"]["mean"] = entry["image_transform"]["mean"] + 1e-3
    with pytest.raises(RuntimeError, match="image mean"):
        validate_registered_bank_payload_v2(broken, reg, c)
    with pytest.raises(RuntimeError, match="different registration"):
        validate_registered_bank_payload_v2(payload, {**reg, "registration_sha256": "1" * 64}, c)
    with pytest.raises(RuntimeError, match="different cell"):
        validate_registered_bank_payload_v2(payload, reg, {**c, "cell_id": "other"})
    broken = copy.deepcopy(payload)
    broken["candidate_order"] = list(reversed(broken["candidate_order"]))
    with pytest.raises(RuntimeError, match="registered order"):
        validate_registered_bank_payload_v2(broken, reg, c)


def test_loader_rejects_a_bank_file_that_differs_from_its_sidecar(tmp_path):
    payload, _, _ = full_payload()
    pt, _ = save_bank_payload(payload, tmp_path / "cell__banks.pt")
    tampered = dict(payload)
    tampered["candidates"] = copy.deepcopy(payload["candidates"])
    tampered["candidates"]["no_correction"]["metadata"]["role"] = "tampered"
    torch.save(tampered, pt)
    with pytest.raises(RuntimeError, match="sidecar"):
        load_bank_payload(pt)


def test_v3_m1a_provenance_hashes_must_equal_the_stored_sources():
    payload, reg, c = full_payload()
    for block, key in (("feature_hashes", "development_clean"), ("feature_hashes", "control_noisy"),
                       ("feature_hashes", "control_labels"), ("direction_hashes", "noisy"),
                       ("prompt_control", "prototypes_sha256"), ("prompt_control", "context_vectors_sha256")):
        broken = copy.deepcopy(payload)
        broken["provenance"][block][key] = "9" * 64
        with pytest.raises(RuntimeError, match=f"{block}.{key}"):
            validate_registered_bank_payload_v2(broken, reg, c, recompute=False)
    broken = copy.deepcopy(payload)
    broken["provenance"]["control_cache_tensor_sha256"] = {"clean_features": "1" * 64, "noisy_features": "2" * 64, "labels": "3" * 64}
    with pytest.raises(RuntimeError, match="control cache"):
        validate_registered_bank_payload_v2(broken, reg, c, recompute=False)


def test_v3_m1c_operator_records_must_reproduce_the_bank_and_the_recomputation():
    payload, reg, c = full_payload()
    broken = copy.deepcopy(payload)
    broken["candidates"]["clean_boundary_active__step_0.16"]["metadata"]["operator"]["row_norms"][0] += 1e-3
    with pytest.raises(RuntimeError, match="row norms"):
        validate_registered_bank_payload_v2(broken, reg, c, recompute=False)
    broken = copy.deepcopy(payload)
    broken["candidates"]["text_only_centering__coefficient_1"]["metadata"]["operator"] = {"known": False}
    with pytest.raises(RuntimeError, match="does not record its exact operator"):
        validate_registered_bank_payload_v2(broken, reg, c, recompute=False)
    broken = copy.deepcopy(payload)
    tangent = broken["candidates"]["control__learned_shared_translation_tangent"]["metadata"]["operator"]
    vector = torch.tensor(tangent["shared_vector"], dtype=torch.float64) * 2.0
    tangent.update(shared_vector=[float(value) for value in vector],
                   row_scales=[2.0 * float(value) for value in tangent["row_scales"]],
                   row_norms=[2.0 * float(value) for value in tangent["row_norms"]],
                   shared_vector_norm=float(torch.linalg.vector_norm(vector)),
                   shared_vector_sha256=tensor_scientific_hash(vector.to(torch.float32)))
    validate_registered_bank_payload_v2(broken, reg, c, recompute=False)  # a rescaled record still reproduces the bank
    with pytest.raises(RuntimeError, match="differs from its recomputation"):
        validate_registered_bank_payload_v2(broken, reg, c)


def test_v3_n2_tangent_control_counters_are_registered_as_exploratory():
    from satml2027ext import candidates_ext as ce

    block = ce.canonical_candidates_block()
    tangent = "control__learned_shared_translation_tangent"
    assert block["exploratory_containment_candidates"] == [tangent]
    assert tangent not in block["flip_theorem_candidates"] and len(block["flip_theorem_candidates"]) == 25
    entry = next(value for value in block["candidates"] if value["candidate_id"] == tangent)
    assert entry["flip_theorem"].startswith("exploratory")


# --------------------------------------------------------------------------- V4 (reviews of reviewer bundle V3)


def legacy_registration() -> dict:
    reg = registration()
    reg["experiment_id"] = "EXP-20260921-021A"
    reg["bank_construction"] = {**reg["bank_construction"], "source_mode": "exp016_saved_tensors"}
    return reg


def legacy_payload(*, perturb: float = 0.0):
    """A 021A-like payload whose legacy banks are 'saved' tensors; ``perturb`` mimics another runtime's last bits."""

    reg, c = legacy_registration(), cell()
    sources = random_sources()
    specs = candidate_specs(reg, c)
    formula = build_registered_banks(specs, sources)
    legacy = {}
    for index, spec in enumerate(spec for spec in specs if spec.get("legacy_exp017")):
        tensor = formula[spec["candidate_id"]]["prototypes"].clone()
        # the six banks that differed in the last bits on the auditors' runtime (global mean and projected gap)
        if perturb and spec["family"] in ("global_mean_centering", "chowers_exact_projected_gap"):
            noise = torch.randn(tensor.shape, generator=torch.Generator().manual_seed(index), dtype=torch.float64)
            tensor = normalize_rows(tensor.double() + perturb * noise).to(torch.float32)
        legacy[spec["candidate_id"]] = tensor.contiguous()
    sources = {**sources, "legacy_banks": legacy}
    banks = build_registered_banks(specs, sources)
    record = provenance(reg, c)
    record.update(banks_ext.provenance_source_bindings(sources))
    return make_bank_payload(banks, registration=reg, cell=c, sources=sources, provenance=record), reg, c, legacy


def test_v4_legacy_banks_are_sampled_bit_for_bit_and_the_formula_is_only_a_conformance_check():
    payload, reg, c, legacy = legacy_payload(perturb=1e-7)
    assert len(legacy) == 18
    validate_registered_bank_payload_v2(payload, reg, c)
    for key, tensor in legacy.items():
        assert torch.equal(payload["candidates"][key]["prototypes"], tensor)
        assert payload["candidates"][key]["metadata"]["prototype_source"] == "saved EXP-017 tensor, bit for bit"
    assert max(banks_ext.recomputation_report(payload, reg, c).values()) == 0.0
    conformance = banks_ext.legacy_formula_conformance(payload, reg, c)
    assert conformance["banks"] == 18 and conformance["within_tolerance"]
    assert 0.0 < conformance["max_abs_delta"] <= banks_ext.PORTABLE_RECOMPUTE_ATOL
    assert conformance["exact_matches_diagnostic"] == 12  # six perturbed banks: the sampled banks are unaffected


def test_v4_legacy_tensor_tampering_and_misplaced_legacy_tensors_are_refused():
    payload, reg, c, legacy = legacy_payload()
    key = "chowers_exact_projected_gap__coefficient_1"
    broken = copy.deepcopy(payload)
    broken["sources"]["legacy_banks"][key] = torch.nextafter(legacy[key], torch.tensor(2.0)).contiguous()
    with pytest.raises(RuntimeError, match="stored legacy tensor"):
        validate_registered_bank_payload_v2(broken, reg, c)
    broken = copy.deepcopy(payload)
    del broken["sources"]["legacy_banks"][key]
    with pytest.raises(RuntimeError, match="saved EXP-017 tensor of every legacy bank"):
        validate_registered_bank_payload_v2(broken, reg, c)
    ordinary, reg_b, c_b = full_payload()
    broken = copy.deepcopy(ordinary)
    broken["sources"]["legacy_banks"] = dict(legacy)
    with pytest.raises(RuntimeError, match="only a 021A payload"):
        validate_registered_bank_payload_v2(broken, reg_b, c_b)


def test_v4_source_mode_is_bound_to_the_registration():
    payload, reg, c = full_payload()
    broken = copy.deepcopy(payload)
    broken["provenance"]["source_mode"] = "exp016_saved_tensors"
    with pytest.raises(RuntimeError, match="source_mode"):
        validate_registered_bank_payload_v2(broken, reg, c)
    broken = copy.deepcopy(payload)
    del broken["provenance"]["source_mode"]
    with pytest.raises(RuntimeError, match="source_mode"):
        validate_registered_bank_payload_v2(broken, reg, c)
    for mode in (None, "parent_payload", "something_else"):
        with pytest.raises(RuntimeError, match="source_mode"):
            banks_ext.registered_source_mode({"bank_construction": {"source_mode": mode}})


@pytest.mark.parametrize("field,value", [("row_norms", float("nan")), ("row_scales", float("nan")),
                                         ("shared_vector_norm", float("nan")), ("shared_vector", float("inf")),
                                         ("row_norms", 0.0), ("known", "yes")])
def test_v4_non_finite_or_malformed_operator_records_are_refused(field, value):
    payload, reg, c = full_payload()
    broken = copy.deepcopy(payload)
    operator = broken["candidates"]["clean_boundary_active__step_0.16"]["metadata"]["operator"]
    if field in ("shared_vector_norm", "known"):
        operator[field] = value
    else:
        operator[field][0] = value
    with pytest.raises(RuntimeError):
        validate_registered_bank_payload_v2(broken, reg, c, recompute=False)


def test_v4_portable_column_mean_is_close_to_the_exact_mean_and_layout_independent():
    generator = torch.Generator().manual_seed(SEED)
    for rows in (1, 15, 16, 17, 50, 257, 500, 1500):
        for width in (512, 768):
            features = normalize_rows(torch.randn(rows, width, generator=generator, dtype=torch.float64)).to(torch.float32)
            portable = banks_ext.portable_column_mean(features)
            assert portable.dtype == torch.float32 and portable.shape == (width,)
            assert float((portable.double() - features.double().mean(dim=0)).abs().max()) <= 1e-6
            strided = torch.empty(width, rows).T.copy_(features)  # same values, non-contiguous layout
            assert torch.equal(banks_ext.portable_column_mean(strided), portable)
