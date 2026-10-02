"""Tests for the V2 preparation stage (V1 review findings P0-C, P0-D).

The 021A path runs on the real saved EXP-016/EXP-017 tensors of one EuroSAT
fold (no model: the prompt control is faked); the 021B path runs the approved
direction builder and the EXP-019 control fitter on a fake encoder.

  .venv/Scripts/python -m pytest satml2027ext/tests/test_prepare_banks_ext.py -q
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from satml2027ext import banks_ext, candidates_ext, items_ext  # noqa: E402
from satml2027ext import make_registrations_ext as mr  # noqa: E402
from satml2027ext import prepare_banks_ext as prep  # noqa: E402
from satml2027ext._common import canonical_json_hash, read_json, sha256_file  # noqa: E402

ITEMS = PROJECT / "results" / "satml2027ext" / "items"
EXP016 = PROJECT / "results" / "EXP-20260906-016"
EXP017 = PROJECT / "results" / "EXP-20260906-017"
MODEL = "openai-clip-vit-b32-quickgelu"
needs_real = pytest.mark.skipif(not (EXP016 / "artifact_manifest.json").is_file() or not (ITEMS / "item_manifest_ext.json").is_file(),
                                reason="saved EXP-016/EXP-017 results or item lists are not present")


def unit(value):
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True)


def fake_prompt(classes: int, dim: int):
    def learn(names, clean, noisy, labels, recipe, *, shots_per_class):
        assert len(names) == classes and shots_per_class >= int(torch.bincount(labels).max())
        generator = torch.Generator().manual_seed(5)
        return SimpleNamespace(prototypes=unit(torch.randn(classes, dim, generator=generator)),
                               context_vectors=torch.randn(4, 8, generator=generator), initial_loss=2.0, final_loss=1.0,
                               hook_difference=0.0)
    return learn


def args_namespace():
    return argparse.Namespace(split_manifest=PROJECT / "artifacts/day14/phase3_splits_v1/manifest.json",
                              exclusion_manifest=PROJECT / "artifacts/day14/phase3_splits_v1/exclusion_manifest.json")


def registration_021a(cell: dict) -> dict:
    return {
        "experiment_id": "EXP-20260921-021A", "registration_sha256": "0" * 64,
        "candidates": candidates_ext.canonical_candidates_block(),
        "bank_construction": mr.bank_construction_021a(),
        "models": mr.MODEL_BINDINGS,
        "prompt_configs": mr.prompt_config_block(["cifar100", "eurosat"]),
        "cells": [cell],
    }


def real_021a_cell(dataset: str = "eurosat", fold: str = "0") -> dict:
    manifest = read_json(ITEMS / "item_manifest_ext.json")
    roles = manifest["studies"]["EXP-20260921-021A"]["datasets"][dataset]["folds"][fold]
    return {"cell_id": f"{MODEL}__{dataset}__fold{fold}__sigma0.25", "model_id": MODEL, "dataset_id": dataset,
            "fold": int(fold), "sigma": 0.25, "class_count": 10 if dataset == "eurosat" else 100,
            "item_count": roles["evaluation"]["count"], "evaluation_items_sha256": roles["evaluation"]["item_ids_sha256"],
            "development_items_sha256": roles["development"]["item_ids_sha256"],
            "development_item_count": roles["development"]["count"]}


def preflight_for(items_dir: Path) -> dict:
    return {"item_list_file_sha256": {path.name: sha256_file(path) for path in Path(items_dir).glob("*.csv")},
            "image_file_sha256": {"eurosat": {}, "imagenette": {}}}


@needs_real
def test_021a_payload_is_built_from_the_exp016_tensors_and_reproduces_exp017():
    cell = real_021a_cell()
    registration = registration_021a(cell)
    backend = prep.Backend(device="cpu", mean=(0.5,) * 3, std=(0.25,) * 3, encode_images=None, load_pixels=None,
                           class_names=lambda dataset: [f"c{i}" for i in range(10)], text_prototypes=None,
                           learn_prompt=fake_prompt(10, 512),
                           checkpoint={"file": "open_clip_model.safetensors", "sha256": mr.MODEL_BINDINGS[MODEL]["sha256"],
                                       "model_state_sha256": "e" * 64})
    payload, cache = prep.prepare_cell(registration, cell, backend=backend, items_dir=ITEMS, preflight=preflight_for(ITEMS),
                                       preflight_sha256="f" * 64, approval_sha256="9" * 64, exp016_root=EXP016,
                                       exp017_root=EXP017, log=lambda message: None)
    assert payload["candidate_order"] == candidates_ext.canonical_candidate_ids()
    reproduction = payload["provenance"]["legacy_reproduction"]
    # V4 (text review of reviewer bundle V3, item 5): the sampled legacy banks are the saved EXP-017 tensors
    assert reproduction["sampled_banks_equal_saved_exp017_bitwise"] == 18 and reproduction["image_mean_equals_exp017_record"]
    assert len(reproduction["formula_conformance"]["per_candidate"]) == 18 and reproduction["max_abs_delta"] <= 2e-6
    saved = torch.load(EXP017 / "cells" / f"{MODEL}__eurosat__fold0" / "candidate_prototype_banks.pt", map_location="cpu",
                       weights_only=True)
    for key, tensor in saved.items():
        assert torch.equal(payload["candidates"][key]["prototypes"], tensor)
        assert payload["candidates"][key]["metadata"]["prototype_source"] == "saved EXP-017 tensor, bit for bit"
        assert torch.equal(payload["sources"]["legacy_banks"][key], tensor)
    assert payload["provenance"]["roles"]["development"]["items_sha256"] == cell["development_items_sha256"]
    assert payload["provenance"]["source_mode"] == "exp016_saved_tensors" and payload["sealed_evaluation_access"] is False
    assert payload["provenance"]["exp016_sources"]["control_noisy_draws_per_item"] == 16
    assert cache["noisy_features"].shape == (50, 16, 512) and cache["role"] == "development"
    banks_ext.validate_registered_bank_payload_v2(payload, registration, cell)
    stats = payload["provenance"]["control_training"]
    assert 0.0 <= stats["lowrank"]["clean_accuracy"] <= 1.0 and stats["shared_delta_norm"] >= 0.0


@needs_real
def test_preparation_refuses_evaluation_lists_and_changed_item_files():
    cell = real_021a_cell()
    registration = registration_021a(cell)
    with pytest.raises(RuntimeError):
        items_ext.load_verified(ITEMS, registration, cell, "evaluation", preparation=True)
    preflight = preflight_for(ITEMS)
    name = items_ext.item_list_path(ITEMS, registration, cell, "development").name
    preflight["item_list_file_sha256"][name] = "0" * 64
    with pytest.raises(RuntimeError, match="preflight"):
        items_ext.load_verified(ITEMS, registration, cell, "development", preflight=preflight, preparation=True)
    with pytest.raises(RuntimeError, match="021C"):
        prep.prepare_cell({**registration, "candidates": candidates_ext.subset_candidates_block(
            candidates_ext.BUDGET_SUBSET_021C, parent_experiment_id="P", parent_registration_sha256="0" * 64)}, cell,
            backend=None, items_dir=ITEMS, preflight=preflight_for(ITEMS), preflight_sha256="f" * 64, approval_sha256="9" * 64,
            exp016_root=EXP016, exp017_root=EXP017)


@needs_real
def test_021a_rejects_an_exp016_file_that_differs_from_its_manifest(tmp_path):
    cell = real_021a_cell()
    registration = registration_021a(cell)
    copy016 = tmp_path / "EXP-016"
    shutil.copytree(EXP016 / "cells" / "openai-clip-vit-b32-quickgelu__eurosat__fold0", copy016 / "cells" / "openai-clip-vit-b32-quickgelu__eurosat__fold0")
    shutil.copy2(EXP016 / "artifact_manifest.json", copy016 / "artifact_manifest.json")
    tensor = torch.load(copy016 / "cells" / "openai-clip-vit-b32-quickgelu__eurosat__fold0" / "clean_features.pt", weights_only=True)
    torch.save(tensor * 1.0001, copy016 / "cells" / "openai-clip-vit-b32-quickgelu__eurosat__fold0" / "clean_features.pt")
    rows = items_ext.load_items(ITEMS, cell, "development", registration)
    with pytest.raises(RuntimeError, match="artifact manifest"):
        prep.load_exp016_sources(registration, cell, rows, exp016_root=copy016, exp017_root=EXP017)


# --------------------------------------------------------------------------- 021B path with a fake encoder


CLASSES, DIM, SIZE = 4, 16, 8


class FakeEncoder:
    def __init__(self):
        self.weight = torch.randn(3 * SIZE * SIZE, DIM, generator=torch.Generator().manual_seed(0))

    def __call__(self, batch):
        return torch.tanh(batch.reshape(batch.shape[0], -1) @ self.weight)


def write_list(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset_id", "item_id", "dataset_index", "label", "source_split"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_021b_path_builds_directions_controls_and_a_valid_payload(tmp_path):
    development = [{"dataset_id": "cifar100", "item_id": f"cifar100-train-{90000 + i}", "dataset_index": str(i),
                    "label": str(i % CLASSES), "source_split": "cifar100_reserved_not_accessed"} for i in range(12)]
    control = [{"dataset_id": "cifar100", "item_id": f"cifar100-train-{91000 + i}", "dataset_index": str(100 + i),
                "label": str(i % CLASSES), "source_split": "cifar100_reserved_not_accessed"} for i in range(8)]
    write_list(tmp_path / "exp021b__cifar100_test__development.csv", development)
    write_list(tmp_path / "exp021b__cifar100_test__control_train.csv", control)
    cell = {"cell_id": f"{MODEL}__cifar100_test__sigma0.25", "model_id": MODEL, "dataset_id": "cifar100_test", "sigma": 0.25,
            "class_count": CLASSES, "item_count": 1, "evaluation_items_sha256": "0" * 64,
            "development_items_sha256": canonical_json_hash([row["item_id"] for row in development]), "development_item_count": 12,
            "control_train_items_sha256": canonical_json_hash([row["item_id"] for row in control]), "control_train_item_count": 8}
    construction = mr.bank_construction_021b(args_namespace())
    construction["direction_construction"]["development_draws_per_item"] = 8   # keep the test small
    construction["control_training"]["epochs"] = 5
    registration = {"experiment_id": "EXP-20260921-021B", "registration_sha256": "1" * 64,
                    "candidates": candidates_ext.canonical_candidates_block(), "bank_construction": construction,
                    "models": mr.MODEL_BINDINGS, "prompt_configs": mr.prompt_config_block(["cifar100_test"]), "cells": [cell]}
    encoder = FakeEncoder()
    prototypes = unit(torch.randn(CLASSES, DIM, generator=torch.Generator().manual_seed(3)))
    backend = prep.Backend(
        device="cpu", mean=(0.5,) * 3, std=(0.25,) * 3, encode_images=encoder,
        load_pixels=lambda key, rows: torch.rand((len(rows), 3, SIZE, SIZE), generator=torch.Generator().manual_seed(len(rows))),
        class_names=lambda dataset: [f"c{i}" for i in range(CLASSES)], text_prototypes=lambda dataset: prototypes,
        learn_prompt=fake_prompt(CLASSES, DIM),
        checkpoint={"file": "x", "sha256": mr.MODEL_BINDINGS[MODEL]["sha256"], "model_state_sha256": "e" * 64})
    payload, cache = prep.prepare_cell(registration, cell, backend=backend, items_dir=tmp_path, preflight=preflight_for(tmp_path),
                                       preflight_sha256="f" * 64, approval_sha256="9" * 64, exp016_root=EXP016,
                                       exp017_root=EXP017, log=lambda message: None)
    banks_ext.validate_registered_bank_payload_v2(payload, registration, cell)
    provenance = payload["provenance"]
    assert set(provenance["roles"]) == {"development", "control_train"} and provenance["evaluation_items_loaded"] == 0
    assert provenance["feature_hashes"]["development_noisy"] is not None
    assert cache["noisy_features"].shape == (8, 16, DIM) and cache["item_ids"] == [row["item_id"] for row in control]
    summary = prep.write_cell(payload, cache, banks_dir=tmp_path / "banks", preparation_dir=tmp_path / "prep")
    assert summary["bank_file_sha256"] == sha256_file(tmp_path / "banks" / f"{cell['cell_id']}__banks.pt")
    with pytest.raises(FileExistsError):
        prep.write_cell(payload, cache, banks_dir=tmp_path / "banks", preparation_dir=tmp_path / "prep")
    reloaded = banks_ext.load_bank_payload(tmp_path / "banks" / f"{cell['cell_id']}__banks.pt")
    banks_ext.validate_registered_bank_payload_v2(reloaded, registration, cell)


def test_v3_m1_the_registered_few_shot_budget_is_an_integer_and_enforced():
    from satml2027ext import prompt_bank

    assert prep.registered_shots({"prompt_control": {"shots_per_class": 5}}) == 5
    for value in ("5", None, 0, True):
        with pytest.raises(RuntimeError):
            prep.registered_shots({"prompt_control": {"shots_per_class": value}})
    assert mr.bank_construction_021a()["prompt_control"]["shots_per_class"] == 5
    assert mr.bank_construction_021b(args_namespace())["prompt_control"]["shots_per_class"] == 20
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    features = unit(torch.randn(6, 8, generator=torch.Generator().manual_seed(1)))
    noisy = unit(torch.randn(6, 2, 8, generator=torch.Generator().manual_seed(2)))
    prompt_bank._validate_supervision(noisy, features, labels, 2, 3)
    with pytest.raises(ValueError, match="few-shot budget"):
        prompt_bank._validate_supervision(noisy, features, labels, 2, 2)


@needs_real
def test_v3_m1b_bank_manifest_rederives_the_021a_sources_from_exp016():
    from satml2027ext import bank_manifest_ext

    cell = real_021a_cell()
    registration = registration_021a(cell)
    backend = prep.Backend(device="cpu", mean=(0.5,) * 3, std=(0.25,) * 3, encode_images=None, load_pixels=None,
                           class_names=lambda dataset: [f"c{i}" for i in range(10)], text_prototypes=None,
                           learn_prompt=fake_prompt(10, 512),
                           checkpoint={"file": "open_clip_model.safetensors", "sha256": mr.MODEL_BINDINGS[MODEL]["sha256"],
                                       "model_state_sha256": "e" * 64})
    payload, _ = prep.prepare_cell(registration, cell, backend=backend, items_dir=ITEMS, preflight=preflight_for(ITEMS),
                                   preflight_sha256="f" * 64, approval_sha256="9" * 64, exp016_root=EXP016,
                                   exp017_root=EXP017, log=lambda message: None)
    origin = bank_manifest_ext.exp021a_origin_check(payload, registration, cell, items_dir=ITEMS, exp016_root=EXP016,
                                                    exp017_root=EXP017)
    assert origin["sources_equal_exp016"] and origin["legacy_formula_conformance_max_delta_recomputed"] <= 2e-6
    # V4: identity is required bit for bit; the exact reconstruction count is a runtime-dependent diagnostic
    assert origin["legacy_banks_equal_exp017_bitwise"] == 18 and origin["image_mean_equals_exp017_record"] is True
    assert 0 <= origin["legacy_formula_exact_matches_diagnostic"] <= 18
    forged = dict(payload)
    forged["sources"] = dict(payload["sources"])
    forged["sources"]["legacy_banks"] = dict(payload["sources"]["legacy_banks"])
    key = "global_mean_centering__coefficient_1"
    forged["sources"]["legacy_banks"][key] = torch.nextafter(forged["sources"]["legacy_banks"][key],
                                                             torch.tensor(2.0)).contiguous()
    with pytest.raises(RuntimeError, match="differs from the bound EXP-017 file"):
        bank_manifest_ext.exp021a_origin_check(forged, registration, cell, items_dir=ITEMS, exp016_root=EXP016, exp017_root=EXP017)
    forged = dict(payload)
    forged["source_hashes"] = dict(payload["source_hashes"], clean_direction="0" * 64)
    with pytest.raises(RuntimeError, match="differ from the bound EXP-016 tensors"):
        bank_manifest_ext.exp021a_origin_check(forged, registration, cell, items_dir=ITEMS, exp016_root=EXP016, exp017_root=EXP017)


@needs_real
def test_v4_portable_image_mean_equals_the_exp017_calibration_record_in_every_cell():
    """The registered image-mean algorithm reproduces the calibration mean hash that EXP-017 recorded (all 12 cells)."""

    cells = sorted(path.name for path in (EXP017 / "cells").iterdir())
    assert len(cells) == 12
    for name in cells:
        features = torch.load(EXP016 / "cells" / name / "clean_features.pt", map_location="cpu", weights_only=True).to(torch.float32)
        recorded = {entry.get("calibration_image_mean_sha256")
                    for entry in read_json(EXP017 / "cells" / name / "candidate_metadata.json").values()
                    if entry.get("calibration_image_mean_sha256")}
        assert {banks_ext.tensor_scientific_hash(banks_ext.portable_column_mean(features.contiguous()))} == recorded, name
