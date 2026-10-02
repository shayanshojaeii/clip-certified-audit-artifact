"""Tests for the EXP-021 V2 worker on a fake encoder (no model, no dataset, no GPU).

  .venv/Scripts/python -m pytest satml2027ext/tests/test_run_shard_ext.py -q
"""

from __future__ import annotations

import copy
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import torch  # noqa: E402

from satml2027ext import banks_ext, candidates_ext, guard_ext  # noqa: E402
from satml2027ext import run_shard_ext as worker  # noqa: E402
from satml2027ext._common import canonical_json_hash, read_json, write_json_atomic  # noqa: E402
from common.banks import normalize_rows  # noqa: E402
from common.seeds import derive_block_seed, derive_item_seed, block_plan, NOISE_BLOCK  # noqa: E402

DIM, SIZE, CLASSES = 16, 8, 4
MODEL, DATASET = "openai-clip-vit-b32-quickgelu", "cifar100_test"


class FakeEncoder:
    """Deterministic image encoder: tanh of a fixed random projection of the pixels."""

    def __init__(self, seed: int = 0):
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(3 * SIZE * SIZE, DIM, generator=generator)

    def __call__(self, batch):
        return torch.tanh(batch.reshape(batch.shape[0], -1) @ self.weight)


def unit(value):
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True)


def make_registration(n_items: int = 6, selection: int = 64, confirmation: int = 200, *, checkpoints=(),
                      experiment_id: str = "EXP-20260921-021B", candidates=None) -> tuple[dict, list[dict]]:
    item_ids = [f"fake-{index:02d}" for index in (5, 1, 3, 0, 4, 2, 7, 6, 9, 8)[:n_items]]  # deliberately unsorted
    body = {
        "schema_version": guard_ext.REGISTRATION_SCHEMA,
        "experiment_id": experiment_id,
        "status": "preregistered_pending_independent_approval",
        "candidates": candidates or candidates_ext.canonical_candidates_block(),
        "bank_construction": {"source_mode": "encode_registered_roles", "control_training": {"role": "control_train"},
                              "prompt_control": {"steps": 1}},
        "models": {MODEL: {"sha256": "a" * 64}},
        "prompt_configs": {DATASET: {"path": "p.json", "sha256": "b" * 64}},
        "sharding": {"max_items_per_shard": 1000},
        "certification": {
            "selection_draws": selection, "confirmation_draws": confirmation, "alpha_per_example": 0.001,
            "reported_radii": [0.0, 0.1, 0.25, 0.5],
            "seeds": {"selection_base_seed": 20260925001, "confirmation_base_seed": 20260925002},
            "streams": {"selection": "cohen_class_selection", "confirmation": "cohen_class_confirmation"},
            "noise_block_draws": 64, "budget_checkpoints": list(checkpoints),
            "margin_storage": {"bank": "no_correction", "stream": "confirmation"},
            "flip_counters": {"tolerance": 1e-5},
        },
        "cells": [{
            "cell_id": f"{MODEL}__{DATASET}__sigma0.25", "model_id": MODEL, "dataset_id": DATASET,
            "sigma": 0.25, "class_count": CLASSES, "item_count": n_items,
            "evaluation_items_sha256": canonical_json_hash(item_ids),
            "development_items_sha256": "c" * 64, "control_train_items_sha256": "d" * 64,
            "consumes_sealed_split": True,
        }],
    }
    body["registration_sha256"] = canonical_json_hash(body)
    rows = [{"item_id": item_id, "dataset_index": str(index), "label": str(index % CLASSES)} for index, item_id in enumerate(item_ids)]
    return body, rows


def make_payload(registration: dict, seed: int = 1) -> dict:
    cell = registration["cells"][0]
    generator = torch.Generator().manual_seed(seed)
    prototypes = normalize_rows(torch.randn(CLASSES, DIM, generator=generator, dtype=torch.float64)).to(torch.float32)
    features = normalize_rows(torch.randn(40, DIM, generator=generator, dtype=torch.float64)
                              + 0.9 * normalize_rows(torch.randn(1, DIM, generator=generator, dtype=torch.float64))).to(torch.float32)
    sources = {
        "prototypes": prototypes, "development_clean_features": features,
        "clean_direction": normalize_rows(torch.randn(1, DIM, generator=generator, dtype=torch.float64))[0],
        "noisy_direction": normalize_rows(torch.randn(1, DIM, generator=generator, dtype=torch.float64))[0],
        "controls": {
            "shared_delta": 0.3 * torch.randn(DIM, generator=generator),
            "lowrank_left": 0.3 * torch.randn(CLASSES, 8, generator=generator),
            "lowrank_right": 0.3 * torch.randn(8, DIM, generator=generator),
            "noisy_features": unit(torch.randn(40, 3, DIM, generator=generator)),
            "labels": torch.arange(40) % CLASSES,
            "fewshot_prompt_prototypes": unit(torch.randn(CLASSES, DIM, generator=generator)),
        },
    }
    banks = banks_ext.build_registered_banks(candidates_ext.candidate_specs(registration, cell), sources)
    provenance = {
        "registration_sha256": registration["registration_sha256"], "cell_id": cell["cell_id"], "sealed_evaluation_access": False,
        "roles": {"development": {"items_sha256": "c" * 64}, "control_train": {"items_sha256": "d" * 64}},
        "recipe_sha256": canonical_json_hash(registration["bank_construction"]),
        "source_mode": registration["bank_construction"]["source_mode"],
        "checkpoint": {"sha256": "a" * 64, "model_state_sha256": "e" * 64},
        "prompt_config": {"path": "p.json", "sha256": "b" * 64}, "data_preflight_sha256": "f" * 64,
        "code_sha256": {"x": "1" * 64},
    }
    provenance.update(banks_ext.provenance_source_bindings(sources))
    return banks_ext.make_bank_payload(banks, registration=registration, cell=cell, sources=sources, provenance=provenance)


def make_pixels(n_items: int, seed: int = 7):
    return torch.rand((n_items, 3, SIZE, SIZE), generator=torch.Generator().manual_seed(seed))


class WorkerHarness:
    """Everything ``process_shard`` needs, on CPU, with the fake encoder and real V2 banks."""

    def __init__(self, root: Path, n_items: int = 6, selection: int = 64, confirmation: int = 200, checkpoints=()):
        self.root = root
        self.registration, self.rows = make_registration(n_items, selection, confirmation, checkpoints=checkpoints)
        self.cell = dict(self.registration["cells"][0])
        self.payload = make_payload(self.registration)
        self.candidate_ids, self.banks, self.scorer_meta = worker.load_and_validate_banks(self.payload, self.registration, self.cell)
        self.encode = FakeEncoder()
        self.pixels = make_pixels(n_items)
        self.mean = torch.full((1, 3, 1, 1), 0.45)
        self.std = torch.full((1, 3, 1, 1), 0.27)
        worker.configure_precision()
        self.environment = worker.environment_metadata("cpu", 2)
        self.custody = {key: "0" * 64 for key in worker.CUSTODY_KEYS}

    def run(self, out_name: str, *, shard_index: int = 0, shard_count: int = 1, blocks_per_forward: int = 2,
            checkpoint_every: int = 10, store_margins: bool = True, stop_after_items: int | None = None,
            registration=None, cell=None, scorer_meta=None, environment=None, custody=None) -> Path:
        offset, shard_rows = worker.shard_slice(self.rows, shard_index, shard_count)
        return worker.process_shard(
            registration=registration or self.registration, cell=cell or self.cell, shard_rows=shard_rows,
            offset=offset, shard_index=shard_index, shard_count=shard_count, banks=self.banks,
            candidate_ids=self.candidate_ids, scorer_meta=scorer_meta or self.scorer_meta, encode=self.encode,
            pixels=self.pixels[offset : offset + len(shard_rows)], mean=self.mean, std=self.std, device="cpu",
            out_dir=self.root / out_name, blocks_per_forward=blocks_per_forward, checkpoint_every=checkpoint_every,
            store_margins=store_margins, environment=environment or self.environment, approval_sha256="0" * 64,
            custody=custody or self.custody, stop_after_items=stop_after_items, log=lambda message: None,
        )


class ScorerTests(unittest.TestCase):
    def test_precision_switches_are_off(self):
        worker.configure_precision()
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        environment = worker.environment_metadata("cpu", 3)
        self.assertEqual(environment["forward_batch_draws"], 3 * 64)

    def test_image_transform_applied_per_candidate(self):
        registration, _ = make_registration()
        candidate_ids, banks, _ = worker.load_and_validate_banks(make_payload(registration), registration, registration["cells"][0])
        self.assertEqual(candidate_ids, candidates_ext.canonical_candidate_ids())
        scorer = worker.Scorer(banks, candidate_ids, "no_correction", "cpu")
        features = unit(torch.randn(37, DIM, generator=torch.Generator().manual_seed(3)))
        logits = scorer.logits(features)
        for position, candidate_id in enumerate(candidate_ids):
            entry = banks[candidate_id]
            transform = entry["image_transform"]
            z = features if transform is None else unit(features - transform["coefficient"] * transform["mean"])
            self.assertTrue(torch.allclose(logits[:, position, :], z @ entry["prototypes"].T, atol=1e-6), candidate_id)

    def test_identity_candidate_must_not_be_transformed(self):
        registration, _ = make_registration()
        payload = make_payload(registration)

        def validator(payload, registration, cell):
            banks = dict(payload["candidates"])
            banks["no_correction"] = {**banks["no_correction"], "image_transform": {"type": "center_and_renormalize",
                                                                                    "mean": torch.zeros(DIM), "coefficient": 1.0}}
            return banks
        with self.assertRaises(RuntimeError):
            worker.load_and_validate_banks(payload, registration, registration["cells"][0], validator=validator)

    def test_validator_import_fails_closed(self):
        registration, _ = make_registration()
        with mock.patch.object(worker.importlib, "import_module", side_effect=ImportError("no banks_ext")):
            with self.assertRaises(RuntimeError):
                worker.load_and_validate_banks(make_payload(registration), registration, registration["cells"][0])

    def test_subset_registration_scores_exactly_its_registered_banks(self):
        parent, _ = make_registration()
        subset_block = candidates_ext.subset_candidates_block(candidates_ext.BUDGET_SUBSET_021C,
                                                              parent_experiment_id=parent["experiment_id"],
                                                              parent_registration_sha256=parent["registration_sha256"])
        child, _ = make_registration(experiment_id="EXP-20260921-021C", candidates=subset_block)
        child_cell = {**child["cells"][0], "cell_id": "child", "parent_cell_id": parent["cells"][0]["cell_id"]}
        payload = make_payload(parent)
        ids, banks, _ = worker.load_and_validate_banks(payload, child, child_cell, parent_registration=parent,
                                                       parent_cell=parent["cells"][0])
        self.assertEqual(ids, list(candidates_ext.BUDGET_SUBSET_021C))
        self.assertEqual(list(banks), ids)
        with self.assertRaises(RuntimeError):
            worker.load_and_validate_banks(payload, child, child_cell)
        with self.assertRaises(RuntimeError):
            worker.load_and_validate_banks(payload, child, child_cell, parent_registration={**parent, "registration_sha256": "9" * 64},
                                           parent_cell=parent["cells"][0])


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp())
        self.harness = WorkerHarness(self.temp)

    def tearDown(self):
        shutil.rmtree(self.temp, ignore_errors=True)

    def test_block_invariance_and_shard_independence(self):
        one = worker.load_npz_closed(self.harness.run("bpf1", blocks_per_forward=1))
        three = worker.load_npz_closed(self.harness.run("bpf3", blocks_per_forward=3))
        keys = ("selection_counts", "confirmation_counts", "raw_predictions", "selection_seeds", "confirmation_seeds") + worker.FLIP_COUNTERS
        for key in keys:
            self.assertTrue(np.array_equal(one[key], three[key]), key)
        parts = [worker.load_npz_closed(self.harness.run("sharded", shard_index=index, shard_count=2, blocks_per_forward=4))
                 for index in range(2)]
        for key in ("selection_counts", "confirmation_counts") + worker.FLIP_COUNTERS:
            self.assertTrue(np.array_equal(np.concatenate([part[key] for part in parts], axis=1), one[key]), key)
        self.assertEqual(list(np.concatenate([part["item_ids"] for part in parts])), [row["item_id"] for row in self.harness.rows])
        self.assertTrue(np.all(one["confirmation_counts"].sum(axis=2) == 200))

    def test_flip_counters_match_a_brute_force_recount_and_the_theorem_holds(self):
        shard = worker.load_npz_closed(self.harness.run("counters"))
        scorer = worker.Scorer(self.harness.banks, self.harness.candidate_ids, "no_correction", "cpu")
        certification = guard_ext.certification_parameters(self.harness.registration)
        identity = scorer.identity_position
        expected = {name: np.zeros((len(self.harness.candidate_ids), len(self.harness.rows)), dtype=np.int64)
                    for name in ("changed", "useful", "harmful", "containment", "loose", "budget")}
        for position, row in enumerate(self.harness.rows):
            seed = derive_item_seed(base_seed=certification["confirmation_base_seed"], model_id=MODEL, dataset_id=DATASET,
                                    sigma=0.25, item_id=row["item_id"], stream_role=certification["confirmation_stream_role"])
            pieces = []
            for block_index, block_draws in block_plan(200):
                generator = torch.Generator().manual_seed(derive_block_seed(seed, block_index))
                pieces.append(torch.randn((NOISE_BLOCK, 3, SIZE, SIZE), generator=generator)[:block_draws])
            noise = torch.cat(pieces)
            pixel = self.harness.pixels[position]
            features = unit(self.harness.encode(((pixel.unsqueeze(0) + 0.25 * noise) - self.harness.mean) / self.harness.std))
            logits = scorer.logits(features).double().numpy()
            labels = logits.argmax(axis=2)
            truth = int(row["label"])
            j = labels[:, identity]
            scores = logits[:, identity, :]
            top = np.sort(scores, axis=1)
            margin = top[:, -1] - top[:, -2]
            for c, candidate_id in enumerate(self.harness.candidate_ids):
                k = labels[:, c]
                changed = k != j
                expected["changed"][c, position] = changed.sum()
                expected["useful"][c, position] = (changed & (j != truth) & (k == truth)).sum()
                expected["harmful"][c, position] = (changed & (j == truth) & (k != truth)).sum()
                operator = self.harness.banks[candidate_id]["metadata"]["operator"]
                if c == identity or self.harness.banks[candidate_id]["image_transform"] is not None or not operator.get("known"):
                    continue
                a, n = np.array(operator["row_scales"]), np.array(operator["row_norms"])
                if not np.all(a > 0):
                    continue
                pair = scores[np.arange(len(j)), j] - scores[np.arange(len(j)), k]
                threshold = np.abs(n[j] - n[k]) + np.abs(a[j] - a[k])
                expected["containment"][c, position] = (changed & (a[j] * pair > threshold + 1e-5)).sum()
                if np.all(np.abs(a - 1) <= 1e-12):
                    bound = 2 * min(1.0, operator["shared_vector_norm"])
                    expected["loose"][c, position] = (changed & (margin > bound + 1e-5)).sum()
                    expected["budget"][c, position] = (margin <= bound + 1e-5).sum()
        mapping = {"changed": "flip_changed_draws", "useful": "flip_useful_draws", "harmful": "flip_harmful_draws",
                   "containment": "flip_containment_violations", "loose": "flip_loose_violations", "budget": "flip_loose_budget_draws"}
        for name, key in mapping.items():
            self.assertTrue(np.array_equal(expected[name], shard[key]), key)
        # the decision-change proposition: no applicable bank ever violates it
        self.assertEqual(int(shard["flip_containment_violations"].sum()), 0)
        self.assertEqual(int(shard["flip_loose_violations"].sum()), 0)
        applicable = shard["flip_containment_applicable"]
        steps = [c for c, candidate_id in enumerate(self.harness.candidate_ids) if candidates_ext.step_value(candidate_id)]
        self.assertTrue(all(applicable[c] and shard["flip_loose_applicable"][c] for c in steps))
        self.assertGreater(int(shard["flip_changed_draws"][steps].sum()), 0)  # the large steps do change labels
        projected = [c for c, candidate_id in enumerate(self.harness.candidate_ids) if candidate_id.startswith("chowers")]
        self.assertEqual(int(shard["flip_changed_draws"][projected].sum()), 0)

    def test_counters_detect_a_violation_when_the_operator_record_is_wrong(self):
        banks = copy.deepcopy(self.harness.banks)
        target = "clean_boundary_active__step_0.32"
        operator = banks[target]["metadata"]["operator"]
        operator["row_norms"] = [1.0] * CLASSES  # claims equal normalizers: every change would violate
        operator["shared_vector_norm"] = 0.0
        scorer = worker.Scorer(banks, self.harness.candidate_ids, "no_correction", "cpu")
        result = worker.count_item(encode=self.harness.encode, scorer=scorer, pixel=self.harness.pixels[0],
                                   mean=self.harness.mean, std=self.harness.std, item_seed=123, draws=640, sigma=0.25,
                                   blocks_per_forward=2, device="cpu", counters=True, truth=0)
        position = self.harness.candidate_ids.index(target)
        changed = int(result["counters"]["flip_changed_draws"][position])
        self.assertGreater(changed, 0)
        self.assertEqual(int(result["counters"]["flip_loose_violations"][position]), changed)

    def test_budget_prefix_equals_a_shorter_run_of_the_same_stream(self):
        long = WorkerHarness(self.temp / "long", confirmation=200, checkpoints=(96,))
        short = WorkerHarness(self.temp / "short", confirmation=96)
        long_data = worker.load_npz_closed(long.run("long", blocks_per_forward=3))  # 192-draw chunks: the prefix ends mid-chunk
        self.assertEqual(long_data["prefix_confirmation_counts"].shape, (1, 36, 6, CLASSES))
        # identical seeds and items; the 96-draw run is the prefix of the 200-draw run
        short.registration = {**short.registration}
        short_data = worker.load_npz_closed(short.run("short", blocks_per_forward=1))
        self.assertTrue(np.array_equal(long_data["prefix_confirmation_counts"][0], short_data["confirmation_counts"]))
        self.assertTrue(np.array_equal(long_data["confirmation_seeds"], short_data["confirmation_seeds"]))

    def test_large_confirmation_budget_and_cap(self):
        harness = WorkerHarness(self.temp / "big", n_items=1, confirmation=100_000, checkpoints=(4096,))
        data = worker.load_npz_closed(harness.run("big", blocks_per_forward=32, store_margins=False))
        self.assertTrue(np.all(data["confirmation_counts"].sum(axis=2) == 100_000))
        self.assertTrue(np.all(data["prefix_confirmation_counts"].sum(axis=3) == 4096))
        registration, _ = make_registration(n_items=1, confirmation=100_001)
        harness.registration = registration
        harness.cell = dict(registration["cells"][0])
        with self.assertRaises(RuntimeError):
            harness.run("too_big", store_margins=False)

    def test_sidecar_records_custody_and_sealed_access(self):
        shard = self.harness.run("meta")
        meta = read_json(shard.with_suffix(".meta.json"))
        self.assertEqual(meta["schema_version"], worker.SHARD_META_SCHEMA)
        self.assertTrue(meta["sealed_evaluation_access"])
        self.assertEqual(meta["custody"], self.harness.custody)
        self.assertEqual(meta["scored_candidate_ids"], self.harness.candidate_ids)
        self.assertEqual(meta["flip_counters"]["tolerance"], 1e-5)

    def test_resume_provenance(self):
        reference = worker.load_npz_closed(self.harness.run("reference"))
        with self.assertRaises(worker.StopRequested):
            self.harness.run("resume", checkpoint_every=1, stop_after_items=2)
        shard = self.temp / "resume" / "cells" / self.harness.cell["cell_id"] / "shard_000_of_001.npz"
        self.assertEqual(int(worker.load_npz_closed(shard)["done"].sum()), 2)
        resumed = worker.load_npz_closed(self.harness.run("resume", checkpoint_every=1))
        for key in ("selection_counts", "confirmation_counts", "raw_predictions", "selection_seeds", "confirmation_seeds") + worker.FLIP_COUNTERS:
            self.assertTrue(np.array_equal(resumed[key], reference[key]), key)
        with self.assertRaises(worker.StopRequested):
            self.harness.run("tamper", checkpoint_every=1, stop_after_items=1)
        tamper = self.temp / "tamper" / "cells" / self.harness.cell["cell_id"] / "shard_000_of_001.npz"
        with self.assertRaises(RuntimeError):  # a different bank manifest
            self.harness.run("tamper", custody={**self.harness.custody, "bank_manifest_sha256": "1" * 64})
        with self.assertRaises(RuntimeError):  # a different device model
            self.harness.run("tamper", environment={**self.harness.environment, "device_name": "NVIDIA RTX 9999"})
        meta = read_json(tamper.with_suffix(".meta.json"))
        meta["registration_sha256"] = "f" * 64
        write_json_atomic(tamper.with_suffix(".meta.json"), meta)
        with self.assertRaises(RuntimeError):
            self.harness.run("tamper")


class ItemOrderTests(unittest.TestCase):
    def test_registered_order_is_kept_and_verified(self):
        registration, rows = make_registration()
        temp = Path(tempfile.mkdtemp())
        try:
            import csv

            path = temp / f"exp021b__{DATASET}__evaluation.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["item_id", "dataset_index", "label"])
                writer.writeheader()
                writer.writerows(rows)
            loaded = worker.load_items(temp, registration["cells"][0], "evaluation", registration)
            self.assertEqual([row["item_id"] for row in loaded], [row["item_id"] for row in rows])
            worker.verify_item_list(loaded, registration["cells"][0], "evaluation")
            with self.assertRaises(RuntimeError):  # the V1 sort would fail the ordered hash
                worker.verify_item_list(sorted(loaded, key=lambda row: row["item_id"]), registration["cells"][0], "evaluation")
        finally:
            shutil.rmtree(temp, ignore_errors=True)


class SealedEuroSatIndexingTests(unittest.TestCase):
    def test_sealed_rows_are_located_by_item_id_not_by_the_global_index(self):
        """V1 indexed the 1,000-item sealed dataset with the global EuroSAT index from the CSV."""

        from satml2027ext import datasets_ext

        ids = [f"eurosat/2750/Forest/Forest_{index}.jpg" for index in (40, 10, 30, 20)]
        labels = [1, 1, 1, 1]

        def fake_loader(dataset_id, root, transform=None, **kwargs):
            return "dataset", ids, labels

        rows = [{"item_id": ids[2], "dataset_index": "9030", "label": "1"}, {"item_id": ids[0], "dataset_index": "9040", "label": "1"}]
        seen = {}

        def fake_recover(dataset, positions, mean, std):
            seen["positions"] = list(positions)
            return torch.zeros((len(positions), 3, 2, 2))

        loaded = mock.Mock(preprocess=None, mean=(0.5,) * 3, std=(0.5,) * 3)
        with mock.patch.object(datasets_ext, "load_dataset_ext", fake_loader), mock.patch("common.datasets.recover_pixels", fake_recover):
            worker.load_pixels_for_cell({"dataset_id": "eurosat_sealed"}, rows, Path("."), loaded, None, None)
        self.assertEqual(seen["positions"], [2, 0])
        with mock.patch.object(datasets_ext, "load_dataset_ext", fake_loader), mock.patch("common.datasets.recover_pixels", fake_recover):
            with self.assertRaises(RuntimeError):
                worker.load_pixels_for_cell({"dataset_id": "eurosat_sealed"}, [{**rows[0], "label": "7"}], Path("."), loaded, None, None)


class MainGuardTests(unittest.TestCase):
    def test_main_refuses_without_an_approval_or_its_stage_files(self):
        temp = Path(tempfile.mkdtemp())
        try:
            registration, _ = make_registration()
            path = temp / "registration.json"
            write_json_atomic(path, registration)
            argv = ["--registration", str(path), "--cell-id", registration["cells"][0]["cell_id"], "--out", str(temp / "out"),
                    "--bank-manifest", str(temp / "manifest.json"), "--data-preflight", str(temp / "preflight.json"),
                    "--data-root", str(temp)]
            with self.assertRaises(guard_ext.ApprovalError):
                worker.main(argv, approval_path=temp / "missing.json")
            with self.assertRaises(SystemExit):
                worker.main(["--registration", str(path), "--cell-id", "x", "--out", str(temp)], approval_path=temp / "missing.json")
            self.assertEqual(worker.DEFAULT_APPROVAL_PATH, PROJECT / "research" / "EXP021_APPROVAL.json")
        finally:
            shutil.rmtree(temp, ignore_errors=True)


class V3WorkerTests(unittest.TestCase):
    """Internal re-review of V2 (2026-09-26): n1 (021C full-stack scoring), m3 (custody labels), determinism."""

    def test_v3_n1_subset_scored_with_the_parent_stack_equals_the_parent_rows(self):
        import json

        with tempfile.TemporaryDirectory() as temp:
            harness = WorkerHarness(Path(temp), n_items=4, selection=64, confirmation=200)
            full_path = harness.run("full", store_margins=False)
            subset = list(candidates_ext.BUDGET_SUBSET_021C)
            offset, rows = worker.shard_slice(harness.rows, 0, 1)
            path = worker.process_shard(
                registration=harness.registration, cell=harness.cell, shard_rows=rows, offset=offset, shard_index=0,
                shard_count=1, banks=harness.banks, candidate_ids=subset,
                scorer_meta={key: harness.scorer_meta[key] for key in subset}, encode=harness.encode, pixels=harness.pixels,
                mean=harness.mean, std=harness.std, device="cpu", out_dir=Path(temp) / "subset", blocks_per_forward=2,
                checkpoint_every=10, store_margins=False, environment=harness.environment, approval_sha256="0" * 64,
                custody=harness.custody, log=lambda message: None, scoring_ids=harness.candidate_ids)
            with np.load(full_path) as archive:
                full = {key: np.array(archive[key]) for key in archive.files}
            with np.load(path) as archive:
                part = {key: np.array(archive[key]) for key in archive.files}
            index = [harness.candidate_ids.index(value) for value in subset]
            for key in ("selection_counts", "confirmation_counts", "raw_predictions", *worker.FLIP_COUNTERS,
                        "flip_containment_applicable", "flip_loose_applicable"):
                np.testing.assert_array_equal(part[key], full[key][index], err_msg=key)
            self.assertEqual([str(value) for value in part["candidate_ids"]], subset)
            meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["flip_counters"]["scoring_stack"], harness.candidate_ids)

    def test_v3_m3_custody_refuses_labels_that_differ_from_the_registered_list(self):
        import csv as csv_module

        from satml2027ext import custody_ext

        with tempfile.TemporaryDirectory() as temp:
            harness = WorkerHarness(Path(temp), n_items=4, selection=32, confirmation=64)
            harness.run("results", store_margins=False)
            items = Path(temp) / "items"
            items.mkdir()
            stem = "exp021b__" + harness.cell["dataset_id"] + "__evaluation.csv"

            def write(rows):
                with (items / stem).open("w", encoding="utf-8", newline="") as handle:
                    writer = csv_module.DictWriter(handle, fieldnames=["dataset_id", "item_id", "dataset_index", "label", "source_split"])
                    writer.writeheader()
                    for row in rows:
                        writer.writerow({"dataset_id": harness.cell["dataset_id"], "source_split": "x", **row})

            expected = {"approval_sha256": "0" * 64, "custody": dict(harness.custody)}
            write(harness.rows)
            record = custody_ext.verify_cell(harness.registration, harness.cell, Path(temp) / "results", items, expected=expected)
            self.assertEqual(record["items"], 4)
            wrong = [dict(row) for row in harness.rows]
            wrong[1]["label"] = str((int(wrong[1]["label"]) + 1) % CLASSES)
            write(wrong)
            with self.assertRaisesRegex(RuntimeError, "ground_truth"):
                custody_ext.verify_cell(harness.registration, harness.cell, Path(temp) / "results", items, expected=expected)

    def test_v41_custody_build_refuses_unlisted_files_in_a_real_worker_output(self):
        import csv as csv_module
        import shutil

        from satml2027ext import custody_ext

        with tempfile.TemporaryDirectory() as temp:
            harness = WorkerHarness(Path(temp), n_items=4, selection=32, confirmation=64)
            harness.run("results", store_margins=False)
            items = Path(temp) / "items"
            items.mkdir()
            with (items / ("exp021b__" + harness.cell["dataset_id"] + "__evaluation.csv")).open("w", encoding="utf-8", newline="") as handle:
                writer = csv_module.DictWriter(handle, fieldnames=["dataset_id", "item_id", "dataset_index", "label", "source_split"])
                writer.writeheader()
                for row in harness.rows:
                    writer.writerow({"dataset_id": harness.cell["dataset_id"], "source_split": "x", **row})
            expected = {"approval_sha256": "0" * 64, "custody": dict(harness.custody)}
            results = Path(temp) / "results"
            cell_dir = results / "cells" / harness.cell["cell_id"]
            record = custody_ext.verify_cell(harness.registration, harness.cell, results, items, expected=expected)
            self.assertEqual(sorted(record["files"]), sorted(path.name for path in cell_dir.iterdir()))
            sidecar = next(cell_dir.glob("*.meta.json"))
            for stray in ("shard_-001_of_1.meta.json", sidecar.name.replace(".meta.json", "_margins.npz")):
                shutil.copyfile(sidecar, cell_dir / stray)
                with self.assertRaises(RuntimeError):
                    custody_ext.verify_cell(harness.registration, harness.cell, results, items, expected=expected)
                (cell_dir / stray).unlink()
            custody_ext.verify_cell(harness.registration, harness.cell, results, items, expected=expected)

    def test_v3_m4_not_run_cells_must_have_no_result_files(self):
        from satml2027ext import custody_ext

        with tempfile.TemporaryDirectory() as temp:
            registration, _ = make_registration()
            registration["cells"].append(dict(registration["cells"][0], cell_id="second"))
            results = Path(temp)
            self.assertEqual(custody_ext.not_run_check(registration, results, ["second"]), ["second"])
            (results / "cells" / "second").mkdir(parents=True)
            (results / "cells" / "second" / "shard_000_of_001.npz").write_bytes(b"x")
            with self.assertRaisesRegex(RuntimeError, "result files"):
                custody_ext.not_run_check(registration, results, ["second"])
            with self.assertRaisesRegex(RuntimeError, "unregistered"):
                custody_ext.not_run_check(registration, results, ["nope"])
            with self.assertRaisesRegex(RuntimeError, "every registered cell"):
                custody_ext.not_run_check(registration, Path(temp) / "empty", [cell["cell_id"] for cell in registration["cells"]])

    def test_v3_precision_requests_deterministic_algorithms(self):
        import os

        extra = worker.configure_precision()
        self.assertTrue(extra["deterministic_algorithms"])
        self.assertEqual(os.environ.get("CUBLAS_WORKSPACE_CONFIG"), extra["cublas_workspace_config"])


class V3CustodyPreflightTests(unittest.TestCase):
    def test_v3_m2c_custody_refuses_a_manifest_that_names_another_preflight(self):
        from satml2027ext import bank_manifest_ext, custody_ext

        with tempfile.TemporaryDirectory() as temp:
            registration, _ = make_registration()
            preflight = Path(temp) / "preflight.json"
            write_json_atomic(preflight, {"any": "record"})
            manifest = {"schema_version": bank_manifest_ext.MANIFEST_SCHEMA, "data_preflight_sha256": "0" * 64,
                        "registrations": {}, "cells": {}}
            manifest["content_sha256"] = canonical_json_hash(manifest)
            path = Path(temp) / "manifest.json"
            write_json_atomic(path, manifest)
            with self.assertRaisesRegex(RuntimeError, "another data preflight"):
                custody_ext.build_custody(registration, Path(temp), Path(temp), bank_manifest=path, data_preflight=preflight)


if __name__ == "__main__":
    unittest.main()
