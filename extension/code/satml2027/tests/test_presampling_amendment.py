from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.devices import Device, plan_shards
from common.banks import (
    EXPECTED_CANDIDATE_IDS,
    FROZEN_GRID_CANDIDATE_IDS,
    GR_CLIP_IMAGE_TRANSFORM,
    IDENTITY_IMAGE_TRANSFORM,
    build_eighteen_candidate_banks,
)
from common.hashing import tensor_scientific_hash
from common.prompts import render_prompts
from day1.d1_05_run_shard import _save_npz_atomic
from day1.d1_05_run_shard import validate_bank_payload, validate_resume_metadata
from day1.d1_06_merge_verify import merge_cell
from day1.d1_07_analyze import simultaneous_bootstrap
from day1.d1_08_n3c_predictions import (
    evaluate_cell,
    validate_count_feature_alignment,
    validate_feature_identity,
)


def valid_registered_bank_payload(registration_hash: str = "r", cell_id: str = "c") -> dict:
    prototypes = torch.eye(2, dtype=torch.float32)
    source_hash = tensor_scientific_hash(prototypes)
    image_mean = torch.tensor([0.25, -0.5], dtype=torch.float32)
    banks = {}
    for candidate_id in sorted(EXPECTED_CANDIDATE_IDS):
        tensor = prototypes.clone()
        is_gr = candidate_id == "gr_clip_style_two_sided__coefficient_1"
        entry = {
            "prototypes": tensor,
            "metadata": {
                "candidate_id": candidate_id,
                "source_prototype_sha256": source_hash,
                "output_prototype_sha256": tensor_scientific_hash(tensor),
                "image_transform": GR_CLIP_IMAGE_TRANSFORM if is_gr else IDENTITY_IMAGE_TRANSFORM,
                "calibration_image_mean_sha256": tensor_scientific_hash(image_mean) if is_gr else None,
            },
        }
        if is_gr:
            entry["image_mean"] = image_mean.clone()
        banks[candidate_id] = entry
    return {"registration_sha256": registration_hash, "cell_id": cell_id, "banks": banks}


class PresamplingAmendmentTests(unittest.TestCase):
    def test_exact_frozen_candidate_ids_match_the_builder(self):
        bank = torch.eye(3, dtype=torch.float32)
        built = build_eighteen_candidate_banks(
            prototypes=bank,
            development_clean_features=bank,
            clean_direction=torch.tensor([1.0, 0.0, 0.0]),
            noisy_direction=torch.tensor([0.0, 1.0, 0.0]),
            common_steps=[0.0025, 0.005, 0.01, 0.02, 0.04],
            gap_coefficients=[0.25, 0.5, 1.0],
        )
        self.assertEqual(set(built), FROZEN_GRID_CANDIDATE_IDS)

    def test_prompt_renderer_accepts_named_and_positional_templates(self):
        self.assertEqual(render_prompts(["cat"], ["a {}", "the {class_name}"]), [["a cat", "the cat"]])

    def test_atomic_npz_does_not_append_a_second_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "checkpoint.npz"
            _save_npz_atomic(target, value=np.arange(3))
            self.assertTrue(target.is_file())
            self.assertFalse((Path(directory) / "checkpoint.npz.tmp.npz").exists())
            np.testing.assert_array_equal(np.load(target)["value"], np.arange(3))

    def test_planner_keeps_every_cell_on_one_server(self):
        cells = [
            {"cell_id": "a", "model_id": "m", "dataset_id": "d", "sigma": 0.25, "item_count": 20},
            {"cell_id": "b", "model_id": "m", "dataset_id": "d", "sigma": 0.25, "item_count": 20},
        ]
        devices = [Device("A", 0, "other"), Device("A", 1, "other"), Device("B", 0, "other")]
        jobs = plan_shards(cells=cells, devices=devices, throughput_table={}, draws_per_item=8)
        for cell_id in ("a", "b"):
            self.assertEqual(len({job.device_key.split(":")[0] for job in jobs if job.cell_id == cell_id}), 1)

    def test_simultaneous_band_uses_one_shared_resample_family(self):
        cell_ids = np.array(["a"] * 4 + ["b"] * 4)
        labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        differences = {"x": np.array([1, 0, 1, 0, 0, 0, 0, 0]),
                       "y": np.array([0, 0, 0, 0, 1, 0, 1, 0])}
        intervals, metadata = simultaneous_bootstrap(
            differences, cell_ids, labels, replicates=200, seed=7, level=0.95
        )
        self.assertEqual(set(intervals), {"x", "y"})
        self.assertEqual(metadata["method"], "shared_item_multiplicity_dataset_class_stratified_max_absolute_deviation")
        widths = {round(value["upper"] - value["point_difference"], 12) for value in intervals.values()}
        self.assertEqual(len(widths), 1)

    def test_n3c_p4a_uses_fixed_clean_pair_and_keeps_signed_margins(self):
        base = np.eye(2, dtype=np.float32)
        clean = np.array([[0.8, 0.2], [0.2, 0.8]], dtype=np.float32)
        features = np.array([
            [[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]],
            [[0.2, 0.8], [0.9, 0.1], [0.3, 0.7], [0.8, 0.2]],
        ], dtype=np.float32)
        report = evaluate_cell(features, clean, base, {"no_correction": base}, 3, {})
        self.assertEqual(report["P4a"]["pair_definition"],
                         "fixed raw-clean top-1 anchor versus fixed raw-clean runner-up")
        self.assertAlmostEqual(report["P6_winner_runner_up_margin_quantiles"]["0.5"], 0.6, places=6)

    def test_planner_uses_per_cell_draw_budget(self):
        cells = [
            {"cell_id": "exp", "model_id": "m", "dataset_id": "d", "sigma": .25, "item_count": 10},
            {"cell_id": "n3c", "model_id": "m", "dataset_id": "d", "sigma": .25, "item_count": 10},
        ]
        jobs = plan_shards(cells=cells, devices=[Device("A", 0, "other")], throughput_table={},
                           draws_per_item={"exp": 4224, "n3c": 512})
        encodes = {job.cell_id: job.encodes for job in jobs}
        self.assertEqual(encodes, {"exp": 42240, "n3c": 5120})

    def test_launch_paths_resolve_from_project_root(self):
        prepare = (ROOT / "launch" / "day1_prepare.sh").read_text(encoding="utf-8")
        self.assertIn("python satml2027/day1/d1_02_build_development.py", prepare)
        self.assertIn("configs/prompts/cifar100_openai_readme.json", prepare)
        self.assertIn("configs/prompts/eurosat_openai_ensemble_v1.json", prepare)
        planner = (ROOT / "day1" / "d1_04_plan.py").read_text(encoding="utf-8")
        self.assertIn('cd "$(dirname "$0")/../../../.."', planner)
        self.assertIn("python satml2027/day1/d1_05_run_shard.py", planner)

    def test_n3c_chowers_candidate_is_not_allclose_skipped_and_p2_is_emitted(self):
        base = np.eye(2, dtype=np.float32)
        candidate = base.copy()
        candidate[0, 1] = 1e-7
        candidate /= np.linalg.norm(candidate, axis=1, keepdims=True)
        features = np.array([[[.8, .2], [.7, .3], [.6, .4], [.9, .1]]], dtype=np.float32)
        metadata = {"chowers_test": {"image_transform": "identity", "operator": {
            "known": True, "row_scales": [1.0, 1.0], "row_norms": [1.0, 1.0],
            "shared_vector": [0.0, 0.0]}}}
        report = evaluate_cell(features, np.array([[.8, .2]], dtype=np.float32), base,
                               {"no_correction": base, "chowers_test": candidate}, 3, metadata,
                               ground_truth=np.array([0]))
        self.assertIn("chowers_test", report["P1"])
        self.assertIn("chowers_test", report["P2"])
        self.assertIn("chowers_test", report["P9"])

    def test_n3c_nonpositive_row_scale_fails_closed(self):
        base = np.eye(2, dtype=np.float32)
        features = np.array([[[.8, .2], [.7, .3], [.75, .25], [.65, .35]]], dtype=np.float32)
        metadata = {"bad": {"image_transform": "identity", "operator": {
            "known": True, "row_scales": [1.0, 0.0], "row_norms": [1.0, 1.0]}}}
        report = evaluate_cell(features, np.array([[.8, .2]], dtype=np.float32), base,
                               {"no_correction": base, "bad": base.copy()}, 2, metadata)
        self.assertFalse(report["P1"]["bad"]["applicable"])
        self.assertIn("a_k > 0", report["P1"]["bad"]["reason"])

    def test_exp019b_registration_names_all_eighteen_primary_cells(self):
        registration = json.loads((ROOT.parent / "configs" / "satml2027" / "exp-20260920-019b.json").read_text())
        self.assertIn("all 18 backbone-extension cells", registration["primary_outcome"])
        self.assertIsNone(registration["bridge_cells"])

    def test_bank_and_resume_metadata_are_exactly_bound(self):
        registration = {"experiment_id": "E", "registration_sha256": "r"}
        cell = {"cell_id": "c"}
        payload = valid_registered_bank_payload()
        banks = payload["banks"]
        self.assertEqual(validate_bank_payload(payload, registration, cell), banks)
        with self.assertRaises(RuntimeError):
            validate_bank_payload({"cell_id": "c", "banks": banks}, registration, cell)
        current = {key: value["metadata"] for key, value in banks.items()}
        stale = dict(current)
        stale["no_correction"] = {"output_prototype_sha256": "STALE"}
        with self.assertRaises(RuntimeError):
            validate_resume_metadata({"experiment_id": "E", "registration_sha256": "r",
                                      "cell": cell, "candidate_metadata": stale}, registration, cell, current)

    def test_bank_tensor_hash_candidate_set_and_gr_mean_fail_closed(self):
        registration = {"experiment_id": "E", "registration_sha256": "r"}
        cell = {"cell_id": "c"}
        tampered = valid_registered_bank_payload()
        tampered["banks"]["clean_boundary_active__step_0.0025"]["prototypes"][0, 0] = 0.5
        with self.assertRaisesRegex(RuntimeError, "prototype tensor hash mismatch"):
            validate_bank_payload(tampered, registration, cell)

        arbitrary = valid_registered_bank_payload()
        entry = arbitrary["banks"].pop("no_correction")
        entry["metadata"]["candidate_id"] = "b00"
        arbitrary["banks"]["b00"] = entry
        with self.assertRaisesRegex(RuntimeError, "candidate IDs differ"):
            validate_bank_payload(arbitrary, registration, cell)

        stale_mean = valid_registered_bank_payload()
        stale_mean["banks"]["gr_clip_style_two_sided__coefficient_1"]["image_mean"][0] = 0.75
        with self.assertRaisesRegex(RuntimeError, "image mean hash mismatch"):
            validate_bank_payload(stale_mean, registration, cell)

    def test_merge_rejects_incomplete_bank_family(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            banks_dir = root / "banks"
            banks_dir.mkdir()
            cell = {"cell_id": "c", "item_count": 1, "development_items_sha256": "unused"}
            registration = {"experiment_id": "E", "registration_sha256": "r"}
            torch.save({"registration_sha256": "r", "cell_id": "c", "banks": {"only": {"metadata": {}}}},
                       banks_dir / "c__banks.pt")
            with self.assertRaisesRegex(RuntimeError, "candidate IDs differ"):
                merge_cell(cell, [root], registration, banks_dir)

    def test_merge_rejects_tensor_tampering_even_when_metadata_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            banks_dir = root / "banks"
            banks_dir.mkdir()
            cell = {"cell_id": "c", "item_count": 1, "development_items_sha256": "unused"}
            registration = {"experiment_id": "E", "registration_sha256": "r"}
            payload = valid_registered_bank_payload()
            payload["banks"]["clean_boundary_active__step_0.0025"]["prototypes"][0, 0] = 0.25
            torch.save(payload, banks_dir / "c__banks.pt")
            with self.assertRaisesRegex(RuntimeError, "prototype tensor hash mismatch"):
                merge_cell(cell, [root], registration, banks_dir)

    def test_n3c_feature_identity_is_registration_bound(self):
        registration = {"experiment_id": "N", "registration_sha256": "r",
                        "sampling": {"stream_role": "diagnostic", "base_seed": 7}}
        cell = {"cell_id": "c", "sigma": .25}
        data = {"experiment_id": np.array(["N"]), "registration_sha256": np.array(["r"]),
                "cell_id": np.array(["c"]), "sigma": np.array([.25]),
                "stream_role": np.array(["diagnostic"]), "base_seed": np.array([7]),
                "shard_index": np.array([0]), "shard_count": np.array([2])}
        validate_feature_identity(data, registration, cell, shard_index=0, shard_count=2)
        data["registration_sha256"] = np.array(["stale"])
        with self.assertRaises(RuntimeError):
            validate_feature_identity(data, registration, cell, shard_index=0, shard_count=2)

    def test_n3c_feature_and_count_ground_truth_must_match(self):
        feature = {
            "done": np.array([True]),
            "item_ids": np.array(["item-0"]),
            "ground_truth": np.array([1], dtype=np.int64),
        }
        counts = {
            "done": np.array([True]),
            "item_ids": np.array(["item-0"]),
            "ground_truth": np.array([0], dtype=np.int64),
        }
        with self.assertRaisesRegex(RuntimeError, "ground-truth mismatch"):
            validate_count_feature_alignment(feature, counts, "fixture_features.npz")

    def test_p1_checks_actual_new_winner_not_any_eligible_competitor(self):
        base = np.eye(3, dtype=np.float32)
        candidate = base.copy()
        candidate[2] = np.array([1.0, .2, 0.0], dtype=np.float32)
        candidate[2] /= np.linalg.norm(candidate[2])
        features = np.array([[[.8, .7, 0.0], [.8, .7, 0.0], [.8, .7, 0.0], [.8, .7, 0.0]]], dtype=np.float32)
        metadata = {"candidate": {"image_transform": "identity", "operator": {
            "known": True, "row_scales": [1.0, 1.0, 1.0], "row_norms": [1.0, 2.0, 1.0],
            "shared_vector": [0.0, 0.0, 0.0]}}}
        report = evaluate_cell(features, np.array([[.8, .7, 0.0]], dtype=np.float32), base,
                               {"no_correction": base, "candidate": candidate}, 2, metadata)
        self.assertEqual(report["P1"]["candidate"]["flips"], 4)
        self.assertEqual(report["P1"]["candidate"]["flips_outside_eligible_set"], 4)

    def test_p2_uses_construction_vector_and_p9_covers_nonapplicable_control(self):
        base = np.eye(2, dtype=np.float32)
        vector = np.array([0.0, .1], dtype=np.float32)
        raw = base + vector
        candidate = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.linalg.norm(raw, axis=1)
        features = np.array([[[.8, .2], [.2, .8], [.7, .3], [.3, .7]]], dtype=np.float32)
        other = candidate[::-1].copy()
        metadata = {
            "chowers_test": {"image_transform": "identity", "operator": {
                "known": True, "row_scales": [1.0, 1.0], "row_norms": norms.tolist(),
                "shared_vector": vector.tolist()}},
            "lowrank": {"image_transform": "identity", "operator": {"known": False}},
        }
        report = evaluate_cell(features, np.array([[.8, .2]], dtype=np.float32), base,
                               {"no_correction": base, "chowers_test": candidate, "lowrank": other}, 3,
                               metadata, ground_truth=np.array([0]))
        self.assertLess(report["P2"]["chowers_test"]["recorded_vs_reconstructed_shared_vector_max"], 1e-6)
        self.assertIn("lowrank", report["P9"])
        self.assertFalse(report["P1"]["lowrank"]["applicable"])


if __name__ == "__main__":
    unittest.main()
