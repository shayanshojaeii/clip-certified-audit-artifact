"""Tests for the EXP-021 multi-GPU planner (numpy-free; runs in either environment)."""

from __future__ import annotations

import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from satml2027ext import candidates_ext, guard_ext, plan_ext  # noqa: E402
from satml2027ext._common import canonical_json_hash, read_json, write_json_atomic  # noqa: E402

L14 = "openai-clip-vit-l14-quickgelu"
B32 = "openai-clip-vit-b32-quickgelu"
LAION = "openclip-vit-b32-laion2b"


def synthetic_cells(*, l14: int = 6, b32: int = 12, items: int = 1000, draws: int = 4224) -> list[dict]:
    cells = []
    for index in range(l14):
        cells.append({"cell_id": f"{L14}__ds{index}__sigma0.25", "model_id": L14, "dataset_id": f"ds{index}", "sigma": 0.25,
                      "item_count": items, "class_count": 10, "experiment_id": "EXP-TEST-021B", "registration_path": "reg.json",
                      "registration_sha256": "0" * 64, "draws_per_item": draws, "store_margins": True})
    for index in range(b32):
        model = B32 if index % 2 else LAION
        cells.append({"cell_id": f"{model}__ds{index}__sigma0.12", "model_id": model, "dataset_id": f"ds{index}", "sigma": 0.12,
                      "item_count": items, "class_count": 10, "experiment_id": "EXP-TEST-021B", "registration_path": "reg.json",
                      "registration_sha256": "0" * 64, "draws_per_item": draws, "store_margins": True})
    return cells


def synthetic_registration(cells: list[dict], experiment_id: str = "EXP-20260921-021B", candidates=None) -> dict:
    body = {
        "schema_version": guard_ext.REGISTRATION_SCHEMA,
        "experiment_id": experiment_id,
        "status": "preregistered_pending_independent_approval",
        "candidates": candidates or candidates_ext.canonical_candidates_block(),
        "sharding": {"max_items_per_shard": 1000},
        "bank_construction": {"source_mode": "encode_registered_roles",
                              "direction_construction": {"development_draws_per_item": 64},
                              "control_training": {"train_draws_per_item": 16}},
        "certification": {
            "selection_draws": 128, "confirmation_draws": 4096, "alpha_per_example": 0.001, "reported_radii": [0.0, 0.25],
            "seeds": {"selection_base_seed": 1, "confirmation_base_seed": 2},
            "streams": {"selection": "cohen_class_selection", "confirmation": "cohen_class_confirmation"},
            "noise_block_draws": 64, "margin_storage": {"bank": "no_correction", "stream": "confirmation"},
        },
        "cells": [{**{key: cell[key] for key in ("cell_id", "model_id", "dataset_id", "sigma", "item_count", "class_count")},
                   "development_item_count": 300, "control_train_item_count": 200, "store_identity_margins": True} for cell in cells],
    }
    body["registration_sha256"] = canonical_json_hash(body)
    return body


class DeviceAndThroughputTests(unittest.TestCase):
    def test_parse_devices_both_syntaxes(self):
        single = plan_ext.parse_devices(["5090:6"])
        self.assertEqual([device.key for device in single], [f"host:{index}" for index in range(6)])
        self.assertTrue(all(device.gpu_class == "5090" for device in single))
        named = plan_ext.parse_devices(["A:1:4090", "B:5:5090"])
        self.assertEqual(len(named), 6)
        self.assertEqual(named[0], plan_ext.Device("A", 0, "4090"))
        self.assertEqual(named[-1], plan_ext.Device("B", 4, "5090"))
        for bad in (["5090"], ["5090:0"], ["A:x:4090"], ["A:1:4090:extra"]):
            with self.assertRaises(ValueError):
                plan_ext.parse_devices(bad)

    def test_throughput_table_has_explicit_5090_row(self):
        table = plan_ext.default_throughput_table()
        self.assertIn("5090", table)
        self.assertAlmostEqual(table["4090"][L14], 4224 / 24.2, places=3)
        self.assertAlmostEqual(table["5090"][L14] / table["4090"][L14], 1.4, places=3)
        custom = plan_ext.default_throughput_table(2.0)
        self.assertAlmostEqual(custom["5090"][B32] / custom["4090"][B32], 2.0, places=3)
        merged = plan_ext.merge_throughput(table, {"5090": {L14: 999.0}})
        self.assertEqual(merged["5090"][L14], 999.0)
        self.assertEqual(merged["4090"][L14], table["4090"][L14])
        with self.assertRaises(KeyError):
            plan_ext.rate(table, "5090", "unknown-model")


class AssignmentTests(unittest.TestCase):
    def setUp(self):
        self.table = plan_ext.default_throughput_table()
        self.cells = synthetic_cells()

    def test_whole_cells_balanced_over_six_5090s(self):
        devices = plan_ext.parse_devices(["5090:6"])
        jobs = plan_ext.assign_jobs(plan_ext.expand_jobs(self.cells), devices, self.table)
        self.assertEqual(len(jobs), len(self.cells))
        self.assertEqual(sorted(job.cell_id for job in jobs), sorted(cell["cell_id"] for cell in self.cells))
        self.assertTrue(all(job.shard_count == 1 and job.shard_index == 0 for job in jobs))
        self.assertTrue(all(job.device_key in {device.key for device in devices} for job in jobs))
        for job in jobs:
            self.assertAlmostEqual(job.estimated_seconds, job.encodes / self.table["5090"][job.model_id])
        summary = plan_ext.summarize(jobs, devices)
        self.assertLessEqual(summary["max_min_load_ratio"], plan_ext.BALANCE_TARGET_RATIO)
        self.assertTrue(summary["balanced_within_target"])
        self.assertAlmostEqual(summary["wall_clock_seconds"], max(summary["per_device_seconds"].values()), places=0)
        # each 5090 carries exactly one ViT-L/14 cell (LPT places the six longest first)
        per_device_l14 = {}
        for job in jobs:
            if job.model_id == L14:
                per_device_l14[job.device_key] = per_device_l14.get(job.device_key, 0) + 1
        self.assertEqual(sorted(per_device_l14.values()), [1] * 6)

    def test_l14_prefers_5090_on_mixed_hosts(self):
        devices = plan_ext.parse_devices(["A:1:4090", "B:5:5090"])
        jobs = plan_ext.assign_jobs(plan_ext.expand_jobs(self.cells), devices, self.table)
        for job in jobs:
            if job.model_id == L14:
                self.assertEqual(job.gpu_class, "5090", job.cell_id)
        self.assertTrue(any(job.device_key == "A:0" for job in jobs), "the 4090 receives non-L/14 work")

    def test_split_cells_keep_one_gpu_class_and_resume_safe_indices(self):
        devices = plan_ext.parse_devices(["A:1:4090", "B:5:5090"])
        jobs = plan_ext.assign_jobs(plan_ext.expand_jobs(self.cells, max_shard_items=400), devices, self.table)
        by_cell: dict[str, list] = {}
        for job in jobs:
            by_cell.setdefault(job.cell_id, []).append(job)
        for cell_id, cell_jobs in by_cell.items():
            self.assertEqual(len({job.gpu_class for job in cell_jobs}), 1, cell_id)
            self.assertEqual(sorted(job.shard_index for job in cell_jobs), list(range(3)))
            self.assertTrue(all(job.shard_count == 3 for job in cell_jobs))
            self.assertEqual(sum(job.item_count for job in cell_jobs), 1000)
        self.assertEqual(len(jobs), 3 * len(self.cells))

    def test_estimates_scale_with_throughput_and_draws(self):
        devices = plan_ext.parse_devices(["5090:6"])
        base = plan_ext.summarize(plan_ext.assign_jobs(plan_ext.expand_jobs(self.cells), devices, self.table), devices)
        doubled = {gpu_class: {model: 2.0 * value for model, value in models.items()} for gpu_class, models in self.table.items()}
        fast = plan_ext.summarize(plan_ext.assign_jobs(plan_ext.expand_jobs(self.cells), devices, doubled), devices)
        self.assertAlmostEqual(fast["total_gpu_hours"], base["total_gpu_hours"] / 2, places=2)
        self.assertAlmostEqual(fast["wall_clock_seconds"], base["wall_clock_seconds"] / 2, delta=1.0)
        more_draws = synthetic_cells(draws=2 * 4224)
        heavy = plan_ext.summarize(plan_ext.assign_jobs(plan_ext.expand_jobs(more_draws), devices, self.table), devices)
        self.assertAlmostEqual(heavy["total_gpu_hours"], 2 * base["total_gpu_hours"], places=2)
        self.assertGreater(heavy["wall_clock_seconds"], base["wall_clock_seconds"])
        faster_5090 = plan_ext.default_throughput_table(2.0)
        quick = plan_ext.summarize(plan_ext.assign_jobs(plan_ext.expand_jobs(self.cells), devices, faster_5090), devices)
        self.assertLess(quick["wall_clock_seconds"], base["wall_clock_seconds"])
        one_gpu = plan_ext.parse_devices(["5090:1"])
        serial = plan_ext.summarize(plan_ext.assign_jobs(plan_ext.expand_jobs(self.cells), one_gpu, self.table), one_gpu)
        self.assertGreaterEqual(serial["wall_clock_seconds"], base["wall_clock_seconds"])
        self.assertAlmostEqual(serial["total_gpu_hours"], base["total_gpu_hours"], places=2)


class ScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp())
        self.cells = synthetic_cells(l14=2, b32=4)
        self.registration = synthetic_registration(self.cells)
        self.registration_path = self.temp / "exp-test-021b.json"
        write_json_atomic(self.registration_path, self.registration)

    def tearDown(self):
        shutil.rmtree(self.temp, ignore_errors=True)

    def test_main_writes_plan_scripts_and_hash(self):
        out = self.temp / "plan"
        code = plan_ext.main(["--registrations", str(self.registration_path), "--devices", "5090:6", "--out", str(out),
                              "--results-root", "results/satml2027ext"])
        self.assertEqual(code, 0)
        plan = read_json(out / "plan.json")
        body = {key: value for key, value in plan.items() if key != "plan_sha256"}
        self.assertEqual(plan["plan_sha256"], canonical_json_hash(body))
        self.assertEqual((out / "plan.sha256").read_text(encoding="utf-8").split()[0], plan["plan_sha256"])
        self.assertEqual(len(plan["devices"]), 6)
        self.assertTrue(all(device["device_model"].startswith("NVIDIA GeForce RTX 5090") for device in plan["devices"]))
        self.assertEqual(len(plan["per_gpu"]), 6)
        self.assertEqual(sum(len(entry["jobs"]) for entry in plan["per_gpu"].values()), len(self.cells))
        self.assertTrue(all("estimated_seconds" in job for job in plan["jobs"]))
        self.assertEqual(plan["summary"]["cell_count"], len(self.cells))
        launch = out / "launch"
        scripts = sorted(path.name for path in launch.glob("run_gpu*.sh"))
        self.assertEqual(scripts, [f"run_gpu{index}.sh" for index in range(6)])
        self.assertTrue((launch / "launch_all.sh").is_file())
        self.assertTrue((launch / "status.sh").is_file())
        text = "\n".join((launch / name).read_text(encoding="utf-8") for name in scripts)
        for cell in self.cells:
            self.assertEqual(len(re.findall(rf"--cell-id {re.escape(cell['cell_id'])} ", text)), 1, cell["cell_id"])
        self.assertEqual(text.count("satml2027ext/run_shard_ext.py"), len(self.cells))
        self.assertNotIn("--store-margins", text)  # the worker reads the registered per-cell flag
        self.assertEqual(text.count("--bank-manifest results/satml2027ext/EXP021_BANK_MANIFEST.json"), len(self.cells))
        self.assertEqual(text.count("--data-preflight results/satml2027ext/EXP021_DATA_PREFLIGHT.json"), len(self.cells))
        for index in range(6):
            script = (launch / f"run_gpu{index}.sh").read_text(encoding="utf-8")
            self.assertIn("set -euo pipefail", script)
            self.assertIn("export PYTHONUNBUFFERED=1", script)
            self.assertIn(f"export CUDA_VISIBLE_DEVICES={index}", script)
            self.assertIn("export NVIDIA_TF32_OVERRIDE=0", script)
            self.assertIn("export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0", script)
            self.assertIn("export HF_HUB_OFFLINE=1", script)
            self.assertIn(plan["plan_sha256"], script)
        launch_all = (launch / "launch_all.sh").read_text(encoding="utf-8")
        self.assertIn("setsid nohup bash", launch_all)
        self.assertIn("pids/gpu$gpu.pid", launch_all)
        self.assertIn("logs/gpu$gpu.log", launch_all)
        status = (launch / "status.sh").read_text(encoding="utf-8")
        self.assertIn("items_done", status)
        self.assertIn(".meta.json", status)
        self.assertNotIn(".npz", status.replace(".meta.json", ""))
        # never overwrite an existing plan
        with self.assertRaises(FileExistsError):
            plan_ext.main(["--registrations", str(self.registration_path), "--devices", "5090:6", "--out", str(out)])

    def test_shards_may_not_exceed_the_registered_size_and_split_deterministically(self):
        with self.assertRaises(ValueError):
            plan_ext.main(["--registrations", str(self.registration_path), "--devices", "5090:6", "--out", str(self.temp / "p1"),
                           "--max-shard-items", "1500"])
        code = plan_ext.main(["--registrations", str(self.registration_path), "--devices", "5090:6", "--out", str(self.temp / "p2"),
                              "--max-shard-items", "400"])
        self.assertEqual(code, 0)
        plan = read_json(self.temp / "p2" / "plan.json")
        self.assertTrue(all(job["shard_count"] == 3 and job["item_count"] <= 400 for job in plan["jobs"]))
        self.assertEqual(plan["note"], "planning reads registrations only; it touches no data and no outcome and is not gated by the approval")

    def test_subset_registration_needs_and_names_its_parent(self):
        subset = candidates_ext.subset_candidates_block(candidates_ext.BUDGET_SUBSET_021C, parent_experiment_id=self.registration["experiment_id"],
                                                        parent_registration_sha256=self.registration["registration_sha256"])
        child_cells = [{**self.cells[0], "cell_id": self.cells[0]["cell_id"] + "__subset100", "item_count": 100}]
        child = synthetic_registration(child_cells, experiment_id="EXP-20260921-021C", candidates=subset)
        child_path = self.temp / "exp-test-021c.json"
        write_json_atomic(child_path, child)
        with self.assertRaises(ValueError):
            plan_ext.main(["--registrations", str(child_path), "--devices", "5090:1", "--out", str(self.temp / "p3")])
        plan_ext.main(["--registrations", str(self.registration_path), str(child_path), "--devices", "5090:2",
                       "--out", str(self.temp / "p4")])
        text = "\n".join(path.read_text(encoding="utf-8") for path in (self.temp / "p4" / "launch").glob("run_gpu*.sh"))
        self.assertEqual(text.count(f"--parent-registration {self.registration_path.as_posix()}"), 1)

    def test_preparation_stage_scripts(self):
        out = self.temp / "prep"
        code = plan_ext.main(["--registrations", str(self.registration_path), "--devices", "5090:6", "--out", str(out),
                              "--stage", "preparation"])
        self.assertEqual(code, 0)
        scripts = sorted((out / "launch_preparation").glob("prepare_gpu*.sh"))
        self.assertEqual(len(scripts), 6)
        text = "\n".join(path.read_text(encoding="utf-8") for path in scripts)
        self.assertEqual(text.count("satml2027ext/prepare_banks_ext.py"), len(self.cells))
        self.assertIn("export HF_HUB_OFFLINE=1", text)
        self.assertIn("bank_manifest_ext.py", (out / "launch_preparation" / "after_all_gpus.txt").read_text(encoding="utf-8"))

    def test_named_hosts_scripts(self):
        out = self.temp / "plan_named"
        plan_ext.main(["--registrations", str(self.registration_path), "--devices", "A:1:4090", "B:2:5090", "--out", str(out)])
        names = sorted(path.name for path in (out / "launch").glob("run_*.sh"))
        self.assertEqual(names, ["run_A_gpu0.sh", "run_B_gpu0.sh", "run_B_gpu1.sh"])
        self.assertFalse((out / "launch" / "launch_all.sh").exists())


class V3PlannerTests(unittest.TestCase):
    """Internal re-review of V2 (2026-09-26): cache-dir pass-through, cuBLAS determinism, registered forward batch."""

    def setUp(self):
        self.temp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp, ignore_errors=True)

    def test_v3_scripts_pass_the_cache_and_export_cublas_determinism(self):
        registrations = [str(PROJECT / "configs/satml2027ext/exp-20260921-021a.json")]
        code = plan_ext.main(["--registrations", *registrations, "--devices", "5090:2", "--out", str(self.temp / "p"),
                              "--cache-dir", "/srv/hf"])
        self.assertEqual(code, 0)
        script = (self.temp / "p" / "launch" / "run_gpu0.sh").read_text(encoding="utf-8")
        self.assertIn("export CUBLAS_WORKSPACE_CONFIG=:4096:8", script)
        self.assertIn("--cache-dir /srv/hf", script)
        self.assertIn("--blocks-per-forward 4", script)
        with self.assertRaises(SystemExit):
            plan_ext.main(["--registrations", *registrations, "--devices", "5090:2", "--out", str(self.temp / "q"),
                           "--blocks-per-forward", "8"])

    def test_v3_docstring_describes_registered_shards(self):
        self.assertNotIn("whole cells per GPU", plan_ext.__doc__.splitlines()[0])


if __name__ == "__main__":
    unittest.main()
