"""Focused tests for the outcome-blind analysis addendum V1 (d1_09 and d1_10).

Everything runs on tiny synthetic inputs inside a temporary directory.  No
EXP-019/N3C result is read or looked for; the only project files touched are
the registrations, the registered item lists, and the new scripts.  Torch is
optional: the `.pt` round trip is exercised only when torch is importable.
"""

from __future__ import annotations

import contextlib
import copy
import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))

try:  # torch is optional on analysis-only machines
    import torch

    TORCH_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - depends on the host
    torch = None
    TORCH_AVAILABLE = False

from common.hashing import canonical_json_hash, read_json, sha256_file, tensor_scientific_hash, write_json_atomic  # noqa: E402
from day1 import d1_07_analyze as d1_07  # noqa: E402
from day1 import d1_09_sensitivity_analysis as d1_09  # noqa: E402
from day1 import d1_10_tf32_preparation_adjudication as d1_10  # noqa: E402

TEST_REPLICATES = 1000
STEPS = [0.0025, 0.005, 0.01, 0.02, 0.04]
COEFFICIENTS = [0.25, 0.5, 1.0]
CONTROL_IDS = [
    "learned_shared_translation_tangent",
    "learned_shared_translation_pure",
    "noisy_class_mean",
    "lowrank_tangent_r8",
]


# --------------------------------------------------------------------------- synthetic inputs


def synthetic_registration(experiment_id: str = "EXP-TEST-019A") -> dict:
    """Two models x two datasets (heterogeneous strata) plus two EXP-019A-style bridge cells."""

    datasets = {"dsa": (3, 2), "cifar10": (2, 3)}  # dataset -> (classes, items per class)
    cells = []
    for model in ("m1", "m2"):
        for dataset, (classes, per_class) in datasets.items():
            cells.append(_cell(model, dataset, 0.5, classes, classes * per_class))
        cells.append(_cell(model, "cifar10", 0.25, 2, 6))  # bridge cell: excluded from the primary
    registration = {
        "experiment_id": experiment_id,
        "cells": cells,
        "certification": {
            "alpha_per_example": 0.001,
            "reported_radii": [0.0, 0.12, 0.25, 0.5],
            "selection_draws": 128,
            "confirmation_draws": 4096,
            "primary_radius_rule": "r = sigma",
        },
        "prospective_practical_effect_region": {"absolute_ca": 0.005, "interpretation": "reporting region only"},
        "candidates": {
            "frozen_grid": {
                "bank_count": 18,
                "common_unit_direction_step_grid": STEPS,
                "gap_coefficient_grid": COEFFICIENTS,
            },
            "controls": [{"control_id": control_id} for control_id in CONTROL_IDS],
        },
        "final_test_access": False,
    }
    registration["registration_sha256"] = canonical_json_hash(registration)
    return registration


def _cell(model: str, dataset: str, sigma: float, classes: int, items: int) -> dict:
    return {
        "cell_id": f"{model}__{dataset}__sigma{sigma:.12g}",
        "model_id": model,
        "dataset_id": dataset,
        "sigma": sigma,
        "class_count": classes,
        "item_count": items,
    }


def write_synthetic_merged(directory: Path, registration: dict, rng: np.random.Generator, *, mode: str = "random") -> None:
    """Merged `.npz` files with the exact keys d1_06 writes and outcomes controlled through the counts.

    mode="random": reference certified-correct with probability 0.2, candidates flip 25% of
    items, the X3 control equals the reference, the rank-8 control is certified-correct
    everywhere.  mode="stratum_constant": every difference is constant within each
    (dataset, class) stratum of each cell, so every stratified band must be exactly zero.
    """

    family = d1_09.registered_candidate_family(registration)
    banks = family["all_banks"]
    items_by_dataset: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cell_index, cell in enumerate(registration["cells"]):
        dataset, classes, count = cell["dataset_id"], int(cell["class_count"]), int(cell["item_count"])
        if dataset not in items_by_dataset:
            labels = np.repeat(np.arange(classes), count // classes)
            item_ids = np.array([f"{dataset}-{index:03d}" for index in range(count)])
            items_by_dataset[dataset] = (item_ids, labels)
        item_ids, labels = items_by_dataset[dataset]
        if mode == "random":
            reference = rng.random(count) < 0.2
        else:
            reference = np.zeros(count, dtype=bool)
        outcomes = np.zeros((len(banks), count), dtype=bool)
        for position, bank in enumerate(banks):
            if bank in ("no_correction", d1_09.X3_CONTROL_ID):
                outcomes[position] = reference
            elif bank == "control__lowrank_tangent_r8":
                outcomes[position] = True
            elif mode == "random":
                outcomes[position] = reference ^ (rng.random(count) < 0.25)
            else:
                per_class = np.array([(position * 7 + cell_index * 3 + klass) % 2 == 0 for klass in range(classes)])
                outcomes[position] = per_class[labels]
        selection = np.zeros((len(banks), count, classes), dtype=np.int32)
        confirmation = np.zeros((len(banks), count, classes), dtype=np.int32)
        for position in range(len(banks)):
            for item in range(count):
                truth = int(labels[item])
                if outcomes[position, item]:
                    selected, successes = truth, 4096
                elif item % 2 == 0:
                    selected, successes = (truth + 1) % classes, 4096  # certified but wrong
                else:
                    selected, successes = truth, 2048  # abstains
                selection[position, item, selected] = 128
                confirmation[position, item, selected] = successes
                if successes < 4096:
                    confirmation[position, item, (selected + 1) % classes] = 4096 - successes
        np.savez_compressed(
            directory / f"{cell['cell_id']}__merged.npz",
            candidate_ids=np.array(banks),
            item_ids=item_ids,
            ground_truth=labels.astype(np.int64),
            raw_predictions=np.tile(labels.astype(np.int64), (len(banks), 1)),
            selection_counts=selection,
            confirmation_counts=confirmation,
            selection_seeds=np.arange(1, count + 1, dtype=np.int64),
            confirmation_seeds=np.arange(1001, 1001 + count, dtype=np.int64),
        )


def run_primary(merged: Path, registration_path: Path, out: Path) -> int:
    """Run the registered d1_07 in-process with a reduced replicate count (its constants are patched)."""

    original = d1_07.simultaneous_bootstrap

    def reduced(*args, **kwargs):
        kwargs.setdefault("replicates", TEST_REPLICATES)
        return original(*args, **kwargs)

    argv = ["d1_07", "--merged", str(merged), "--registration", str(registration_path), "--out", str(out)]
    with mock.patch.object(d1_07, "simultaneous_bootstrap", reduced), \
            mock.patch.object(d1_07, "BOOTSTRAP_REPLICATES", TEST_REPLICATES), \
            mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
        return d1_07.main()


def run_sensitivity(merged: Path, registration_path: Path, primary: Path, root: Path) -> int:
    argv = ["--merged", str(merged), "--registration", str(registration_path), "--primary", str(primary),
            "--root", str(root)]
    with mock.patch.object(d1_07, "BOOTSTRAP_REPLICATES", TEST_REPLICATES), contextlib.redirect_stdout(io.StringIO()):
        return d1_09.main(argv)


def write_approval(root: Path, *, overrides: dict | None = None) -> Path:
    research = root / "research"
    research.mkdir(parents=True, exist_ok=True)
    addendum = research / d1_09.ADDENDUM_RELATIVE_PATH.name
    if not addendum.is_file():
        addendum.write_text("synthetic addendum text for the guard test\n", encoding="utf-8")
    amendment = research / d1_09.RETROSPECTIVE_AMENDMENT_RELATIVE_PATH.name
    if not amendment.is_file():
        amendment.write_text("synthetic retrospective amendment for the guard test\n", encoding="utf-8")
    approval = {
        "status": d1_09.REQUIRED_APPROVAL_STATUS,
        "addendum_sha256": sha256_file(addendum),
        "retrospective_amendment_sha256": sha256_file(amendment),
        "sensitivity_source_sha256": sha256_file(Path(d1_09.__file__).resolve()),
        "tf32_adjudication_source_sha256": sha256_file(Path(d1_10.__file__).resolve()),
        "primary_analysis_source_sha256": sha256_file(Path(d1_07.__file__).resolve()),
        "new_sampling_authorized": False,
        "equivalence_claim_authorized": False,
        "confirmatory_claim_authorized": False,
        "causal_claim_authorized": False,
        "final_test_authorized": False,
        "outcome_blind_at_approval": False,
        "outcomes_accessed_before_review": True,
        "unblinding_decision": d1_09.UNBLINDING_DECISION,
        "unblinding_date": d1_09.UNBLINDING_DATE,
        "review_date": "2026-09-27",
        "reviewer": "synthetic test reviewer",
        "reviewer_conclusion": "synthetic approval for the guard test only",
    }
    approval.update(overrides or {})
    path = research / d1_09.APPROVAL_RELATIVE_PATH.name
    write_json_atomic(path, approval)
    return path


# --------------------------------------------------------------------------- synthetic preparation trees


def _unit(value: np.ndarray) -> np.ndarray:
    return value / np.linalg.norm(value, axis=-1, keepdims=True)


def synthetic_preparation(cell: dict, registration: dict, rng: np.random.Generator, *, dim: int = 4,
                          basis_prototypes: bool = False) -> dict:
    classes = int(cell["class_count"])
    dev_items, draws, train_items, train_draws, rank = 6, 2, 4, 2, 2
    if basis_prototypes:
        prototypes = np.eye(classes, dim, dtype=np.float32)
    else:
        prototypes = _unit(rng.normal(size=(classes, dim))).astype(np.float32)
    clean = _unit(rng.normal(size=(dev_items, dim))).astype(np.float32)
    image_mean = clean.mean(axis=0).astype(np.float32)
    family = d1_09.registered_candidate_family(registration)
    banks = {}
    for candidate_id in family["all_banks"]:
        if candidate_id == "no_correction":
            bank = prototypes.copy()
        else:
            bank = _unit(prototypes + 0.05 * rng.normal(size=(classes, dim))).astype(np.float32)
        is_gr = candidate_id == d1_10.GR_BANK_ID
        is_control = candidate_id.startswith("control__")
        banks[candidate_id] = {
            "prototypes": bank,
            "metadata": {
                "candidate_id": candidate_id,
                "family": "registered_control" if is_control else "frozen_exp017_grid",
                "objective_id": None,
                "step_or_coefficient": None if is_control or candidate_id == "no_correction" else 0.5,
                "image_transform": d1_10.GR_IMAGE_TRANSFORM if is_gr else "identity",
                "source_prototype_sha256": tensor_scientific_hash(prototypes),
                "output_prototype_sha256": tensor_scientific_hash(bank),
                "calibration_image_mean_sha256": tensor_scientific_hash(image_mean) if is_gr else None,
                "role": "registered_control" if is_control else "frozen_exp017_grid",
                "operator": {"known": True, "shared_vector": [float(v) for v in (bank - prototypes).mean(axis=0)],
                             "row_norms": [float(v) for v in np.linalg.norm(bank, axis=1)]},
            },
            "image_mean": image_mean if is_gr else None,
        }
    return {
        "development": {
            "cell_id": cell["cell_id"], "model_id": cell["model_id"], "dataset_id": cell["dataset_id"],
            "sigma": float(cell["sigma"]), "prototypes": prototypes, "clean_features": clean,
            "noisy_features": _unit(rng.normal(size=(dev_items, draws, dim))).astype(np.float32),
            "clean_direction": _unit(rng.normal(size=dim)).astype(np.float32),
            "noisy_direction": _unit(rng.normal(size=dim)).astype(np.float32),
            "item_ids": [f"{cell['dataset_id']}-dev-{index}" for index in range(dev_items)],
            "direction_builder": "project", "noisy_temperature": 0.05, "development_draws": draws,
            "prompt_config": {"templates": ["a photo of a {}."]}, "final_test_access": False,
            "registration_sha256": registration["registration_sha256"],
        },
        "controls": {
            "cell_id": cell["cell_id"], "shared_delta": (0.01 * rng.normal(size=dim)).astype(np.float32),
            "lowrank_left": (0.01 * rng.normal(size=(classes, rank))).astype(np.float32),
            "lowrank_right": (0.01 * rng.normal(size=(rank, dim))).astype(np.float32), "rank": rank,
            "registration_sha256": registration["registration_sha256"],
            "training": {"epochs": 80, "batch_size": 1024, "learning_rate": 0.005, "weight_decay": 0.0001,
                         "temperature": 0.05, "clean_weight": 0.5, "train_draws": train_draws,
                         "noise_base_seed": 20260920005, "optimizer_seed": 20260920006,
                         "minibatch_seed": 20260920007},
            "final_test_access": False,
        },
        "control_train": {
            "item_ids": [f"{cell['dataset_id']}-train-{index}" for index in range(train_items)],
            "clean_features": _unit(rng.normal(size=(train_items, dim))).astype(np.float32),
            "noisy_features": _unit(rng.normal(size=(train_items, train_draws, dim))).astype(np.float32),
            "labels": (np.arange(train_items) % classes).astype(np.int64), "sigma": float(cell["sigma"]),
            "final_test_access": False,
        },
        "banks": {
            "cell_id": cell["cell_id"], "banks": banks, "registration_sha256": registration["registration_sha256"],
            "frozen_steps": STEPS, "frozen_coefficients": COEFFICIENTS, "final_test_access": False,
        },
    }


def to_cell_preparation(prep: dict) -> d1_10.CellPreparation:
    """Build the loader's in-memory form from a synthetic preparation (mirrors `_load_cell_pt`)."""

    development, controls, cache, bank_payload = prep["development"], prep["controls"], prep["control_train"], prep["banks"]
    prepared = d1_10.CellPreparation(cell_id=development["cell_id"], source_format="npz")
    for name in d1_10.DEVELOPMENT_ARRAYS:
        prepared.arrays[f"development.{name}"] = development[name]
    prepared.metadata["development"] = {k: v for k, v in development.items() if k not in d1_10.DEVELOPMENT_ARRAYS}
    for name in d1_10.CONTROL_ARRAYS:
        prepared.arrays[f"controls.{name}"] = controls[name]
    prepared.metadata["controls"] = {k: v for k, v in controls.items() if k not in d1_10.CONTROL_ARRAYS}
    for name in d1_10.CONTROL_TRAIN_ARRAYS:
        prepared.arrays[f"control_train.{name}"] = cache[name]
    prepared.metadata["control_train"] = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cache.items() if k not in d1_10.CONTROL_TRAIN_ARRAYS
    }
    banks = bank_payload["banks"]
    for candidate_id in sorted(banks):
        prepared.arrays[f"banks.{candidate_id}"] = banks[candidate_id]["prototypes"]
        if banks[candidate_id]["image_mean"] is not None:
            prepared.arrays["banks.gr_clip_image_mean"] = banks[candidate_id]["image_mean"]
    prepared.metadata["banks"] = {
        **{k: v for k, v in bank_payload.items() if k != "banks"},
        "candidate_ids": sorted(banks),
        "metadata": {candidate_id: copy.deepcopy(banks[candidate_id]["metadata"]) for candidate_id in sorted(banks)},
    }
    prepared.sidecars = {
        "development": {
            "cell_id": development["cell_id"],
            "prototype_sha256": tensor_scientific_hash(development["prototypes"]),
            "clean_direction_sha256": tensor_scientific_hash(development["clean_direction"]),
            "noisy_direction_sha256": tensor_scientific_hash(development["noisy_direction"]),
            "clean_features_sha256": tensor_scientific_hash(development["clean_features"]),
            "clean_zero_shot_accuracy": 0.5, "final_test_access": False,
        },
        "controls": {
            "cell_id": controls["cell_id"], "shared_delta_sha256": tensor_scientific_hash(controls["shared_delta"]),
            "shared_train_accuracy": {"noisy_accuracy": 0.5, "clean_accuracy": 0.5},
            "lowrank_train_accuracy": {"noisy_accuracy": 0.6, "clean_accuracy": 0.6}, "final_test_access": False,
        },
        "banks": {"cell_id": bank_payload["cell_id"], "bank_count": len(banks), "final_test_access": False},
    }
    return prepared


def write_mirror_tree(root: Path, preps: dict[str, dict]) -> None:
    for prep in preps.values():
        d1_10.export_cell_npz(to_cell_preparation(prep), root)


def write_pt_tree(root: Path, preps: dict[str, dict]) -> None:
    """The real server layout, written with torch.save exactly like d1_02, d1_03b and d1_03."""

    for cell_id, prep in preps.items():
        development = {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in prep["development"].items()}
        controls = {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in prep["controls"].items()}
        cache = {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in prep["control_train"].items()}
        banks = {}
        for candidate_id, entry in prep["banks"]["banks"].items():
            bank_entry = {"prototypes": torch.from_numpy(entry["prototypes"]), "metadata": copy.deepcopy(entry["metadata"])}
            if entry["image_mean"] is not None:
                bank_entry["image_mean"] = torch.from_numpy(entry["image_mean"])
            banks[candidate_id] = bank_entry
        payload = {**{k: v for k, v in prep["banks"].items() if k != "banks"}, "banks": banks}
        (root / "development").mkdir(parents=True, exist_ok=True)
        (root / "controls").mkdir(parents=True, exist_ok=True)
        (root / "banks").mkdir(parents=True, exist_ok=True)
        torch.save(development, root / "development" / f"{cell_id}__development.pt")
        torch.save(controls, root / "controls" / f"{cell_id}__controls.pt")
        torch.save(cache, root / "controls" / f"{cell_id}__control_train_features.pt")
        torch.save(payload, root / "banks" / f"{cell_id}__banks.pt")
        sidecars = to_cell_preparation(prep).sidecars
        write_json_atomic(root / "development" / f"{cell_id}__development.json", sidecars["development"])
        write_json_atomic(root / "controls" / f"{cell_id}__controls.json", sidecars["controls"])
        write_json_atomic(root / "banks" / f"{cell_id}__banks.json", sidecars["banks"])


def perturb(preps: dict[str, dict], rng: np.random.Generator, *, scale: float) -> dict[str, dict]:
    """A recompute-like copy: every float tensor moves by `scale`; sidecar hashes are recomputed on write."""

    copied = copy.deepcopy(preps)
    for prep in copied.values():
        for section in ("development", "controls", "control_train"):
            for key, value in prep[section].items():
                if isinstance(value, np.ndarray) and value.dtype.kind == "f":
                    prep[section][key] = (value + scale * rng.choice([-1.0, 1.0], size=value.shape)).astype(np.float32)
        for entry in prep["banks"]["banks"].values():
            entry["prototypes"] = (entry["prototypes"] + scale * rng.choice([-1.0, 1.0], size=entry["prototypes"].shape)).astype(np.float32)
            if entry["image_mean"] is not None:
                entry["image_mean"] = (entry["image_mean"] + scale).astype(np.float32)
            entry["metadata"]["output_prototype_sha256"] = tensor_scientific_hash(entry["prototypes"])
    return copied


def run_compare(server: Path, recompute: Path, registration_path: Path, out: Path, *, cells=None) -> dict:
    argv = ["compare", "--server-root", str(server), "--recompute-root", str(recompute),
            "--registrations", str(registration_path), "--out", str(out)]
    if cells:
        argv += ["--cells", *cells]
    with contextlib.redirect_stdout(io.StringIO()):
        code = d1_10.main(argv)
    assert code == 0
    return read_json(out)


# --------------------------------------------------------------------------- tests


class RegisteredFamilyAndStrataTests(unittest.TestCase):
    def test_real_registrations_enumerate_exactly_21_contrasts(self):
        for name in ("exp-20260920-019a.json", "exp-20260920-019b.json"):
            registration = read_json(PROJECT / "configs" / "satml2027" / name)
            family = d1_09.registered_candidate_family(registration)
            self.assertEqual(len(family["frozen_grid_contrasts"]), 17)
            self.assertEqual(len(family["control_contrasts"]), 4)
            self.assertEqual(len(family["contrasts"]), 21)
            self.assertEqual(len(family["all_banks"]), 22)
            self.assertEqual(family["reference"], "no_correction")
            self.assertIn(d1_09.X3_CONTROL_ID, family["control_contrasts"])
            for control in d1_09.X4_PER_CLASS_CONTROL_IDS:
                self.assertIn(control, family["control_contrasts"])
            self.assertEqual(d1_09.verify_registration_hash(registration), registration["registration_sha256"])

    def test_registered_evaluation_strata_sizes(self):
        items = PROJECT / "results" / "satml2027" / "items"
        if not items.is_dir():
            self.skipTest("registered item lists are not present on this host")
        expected = {"cifar100": (500, 100, 5), "cifar10": (500, 10, 50), "eurosat": (150, 10, 15)}
        registration = read_json(PROJECT / "configs" / "satml2027" / "exp-20260920-019a.json")
        expected_hashes = {cell["dataset_id"]: cell["evaluation_items_sha256"] for cell in registration["cells"]}
        for dataset, (count, classes, per_class) in expected.items():
            with (items / f"{dataset}__evaluation.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = sorted(csv.DictReader(handle), key=lambda row: row["item_id"])
            labels = np.array([int(row["label"]) for row in rows])
            self.assertEqual(len(rows), count)
            counts = np.bincount(labels, minlength=classes)
            self.assertEqual(len(counts), classes)
            self.assertTrue(np.all(counts == per_class), f"{dataset}: per-class sizes {sorted(set(counts))}")
            self.assertEqual(canonical_json_hash([row["item_id"] for row in rows]), expected_hashes[dataset])
        self.assertAlmostEqual(d1_09.rao_wu_factor(5), np.sqrt(5 / 4))
        self.assertAlmostEqual(d1_09.rao_wu_factor(50), np.sqrt(50 / 49))
        self.assertAlmostEqual(d1_09.rao_wu_factor(15), np.sqrt(15 / 14))
        self.assertEqual(d1_09.rao_wu_factor(1), 1.0)


class RaoWuBootstrapTests(unittest.TestCase):
    def _strata(self, sizes, width, seed):
        rng = np.random.default_rng(seed)
        return [{"dataset_id": f"d{index}", "label": index, "size": size, "positions_per_identity": [2],
                 "aggregate": rng.normal(size=(size, width)) / 20.0} for index, size in enumerate(sizes)]

    def test_vectorized_bootstrap_matches_naive_reference_and_is_chunk_invariant(self):
        strata = self._strata((2, 3, 5), 4, seed=3)
        replicates, seed = 300, 11
        fast = d1_09.rao_wu_bootstrap(strata, replicates=replicates, seed=seed, chunk=7, return_replicates=True)
        other_chunk = d1_09.rao_wu_bootstrap(strata, replicates=replicates, seed=seed, chunk=1000, return_replicates=True)
        np.testing.assert_array_equal(fast["deviation_unscaled"], other_chunk["deviation_unscaled"])
        np.testing.assert_array_equal(fast["deviation_rao_wu"], other_chunk["deviation_rao_wu"])
        generator = np.random.Generator(np.random.PCG64DXSM(seed))
        naive_plain = np.zeros((replicates, 4))
        naive_rw = np.zeros((replicates, 4))
        for stratum in strata:
            size = stratum["size"]
            factor = np.sqrt(size / (size - 1))
            draws = generator.integers(0, size, size=(replicates, size), endpoint=False)
            for replicate in range(replicates):
                weights = np.bincount(draws[replicate], minlength=size).astype(float)
                contribution = (weights - 1.0) @ stratum["aggregate"]
                naive_plain[replicate] += contribution
                naive_rw[replicate] += factor * contribution
        np.testing.assert_allclose(fast["deviation_unscaled"], naive_plain, atol=1e-12)
        np.testing.assert_allclose(fast["deviation_rao_wu"], naive_rw, atol=1e-12)
        critical = float(np.quantile(np.max(np.abs(naive_rw), axis=1), 0.95, method="higher"))
        self.assertAlmostEqual(fast["critical_rao_wu"], critical, places=12)
        self.assertEqual(fast["rao_wu_factors"], [np.sqrt(2), np.sqrt(1.5), np.sqrt(1.25)])
        self.assertEqual(fast["singleton_strata"], 0)

    def test_uniform_strata_scale_the_critical_value_exactly(self):
        strata = self._strata((4, 4, 4), 3, seed=5)
        result = d1_09.rao_wu_bootstrap(strata, replicates=500, seed=2)
        self.assertAlmostEqual(result["critical_rao_wu"], np.sqrt(4 / 3) * result["critical_unscaled"], places=12)
        self.assertGreater(result["critical_unscaled"], 0.0)

    def test_prediction_rules_and_region_language(self):
        intervals = {
            d1_09.X3_CONTROL_ID: {"point_difference": 0.004, "lower": -0.006, "upper": 0.014},
            d1_09.X3_COMPANION_ID: {"point_difference": 0.02, "lower": 0.01, "upper": 0.03},
            "control__noisy_class_mean": {"point_difference": 0.03, "lower": 0.02, "upper": 0.04},
            "control__lowrank_tangent_r8": {"point_difference": -0.02, "lower": -0.03, "upper": -0.01},
        }
        verdict = d1_09.evaluate_predictions(intervals, region=0.005, critical=0.01)
        self.assertTrue(verdict["X3"]["confirmed"])
        self.assertFalse(verdict["X3"]["companion_contains_zero"])
        self.assertTrue(verdict["X4"]["confirmed"])
        self.assertEqual(verdict["X4"]["positive_exclusions"],
                         {"control__noisy_class_mean": True, "control__lowrank_tangent_r8": False})
        self.assertFalse(verdict["practical_effect_region"]["band_narrower_than_region"])
        narrow = d1_09.interval_flags({"lower": -0.003, "upper": 0.003}, region=0.005, critical=0.003)
        self.assertTrue(narrow["inside_region_language_permitted"])
        wide = d1_09.interval_flags({"lower": -0.004, "upper": 0.004}, region=0.005, critical=0.006)
        self.assertFalse(wide["inside_region_language_permitted"])
        with self.assertRaises(RuntimeError):
            d1_09.evaluate_predictions({d1_09.X3_CONTROL_ID: intervals[d1_09.X3_CONTROL_ID]}, region=0.005, critical=0.01)


class SensitivityEndToEndTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.registration = synthetic_registration()
        self.registration_path = self.root / "registration.json"
        write_json_atomic(self.registration_path, self.registration)
        self.merged = self.root / "merged"
        self.merged.mkdir()
        self.primary = self.root / "analysis"

    def tearDown(self):
        self._temporary.cleanup()

    def test_guard_refuses_then_runs_after_approval_and_matches_primary(self):
        write_synthetic_merged(self.merged, self.registration, np.random.Generator(np.random.PCG64DXSM(7)))
        self.assertEqual(run_primary(self.merged, self.registration_path, self.primary), 0)
        contrasts = read_json(self.primary / "contrasts.json")
        self.assertEqual(sorted(contrasts["contrasts"]), d1_09.registered_candidate_family(self.registration)["contrasts"])
        self.assertEqual(contrasts["simultaneous_band"]["replicates"], TEST_REPLICATES)

        with self.assertRaisesRegex(RuntimeError, "addendum V1 is absent"):
            run_sensitivity(self.merged, self.registration_path, self.primary, self.root)
        (self.root / "research").mkdir()
        (self.root / "research" / d1_09.ADDENDUM_RELATIVE_PATH.name).write_text("synthetic addendum\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "approval .* is absent"):
            run_sensitivity(self.merged, self.registration_path, self.primary, self.root)
        write_approval(self.root, overrides={"addendum_sha256": "0" * 64})
        with self.assertRaisesRegex(RuntimeError, "addendum hash mismatch"):
            run_sensitivity(self.merged, self.registration_path, self.primary, self.root)
        write_approval(self.root, overrides={"sensitivity_source_sha256": "0" * 64})
        with self.assertRaisesRegex(RuntimeError, "sensitivity-source hash mismatch"):
            run_sensitivity(self.merged, self.registration_path, self.primary, self.root)
        write_approval(self.root, overrides={"status": "PENDING"})
        with self.assertRaisesRegex(RuntimeError, "not independently approved"):
            run_sensitivity(self.merged, self.registration_path, self.primary, self.root)
        write_approval(self.root, overrides={"new_sampling_authorized": True})
        with self.assertRaisesRegex(RuntimeError, "new_sampling_authorized"):
            run_sensitivity(self.merged, self.registration_path, self.primary, self.root)
        # retrospective-review schema (V3 audit G1, D-135)
        write_approval(self.root, overrides={"outcome_blind_at_approval": True})
        with self.assertRaisesRegex(RuntimeError, "outcome_blind_at_approval as false"):
            run_sensitivity(self.merged, self.registration_path, self.primary, self.root)
        write_approval(self.root, overrides={"unblinding_decision": None})
        with self.assertRaisesRegex(RuntimeError, "recorded unblinding"):
            run_sensitivity(self.merged, self.registration_path, self.primary, self.root)
        write_approval(self.root, overrides={"review_date": "2026-09-24"})
        with self.assertRaisesRegex(RuntimeError, "cannot predate"):
            run_sensitivity(self.merged, self.registration_path, self.primary, self.root)
        write_approval(self.root, overrides={"retrospective_amendment_sha256": "0" * 64})
        with self.assertRaisesRegex(RuntimeError, "retrospective-amendment hash mismatch"):
            run_sensitivity(self.merged, self.registration_path, self.primary, self.root)
        retired = self.root / d1_09.RETIRED_APPROVAL_RELATIVE_PATH
        retired.write_text("{}", encoding="utf-8")
        write_approval(self.root)
        with self.assertRaisesRegex(RuntimeError, "retired"):
            run_sensitivity(self.merged, self.registration_path, self.primary, self.root)
        retired.unlink()
        self.assertFalse((self.primary / d1_09.OUTPUT_NAME).exists())

        write_approval(self.root)
        self.assertEqual(run_sensitivity(self.merged, self.registration_path, self.primary, self.root), 0)
        output = read_json(self.primary / d1_09.OUTPUT_NAME)
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            run_sensitivity(self.merged, self.registration_path, self.primary, self.root)

        self.assertEqual(output["contrast_family"]["count"], 21)
        self.assertEqual(len(output["cells"]), 4)
        self.assertEqual(len(output["bridge_cells_excluded"]), 2)
        self.assertEqual(output["strata"]["count"], 5)
        self.assertEqual(output["strata"]["unique_identities"], 12)
        self.assertEqual(output["strata"]["by_dataset"]["dsa"]["sizes"], [2])
        self.assertEqual(output["strata"]["by_dataset"]["cifar10"]["sizes"], [3])
        self.assertEqual(output["strata"]["by_dataset"]["dsa"]["positions_per_identity"], [2])
        self.assertAlmostEqual(output["strata"]["by_dataset"]["dsa"]["rao_wu_factors"][0], np.sqrt(2))
        self.assertAlmostEqual(output["strata"]["by_dataset"]["cifar10"]["rao_wu_factors"][0], np.sqrt(1.5))
        self.assertEqual(output["sensitivity_bootstrap"]["replicates"], TEST_REPLICATES)
        self.assertEqual(output["sensitivity_bootstrap"]["seed"], d1_07.BOOTSTRAP_SEED)
        self.assertEqual(output["sensitivity_bootstrap"]["generator"], "PCG64DXSM")
        self.assertEqual(output["registered_band"]["critical_max_absolute_deviation"],
                         contrasts["simultaneous_band"]["critical_max_absolute_deviation"])
        self.assertEqual(output["point_agreement_with_primary_max_abs"], 0.0)
        self.assertAlmostEqual(output["sensitivity_bootstrap"]["global_inflation_bound"]["factor"], np.sqrt(2))
        registered = output["registered_band"]["critical_max_absolute_deviation"]
        unscaled = output["sensitivity_bootstrap"]["critical_unscaled"]
        self.assertGreater(registered, 0.0)
        self.assertLess(abs(unscaled - registered) / registered, 0.5)  # Monte-Carlo agreement only
        self.assertGreater(output["sensitivity_bootstrap"]["critical_rao_wu"], unscaled)

        differences = d1_09.load_primary_differences(self.merged, self.registration)["differences"]
        region, critical = 0.005, registered
        for candidate, entry in output["contrasts"].items():
            interval = contrasts["contrasts"][candidate]["simultaneous_confidence_interval"]
            self.assertEqual(entry["registered_interval"]["lower"], interval["lower"])
            self.assertEqual(entry["point_difference_equal_cell_macro"], interval["point_difference"])
            self.assertAlmostEqual(entry["item_pooled_point_difference"], float(np.mean(differences[candidate])), places=12)
            self.assertAlmostEqual(
                sum(cell["item_count"] * cell["cell_difference"] for cell in entry["per_cell"])
                / sum(cell["item_count"] for cell in entry["per_cell"]),
                entry["item_pooled_point_difference"], places=12)
            self.assertEqual(entry["registered_interval_flags"], d1_09.interval_flags(interval, region=region, critical=critical))
            self.assertAlmostEqual(entry["rao_wu_interval"]["upper"] - entry["rao_wu_interval"]["lower"],
                                   2 * output["sensitivity_bootstrap"]["critical_rao_wu"], places=12)
        x3 = output["predictions_registered_band"]["X3"]
        x4 = output["predictions_registered_band"]["X4"]
        self.assertTrue(x3["confirmed"])  # the X3 control was constructed identical to the reference
        self.assertEqual(x3["interval"]["point_difference"], 0.0)
        self.assertTrue(x4["confirmed"])  # the rank-8 control was constructed certified-correct everywhere
        self.assertTrue(x4["positive_exclusions"]["control__lowrank_tangent_r8"])
        self.assertEqual(output["approval"]["status"], d1_09.REQUIRED_APPROVAL_STATUS)
        self.assertIs(output["approval"]["outcome_blind_at_approval"], False)
        self.assertIn("RETROSPECTIVELY", output["status"])
        self.assertFalse(output["final_test_access"])

    def test_constant_within_stratum_differences_give_zero_bands(self):
        write_synthetic_merged(self.merged, self.registration, np.random.Generator(np.random.PCG64DXSM(9)),
                               mode="stratum_constant")
        self.assertEqual(run_primary(self.merged, self.registration_path, self.primary), 0)
        contrasts = read_json(self.primary / "contrasts.json")
        self.assertLessEqual(contrasts["simultaneous_band"]["critical_max_absolute_deviation"], 1e-12)
        write_approval(self.root)
        self.assertEqual(run_sensitivity(self.merged, self.registration_path, self.primary, self.root), 0)
        output = read_json(self.primary / d1_09.OUTPUT_NAME)
        self.assertLessEqual(output["sensitivity_bootstrap"]["critical_unscaled"], 1e-12)
        self.assertLessEqual(output["sensitivity_bootstrap"]["critical_rao_wu"], 1e-12)
        self.assertEqual(output["point_agreement_with_primary_max_abs"], 0.0)

    def test_registration_edit_is_refused(self):
        write_synthetic_merged(self.merged, self.registration, np.random.Generator(np.random.PCG64DXSM(1)))
        self.assertEqual(run_primary(self.merged, self.registration_path, self.primary), 0)
        write_approval(self.root)
        edited = dict(self.registration)
        edited["prospective_practical_effect_region"] = {"absolute_ca": 0.01, "interpretation": "edited"}
        write_json_atomic(self.registration_path, edited)
        with self.assertRaisesRegex(RuntimeError, "registration hash mismatch"):
            run_sensitivity(self.merged, self.registration_path, self.primary, self.root)


class TF32AdjudicationTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.registration = synthetic_registration("EXP-TEST-019B")
        self.registration_path = self.root / "registration.json"
        write_json_atomic(self.registration_path, self.registration)
        rng = np.random.Generator(np.random.PCG64DXSM(21))
        self.preps = {cell["cell_id"]: synthetic_preparation(cell, self.registration, rng)
                      for cell in self.registration["cells"]}
        self.server = self.root / "server"
        write_mirror_tree(self.server, self.preps)

    def tearDown(self):
        self._temporary.cleanup()

    def test_identical_and_within_tolerance_trees_are_immaterial(self):
        identical = self.root / "identical"
        write_mirror_tree(identical, self.preps)
        report = run_compare(self.server, identical, self.registration_path, self.root / "identical.json")
        self.assertEqual(report["decision"]["decision"], d1_10.DECISION_IMMATERIAL)
        self.assertEqual(report["decision"]["cells_compared"], len(self.preps))
        for cell in report["cells"].values():
            self.assertEqual(cell["summary"]["bank_count"], 22)
            self.assertEqual(cell["summary"]["banks_exact_hash_equal"], 22)
            self.assertEqual(cell["summary"]["max_bank_deviation"], 0.0)
            self.assertTrue(cell["summary"]["gr_clip_image_mean_within_tolerance"])
            self.assertEqual(cell["discrete_differences"], [])
            self.assertEqual(cell["tie_flags"]["critical_set_differences"], 0)
            self.assertTrue(cell["deployability"]["other"]["clean_direction"]["implied_deployable"])
        self.assertFalse(report["result_files_touched"])
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            run_compare(self.server, identical, self.registration_path, self.root / "identical.json")

        nearby = self.root / "nearby"
        write_mirror_tree(nearby, perturb(self.preps, np.random.Generator(np.random.PCG64DXSM(2)), scale=1e-7))
        report = run_compare(self.server, nearby, self.registration_path, self.root / "nearby.json")
        self.assertEqual(report["decision"]["decision"], d1_10.DECISION_IMMATERIAL)
        self.assertEqual(report["decision"]["banks_beyond_tolerance"], {})
        self.assertLessEqual(report["decision"]["max_bank_deviation_over_cells"], 2e-6)
        self.assertGreater(report["decision"]["max_bank_deviation_over_cells"], 0.0)
        cell = next(iter(report["cells"].values()))
        self.assertGreater(cell["arrays"]["development.clean_features"]["max_abs"], 0.0)
        self.assertIsNotNone(cell["arrays"]["development.noisy_features"]["relative_frobenius"])
        self.assertTrue(cell["sidecars"]["hash_mismatches"])  # recomputed tensor hashes differ, which is not discrete

    def test_bank_beyond_tolerance_downgrades(self):
        beyond = perturb(self.preps, np.random.Generator(np.random.PCG64DXSM(3)), scale=1e-7)
        cell_id = self.registration["cells"][0]["cell_id"]
        target = beyond[cell_id]["banks"]["banks"]["control__noisy_class_mean"]
        target["prototypes"] = (target["prototypes"] + 1e-5).astype(np.float32)
        target["metadata"]["output_prototype_sha256"] = tensor_scientific_hash(target["prototypes"])
        recompute = self.root / "beyond"
        write_mirror_tree(recompute, beyond)
        report = run_compare(self.server, recompute, self.registration_path, self.root / "beyond.json")
        self.assertEqual(report["decision"]["decision"], d1_10.DECISION_DOWNGRADE)
        self.assertEqual(report["decision"]["banks_beyond_tolerance"], {cell_id: ["control__noisy_class_mean"]})
        self.assertEqual(report["decision"]["discrete_differences"], {})
        self.assertGreater(report["cells"][cell_id]["banks"]["control__noisy_class_mean"]["max_abs"], 2e-6)

    def test_discrete_differences_downgrade(self):
        cells = self.registration["cells"]
        # (a) a missing candidate bank
        missing = copy.deepcopy(self.preps)
        del missing[cells[0]["cell_id"]]["banks"]["banks"]["clean_boundary_active__step_0.01"]
        recompute = self.root / "missing"
        write_mirror_tree(recompute, missing)
        report = run_compare(self.server, recompute, self.registration_path, self.root / "missing.json",
                             cells=[cells[0]["cell_id"]])
        self.assertEqual(report["decision"]["decision"], d1_10.DECISION_DOWNGRADE)
        self.assertFalse(report["cells"][cells[0]["cell_id"]]["candidate_ids"]["equal"])
        paths = {entry["path"] for entry in report["cells"][cells[0]["cell_id"]]["discrete_differences"]}
        self.assertIn("banks.candidate_ids", paths)

        # (b) a discrete metadata field
        edited = copy.deepcopy(self.preps)
        edited[cells[1]["cell_id"]]["banks"]["banks"]["global_mean_centering__coefficient_0.5"]["metadata"]["step_or_coefficient"] = 0.25
        recompute = self.root / "edited"
        write_mirror_tree(recompute, edited)
        report = run_compare(self.server, recompute, self.registration_path, self.root / "edited.json",
                             cells=[cells[1]["cell_id"]])
        self.assertEqual(report["decision"]["decision"], d1_10.DECISION_DOWNGRADE)
        paths = {entry["path"] for entry in report["cells"][cells[1]["cell_id"]]["discrete_differences"]}
        self.assertIn("banks.metadata.global_mean_centering__coefficient_0.5.step_or_coefficient", paths)

        # (c) a tie flag: an exact two-way tie on one development item under basis prototypes
        rng = np.random.Generator(np.random.PCG64DXSM(5))
        basis = {cells[2]["cell_id"]: synthetic_preparation(cells[2], self.registration, rng, basis_prototypes=True)}
        server = self.root / "basis_server"
        write_mirror_tree(server, basis)
        tied = copy.deepcopy(basis)
        features = tied[cells[2]["cell_id"]]["development"]["clean_features"].copy()
        features[0] = 0.0
        features[0, 0] = features[0, 1] = np.float32(1.0 / np.sqrt(2.0))
        tied[cells[2]["cell_id"]]["development"]["clean_features"] = features
        recompute = self.root / "tied"
        write_mirror_tree(recompute, tied)
        report = run_compare(server, recompute, self.registration_path, self.root / "tied.json",
                             cells=[cells[2]["cell_id"]])
        self.assertEqual(report["decision"]["decision"], d1_10.DECISION_DOWNGRADE)
        ties = report["cells"][cells[2]["cell_id"]]["tie_flags"]
        self.assertEqual(ties["tie_flag_differences"], 1)
        self.assertEqual(ties["tied_items_other"], 1)
        self.assertEqual(ties["tied_items_reference"], 0)

    def test_manifest_check_and_result_path_guard(self):
        names = sorted(path.name for path in self.server.iterdir() if path.is_file())
        manifest = self.root / "prep.manifest.sha256"
        lines = [f"{sha256_file(self.server / name)}  results/satml2027/{name}" for name in names]
        lines.append("0" * 64 + "  results/other/ignored.txt")
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        report = d1_10.verify_manifest(self.server, manifest, prefix="results/satml2027/")
        self.assertEqual(report["matched"], len(names))
        self.assertEqual(report["skipped_outside_prefix"], 1)
        self.assertTrue(report["all_listed_files_match"])
        lines[0] = "f" * 64 + lines[0][64:]
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            code = d1_10.main(["manifest-check", "--root", str(self.server), "--manifest", str(manifest),
                               "--out", str(self.root / "manifest_check.json")])
        self.assertEqual(code, 2)
        checked = read_json(self.root / "manifest_check.json")
        self.assertEqual(checked["manifest"]["mismatched"], [f"results/satml2027/{names[0]}"])
        with self.assertRaisesRegex(RuntimeError, "result directory"):
            d1_10.assert_not_result_path(self.root / "results" / "satml2027" / "EXP-20260920-019A" / "cells", "--root")
        with self.assertRaisesRegex(RuntimeError, "result directory"):
            d1_10.main(["compare", "--server-root", str(self.server), "--recompute-root", str(self.server),
                        "--registrations", str(self.registration_path),
                        "--out", str(self.root / "merged" / "x.json")])

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not importable on this host")
    def test_pt_tree_round_trip_matches_mirror(self):
        cell = self.registration["cells"][0]
        single = {cell["cell_id"]: self.preps[cell["cell_id"]]}
        pt_root = self.root / "pt"
        write_pt_tree(pt_root, single)
        loaded = d1_10.load_cell(pt_root, cell["cell_id"])
        self.assertEqual(loaded.source_format, "pt")
        mirror = to_cell_preparation(single[cell["cell_id"]])
        self.assertEqual(loaded.metadata, mirror.metadata)
        self.assertEqual(sorted(loaded.arrays), sorted(mirror.arrays))
        for name, value in mirror.arrays.items():
            np.testing.assert_array_equal(loaded.arrays[name], value)
        exported = self.root / "exported"
        with contextlib.redirect_stdout(io.StringIO()):
            code = d1_10.main(["export-npz", "--root", str(pt_root), "--registrations", str(self.registration_path),
                               "--cells", cell["cell_id"], "--out", str(exported)])
        self.assertEqual(code, 0)
        report = run_compare(pt_root, exported, self.registration_path, self.root / "roundtrip.json",
                             cells=[cell["cell_id"]])
        self.assertEqual(report["decision"]["decision"], d1_10.DECISION_IMMATERIAL)
        summary = report["cells"][cell["cell_id"]]["summary"]
        self.assertEqual(summary["banks_exact_hash_equal"], 22)
        self.assertEqual(summary["max_bank_deviation"], 0.0)
        self.assertEqual(report["cells"][cell["cell_id"]]["source_formats"], {"reference": "pt", "other": "npz"})


if __name__ == "__main__":
    unittest.main()
