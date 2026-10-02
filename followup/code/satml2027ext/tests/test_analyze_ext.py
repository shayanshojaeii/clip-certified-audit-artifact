"""Known-answer tests for the V2 analysis (torch-free).

  .venv/Scripts/python -m pytest satml2027ext/tests/test_analyze_ext.py -q
"""

from __future__ import annotations

import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from satml2027ext import analyze_ext as analysis  # noqa: E402
from satml2027ext import candidates_ext, guard_ext  # noqa: E402
from satml2027ext import make_registrations_ext as mr  # noqa: E402
from satml2027ext._common import canonical_json_hash, sha256_file, write_json_atomic  # noqa: E402

IDS = candidates_ext.canonical_candidate_ids()
REFERENCE = "no_correction"
CLASSES = 10
SIGMA = 0.25
SELECTION, CONFIRMATION = 128, 4096


def registration(datasets: dict[str, int], models=("mA", "mB"), predictions=None) -> dict:
    cells = []
    for model in models:
        for dataset, items in datasets.items():
            ids = [f"{dataset}-{index:05d}" for index in range(items)]
            cells.append({"cell_id": f"{model}__{dataset}__sigma0.25", "model_id": model, "dataset_id": dataset, "sigma": SIGMA,
                          "class_count": CLASSES, "item_count": items, "evaluation_items_sha256": canonical_json_hash(ids)})
    body = {
        "schema_version": guard_ext.REGISTRATION_SCHEMA, "experiment_id": "EXP-20260921-021B",
        "status": "preregistered_pending_independent_approval",
        "candidates": candidates_ext.canonical_candidates_block(),
        "certification": {"selection_draws": SELECTION, "confirmation_draws": CONFIRMATION, "alpha_per_example": 0.001,
                          "reported_radii": [0.0, 0.1, 0.25, 0.5],
                          "seeds": {"selection_base_seed": 1, "confirmation_base_seed": 2},
                          "streams": {"selection": "cohen_class_selection", "confirmation": "cohen_class_confirmation"},
                          "noise_block_draws": 64, "budget_checkpoints": [], "flip_counters": {"tolerance": 1e-5}},
        "inference": mr.inference_block_primary({}),
        "predictions": predictions if predictions is not None else mr.predictions_021a()[:1] + mr.predictions_021b(),
        "cells": cells,
    }
    body["inference"]["uncertainty"]["replicates"] = 3000
    body["registration_sha256"] = canonical_json_hash(body)
    return body


def certified_counts(correct: np.ndarray, labels: np.ndarray, *, wrong_class: int = 0):
    """Selection/confirmation counts that certify the true class (correct) or a wrong class, far above r = sigma."""

    n = len(labels)
    selected = np.where(correct, labels, (labels + 1 + wrong_class) % CLASSES)
    selection = np.zeros((n, CLASSES), dtype=np.int32)
    confirmation = np.zeros((n, CLASSES), dtype=np.int32)
    selection[np.arange(n), selected] = SELECTION
    confirmation[np.arange(n), selected] = CONFIRMATION
    return selection, confirmation


def write_cell(root: Path, cell: dict, outcomes: dict[str, np.ndarray], *, violations: dict[str, int] | None = None,
               shards: int = 2, prefix: dict[str, np.ndarray] | None = None, candidate_ids=IDS):
    n = int(cell["item_count"])
    ids = np.array([f"{cell['dataset_id']}-{index:05d}" for index in range(n)])
    labels = np.arange(n) % CLASSES
    selection = np.zeros((len(candidate_ids), n, CLASSES), dtype=np.int32)
    confirmation = np.zeros_like(selection)
    for position, candidate_id in enumerate(candidate_ids):
        selection[position], confirmation[position] = certified_counts(outcomes.get(candidate_id, outcomes[REFERENCE]), labels)
    counters = {name: np.zeros((len(candidate_ids), n), dtype=np.int32) for name in analysis.COUNTER_NAMES}
    applicable = np.array([candidate_id in candidates_ext.canonical_candidates_block()["flip_theorem_candidates"] for candidate_id in candidate_ids])
    for candidate_id, count in (violations or {}).items():
        counters["flip_containment_violations"][candidate_ids.index(candidate_id), 0] = count
        counters["flip_changed_draws"][candidate_ids.index(candidate_id), 0] = count
    folder = root / "cells" / cell["cell_id"]
    folder.mkdir(parents=True, exist_ok=True)
    bounds = np.linspace(0, n, shards + 1).astype(int)
    for index in range(shards):
        start, stop = bounds[index], bounds[index + 1]
        arrays = dict(candidate_ids=np.array(candidate_ids), item_ids=ids[start:stop], ground_truth=labels[start:stop],
                      raw_predictions=np.tile(labels[start:stop], (len(candidate_ids), 1)),
                      selection_counts=selection[:, start:stop], confirmation_counts=confirmation[:, start:stop],
                      confirmation_seeds=np.arange(start, stop) + 1, selection_seeds=np.arange(start, stop) + 500001,
                      done=np.ones(stop - start, dtype=bool),
                      dataset_indices=np.arange(start, stop),
                      item_offset=np.array([start]), flip_containment_applicable=applicable,
                      flip_loose_applicable=applicable,
                      **{name: counters[name][:, start:stop] for name in analysis.COUNTER_NAMES})
        if prefix is not None:
            arrays["budget_checkpoints"] = np.array([4096])
            arrays["prefix_confirmation_counts"] = np.stack([prefix["confirmation"][:, start:stop]])
            arrays["selection_counts"] = prefix["selection"][:, start:stop]
            arrays["confirmation_counts"] = prefix["full"][:, start:stop]
        np.savez(folder / f"shard_{index:03d}_of_{shards:03d}.npz", **arrays)


def outcomes_for(n: int, *, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    identity = rng.random(n) < 0.5
    two_sided = identity | (rng.random(n) < 0.20)          # large gain
    image_only = identity & (rng.random(n) >= 0.20)        # large loss
    small = identity.copy()
    wrong = np.flatnonzero(~identity)
    small[wrong[: int(round(0.01 * n))]] = True             # +1 point, 1% discordance
    lowrank = identity | (rng.random(n) < 0.40)
    return {REFERENCE: identity, "gr_clip_style_two_sided__coefficient_1": two_sided,
            "image_only_centering__coefficient_1": image_only, "clean_boundary_active__step_0.16": small,
            "control__lowrank_tangent_r8": lowrank}


class PureFunctionTests(unittest.TestCase):
    def test_holm_and_p_values(self):
        self.assertEqual(analysis.holm([0.5, 0.5]), [1.0, 1.0])
        adjusted = analysis.holm([0.01, 0.04, 0.03, 0.2])
        self.assertAlmostEqual(adjusted[0], 0.04)
        values = np.full(1000, 3.0)
        self.assertEqual(analysis.two_sided_p(values), 0.0)
        self.assertEqual(analysis.tost_p_values(values, 2.0), (0.0, 1.0, 1.0))
        self.assertEqual(analysis.gain_p(values, 2.0), 0.0)
        self.assertEqual(analysis.loss_p(-values, 2.0), 0.0)
        self.assertEqual(analysis.rao_wu_factor(5), math.sqrt(5 / 4))

    def test_direction_and_magnitude_are_orthogonal(self):
        self.assertEqual(analysis.classify_direction(0.6, 0.001, 0.05), "positive")
        self.assertEqual(analysis.classify_direction(-3.0, 0.001, 0.05), "negative")
        self.assertEqual(analysis.classify_direction(3.0, 0.2, 0.05), "unresolved")
        # V1 would have called a precise +0.6 [+0.2, +1.0] "superior" only; V2 reports both facts
        self.assertEqual(analysis.classify_magnitude(1.0, 1.0, 0.001, 0.05), "practically_equivalent")
        self.assertEqual(analysis.classify_magnitude(0.001, 1.0, 1.0, 0.05), "gain_at_least_sesoi")
        self.assertEqual(analysis.classify_magnitude(1.0, 0.001, 1.0, 0.05), "loss_at_least_sesoi")
        self.assertEqual(analysis.classify_magnitude(0.2, 0.3, 0.4, 0.05), "inconclusive")

    def test_romano_wolf_uses_standard_stepdown_adjusted_p_values(self):
        rng = np.random.default_rng(3)
        deviations = rng.normal(0.0, 1.0, size=(20000, 4))
        points = np.array([6.0, 0.3, -0.2, 2.6])
        result = analysis.romano_wolf_stepdown(points, deviations, 0.05)
        self.assertTrue(result["rejected"][0] and result["rejected"][3])
        self.assertFalse(result["rejected"][1] or result["rejected"][2])
        z = np.abs(deviations) / deviations.std(axis=0, ddof=1)
        t = np.abs(points / deviations.std(axis=0, ddof=1))
        # rank 3 (|t| = 0.3): the maximum runs over itself and the smaller statistic only (not over rejected ones)
        expected = max(float(np.mean(z[:, [1, 2]].max(axis=1) >= t[1])), float(np.mean(z[:, [3, 1, 2]].max(axis=1) >= t[3])))
        self.assertAlmostEqual(float(result["adjusted_p"][1]), expected)
        self.assertGreaterEqual(result["adjusted_p"][2], result["adjusted_p"][1])


class SyntheticAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = Path(tempfile.mkdtemp())
        cls.registration = registration({"dsA": 1000, "dsB": 200})
        for cell in cls.registration["cells"]:
            seed = 11 if cell["dataset_id"] == "dsA" else 12
            violations = {"cohen_aligned_noisy_margin__step_0.08": 1} if cell["cell_id"] == "mB__dsB__sigma0.25" else None
            write_cell(cls.temp, cell, outcomes_for(int(cell["item_count"]), seed=seed), violations=violations)
        cls.report = analysis.analyze(cls.registration, results=cls.temp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp, ignore_errors=True)

    def test_families_and_known_effects(self):
        self.assertEqual(set(self.report["families"]), {"dsA__sigma0.25", "dsB__sigma0.25", "pooled__sigma0.25"})
        pooled = self.report["families"]["pooled__sigma0.25"]
        self.assertTrue(pooled["adjudication"])
        self.assertEqual(pooled["n_images"], 1200)
        self.assertEqual(pooled["n_positions"], 2400)
        gain = pooled["contrasts"]["gr_clip_style_two_sided__coefficient_1"]
        self.assertEqual((gain["direction"], gain["magnitude"]), ("positive", "gain_at_least_sesoi"))
        loss = pooled["contrasts"]["image_only_centering__coefficient_1"]
        self.assertEqual((loss["direction"], loss["magnitude"]), ("negative", "loss_at_least_sesoi"))
        zero = pooled["contrasts"]["text_only_centering__coefficient_1"]
        self.assertEqual((zero["direction"], zero["magnitude"]), ("unresolved", "practically_equivalent"))
        small = pooled["contrasts"]["clean_boundary_active__step_0.16"]
        self.assertAlmostEqual(small["point_points"], 1.0, delta=0.05)
        self.assertEqual((small["direction"], small["magnitude"]), ("positive", "practically_equivalent"))
        self.assertIn("ci95_unadjusted_lower", small)
        self.assertNotIn("holm_ci_lower", small)
        control = pooled["contrasts"]["control__lowrank_tangent_r8"]
        self.assertEqual((control["family_role"], control["direction"]), ("positive_control", "positive"))
        self.assertIn("minus_comparator_point", control)
        self.assertEqual(pooled["contrasts"]["global_mean_centering__coefficient_1"]["family_role"], "secondary")

    def test_decomposition_is_reported_with_the_interaction(self):
        decomposition = self.report["families"]["pooled__sigma0.25"]["decomposition"]["1"]
        self.assertAlmostEqual(decomposition["interaction_point"],
                               decomposition["two_sided"] - decomposition["text_only"] - decomposition["image_only"])

    def test_question_verdicts(self):
        questions = self.report["questions"]
        self.assertEqual(questions["Q1"]["verdict"], "reproduced")
        # V4 (audit 6.2): the difference, both p-values and the registered meaning accompany the label
        primary = questions["Q1"]["primary_equal_cell_macro"]
        self.assertAlmostEqual(primary["fresh_minus_saved_points"], primary["point_points"] - 89 / 30)
        self.assertEqual(primary["verdict_meaning"], mr.Q1_VERDICT_MEANING["reproduced"])
        self.assertEqual(primary["verdict_meaning"], analysis.Q1_VERDICT_MEANING["reproduced"])
        self.assertIn("fresh minus saved", questions["Q1"]["sentence"])
        self.assertIn("shortfall p =", questions["Q1"]["sentence"])
        self.assertEqual(primary["reference_points"], 89 / 30)  # the exact fraction, not the rounded planning float
        self.assertEqual(questions["Q3"]["verdict"], "violated")
        self.assertEqual(questions["Q3"]["exceptions"], ["mB__dsB__sigma0.25::cohen_aligned_noisy_margin__step_0.08"])
        self.assertEqual(questions["Q4"]["verdict"], "conforms")
        self.assertIn("changed_projected_gap_draws", questions["Q4"])  # V4 (audit E7): no "numerical ties" label
        self.assertIn("does not identify the cause", questions["Q4"]["tie_budget"]["statement"])
        q6 = questions["Q6"]["pooled__sigma0.25"]["clean_boundary_active__step_0.16"]
        self.assertEqual(q6["sentence_magnitude"], mr.Q6_MAGNITUDE["practically_equivalent"])
        self.assertEqual(questions["Q6"]["dsB__sigma0.25"]["clean_boundary_active__step_0.16"]["role"], "secondary (dataset-specific)")
        self.assertEqual(len(questions["Q7"]["identity_regime"]), 4)
        self.assertIn("control__noisy_class_mean", questions["Q5"]["pooled__sigma0.25"])
        self.assertNotIn("conforms", self.report["tables"]["flip"][0])  # the V1 aggregate comparison is gone

    def test_merge_rejects_sorted_or_incomplete_shards(self):
        cell = dict(self.registration["cells"][0])
        certification = guard_ext.certification_parameters(self.registration)
        with self.assertRaises(RuntimeError):
            analysis.merge_cell_from_shards({**cell, "evaluation_items_sha256": "0" * 64}, self.temp, IDS, certification,
                                            registration=self.registration)
        with self.assertRaises(RuntimeError):
            analysis.merge_cell_from_shards(cell, self.temp, list(reversed(IDS)), certification, registration=self.registration)


class BudgetAnalysisTests(unittest.TestCase):
    def test_prefix_versus_full_certificates_and_the_021b_join(self):
        temp = Path(tempfile.mkdtemp())
        try:
            parent = registration({"dsA": 200}, models=("mA",))
            subset = candidates_ext.subset_candidates_block(candidates_ext.BUDGET_SUBSET_021C, parent_experiment_id=parent["experiment_id"],
                                                            parent_registration_sha256=parent["registration_sha256"])
            child_ids = list(candidates_ext.BUDGET_SUBSET_021C)
            child = registration({"dsA": 100}, models=("mA",), predictions=mr.predictions_021c())
            child.update({"experiment_id": "EXP-20260921-021C", "candidates": subset})
            child["certification"] = dict(child["certification"], confirmation_draws=100000, budget_checkpoints=[4096])
            child["cells"][0].update({"cell_id": "mA__dsA__subset", "parent_cell_id": parent["cells"][0]["cell_id"]})
            child.pop("registration_sha256")
            child["registration_sha256"] = canonical_json_hash(child)
            n = 100
            labels = np.arange(n) % CLASSES
            selection = np.zeros((3, n, CLASSES), dtype=np.int32)
            prefix = np.zeros_like(selection)
            full = np.zeros_like(selection)
            selection[:, np.arange(n), labels] = SELECTION
            prefix[:, np.arange(n), labels] = 3500              # below k_min(4096) = 3518: not certified at r = sigma
            prefix[:, np.arange(n), (labels + 1) % CLASSES] = 596
            full[:, np.arange(n), labels] = 85500               # 0.855 > 0.8449: certified with 100,000 draws
            full[:, np.arange(n), (labels + 1) % CLASSES] = 14500
            prefix[:, :50, :] = 0
            prefix[:, np.arange(50), labels[:50]] = 4096          # first 50 items: certified at both budgets
            full[:, :50, :] = 0
            full[:, np.arange(50), labels[:50]] = 100000
            write_cell(temp / "c", child["cells"][0], {REFERENCE: np.ones(n, dtype=bool)}, shards=2, candidate_ids=child_ids,
                       prefix={"selection": selection, "confirmation": prefix, "full": full})
            parent_counts = np.zeros((len(IDS), 200, CLASSES), dtype=np.int32)
            write_cell(temp / "b", parent["cells"][0], {REFERENCE: np.ones(200, dtype=bool)}, shards=1)
            shard = temp / "b" / "cells" / parent["cells"][0]["cell_id"] / "shard_000_of_001.npz"
            with np.load(shard) as archive:
                arrays = {key: np.array(archive[key]) for key in archive.files}
            for position, candidate_id in enumerate(child_ids):
                arrays["confirmation_counts"][IDS.index(candidate_id), :100] = prefix[position]
            arrays["confirmation_seeds"][:100] = np.arange(100) + 1
            np.savez(shard, **arrays)
            report = analysis.analyze_budget(child, results=temp / "c", parent_registration=parent, parent_results=temp / "b")
            rows = {row["candidate_id"]: row for row in report["questions"]["Q9"]["rows"]}
            self.assertEqual(set(rows), set(child_ids))
            row = rows[REFERENCE]
            self.assertEqual((row["certified_correct_4096"], row["certified_correct_100000"]), (50, 100))
            self.assertEqual((row["changed"], row["gained_with_100000"], row["lost_with_100000"]), (50, 50, 0))
            join = report["questions"]["Q9"]["prefix_identity_with_021b"][0]
            self.assertTrue(join["same_items"] and join["same_confirmation_seeds"] and join["same_selection_seeds"])
            self.assertEqual(join["prefix_equal_to_021b_counts"], {candidate_id: 100 for candidate_id in child_ids})
            self.assertTrue(join["confirmation_prefix_identity"] and join["certificate_input_identity"] and join["exact"])
            # V4 (audit E4): a differing certificate input is reported even when the confirmation prefix is identical
            arrays["raw_predictions"][IDS.index(REFERENCE), 3] = (arrays["raw_predictions"][IDS.index(REFERENCE), 3] + 1) % CLASSES
            np.savez(shard, **arrays)
            report = analysis.analyze_budget(child, results=temp / "c", parent_registration=parent, parent_results=temp / "b")
            join = report["questions"]["Q9"]["prefix_identity_with_021b"][0]
            self.assertTrue(join["confirmation_prefix_identity"])
            self.assertFalse(join["certificate_input_identity"] or join["exact"])
            self.assertEqual(join["raw_predictions_equal_to_021b"][REFERENCE], 99)
            self.assertFalse(report["questions"]["Q9"]["certificate_input_identity_everywhere"])
            with self.assertRaises(RuntimeError):
                analysis.analyze_budget(parent, results=temp / "b")
        finally:
            shutil.rmtree(temp, ignore_errors=True)


class MainGuardTests(unittest.TestCase):
    def test_main_refuses_without_the_analysis_approval(self):
        temp = Path(tempfile.mkdtemp())
        try:
            reg = registration({"dsA": 20})
            path = temp / "registration.json"
            write_json_atomic(path, reg)
            for name in ("custody.json", "manifest.json", "preflight.json"):
                write_json_atomic(temp / name, {"x": 1})
            argv = ["--registration", str(path), "--results", str(temp), "--custody", str(temp / "custody.json"),
                    "--bank-manifest", str(temp / "manifest.json"), "--data-preflight", str(temp / "preflight.json"),
                    "--out", str(temp / "out")]
            with self.assertRaises(guard_ext.ApprovalError):
                analysis.main(argv, approval_path=temp / "missing.json")
            approval = guard_ext.approval_template({key: reg["registration_sha256"] if key == reg["experiment_id"] else "0" * 64
                                                    for key in guard_ext.EXPERIMENT_IDS})
            approval.update(status=guard_ext.REQUIRED_APPROVAL_STATUS, preparation_authorized=True, sampling_authorized=True,
                            analysis_authorized=False, approved_by="r", approved_on="d")
            approval.pop("instructions")
            write_json_atomic(temp / "approval.json", approval)
            with self.assertRaises(guard_ext.ApprovalError):
                analysis.main(argv, approval_path=temp / "approval.json")
            self.assertFalse((temp / "out").exists())
        finally:
            shutil.rmtree(temp, ignore_errors=True)


class V3AnalysisTests(unittest.TestCase):
    """Internal re-review of V2 (2026-09-26): B1, M1d, m2, m3, m4, m6."""

    @classmethod
    def setUpClass(cls):
        cls.temp = Path(tempfile.mkdtemp())
        reg = registration({"dsA": 400, "dsB": 100})
        reg.pop("registration_sha256")
        reg["inference"]["family_power"] = {"dsB__sigma0.25": {"label": "underpowered", "unique_images": 100}}
        reg["registration_sha256"] = canonical_json_hash(reg)
        cls.registration = reg
        for cell in reg["cells"]:
            write_cell(cls.temp, cell, outcomes_for(int(cell["item_count"]), seed=21 if cell["dataset_id"] == "dsA" else 22))
        cls.report = analysis.analyze(reg, results=cls.temp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp, ignore_errors=True)

    def test_v3_b1_q1_classify_covers_the_five_verdicts(self):
        rng = np.random.default_rng(7)
        cases = {"reproduced": (3.0, 0.3), "reproduced_smaller": (1.5, 0.3), "not_reproduced": (0.2, 0.5),
                 "inconclusive": (1.0, 1.5), "reversed": (-2.0, 0.3)}
        for expected, (mean, sd) in cases.items():
            values = mean + sd * rng.standard_normal(200000)
            self.assertEqual(analysis.q1_classify(mean, values, 2.97, 0.05)["verdict"], expected, expected)
        with self.assertRaises(ValueError):
            analysis.q1_classify(1.0, np.ones(10), 0.0, 0.05)

    def test_v3_b1_registered_q1_block_matches_the_saved_draws(self):
        import json

        entry = mr.q1_block()
        references = entry["saved_draw_reference_points"]
        self.assertEqual((references["equal_cell_macro"]["exact"], references["item_pooled"]["exact"]), ("89/30", "34/33"))
        self.assertNotIn("otherwise", entry)
        self.assertNotIn("conditional on its saved draws", json.dumps(entry))
        self.assertEqual(set(entry["sentences"]["equal_cell_macro"]), set(analysis.Q1_VERDICTS))
        self.assertEqual(set(entry["sentences"]["item_pooled"]), set(analysis.Q1_VERDICTS))
        gate = json.loads((PROJECT / "artifacts/negative_paper/gate_n2_simultaneous_analysis_v1.json").read_text(encoding="utf-8"))
        point = next(row["point_estimate"] for row in gate["contrasts"] if row["candidate_id"] == entry["contrast"]
                     and row["radius"] == 0.25 and row["metric"] == "standard_cohen_certified_accuracy")
        self.assertAlmostEqual(100 * point, references["equal_cell_macro"]["value"], places=9)
        self.assertEqual(entry["reference_source"]["sha256"], sha256_file(PROJECT / mr.Q1_PLANNING_PATH))
        planning = json.loads((PROJECT / mr.Q1_PLANNING_PATH).read_text(encoding="utf-8"))
        self.assertEqual(planning["saved_draws_under_registered_q1_analysis"]["macro"]["verdict"], "reproduced")
        self.assertTrue(all(cell["items_equal"] and cell["labels_equal"] for cell in planning["identity_checks"]["cells"].values()))

    def test_v3_b1_q1_verdict_uses_the_equal_cell_estimand_and_descriptive_sentences(self):
        q1 = self.report["questions"]["Q1"]
        self.assertEqual((q1["verdict"], q1["estimand"]), ("reproduced", "equal_cell_macro"))
        self.assertIn("equal-cell gain", q1["sentence"])
        self.assertIn("no Q1 verdict is an item-level replication", q1["sentence"])
        self.assertIn("item-pooled gain", q1["secondary_item_pooled"]["sentence"])
        registered = mr.q1_block()["saved_draw_reference_points"]["equal_cell_macro"]
        # V4: the analysis uses the exact fraction; the registered float is rounded to 10 significant digits
        self.assertEqual(q1["primary_equal_cell_macro"]["reference_points"], 89 / 30)
        self.assertEqual(registered["exact"], "89/30")
        self.assertAlmostEqual(q1["primary_equal_cell_macro"]["reference_points"], registered["value"], delta=1e-8)

    def test_v3_m4_not_run_cells_are_reported_and_q1_is_not_evaluable(self):
        report = analysis.analyze(self.registration, results=self.temp, replicates_override=500, not_run=["mA__dsB__sigma0.25"])
        self.assertEqual(report["not_run_cells"], ["mA__dsB__sigma0.25"])
        self.assertEqual(report["questions"]["Q1"]["verdict"], "not_evaluable")
        family = report["families"]["dsB__sigma0.25"]
        self.assertEqual((family["cells"], family["not_run_cells_of_family"]), (["mB__dsB__sigma0.25"], ["mA__dsB__sigma0.25"]))
        # V4: under scope descent the registered power statement no longer applies
        self.assertEqual(family["power"]["cells_not_run"], 1)
        self.assertNotEqual(family["power"]["effective"], family["power"]["registered"])
        self.assertIn("no longer applies", family["power"]["effective"])
        self.assertEqual(report["families"]["pooled__sigma0.25"]["power"]["cells_not_run"], 1)
        with self.assertRaises(RuntimeError):
            analysis.analyze(self.registration, results=self.temp, replicates_override=500, not_run=["unregistered"])

    def test_v3_m3_merge_checks_labels_and_indices_against_the_registered_list(self):
        cell = next(value for value in self.registration["cells"] if value["cell_id"] == "mA__dsA__sigma0.25")
        certification = guard_ext.certification_parameters(self.registration)
        rows = [{"item_id": f"dsA-{index:05d}", "label": str(index % CLASSES), "dataset_index": str(index)} for index in range(400)]
        analysis.merge_cell_from_shards(cell, self.temp, IDS, certification, registered_rows=rows, registration=self.registration)
        wrong_label = [dict(row) for row in rows]
        wrong_label[7]["label"] = str((int(wrong_label[7]["label"]) + 1) % CLASSES)
        with self.assertRaisesRegex(RuntimeError, "labels"):
            analysis.merge_cell_from_shards(cell, self.temp, IDS, certification, registered_rows=wrong_label,
                                            registration=self.registration)
        wrong_index = [dict(row) for row in rows]
        wrong_index[3]["dataset_index"] = "99999"
        with self.assertRaisesRegex(RuntimeError, "dataset indices"):
            analysis.merge_cell_from_shards(cell, self.temp, IDS, certification, registered_rows=wrong_index,
                                            registration=self.registration)

    def test_v3_m6_family_power_labels_reach_the_report_and_q6(self):
        power = self.report["families"]["dsB__sigma0.25"]["power"]
        self.assertEqual(power["registered"]["label"], "underpowered")
        self.assertEqual(power["effective"], power["registered"])  # no scope descent: the registered label applies
        self.assertIsNone(self.report["families"]["dsA__sigma0.25"]["power"]["registered"])
        row = self.report["questions"]["Q6"]["dsB__sigma0.25"]["clean_boundary_active__step_0.16"]
        self.assertEqual(row["power"]["registered"]["label"], "underpowered")
        self.assertEqual(mr.family_power_021b()["eurosat_sealed__sigma0.25"]["label"], "underpowered (power about 0.25 for 2 points)")

    def test_v3_m1d_flip_conformance_is_never_vacuous(self):
        def row(candidate, *, step=None, applicable=True, violations=0, cell="c1"):
            return {"cell_id": cell, "candidate_id": candidate, "step": step, "containment_applicable": applicable,
                    "loose_applicable": applicable, "flip_changed_draws": 3, "flip_containment_violations": violations,
                    "flip_loose_violations": 0}
        theorem = ["clean_boundary_active__step_0.16", "chowers_exact_projected_gap__coefficient_1"]
        tangent = "control__learned_shared_translation_tangent"
        rows = [row(theorem[0], step=0.16, applicable=False), row(theorem[1]), row(tangent, applicable=False)]
        summary = analysis.flip_summary(rows, theorem_ids=theorem, exploratory_ids=[tangent])
        self.assertEqual((summary["q3_verdict"], summary["q4_verdict"]), ("not_evaluable", "not_evaluable"))
        self.assertEqual(summary["not_applicable_pairs"], ["c1::clean_boundary_active__step_0.16"])
        self.assertFalse(summary["containment_conforms_everywhere"])
        self.assertEqual([entry["candidate_id"] for entry in summary["exploratory_containment"]], [tangent])
        missing = analysis.flip_summary([row(theorem[1])], theorem_ids=theorem)
        self.assertEqual(missing["q3_verdict"], "not_evaluable")
        self.assertIn("*::clean_boundary_active__step_0.16", missing["not_applicable_pairs"])
        ok = analysis.flip_summary([row(theorem[0], step=0.16), row(theorem[1])], theorem_ids=theorem)
        self.assertEqual((ok["q3_verdict"], ok["q4_verdict"]), ("conforms", "conforms"))
        bad = analysis.flip_summary([row(theorem[0], step=0.16, violations=1), row(theorem[1])], theorem_ids=theorem)
        self.assertEqual(bad["q3_verdict"], "violated")

    def test_v3_m2_budget_join_is_required_and_bound_to_the_named_parent(self):
        parent = registration({"dsA": 20}, models=("mA",))
        subset = candidates_ext.subset_candidates_block(candidates_ext.BUDGET_SUBSET_021C, parent_experiment_id=parent["experiment_id"],
                                                        parent_registration_sha256=parent["registration_sha256"])
        child = registration({"dsA": 10}, models=("mA",), predictions=mr.predictions_021c())
        child.update({"experiment_id": "EXP-20260921-021C", "candidates": subset})
        child["certification"] = dict(child["certification"], confirmation_draws=100000, budget_checkpoints=[4096])
        with self.assertRaisesRegex(RuntimeError, "prefix join"):
            analysis.analyze_budget(child, results=self.temp)
        other = dict(parent, registration_sha256="9" * 64)
        with self.assertRaisesRegex(RuntimeError, "names"):
            analysis.analyze_budget(child, results=self.temp, parent_registration=other, parent_results=self.temp)


if __name__ == "__main__":
    unittest.main()


class V4TieBudgetTests(unittest.TestCase):
    def test_projected_gap_tie_budget_reads_the_operator_records_of_the_sidecar(self):
        temp = Path(tempfile.mkdtemp())
        try:
            cell = {"cell_id": "mA__dsA__sigma0.25"}
            folder = temp / "cells" / cell["cell_id"]
            folder.mkdir(parents=True)
            operator = {"row_norms": [1.0, 1.0 + 3e-7, 1.0 - 1e-7], "row_scales": [1.0, 1.0, 1.0]}
            write_json_atomic(folder / "shard_000_of_001.meta.json", {"candidate_metadata": {
                "chowers_exact_projected_gap__coefficient_1": {"metadata": {"operator": operator}},
                "no_correction": {"metadata": {"operator": {"row_norms": [1.0, 2.0]}}}}})
            (folder / "shard_000_of_001.npz").write_bytes(b"")  # V4.1: a cell directory holds one complete shard set
            budget = analysis.projected_gap_tie_budget(temp, cell, ["no_correction", "chowers_exact_projected_gap__coefficient_1"], 1e-5)
            self.assertEqual(list(budget), ["chowers_exact_projected_gap__coefficient_1"])
            entry = budget["chowers_exact_projected_gap__coefficient_1"]
            self.assertAlmostEqual(entry["max_normalizer_difference"], 4e-7)
            self.assertAlmostEqual(entry["margin_bound"], 4e-7 + 1e-5)
            summary = analysis.flip_summary([], theorem_ids=None, tie_budgets={cell["cell_id"]: budget})
            self.assertAlmostEqual(summary["q4_tie_budget"]["max_margin_bound"], 4e-7 + 1e-5)
        finally:
            shutil.rmtree(temp, ignore_errors=True)


class V41MergeTests(unittest.TestCase):
    def test_mixed_unsigned_and_signed_63_bit_seeds_merge_exactly(self):
        temp = Path(tempfile.mkdtemp())
        try:
            reg = registration({"dsA": 6}, models=("mA",))
            cell = reg["cells"][0]
            write_cell(temp, cell, outcomes_for(6, seed=5), shards=2)
            expected = {}
            for index, dtype in ((0, np.uint64), (1, np.int64)):
                path = temp / "cells" / cell["cell_id"] / f"shard_{index:03d}_of_002.npz"
                with np.load(path) as archive:
                    arrays = {key: np.array(archive[key]) for key in archive.files}
                for name, base in (("selection_seeds", (1 << 62) + 1), ("confirmation_seeds", (1 << 62) + 3)):
                    values = [base + 2 * (int(arrays["item_offset"][0]) + position) for position in range(len(arrays[name]))]
                    arrays[name] = np.array(values, dtype=dtype)
                    expected.setdefault(name, []).extend(values)
                np.savez(path, **arrays)
            merged = analysis.merge_cell_from_shards(cell, temp, IDS, guard_ext.certification_parameters(reg), registration=reg)
            for name, values in expected.items():
                self.assertEqual(merged[name].dtype, np.int64)
                self.assertEqual([int(value) for value in merged[name]], values)  # no float64 rounding
        finally:
            shutil.rmtree(temp, ignore_errors=True)
