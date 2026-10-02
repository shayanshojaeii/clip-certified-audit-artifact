"""V4 count-contract tests (reviewer-bundle V3 audit, finding E3), plus strict JSON (E2).

Each audit probe that the V3 custody stage accepted is replayed here and must be refused.

  .venv/Scripts/python -m pytest satml2027ext/tests/test_contract_ext.py -q
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from satml2027ext import candidates_ext, contract_ext  # noqa: E402
from satml2027ext._common import NonFiniteJSONError, read_json  # noqa: E402

IDS = candidates_ext.canonical_candidate_ids()
CLASSES = 10
ITEMS = 6
SELECTION, CONFIRMATION = 128, 4096
CELL = {"cell_id": "mA__dsA__sigma0.25", "class_count": CLASSES, "sigma": 0.25}
REGISTRATION = {"experiment_id": "EXP-20260921-021B", "candidates": candidates_ext.canonical_candidates_block()}
CERTIFICATION = {"selection_draws": SELECTION, "confirmation_draws": CONFIRMATION, "budget_checkpoints": []}
THEOREM = set(REGISTRATION["candidates"]["flip_theorem_candidates"])


def shard(ids=IDS, *, checkpoints=()) -> dict[str, np.ndarray]:
    count = len(ids)
    labels = np.arange(ITEMS) % CLASSES
    selection = np.zeros((count, ITEMS, CLASSES), dtype=np.int32)
    selection[:, np.arange(ITEMS), labels] = SELECTION
    confirmation = np.zeros_like(selection)
    confirmation[:, np.arange(ITEMS), labels] = CONFIRMATION - 100
    confirmation[:, np.arange(ITEMS), (labels + 1) % CLASSES] = 100
    applicable = np.array([candidate in THEOREM for candidate in ids])
    counters = {name: np.zeros((count, ITEMS), dtype=np.int32) for name in contract_ext.COUNTER_NAMES}
    step = ids.index("clean_boundary_active__step_0.16")
    counters["flip_changed_draws"][step] = 40
    counters["flip_useful_draws"][step] = 10
    counters["flip_harmful_draws"][step] = 5
    counters["flip_loose_budget_draws"][step] = 300
    data = dict(item_ids=np.array([f"i{index}" for index in range(ITEMS)]), ground_truth=labels.astype(np.int64),
                raw_predictions=np.tile(labels, (count, 1)).astype(np.int64), selection_counts=selection,
                confirmation_counts=confirmation, selection_seeds=np.arange(ITEMS, dtype=np.int64) + 7,
                confirmation_seeds=np.arange(ITEMS, dtype=np.int64) + 11, done=np.ones(ITEMS, dtype=bool),
                flip_containment_applicable=applicable.copy(), flip_loose_applicable=applicable.copy(), **counters)
    if checkpoints:
        prefix = np.zeros((len(checkpoints), count, ITEMS, CLASSES), dtype=np.int32)
        for index, checkpoint in enumerate(checkpoints):
            prefix[index, :, np.arange(ITEMS), labels] = checkpoint - 10
            prefix[index, :, np.arange(ITEMS), (labels + 1) % CLASSES] = 10
        data["prefix_confirmation_counts"] = prefix
        data["budget_checkpoints"] = np.array(list(checkpoints), dtype=np.int64)
    return data


def check(data, *, certification=CERTIFICATION, ids=IDS, registration=REGISTRATION):
    return contract_ext.check_shard(data, registration=registration, cell=CELL, candidate_ids=list(ids),
                                    certification=certification, where="probe")


def test_a_valid_shard_passes_and_returns_its_flags():
    flags = check(shard())
    assert flags["flip_containment_applicable"].sum() == len(THEOREM)


def mutate(function):
    data = shard()
    function(data)
    return data


PROBES = {
    "fractional_counts": lambda d: d.update(confirmation_counts=d["confirmation_counts"].astype(np.float64) + 0.25),
    "integral_float_counts": lambda d: d.update(selection_counts=d["selection_counts"].astype(np.float64)),
    "negative_counts_sum_preserved": lambda d: (d["confirmation_counts"].__setitem__((0, 0, 2), -5),
                                                d["confirmation_counts"].__setitem__((0, 0, 3), 5)),
    "changed_above_budget": lambda d: d["flip_changed_draws"].__setitem__((IDS.index("clean_boundary_active__step_0.16"), 0),
                                                                          CONFIRMATION + 1),
    "harmful_above_changed": lambda d: d["flip_harmful_draws"].__setitem__((IDS.index("clean_boundary_active__step_0.16"), 0), 39),
    "containment_violation_above_changed": lambda d: d["flip_containment_violations"].__setitem__(
        (IDS.index("clean_boundary_active__step_0.16"), 1), 41),
    "loose_budget_above_draws": lambda d: d["flip_loose_budget_draws"].__setitem__(
        (IDS.index("clean_boundary_active__step_0.16"), 1), CONFIRMATION + 1),
    "all_applicability_false": lambda d: (d["flip_containment_applicable"].fill(False), d["flip_loose_applicable"].fill(False)),
    "applicability_on_an_image_side_bank": lambda d: d["flip_containment_applicable"].__setitem__(
        IDS.index("gr_clip_style_two_sided__coefficient_1"), True),
    "loose_without_containment": lambda d: d["flip_containment_applicable"].__setitem__(
        IDS.index("clean_boundary_active__step_0.16"), False),
    "violations_where_not_applicable": lambda d: (d["flip_changed_draws"].__setitem__((IDS.index("control__lowrank_tangent_r8"), 0), 3),
                                                  d["flip_containment_violations"].__setitem__((IDS.index("control__lowrank_tangent_r8"), 0), 1)),
    "identity_with_changes": lambda d: d["flip_changed_draws"].__setitem__((IDS.index("no_correction"), 0), 1),
    "non_boolean_flags": lambda d: d.update(flip_loose_applicable=d["flip_loose_applicable"].astype(np.int64)),
    "label_out_of_range": lambda d: d["raw_predictions"].__setitem__((0, 0), CLASSES),
    "selection_total_wrong": lambda d: d["selection_counts"].__setitem__((1, 1, 0), 1),
    "unregistered_prefix": lambda d: d.update(prefix_confirmation_counts=np.zeros((1, len(IDS), ITEMS, CLASSES), dtype=np.int32)),
    "missing_selection_seeds": lambda d: d.pop("selection_seeds"),
    "incomplete": lambda d: d["done"].__setitem__(0, False),
}


@pytest.mark.parametrize("name", sorted(PROBES))
def test_each_audit_probe_is_refused(name):
    with pytest.raises(contract_ext.CountContractError):
        check(mutate(PROBES[name]))


# V4.1: the six integer/overflow mutations of the reviewer-bundle V5 audit (probe_exp021_overflow.py), each of
# which the V4 contract accepted, plus two seed-domain probes.  Every one must be refused.
UINT64_MAX, INT64_MAX = int(np.iinfo(np.uint64).max), int(np.iinfo(np.int64).max)
STEP = IDS.index("clean_boundary_active__step_0.16")


def _selection_uint64_wrap(d):
    d["selection_counts"] = d["selection_counts"].astype(np.uint64)
    d["selection_counts"][0, 0, :] = 0
    d["selection_counts"][0, 0, 0] = UINT64_MAX  # UINT64_MAX + 129 wraps to the registered 128
    d["selection_counts"][0, 0, 1] = 129


def _selection_int64_sum_overflow(d):
    d["selection_counts"] = d["selection_counts"].astype(np.int64)
    d["selection_counts"][0, 0, :] = 0
    d["selection_counts"][0, 0, 0] = INT64_MAX  # 2 * INT64_MAX + 130 wraps to 128
    d["selection_counts"][0, 0, 1] = INT64_MAX
    d["selection_counts"][0, 0, 2] = 130


def _confirmation_int64_sum_overflow(d):
    d["confirmation_counts"] = d["confirmation_counts"].astype(np.int64)
    d["confirmation_counts"][0, 0, :] = 0
    d["confirmation_counts"][0, 0, 0] = INT64_MAX  # 2 * INT64_MAX + 4098 wraps to 4096
    d["confirmation_counts"][0, 0, 1] = INT64_MAX
    d["confirmation_counts"][0, 0, 2] = 4098


def _raw_prediction_uint64_wrap(d):
    d["raw_predictions"] = d["raw_predictions"].astype(np.uint64)
    d["raw_predictions"][0, 0] = UINT64_MAX


def _ground_truth_uint64_wrap(d):
    d["ground_truth"] = d["ground_truth"].astype(np.uint64)
    d["ground_truth"][0] = UINT64_MAX


def _counter_addition_overflow(d):
    for name in ("flip_useful_draws", "flip_harmful_draws"):
        d[name] = d[name].astype(np.int64)
        d[name][STEP, 0] = INT64_MAX  # U + H wraps to -2, below the changed draws


def _seed_above_63_bits(d):
    d["selection_seeds"] = d["selection_seeds"].astype(np.uint64)
    d["selection_seeds"][0] = 1 << 63  # wraps negative in int64; seeds are 63-bit


def _negative_seed(d):
    d["confirmation_seeds"][0] = -1


def _timedelta_counts(d):
    d["selection_counts"] = d["selection_counts"].astype("m8[ns]")  # a numpy signed-integer subclass (re-review)


def _zero_d_item_ids(d):
    d["item_ids"] = np.array("i0")


def _dataset_index_wrap(d):
    d["dataset_indices"] = np.full(ITEMS, UINT64_MAX, dtype=np.uint64)


def _negative_item_offset(d):
    d["item_offset"] = np.array([-1])


OVERFLOW_PROBES = {
    "timedelta_counts": _timedelta_counts,
    "zero_d_item_ids": _zero_d_item_ids,
    "dataset_index_wrap": _dataset_index_wrap,
    "negative_item_offset": _negative_item_offset,
    "selection_uint64_cast_wrap": _selection_uint64_wrap,
    "selection_int64_sum_overflow": _selection_int64_sum_overflow,
    "confirmation_int64_sum_overflow": _confirmation_int64_sum_overflow,
    "raw_prediction_uint64_cast_wrap": _raw_prediction_uint64_wrap,
    "ground_truth_uint64_cast_wrap": _ground_truth_uint64_wrap,
    "counter_addition_overflow": _counter_addition_overflow,
    "seed_above_63_bits": _seed_above_63_bits,
    "negative_seed": _negative_seed,
}


@pytest.mark.parametrize("name", sorted(OVERFLOW_PROBES))
def test_v41_each_overflow_probe_is_refused(name):
    with pytest.raises(contract_ext.CountContractError):
        check(mutate(OVERFLOW_PROBES[name]))


def test_v41_in_range_unsigned_storage_still_passes():
    data = shard()
    for name in ("selection_counts", "confirmation_counts", *contract_ext.COUNTER_NAMES):
        data[name] = data[name].astype(np.uint16)
    for name in ("raw_predictions", "ground_truth", "selection_seeds", "confirmation_seeds"):
        data[name] = data[name].astype(np.uint64)
    check(data)


def test_prefix_contract_for_budget_checkpoints():
    budget = dict(CERTIFICATION, confirmation_draws=CONFIRMATION, budget_checkpoints=[1024])
    check(shard(checkpoints=(1024,)), certification=budget)
    wrong_label = shard(checkpoints=(1024,))
    wrong_label["budget_checkpoints"] = np.array([2048], dtype=np.int64)
    zero_totals = shard(checkpoints=(1024,))
    zero_totals["prefix_confirmation_counts"][:] = 0
    above_full = shard(checkpoints=(1024,))
    above_full["prefix_confirmation_counts"][0, 0, 0, :] = 0
    above_full["prefix_confirmation_counts"][0, 0, 0, 5] = 1024  # the full stream has no vote for class 5
    missing = shard()
    for data in (wrong_label, zero_totals, above_full, missing):
        with pytest.raises(contract_ext.CountContractError):
            check(data, certification=budget)
    two = dict(budget, budget_checkpoints=[1024, 2048])
    decreasing = shard(checkpoints=(1024, 2048))
    decreasing["prefix_confirmation_counts"][1, 0, 0, :] = 0
    decreasing["prefix_confirmation_counts"][1, 0, 0, 9] = 2048
    with pytest.raises(contract_ext.CountContractError):
        check(decreasing, certification=two)


def test_flags_must_agree_across_shards():
    first = check(shard())
    other = copy.deepcopy(first)
    other["flip_loose_applicable"][IDS.index("control__learned_shared_translation_tangent")] = True
    with pytest.raises(contract_ext.CountContractError):
        contract_ext.check_flags_agree(first, other, "shard 1")
    contract_ext.check_flags_agree(first, copy.deepcopy(first), "shard 1")


def test_the_exploratory_tangent_control_is_only_type_checked():
    data = shard()
    position = IDS.index("control__learned_shared_translation_tangent")
    data["flip_containment_applicable"][position] = True
    data["flip_changed_draws"][position] = 7
    data["flip_containment_violations"][position] = 2
    check(data)


def test_021c_subset_applicability_follows_the_registered_operator_classes():
    subset = list(candidates_ext.BUDGET_SUBSET_021C)
    block = candidates_ext.subset_candidates_block(candidates_ext.BUDGET_SUBSET_021C, parent_experiment_id="EXP-20260921-021B",
                                                   parent_registration_sha256="0" * 64)
    registration = {"experiment_id": "EXP-20260921-021C", "candidates": block}
    required = contract_ext.registered_applicability(registration, CELL, subset)
    assert required == {"no_correction": (False, False), "gr_clip_style_two_sided__coefficient_1": (False, False),
                        "cohen_aligned_noisy_margin__step_0.16": (True, True)}


def test_strict_json_refuses_nan_and_infinity(tmp_path):
    for token in ("NaN", "Infinity", "-Infinity"):
        path = tmp_path / f"{token.strip('-')}.json"
        path.write_text('{"a": 1, "b": ' + token + "}", encoding="utf-8")
        with pytest.raises(NonFiniteJSONError):
            read_json(path)
    path = tmp_path / "ok.json"
    path.write_text('{"a": 1.5e300, "b": -0.0}', encoding="utf-8")
    assert read_json(path) == {"a": 1.5e300, "b": -0.0}


def test_v41_strict_json_refuses_literals_that_overflow_to_infinity(tmp_path):
    for token in ("1e999", "-1e999", "1.8e308"):
        path = tmp_path / "overflow.json"
        path.write_text('{"a": ' + token + "}", encoding="utf-8")
        with pytest.raises(NonFiniteJSONError):
            read_json(path)
