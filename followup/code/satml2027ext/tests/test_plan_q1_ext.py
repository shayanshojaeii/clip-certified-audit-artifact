"""V4 tests of the 021A Q1 planning file (reviewer-bundle V3 audit E5, E6; text review items 7-8).

  .venv/Scripts/python -m pytest satml2027ext/tests/test_plan_q1_ext.py -q
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from satml2027ext import make_registrations_ext as mr  # noqa: E402
from satml2027ext import plan_q1_ext as plan  # noqa: E402
from satml2027ext._common import canonical_json_hash  # noqa: E402

PLANNING = PROJECT / mr.Q1_PLANNING_PATH


def test_rounding_is_idempotent_and_keeps_exact_values_exact():
    record = {"a": 0.9499999999999998, "b": [0.03430548029427494, 1, True, None, "89/30"], "c": {"d": 2.0 / 3.0}}
    rounded = plan.round_floats(record)
    assert rounded == {"a": 0.95, "b": [0.03430548029, 1, True, None, "89/30"], "c": {"d": 0.6666666667}}
    assert plan.round_floats(rounded) == rounded
    with pytest.raises(ValueError):
        plan.round_floats({"x": float("nan")})
    # the two runtimes of the text review (0.03430548029427494 vs ...495; 0.9499999999999998 vs 0.95) now serialize alike
    assert plan.round_floats(0.03430548029427495) == plan.round_floats(0.03430548029427494)
    assert plan.round_floats(0.95) == plan.round_floats(0.9499999999999998)


def test_portable_comparison_accepts_last_digit_noise_and_refuses_real_differences():
    stored = json.loads(PLANNING.read_text(encoding="utf-8"))
    assert plan.compare_planning(stored, copy.deepcopy(stored)) == []
    noisy = copy.deepcopy(stored)
    se = noisy["saved_draws_under_registered_q1_analysis"]["macro"]["bootstrap_se_points"]
    noisy["saved_draws_under_registered_q1_analysis"]["macro"]["bootstrap_se_points"] = se * (1 + 5e-10)
    noisy["content_sha256"] = "0" * 64  # derived; not compared
    assert plan.compare_planning(stored, noisy) == []
    for mutate in (
        lambda r: r["saved_draws_under_registered_q1_analysis"]["macro"].__setitem__("bootstrap_se_points", se * 1.001),
        lambda r: r["reference_points"]["equal_cell_macro"].__setitem__("exact_points", "90/30"),
        lambda r: r["saved_draws_under_registered_q1_analysis"]["macro"].__setitem__("verdict", "inconclusive"),
        lambda r: r["reference_points"].__setitem__("positions", 3301),
        lambda r: r.pop("portability"),
    ):
        changed = copy.deepcopy(stored)
        mutate(changed)
        assert plan.compare_planning(stored, changed), mutate


def test_the_archived_planning_file_carries_the_portability_contract_and_the_sharp_bound():
    stored = json.loads(PLANNING.read_text(encoding="utf-8"))
    assert stored["schema_version"] == "satml2027ext.q1_planning.v2"
    assert canonical_json_hash({key: value for key, value in stored.items() if key != "content_sha256"}) == stored["content_sha256"]
    assert stored["portability"]["abs_tol"] == plan.PORTABLE_ABS_TOL and stored["portability"]["rel_tol"] == plan.PORTABLE_REL_TOL
    assert plan.round_floats(stored) == stored  # every float already has at most 10 significant digits
    for model in ("plug_in", "posterior_predictive"):
        sd = stored["fresh_draw_models"][model]["monte_carlo_sd_points"]
        for estimand in ("macro", "pooled"):
            assert sd["sharp_upper_bound_any_coupling"][estimand] >= sd["independent_coupling_model"][estimand] > 0
        assert "model-based" in stored["fresh_draw_models"][model]["verdict_probabilities_basis"]
    assert "upper bound on the Monte-Carlo variance" not in json.dumps(stored)
    # the audit's table (section 5.5): posterior-predictive macro 0.086093, pooled 0.077379 points
    predictive = stored["fresh_draw_models"]["posterior_predictive"]["monte_carlo_sd_points"]["sharp_upper_bound_any_coupling"]
    assert abs(predictive["macro"] - 0.086093) < 5e-7 and abs(predictive["pooled"] - 0.077379) < 5e-7


def test_sharp_bound_is_attained_by_an_extreme_coupling_and_exceeds_independence():
    for p, q in ((0.5, 0.5), (0.3, 0.6), (0.9, 0.2), (1.0, 0.0), (0.97, 0.99)):
        bound = min(p + q, 2 - p - q) - (p - q) ** 2
        # the countermonotone coupling C = 1{U < p}, I = 1{U > 1 - q} attains it
        u = (np.arange(200000) + 0.5) / 200000
        c = (u < p).astype(float)
        i = (u > 1 - q).astype(float)
        assert math.isclose(float(np.var(c - i)), bound, abs_tol=1e-4)
        assert bound >= p * (1 - p) + q * (1 - q) - 1e-12  # independence is one coupling, so never above the bound
    # the audit's example: C = 1{U < 1/2}, I = 1 - C has variance 1, the independence value is 1/2
    assert min(0.5 + 0.5, 2 - 0.5 - 0.5) - 0.0 == 1.0 and 0.5 * 0.5 + 0.5 * 0.5 == 0.5
