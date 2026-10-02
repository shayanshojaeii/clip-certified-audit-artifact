import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest


SCRIPT = Path("analysis/negative_paper_gate_n3.py")
SPEC = importlib.util.spec_from_file_location("negative_paper_gate_n3", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
import sys
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _config():
    return json.loads(Path("configs/negative_paper_gate_n3_theory_v1.json").read_text(encoding="utf-8"))


def test_frozen_config_is_non_authorizing_and_has_three_witnesses():
    config = _config()
    MODULE.validate_config(config)
    assert len(config["parameter_cases"]) == 3
    assert all(value is False for value in config["authority"].values())


@pytest.mark.parametrize("case", _config()["parameter_cases"], ids=lambda case: case["id"])
def test_exact_invariants_and_population_radii(case):
    result = MODULE.analyze_case(case)
    checks = MODULE.verify_result(result, 1e-12)
    assert "same complete global gap" in checks
    assert "different Cohen radii" in checks
    assert result["population_cohen_radii"][0] == pytest.approx(result["q"], abs=1e-12)
    assert result["population_cohen_radii"][1] == pytest.approx(
        result["q"] / case["kappa"], abs=1e-12
    )


@pytest.mark.parametrize("case", _config()["parameter_cases"], ids=lambda case: case["id"])
def test_decision_boundaries_follow_from_normalized_scores(case):
    result = MODULE.analyze_case(case)
    q = result["q"]
    kappa = case["kappa"]
    epsilon = min(q, q / kappa) / 10.0
    assert q + (-q + epsilon) > 0.0
    assert q + (-q - epsilon) < 0.0
    assert q + kappa * (-q / kappa + epsilon) > 0.0
    assert q + kappa * (-q / kappa - epsilon) < 0.0


def test_kappa_controls_radius_order_without_changing_clean_invariants():
    high, low = _config()["parameter_cases"][:2]
    high_result = MODULE.analyze_case(high)
    low_result = MODULE.analyze_case(low)
    assert high["kappa"] > 1 and high_result["radius_difference"] > 0
    assert low["kappa"] < 1 and low_result["radius_difference"] < 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gamma", 0.0),
        ("gamma", 1.0),
        ("b", 0.0),
        ("d", 0.0),
        ("kappa", 0.0),
        ("kappa", 1.0),
        ("sigma", 0.0),
    ],
)
def test_invalid_or_vacuous_parameter_cases_fail_closed(field, value):
    case = copy.deepcopy(_config()["parameter_cases"][0])
    case[field] = value
    with pytest.raises(ValueError):
        MODULE.validate_case(case)


def test_encoder_is_globally_defined_and_derivative_bound_is_finite():
    case = _config()["parameter_cases"][0]
    result = MODULE.analyze_case(case)
    z0 = tuple(result["z0"])
    kappa = case["kappa"]
    bound = result["encoder_global_lipschitz_bound"]
    for u in (-1e6, -10.0, -1.0, 0.0, 1.0, 10.0, 1e6):
        g = (z0[0] + u, z0[1] + kappa * u, z0[2])
        assert MODULE.norm(g) >= result["encoder_nonzero_floor"]
        f = MODULE.normalize(g)
        v = (1.0, kappa, 0.0)
        projection = MODULE.dot(f, v)
        derivative = tuple((v_i - projection * f_i) / MODULE.norm(g) for v_i, f_i in zip(v, f, strict=True))
        assert MODULE.norm(derivative) <= bound + 1e-12


def test_candidate_latex_keeps_certificate_engines_separated():
    text = Path("paper/generated/negative_paper_gate_n3_theory_candidate_v1.tex").read_text(
        encoding="utf-8"
    )
    proposition = text.split("% END COHEN PROPOSITION", 1)[0]
    for forbidden in ("D^*", "tau^*", "FCSB", "Smoothed Embeddings"):
        assert forbidden not in proposition
    assert "population" in proposition.lower()
    assert "finite-sample" in text.lower()


def test_full_primary_verifier_checks_all_bound_sources():
    result = MODULE.run(Path("configs/negative_paper_gate_n3_theory_v1.json").resolve())
    assert result["case_count"] == 3
    assert result["source_hash_checks"] == 4
    assert result["sampling_performed"] is False
    assert result["scientific_data_accessed"] is False
    assert result["gate_n3_independently_verified"] is False
