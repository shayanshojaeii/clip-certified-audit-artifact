"""Encoder-level Monte Carlo check of the Gate-N3 Cohen non-identification witness.

Reads only the frozen synthetic cases in
``configs/negative_paper_gate_n3_theory_v1.json`` (see
``research/NEGATIVE_PAPER_GATE_N3_COHEN_NONIDENTIFICATION.md``).  The Gaussian
draws are synthetic scalar perturbations of the analytic encoder
``f(u) = Normalize(z0 + u (e1 + kappa e2))``; they are not randomized-smoothing
samples of any CLIP classifier, no registered noise stream is touched, no
model or dataset is read, and nothing here is a certificate.  The population
radii reconstructed below are plug-in population quantities, not finite-sample
Cohen certificates.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest

CONFIG_PATH = Path("configs/negative_paper_gate_n3_theory_v1.json")
DRAWS = 200_000
SEED = 20260925
STANDARD_ERROR_MULTIPLIER = 4.0


def _phi(x: float) -> float:
    """Standard normal CDF through math.erf."""

    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _construction(case: dict) -> dict:
    gamma, b, d = float(case["gamma"]), float(case["b"]), float(case["d"])
    kappa, sigma = float(case["kappa"]), float(case["sigma"])
    s = math.sqrt(1.0 - gamma * gamma)
    scale = math.sqrt(2.0 * b * b + d * d)
    q = b / scale
    z0 = np.array([b, b, d]) / scale
    direction = np.array([1.0, kappa, 0.0])
    bank_1 = np.array([[s, 0.0, gamma], [-s, 0.0, gamma]])  # t1+, t1-
    bank_2 = np.array([[0.0, s, gamma], [0.0, -s, gamma]])  # t2+, t2-
    return {
        "q": q,
        "kappa": kappa,
        "sigma": sigma,
        "z0": z0,
        "direction": direction,
        "banks": (bank_1, bank_2),
        "population_probabilities": (_phi(q / sigma), _phi(q / (kappa * sigma))),
        "population_radii": (q, q / kappa),
    }


def _encode(z0: np.ndarray, direction: np.ndarray, u: np.ndarray) -> np.ndarray:
    raw = z0[None, :] + u[:, None] * direction[None, :]
    norms = np.linalg.norm(raw, axis=1)
    assert np.all(norms > 0.0)
    return raw / norms[:, None]


def _positive(features: np.ndarray, bank: np.ndarray) -> np.ndarray:
    scores = features @ bank.T
    return scores[:, 0] > scores[:, 1]


def test_frozen_config_is_non_authorizing_and_unchanged_in_shape() -> None:
    config = _config()
    assert config["proposition_id"] == "N3.COHEN.NONIDENTIFICATION.V1"
    assert len(config["parameter_cases"]) == 3
    assert all(value is False for value in config["authority"].values())
    assert config["construction"]["encoder"] == "f(u)=Normalize(z0+u(e1+kappa*e2))"


@pytest.mark.parametrize("index_and_case", list(enumerate(_config()["parameter_cases"])), ids=lambda pair: pair[1]["id"])
def test_empirical_class_probabilities_match_phi(index_and_case: tuple[int, dict]) -> None:
    index, case = index_and_case
    built = _construction(case)
    rng = np.random.default_rng(SEED + index)
    u = rng.normal(0.0, built["sigma"], DRAWS)
    features = _encode(built["z0"], built["direction"], u)
    assert np.max(np.abs(np.linalg.norm(features, axis=1) - 1.0)) <= 1e-12

    for bank, p_population, boundary_slope in zip(
        built["banks"], built["population_probabilities"], (1.0, built["kappa"]), strict=True
    ):
        positive = _positive(features, bank)
        # the normalized score difference is 2 s (q + slope u) / ||raw||, so the hard
        # label is sign(q + slope u) except at numerical ties of measure zero
        argument = built["q"] + boundary_slope * u
        clear = np.abs(argument) > 1e-9
        assert np.array_equal(positive[clear], argument[clear] > 0.0)
        p_hat = float(positive.mean())
        standard_error = math.sqrt(p_population * (1.0 - p_population) / DRAWS)
        assert abs(p_hat - p_population) <= STANDARD_ERROR_MULTIPLIER * standard_error, (
            case["id"],
            p_hat,
            p_population,
            standard_error,
        )

    # the two banks differ by far more than Monte Carlo error on the same draws
    p1_hat = float(_positive(features, built["banks"][0]).mean())
    p2_hat = float(_positive(features, built["banks"][1]).mean())
    assert abs(p1_hat - p2_hat) > 20.0 * math.sqrt(0.25 / DRAWS)
    assert (p1_hat > p2_hat) == (built["kappa"] > 1.0)


@pytest.mark.parametrize("index_and_case", list(enumerate(_config()["parameter_cases"])), ids=lambda pair: pair[1]["id"])
def test_plug_in_population_radii_match_closed_form(index_and_case: tuple[int, dict]) -> None:
    """sigma * Phi^{-1}(p_hat) reconstructs q and q/kappa within a delta-method band (not a certificate)."""

    index, case = index_and_case
    built = _construction(case)
    rng = np.random.default_rng(SEED + index)
    u = rng.normal(0.0, built["sigma"], DRAWS)
    features = _encode(built["z0"], built["direction"], u)
    normal = NormalDist()
    for bank, p_population, radius in zip(built["banks"], built["population_probabilities"], built["population_radii"], strict=True):
        p_hat = float(_positive(features, bank).mean())
        plug_in_radius = built["sigma"] * normal.inv_cdf(p_hat)
        standard_error_p = math.sqrt(p_population * (1.0 - p_population) / DRAWS)
        density = normal.pdf(normal.inv_cdf(p_population))
        standard_error_radius = built["sigma"] * standard_error_p / density
        assert abs(plug_in_radius - radius) <= STANDARD_ERROR_MULTIPLIER * standard_error_radius


@pytest.mark.parametrize("case", _config()["parameter_cases"], ids=lambda case: case["id"])
def test_two_point_reference_set_gives_identical_gap_vectors_and_correct_labels(case: dict) -> None:
    built = _construction(case)
    q, kappa = built["q"], built["kappa"]
    reference_distance = 2.0 * max(q, q / kappa)  # L > max(q, q / kappa)
    assert reference_distance > max(q, q / kappa)
    u = np.array([0.0, -reference_distance])  # u_plus = 0 (positive label), u_minus = -L (negative label)
    features = _encode(built["z0"], built["direction"], u)

    # the encoder and the reference set are shared, so the class-balanced image mean
    # is one object for both banks
    image_mean = features.mean(axis=0)
    assert np.allclose(image_mean, 0.5 * (features[0] + features[1]))

    bank_1, bank_2 = built["banks"]
    prototype_mean_1, prototype_mean_2 = bank_1.mean(axis=0), bank_2.mean(axis=0)
    assert np.array_equal(prototype_mean_1, prototype_mean_2)
    gap_1 = prototype_mean_1 - image_mean
    gap_2 = prototype_mean_2 - image_mean
    assert np.array_equal(gap_1, gap_2)  # identical complete global-gap vectors, bit for bit

    labels = np.array([True, False])  # positive at u_plus, negative at u_minus
    for bank in (bank_1, bank_2):
        assert np.array_equal(_positive(features, bank), labels)

    # same clean margin and prototype correlation at the certified query u = 0
    clean = features[0]
    assert math.isclose(clean @ (bank_1[0] - bank_1[1]), clean @ (bank_2[0] - bank_2[1]), abs_tol=1e-12)
    assert math.isclose(bank_1[0] @ bank_1[1], bank_2[0] @ bank_2[1], abs_tol=1e-12)
    assert clean @ (bank_1[0] - bank_1[1]) > 0.0
