"""Cohen CERTIFY arithmetic and outcome metrics, torch-free (numpy + scipy).

Mirrors `phase4/certification.py` exactly: the class is selected on the
selection draws, the Clopper-Pearson lower bound is computed on the confirmation
draws for the selected class, `p_B = 1 - p_A` (CERTIFY's one-sided form), and an
example abstains when the lower bound does not exceed one half.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import beta, norm


@lru_cache(maxsize=None)
def clopper_pearson_lower(successes: int, trials: int, alpha: float) -> float:
    if trials <= 0 or not 0 <= successes <= trials or not 0.0 < alpha < 1.0:
        raise ValueError("invalid binomial-bound arguments")
    if successes == 0:
        return 0.0
    return float(beta.ppf(alpha, successes, trials - successes + 1))


def probability_threshold(radius: float, sigma: float) -> float:
    if radius < 0.0 or sigma <= 0.0:
        raise ValueError("radius must be nonnegative and sigma positive")
    return float(norm.cdf(radius / sigma))


@lru_cache(maxsize=None)
def min_successes_for_radius(trials: int, alpha: float, radius: float, sigma: float) -> int:
    """Smallest confirmation count whose exact lower bound reaches ``radius``."""

    target = probability_threshold(radius, sigma)
    low, high = 0, trials
    while low < high:
        middle = (low + high) // 2
        if clopper_pearson_lower(middle, trials, alpha) >= target:
            high = middle
        else:
            low = middle + 1
    if clopper_pearson_lower(low, trials, alpha) < target:
        raise RuntimeError("target radius is unattainable at this sample budget")
    return low


def certify_from_counts(
    *,
    raw_predictions: np.ndarray,
    ground_truth: np.ndarray,
    selection_counts: np.ndarray,
    confirmation_counts: np.ndarray,
    sigma: float,
    alpha: float,
    radii: Sequence[float],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Return (summary metrics, per-item outcome arrays) for one candidate bank."""

    if selection_counts.ndim != 2 or confirmation_counts.shape != selection_counts.shape:
        raise ValueError("counts must be matching [item, class] matrices")
    trials_by_row = confirmation_counts.sum(axis=1)
    if not np.all(trials_by_row == trials_by_row[0]):
        raise ValueError("confirmation rows have unequal sample counts")
    trials = int(trials_by_row[0])
    selected = selection_counts.argmax(axis=1)
    successes = confirmation_counts[np.arange(len(selected)), selected]
    lower = np.array([clopper_pearson_lower(int(k), trials, float(alpha)) for k in successes])
    abstained = lower <= 0.5
    radius = np.zeros_like(lower)
    valid = ~abstained
    if valid.any():
        radius[valid] = float(sigma) * norm.ppf(np.clip(lower[valid], 1e-15, 1 - 1e-15))
    correct = selected == ground_truth
    anchored = selected == raw_predictions
    summary: dict[str, float] = {
        "example_count": int(len(ground_truth)),
        "clean_accuracy": float(np.mean(raw_predictions == ground_truth)),
        "smoothed_accuracy": float(np.mean(correct)),
        "abstention_rate": float(np.mean(abstained)),
        "selection_raw_agreement": float(np.mean(anchored)),
        "average_selected_class_radius": float(np.mean(radius)),
    }
    outcomes: dict[str, np.ndarray] = {
        "selected_classes": selected,
        "selected_successes": successes,
        "probability_lowers": lower,
        "selected_class_radii": radius,
        "abstained": abstained,
    }
    for value in radii:
        key = format(float(value), ".12g")
        reaches = valid & (radius >= float(value))
        summary[f"standard_certified_accuracy@{key}"] = float(np.mean(reaches & correct))
        summary[f"anchored_certified_accuracy@{key}"] = float(np.mean(reaches & correct & anchored))
        summary[f"certified_wrong_rate@{key}"] = float(np.mean(reaches & ~correct))
        outcomes[f"standard@{key}"] = reaches & correct
        outcomes[f"wrong_stable@{key}"] = reaches & ~correct
    return summary, outcomes


def prediction_concentration(selected: np.ndarray, class_count: int) -> dict[str, float]:
    """Hub diagnostics of the smoothed predictions (Sato-comparable)."""

    counts = np.bincount(selected, minlength=class_count).astype(np.float64)
    share = counts / counts.sum()
    ordered = np.sort(share)[::-1]
    positive = share[share > 0]
    entropy = float(-(positive * np.log(positive)).sum() / np.log(class_count)) if class_count > 1 else 0.0
    sorted_counts = np.sort(counts)
    index = np.arange(1, class_count + 1)
    gini = float((2.0 * (index * sorted_counts).sum()) / (class_count * sorted_counts.sum()) - (class_count + 1) / class_count)
    return {
        "top_class": int(share.argmax()),
        "top_class_share": float(ordered[0]),
        "classes_for_half": int(np.searchsorted(np.cumsum(ordered), 0.5) + 1),
        "classes_predicted": int((counts > 0).sum()),
        "predicted_class_gini": gini,
        "normalized_prediction_entropy": entropy,
    }


def macro_cell_mean(values: Mapping[str, float]) -> float:
    return float(np.mean(list(values.values())))
