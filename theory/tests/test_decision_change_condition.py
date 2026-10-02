"""Numerical verification of ``research/DECISION_CHANGE_CONDITION.md``.

Scope: synthetic random geometry only.  No model, dataset, saved result, or
registered noise stream is read, and nothing computed here is a certificate.

Checked statements (numbering follows the note):

* Proposition 1, per-draw identity for a renormalized common text translation
  ``t'_k = (t_k + w) / n_k`` with ``n_k = ||t_k + w||``:
  ``score'_k = (s_k + c) / n_k`` and the exact pairwise margin formula.
* Proposition 2, necessary decision-change condition, on every observed
  argmax flip over more than 10,000 random ``(z, w)`` trials:
  ``s_j - s_k <= (n_j - n_k) z^T t'_j <= |n_j - n_k| <= ||w|| ||t_j - t_k||``.
* Lemma (normalizer difference), including adversarial translations for which
  ``n_j + n_k < 2`` so that the naive Cauchy-Schwarz route would not suffice,
  and the exact tightness example.
* Corollary 1, a translation orthogonal to every prototype difference
  (Chowers-style exact projection) never changes an argmax.
* Corollary 2, the finite flip budget, against brute-force vote counts.
* The two-sided remark: a GR-CLIP-style two-sided centering can flip a draw
  whose original margin exceeds every text-only bound (existence).
* The per-class-scale extension used by the registered N3C P1 statement for
  tangent controls.
"""

from __future__ import annotations

import numpy as np
import pytest

DIM = 512
CLASS_COUNTS = (10, 100)
TRANSLATION_NORMS = (0.0025, 0.04, 0.5, 2.0)
TRIALS_PER_CELL = 1500  # 2 class counts x 4 norms x 1500 = 12,000 (z, w) trials
SEED = 20260925
ROUNDOFF = 1e-12


# --------------------------------------------------------------------------- helpers


def _unit_rows(rng: np.random.Generator, count: int, dim: int = DIM) -> np.ndarray:
    rows = rng.standard_normal((count, dim))
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _translate(prototypes: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact operator of the frozen grid: t'_k = (t_k + w) / ||t_k + w||."""

    raw = prototypes + w[None, :]
    norms = np.linalg.norm(raw, axis=1)
    assert np.all(norms > 1e-12)
    return raw / norms[:, None], norms


def _near_tie_feature(
    rng: np.random.Generator,
    prototypes: np.ndarray,
    j: int,
    k: int,
    margin: float,
    perpendicular_noise: float = 0.1,
) -> np.ndarray:
    """Unit z whose two leading classes are j and k with s_j - s_k close to ``margin``.

    With d = t_j - t_k, (t_j + t_k)^T d = 0 for unit prototypes, so the signed
    margin of normalize(t_j + t_k + tau d_hat + eta) with eta orthogonal to d is
    tau ||d|| / ||.||.  ``margin`` may be negative.
    """

    diff = prototypes[j] - prototypes[k]
    diff_unit = _unit(diff)
    base = prototypes[j] + prototypes[k]
    eta = rng.standard_normal(prototypes.shape[1])
    eta -= (eta @ diff_unit) * diff_unit
    eta *= perpendicular_noise / np.linalg.norm(eta)
    tau = margin * np.linalg.norm(base) / np.linalg.norm(diff)
    return _unit(base + tau * diff_unit + eta)


def _random_translation(rng: np.random.Generator, norm: float) -> np.ndarray:
    return norm * _unit(rng.standard_normal(DIM))


def _adversarial_translation(
    rng: np.random.Generator, prototypes: np.ndarray, j: int, k: int, norm: float
) -> np.ndarray:
    """Translation mostly along t_j - t_k, which lengthens t_j more than t_k."""

    direction = _unit(prototypes[j] - prototypes[k]) + 0.3 * rng.standard_normal(DIM) / np.sqrt(DIM)
    return norm * _unit(direction)


# --------------------------------------------------------------------------- Proposition 1


@pytest.mark.parametrize("class_count", CLASS_COUNTS)
@pytest.mark.parametrize("norm", TRANSLATION_NORMS)
def test_per_draw_identity_is_exact(class_count: int, norm: float) -> None:
    rng = np.random.default_rng(SEED + class_count)
    prototypes = _unit_rows(rng, class_count)
    features = _unit_rows(rng, 2000)
    w = _random_translation(rng, norm)
    translated, n = _translate(prototypes, w)

    scores = features @ prototypes.T  # s_k = z^T t_k
    shift = features @ w  # c = z^T w
    direct = features @ translated.T
    predicted = (scores + shift[:, None]) / n[None, :]
    assert np.max(np.abs(direct - predicted)) <= ROUNDOFF

    # exact pairwise transformed margin for every ordered pair (j, k)
    s_j = scores[:, :, None]
    s_k = scores[:, None, :]
    c = shift[:, None, None]
    margin_formula = (n[None, None, :] * (s_j + c) - n[None, :, None] * (s_k + c)) / (
        n[None, :, None] * n[None, None, :]
    )
    margin_direct = direct[:, :, None] - direct[:, None, :]
    assert np.max(np.abs(margin_direct - margin_formula)) <= ROUNDOFF

    # n_j^2 - n_k^2 = 2 (t_j - t_k)^T w exactly, hence n_j - n_k = 2 (t_j - t_k)^T w / (n_j + n_k)
    squared_difference = n[:, None] ** 2 - n[None, :] ** 2
    pair_dot = 2.0 * ((prototypes[:, None, :] - prototypes[None, :, :]) @ w)
    assert np.max(np.abs(squared_difference - pair_dot)) <= 1e-11
    ratio = pair_dot / (n[:, None] + n[None, :])
    assert np.max(np.abs((n[:, None] - n[None, :]) - ratio)) <= 1e-11


# --------------------------------------------------------------------------- Lemma


def test_normalizer_difference_lemma_including_adversarial_translations() -> None:
    """|n_j - n_k| <= min(1, ||w||) ||t_j - t_k||, also when n_j + n_k < 2."""

    rng = np.random.default_rng(SEED + 11)
    prototypes = _unit_rows(rng, 10)
    checked = 0
    small_denominator_cases = 0
    for _ in range(400):
        j, k = rng.choice(10, size=2, replace=False)
        t_j, t_k = prototypes[j], prototypes[k]
        difference = np.linalg.norm(t_j - t_k)
        bisector = _unit(t_j + t_k)
        candidates = [_random_translation(rng, norm) for norm in TRANSLATION_NORMS]
        candidates += [-beta * t_k for beta in np.linspace(0.05, 2.0, 20)]
        candidates += [-beta * t_j for beta in np.linspace(0.05, 2.0, 20)]
        candidates += [-t_k + lam * bisector for lam in np.linspace(0.0, 1.5, 16)]
        candidates += [-beta * bisector for beta in np.linspace(0.05, 2.0, 10)]
        for w in candidates:
            n_j, n_k = np.linalg.norm(t_j + w), np.linalg.norm(t_k + w)
            if min(n_j, n_k) <= 1e-9:
                continue  # the operator is undefined at a zero raw norm
            gap = abs(n_j - n_k)
            assert gap <= np.linalg.norm(w) * difference + ROUNDOFF
            assert gap <= difference + ROUNDOFF
            assert abs(n_j**2 - n_k**2 - 2.0 * (t_j - t_k) @ w) <= 1e-10
            if n_j + n_k < 2.0 - 1e-9:
                small_denominator_cases += 1
            checked += 1
    assert checked > 10000
    # The naive route 2|(t_j - t_k)^T w| / (n_j + n_k) <= ||w|| ||t_j - t_k|| needs
    # n_j + n_k >= 2, which fails on many of the translations above; the inversion
    # (Ptolemy) argument in the note does not need it.
    assert small_denominator_cases > 100


def test_normalizer_difference_lemma_is_tight() -> None:
    """Equality case: unit a, b with a - b of length D, w = -b + lam m_hat, lam at tangency."""

    rng = np.random.default_rng(SEED + 12)
    m_hat = _unit(rng.standard_normal(DIM))
    d_hat = rng.standard_normal(DIM)
    d_hat -= (d_hat @ m_hat) * m_hat
    d_hat = _unit(d_hat)
    half_length, height = 0.6, 0.8  # D = 1.2, M = 0.8, M^2 + (D/2)^2 = 1
    a = height * m_hat + half_length * d_hat
    b = height * m_hat - half_length * d_hat
    assert abs(np.linalg.norm(a) - 1.0) <= ROUNDOFF and abs(np.linalg.norm(b) - 1.0) <= ROUNDOFF
    w = -b + 0.35 * m_hat
    gap = abs(np.linalg.norm(a + w) - np.linalg.norm(b + w))
    bound = np.linalg.norm(w) * np.linalg.norm(a - b)
    assert gap <= bound + ROUNDOFF
    assert abs(gap - bound) <= 1e-9  # attained, so the factor ||w|| cannot be improved
    assert np.linalg.norm(a + w) + np.linalg.norm(b + w) < 2.0  # outside the naive route


# --------------------------------------------------------------------------- Proposition 2


def test_necessary_condition_holds_on_every_observed_flip() -> None:
    rng = np.random.default_rng(SEED + 2)
    total_trials = 0
    flips_by_cell: dict[tuple[int, float], int] = {}
    violations = 0
    positive_score_flips_with_wrong_normalizer_order = 0
    for class_count in CLASS_COUNTS:
        prototypes = _unit_rows(rng, class_count)
        pair_distance = np.linalg.norm(
            prototypes[:, None, :] - prototypes[None, :, :], axis=2
        )
        for norm in TRANSLATION_NORMS:
            flips = 0
            for trial in range(TRIALS_PER_CELL):
                j0, k0 = rng.choice(class_count, size=2, replace=False)
                mode = trial % 3
                if mode == 0:
                    z = _unit(rng.standard_normal(DIM))
                    w = _random_translation(rng, norm)
                elif mode == 1:
                    z = _near_tie_feature(rng, prototypes, j0, k0, rng.uniform(-1.5, 1.5) * norm)
                    w = _random_translation(rng, norm)
                else:
                    z = _near_tie_feature(rng, prototypes, j0, k0, rng.uniform(-3.0, 3.0) * norm)
                    w = _adversarial_translation(rng, prototypes, j0, k0, norm)
                total_trials += 1

                scores = prototypes @ z
                j = int(np.argmax(scores))
                translated, n = _translate(prototypes, w)
                new_scores = translated @ z
                k = int(np.argmax(new_scores))
                if k == j:
                    continue
                flips += 1
                original_pair_margin = scores[j] - scores[k]
                assert original_pair_margin >= -ROUNDOFF  # j was the original pair winner
                normalizer_gap = n[j] - n[k]
                exact_form_old_winner = normalizer_gap * new_scores[j]
                exact_form_new_winner = normalizer_gap * new_scores[k]
                bound_chain_ok = (
                    original_pair_margin <= exact_form_old_winner + ROUNDOFF
                    and original_pair_margin <= exact_form_new_winner + ROUNDOFF
                    and exact_form_old_winner <= abs(normalizer_gap) + ROUNDOFF
                    and abs(normalizer_gap) <= norm * pair_distance[j, k] + ROUNDOFF
                    and abs(normalizer_gap) <= pair_distance[j, k] + ROUNDOFF
                    and norm * pair_distance[j, k] <= 2.0 * norm + ROUNDOFF
                )
                # registered N3C P1 form: |<z, t_j - t_k>| <= |n_k - n_j|
                registered_ok = abs(original_pair_margin) <= abs(normalizer_gap) + ROUNDOFF
                # top-1 form: the original winner-runner-up margin obeys the same bound
                runner_up = np.max(np.delete(scores, j))
                top1_ok = scores[j] - runner_up <= np.max(np.abs(n[:, None] - n[None, :])) + ROUNDOFF
                if not (bound_chain_ok and registered_ok and top1_ok):
                    violations += 1
                if new_scores[j] > ROUNDOFF and original_pair_margin > ROUNDOFF and normalizer_gap <= 0.0:
                    positive_score_flips_with_wrong_normalizer_order += 1
            flips_by_cell[(class_count, norm)] = flips
    assert total_trials >= 10000
    assert violations == 0
    # sign statement: with a positive post-translation score of the old winner, the
    # old winner can only lose to a class whose raw translated norm is smaller
    assert positive_score_flips_with_wrong_normalizer_order == 0
    # non-vacuous: flips were observed in every (class count, norm) cell
    assert all(count > 0 for count in flips_by_cell.values()), flips_by_cell


def test_per_class_scale_extension_on_tangent_translations() -> None:
    """t'_k = (a_k t_k + v)/n_k with a_k = 1 - v^T t_k: a_j <z,t_j - t_k> <= |n_k - n_j| + |a_j - a_k|."""

    rng = np.random.default_rng(SEED + 3)
    flips = 0
    for class_count in CLASS_COUNTS:
        prototypes = _unit_rows(rng, class_count)
        for norm in TRANSLATION_NORMS:
            for trial in range(300):
                j0, k0 = rng.choice(class_count, size=2, replace=False)
                v = (
                    _random_translation(rng, norm)
                    if trial % 2 == 0
                    else _adversarial_translation(rng, prototypes, j0, k0, norm)
                )
                z = _near_tie_feature(rng, prototypes, j0, k0, rng.uniform(-3.0, 3.0) * norm)
                scale = 1.0 - prototypes @ v  # a_k
                if np.any(scale <= 0.0):
                    continue  # the registered condition requires every a_k > 0
                raw = scale[:, None] * prototypes + v[None, :]
                tangent_form = prototypes + v[None, :] - (prototypes @ v)[:, None] * prototypes
                assert np.max(np.abs(raw - tangent_form)) <= ROUNDOFF
                n = np.linalg.norm(raw, axis=1)
                translated = raw / n[:, None]
                scores = prototypes @ z
                new_scores = translated @ z
                j, k = int(np.argmax(scores)), int(np.argmax(new_scores))
                if j == k:
                    continue
                flips += 1
                assert scale[j] * (scores[j] - scores[k]) <= abs(n[k] - n[j]) + abs(scale[j] - scale[k]) + ROUNDOFF
                # sharper intermediate statement used in the proof
                assert scale[j] * scores[j] - scale[k] * scores[k] <= (n[j] - n[k]) * new_scores[j] + ROUNDOFF
    assert flips > 50


# --------------------------------------------------------------------------- Corollary 1


@pytest.mark.parametrize("class_count", CLASS_COUNTS)
def test_orthogonal_projection_translation_never_changes_the_argmax(class_count: int) -> None:
    rng = np.random.default_rng(SEED + 4 + class_count)
    prototypes = _unit_rows(rng, class_count)
    calibration_images = _unit_rows(rng, 200)
    image_mean = calibration_images.mean(axis=0)
    text_mean = prototypes.mean(axis=0)

    # numpy mirror of satml2027/common/banks.py:218-223 (exact affine-span projection)
    centered = prototypes - text_mean[None, :]
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    threshold = max(centered.shape) * np.finfo(centered.dtype).eps * singular_values.max()
    basis = vh[singular_values > threshold].T
    toward_images = image_mean - text_mean
    projected = toward_images - basis @ (basis.T @ toward_images)
    assert np.linalg.norm(projected) > 1e-3

    # independent construction: QR of the K-1 differences t_k - t_K, which span the
    # same subspace as the centered prototypes (the centered matrix itself has rank
    # K-1 because its rows sum to zero, so QR must not be applied to it directly)
    differences = (prototypes[:-1] - prototypes[-1][None, :]).T
    q_basis, _ = np.linalg.qr(differences)
    alternative = toward_images - q_basis @ (q_basis.T @ toward_images)
    assert np.max(np.abs(alternative - projected)) <= 1e-9

    features = np.concatenate(
        [
            _unit_rows(rng, 5000),
            np.stack(
                [
                    _near_tie_feature(
                        rng,
                        prototypes,
                        *rng.choice(class_count, size=2, replace=False),
                        rng.choice([-1.0, 1.0]) * 10.0 ** rng.uniform(-8.0, -2.0),
                    )
                    for _ in range(5000)
                ]
            ),
        ]
    )
    scores = features @ prototypes.T
    original = np.argmax(scores, axis=1)
    original_counts = np.bincount(original, minlength=class_count)

    for scale in (0.25, 0.5, 1.0, 1.0 / np.linalg.norm(projected), 3.0 / np.linalg.norm(projected)):
        w = scale * projected
        # orthogonal to every prototype difference, so every t_k^T w is the same
        assert np.max(np.abs(centered @ w)) <= 1e-10 * max(1.0, np.linalg.norm(w))
        translated, n = _translate(prototypes, w)
        assert n.max() - n.min() <= 1e-10
        new_scores = features @ translated.T
        common_shift = features @ w
        assert np.max(np.abs(new_scores - (scores + common_shift[:, None]) / n[0])) <= 1e-10
        changed = np.argmax(new_scores, axis=1) != original
        assert int(changed.sum()) == 0
        assert np.array_equal(np.bincount(np.argmax(new_scores, axis=1), minlength=class_count), original_counts)


def test_orthogonal_projection_float32_flips_are_confined_to_numerical_ties() -> None:
    """In float32 arithmetic the invariance is exact up to roundoff: any flip is a numerical tie."""

    rng = np.random.default_rng(SEED + 6)
    prototypes = _unit_rows(rng, 100)
    differences = (prototypes[:-1] - prototypes[-1][None, :]).T
    q_basis, _ = np.linalg.qr(differences)
    g = rng.standard_normal(DIM)
    w = 0.5 * _unit(g - q_basis @ (q_basis.T @ g))
    translated, _ = _translate(prototypes, w)
    features = np.stack(
        [
            _near_tie_feature(
                rng, prototypes, *rng.choice(100, size=2, replace=False), 10.0 ** rng.uniform(-9.0, -3.0)
            )
            for _ in range(4000)
        ]
    )
    exact_scores = features @ prototypes.T
    exact_margin = np.sort(exact_scores, axis=1)[:, -1] - np.sort(exact_scores, axis=1)[:, -2]
    single = np.float32
    original = np.argmax(features.astype(single) @ prototypes.astype(single).T, axis=1)
    after = np.argmax(features.astype(single) @ translated.astype(single).T, axis=1)
    flipped = original != after
    assert np.all(exact_margin[flipped] <= 1e-5)


# --------------------------------------------------------------------------- Corollary 2


def _item_draws(rng: np.random.Generator, prototypes: np.ndarray, kind: str, draws: int, margin_scale: float) -> np.ndarray:
    def concentrated(count: int, klass: int) -> np.ndarray:
        rows = prototypes[klass][None, :] + 0.3 * rng.standard_normal((count, DIM)) / np.sqrt(DIM) * np.sqrt(DIM)
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)

    if kind == "already":
        return concentrated(draws, 0)
    if kind == "split":
        return np.concatenate([concentrated(draws // 2, 0), concentrated(draws - draws // 2, 1)])
    near = np.stack(
        [_near_tie_feature(rng, prototypes, 0, 1, rng.uniform(-2.0, 2.0) * margin_scale) for _ in range(draws - 300)]
    )
    return np.concatenate([concentrated(300, 0), near])


def test_flip_budget_bounds_brute_force_counts() -> None:
    rng = np.random.default_rng(SEED + 5)
    draws = 512
    k_min = 456  # the registered N3C threshold at 512 draws (value used only as a fixed cut-off here)
    partition_seen = {"already": 0, "unattainable": 0, "undetermined": 0}
    flipped_total = 0
    for class_count in CLASS_COUNTS:
        prototypes = _unit_rows(rng, class_count)
        for norm in TRANSLATION_NORMS:
            for kind in ("already", "split", "mixed"):
                w = (
                    _adversarial_translation(rng, prototypes, 0, 1, norm)
                    if kind == "mixed"
                    else _random_translation(rng, norm)
                )
                translated, n = _translate(prototypes, w)
                normalizer_gap = np.abs(n[:, None] - n[None, :])
                features = _item_draws(rng, prototypes, kind, draws, margin_scale=1.5 * normalizer_gap[0, 1] + 1e-6)

                scores = features @ prototypes.T
                winners = np.argmax(scores, axis=1)
                top = np.take_along_axis(scores, winners[:, None], axis=1)
                margins = top - scores  # s_winner - s_k >= 0
                counts0 = np.bincount(winners, minlength=class_count)
                k0 = int(counts0.max())

                # per-draw eligibility exactly as satml2027/day1/d1_08_n3c_predictions.py:197-201 (a_k = 1)
                eligible = margins <= normalizer_gap[winners] + ROUNDOFF
                np.put_along_axis(eligible, winners[:, None], False, axis=1)
                eligible_draw = eligible.any(axis=1)
                budget = int(eligible_draw.sum())  # E_F
                runner_up_margin = np.partition(margins, 1, axis=1)[:, 1]
                coarse_budget = int((runner_up_margin <= normalizer_gap.max() + ROUNDOFF).sum())

                new_scores = features @ translated.T
                new_winners = np.argmax(new_scores, axis=1)
                counts1 = np.bincount(new_winners, minlength=class_count)
                flipped = new_winners != winners
                flipped_total += int(flipped.sum())

                assert int(flipped.sum()) <= budget <= coarse_budget
                assert bool(np.all(eligible[np.flatnonzero(flipped), new_winners[flipped]]))
                assert int(counts1.max()) <= k0 + budget
                assert bool(np.all(counts1 <= counts0 + budget))
                assert bool(np.all(counts1 >= counts0 - budget))

                if k0 >= k_min:
                    partition_seen["already"] += 1
                elif k0 + budget < k_min:
                    partition_seen["unattainable"] += 1
                    assert int(counts1.max()) < k_min  # brute force agrees with the registered P7 prediction
                else:
                    partition_seen["undetermined"] += 1
                for cutoff in range(0, draws + 1, 16):
                    if k0 + budget < cutoff:
                        assert int(counts1.max()) < cutoff
    assert flipped_total > 0
    assert all(count > 0 for count in partition_seen.values()), partition_seen


# --------------------------------------------------------------------------- two-sided remark


def test_two_sided_centering_can_flip_beyond_every_text_only_bound() -> None:
    """GR-CLIP-style two-sided centering escapes Propositions 1-2 (existence)."""

    prototypes = np.zeros((3, DIM))
    prototypes[0, 0] = prototypes[1, 1] = prototypes[2, 2] = 1.0  # t_1 = e1, t_2 = e2, t_3 = e3
    text_mean = prototypes.mean(axis=0)
    image_mean = np.zeros(DIM)
    image_mean[0] = 0.98  # a mean of unit image features concentrated near t_1 (hub-like)
    z = np.zeros(DIM)
    z[0], z[1] = 1.0, 0.05
    z = _unit(z)

    scores = prototypes @ z
    j = int(np.argmax(scores))
    assert j == 0
    original_margin = scores[0] - scores[1]

    # text-only part of the operator: w = -mu_T, exact normalizers n_k
    w = -text_mean
    translated, n = _translate(prototypes, w)
    pair_distance = np.linalg.norm(prototypes[0] - prototypes[1])
    assert abs(n[0] - n[1]) <= ROUNDOFF  # symmetric bank: the exact text-only bound is zero
    assert original_margin > np.linalg.norm(w) * pair_distance > abs(n[0] - n[1])
    assert int(np.argmax(translated @ z)) == 0  # as Proposition 2 requires, no text-only flip

    # two-sided operator as implemented (banks.py:210-215, phase3_candidates.py:40-58)
    centered_image = z - image_mean
    centered_image /= np.linalg.norm(centered_image)
    two_sided_scores = translated @ centered_image
    identity = ((z - image_mean) @ (prototypes - text_mean[None, :]).T) / (n * np.linalg.norm(z - image_mean))
    assert np.max(np.abs(two_sided_scores - identity)) <= ROUNDOFF
    assert int(np.argmax(two_sided_scores)) == 1  # flipped to t_2 although the margin exceeded every text-only bound

    # the escape is the class-dependent image-side term -mu_I^T t_k, absent from a text-only translation
    class_bias = -(prototypes @ image_mean)
    assert class_bias[0] < class_bias[1]
