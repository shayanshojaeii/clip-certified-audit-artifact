"""Candidate and control prototype banks.

`build_eighteen_candidate_banks` is a line-by-line port of the frozen EXP-017
construction rule (see `analysis/exp017_stage_b_independent.py`).  It is used
unchanged for the EXP-019 extension so that the extension tests the *same*
operators at new sigma, backbone and dataset.  `d1_03_build_banks.py
--verify-against-exp017` rebuilds one existing cell and compares tensor hashes
with the saved EXP-017 banks, which is the custody proof that the rule did not
drift.

Control banks (learned shared translation, noisy class mean, low-rank tangent
adapter) are *not* part of the frozen grid.  They are registered separately as
controls and are always unit-norm prototype banks, so the worker scores them
exactly like any candidate.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from .hashing import tensor_scientific_hash

IDENTITY_IMAGE_TRANSFORM = "identity"
GR_CLIP_IMAGE_TRANSFORM = "gr_clip_style_center_and_renormalize"

FROZEN_GRID_CANDIDATE_IDS = frozenset(
    {
        "no_correction",
        "gr_clip_style_two_sided__coefficient_1",
        *(f"global_mean_centering__coefficient_{value:.12g}" for value in (0.25, 0.5, 1.0)),
        *(f"chowers_exact_projected_gap__coefficient_{value:.12g}" for value in (0.25, 0.5, 1.0)),
        *(f"clean_boundary_active__step_{value:.12g}" for value in (0.0025, 0.005, 0.01, 0.02, 0.04)),
        *(f"cohen_aligned_noisy_margin__step_{value:.12g}" for value in (0.0025, 0.005, 0.01, 0.02, 0.04)),
    }
)
REGISTERED_CONTROL_CANDIDATE_IDS = frozenset(
    {
        "control__learned_shared_translation_tangent",
        "control__learned_shared_translation_pure",
        "control__lowrank_tangent_r8",
        "control__noisy_class_mean",
    }
)
EXPECTED_CANDIDATE_IDS = FROZEN_GRID_CANDIDATE_IDS | REGISTERED_CONTROL_CANDIDATE_IDS


def validate_registered_bank_payload(
    payload: Mapping[str, Any], *, registration_sha256: str, cell_id: str
) -> dict[str, dict[str, Any]]:
    """Authenticate the complete registered 18+4 bank artifact.

    Metadata agreement alone is not custody: every stored tensor is re-hashed,
    the exact candidate-ID set is enforced, and the GR-CLIP image statistic is
    independently authenticated.  Worker, merge, and N3C analysis all call
    this one fail-closed validator.
    """

    if payload.get("registration_sha256") != registration_sha256:
        raise RuntimeError("bank file was built under a different registration")
    if payload.get("cell_id") != cell_id:
        raise RuntimeError("bank file was built for a different cell")
    banks = payload.get("banks")
    if not isinstance(banks, Mapping):
        raise RuntimeError("bank artifact has no bank mapping")
    observed_ids = set(banks)
    if observed_ids != EXPECTED_CANDIDATE_IDS:
        missing = sorted(EXPECTED_CANDIDATE_IDS - observed_ids)
        extra = sorted(observed_ids - EXPECTED_CANDIDATE_IDS)
        raise RuntimeError(
            "registered bank candidate IDs differ from the exact 18+4 contract; "
            f"missing={missing}, extra={extra}"
        )

    source_entry = banks["no_correction"]
    source_tensor = source_entry.get("prototypes")
    if not isinstance(source_tensor, torch.Tensor):
        raise RuntimeError("no_correction prototypes are absent or not a tensor")
    source_hash = tensor_scientific_hash(source_tensor)

    for candidate_id in sorted(EXPECTED_CANDIDATE_IDS):
        entry = banks[candidate_id]
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"bank {candidate_id} is not a mapping")
        metadata = entry.get("metadata")
        prototypes = entry.get("prototypes")
        if not isinstance(metadata, Mapping):
            raise RuntimeError(f"bank {candidate_id} lacks metadata")
        if metadata.get("candidate_id") != candidate_id:
            raise RuntimeError(f"bank key/metadata candidate ID mismatch for {candidate_id}")
        if not isinstance(prototypes, torch.Tensor):
            raise RuntimeError(f"bank {candidate_id} prototypes are absent or not a tensor")
        actual_output_hash = tensor_scientific_hash(prototypes)
        if metadata.get("output_prototype_sha256") != actual_output_hash:
            raise RuntimeError(f"bank {candidate_id} prototype tensor hash mismatch")
        if metadata.get("source_prototype_sha256") != source_hash:
            raise RuntimeError(f"bank {candidate_id} source prototype hash mismatch")

        expected_transform = (
            GR_CLIP_IMAGE_TRANSFORM
            if candidate_id == "gr_clip_style_two_sided__coefficient_1"
            else IDENTITY_IMAGE_TRANSFORM
        )
        if metadata.get("image_transform", IDENTITY_IMAGE_TRANSFORM) != expected_transform:
            raise RuntimeError(f"bank {candidate_id} image-transform contract mismatch")
        if expected_transform == GR_CLIP_IMAGE_TRANSFORM:
            image_mean = entry.get("image_mean")
            if not isinstance(image_mean, torch.Tensor):
                raise RuntimeError("GR-CLIP bank lacks its calibration image mean tensor")
            if metadata.get("calibration_image_mean_sha256") != tensor_scientific_hash(image_mean):
                raise RuntimeError("GR-CLIP calibration image mean hash mismatch")
    return dict(banks)


def normalize_rows(value: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.vector_norm(value, dim=1)
    if torch.any(norms <= 1e-12) or not torch.isfinite(norms).all():
        raise ValueError("bank construction produced a negligible row")
    return (value / norms[:, None]).contiguous()


def build_eighteen_candidate_banks(
    *,
    prototypes: torch.Tensor,
    development_clean_features: torch.Tensor,
    clean_direction: torch.Tensor,
    noisy_direction: torch.Tensor,
    common_steps: Sequence[float],
    gap_coefficients: Sequence[float],
) -> dict[str, dict[str, Any]]:
    """Frozen EXP-017 grid: 1 + 3 + 1 + 3 + 5 + 5 = 18 banks."""

    bank = torch.as_tensor(prototypes).detach().cpu().contiguous()
    features = torch.as_tensor(development_clean_features).detach().cpu().contiguous()
    clean = torch.as_tensor(clean_direction).detach().cpu().contiguous()
    noisy = torch.as_tensor(noisy_direction).detach().cpu().contiguous()
    if bank.ndim != 2 or features.ndim != 2 or bank.shape[1] != features.shape[1]:
        raise ValueError("prototype and development-feature shapes are incompatible")
    if bank.dtype != features.dtype or not bank.is_floating_point():
        raise ValueError("prototypes and features must share a floating dtype")
    if clean.shape != (bank.shape[1],) or noisy.shape != (bank.shape[1],):
        raise ValueError("proposal directions have the wrong shape")
    if not all(torch.isfinite(value).all() for value in (bank, features, clean, noisy)):
        raise ValueError("bank construction inputs must be finite")
    ones = torch.ones(bank.shape[0], dtype=torch.float64)
    if not torch.allclose(torch.linalg.vector_norm(bank.double(), dim=1), ones, atol=2e-5, rtol=0.0):
        raise ValueError("source prototype bank must be unit normalized")
    if not torch.allclose(
        torch.linalg.vector_norm(features.double(), dim=1),
        torch.ones(features.shape[0], dtype=torch.float64),
        atol=2e-5,
        rtol=0.0,
    ):
        raise ValueError("development features must be unit normalized")
    for direction in (clean, noisy):
        if not torch.isclose(direction.double().norm(), torch.tensor(1.0, dtype=torch.float64), atol=2e-5, rtol=0.0):
            raise ValueError("proposal directions must be unit normalized")

    steps = tuple(float(value) for value in common_steps)
    coefficients = tuple(float(value) for value in gap_coefficients)
    if len(steps) != 5 or len(coefficients) != 3:
        raise ValueError("frozen grids require five steps and three coefficients")

    image_mean = features.mean(dim=0).to(dtype=bank.dtype)
    text_mean = bank.mean(dim=0)
    uniform_weights = torch.full((bank.shape[0],), 1.0 / bank.shape[0], dtype=bank.dtype)
    weighted_text_mean = uniform_weights @ bank
    source_hash = tensor_scientific_hash(bank)
    image_mean_hash = tensor_scientific_hash(image_mean)
    text_mean_hash = tensor_scientific_hash(text_mean)
    output: dict[str, dict[str, Any]] = {}

    def add(candidate_id, candidate_bank, *, family, objective_id, value,
            image_transform=IDENTITY_IMAGE_TRANSFORM, shared_vector=None, row_scales=None):
        candidate_bank = candidate_bank.detach().cpu().contiguous()
        operator = operator_parameters(bank, shared_vector=shared_vector, row_scales=row_scales)
        output[candidate_id] = {
            "prototypes": candidate_bank,
            "metadata": {
                "candidate_id": candidate_id,
                "family": family,
                "objective_id": objective_id,
                "step_or_coefficient": value,
                "image_transform": image_transform,
                "source_prototype_sha256": source_hash,
                "output_prototype_sha256": tensor_scientific_hash(candidate_bank),
                "calibration_image_mean_sha256": image_mean_hash if image_transform != IDENTITY_IMAGE_TRANSFORM else None,
                "role": "frozen_exp017_grid",
                "operator": operator,
            },
        }

    add("no_correction", bank.clone(), family="no_correction", objective_id=None, value=None,
        shared_vector=torch.zeros(bank.shape[1], dtype=bank.dtype))

    gap = weighted_text_mean - image_mean
    for coefficient in coefficients:
        add(
            f"global_mean_centering__coefficient_{coefficient:.12g}",
            normalize_rows(bank - coefficient * gap.unsqueeze(0)),
            family="global_mean_centering",
            objective_id=None,
            value=coefficient,
            shared_vector=-coefficient * gap,
        )

    add(
        "gr_clip_style_two_sided__coefficient_1",
        normalize_rows(bank - text_mean.unsqueeze(0)),
        family="gr_clip_style_two_sided",
        objective_id=None,
        value=1.0,
        image_transform=GR_CLIP_IMAGE_TRANSFORM,
        shared_vector=-text_mean,
    )

    centered = bank - weighted_text_mean.unsqueeze(0)
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    threshold = max(centered.shape) * torch.finfo(centered.dtype).eps * float(singular_values.max().item())
    basis = vh[singular_values > threshold].T
    toward_images = image_mean - weighted_text_mean
    projected = toward_images.clone() if basis.shape[1] == 0 else toward_images - basis @ (basis.T @ toward_images)
    for coefficient in coefficients:
        add(
            f"chowers_exact_projected_gap__coefficient_{coefficient:.12g}",
            normalize_rows(bank + coefficient * projected.unsqueeze(0)),
            family="chowers_exact_projected_gap",
            objective_id=None,
            value=coefficient,
            shared_vector=coefficient * projected,
        )

    for objective_id, direction in (("clean_boundary_active", clean), ("cohen_aligned_noisy_margin", noisy)):
        cast_direction = direction.to(dtype=bank.dtype)
        for step in steps:
            add(
                f"{objective_id}__step_{step:.12g}",
                normalize_rows(bank + step * cast_direction.unsqueeze(0)),
                family="proposal",
                objective_id=objective_id,
                value=step,
                shared_vector=step * cast_direction,
            )

    if len(output) != 18:
        raise AssertionError("candidate construction did not produce 18 banks")
    return output


# --------------------------------------------------------------------------- controls


def noisy_class_mean_bank(noisy_features: torch.Tensor, labels: torch.Tensor, class_count: int) -> torch.Tensor:
    """Training-free positive control: normalized mean of normalized noisy draws."""

    if noisy_features.ndim != 3:
        raise ValueError("noisy_features must be [items, draws, dim]")
    flat = noisy_features.reshape(-1, noisy_features.shape[-1]).to(dtype=torch.float32)
    flat = flat / torch.linalg.vector_norm(flat, dim=1, keepdim=True)
    repeated = labels.repeat_interleave(noisy_features.shape[1])
    rows = []
    for klass in range(class_count):
        mask = repeated == klass
        if not bool(mask.any()):
            raise ValueError(f"class {klass} has no training draws")
        rows.append(flat[mask].mean(dim=0))
    return normalize_rows(torch.stack(rows))


def tangent_shared_translation(prototypes: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """t'_k = normalize(t_k + (v - <v,t_k> t_k)); the Phase-4 shared-translation form."""

    base = prototypes.to(dtype=torch.float32)
    vector = delta.to(dtype=torch.float32)
    tangent = vector.unsqueeze(0) - (base @ vector)[:, None] * base
    return normalize_rows(base + tangent)


def pure_shared_translation(prototypes: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """t'_k = normalize(t_k + v); the unprojected form of the same vector."""

    base = prototypes.to(dtype=torch.float32)
    return normalize_rows(base + delta.to(dtype=torch.float32).unsqueeze(0))


def low_rank_tangent_bank(prototypes: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """t'_k = normalize(t_k + tangent(LR)_k); the Phase-4 rank-r adapter form."""

    base = prototypes.to(dtype=torch.float32)
    delta = (left.to(dtype=torch.float32) @ right.to(dtype=torch.float32))
    tangent = delta - (base * delta).sum(dim=1, keepdim=True) * base
    return normalize_rows(base + tangent)


def operator_parameters(base: torch.Tensor, *, shared_vector=None, row_scales=None) -> dict:
    """Exact (a_k, n_k) of t'_k = (a_k t_k + v)/n_k, recorded at construction time.

    The winner-pair flip condition of Proposition 2 needs only differences of the
    per-class normalizers `n_k` and scales `a_k`, so storing them here makes the
    N3c containment test and the P7 flip budget exact instead of recovered.
    """

    if shared_vector is None:
        return {"known": False}
    reference = base.to(dtype=torch.float64)
    vector = shared_vector.to(dtype=torch.float64)
    scales = (
        torch.ones(reference.shape[0], dtype=torch.float64)
        if row_scales is None
        else row_scales.to(dtype=torch.float64)
    )
    raw = scales[:, None] * reference + vector.unsqueeze(0)
    norms = torch.linalg.vector_norm(raw, dim=1)
    return {
        "known": True,
        "form": "t'_k = (a_k t_k + v) / n_k",
        "shared_vector": [float(value) for value in vector],
        "shared_vector_sha256": tensor_scientific_hash(vector.to(dtype=torch.float32)),
        "shared_vector_norm": float(torch.linalg.vector_norm(vector)),
        "row_scales": [float(value) for value in scales],
        "row_norms": [float(value) for value in norms],
        "max_normalizer_difference": float((norms[None, :] - norms[:, None]).abs().max()),
        "max_scale_difference": float((scales[None, :] - scales[:, None]).abs().max()),
    }


def normalizer_statistics(prototypes: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    """Per-class normalizers and the mismatch that Proposition 2 bounds."""

    base = prototypes.to(dtype=torch.float64)
    other = candidate.to(dtype=torch.float64)
    scale = (other * base).sum(dim=1)
    residual = other - scale[:, None] * base
    norms = torch.linalg.vector_norm(residual, dim=1)
    return {
        "cosine_min": float(scale.min()),
        "cosine_max": float(scale.max()),
        "orthogonal_residual_max": float(norms.max()),
        "frobenius_displacement": float(torch.linalg.matrix_norm(other - base)),
    }
