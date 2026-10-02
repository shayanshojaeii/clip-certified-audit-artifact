#!/usr/bin/env python3
"""Day 1 - N3c predictions P1, P4a, P5, P6, P7 from the stored fresh-noise features.

Implemented here (the falsifiable and structural core of the registration):

  P1  winner-pair containment.  For every draw whose argmax changes from j to k
      under candidate F, check |<z, t_j - t_k>| <= |n_k - n_j| for a normalized
      shared translation, or the per-class-scale form
      a_j <z, t_j - t_k> <= |n_k - n_j| + |a_j - a_k| for a tangent control,
      where t'_k = (a_k t_k + v) / n_k is recovered from the bank itself.
  P4a marginal Gaussian calibration of the fixed clean-anchor versus fixed
      clean-runner-up signed noisy margin: Phi(mu/s)
      from the first half of the draws against the held-out frequency in the
      second half, 10 equal-mass bins.
  P5  attractor geometry: Spearman rho between <t_k, mean noisy feature> and the
      noisy vote share, with a label-permutation reference.
  P6  descriptive noisy-margin anatomy including per-draw winner-runner-up
      margin quantiles.
  P7  flip budget versus the finite-budget threshold: max_j c'_j <= k_0 + E_F,
      reported as the partition {k_0 >= k_min} / {k_0 + E_F < k_min} / undetermined,
      overall and conditional on the k_0 regime.  No certificate is issued and
      the phrase "cannot be certified" is not used.

P2 orthogonality residuals are reported by the bank builder. P3/P4b were removed
by the pre-sampling amendment because they were not implemented as registered.

Usage:
  python day1/d1_08_n3c_predictions.py --features results/satml2027/N3C/cells \
      --banks results/satml2027/banks --registration configs/satml2027/n3c_v3.json \
      --out analysis/n3c
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.certify import min_successes_for_radius  # noqa: E402
from common.banks import validate_registered_bank_payload  # noqa: E402
from common.hashing import canonical_json_hash, read_json, write_json_atomic  # noqa: E402

TIE_TOLERANCE = 1e-6


def validate_feature_identity(data, registration: dict, cell: dict,
                              shard_index: int | None = None, shard_count: int | None = None) -> None:
    sampling = registration["sampling"]
    observed = {
        "experiment_id": str(data["experiment_id"][0]),
        "registration_sha256": str(data["registration_sha256"][0]),
        "cell_id": str(data["cell_id"][0]),
        "sigma": float(data["sigma"][0]),
        "stream_role": str(data["stream_role"][0]),
        "base_seed": int(data["base_seed"][0]),
        "shard_index": int(data["shard_index"][0]),
        "shard_count": int(data["shard_count"][0]),
    }
    expected = {
        "experiment_id": registration["experiment_id"],
        "registration_sha256": registration["registration_sha256"],
        "cell_id": cell["cell_id"],
        "sigma": float(cell["sigma"]),
        "stream_role": sampling["stream_role"],
        "base_seed": int(sampling["base_seed"]),
        "shard_index": int(data["shard_index"][0]) if shard_index is None else int(shard_index),
        "shard_count": int(data["shard_count"][0]) if shard_count is None else int(shard_count),
    }
    if observed != expected:
        raise RuntimeError("N3C feature provenance differs from the active registration/cell")


def validate_count_feature_alignment(data, counts, path: Path | str) -> None:
    """Bind every P9-relevant feature-shard field to its count shard."""

    if not np.array_equal(data["done"], counts["done"]):
        raise RuntimeError(f"N3C count/feature completion mismatch for {path}")
    if not np.array_equal(data["item_ids"], counts["item_ids"]):
        raise RuntimeError(f"N3C count/feature item mismatch for {path}")
    if not np.array_equal(data["ground_truth"], counts["ground_truth"]):
        raise RuntimeError(f"N3C count/feature ground-truth mismatch for {path}")


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    a, b = rankdata(first), rankdata(second)
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denominator) if denominator > 0 else 0.0


def recover_operator(base: np.ndarray, candidate: np.ndarray) -> dict[str, np.ndarray]:
    """Recover (a_k, n_k, v) for t'_k = (a_k t_k + v)/n_k from the two banks.

    With `r_k = t'_k - <t'_k, t_k> t_k` the component orthogonal to `t_k`, a pure
    or tangent shared translation has `v_perp_k = n_k r_k`; `n_k` is identified up
    to the shared scale by least squares against the mean direction, which is all
    the flip condition needs (it uses differences of `n` and of `a`).
    """

    scale = (candidate * base).sum(axis=1)
    residual = candidate - scale[:, None] * base
    norms = np.linalg.norm(residual, axis=1)
    reference = residual[np.argmax(norms)]
    reference = reference / max(np.linalg.norm(reference), 1e-30)
    projection = residual @ reference
    n = np.where(np.abs(projection) > 1e-12, np.abs(projection).max() / np.maximum(np.abs(projection), 1e-12), 1.0)
    n = n / n.mean()
    return {"a": scale * n, "n": n}


def evaluate_cell(features: np.ndarray, clean_features: np.ndarray, base: np.ndarray,
                  banks: dict[str, np.ndarray], k_min: int,
                  metadata: dict[str, dict] | None = None,
                  ground_truth: np.ndarray | None = None,
                  permutation_seed: int = 20260920008,
                  image_means: dict[str, np.ndarray] | None = None) -> dict:
    items, draws, _ = features.shape
    scores = np.einsum("idf,kf->idk", features.astype(np.float32), base.astype(np.float32))
    winners = scores.argmax(axis=2)
    top = np.take_along_axis(scores, winners[..., None], axis=2)
    margins = top - scores
    partner = margins.copy()
    np.put_along_axis(partner, winners[..., None], np.inf, axis=2)
    runner_margin = partner.min(axis=2)
    counts0 = np.stack([np.bincount(winners[i], minlength=base.shape[0]) for i in range(items)])
    k0 = counts0.max(axis=1)

    report: dict[str, object] = {
        "items": int(items),
        "draws": int(draws),
        "k_min": int(k_min),
        "P6_winner_runner_up_margin_quantiles": {
            str(q): float(np.quantile(runner_margin, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        },
        "P6_items_already_at_threshold": int((k0 >= k_min).sum()),
        "P1": {},
        "P2": {},
        "P7": {},
        "P9": {},
    }

    for candidate_id, candidate in sorted(banks.items()):
        if candidate_id == "no_correction":
            continue
        if candidate.shape != base.shape:
            raise RuntimeError(f"candidate {candidate_id} has the wrong shape")
        candidate_metadata = (metadata or {}).get(candidate_id, {})
        candidate_features = features.astype(np.float32)
        if candidate_metadata.get("image_transform") == "gr_clip_style_center_and_renormalize":
            if candidate_id not in (image_means or {}):
                raise RuntimeError(f"candidate {candidate_id} requires its frozen image mean")
            candidate_features = candidate_features - np.asarray(image_means[candidate_id], dtype=np.float32)
            norms = np.linalg.norm(candidate_features, axis=2, keepdims=True)
            if np.any(norms <= 1e-12):
                raise RuntimeError(f"candidate {candidate_id} produced an undefined centered image feature")
            candidate_features = candidate_features / norms
        new_scores = np.einsum("idf,kf->idk", candidate_features, candidate.astype(np.float32))
        new_winners = new_scores.argmax(axis=2)
        flipped = new_winners != winners
        counts1 = np.stack([np.bincount(new_winners[i], minlength=base.shape[0]) for i in range(items)])
        k1 = counts1.max(axis=1)
        if ground_truth is not None:
            truth = np.asarray(ground_truth, dtype=np.int64)[:, None]
            report["P9"][candidate_id] = {
                "unchanged": int((~flipped).sum()),
                "useful_wrong_to_correct": int((flipped & (winners != truth) & (new_winners == truth)).sum()),
                "harmful_correct_to_wrong": int((flipped & (winners == truth) & (new_winners != truth)).sum()),
                "wrong_to_wrong": int((flipped & (winners != truth) & (new_winners != truth)).sum()),
                "total_flips": int(flipped.sum()),
            }
        recorded = candidate_metadata.get("operator", {})
        if candidate_metadata.get("image_transform") != "identity" or not recorded.get("known"):
            report["P1"][candidate_id] = {
                "applicable": False,
                "reason": "containment applies only to pure/shared-tangent prototype operators with identity image transform",
            }
            report["P7"][candidate_id] = {"applicable": False, "reason": "P1 operator class not applicable"}
            continue
        if recorded.get("known"):
            a = np.asarray(recorded["row_scales"], dtype=np.float64)
            n = np.asarray(recorded["row_norms"], dtype=np.float64)
            exact = True
        else:
            operator = recover_operator(base.astype(np.float64), candidate.astype(np.float64))
            a, n, exact = operator["a"], operator["n"], False
        if not bool(np.all(a > 0)):
            reason = "the registered necessary condition requires every row scale a_k > 0"
            report["P1"][candidate_id] = {"applicable": False, "reason": reason, "minimum_row_scale": float(a.min())}
            report["P7"][candidate_id] = {"applicable": False, "reason": reason}
            continue
        threshold = np.abs(n[None, :] - n[:, None]) + np.abs(a[None, :] - a[:, None])
        eligible = (a[winners][..., None] * margins) <= (threshold[winners] + TIE_TOLERANCE)
        np.put_along_axis(eligible, winners[..., None], False, axis=2)
        eligible_draw = eligible.any(axis=2)
        budget = eligible_draw.sum(axis=1)
        actual_pair_eligible = np.take_along_axis(eligible, new_winners[..., None], axis=2)[..., 0]
        outside = int((flipped & ~actual_pair_eligible).sum())

        report["P1"][candidate_id] = {
            "applicable": True,
            "flips": int(flipped.sum()),
            "flips_outside_eligible_set": outside,
            "conforms": outside == 0,
            "max_normalizer_mismatch": float(np.abs(n[None, :] - n[:, None]).max()),
            "max_scale_mismatch": float(np.abs(a[None, :] - a[:, None]).max()),
            "operator_parameters": "exact (recorded at construction)" if exact else "recovered from the banks (conservative)",
        }
        already = k0 >= k_min
        unattainable = (~already) & (k0 + budget < k_min)
        report["P7"][candidate_id] = {
            "applicable": True,
            "already_at_threshold": int(already.sum()),
            "threshold_unattainable_on_these_draws": int(unattainable.sum()),
            "undetermined": int(((~already) & ~unattainable).sum()),
            "bound_violations": int((k1 > k0 + budget).sum()),
            "E_F_median": float(np.median(budget)),
            "E_F_q90": float(np.quantile(budget, 0.9)),
            "realized_at_threshold_after": int((k1 >= k_min).sum()),
            "by_k0_regime": {
                name: {"items": int(mask.sum()), "E_F_median": float(np.median(budget[mask])) if mask.any() else None}
                for name, mask in (("k0>=k_min", already), ("k0<k_min", ~already))
            },
        }
        if "chowers" in candidate_id.lower():
            reconstructed_rows = n[:, None] * candidate.astype(np.float64) - a[:, None] * base.astype(np.float64)
            reconstructed_vector = reconstructed_rows.mean(axis=0)
            row_agreement = float(np.max(np.abs(reconstructed_rows - reconstructed_vector[None, :])))
            if "shared_vector" not in recorded:
                raise RuntimeError(f"Chowers candidate {candidate_id} lacks its construction vector")
            construction_vector = np.asarray(recorded["shared_vector"], dtype=np.float64)
            construction_agreement = float(np.max(np.abs(reconstructed_vector - construction_vector)))
            centered = base.astype(np.float64) - base.astype(np.float64).mean(axis=0, keepdims=True)
            report["P2"][candidate_id] = {
                "centered_span_orthogonality_residual_max": float(np.abs(centered @ construction_vector).max()),
                "reconstructed_shared_vector_row_agreement_max": row_agreement,
                "recorded_vs_reconstructed_shared_vector_max": construction_agreement,
                "normalizer_mismatch_max": float(np.abs(n[:, None] - n[None, :]).max()),
                "p1_flips": int(flipped.sum()),
                "p1_exceptions": outside,
            }

    # P5 attractor geometry: first half estimates the geometry, second half the votes
    half = draws // 2
    mean_feature = features[:, :half].astype(np.float64).mean(axis=(0, 1))
    alignment = base.astype(np.float64) @ mean_feature
    share = np.bincount(winners[:, half:].reshape(-1), minlength=base.shape[0]) / (items * (draws - half))
    observed = spearman(alignment, share)
    generator = np.random.default_rng(permutation_seed)
    reference = np.array([spearman(generator.permutation(alignment), share) for _ in range(10000)])
    report["P5"] = {
        "spearman_rho": observed,
        "permutation_reference_p": float((np.abs(reference) >= abs(observed)).mean()),
        "observed_hub_class": int(share.argmax()),
        "hub_rank_by_alignment": int(1 + (alignment > alignment[int(share.argmax())]).sum()),
        "top_class_share": float(share.max()),
    }

    # P4a: freeze anchor and runner-up from the clean feature, then measure the
    # signed noisy margin for that fixed pair. This preserves negative margins.
    clean_scores = clean_features.astype(np.float64) @ base.astype(np.float64).T
    clean_anchor = clean_scores.argmax(axis=1)
    masked = clean_scores.copy()
    masked[np.arange(items), clean_anchor] = -np.inf
    clean_runner_up = masked.argmax(axis=1)
    signed_margin = (
        scores[np.arange(items)[:, None], np.arange(draws)[None, :], clean_anchor[:, None]]
        - scores[np.arange(items)[:, None], np.arange(draws)[None, :], clean_runner_up[:, None]]
    )
    mu, sd = signed_margin[:, :half].mean(axis=1), signed_margin[:, :half].std(axis=1, ddof=1)
    predicted = norm.cdf(np.divide(mu, sd, out=np.zeros_like(mu), where=sd > 0))
    observed_frequency = (signed_margin[:, half:] > 0).mean(axis=1)
    order = np.argsort(predicted)
    bins = np.array_split(order, min(10, len(order)))
    calibration = [
        {"bin": index, "items": int(len(chunk)), "predicted": float(predicted[chunk].mean()),
         "observed": float(observed_frequency[chunk].mean())}
        for index, chunk in enumerate(bins)
    ]
    report["P4a"] = {
        "bins": calibration,
        "mean_absolute_calibration_error": float(np.mean([abs(b["predicted"] - b["observed"]) for b in calibration])),
        "note": "diagnostic only; a Gaussian plug-in score, never a bound",
        "pair_definition": "fixed raw-clean top-1 anchor versus fixed raw-clean runner-up",
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=Path, help="directory of *_features.npz shard files")
    parser.add_argument("--banks", required=True, type=Path)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    import torch  # bank files are torch archives

    registration = read_json(args.registration)
    sampling = registration["sampling"]
    alpha = float(sampling["alpha_per_example"])
    permutation_seed = int(sampling["p5_permutation_seed"])
    k_min = min_successes_for_radius(int(sampling["draws_per_item"]), alpha, float(sampling["sigma"]), float(sampling["sigma"]))
    results = {}
    for cell in registration["cells"]:
        cell_id = cell["cell_id"]
        bank_path = args.banks / f"{cell_id}__banks.pt"
        if not bank_path.is_file():
            raise FileNotFoundError(f"registered N3C bank file is absent: {bank_path}")
        shards = sorted(args.features.glob(f"{cell_id}/*_features.npz"))
        if not shards:
            raise FileNotFoundError(f"registered N3C feature shards are absent for {cell_id}")
        loaded_shards = [np.load(path, allow_pickle=False) for path in shards]
        for path, data in zip(shards, loaded_shards):
            shard_token = path.name.removesuffix("_features.npz").removeprefix("shard_")
            shard_index_text, shard_count_text = shard_token.split("_of_")
            validate_feature_identity(data, registration, cell,
                                      shard_index=int(shard_index_text), shard_count=int(shard_count_text))
            count_name = path.name.replace("_features.npz", ".npz")
            count_path = path.with_name(count_name)
            meta_path = count_path.with_suffix(".meta.json")
            if not count_path.is_file() or not meta_path.is_file():
                raise RuntimeError(f"N3C feature shard lacks its corresponding count shard/metadata: {path}")
            meta = read_json(meta_path)
            if (meta.get("experiment_id") != registration["experiment_id"]
                    or meta.get("registration_sha256") != registration["registration_sha256"]
                    or meta.get("cell", {}).get("cell_id") != cell_id):
                raise RuntimeError(f"N3C count-shard metadata mismatch for {path}")
            counts = np.load(count_path, allow_pickle=False)
            validate_count_feature_alignment(data, counts, path)
        if any("done" not in data or not bool(np.all(data["done"])) for data in loaded_shards):
            raise RuntimeError(f"N3C feature shard is incomplete for {cell_id}")
        item_ids = np.concatenate([data["item_ids"] for data in loaded_shards]).tolist()
        if len(set(item_ids)) != len(item_ids):
            raise RuntimeError(f"N3C feature shards contain duplicate items for {cell_id}")
        if canonical_json_hash(item_ids) != cell["development_items_sha256"]:
            raise RuntimeError(f"N3C feature item-list hash mismatch for {cell_id}")
        features = np.concatenate([data["features"].astype(np.float32) for data in loaded_shards], axis=0)
        clean_features = np.concatenate([data["clean_features"].astype(np.float32) for data in loaded_shards], axis=0)
        ground_truth = np.concatenate([data["ground_truth"].astype(np.int64) for data in loaded_shards], axis=0)
        payload = torch.load(bank_path, map_location="cpu", weights_only=False)
        validated_banks = validate_registered_bank_payload(
            payload,
            registration_sha256=registration["registration_sha256"],
            cell_id=cell_id,
        )
        banks = {key: entry["prototypes"].numpy() for key, entry in validated_banks.items()}
        metadata = {key: entry["metadata"] for key, entry in validated_banks.items()}
        image_means = {
            key: entry["image_mean"].numpy()
            for key, entry in validated_banks.items()
            if "image_mean" in entry
        }
        base = banks["no_correction"]
        results[cell_id] = evaluate_cell(features, clean_features, base, banks, k_min, metadata,
                                         ground_truth=ground_truth, permutation_seed=permutation_seed,
                                         image_means=image_means)
        p1 = results[cell_id]["P1"]
        applicable = [v for v in p1.values() if v.get("applicable")]
        print(f"{cell_id}: P1 conforms for {sum(1 for v in applicable if v['conforms'])}/{len(applicable)} applicable banks; "
              f"P5 rho={results[cell_id]['P5']['spearman_rho']:+.3f}; "
              f"P4a MACE={results[cell_id]['P4a']['mean_absolute_calibration_error']:.4f}")
    if set(results) != {cell["cell_id"] for cell in registration["cells"]}:
        raise RuntimeError("N3C result cell set differs from the registered four-cell set")
    write_json_atomic(args.out / "n3c_predictions.json", {
        "schema_version": "satml2027.n3c_predictions.v1",
        "registration_sha256": registration["registration_sha256"],
        "k_min": k_min,
        "cells": results,
        "final_test_access": False,
    })
    print(f"wrote {args.out}/n3c_predictions.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
