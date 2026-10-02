"""Registration-driven candidate banks for EXP-021 (V2: 36 per cell).

Every operator form is the frozen EXP-017 form, reproduced from the same
expressions as ``satml2027/common/banks.py::build_eighteen_candidate_banks`` so
that the 18 legacy candidate ids rebuild bit-for-bit (or within the registered
portable tolerance 2e-6) from the saved EXP-016 tensors; since V4, 021A stores the saved EXP-017 tensors and
scores them bit for bit, and the formula reconstruction is a conformance check.  The candidate family,
its order and every printed formula come from ``satml2027ext/candidates_ext.py``
(the single source of truth shared with the registration generator, the guard,
the worker and the analysis).  New members relative to EXP-017:

* two-sided centring at coefficients 0.25 and 0.5 (coefficient 1 is legacy);
* text-only centring ``t'_k = normalize(t_k - c mean(t))`` with ``z`` unchanged
  (V2: the exact text half of the two-sided operator; same prototype tensor);
* image-only centring: prototypes unchanged, ``z' = normalize(z - c mu_I)``;
* common-translation steps 0.08, 0.16, 0.32 for both development directions;
* four supervised controls: learned shared translation (tangent form), rank-8
  tangent adapter (EXP-019 noisy+clean cross-entropy recipe), noisy class mean,
  and the few-shot context-prompt bank (``satml2027ext/prompt_bank.py``).

A V2 payload stores its source tensors (prototypes, development clean features,
image mean, both directions, the learned control parameters) and a provenance
block bound to the registration (ordered role-item hashes, prompt and recipe
hashes, checkpoint bytes, model state, data preflight).  The validator checks
that provenance against the registration and recomputes every deterministic
operator from the stored sources, so a payload can no longer validate merely by
agreeing with itself (V1 review findings P0-C/P0-D).

Conventions copied from the frozen rule (do not "simplify" them; they decide
the last float32 bits):
  * ``mu_I = development_clean_features.mean(dim=0)``;
  * text mean-centering and the projected gap use the *weighted* uniform mean
    ``w @ bank`` with ``w = 1/K``; the two-sided and text-only operators use
    ``bank.mean(dim=0)``;
  * raw rows are ``bank +/- value * vector.unsqueeze(0)`` and are renormalized with
    ``normalize_rows``.
"""

from __future__ import annotations

import copy
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from satml2027ext._common import (  # puts satml2027/ on sys.path; frozen hashing helpers
    PROJECT_ROOT,
    canonical_json_hash,
    read_json,
    sha256_file,
    write_json_atomic,
)

if str(PROJECT_ROOT) not in sys.path:  # `interventions/` and `certification/` live at the root
    sys.path.insert(0, str(PROJECT_ROOT))

from common.banks import (  # noqa: E402  (approved EXP-019 port of the EXP-017 rule)
    low_rank_tangent_bank,
    noisy_class_mean_bank,
    normalize_rows,
    normalizer_statistics,
    operator_parameters,
    tangent_shared_translation,
)
from common.hashing import tensor_scientific_hash  # noqa: E402
from satml2027ext.candidates_ext import (  # noqa: E402
    CANDIDATES_SCHEMA_VERSION,
    CONTROL_PREFIX,
    IMAGE_TRANSFORM_CENTER,
    canonical_candidates_block,
    candidate_specs,
)
from satml2027ext.candidates_ext import registered_candidate_ids as _registered_candidate_ids  # noqa: E402
from satml2027ext import candidates_ext  # noqa: E402

BANK_PAYLOAD_SCHEMA_VERSION = "satml2027ext.bank_payload.v2"
UNIT_ATOL = 2e-5
PORTABLE_RECOMPUTE_ATOL = 2e-6
KNOWN_SOURCE_MODES = ("exp016_saved_tensors", "encode_registered_roles")
IMAGE_MEAN_ALGORITHM = candidates_ext.IMAGE_MEAN_ALGORITHM
SAVED_LEGACY_MODE = "exp016_saved_tensors"
ALLOWED_PREPARATION_ROLES = ("development", "control_train")
STORED_CONTROL_TENSORS = ("shared_delta", "lowrank_left", "lowrank_right", "fewshot_prompt_prototypes",
                          "fewshot_context_vectors")
RECOMPUTABLE_FROM_PAYLOAD = frozenset({
    "no_correction", "global_mean_centering", "chowers_exact_projected_gap", "gr_clip_style_two_sided",
    "text_only_centering", "image_only_centering", "proposal",
})
RECOMPUTABLE_CONTROLS = frozenset({"learned_shared_translation_tangent", "lowrank_tangent_r8", "fewshot_prompt_bank"})

__all__ = [
    "BANK_PAYLOAD_SCHEMA_VERSION",
    "CANDIDATES_SCHEMA_VERSION",
    "CONTROL_PREFIX",
    "IMAGE_TRANSFORM_CENTER",
    "build_registered_banks",
    "candidate_specs",
    "default_candidates_block",
    "load_bank_payload",
    "make_bank_payload",
    "registered_candidate_ids",
    "save_bank_payload",
    "validate_registered_bank_payload_v2",
]


def default_candidates_block() -> dict[str, Any]:
    """The canonical registration block (kept under its V1 name for callers and tests)."""

    return canonical_candidates_block()


def registered_candidate_ids(registration: Mapping[str, Any], cell: Mapping[str, Any]) -> list[str]:
    return _registered_candidate_ids(registration, cell)


# --------------------------------------------------------------------------- sources


def _as_cpu(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    tensor = value.detach().cpu().contiguous()
    if not tensor.is_floating_point() and not name.endswith("labels"):
        raise TypeError(f"{name} must be a floating tensor")
    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite")
    return tensor


def _check_unit_rows(tensor: torch.Tensor, name: str, *, atol: float = UNIT_ATOL) -> None:
    norms = torch.linalg.vector_norm(tensor.double(), dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=atol, rtol=0.0):
        raise ValueError(f"{name} must have unit rows (atol={atol})")


def _direction(value: torch.Tensor, name: str, dimension: int, dtype: torch.dtype) -> torch.Tensor:
    vector = _as_cpu(value, name)
    if vector.shape != (dimension,):
        raise ValueError(f"{name} has shape {tuple(vector.shape)}, expected ({dimension},)")
    _check_unit_rows(vector.unsqueeze(0), name)
    return vector.to(dtype=dtype)


def _ceil_log2(value: int) -> int:
    return 0 if value <= 1 else (int(value) - 1).bit_length()


def portable_column_mean(features: torch.Tensor) -> torch.Tensor:
    """Mean over rows with a fixed summation order (V4, audit E6; see ``IMAGE_MEAN_ALGORITHM``).

    The order is torch's CPU cascade sum for an outer (row) reduction: rows are added in blocks of
    ``2**max(4, ceil(log2 n) // 4)``; after each block the accumulator levels are carried upward.
    NumPy performs every addition elementwise in the feature dtype, so the result is the same on any
    IEEE-754 platform.  Only 2-D tensors are accepted.
    """

    import numpy as np

    if features.ndim != 2 or features.shape[0] < 1:
        raise ValueError("portable_column_mean expects a nonempty 2-D tensor")
    values = np.ascontiguousarray(features.detach().cpu().numpy())
    size = int(values.shape[0])
    levels = 4
    level_power = max(4, _ceil_log2(size) // levels)
    level_step = 1 << level_power
    level_mask = level_step - 1
    accumulators = [np.zeros(values.shape[1], dtype=values.dtype) for _ in range(levels)]
    index = 0
    while index + level_step <= size:
        for _ in range(level_step):
            accumulators[0] = accumulators[0] + values[index]
            index += 1
        for level in range(1, levels):
            accumulators[level] = accumulators[level] + accumulators[level - 1]
            accumulators[level - 1] = np.zeros_like(accumulators[level - 1])
            if (index & (level_mask << (level * level_power))) != 0:
                break
    while index < size:
        accumulators[0] = accumulators[0] + values[index]
        index += 1
    for level in range(1, levels):
        accumulators[0] = accumulators[0] + accumulators[level]
    mean = accumulators[0] / values.dtype.type(size)
    return torch.from_numpy(np.ascontiguousarray(mean.astype(values.dtype))).to(dtype=features.dtype)


def _prepare_sources(sources: Mapping[str, Any]) -> dict[str, Any]:
    bank = _as_cpu(sources["prototypes"], "prototypes")
    features = _as_cpu(sources["development_clean_features"], "development_clean_features")
    if bank.ndim != 2 or features.ndim != 2 or bank.shape[1] != features.shape[1]:
        raise ValueError("prototype and development-feature shapes are incompatible")
    if bank.dtype != features.dtype:
        raise ValueError("prototypes and development features must share a dtype")
    if bank.shape[0] < 2:
        raise ValueError("at least two prototypes are required")
    _check_unit_rows(bank, "prototypes")
    _check_unit_rows(features, "development_clean_features")
    dimension = int(bank.shape[1])

    prepared: dict[str, Any] = {
        "prototypes": bank,
        "development_clean_features": features,
        "image_mean": portable_column_mean(features).to(dtype=bank.dtype),
    }
    supplied_mean = sources.get("image_mean")
    if supplied_mean is not None:
        supplied_mean = _as_cpu(supplied_mean, "image_mean")
        if supplied_mean.shape != (dimension,):
            raise ValueError("supplied image_mean has the wrong shape")
        if not torch.allclose(supplied_mean.double(), prepared["image_mean"].double(), atol=1e-6, rtol=0.0):
            raise ValueError("supplied image_mean is not the mean of the development clean features")
    labels = sources.get("development_labels")
    if labels is not None:
        labels = _as_cpu(labels, "development_labels")
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError("development_labels must align with development_clean_features")
        prepared["development_labels"] = labels.to(torch.long)
    for key in ("clean_direction", "noisy_direction"):
        if sources.get(key) is not None:
            prepared[key] = _direction(sources[key], key, dimension, bank.dtype)
    controls = sources.get("controls") or {}
    if not isinstance(controls, Mapping):
        raise TypeError("sources['controls'] must be a mapping")
    prepared["controls"] = {}
    for key, value in controls.items():
        if isinstance(value, torch.Tensor):
            prepared["controls"][key] = _as_cpu(value, f"controls.{key}")
        else:
            prepared["controls"][key] = value
    legacy = sources.get("legacy_banks")
    if legacy is not None:
        # V4 (text review of reviewer bundle V3, item 5): 021A samples the saved EXP-017 tensors bit for bit
        if not isinstance(legacy, Mapping) or not legacy:
            raise TypeError("sources['legacy_banks'] must be a nonempty mapping of candidate id to tensor")
        prepared["legacy_banks"] = {}
        for key, value in sorted(legacy.items()):
            tensor = _as_cpu(value, f"legacy_banks.{key}")
            if tensor.shape != bank.shape or tensor.dtype != bank.dtype:
                raise ValueError(f"legacy bank {key} has shape/dtype {tuple(tensor.shape)}/{tensor.dtype}, "
                                 f"expected {tuple(bank.shape)}/{bank.dtype}")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"legacy bank {key} is not finite")
            prepared["legacy_banks"][str(key)] = tensor
    return prepared


def source_hashes(prepared: Mapping[str, Any]) -> dict[str, Any]:
    """Hashes of every input tensor a bank can depend on (recorded in metadata)."""

    output: dict[str, Any] = {
        "prototypes": tensor_scientific_hash(prepared["prototypes"]),
        "development_clean_features": tensor_scientific_hash(prepared["development_clean_features"]),
        "image_mean": tensor_scientific_hash(prepared["image_mean"]),
    }
    for key in ("development_labels", "clean_direction", "noisy_direction"):
        if key in prepared:
            output[key] = tensor_scientific_hash(prepared[key])
    output["controls"] = {
        key: tensor_scientific_hash(value)
        for key, value in prepared.get("controls", {}).items()
        if isinstance(value, torch.Tensor)
    }
    if "legacy_banks" in prepared:
        output["legacy_banks"] = {key: tensor_scientific_hash(value) for key, value in sorted(prepared["legacy_banks"].items())}
    return output


# --------------------------------------------------------------------------- operators


def projected_gap_direction(bank: torch.Tensor, image_mean: torch.Tensor) -> torch.Tensor:
    """``(I - V V^T)(mu_I - mu_T)`` exactly as the frozen rule computes it."""

    uniform_weights = torch.full((bank.shape[0],), 1.0 / bank.shape[0], dtype=bank.dtype)
    weighted_text_mean = uniform_weights @ bank
    centered = bank - weighted_text_mean.unsqueeze(0)
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    threshold = max(centered.shape) * torch.finfo(centered.dtype).eps * float(singular_values.max().item())
    basis = vh[singular_values > threshold].T
    toward_images = image_mean - weighted_text_mean
    if basis.shape[1] == 0:
        return toward_images.clone()
    return toward_images - basis @ (basis.T @ toward_images)


def apply_image_transform(features: torch.Tensor, transform: Mapping[str, Any] | None) -> torch.Tensor:
    """Apply a candidate's image-side transform to unit image features [N, d]."""

    if transform is None:
        return features
    if transform.get("type") != IMAGE_TRANSFORM_CENTER:
        raise ValueError(f"unsupported image transform {transform.get('type')!r}")
    mean = transform["mean"].to(dtype=features.dtype, device=features.device)
    coefficient = float(transform["coefficient"])
    raw = features - coefficient * mean.unsqueeze(0)
    norms = torch.linalg.vector_norm(raw, dim=1)
    if torch.any(norms <= 1e-12):
        raise ValueError("image centring produced a negligible feature vector")
    return raw / norms[:, None]


def build_registered_banks(
    specs: Sequence[Mapping[str, Any]], sources: Mapping[str, Any]
) -> "OrderedDict[str, dict[str, Any]]":
    """Build every spec's bank from the development sources.

    ``sources`` keys: ``prototypes`` [K, d] unit rows; ``development_clean_features``
    [N, d] unit rows; optional ``development_labels`` [N]; ``clean_direction`` and
    ``noisy_direction`` [d] unit (required by the step families); optional
    ``image_mean`` (cross-checked against the feature mean); ``controls``: a
    mapping with ``shared_delta`` [d], ``lowrank_left`` [K, r] and
    ``lowrank_right`` [r, d], ``noisy_features`` [M, draws, d] with ``labels`` [M]
    for the noisy class mean, ``fewshot_prompt_prototypes`` [K, d] unit rows
    (from ``prompt_bank.learn_context_prompts``) and optionally
    ``fewshot_context_vectors`` [C, D] (stored for custody, not used here).  Only the sources needed by the
    given specs are required, so tests can rebuild subsets.
    """

    prepared = _prepare_sources(sources)
    bank = prepared["prototypes"]
    image_mean = prepared["image_mean"]
    hashes = source_hashes(prepared)
    class_count, dimension = int(bank.shape[0]), int(bank.shape[1])
    legacy_banks = prepared.get("legacy_banks")

    # Frozen-rule statistics (see module docstring for why both means exist).
    text_mean = bank.mean(dim=0)
    uniform_weights = torch.full((class_count,), 1.0 / class_count, dtype=bank.dtype)
    weighted_text_mean = uniform_weights @ bank
    gap = weighted_text_mean - image_mean
    projected: torch.Tensor | None = None

    output: "OrderedDict[str, dict[str, Any]]" = OrderedDict()

    def add(spec: Mapping[str, Any], candidate_bank: torch.Tensor, *, shared_vector=None, row_scales=None,
            uses_image_mean: bool = False, extra: Mapping[str, Any] | None = None) -> None:
        candidate_bank = candidate_bank.detach().cpu().contiguous()
        if candidate_bank.shape != bank.shape:
            raise AssertionError(f"{spec['candidate_id']} produced shape {tuple(candidate_bank.shape)}")
        legacy_info = None
        if spec.get("legacy_exp017") and legacy_banks is not None:
            saved = legacy_banks.get(spec["candidate_id"])
            if saved is None:
                raise KeyError(f"legacy bank {spec['candidate_id']} is missing from sources['legacy_banks']")
            legacy_info = {
                "prototype_source": "saved EXP-017 tensor, bit for bit",
                "formula_reconstruction_max_abs_delta": float((candidate_bank.double() - saved.double()).abs().max()),
                "formula_reconstruction_exact": bool(torch.equal(candidate_bank, saved)),
            }
            candidate_bank = saved.clone().contiguous()
        _check_unit_rows(candidate_bank, spec["candidate_id"])
        transform_spec = spec.get("image_transform")
        image_transform = None
        if transform_spec is not None:
            image_transform = {
                "type": transform_spec["type"],
                "mean": image_mean.clone(),
                "coefficient": float(transform_spec["coefficient"]),
            }
            uses_image_mean = True
        metadata = {
            "candidate_id": spec["candidate_id"],
            "cell_id": spec.get("cell_id"),
            "position": int(spec["position"]),
            "family": spec["family"],
            "objective_id": spec.get("objective_id"),
            "parameter": spec.get("parameter"),
            "step_or_coefficient": None if spec.get("value") is None else float(spec["value"]),
            "role": spec.get("role"),
            "legacy_exp017": bool(spec.get("legacy_exp017", False)),
            "image_transform": None if image_transform is None else {
                "type": image_transform["type"],
                "coefficient": image_transform["coefficient"],
                "mean_sha256": hashes["image_mean"],
            },
            "source_prototype_sha256": hashes["prototypes"],
            "output_prototype_sha256": tensor_scientific_hash(candidate_bank),
            "calibration_image_mean_sha256": hashes["image_mean"] if uses_image_mean else None,
            "source_hashes": copy.deepcopy(hashes),
            "operator": operator_parameters(bank, shared_vector=shared_vector, row_scales=row_scales),
            "normalizer_statistics": normalizer_statistics(bank, candidate_bank),
        }
        if extra:
            metadata.update(extra)
        if legacy_info is not None:
            metadata.update(legacy_info)
        output[spec["candidate_id"]] = {
            "prototypes": candidate_bank,
            "image_transform": image_transform,
            "metadata": metadata,
        }

    for spec in specs:
        family = spec["family"]
        candidate_id = spec["candidate_id"]
        if candidate_id in output:
            raise ValueError(f"duplicate spec {candidate_id}")
        value = spec.get("value")
        if family == "no_correction":
            add(spec, bank.clone(), shared_vector=torch.zeros(dimension, dtype=bank.dtype))
        elif family == "global_mean_centering":
            coefficient = float(value)
            add(
                spec,
                normalize_rows(bank - coefficient * gap.unsqueeze(0)),
                shared_vector=-coefficient * gap,
                uses_image_mean=True,
            )
        elif family == "chowers_exact_projected_gap":
            coefficient = float(value)
            if projected is None:
                projected = projected_gap_direction(bank, image_mean)
            add(
                spec,
                normalize_rows(bank + coefficient * projected.unsqueeze(0)),
                shared_vector=coefficient * projected,
                uses_image_mean=True,
            )
        elif family in ("gr_clip_style_two_sided", "text_only_centering"):
            # identical prototype tensors at the same coefficient; only the image transform differs
            coefficient = float(value)
            add(
                spec,
                normalize_rows(bank - coefficient * text_mean.unsqueeze(0)),
                shared_vector=-coefficient * text_mean,
            )
        elif family == "image_only_centering":
            add(spec, bank.clone(), shared_vector=torch.zeros(dimension, dtype=bank.dtype))
        elif family == "proposal":
            objective = spec["objective_id"]
            key = {"clean_boundary_active": "clean_direction", "cohen_aligned_noisy_margin": "noisy_direction"}[objective]
            if key not in prepared:
                raise KeyError(f"{candidate_id} needs sources['{key}']")
            direction = prepared[key]
            step = float(value)
            add(
                spec,
                normalize_rows(bank + step * direction.unsqueeze(0)),
                shared_vector=step * direction,
            )
        elif family == "registered_control":
            _add_control(spec, prepared, add, class_count)
        else:
            raise ValueError(f"unsupported family {family!r} for {candidate_id}")

    if len(output) != len(specs):
        raise AssertionError("bank construction did not produce one bank per spec")
    return output


def _add_control(spec: Mapping[str, Any], prepared: Mapping[str, Any], add, class_count: int) -> None:
    bank = prepared["prototypes"]
    controls = prepared["controls"]
    control_id = spec["objective_id"]
    candidate_id = spec["candidate_id"]
    if control_id == "learned_shared_translation_tangent":
        if "shared_delta" not in controls:
            raise KeyError(f"{candidate_id} needs sources['controls']['shared_delta']")
        delta = controls["shared_delta"].to(torch.float32)
        if delta.shape != (bank.shape[1],):
            raise ValueError("shared_delta has the wrong shape")
        add(
            spec,
            tangent_shared_translation(bank, delta),
            shared_vector=delta,
            row_scales=(1.0 - bank.to(torch.float32) @ delta),
            extra={"control_source": "shared_delta"},
        )
    elif control_id == "lowrank_tangent_r8":
        for key in ("lowrank_left", "lowrank_right"):
            if key not in controls:
                raise KeyError(f"{candidate_id} needs sources['controls']['{key}']")
        left = controls["lowrank_left"].to(torch.float32)
        right = controls["lowrank_right"].to(torch.float32)
        if left.shape[0] != bank.shape[0] or right.shape[1] != bank.shape[1] or left.shape[1] != right.shape[0]:
            raise ValueError("low-rank factors have incompatible shapes")
        add(
            spec,
            low_rank_tangent_bank(bank, left, right),
            extra={"control_source": "lowrank_left,lowrank_right", "rank": int(left.shape[1]),
                   "objective": "noisy_clean_cross_entropy (EXP-019 recipe)"},
        )
    elif control_id == "noisy_class_mean":
        for key in ("noisy_features", "labels"):
            if key not in controls:
                raise KeyError(f"{candidate_id} needs sources['controls']['{key}']")
        add(
            spec,
            noisy_class_mean_bank(controls["noisy_features"], controls["labels"].to(torch.long), class_count),
            extra={"control_source": "noisy_features,labels"},
        )
    elif control_id == "fewshot_prompt_bank":
        if "fewshot_prompt_prototypes" not in controls:
            raise KeyError(f"{candidate_id} needs sources['controls']['fewshot_prompt_prototypes']")
        prototypes = controls["fewshot_prompt_prototypes"].to(torch.float32)
        if prototypes.shape != bank.shape:
            raise ValueError("few-shot prompt prototypes have the wrong shape")
        _check_unit_rows(prototypes, candidate_id)
        add(
            spec,
            prototypes.clone(),
            extra={"control_source": "fewshot_prompt_prototypes", "context_tokens": spec.get("context_tokens")},
        )
    else:
        raise ValueError(f"unsupported control {control_id!r}")


# --------------------------------------------------------------------------- payload


def _stored_sources(prepared: Mapping[str, Any]) -> dict[str, Any]:
    stored: dict[str, Any] = {
        "prototypes": prepared["prototypes"].clone(),
        "development_clean_features": prepared["development_clean_features"].clone(),
        "image_mean": prepared["image_mean"].clone(),
    }
    for key in ("clean_direction", "noisy_direction"):
        if key in prepared:
            stored[key] = prepared[key].clone()
    stored["controls"] = {
        key: value.clone()
        for key, value in prepared.get("controls", {}).items()
        if key in STORED_CONTROL_TENSORS and isinstance(value, torch.Tensor)
    }
    if "legacy_banks" in prepared:
        stored["legacy_banks"] = {key: value.clone() for key, value in sorted(prepared["legacy_banks"].items())}
    return stored


def make_bank_payload(
    banks: Mapping[str, Mapping[str, Any]],
    *,
    registration: Mapping[str, Any],
    cell: Mapping[str, Any],
    sources: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the ``.pt`` payload for one cell (registered order, stored sources, provenance)."""

    order = registered_candidate_ids(registration, cell)
    if list(banks) != order:
        raise ValueError("banks must be supplied in the registered candidate order and be complete")
    prepared = _prepare_sources(sources)
    first = next(iter(banks.values()))
    hashes = first["metadata"]["source_hashes"]
    if hashes != source_hashes(prepared):
        raise ValueError("the supplied sources are not the sources the banks were built from")
    if not isinstance(provenance, Mapping) or provenance.get("sealed_evaluation_access") is not False:
        raise ValueError("provenance must record sealed_evaluation_access=False")
    return {
        "schema_version": BANK_PAYLOAD_SCHEMA_VERSION,
        "experiment_id": registration.get("experiment_id"),
        "registration_sha256": registration["registration_sha256"],
        "cell_id": cell["cell_id"],
        "candidate_order": order,
        "candidates": {candidate_id: dict(banks[candidate_id]) for candidate_id in order},
        "image_mean": prepared["image_mean"].clone(),
        "source_hashes": copy.deepcopy(hashes),
        "sources": _stored_sources(prepared),
        "provenance": copy.deepcopy(dict(provenance)),
        "sealed_evaluation_access": False,
    }


def _json_safe_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    transform = entry.get("image_transform")
    return {
        "metadata": copy.deepcopy(entry["metadata"]),
        "image_transform": None if transform is None else {
            "type": transform["type"],
            "coefficient": float(transform["coefficient"]),
            "mean_sha256": tensor_scientific_hash(transform["mean"]),
        },
        "prototype_shape": list(entry["prototypes"].shape),
        "prototype_dtype": str(entry["prototypes"].dtype),
    }


def _source_tensor_hashes(stored: Mapping[str, Any]) -> dict[str, Any]:
    output = {key: tensor_scientific_hash(value) for key, value in stored.items() if isinstance(value, torch.Tensor)}
    output["controls"] = {key: tensor_scientific_hash(value) for key, value in stored.get("controls", {}).items()}
    return output


def save_bank_payload(payload: Mapping[str, Any], pt_path: str | Path, *, overwrite: bool = False) -> tuple[Path, Path]:
    """Write ``<stem>.pt`` and its ``<stem>.json`` sidecar; never overwrites unless asked (tests only)."""

    target = Path(pt_path)
    if target.suffix != ".pt":
        raise ValueError("bank payload path must end in .pt")
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite an existing bank payload: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".pt.tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(target)
    sidecar = {
        "schema_version": payload["schema_version"],
        "experiment_id": payload.get("experiment_id"),
        "registration_sha256": payload["registration_sha256"],
        "cell_id": payload["cell_id"],
        "candidate_order": list(payload["candidate_order"]),
        "bank_count": len(payload["candidates"]),
        "bank_file_sha256": sha256_file(target),
        "image_mean_sha256": tensor_scientific_hash(payload["image_mean"]),
        "source_hashes": copy.deepcopy(payload["source_hashes"]),
        "stored_source_hashes": _source_tensor_hashes(payload["sources"]),
        "provenance": copy.deepcopy(payload["provenance"]),
        "candidates": {key: _json_safe_entry(entry) for key, entry in payload["candidates"].items()},
        "sealed_evaluation_access": False,
    }
    sidecar["candidates_sha256"] = canonical_json_hash(sidecar["candidates"])
    json_path = write_json_atomic(target.with_suffix(".json"), sidecar)
    return target, json_path


def load_bank_payload(pt_path: str | Path, *, require_sidecar: bool = True) -> dict[str, Any]:
    """Load a payload and authenticate it against its sidecar's file hash."""

    target = Path(pt_path)
    payload = torch.load(target, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != BANK_PAYLOAD_SCHEMA_VERSION:
        raise RuntimeError(f"bank file does not carry the {BANK_PAYLOAD_SCHEMA_VERSION} payload schema")
    sidecar_path = target.with_suffix(".json")
    if sidecar_path.is_file():
        sidecar = read_json(sidecar_path)
        if sidecar.get("bank_file_sha256") != sha256_file(target):
            raise RuntimeError("bank file hash differs from its sidecar")
        if list(sidecar.get("candidate_order", [])) != list(payload.get("candidate_order", [])):
            raise RuntimeError("sidecar candidate order differs from the bank file")
        for key, entry in payload["candidates"].items():
            recorded = sidecar["candidates"].get(key, {}).get("metadata", {}).get("output_prototype_sha256")
            if recorded != entry["metadata"].get("output_prototype_sha256"):
                raise RuntimeError(f"sidecar/bank metadata hash mismatch for {key}")
        if sidecar.get("provenance") != payload.get("provenance"):
            raise RuntimeError("sidecar provenance differs from the bank file")
    elif require_sidecar:
        raise FileNotFoundError(f"bank sidecar missing: {sidecar_path}")
    return payload


# --------------------------------------------------------------------------- provenance


_HEX64 = frozenset("0123456789abcdef")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX64


def control_role(registration: Mapping[str, Any]) -> str:
    """The item role the registered supervised controls are trained on."""

    construction = registration.get("bank_construction")
    if not isinstance(construction, Mapping):
        raise RuntimeError("registration lacks a bank_construction block")
    role = (construction.get("control_training") or {}).get("role")
    if role not in ALLOWED_PREPARATION_ROLES:
        raise RuntimeError(f"registered control role {role!r} is not one of {ALLOWED_PREPARATION_ROLES}")
    return str(role)


def provenance_source_bindings(sources: Mapping[str, Any]) -> dict[str, Any]:
    """The provenance fields that must equal hashes of the stored source tensors.

    Used by the preparation stage (and tests) to write them and by
    ``validate_provenance`` to check them (V3, re-review M1a: V2 recorded these
    hashes but never compared them with the stored tensors).
    """

    hashes = source_hashes(_prepare_sources(sources))
    return bindings_from_source_hashes(hashes)


def bindings_from_source_hashes(hashes: Mapping[str, Any]) -> dict[str, Any]:
    controls = hashes.get("controls") or {}
    features = {"development_clean": hashes.get("development_clean_features")}
    if "noisy_features" in controls:
        features["control_noisy"] = controls["noisy_features"]
    if "labels" in controls:
        features["control_labels"] = controls["labels"]
    prompt = {}
    if "fewshot_prompt_prototypes" in controls:
        prompt["prototypes_sha256"] = controls["fewshot_prompt_prototypes"]
    if "fewshot_context_vectors" in controls:
        prompt["context_vectors_sha256"] = controls["fewshot_context_vectors"]
    return {
        "feature_hashes": features,
        "direction_hashes": {"clean": hashes.get("clean_direction"), "noisy": hashes.get("noisy_direction")},
        "prompt_control": prompt,
    }


def registered_source_mode(registration: Mapping[str, Any]) -> str:
    """The registered bank source mode; unknown or missing modes are refused (V4, audit E1)."""

    construction = registration.get("bank_construction")
    mode = construction.get("source_mode") if isinstance(construction, Mapping) else None
    if mode not in KNOWN_SOURCE_MODES:
        raise RuntimeError(f"registered bank_construction.source_mode {mode!r} is not one of {KNOWN_SOURCE_MODES}")
    return str(mode)


def validate_provenance(payload: Mapping[str, Any], registration: Mapping[str, Any], cell: Mapping[str, Any]) -> dict[str, Any]:
    """Check the payload's provenance block against the registration and the stored sources (fail closed)."""

    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RuntimeError("bank payload lacks a provenance block")
    if provenance.get("registration_sha256") != registration.get("registration_sha256"):
        raise RuntimeError("provenance names a different registration")
    if provenance.get("cell_id") != cell.get("cell_id"):
        raise RuntimeError("provenance names a different cell")
    if provenance.get("sealed_evaluation_access") is not False:
        raise RuntimeError("provenance must record sealed_evaluation_access=False")
    roles = provenance.get("roles")
    if not isinstance(roles, Mapping) or set(roles) - set(ALLOWED_PREPARATION_ROLES):
        raise RuntimeError(f"provenance roles must be a subset of {ALLOWED_PREPARATION_ROLES}")
    required_roles = {"development", control_role(registration)}
    for role in sorted(required_roles):
        entry = roles.get(role)
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"provenance lacks the {role} role record")
        expected = cell.get(f"{role}_items_sha256")
        if not _is_sha256(expected) or entry.get("items_sha256") != expected:
            raise RuntimeError(f"provenance {role} item hash differs from the registered ordered list")
    construction = registration.get("bank_construction")
    if provenance.get("recipe_sha256") != canonical_json_hash(construction):
        raise RuntimeError("provenance recipe hash differs from the registered bank_construction block")
    registered_mode = registered_source_mode(registration)
    if provenance.get("source_mode") != registered_mode:
        # V4 (audit E1): the origin check is chosen from the registration, never from the payload
        raise RuntimeError(f"provenance source_mode {provenance.get('source_mode')!r} differs from the registered "
                           f"bank_construction.source_mode {registered_mode!r}")
    models = registration.get("models") or {}
    binding = models.get(cell.get("model_id"))
    checkpoint = provenance.get("checkpoint")
    if not isinstance(binding, Mapping) or not isinstance(checkpoint, Mapping):
        raise RuntimeError("registration model binding or provenance checkpoint record is absent")
    if checkpoint.get("sha256") != binding.get("sha256") or not _is_sha256(checkpoint.get("sha256")):
        raise RuntimeError("provenance checkpoint hash differs from the registered checkpoint binding")
    if not _is_sha256(checkpoint.get("model_state_sha256")):
        raise RuntimeError("provenance lacks the model state hash")
    prompts = registration.get("prompt_configs") or {}
    registered_prompt = prompts.get(cell.get("dataset_id"))
    prompt = provenance.get("prompt_config")
    if not isinstance(registered_prompt, Mapping) or not isinstance(prompt, Mapping) \
            or prompt.get("sha256") != registered_prompt.get("sha256"):
        raise RuntimeError("provenance prompt configuration differs from the registered one")
    if not _is_sha256(provenance.get("data_preflight_sha256")):
        raise RuntimeError("provenance lacks the data preflight hash")
    if not isinstance(provenance.get("code_sha256"), Mapping) or not provenance["code_sha256"]:
        raise RuntimeError("provenance lacks the code hashes")
    hashes = payload.get("source_hashes")
    if not isinstance(hashes, Mapping):
        raise RuntimeError("bank payload lacks source hashes")
    for block, expected in bindings_from_source_hashes(hashes).items():
        recorded = provenance.get(block)
        if not isinstance(recorded, Mapping):
            raise RuntimeError(f"provenance lacks its {block} record")
        for key, value in expected.items():
            if value is None:
                raise RuntimeError(f"stored sources lack the tensor behind provenance {block}.{key}")
            if recorded.get(key) != value:
                raise RuntimeError(f"provenance {block}.{key} differs from the hash of the stored source tensor")
    cache = provenance.get("control_cache_tensor_sha256")
    features = provenance.get("feature_hashes") or {}
    if cache is not None:
        for cache_key, feature_key in (("clean_features", "control_clean"), ("noisy_features", "control_noisy"),
                                       ("labels", "control_labels")):
            if cache.get(cache_key) != features.get(feature_key):
                raise RuntimeError(f"provenance control cache {cache_key} hash differs from feature_hashes.{feature_key}")
    return dict(provenance)


# --------------------------------------------------------------------------- validator


def _recompute_banks(specs: Sequence[Mapping[str, Any]], stored: Mapping[str, Any], *,
                     use_legacy: bool = True) -> "OrderedDict[str, dict[str, Any]]":
    """Rebuild every bank that is a deterministic function of the stored sources."""

    selected = []
    for spec in specs:
        if spec["family"] in RECOMPUTABLE_FROM_PAYLOAD:
            selected.append(spec)
        elif spec["family"] == "registered_control" and spec["objective_id"] in RECOMPUTABLE_CONTROLS:
            if spec["objective_id"] == "fewshot_prompt_bank" and "fewshot_prompt_prototypes" not in stored.get("controls", {}):
                continue
            selected.append(spec)
    sources = {
        "prototypes": stored["prototypes"],
        "development_clean_features": stored["development_clean_features"],
        "controls": dict(stored.get("controls", {})),
    }
    for key in ("clean_direction", "noisy_direction"):
        if key in stored:
            sources[key] = stored[key]
    if use_legacy and stored.get("legacy_banks"):
        sources["legacy_banks"] = stored["legacy_banks"]
    return build_registered_banks(selected, sources)


def validate_registered_bank_payload_v2(
    payload: Mapping[str, Any],
    registration: Mapping[str, Any],
    cell: Mapping[str, Any],
    *,
    require_provenance: bool = True,
    recompute: bool = True,
) -> "OrderedDict[str, dict[str, Any]]":
    """Fail-closed authentication of a complete EXP-021 bank payload.

    Enforces, from the registration only: the registered id set and order,
    each spec's family/parameter/image-transform contract, unit rows, the
    output tensor hashes, the shared source-prototype hash, the image-mean hash
    for every image-side candidate, agreement of every bank's source hashes with
    the payload-level hashes and with the stored source tensors, the provenance
    block (ordered role hashes, recipe, checkpoint, prompt, preflight), and the
    recomputation of every deterministic operator from the stored sources within
    the registered portable tolerance.
    """

    specs = candidate_specs(registration, cell)
    expected_ids = [spec["candidate_id"] for spec in specs]
    if payload.get("schema_version") != BANK_PAYLOAD_SCHEMA_VERSION:
        raise RuntimeError("bank payload schema mismatch")
    if payload.get("registration_sha256") != registration.get("registration_sha256"):
        raise RuntimeError("bank payload was built under a different registration")
    if payload.get("cell_id") != cell.get("cell_id"):
        raise RuntimeError("bank payload was built for a different cell")
    if payload.get("sealed_evaluation_access") is not False:
        raise RuntimeError("bank payload must record sealed_evaluation_access=False")
    candidates = payload.get("candidates")
    if not isinstance(candidates, Mapping):
        raise RuntimeError("bank payload has no candidate mapping")
    observed_ids = set(candidates)
    if observed_ids != set(expected_ids):
        missing = sorted(set(expected_ids) - observed_ids)
        extra = sorted(observed_ids - set(expected_ids))
        raise RuntimeError(
            f"candidate ids differ from the registered set of {len(expected_ids)}; missing={missing}, extra={extra}"
        )
    if list(payload.get("candidate_order", [])) != expected_ids:
        raise RuntimeError("candidate_order differs from the registered order")
    if require_provenance:
        validate_provenance(payload, registration, cell)

    payload_hashes = payload.get("source_hashes")
    if not isinstance(payload_hashes, Mapping):
        raise RuntimeError("bank payload lacks source hashes")
    stored = payload.get("sources")
    if not isinstance(stored, Mapping):
        raise RuntimeError("bank payload lacks its stored source tensors")
    for key in ("prototypes", "development_clean_features", "image_mean", "clean_direction", "noisy_direction"):
        if key in payload_hashes:
            if not isinstance(stored.get(key), torch.Tensor) or tensor_scientific_hash(stored[key]) != payload_hashes[key]:
                raise RuntimeError(f"stored source tensor {key} differs from the recorded source hash")
    for key, value in (stored.get("controls") or {}).items():
        if payload_hashes.get("controls", {}).get(key) != tensor_scientific_hash(value):
            raise RuntimeError(f"stored control tensor {key} differs from the recorded source hash")
    legacy_ids = [spec["candidate_id"] for spec in specs if spec.get("legacy_exp017")]
    stored_legacy = stored.get("legacy_banks")
    if registered_source_mode(registration) == SAVED_LEGACY_MODE:
        if not isinstance(stored_legacy, Mapping) or sorted(stored_legacy) != sorted(legacy_ids):
            raise RuntimeError("a 021A payload must store the saved EXP-017 tensor of every legacy bank")
        for key, value in stored_legacy.items():
            if (payload_hashes.get("legacy_banks") or {}).get(key) != tensor_scientific_hash(value):
                raise RuntimeError(f"stored legacy tensor {key} differs from the recorded source hash")
    elif stored_legacy is not None:
        raise RuntimeError("only a 021A payload may store saved EXP-017 legacy tensors")

    identity = candidates["no_correction"]
    source_tensor = identity.get("prototypes") if isinstance(identity, Mapping) else None
    if not isinstance(source_tensor, torch.Tensor):
        raise RuntimeError("no_correction prototypes are absent or not a tensor")
    source_hash = tensor_scientific_hash(source_tensor)
    if payload_hashes.get("prototypes") != source_hash:
        raise RuntimeError("payload prototype source hash differs from the no_correction tensor")
    payload_mean = payload.get("image_mean")
    payload_mean_hash = None if payload_mean is None else tensor_scientific_hash(payload_mean)
    if payload_hashes.get("image_mean") != payload_mean_hash:
        raise RuntimeError("payload image mean hash differs from its source hash")

    output: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    for spec in specs:
        candidate_id = spec["candidate_id"]
        entry = candidates[candidate_id]
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"bank {candidate_id} is not a mapping")
        metadata = entry.get("metadata")
        prototypes = entry.get("prototypes")
        if not isinstance(metadata, Mapping):
            raise RuntimeError(f"bank {candidate_id} lacks metadata")
        if metadata.get("candidate_id") != candidate_id:
            raise RuntimeError(f"bank key/metadata candidate id mismatch for {candidate_id}")
        if metadata.get("family") != spec["family"] or metadata.get("objective_id") != spec.get("objective_id"):
            raise RuntimeError(f"bank {candidate_id} family/objective differs from the registration")
        expected_value = None if spec.get("value") is None else float(spec["value"])
        if metadata.get("step_or_coefficient") != expected_value:
            raise RuntimeError(f"bank {candidate_id} parameter differs from the registration")
        if not isinstance(prototypes, torch.Tensor) or prototypes.ndim != 2:
            raise RuntimeError(f"bank {candidate_id} prototypes are absent or not a matrix")
        if prototypes.shape != source_tensor.shape or prototypes.dtype != source_tensor.dtype:
            raise RuntimeError(f"bank {candidate_id} shape/dtype differs from the source bank")
        if not torch.isfinite(prototypes).all():
            raise RuntimeError(f"bank {candidate_id} has non-finite entries")
        norms = torch.linalg.vector_norm(prototypes.double(), dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=UNIT_ATOL, rtol=0.0):
            raise RuntimeError(f"bank {candidate_id} has a non-unit row")
        if metadata.get("output_prototype_sha256") != tensor_scientific_hash(prototypes):
            raise RuntimeError(f"bank {candidate_id} prototype tensor hash mismatch")
        if metadata.get("source_prototype_sha256") != source_hash:
            raise RuntimeError(f"bank {candidate_id} source prototype hash mismatch")
        if metadata.get("source_hashes") != dict(payload_hashes):
            raise RuntimeError(f"bank {candidate_id} source hashes differ from the payload source hashes")

        transform_spec = spec.get("image_transform")
        transform = entry.get("image_transform")
        recorded = metadata.get("image_transform")
        if transform_spec is None:
            if transform is not None or recorded is not None:
                raise RuntimeError(f"bank {candidate_id} must not carry an image transform")
        else:
            if not isinstance(transform, Mapping) or not isinstance(recorded, Mapping):
                raise RuntimeError(f"bank {candidate_id} lacks its registered image transform")
            if transform.get("type") != transform_spec["type"] or recorded.get("type") != transform_spec["type"]:
                raise RuntimeError(f"bank {candidate_id} image transform type mismatch")
            if float(transform.get("coefficient")) != float(transform_spec["coefficient"]) or \
                    float(recorded.get("coefficient")) != float(transform_spec["coefficient"]):
                raise RuntimeError(f"bank {candidate_id} image transform coefficient mismatch")
            mean = transform.get("mean")
            if not isinstance(mean, torch.Tensor) or mean.shape != (source_tensor.shape[1],):
                raise RuntimeError(f"bank {candidate_id} image mean is absent or mis-shaped")
            mean_hash = tensor_scientific_hash(mean)
            if metadata.get("calibration_image_mean_sha256") != mean_hash or recorded.get("mean_sha256") != mean_hash:
                raise RuntimeError(f"bank {candidate_id} image mean hash mismatch")
            if payload_hashes.get("image_mean") != mean_hash or payload_mean_hash != mean_hash:
                raise RuntimeError(f"bank {candidate_id} image mean differs from the payload image mean")
            if spec["family"] == "image_only_centering" and tensor_scientific_hash(prototypes) != source_hash:
                raise RuntimeError(f"bank {candidate_id} must leave the prototypes unchanged")
        if stored_legacy is not None and candidate_id in stored_legacy and not torch.equal(prototypes, stored_legacy[candidate_id]):
            raise RuntimeError(f"legacy bank {candidate_id} is not the stored saved EXP-017 tensor bit for bit")
        output[candidate_id] = dict(entry)

    # V3 (re-review M1c): every recorded operator must reproduce its bank, and every
    # registered flip-theorem bank must record a known unit-scale translation.
    theorem_ids = set(_registered_theorem_ids(registration))
    for spec in specs:
        candidate_id = spec["candidate_id"]
        parsed = verify_operator_record(candidate_id, output[candidate_id], source_tensor)
        if candidate_id in theorem_ids:
            if parsed is None:
                raise RuntimeError(f"flip-theorem bank {candidate_id} does not record its exact operator")
            if float((parsed["scales"] - 1.0).abs().max()) > 1e-12:
                raise RuntimeError(f"flip-theorem bank {candidate_id} is not a unit-scale shared translation")
            if output[candidate_id].get("image_transform") is not None:
                raise RuntimeError(f"flip-theorem bank {candidate_id} must not transform the image feature")

    if recompute:
        rebuilt = _recompute_banks(specs, stored)
        for candidate_id, entry in rebuilt.items():
            delta = float((entry["prototypes"].double() - output[candidate_id]["prototypes"].double()).abs().max())
            if not (delta <= PORTABLE_RECOMPUTE_ATOL):
                raise RuntimeError(
                    f"bank {candidate_id} is not the registered operator applied to the stored sources "
                    f"(max |delta| = {delta:.3e} > {PORTABLE_RECOMPUTE_ATOL:.0e})"
                )
            compare_operator_records(candidate_id, output[candidate_id], entry)
    return output


def _registered_theorem_ids(registration: Mapping[str, Any]) -> list[str]:
    block = registration.get("candidates")
    if isinstance(block, Mapping) and isinstance(block.get("flip_theorem_candidates"), list):
        return [str(value) for value in block["flip_theorem_candidates"]]
    return list(candidates_ext.flip_theorem_ids(candidates_ext.canonical_candidate_ids()))


def _vector(values: Any) -> torch.Tensor:
    return torch.as_tensor([float(value) for value in values], dtype=torch.float64)


def verify_operator_record(candidate_id: str, entry: Mapping[str, Any], base: torch.Tensor, *,
                           atol: float = PORTABLE_RECOMPUTE_ATOL) -> dict[str, Any] | None:
    """If the bank records a known operator, it must reproduce the stored bank (V3, re-review M1c).

    For ``t'_k = (a_k t_k + v) / n_k``: the recorded ``n_k`` must equal
    ``||a_k t_k + v||``, the recorded ``shared_vector_norm`` and hash must match
    ``v``, and ``normalize(a_k t_k + v)`` must equal the stored prototypes, all
    within ``atol``.  Returns the parsed operator (or None when not known).
    """

    operator = (entry.get("metadata") or {}).get("operator")
    if not isinstance(operator, Mapping):
        return None
    known = operator.get("known")
    if known is not True:
        if known not in (False, None):
            raise RuntimeError(f"bank {candidate_id} operator 'known' flag must be a JSON boolean")
        return None
    try:
        vector = _vector(operator["shared_vector"])
        scales = _vector(operator["row_scales"])
        norms = _vector(operator["row_norms"])
        vector_norm = operator["shared_vector_norm"]
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"bank {candidate_id} records a malformed operator") from error
    # V4 (audit E2): every recorded number must be finite before any tolerance comparison
    for name, values in (("shared_vector", vector), ("row_scales", scales), ("row_norms", norms)):
        if not bool(torch.isfinite(values).all()):
            raise RuntimeError(f"bank {candidate_id} operator {name} is not finite")
    if isinstance(vector_norm, bool) or not isinstance(vector_norm, (int, float)) or not math.isfinite(float(vector_norm)) \
            or float(vector_norm) < 0:
        raise RuntimeError(f"bank {candidate_id} operator shared_vector_norm is not a finite nonnegative number")
    if not bool((norms > 0).all()):
        raise RuntimeError(f"bank {candidate_id} operator row norms must be positive")
    reference = base.to(dtype=torch.float64)
    if vector.shape != (reference.shape[1],) or scales.shape != (reference.shape[0],) or norms.shape != (reference.shape[0],):
        raise RuntimeError(f"bank {candidate_id} operator record has the wrong shape")
    raw = scales[:, None] * reference + vector[None, :]
    observed_norms = torch.linalg.vector_norm(raw, dim=1)
    if not bool(torch.isfinite(observed_norms).all()) or not bool((observed_norms > 0).all()):
        raise RuntimeError(f"bank {candidate_id} operator raw row norms must be finite and positive")
    if not (float((observed_norms - norms).abs().max()) <= atol):
        raise RuntimeError(f"bank {candidate_id} operator row norms do not follow from its scales and vector")
    if not (abs(float(torch.linalg.vector_norm(vector)) - float(vector_norm)) <= atol):
        raise RuntimeError(f"bank {candidate_id} operator shared_vector_norm differs from its vector")
    if operator.get("shared_vector_sha256") != tensor_scientific_hash(vector.to(dtype=torch.float32)):
        raise RuntimeError(f"bank {candidate_id} operator shared_vector hash differs from its vector")
    rebuilt = raw / observed_norms[:, None]
    delta = float((rebuilt - entry["prototypes"].to(dtype=torch.float64)).abs().max())
    if not (delta <= atol):
        raise RuntimeError(f"bank {candidate_id} is not normalize(a t + v) of its recorded operator (max |delta| = {delta:.3e})")
    return {"vector": vector, "scales": scales, "norms": norms}


def compare_operator_records(candidate_id: str, stored: Mapping[str, Any], rebuilt: Mapping[str, Any], *,
                             atol: float = PORTABLE_RECOMPUTE_ATOL) -> None:
    """The stored operator record must equal the one the registered builder produces from the stored sources."""

    left = (stored.get("metadata") or {}).get("operator") or {}
    right = (rebuilt.get("metadata") or {}).get("operator") or {}
    if bool(left.get("known")) != bool(right.get("known")):
        raise RuntimeError(f"bank {candidate_id} operator 'known' flag differs from its recomputation")
    if not right.get("known"):
        return
    for key in ("shared_vector", "row_scales", "row_norms"):
        try:
            a, b = _vector(left.get(key, [])), _vector(right.get(key, []))
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"bank {candidate_id} operator {key} is malformed") from error
        if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
            raise RuntimeError(f"bank {candidate_id} operator {key} is not finite")
        if a.shape != b.shape or (a.numel() and not (float((a - b).abs().max()) <= atol)):
            raise RuntimeError(f"bank {candidate_id} operator {key} differs from its recomputation")


def legacy_formula_conformance(payload: Mapping[str, Any], registration: Mapping[str, Any],
                               cell: Mapping[str, Any]) -> dict[str, Any] | None:
    """021A: the registered formulas applied to the stored EXP-016 sources versus the sampled saved tensors.

    A conformance check within the portable tolerance; exact matches are a diagnostic only, because the last
    bits of a reconstruction depend on the runtime (V4, audit E6; text review item 5).
    """

    stored = payload.get("sources") or {}
    if not stored.get("legacy_banks"):
        return None
    specs = [spec for spec in candidate_specs(registration, cell) if spec.get("legacy_exp017")]
    formula = _recompute_banks(specs, stored, use_legacy=False)
    per_candidate = {key: float((entry["prototypes"].double() - stored["legacy_banks"][key].double()).abs().max())
                     for key, entry in formula.items()}
    exact = sum(1 for key, entry in formula.items() if torch.equal(entry["prototypes"], stored["legacy_banks"][key]))
    worst = max(per_candidate.values())
    return {"atol": PORTABLE_RECOMPUTE_ATOL, "max_abs_delta": worst, "within_tolerance": bool(worst <= PORTABLE_RECOMPUTE_ATOL),
            "exact_matches_diagnostic": exact, "banks": len(per_candidate), "per_candidate": per_candidate}


def recomputation_report(payload: Mapping[str, Any], registration: Mapping[str, Any], cell: Mapping[str, Any]) -> dict[str, float]:
    """Max |delta| between every stored bank and its recomputation from the stored sources (for manifests)."""

    specs = candidate_specs(registration, cell)
    rebuilt = _recompute_banks(specs, payload["sources"])
    return {
        candidate_id: float((entry["prototypes"].double() - payload["candidates"][candidate_id]["prototypes"].double()).abs().max())
        for candidate_id, entry in rebuilt.items()
    }


def _lookup(mapping: Mapping[str, Any], dotted: str) -> Any:
    value: Any = mapping
    for part in dotted.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value
