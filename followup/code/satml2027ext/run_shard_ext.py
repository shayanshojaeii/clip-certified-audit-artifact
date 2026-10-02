#!/usr/bin/env python3
"""EXP-021 sampling worker (V2): one process per GPU, one shard of one registered cell.

Descends from ``satml2027/day1/d1_05_run_shard.py`` (unchanged, imported only
through ``satml2027/common``).  V2 repairs the V1 pre-sampling review findings:

*   **Registered item order.**  Lists are loaded in their registered order and
    the ordered hash is verified (V1 sorted them and rejected 30/36 cells).
*   **Fail-closed stage approval.**  The ``sampling`` stage must be authorized
    by an approval that binds the three registrations, the complete dependency
    closure, the data preflight and the bank manifest (``guard_ext``).
*   **Environment and data.**  The runtime must equal the preflight environment;
    every item-list CSV and every EuroSAT/Imagenette image loaded must hash to the
    preflight record; CIFAR-100 batches are MD5-verified by the loaders.
*   **Checkpoint and banks.**  The checkpoint file must hash to the registered
    bytes (Hugging Face hub offline) and the loaded model state must equal the
    state the preparation stage recorded; the bank file must hash to the bank
    manifest's entry for this cell (021C: its 021B parent's payload, validated
    against the parent registration) and is validated with recomputation.
*   **Scored candidates.**  Exactly the registered candidate ids are scored
    (021C: the three registered banks of its subset block).
*   **Per-draw decision-change counters** (V1 Q3/Q4 findings).  For every
    confirmation draw and candidate the worker counts, online and exactly:
    label changes against the identity bank, changes that violate the
    winner-pair containment of the decision-change proposition
    (``a_j (s_j - s_k) <= |n_j - n_k| + |a_j - a_k| + tol``, the approved N3C P1
    form), changes whose identity top-1 margin exceeds the loose bound
    ``2 min(1, ||v||)`` (unit-scale translations), the loose flippable budget,
    and the useful / harmful label-change taxonomy against ground truth.
*   **Budget checkpoints** (021C).  Vote counts after the first 4,096 draws of
    the 100,000-draw confirmation stream are stored per item and candidate; the
    seeds make that prefix identical to the 021B confirmation stream.
*   **Deterministic sharding.**  A cell may be split into contiguous shards of
    at most the registered size; seeds depend on item ids only, so the realized
    draws are shard-invariant; every shard of a cell must run on one GPU class.
*   **Custody fields.**  Every sidecar records the registration, approval, bank
    manifest, bank file, data preflight, checkpoint and model-state hashes, and
    ``sealed_evaluation_access`` truthfully (V1 recorded false for sealed cells).

Both TF32 switches are off unconditionally; cuDNN is deterministic.

V3 (internal re-review of V2, 2026-09-26): the bank manifest must name the
approved data preflight (M2c); the forward batch is the registered
``certification.blocks_per_forward`` (a different ``--blocks-per-forward`` is a
refusal); deterministic algorithms are requested; and a 021C cell is scored
with its 021B parent's complete bank stack, in the parent's order, keeping only
its three registered banks, so the logits of its first 4,096 draws come from
the same GEMM shapes as in 021B and the registered prefix identity can hold
bit for bit on one GPU class (n1).

Example:
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=2 python satml2027ext/run_shard_ext.py \
      --registration configs/satml2027ext/exp-20260921-021b.json \
      --cell-id openai-clip-vit-l14-quickgelu__cifar100_test__sigma0.25 --shard-index 0 --shard-count 3 \
      --data-root data --data-preflight results/satml2027ext/EXP021_DATA_PREFLIGHT.json \
      --bank-manifest results/satml2027ext/EXP021_BANK_MANIFEST.json --out results/satml2027ext/EXP-20260921-021B
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:  # allow `python satml2027ext/run_shard_ext.py`
    sys.path.insert(0, str(_PROJECT_ROOT))

from satml2027ext._common import (  # noqa: E402  (puts satml2027/ on sys.path)
    PROJECT_ROOT,
    canonical_json_hash,
    read_json,
    sha256_file,
    write_json_atomic,
)
from satml2027ext import candidates_ext, items_ext  # noqa: E402
from satml2027ext.guard_ext import (  # noqa: E402
    DEFAULT_APPROVAL_PATH,
    certification_parameters,
    find_cell,
    identity_candidate_id,
    item_list_stems,
    load_registration,
    registered_candidate_ids,
    verify_approval,
)
from common.hashing import tensor_scientific_hash  # noqa: E402
from common.seeds import (  # noqa: E402
    CONFIRMATION_STREAM,
    NOISE_BLOCK,
    SELECTION_STREAM,
    block_plan,
    derive_block_seed,
    derive_item_seed,
)

SHARD_META_SCHEMA = "satml2027ext.shard_meta.v3"
MARGIN_STORE_SCHEMA = "satml2027ext.identity_margins.v1"
MAX_CONFIRMATION_DRAWS = 100_000
IDENTITY_TRANSFORM = "identity"
CENTER_TRANSFORM = "center_and_renormalize"
LEGACY_GR_CLIP_TRANSFORM = "gr_clip_style_center_and_renormalize"
DEFAULT_PHASE3_ASSIGNMENTS = Path("artifacts") / "day14" / "phase3_splits_v1" / "assignments.csv"
UNIT_SCALE_TOLERANCE = 1e-12
FLIP_COUNTERS = ("flip_changed_draws", "flip_containment_violations", "flip_loose_violations",
                 "flip_loose_budget_draws", "flip_useful_draws", "flip_harmful_draws")

# V1 names kept for callers and tests
load_items = items_ext.load_items
verify_item_list = items_ext.verify_item_list


class StopRequested(RuntimeError):
    """Test hook: raised by ``process_shard`` after ``stop_after_items`` new items."""


# --------------------------------------------------------------------------- precision


def configure_precision() -> dict[str, Any]:
    """Disable both TF32 switches, make cuDNN deterministic and request deterministic algorithms.

    V3: ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` is set before any cuBLAS handle exists
    (the planner's scripts also export it) and ``torch.use_deterministic_algorithms``
    is enabled in warn-only mode, so identical GEMM shapes give identical bits on
    one GPU class (the 021C prefix identity relies on it) and any operation
    without a deterministic kernel is reported instead of silently varying.
    """

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    extra: dict[str, Any] = {}
    try:  # torch >= 2.9 exposes the precision as a string as well
        torch.set_float32_matmul_precision("highest")
        extra["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    except Exception:  # pragma: no cover - depends on the torch build
        pass
    for attribute, target in (("matmul", torch.backends.cuda.matmul), ("conv", getattr(torch.backends.cudnn, "conv", None))):
        if target is not None and hasattr(target, "fp32_precision"):
            try:
                target.fp32_precision = "ieee"
                extra[f"{attribute}_fp32_precision"] = target.fp32_precision
            except Exception:  # pragma: no cover
                pass
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("TF32 could not be disabled")
    extra["deterministic_algorithms"] = bool(torch.are_deterministic_algorithms_enabled())
    extra["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    return extra


# --------------------------------------------------------------------------- environment


def _package_version(*names: str) -> str | None:
    from importlib import metadata

    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def _nvidia_driver_version() -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            return completed.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def environment_metadata(device: str, blocks_per_forward: int, precision_extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Provenance recorded in every shard sidecar."""

    import torch

    info: dict[str, Any] = {
        "device": device,
        "device_name": "cpu",
        "device_capability": None,
        "device_total_memory_gib": None,
        "driver_version": None,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "torch_version": torch.__version__,
        "open_clip_version": _package_version("open_clip_torch", "open-clip-torch", "open_clip"),
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn_allowed": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "blocks_per_forward": int(blocks_per_forward),
        "forward_batch_draws": int(blocks_per_forward) * NOISE_BLOCK,
        "noise_block_draws": NOISE_BLOCK,
    }
    if precision_extra:
        info.update({f"precision_{key}": value for key, value in precision_extra.items()})
    if device.startswith("cuda") and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        info["device_name"] = properties.name
        info["device_capability"] = f"{properties.major}.{properties.minor}"
        info["device_total_memory_gib"] = round(properties.total_memory / 2**30, 2)
        info["driver_version"] = _nvidia_driver_version()
    else:
        info["device_name"] = platform.processor() or "cpu"
    if info["tf32_matmul_allowed"] or info["tf32_cudnn_allowed"]:
        raise RuntimeError("TF32 is enabled; the registered precision contract forbids it")
    return info


# --------------------------------------------------------------------------- items and sharding


def shard_slice(rows: list[dict[str, str]], shard_index: int, shard_count: int) -> tuple[int, list[dict[str, str]]]:
    """Contiguous deterministic shard of the registered order (seeds are item-id based, so shard-invariant)."""

    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must lie in [0, shard_count)")
    base, extra = divmod(len(rows), shard_count)
    start = shard_index * base + min(shard_index, extra)
    size = base + (1 if shard_index < extra else 0)
    return start, rows[start : start + size]


def max_items_per_shard(registration: Mapping[str, Any]) -> int:
    sharding = registration.get("sharding")
    if not isinstance(sharding, Mapping) or not isinstance(sharding.get("max_items_per_shard"), int):
        raise RuntimeError("registration lacks sharding.max_items_per_shard")
    return int(sharding["max_items_per_shard"])


# --------------------------------------------------------------------------- banks


def _import_bank_validator() -> Callable[..., Mapping[str, Any]]:
    """Import the V2 validator lazily; tests monkeypatch this to inject a fake."""

    try:
        module = importlib.import_module("satml2027ext.banks_ext")
    except ImportError as error:
        raise RuntimeError(
            "satml2027ext.banks_ext is not importable, so the bank payload cannot be authenticated (fail closed)"
        ) from error
    validator = getattr(module, "validate_registered_bank_payload_v2", None)
    if validator is None:
        raise RuntimeError("satml2027ext.banks_ext lacks validate_registered_bank_payload_v2 (fail closed)")
    return validator


def candidate_transform(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a bank entry's image-side transform to ``{"type", "mean", "coefficient"}``."""

    import torch

    spec = entry.get("image_transform")
    if spec is None:
        metadata = entry.get("metadata") or {}
        spec = metadata.get("image_transform")
        if isinstance(spec, Mapping) and "mean" not in spec:
            spec = {**spec, "mean": entry.get("image_mean")}
    if spec is None or spec == IDENTITY_TRANSFORM:
        return {"type": IDENTITY_TRANSFORM}
    if spec == LEGACY_GR_CLIP_TRANSFORM:
        spec = {"type": CENTER_TRANSFORM, "mean": entry.get("image_mean"), "coefficient": 1.0}
    if not isinstance(spec, Mapping):
        raise RuntimeError(f"unsupported image transform specification {spec!r}")
    kind = spec.get("type")
    if kind == IDENTITY_TRANSFORM:
        return {"type": IDENTITY_TRANSFORM}
    if kind != CENTER_TRANSFORM:
        raise RuntimeError(f"unsupported image transform type {kind!r}")
    mean = spec.get("mean")
    if not isinstance(mean, torch.Tensor) or mean.ndim != 1 or not torch.isfinite(mean).all():
        raise RuntimeError("centring transform lacks a finite 1-D mean tensor")
    coefficient = float(spec.get("coefficient", 1.0))
    if not np.isfinite(coefficient):
        raise RuntimeError("centring transform coefficient must be finite")
    return {"type": CENTER_TRANSFORM, "mean": mean.detach().cpu().to(torch.float32).contiguous(), "coefficient": coefficient}


def transform_summary(transform: Mapping[str, Any]) -> dict[str, Any]:
    if transform["type"] == IDENTITY_TRANSFORM:
        return {"type": IDENTITY_TRANSFORM}
    return {"type": CENTER_TRANSFORM, "coefficient": float(transform["coefficient"]),
            "mean_sha256": tensor_scientific_hash(transform["mean"]), "mean_dim": int(transform["mean"].shape[0])}


def json_safe(value: Any) -> Any:
    """JSON-serializable copy of bank metadata: tensors become hashes, tuples lists."""

    try:
        import torch

        if isinstance(value, torch.Tensor):
            return {"tensor_sha256": tensor_scientific_hash(value), "shape": list(value.shape), "dtype": str(value.dtype)}
    except ImportError:  # pragma: no cover
        pass
    if isinstance(value, np.ndarray):
        return {"tensor_sha256": tensor_scientific_hash(value), "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def load_and_validate_banks(
    payload: Mapping[str, Any], registration: Mapping[str, Any], cell: Mapping[str, Any], *, validator=None,
    parent_registration: Mapping[str, Any] | None = None, parent_cell: Mapping[str, Any] | None = None,
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, Any]]:
    """Authenticate the payload and return (scored ids in registered order, banks, JSON metadata).

    A subset registration (021C) validates the parent's payload against the
    parent registration and cell, then keeps exactly its registered ids.
    """

    import torch

    validator = validator or _import_bank_validator()
    scored = registered_candidate_ids(registration, cell)
    if candidates_ext.is_subset_registration(registration):
        parent = candidates_ext.subset_parent(registration)
        if parent_registration is None or parent_cell is None:
            raise RuntimeError("a subset registration needs its parent registration and parent cell")
        if parent_registration.get("registration_sha256") != parent["registration_sha256"]:
            raise RuntimeError("the parent registration is not the one this subset registration names")
        if cell.get("parent_cell_id") != parent_cell.get("cell_id"):
            raise RuntimeError("the parent cell is not the one this cell names")
        banks = dict(validator(payload, parent_registration, parent_cell))
    else:
        banks = dict(validator(payload, registration, cell))
    if banks is None:
        raise RuntimeError("bank validator returned nothing")
    missing = [candidate_id for candidate_id in scored if candidate_id not in banks]
    if missing:
        raise RuntimeError(f"validated payload lacks registered candidates {missing}")
    banks = {candidate_id: banks[candidate_id] for candidate_id in scored}
    identity = identity_candidate_id(registration)
    if identity not in banks:
        raise RuntimeError(f"identity candidate {identity!r} is absent from the bank payload")
    class_count = int(cell["class_count"])
    dimension = None
    scorer_meta: dict[str, Any] = {}
    for candidate_id in scored:
        entry = banks[candidate_id]
        prototypes = entry.get("prototypes")
        if not isinstance(prototypes, torch.Tensor) or prototypes.ndim != 2 or prototypes.shape[0] != class_count:
            raise RuntimeError(f"bank {candidate_id} prototypes are not a [{class_count}, d] tensor")
        if not torch.isfinite(prototypes).all():
            raise RuntimeError(f"bank {candidate_id} has non-finite prototypes")
        dimension = int(prototypes.shape[1]) if dimension is None else dimension
        if int(prototypes.shape[1]) != dimension:
            raise RuntimeError("banks disagree on the embedding dimension")
        transform = candidate_transform(entry)
        if transform["type"] == CENTER_TRANSFORM and int(transform["mean"].shape[0]) != dimension:
            raise RuntimeError(f"bank {candidate_id} image mean has the wrong dimension")
        metadata = entry.get("metadata")
        if not isinstance(metadata, Mapping) or metadata.get("candidate_id") != candidate_id:
            raise RuntimeError(f"bank {candidate_id} metadata is absent or names another candidate")
        scorer_meta[candidate_id] = {
            "metadata": json_safe(metadata),
            "image_transform": transform_summary(transform),
            "prototype_sha256": tensor_scientific_hash(prototypes),
        }
    if candidate_transform(banks[identity])["type"] != IDENTITY_TRANSFORM:
        raise RuntimeError("the identity candidate must not carry an image transform")
    return list(scored), banks, scorer_meta


# --------------------------------------------------------------------------- scorer and per-draw counters


class FlipPlan:
    """Per-candidate operator parameters for the per-draw decision-change counters.

    Containment applies to every candidate whose bank records its exact operator
    ``t'_k = (a_k t_k + v)/n_k`` (``metadata.operator.known``), keeps the image
    feature unchanged, and has every ``a_k > 0``.  The loose bound applies when
    additionally every ``a_k = 1`` (a pure shared translation).
    """

    def __init__(self, banks: Mapping[str, Mapping[str, Any]], candidate_ids: list[str], identity_id: str,
                 device: str, tolerance: float):
        import torch

        self.tolerance = float(tolerance)
        self.count = len(candidate_ids)
        self.containment_applicable = np.zeros(self.count, dtype=bool)
        self.loose_applicable = np.zeros(self.count, dtype=bool)
        thresholds, scales, bounds, positions, loose = [], [], [], [], []
        for position, candidate_id in enumerate(candidate_ids):
            if candidate_id == identity_id:
                continue
            entry = banks[candidate_id]
            operator = (entry.get("metadata") or {}).get("operator") or {}
            if candidate_transform(entry)["type"] != IDENTITY_TRANSFORM or not operator.get("known"):
                continue
            a = torch.tensor(operator["row_scales"], dtype=torch.float64)
            n = torch.tensor(operator["row_norms"], dtype=torch.float64)
            vector_norm = float(operator.get("shared_vector_norm", float("nan")))
            if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(n).all()) or not bool((n > 0).all()) \
                    or not np.isfinite(vector_norm):
                # V4 (audit E2): the validator refuses such records; the worker never counts against them
                raise RuntimeError(f"bank {candidate_id} carries a non-finite operator record")
            if not bool((a > 0).all()):
                continue
            unit_scale = bool(((a - 1.0).abs() <= UNIT_SCALE_TOLERANCE).all())
            self.containment_applicable[position] = True
            self.loose_applicable[position] = unit_scale
            positions.append(position)
            thresholds.append((n[None, :] - n[:, None]).abs() + (a[None, :] - a[:, None]).abs())
            scales.append(a)
            loose.append(unit_scale)
            bounds.append(2.0 * min(1.0, float(operator.get("shared_vector_norm", 0.0))) if unit_scale else float("inf"))
        self.positions = torch.tensor(positions, dtype=torch.long, device=device)
        self.size = len(positions)
        if self.size:
            classes = int(thresholds[0].shape[0])
            self.classes = classes
            self.thresholds = torch.stack(thresholds).to(device=device, dtype=torch.float32).reshape(-1)
            self.scales = torch.stack(scales).to(device=device, dtype=torch.float32).reshape(-1)
            self.loose_mask = torch.tensor(loose, dtype=torch.bool, device=device)
            self.loose_bounds = torch.tensor([value if np.isfinite(value) else 0.0 for value in bounds],
                                             dtype=torch.float32, device=device)
            self.plan_index = torch.arange(self.size, device=device)

    def summary(self, candidate_ids: list[str]) -> dict[str, Any]:
        return {
            "tolerance": self.tolerance,
            "containment_applicable": [candidate_ids[i] for i in range(self.count) if self.containment_applicable[i]],
            "loose_applicable": [candidate_ids[i] for i in range(self.count) if self.loose_applicable[i]],
        }


class Scorer:
    """Stacked prototype banks with per-candidate image-side transforms.

    Every candidate is scored on the same unit feature ``z``.  Identity
    candidates use ``z`` directly; centring candidates use
    ``normalize(z - c * mu)`` with their own ``(mu, c)``.  Labels are returned in
    the registered candidate order, so paired contrasts share random numbers at
    the draw level.
    """

    def __init__(self, banks: Mapping[str, Mapping[str, Any]], candidate_ids: list[str], identity_id: str, device: str,
                 *, flip_tolerance: float = 1e-5):
        import torch

        self.candidate_ids = list(candidate_ids)
        self.count = len(self.candidate_ids)
        self.device = device
        self.class_count = int(banks[self.candidate_ids[0]]["prototypes"].shape[0])
        if self.class_count < 2:
            raise ValueError("at least two classes are required for a margin")
        if self.class_count > np.iinfo(np.int16).max:
            raise ValueError("class ids do not fit int16")
        plain, centred = [], []
        for position, candidate_id in enumerate(self.candidate_ids):
            entry = banks[candidate_id]
            transform = candidate_transform(entry)
            if transform["type"] == IDENTITY_TRANSFORM:
                plain.append((position, entry))
            else:
                centred.append((position, entry, transform))
        self.identity_position = self.candidate_ids.index(identity_id)
        if self.identity_position not in {position for position, _ in plain}:
            raise ValueError("the identity candidate must use the identity image transform")
        to_device = lambda tensor: tensor.detach().to(device=device, dtype=torch.float32)  # noqa: E731
        self.plain_positions = torch.tensor([position for position, _ in plain], device=device, dtype=torch.long)
        self.plain_banks = torch.stack([to_device(entry["prototypes"]) for _, entry in plain])
        self.centred_positions = torch.tensor([position for position, _, _ in centred], device=device, dtype=torch.long)
        if centred:
            self.centred_banks = torch.stack([to_device(entry["prototypes"]) for _, entry, _ in centred])
            self.centred_means = torch.stack([to_device(transform["mean"]) for _, _, transform in centred])
            self.centred_coefficients = torch.tensor(
                [float(transform["coefficient"]) for _, _, transform in centred], device=device, dtype=torch.float32
            )
        else:
            self.centred_banks = self.centred_means = self.centred_coefficients = None
        self.flip = FlipPlan(banks, self.candidate_ids, identity_id, device, flip_tolerance)

    def logits(self, features):
        """Cosine logits [draws, candidates, classes] for unit features [draws, d]."""

        import torch

        out = torch.empty((features.shape[0], self.count, self.class_count), dtype=torch.float32, device=self.device)
        out[:, self.plain_positions] = torch.einsum("bd,ckd->bck", features, self.plain_banks)
        if self.centred_banks is not None:
            centred = features.unsqueeze(0) - self.centred_coefficients[:, None, None] * self.centred_means[:, None, :]
            norms = torch.linalg.vector_norm(centred, dim=2, keepdim=True)
            if bool(torch.any(norms <= 1e-12)):
                raise RuntimeError("image centring produced a negligible feature vector")
            out[:, self.centred_positions] = torch.einsum("cbd,ckd->bck", centred / norms, self.centred_banks)
        return out

    def score(self, features, *, with_identity_margins: bool = False, with_logits: bool = False):
        """Return (labels [draws, candidates], margins or None, top2 ids or None[, logits])."""

        import torch

        logits = self.logits(features)
        labels = torch.argmax(logits, dim=2)
        margins = ids = None
        if with_identity_margins:
            top = torch.topk(logits[:, self.identity_position, :], k=2, dim=1)
            margins = (top.values[:, 0] - top.values[:, 1]).to(dtype=torch.float32)
            ids = top.indices.to(dtype=torch.int16)
        if with_logits:
            return labels, margins, ids, logits
        return labels, margins, ids

    def accumulate(self, labels, counts) -> None:
        """counts [candidates, classes] += one-hot sum of labels [draws, candidates]."""

        import torch

        flat = (labels + torch.arange(self.count, device=self.device) * self.class_count).reshape(-1)
        counts += torch.bincount(flat, minlength=self.count * self.class_count).reshape(
            self.count, self.class_count
        ).to(dtype=counts.dtype)

    def flip_counts(self, logits, labels, truth: int):
        """Per-candidate counters for one batch of confirmation draws (int64 tensors of length ``count``)."""

        import torch

        identity = self.identity_position
        scores = logits[:, identity, :]
        winners = labels[:, identity]
        changed = labels != winners[:, None]
        truth_tensor = torch.as_tensor(int(truth), device=self.device)
        output = {
            "flip_changed_draws": changed.sum(dim=0),
            "flip_useful_draws": (changed & (winners[:, None] != truth_tensor) & (labels == truth_tensor)).sum(dim=0),
            "flip_harmful_draws": (changed & (winners[:, None] == truth_tensor) & (labels != truth_tensor)).sum(dim=0),
        }
        containment = torch.zeros(self.count, dtype=torch.int64, device=self.device)
        loose = torch.zeros(self.count, dtype=torch.int64, device=self.device)
        budget = torch.zeros(self.count, dtype=torch.int64, device=self.device)
        plan = self.flip
        if plan.size:
            candidate_labels = labels[:, plan.positions]
            candidate_changed = candidate_labels != winners[:, None]
            pair = scores.gather(1, winners[:, None]) - scores.gather(1, candidate_labels)
            classes = plan.classes
            threshold = plan.thresholds[(plan.plan_index[None, :] * classes + winners[:, None]) * classes + candidate_labels]
            scale = plan.scales[plan.plan_index[None, :] * classes + winners[:, None]]
            violations = candidate_changed & (scale * pair > threshold + plan.tolerance)
            top = torch.topk(scores, k=2, dim=1).values
            margin = (top[:, 0] - top[:, 1])[:, None]
            within = margin <= plan.loose_bounds[None, :] + plan.tolerance
            containment[plan.positions] = violations.sum(dim=0)
            loose[plan.positions] = (candidate_changed & plan.loose_mask[None, :] & ~within).sum(dim=0)
            budget[plan.positions] = (plan.loose_mask[None, :] & within).sum(dim=0)
        output["flip_containment_violations"] = containment
        output["flip_loose_violations"] = loose
        output["flip_loose_budget_draws"] = budget
        return output


# --------------------------------------------------------------------------- sampling


def rng_selftest(device: str, shape: tuple[int, ...], seed: int = 20260925) -> str:
    import torch

    generator = torch.Generator(device=device)
    generator.manual_seed(derive_block_seed(seed, 0))
    values = torch.randn(shape, generator=generator, dtype=torch.float32, device=device)
    return hashlib.sha256(values.cpu().numpy().tobytes(order="C")).hexdigest()


def count_item(
    *, encode: Callable, scorer: Scorer, pixel, mean, std, item_seed: int, draws: int, sigma: float,
    blocks_per_forward: int, device: str, store_margins: bool = False, counters: bool = False,
    truth: int | None = None, checkpoints: tuple[int, ...] = (),
):
    """Vote counts (and optional margins, per-draw counters, prefix counts) for one item over ``draws`` encodes."""

    import torch

    if isinstance(blocks_per_forward, bool) or not isinstance(blocks_per_forward, int) or blocks_per_forward <= 0:
        raise ValueError("blocks_per_forward must be a positive integer")
    if counters and truth is None:
        raise ValueError("per-draw counters need the item's ground truth")
    counts = torch.zeros((scorer.count, scorer.class_count), dtype=torch.int64, device=device)
    prefix = torch.zeros((len(checkpoints), scorer.count, scorer.class_count), dtype=torch.int64, device=device)
    totals = {name: torch.zeros(scorer.count, dtype=torch.int64, device=device) for name in FLIP_COUNTERS} if counters else None
    margins = np.empty(draws, dtype=np.float32) if store_margins else None
    top2 = np.empty((draws, 2), dtype=np.int16) if store_margins else None
    filled = 0
    plan = block_plan(draws)
    for start in range(0, len(plan), blocks_per_forward):
        chunk = plan[start : start + blocks_per_forward]
        pieces = []
        for block_index, block_draws in chunk:
            generator = torch.Generator(device=device)
            generator.manual_seed(derive_block_seed(item_seed, block_index))
            noise = torch.randn((NOISE_BLOCK, *pixel.shape), generator=generator, dtype=torch.float32, device=device)
            pieces.append(noise[:block_draws])
        noise = torch.cat(pieces, dim=0)
        standardized = ((pixel.unsqueeze(0) + float(sigma) * noise) - mean) / std
        with torch.inference_mode():
            features = encode(standardized).to(dtype=torch.float32)
        features = features / torch.linalg.vector_norm(features, dim=1, keepdim=True)
        labels, margin, ids, logits = scorer.score(features, with_identity_margins=store_margins, with_logits=True)
        scorer.accumulate(labels, counts)
        size = int(labels.shape[0])
        for index, checkpoint in enumerate(checkpoints):
            take = min(size, checkpoint - filled)
            if take > 0:
                scorer.accumulate(labels[:take], prefix[index])
        if counters:
            batch = scorer.flip_counts(logits, labels, int(truth))
            for name in FLIP_COUNTERS:
                totals[name] += batch[name]
        if store_margins:
            margins[filled : filled + size] = margin.cpu().numpy()
            top2[filled : filled + size] = ids.cpu().numpy()
        filled += size
    if filled != draws or not bool(torch.all(counts.sum(dim=1) == draws)):
        raise RuntimeError("vote totals do not match the draw budget")
    for index, checkpoint in enumerate(checkpoints):
        if not bool(torch.all(prefix[index].sum(dim=1) == checkpoint)):
            raise RuntimeError("prefix vote totals do not match the budget checkpoint")
    output = {
        "counts": counts.cpu().numpy().astype(np.int32),
        "margins": margins,
        "top2": top2,
        "prefix": prefix.cpu().numpy().astype(np.int32),
        "counters": None if totals is None else {name: value.cpu().numpy().astype(np.int32) for name, value in totals.items()},
    }
    return output


def clean_features_for(encode: Callable, pixels, mean, std, device: str, batch: int = 64):
    import torch

    clean = []
    with torch.inference_mode():
        for start in range(0, pixels.shape[0], batch):
            standardized = (pixels[start : start + batch].to(device) - mean) / std
            features = encode(standardized).to(dtype=torch.float32)
            clean.append(features / torch.linalg.vector_norm(features, dim=1, keepdim=True))
    return torch.cat(clean)


# --------------------------------------------------------------------------- checkpoint files


def replace_with_retry(temporary: Path, target: Path, *, attempts: int = 8, delay: float = 0.25) -> None:
    """``os.replace`` that tolerates transient Windows sharing violations (AV/OneDrive scans)."""

    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            os.replace(temporary, target)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(delay * (attempt + 1))
    raise RuntimeError(f"could not replace {target} after {attempts} attempts: {last_error}") from last_error


def save_npz_atomic(path: Path, **arrays) -> None:
    """Write an NPZ through a temporary file and an atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    replace_with_retry(temporary, path)


def load_npz_closed(path: Path) -> dict[str, np.ndarray]:
    """Read every array into memory and close the archive (an open handle blocks ``os.replace`` on Windows)."""

    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key]) for key in archive.files}


def margin_store_path(shard_path: Path) -> Path:
    return shard_path.with_name(shard_path.stem + "_margins.npz")


CUSTODY_KEYS = ("bank_manifest_sha256", "bank_file_sha256", "data_preflight_sha256", "checkpoint_sha256",
                "model_state_sha256")


def validate_resume_metadata(prior: Mapping[str, Any], registration: Mapping[str, Any], cell: Mapping[str, Any],
                             scorer_meta: Mapping[str, Any], environment: Mapping[str, Any],
                             custody: Mapping[str, Any] | None = None) -> None:
    if prior.get("schema_version") != SHARD_META_SCHEMA:
        raise RuntimeError("resume checkpoint metadata schema differs from this worker")
    if (prior.get("experiment_id") != registration["experiment_id"]
            or prior.get("registration_sha256") != registration["registration_sha256"]
            or prior.get("cell", {}).get("cell_id") != cell["cell_id"]):
        raise RuntimeError("resume checkpoint metadata differs from the active registration/cell")
    if prior.get("candidate_metadata") != json_safe(scorer_meta):
        raise RuntimeError("resume checkpoint candidate metadata differs from the current bank artifact")
    prior_env = prior.get("environment") or {}
    if prior_env.get("device_name") != environment.get("device_name"):
        raise RuntimeError(
            f"resume on a different device model ({prior_env.get('device_name')!r} -> {environment.get('device_name')!r}); "
            "every shard of a cell must run on one GPU class"
        )
    if prior_env.get("tf32_matmul_allowed") or prior_env.get("tf32_cudnn_allowed"):
        raise RuntimeError("resume checkpoint was produced with TF32 enabled")
    for key in CUSTODY_KEYS:
        if (custody or {}).get(key) != (prior.get("custody") or {}).get(key):
            raise RuntimeError(f"resume checkpoint custody field {key} differs")


def _save_shard(path: Path, *, candidate_ids, rows, offset, selection, confirmation, done, raw_predictions,
                seeds_selection, seeds_confirmation, cell, registration, scorer_meta, environment, approval_sha256,
                started, store_margins: bool, margins=None, top2=None, confirmation_base=None,
                item_hash: str = "", role: str = "evaluation", flip=None, prefix=None, checkpoints=(),
                flip_summary=None, custody=None, shard_index=0, shard_count=1) -> None:
    arrays = dict(
        candidate_ids=np.array(candidate_ids),
        item_ids=np.array([row["item_id"] for row in rows]),
        dataset_indices=np.array([int(row["dataset_index"]) for row in rows], dtype=np.int64),
        ground_truth=np.array([int(row["label"]) for row in rows], dtype=np.int64),
        raw_predictions=raw_predictions,
        selection_counts=selection,
        confirmation_counts=confirmation,
        done=done,
        item_offset=np.array([offset], dtype=np.int64),
        selection_seeds=seeds_selection,
        confirmation_seeds=seeds_confirmation,
        elapsed_seconds=np.array([time.time() - started]),
    )
    if flip is not None:
        arrays.update(flip)
    if prefix is not None and len(checkpoints):
        arrays["budget_checkpoints"] = np.array(list(checkpoints), dtype=np.int64)
        arrays["prefix_confirmation_counts"] = prefix
    save_npz_atomic(path, **arrays)
    if store_margins and margins is not None:
        save_npz_atomic(
            margin_store_path(path),
            schema_version=np.array([MARGIN_STORE_SCHEMA]),
            margins=margins,
            top_classes=top2,
            item_ids=np.array([row["item_id"] for row in rows]),
            ground_truth=np.array([int(row["label"]) for row in rows], dtype=np.int64),
            done=done,
            item_offset=np.array([offset], dtype=np.int64),
            identity_candidate_id=np.array([identity_candidate_id(registration)]),
            experiment_id=np.array([registration["experiment_id"]]),
            registration_sha256=np.array([registration["registration_sha256"]]),
            cell_id=np.array([cell["cell_id"]]),
            sigma=np.array([float(cell["sigma"])], dtype=np.float64),
            stream_role=np.array([CONFIRMATION_STREAM]),
            base_seed=np.array([int(confirmation_base)], dtype=np.int64),
            confirmation_seeds=seeds_confirmation,
        )
    write_json_atomic(
        path.with_suffix(".meta.json"),
        {
            "schema_version": SHARD_META_SCHEMA,
            "experiment_id": registration["experiment_id"],
            "registration_sha256": registration["registration_sha256"],
            "approval_sha256": approval_sha256,
            "cell": cell,
            "shard_index": int(shard_index),
            "shard_count": int(shard_count),
            "item_offset": int(offset),
            "item_role": role,
            "item_list_sha256": item_hash,
            "scored_candidate_ids": list(candidate_ids),
            "candidate_metadata": json_safe(scorer_meta),
            "identity_candidate_id": identity_candidate_id(registration),
            "margins_stored": bool(store_margins),
            "flip_counters": flip_summary,
            "budget_checkpoints": list(checkpoints),
            "custody": dict(custody or {}),
            "environment": dict(environment),
            "items_done": int(done.sum()),
            "items_total": int(len(done)),
            "sealed_evaluation_access": bool(cell.get("consumes_sealed_split", False)),
        },
    )


def process_shard(
    *,
    registration: Mapping[str, Any],
    cell: Mapping[str, Any],
    shard_rows: list[dict[str, str]],
    offset: int,
    shard_index: int,
    shard_count: int,
    banks: Mapping[str, Mapping[str, Any]],
    candidate_ids: list[str],
    scorer_meta: Mapping[str, Any],
    encode: Callable,
    pixels,
    mean,
    std,
    device: str,
    out_dir: Path,
    blocks_per_forward: int,
    checkpoint_every: int,
    store_margins: bool,
    environment: Mapping[str, Any],
    approval_sha256: str,
    item_hash: str = "",
    role: str = "evaluation",
    custody: Mapping[str, Any] | None = None,
    stop_after_items: int | None = None,
    log: Callable[[str], None] = print,
    scoring_ids: list[str] | None = None,
) -> Path:
    """Sample one shard (resuming from an existing checkpoint) and return the shard path.

    ``scoring_ids`` (021C) is the parent's complete ordered candidate stack; the
    scorer runs on all of it and only ``candidate_ids`` are stored.
    """

    certification = certification_parameters(registration)
    selection_draws = certification["selection_draws"]
    confirmation_draws = certification["confirmation_draws"]
    checkpoints = tuple(certification["budget_checkpoints"])
    if selection_draws <= 0 or confirmation_draws <= 0:
        raise RuntimeError("both draw budgets must be positive")
    if confirmation_draws > MAX_CONFIRMATION_DRAWS:
        raise RuntimeError(f"confirmation budget {confirmation_draws} exceeds the supported {MAX_CONFIRMATION_DRAWS}")
    if certification["noise_block_draws"] != NOISE_BLOCK:
        raise RuntimeError("registered noise block size differs from the frozen NOISE_BLOCK")
    selection_base = certification["selection_base_seed"]
    confirmation_base = certification["confirmation_base_seed"]
    selection_role = certification["selection_stream_role"]
    confirmation_role = certification["confirmation_stream_role"]
    if selection_role != SELECTION_STREAM or confirmation_role != CONFIRMATION_STREAM:
        raise RuntimeError("registered stream roles differ from the frozen seed roles")
    model_id, dataset_id, sigma = str(cell["model_id"]), str(cell["dataset_id"]), float(cell["sigma"])
    class_count = int(cell["class_count"])
    identity = identity_candidate_id(registration)

    shard_path = out_dir / "cells" / cell["cell_id"] / f"shard_{shard_index:03d}_of_{shard_count:03d}.npz"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    n_items, n_cand = len(shard_rows), len(candidate_ids)
    selection = np.zeros((n_cand, n_items, class_count), dtype=np.int32)
    confirmation = np.zeros((n_cand, n_items, class_count), dtype=np.int32)
    prefix = np.zeros((len(checkpoints), n_cand, n_items, class_count), dtype=np.int32)
    flip = {name: np.zeros((n_cand, n_items), dtype=np.int32) for name in FLIP_COUNTERS}
    done = np.zeros(n_items, dtype=bool)
    seeds_selection = np.zeros(n_items, dtype=np.int64)
    seeds_confirmation = np.zeros(n_items, dtype=np.int64)
    margins = np.zeros((n_items, confirmation_draws), dtype=np.float32) if store_margins else None
    top2 = np.zeros((n_items, confirmation_draws, 2), dtype=np.int16) if store_margins else None
    expected_ids = [row["item_id"] for row in shard_rows]

    scoring = list(scoring_ids) if scoring_ids is not None else list(candidate_ids)
    absent = [candidate_id for candidate_id in candidate_ids if candidate_id not in scoring]
    if absent:
        raise RuntimeError(f"stored candidates {absent} are not in the scoring stack")
    keep = np.array([scoring.index(candidate_id) for candidate_id in candidate_ids], dtype=np.int64)
    scorer = Scorer(banks, scoring, identity, device, flip_tolerance=certification["flip_tolerance"])
    stack_summary = scorer.flip.summary(scoring)
    flip_summary = {
        "tolerance": stack_summary["tolerance"],
        "containment_applicable": [value for value in stack_summary["containment_applicable"] if value in candidate_ids],
        "loose_applicable": [value for value in stack_summary["loose_applicable"] if value in candidate_ids],
    }
    if scoring != list(candidate_ids):
        flip_summary["scoring_stack"] = scoring
    flip["flip_containment_applicable"] = scorer.flip.containment_applicable[keep].copy()
    flip["flip_loose_applicable"] = scorer.flip.loose_applicable[keep].copy()

    if shard_path.is_file():
        meta_path = shard_path.with_suffix(".meta.json")
        if not meta_path.is_file():
            raise RuntimeError("resume checkpoint metadata sidecar is absent")
        validate_resume_metadata(read_json(meta_path), registration, cell, scorer_meta, environment, custody)
        previous = load_npz_closed(shard_path)
        checks = {
            "candidate_ids": list(previous["candidate_ids"]) == candidate_ids,
            "item_ids": list(previous["item_ids"]) == expected_ids,
            "dataset_indices": np.array_equal(previous["dataset_indices"], np.array([int(row["dataset_index"]) for row in shard_rows])),
            "ground_truth": np.array_equal(previous["ground_truth"], np.array([int(row["label"]) for row in shard_rows])),
            "item_offset": int(previous["item_offset"][0]) == offset,
            "selection_shape": previous["selection_counts"].shape == selection.shape,
            "confirmation_shape": previous["confirmation_counts"].shape == confirmation.shape,
            "done_shape": previous["done"].shape == done.shape,
            "counters": all(name in previous and previous[name].shape == flip[name].shape for name in FLIP_COUNTERS),
            "prefix": (not checkpoints) or ("prefix_confirmation_counts" in previous
                                            and previous["prefix_confirmation_counts"].shape == prefix.shape),
        }
        failed = [name for name, value in checks.items() if not value]
        if failed:
            raise RuntimeError(f"resume checkpoint provenance mismatch: {failed}")
        selection, confirmation, done = previous["selection_counts"], previous["confirmation_counts"], previous["done"]
        for name in FLIP_COUNTERS:
            flip[name] = previous[name].copy()
        if checkpoints:
            prefix = previous["prefix_confirmation_counts"].copy()
        seeds_selection = previous["selection_seeds"].copy()
        seeds_confirmation = previous["confirmation_seeds"].copy()
        if bool(np.any(done & (seeds_selection == 0))) or bool(np.any(done & (seeds_confirmation == 0))):
            raise RuntimeError("completed resume rows contain missing seeds")
        if store_margins and bool(done.any()):
            store = margin_store_path(shard_path)
            if not store.is_file():
                raise RuntimeError("cannot resume with --store-margins: checkpoint exists but its margin store is absent")
            saved = load_npz_closed(store)
            identity_checks = {
                "schema": str(saved["schema_version"][0]) == MARGIN_STORE_SCHEMA,
                "shape": saved["margins"].shape == (n_items, confirmation_draws),
                "top_shape": saved["top_classes"].shape == (n_items, confirmation_draws, 2),
                "dtype": saved["margins"].dtype == np.float32 and saved["top_classes"].dtype == np.int16,
                "done": np.array_equal(saved["done"], done),
                "item_ids": list(saved["item_ids"]) == expected_ids,
                "registration": str(saved["registration_sha256"][0]) == registration["registration_sha256"],
                "cell": str(saved["cell_id"][0]) == cell["cell_id"],
                "identity": str(saved["identity_candidate_id"][0]) == identity,
                "base_seed": int(saved["base_seed"][0]) == confirmation_base,
                "seeds": np.array_equal(saved["confirmation_seeds"], seeds_confirmation),
            }
            failed = [name for name, value in identity_checks.items() if not value]
            if failed:
                raise RuntimeError(f"margin store provenance mismatch: {failed}")
            margins = saved["margins"].astype(np.float32, copy=True)
            top2 = saved["top_classes"].astype(np.int16, copy=True)
        log(f"resuming: {int(done.sum())}/{n_items} items already finished")

    clean = clean_features_for(encode, pixels, mean, std, device)
    raw_predictions = scorer.score(clean)[0].cpu().numpy().T.astype(np.int64)[keep]  # [candidates, items]
    if raw_predictions.shape != (n_cand, n_items):
        raise RuntimeError("raw prediction matrix has the wrong shape")

    started = time.time()
    processed = 0
    save_kwargs = dict(
        candidate_ids=candidate_ids, rows=shard_rows, offset=offset, cell=cell, registration=registration,
        scorer_meta=scorer_meta, environment=environment, approval_sha256=approval_sha256, store_margins=store_margins,
        confirmation_base=confirmation_base, item_hash=item_hash, role=role, checkpoints=checkpoints,
        flip_summary=flip_summary, custody=custody, shard_index=shard_index, shard_count=shard_count,
    )
    for position, row in enumerate(shard_rows):
        if done[position]:
            continue
        pixel = pixels[position].to(device)
        item_seed = derive_item_seed(base_seed=selection_base, model_id=model_id, dataset_id=dataset_id,
                                     sigma=sigma, item_id=row["item_id"], stream_role=selection_role)
        confirmation_seed = derive_item_seed(base_seed=confirmation_base, model_id=model_id, dataset_id=dataset_id,
                                             sigma=sigma, item_id=row["item_id"], stream_role=confirmation_role)
        if confirmation_seed == item_seed:
            raise RuntimeError("selection and confirmation seed collision")
        seeds_selection[position] = item_seed
        seeds_confirmation[position] = confirmation_seed
        selection[:, position] = count_item(
            encode=encode, scorer=scorer, pixel=pixel, mean=mean, std=std, item_seed=item_seed,
            draws=selection_draws, sigma=sigma, blocks_per_forward=blocks_per_forward, device=device,
        )["counts"][keep]
        result = count_item(
            encode=encode, scorer=scorer, pixel=pixel, mean=mean, std=std, item_seed=confirmation_seed,
            draws=confirmation_draws, sigma=sigma, blocks_per_forward=blocks_per_forward, device=device,
            store_margins=store_margins, counters=True, truth=int(row["label"]), checkpoints=checkpoints,
        )
        confirmation[:, position] = result["counts"][keep]
        for name in FLIP_COUNTERS:
            flip[name][:, position] = result["counters"][name][keep]
        if checkpoints:
            prefix[:, :, position] = result["prefix"][:, keep]
        if store_margins:
            margins[position] = result["margins"]
            top2[position] = result["top2"]
        done[position] = True
        processed += 1
        if processed % checkpoint_every == 0 or position == n_items - 1:
            _save_shard(shard_path, selection=selection, confirmation=confirmation, done=done,
                        raw_predictions=raw_predictions, seeds_selection=seeds_selection,
                        seeds_confirmation=seeds_confirmation, started=started, margins=margins, top2=top2,
                        flip=flip, prefix=prefix, **save_kwargs)
            rate = processed * (selection_draws + confirmation_draws) / max(time.time() - started, 1e-9)
            remaining = (n_items - int(done.sum())) * (selection_draws + confirmation_draws) / max(rate, 1e-9) / 60
            log(f"  {int(done.sum())}/{n_items} items  {rate:,.0f} encodes/s  eta {remaining:.1f} min")
        if stop_after_items is not None and processed >= stop_after_items and not bool(done.all()):
            _save_shard(shard_path, selection=selection, confirmation=confirmation, done=done,
                        raw_predictions=raw_predictions, seeds_selection=seeds_selection,
                        seeds_confirmation=seeds_confirmation, started=started, margins=margins, top2=top2,
                        flip=flip, prefix=prefix, **save_kwargs)
            raise StopRequested(f"stopped after {processed} new items (test hook)")

    _save_shard(shard_path, selection=selection, confirmation=confirmation, done=done,
                raw_predictions=raw_predictions, seeds_selection=seeds_selection,
                seeds_confirmation=seeds_confirmation, started=started, margins=margins, top2=top2,
                flip=flip, prefix=prefix, **save_kwargs)
    log(f"done: {shard_path}")
    return shard_path


# --------------------------------------------------------------------------- data loading (real runs)


def load_pixels_for_cell(cell: Mapping[str, Any], shard_rows: list[dict[str, str]], data_root: Path, loaded,
                         phase3_assignments: Path | None, preflight: Mapping[str, Any] | None = None):
    """Pixels in [0, 1] for the shard rows; every file-backed image must match the preflight hash."""

    from satml2027ext.preflight_data_ext import verify_image_file

    dataset_id = str(cell["dataset_id"])
    ext = importlib.import_module("satml2027ext.datasets_ext")
    if dataset_id in getattr(ext, "DATASET_REGISTRY_EXT", {}):
        kwargs: dict[str, Any] = {}
        if dataset_id == "eurosat_sealed":
            kwargs["phase3_assignments"] = phase3_assignments or (PROJECT_ROOT / DEFAULT_PHASE3_ASSIGNMENTS)
        if dataset_id == "imagenette":
            kwargs["split"] = str(cell.get("evaluation_split", "val"))
        dataset, item_ids, labels = ext.load_dataset_ext(dataset_id, data_root, transform=loaded.preprocess, **kwargs)
    else:
        from common.datasets import load_dataset  # noqa: E402

        dataset, item_ids, labels = load_dataset(dataset_id, data_root, transform=loaded.preprocess)
    from common.datasets import recover_pixels  # noqa: E402

    positions: list[int] = []
    index_of = None
    for row in shard_rows:
        index = int(row["dataset_index"])
        if dataset_id == "eurosat_sealed":
            # the sealed dataset object is indexed by the sealed list order, not by the EuroSAT dataset index
            if index_of is None:
                index_of = {item_id: position for position, item_id in enumerate(item_ids)}
            index = index_of[row["item_id"]]
        if item_ids[index] != row["item_id"] or int(labels[index]) != int(row["label"]):
            raise RuntimeError(f"dataset identity differs from the registered item list at {row['item_id']}")
        if preflight is not None:
            verify_image_file(preflight, data_root, row["item_id"])
        positions.append(index)
    return recover_pixels(dataset, positions, loaded.mean, loaded.std)


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None, *, approval_path: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--parent-registration", type=Path, default=None, help="021C only: the 021B registration")
    parser.add_argument("--cell-id", type=str)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--items", type=Path, default=PROJECT_ROOT / "results/satml2027ext/items")
    parser.add_argument("--banks", type=Path, default=PROJECT_ROOT / "results/satml2027ext/banks")
    parser.add_argument("--bank-manifest", type=Path)
    parser.add_argument("--data-preflight", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--phase3-assignments", type=Path, default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--blocks-per-forward", type=int, default=None,
                        help=f"must equal the registered certification.blocks_per_forward (forward batch = this x {NOISE_BLOCK} draws)")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--rng-selftest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision_extra = configure_precision()

    if args.rng_selftest:
        for shape in ((NOISE_BLOCK, 3, 224, 224), (NOISE_BLOCK, 3, 32, 32)):
            print(f"rng digest {shape}: {rng_selftest(device, shape)}")
        return 0

    for name in ("cell_id", "out", "bank_manifest", "data_preflight", "data_root"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")

    from satml2027ext import bank_manifest_ext, models_ext
    from satml2027ext.preflight_data_ext import load_preflight, verify_environment

    registration = load_registration(args.registration)
    approval = verify_approval(registration, stage="sampling", approval_path=approval_path or DEFAULT_APPROVAL_PATH,
                               stage_files={"data_preflight": args.data_preflight, "bank_manifest": args.bank_manifest})
    preflight = load_preflight(args.data_preflight, registrations=approval["registrations"])
    verify_environment(preflight)
    models_ext.require_offline()
    manifest = bank_manifest_ext.load_manifest(args.bank_manifest)
    if manifest.get("data_preflight_sha256") != approval["data_preflight_sha256"]:
        raise RuntimeError("the bank manifest was built from another data preflight than the approved one")
    registered_blocks = certification_parameters(registration)["blocks_per_forward"]
    if not isinstance(registered_blocks, int):
        raise RuntimeError("the registration does not register certification.blocks_per_forward")
    if args.blocks_per_forward is not None and args.blocks_per_forward != registered_blocks:
        raise RuntimeError(f"--blocks-per-forward {args.blocks_per_forward} differs from the registered {registered_blocks}")
    args.blocks_per_forward = registered_blocks
    cell = find_cell(registration, args.cell_id)
    entry = bank_manifest_ext.manifest_entry(manifest, registration, cell["cell_id"])

    rows, item_record = items_ext.load_verified(args.items, registration, cell, "evaluation", preflight=preflight)
    offset, shard_rows = shard_slice(rows, args.shard_index, args.shard_count)
    if len(shard_rows) > max_items_per_shard(registration):
        raise RuntimeError(f"shard of {len(shard_rows)} items exceeds the registered maximum {max_items_per_shard(registration)}")

    bank_path = args.banks / entry["bank_file"]
    if sha256_file(bank_path) != entry["bank_file_sha256"]:
        raise RuntimeError("bank file differs from the approved bank manifest entry")
    payload = importlib.import_module("satml2027ext.banks_ext").load_bank_payload(bank_path)
    parent_registration = parent_cell = None
    if candidates_ext.is_subset_registration(registration):
        if args.parent_registration is None:
            parser.error("--parent-registration is required for a subset registration (021C)")
        parent_registration = load_registration(args.parent_registration)
        parent_cell = find_cell(parent_registration, cell["parent_cell_id"])
    candidate_ids, banks, scorer_meta = load_and_validate_banks(
        payload, registration, cell, parent_registration=parent_registration, parent_cell=parent_cell)
    if {key: scorer_meta[key]["prototype_sha256"] for key in candidate_ids} != {key: entry["prototype_sha256"][key] for key in candidate_ids}:
        raise RuntimeError("scored prototypes differ from the bank manifest")
    scoring_ids = None
    if parent_registration is not None:
        # V3 (re-review n1): score the parent's whole stack so the GEMM shapes equal 021B's
        scoring_ids, stack_banks, stack_meta = load_and_validate_banks(payload, parent_registration, parent_cell)
        if any(stack_meta[key]["prototype_sha256"] != scorer_meta[key]["prototype_sha256"] for key in candidate_ids):
            raise RuntimeError("the parent stack's kept banks differ from the validated subset")
        banks = stack_banks
    environment = environment_metadata(device, args.blocks_per_forward, precision_extra)
    custody = {
        "bank_manifest_sha256": sha256_file(args.bank_manifest),
        "bank_file_sha256": entry["bank_file_sha256"],
        "data_preflight_sha256": sha256_file(args.data_preflight),
        "checkpoint_sha256": registration["models"][str(cell["model_id"])]["sha256"],
        "model_state_sha256": entry["model_state_sha256"],
    }
    print(f"cell={args.cell_id} shard={args.shard_index}/{args.shard_count} items={len(shard_rows)} "
          f"banks={len(candidate_ids)} device={environment['device_name']} "
          f"confirmation_draws={certification_parameters(registration)['confirmation_draws']}")
    if args.dry_run:
        return 0

    loaded, _ = models_ext.load_bound_model(str(cell["model_id"]), registration["models"][str(cell["model_id"])],
                                            device=device, cache_dir=args.cache_dir,
                                            expected_state_sha256=entry["model_state_sha256"])
    pixels = load_pixels_for_cell(cell, shard_rows, args.data_root, loaded, args.phase3_assignments, preflight)
    mean = torch.tensor(loaded.mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(loaded.std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    process_shard(
        registration=registration, cell=cell, shard_rows=shard_rows, offset=offset,
        shard_index=args.shard_index, shard_count=args.shard_count, banks=banks, candidate_ids=candidate_ids,
        scorer_meta=scorer_meta, encode=loaded.model.encode_image, pixels=pixels, mean=mean, std=std, device=device,
        out_dir=args.out, blocks_per_forward=args.blocks_per_forward, checkpoint_every=args.checkpoint_every,
        store_margins=bool(cell.get("store_identity_margins", False)), environment=environment,
        approval_sha256=approval["_approval_sha256"], item_hash=item_record["items_sha256"], role="evaluation",
        custody=custody, scoring_ids=scoring_ids,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
