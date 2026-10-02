"""Checkpoint binding for EXP-021 (V2; V1 review finding P0-D).

The V2 registrations name the exact checkpoint bytes of every backbone
(``registration["models"][model_id]["sha256"]``; the values are the blob hashes
recorded for EXP-019 on 2026-09-25, D-123).  Before any model is used:

1. the file that OpenCLIP will load is resolved with the same pretrained
   configuration and cache directory as ``satml2027/common/models.py::load_model``,
   with the Hugging Face hub forced offline (no silent download of a newer
   revision);
2. its SHA-256 must equal the registered value;
3. after loading, the model state is hashed (sorted parameter and buffer names,
   dtype, shape and bytes), and the preparation stage records it so that the
   sampling worker can require the identical state.

``open_clip`` is imported lazily, so this module is importable on analysis hosts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from satml2027ext._common import PROJECT_ROOT, canonical_json_hash, sha256_file  # noqa: F401  (sys.path)

OFFLINE_VARIABLES = ("HF_HUB_OFFLINE",)


class CheckpointBindingError(RuntimeError):
    """The checkpoint on disk is not the registered checkpoint (fail closed)."""


def require_offline() -> None:
    """Refuse to resolve checkpoints unless the Hugging Face hub is in offline mode."""

    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise CheckpointBindingError(
            "set HF_HUB_OFFLINE=1 before any EXP-021 stage that loads a model; the registered checkpoint must "
            "already be in the cache and must never be re-resolved against the network"
        )
    try:
        from huggingface_hub import constants
    except ImportError:  # pragma: no cover - open_clip requires huggingface_hub for these models
        return
    if not getattr(constants, "HF_HUB_OFFLINE", False):
        raise CheckpointBindingError("huggingface_hub was imported before HF_HUB_OFFLINE=1 took effect")


def resolve_checkpoint_file(model_id: str, *, cache_dir: str | None = None) -> Path:
    """The local file OpenCLIP loads for ``model_id`` (no network)."""

    require_offline()
    from common.models import resolve  # frozen model registry
    from open_clip import pretrained as open_clip_pretrained

    spec = resolve(model_id)
    config = open_clip_pretrained.get_pretrained_cfg(spec["open_clip_name"], spec["pretrained"])
    if not config:
        raise CheckpointBindingError(f"OpenCLIP has no pretrained configuration for {model_id}")
    try:
        target = open_clip_pretrained.download_pretrained(config, cache_dir=cache_dir)
    except Exception as error:  # hf_hub raises LocalEntryNotFoundError offline
        raise CheckpointBindingError(f"registered checkpoint for {model_id} is not in the local cache: {error}") from error
    path = Path(target)
    if not path.is_file():
        raise CheckpointBindingError(f"resolved checkpoint for {model_id} is not a file: {path}")
    return path


def verify_checkpoint(model_id: str, binding: Mapping[str, Any], *, cache_dir: str | None = None) -> dict[str, Any]:
    """Resolve and hash the checkpoint; it must equal the registered binding."""

    expected = binding.get("sha256") if isinstance(binding, Mapping) else None
    if not isinstance(expected, str) or len(expected) != 64:
        raise CheckpointBindingError(f"registration binds no checkpoint hash for {model_id}")
    path = resolve_checkpoint_file(model_id, cache_dir=cache_dir)
    observed = sha256_file(path.resolve())
    if observed != expected:
        raise CheckpointBindingError(f"checkpoint for {model_id} hashes to {observed}, registration binds {expected}")
    expected_name = binding.get("file")
    if isinstance(expected_name, str) and path.name != expected_name:
        raise CheckpointBindingError(f"checkpoint for {model_id} resolves to {path.name}, registration names {expected_name}")
    size = path.resolve().stat().st_size
    expected_bytes = binding.get("bytes")
    if expected_bytes is not None and int(expected_bytes) != size:
        raise CheckpointBindingError(f"checkpoint for {model_id} has {size} bytes, registration binds {expected_bytes}")
    return {"model_id": model_id, "file": path.name, "sha256": observed, "bytes": size}


def model_state_sha256(model) -> str:
    """Canonical hash of every parameter and buffer (sorted names; dtype, shape and bytes)."""

    from common.hashing import tensor_scientific_hash

    state = model.state_dict()
    return canonical_json_hash({name: tensor_scientific_hash(tensor) for name, tensor in sorted(state.items())})


def load_bound_model(model_id: str, binding: Mapping[str, Any], *, device: str, cache_dir: str | None = None,
                     expected_state_sha256: str | None = None):
    """Verify the checkpoint bytes, load through the frozen loader, hash the state.

    Returns ``(loaded, record)``.  When ``expected_state_sha256`` is given (the
    sampling stage passes the value recorded by the preparation stage) the
    loaded state must match it.
    """

    record = verify_checkpoint(model_id, binding, cache_dir=cache_dir)
    from common.models import load_model

    loaded = load_model(model_id, device=device, cache_dir=cache_dir)
    record["model_state_sha256"] = model_state_sha256(loaded.model)
    if expected_state_sha256 is not None and record["model_state_sha256"] != expected_state_sha256:
        raise CheckpointBindingError(
            f"loaded model state {record['model_state_sha256']} differs from the approved {expected_state_sha256}"
        )
    return loaded, record
