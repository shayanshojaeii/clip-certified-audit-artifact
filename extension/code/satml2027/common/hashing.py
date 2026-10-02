"""Hashing and atomic JSON I/O, byte-identical to the frozen Phase-3/Phase-4 schemes.

`canonical_json_hash` reproduces `phase4/common.py::canonical_json_hash` (sorted
keys, compact separators, ASCII).  `tensor_scientific_hash` reproduces
`analysis/exp017_stage_b_independent.py::tensor_scientific_hash`.  Both are
copied deliberately so that this bundle can verify the existing artifacts
without importing the production packages.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_id_list(item_ids: Sequence[str]) -> str:
    """Hash an ordered list of item identifiers (order is part of the identity)."""

    if any(not isinstance(value, str) or not value for value in item_ids):
        raise ValueError("item ids must be nonempty strings")
    return canonical_json_hash(list(item_ids))


def tensor_scientific_hash(value) -> str:
    """Hash dtype, shape and contiguous CPU bytes of a torch tensor or numpy array."""

    try:  # torch is optional on analysis-only machines
        import torch

        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            header = {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
            payload = tensor.numpy().tobytes(order="C")
            return _hash_with_header(header, payload)
    except ModuleNotFoundError:
        pass
    import numpy as np

    array = np.ascontiguousarray(value)
    dtype_name = {
        "float32": "torch.float32",
        "float64": "torch.float64",
        "float16": "torch.float16",
        "int64": "torch.int64",
        "int32": "torch.int32",
        "bool": "torch.bool",
    }.get(array.dtype.name, array.dtype.name)
    return _hash_with_header({"dtype": dtype_name, "shape": list(array.shape)}, array.tobytes(order="C"))


def _hash_with_header(header: dict[str, Any], payload: bytes) -> str:
    buffer = io.BytesIO()
    buffer.write(
        json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode(
            "utf-8"
        )
    )
    buffer.write(b"\0")
    buffer.write(payload)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def write_json_atomic(path: str | os.PathLike[str], value: Any) -> Path:
    """Write pretty, sorted JSON through a temporary file and an atomic replace."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target


def read_json(path: str | os.PathLike[str]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def directory_manifest(root: str | os.PathLike[str], *, exclude: Iterable[str] = ()) -> dict[str, Any]:
    """Build the frozen artifact-manifest structure for a results directory."""

    base = Path(root)
    skip = set(exclude) | {"artifact_manifest.json"}
    records = []
    for path in sorted(base.rglob("*")):
        if path.is_file() and path.name not in skip:
            records.append(
                {
                    "path": path.relative_to(base).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "schema_version": "satml2027.artifact_manifest.v1",
        "artifact_count": len(records),
        "artifact_set_sha256": canonical_json_hash(records),
        "artifacts": records,
    }
