#!/usr/bin/env python3
"""Day 1 - TF32 preparation-deviation adjudication (Addendum V1, item A6).

The deviation.  `d1_02_build_development.py` (line 244) and
`d1_03b_train_controls.py` (line 173) set only
`torch.backends.cuda.matmul.allow_tf32 = False` and leave
`torch.backends.cudnn.allow_tf32` at PyTorch's default (True), whereas the
registrations state "float32 (TF32 disabled)" and the sampling worker
`d1_05_run_shard.py` (`_configure_precision`, lines 58-64) disables both flags.
cuDNN convolutions in the development-feature and control-training feature
passes (the ViT patch-embedding convolution; every RN50 convolution) may
therefore have used TF32 on the RTX 4090.  All evaluation sampling and the N3C
feature store ran with TF32 fully disabled.

This tool compares the preparation tree the server actually produced with a
TF32-off recompute of the same approved commit, seeds and item lists, and
applies the decision rule fixed in the addendum:

  * every one of the 22 banks per cell (plus the GR-CLIP calibration image
    mean) agrees within the registered portable tolerance 2e-6, and
  * no discrete quantity differs (candidate identifiers, implied deployability
    flags, tie flags = the critical-competitor sets of the development items
    under the registered tie tolerance 1e-10, item identities and labels,
    discrete bank/control metadata),

  -> "EXECUTED_WITH_IMMATERIAL_PRECISION_DEVIATION"; otherwise
  -> "DOWNGRADED_TO_DESCRIPTIVE_EXTENSION".

It never opens a shard, merged, feature-store or analysis file: it reads only
the preparation artifacts named in `preparation_paths` and writes one JSON.

Modes
  compare         server preparation tree versus TF32-off recompute tree
  manifest-check  verify a preparation tree against a sha256sum manifest
  export-npz      mirror a `.pt` preparation tree into torch-free `.npz` files
                  (torch is required for this mode only)

A preparation tree is either the server layout (`development/`, `controls/`,
`banks/` directories holding `{cell}__development.pt`, `{cell}__controls.pt`,
`{cell}__control_train_features.pt`, `{cell}__banks.pt` and their `.json`
sidecars) or a mirror directory holding `{cell}__preparation.npz` plus
`{cell}__preparation.meta.json` written by `export-npz`.

Usage:
  python satml2027/day1/d1_10_tf32_preparation_adjudication.py compare \
      --server-root server_pull/results/satml2027 --recompute-root tf32off/results/satml2027 \
      --registrations configs/satml2027/exp-20260920-019a.json configs/satml2027/exp-20260920-019b.json \
                      configs/satml2027/n3c_v3.json \
      --server-manifest server_pull/day1_prepare_attempt2_receipt.txt.manifest.sha256 \
      --out analysis/tf32_adjudication/tf32_preparation_adjudication_v1.json
"""

from __future__ import annotations

import argparse
import math
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.hashing import canonical_json_hash, read_json, sha256_file, tensor_scientific_hash, write_json_atomic  # noqa: E402

SCHEMA_VERSION = "satml2027.tf32_preparation_adjudication.v1"
PORTABLE_REPRODUCTION_ATOL = 2e-6
TIE_TOLERANCE = 1e-10
DIRECTION_UNIT_ATOL = 2e-5  # the bank builder's own unit-norm acceptance tolerance
GR_BANK_ID = "gr_clip_style_two_sided__coefficient_1"
GR_IMAGE_TRANSFORM = "gr_clip_style_center_and_renormalize"
DECISION_IMMATERIAL = "EXECUTED_WITH_IMMATERIAL_PRECISION_DEVIATION"
DECISION_DOWNGRADE = "DOWNGRADED_TO_DESCRIPTIVE_EXTENSION"
RESULT_DIRECTORY_NAMES = frozenset({"EXP-20260920-019A", "EXP-20260920-019B", "N3C-20260920-V3", "merged"})
DEVELOPMENT_ARRAYS = ("prototypes", "clean_features", "noisy_features", "clean_direction", "noisy_direction")
CONTROL_ARRAYS = ("shared_delta", "lowrank_left", "lowrank_right")
CONTROL_TRAIN_ARRAYS = ("clean_features", "noisy_features")
DISCRETE_VALUE_KEYS = frozenset({
    "step_or_coefficient", "sigma", "noisy_temperature", "rank", "development_draws", "epochs", "batch_size",
    "learning_rate", "weight_decay", "temperature", "clean_weight", "train_draws", "noise_base_seed",
    "optimizer_seed", "minibatch_seed", "frozen_steps", "frozen_coefficients", "labels", "dataset_indices",
    "item_ids", "candidate_ids",
})
DEVIATION_STATEMENT = {
    "registered_text": "float32 (TF32 disabled); a registered precision-equivalence check is reported",
    "d1_02_build_development.py": "line 244 sets torch.backends.cuda.matmul.allow_tf32 = False only; "
                                  "torch.backends.cudnn.allow_tf32 stays at its default True",
    "d1_03b_train_controls.py": "line 173 sets torch.backends.cuda.matmul.allow_tf32 = False only; "
                                "torch.backends.cudnn.allow_tf32 stays at its default True",
    "d1_05_run_shard.py": "lines 58-64 (_configure_precision) disable both flags; every evaluation and N3C "
                          "sampling pass ran with TF32 fully disabled",
    "affected_artifacts": "development clean/noisy features, the two proposal directions, the development "
                          "image mean, the 17 non-reference frozen-grid banks, the GR-CLIP calibration mean, "
                          "the control-training feature cache and all four control banks",
    "expected_unaffected": "text prototypes (no convolution in the text encoder), item lists, seeds, the "
                           "no_correction bank, all evaluation vote counts and N3C features",
}


# --------------------------------------------------------------------------- helpers


def _torch():
    try:
        import torch  # noqa: WPS433
    except ModuleNotFoundError:
        return None
    return torch


def _to_numpy(value: Any) -> np.ndarray:
    torch = _torch()
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().numpy()
    return np.ascontiguousarray(value)


def _is_tensor(value: Any) -> bool:
    torch = _torch()
    return torch is not None and isinstance(value, torch.Tensor)


def _plain(value: Any) -> Any:
    """Convert nested metadata to JSON-compatible plain Python values (tensors become lists)."""

    if _is_tensor(value):
        return _to_numpy(value).tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def assert_not_result_path(path: Path, label: str) -> None:
    parts = set(Path(path).resolve().parts)
    if parts & RESULT_DIRECTORY_NAMES:
        raise RuntimeError(f"{label} points into a result directory; this tool never touches result files")


def registered_cells(registrations: list[Path]) -> dict[str, dict[str, Any]]:
    """Map cell_id -> {registration_sha256, experiment_id, model_id, dataset_id, sigma, class_count}."""

    cells: dict[str, dict[str, Any]] = {}
    for path in registrations:
        registration = read_json(path)
        recorded = registration.get("registration_sha256")
        body = {key: value for key, value in registration.items() if key != "registration_sha256"}
        if canonical_json_hash(body) != recorded:
            raise RuntimeError(f"registration hash mismatch: {path}")
        for cell in registration["cells"]:
            cell_id = str(cell["cell_id"])
            if cell_id in cells:
                raise RuntimeError(f"cell {cell_id} appears in more than one registration")
            cells[cell_id] = {
                "registration_sha256": recorded,
                "registration_path": str(path.as_posix()),
                "experiment_id": registration["experiment_id"],
                "model_id": cell["model_id"],
                "dataset_id": cell["dataset_id"],
                "sigma": float(cell["sigma"]),
                "class_count": int(cell["class_count"]),
            }
    return cells


# --------------------------------------------------------------------------- loading


@dataclass
class CellPreparation:
    cell_id: str
    source_format: str
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    sidecars: dict[str, Any] = field(default_factory=dict)


def preparation_paths(root: Path, cell_id: str) -> dict[str, Path]:
    return {
        "development": root / "development" / f"{cell_id}__development.pt",
        "development_json": root / "development" / f"{cell_id}__development.json",
        "controls": root / "controls" / f"{cell_id}__controls.pt",
        "controls_json": root / "controls" / f"{cell_id}__controls.json",
        "control_train_features": root / "controls" / f"{cell_id}__control_train_features.pt",
        "banks": root / "banks" / f"{cell_id}__banks.pt",
        "banks_json": root / "banks" / f"{cell_id}__banks.json",
    }


def mirror_paths(root: Path, cell_id: str) -> dict[str, Path]:
    return {
        "npz": root / f"{cell_id}__preparation.npz",
        "meta": root / f"{cell_id}__preparation.meta.json",
        "development_json": root / f"{cell_id}__development.json",
        "controls_json": root / f"{cell_id}__controls.json",
        "banks_json": root / f"{cell_id}__banks.json",
    }


def _read_optional_json(path: Path) -> Any:
    return read_json(path) if path.is_file() else None


def _load_cell_pt(root: Path, cell_id: str, paths: dict[str, Path]) -> CellPreparation:
    torch = _torch()
    if torch is None:
        raise RuntimeError("torch is required to read .pt preparation files; run export-npz on a torch host "
                           "and compare the mirrors instead")
    for key in ("development", "controls", "control_train_features", "banks"):
        if not paths[key].is_file():
            raise FileNotFoundError(f"{cell_id}: preparation file is absent: {paths[key]}")
    development = torch.load(paths["development"], map_location="cpu", weights_only=False)
    controls = torch.load(paths["controls"], map_location="cpu", weights_only=False)
    cache = torch.load(paths["control_train_features"], map_location="cpu", weights_only=False)
    bank_payload = torch.load(paths["banks"], map_location="cpu", weights_only=False)

    prepared = CellPreparation(cell_id=cell_id, source_format="pt")
    for name in DEVELOPMENT_ARRAYS:
        prepared.arrays[f"development.{name}"] = _to_numpy(development[name])
    prepared.metadata["development"] = {
        key: _plain(value) for key, value in development.items() if key not in DEVELOPMENT_ARRAYS
    }
    for name in CONTROL_ARRAYS:
        if name in controls:
            prepared.arrays[f"controls.{name}"] = _to_numpy(controls[name])
    prepared.metadata["controls"] = {key: _plain(value) for key, value in controls.items() if key not in CONTROL_ARRAYS}
    for name in CONTROL_TRAIN_ARRAYS:
        prepared.arrays[f"control_train.{name}"] = _to_numpy(cache[name])
    prepared.metadata["control_train"] = {
        key: _plain(value) for key, value in cache.items() if key not in CONTROL_TRAIN_ARRAYS
    }
    banks = bank_payload["banks"]
    bank_metadata = {}
    for candidate_id in sorted(banks):
        entry = banks[candidate_id]
        prepared.arrays[f"banks.{candidate_id}"] = _to_numpy(entry["prototypes"])
        bank_metadata[candidate_id] = _plain(entry["metadata"])
        if entry.get("image_mean") is not None:
            prepared.arrays["banks.gr_clip_image_mean"] = _to_numpy(entry["image_mean"])
    prepared.metadata["banks"] = {
        **{key: _plain(value) for key, value in bank_payload.items() if key != "banks"},
        "candidate_ids": sorted(banks),
        "metadata": bank_metadata,
    }
    prepared.sidecars = {
        "development": _read_optional_json(paths["development_json"]),
        "controls": _read_optional_json(paths["controls_json"]),
        "banks": _read_optional_json(paths["banks_json"]),
    }
    return prepared


def _load_cell_npz(root: Path, cell_id: str, paths: dict[str, Path]) -> CellPreparation:
    prepared = CellPreparation(cell_id=cell_id, source_format="npz")
    with np.load(paths["npz"], allow_pickle=False) as data:
        for key in data.files:
            prepared.arrays[key] = np.ascontiguousarray(data[key])
    prepared.metadata = read_json(paths["meta"])
    prepared.sidecars = {
        "development": _read_optional_json(paths["development_json"]),
        "controls": _read_optional_json(paths["controls_json"]),
        "banks": _read_optional_json(paths["banks_json"]),
    }
    return prepared


def load_cell(root: Path, cell_id: str) -> CellPreparation:
    paths = preparation_paths(root, cell_id)
    if any(paths[key].is_file() for key in ("development", "controls", "control_train_features", "banks")):
        return _load_cell_pt(root, cell_id, paths)
    mirror = mirror_paths(root, cell_id)
    if mirror["npz"].is_file() and mirror["meta"].is_file():
        return _load_cell_npz(root, cell_id, mirror)
    raise FileNotFoundError(f"{cell_id}: no preparation artifacts under {root}")


def export_cell_npz(prepared: CellPreparation, out: Path) -> dict[str, str]:
    """Write a torch-free mirror of one cell (arrays in one .npz, metadata in one JSON)."""

    out.mkdir(parents=True, exist_ok=True)
    paths = mirror_paths(out, prepared.cell_id)
    np.savez_compressed(paths["npz"], **prepared.arrays)
    write_json_atomic(paths["meta"], prepared.metadata)
    for key in ("development", "controls", "banks"):
        if prepared.sidecars.get(key) is not None:
            write_json_atomic(paths[f"{key}_json"], prepared.sidecars[key])
    return {"npz": str(paths["npz"]), "meta": str(paths["meta"])}


# --------------------------------------------------------------------------- comparison primitives


def array_deviation(reference: np.ndarray, other: np.ndarray, *, atol: float | None = None,
                    chunk_rows: int = 64) -> dict[str, Any]:
    """Max-abs, relative and exact-hash comparison; float64 arithmetic in row chunks."""

    ref = np.asarray(reference)
    oth = np.asarray(other)
    result: dict[str, Any] = {
        "shape": list(ref.shape),
        "dtype": str(ref.dtype),
        "shape_equal": ref.shape == oth.shape,
        "dtype_equal": str(ref.dtype) == str(oth.dtype),
        "exact_hash_equal": tensor_scientific_hash(ref) == tensor_scientific_hash(oth),
    }
    if not result["shape_equal"]:
        result.update({"max_abs": None, "relative_max_abs": None, "relative_frobenius": None,
                       "reference_max_abs": None, "nonfinite": None})
        if atol is not None:
            result["within_tolerance"] = False
        return result
    ref2 = ref.reshape(ref.shape[0], -1) if ref.ndim >= 1 and ref.size else ref.reshape(1, -1)
    oth2 = oth.reshape(ref2.shape)
    max_abs, max_ref, sq_diff, sq_ref, nonfinite = 0.0, 0.0, 0.0, 0.0, False
    for start in range(0, ref2.shape[0], chunk_rows):
        a = ref2[start:start + chunk_rows].astype(np.float64)
        b = oth2[start:start + chunk_rows].astype(np.float64)
        diff = a - b
        if diff.size == 0:
            continue
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            nonfinite = True
            continue
        max_abs = max(max_abs, float(np.max(np.abs(diff))))
        max_ref = max(max_ref, float(np.max(np.abs(a))))
        sq_diff += float(np.sum(diff * diff))
        sq_ref += float(np.sum(a * a))
    result.update({
        "max_abs": max_abs,
        "relative_max_abs": (max_abs / max_ref) if max_ref > 0.0 else None,
        "relative_frobenius": (math.sqrt(sq_diff) / math.sqrt(sq_ref)) if sq_ref > 0.0 else None,
        "reference_max_abs": max_ref,
        "nonfinite": nonfinite,
    })
    if atol is not None:
        result["within_tolerance"] = bool(not nonfinite and max_abs <= atol)
    return result


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compare_metadata(reference: Any, other: Any, path: str = "") -> dict[str, list[dict[str, Any]]]:
    """Walk two metadata trees: discrete leaves must be equal, hashes are reported, numerics get abs diffs."""

    report: dict[str, list[dict[str, Any]]] = {"discrete_mismatches": [], "hash_mismatches": [], "numeric": []}

    def record_discrete(where: str, left: Any, right: Any) -> None:
        report["discrete_mismatches"].append({"path": where, "reference": _plain(left), "other": _plain(right)})

    def walk(left: Any, right: Any, where: str, key: str) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            for name in sorted(set(left) | set(right)):
                child = f"{where}.{name}" if where else name
                if name not in left or name not in right:
                    record_discrete(child, left.get(name, "<absent>"), right.get(name, "<absent>"))
                    continue
                walk(left[name], right[name], child, name)
            return
        if isinstance(left, list) and isinstance(right, list):
            if left and right and all(_is_number(v) for v in left) and all(_is_number(v) for v in right):
                all_integers = all(isinstance(v, int) for v in left) and all(isinstance(v, int) for v in right)
                if len(left) != len(right):
                    record_discrete(where, f"len={len(left)}", f"len={len(right)}")
                elif key in DISCRETE_VALUE_KEYS or all_integers:
                    if left != right:
                        record_discrete(where, left, right)
                else:
                    diff = float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))))
                    report["numeric"].append({"path": where, "max_abs": diff, "exact_equal": left == right})
                return
            if len(left) != len(right):
                record_discrete(where, f"len={len(left)}", f"len={len(right)}")
                return
            for index, (a, b) in enumerate(zip(left, right)):
                walk(a, b, f"{where}[{index}]", key)
            return
        if isinstance(left, str) and isinstance(right, str) and key.endswith("sha256"):
            if left != right:
                report["hash_mismatches"].append({"path": where, "reference": left, "other": right})
            return
        if isinstance(left, float) and isinstance(right, float) and key not in DISCRETE_VALUE_KEYS:
            report["numeric"].append({"path": where, "max_abs": abs(left - right), "exact_equal": left == right})
            return
        if _is_number(left) and _is_number(right):
            if left != right:
                record_discrete(where, left, right)
            return
        if left != right:
            record_discrete(where, left, right)

    walk(reference, other, path, path.rsplit(".", 1)[-1] if path else "")
    return report


def critical_competitor_sets(prototypes: np.ndarray, clean_features: np.ndarray,
                             tie_tolerance: float = TIE_TOLERANCE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Protected class, critical-competitor set and tie flag per development item (float64, first-max argmax).

    This is the documented selection rule of `clean_boundary_active_direction` in
    d1_02_build_development.py: protected = argmax score; competitors whose
    margin lies within `tie_tolerance` of the smallest margin are critical.  An
    item is tie-flagged when more than one competitor is critical or when the
    protected class is itself tied with its nearest competitor within the
    tolerance (the clean-prediction tie of the registered builders).
    """

    bank = np.asarray(prototypes, dtype=np.float64)
    features = np.asarray(clean_features, dtype=np.float64)
    scores = features @ bank.T
    protected = np.argmax(scores, axis=1)
    rows = np.arange(scores.shape[0])
    margins = scores[rows, protected][:, None] - scores
    margins[rows, protected] = np.inf
    minimum = margins.min(axis=1)
    critical = margins <= (minimum[:, None] + tie_tolerance)
    tie_flag = (critical.sum(axis=1) > 1) | (minimum <= tie_tolerance)
    return protected, critical, tie_flag


def bank_argmax(prepared: CellPreparation, candidate_id: str) -> np.ndarray | None:
    prototypes = prepared.arrays.get(f"banks.{candidate_id}")
    features = prepared.arrays.get("development.clean_features")
    if prototypes is None or features is None:
        return None
    feats = np.asarray(features, dtype=np.float64)
    if candidate_id == GR_BANK_ID:
        mean = prepared.arrays.get("banks.gr_clip_image_mean")
        if mean is None:
            return None
        centered = feats - np.asarray(mean, dtype=np.float64)[None, :]
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            return None
        feats = centered / norms
    return np.argmax(feats @ np.asarray(prototypes, dtype=np.float64).T, axis=1)


def implied_deployability(prepared: CellPreparation) -> dict[str, Any]:
    """d1_02 raises unless both directions are deployable, so their presence implies True; verify unit norm."""

    out: dict[str, Any] = {}
    for name in ("clean_direction", "noisy_direction"):
        vector = prepared.arrays.get(f"development.{name}")
        if vector is None:
            out[name] = {"present": False, "finite": None, "unit_norm": None, "implied_deployable": False}
            continue
        vector = np.asarray(vector, dtype=np.float64)
        finite = bool(np.isfinite(vector).all())
        norm = float(np.linalg.norm(vector)) if finite else float("nan")
        unit = bool(finite and abs(norm - 1.0) <= DIRECTION_UNIT_ATOL)
        out[name] = {"present": True, "finite": finite, "norm": norm, "unit_norm": unit,
                     "implied_deployable": bool(finite and unit)}
    return out


# --------------------------------------------------------------------------- per-cell comparison


def compare_cell(reference: CellPreparation, other: CellPreparation, *, atol: float,
                 expected_registration_sha256: str | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "cell_id": reference.cell_id,
        "source_formats": {"reference": reference.source_format, "other": other.source_format},
        "arrays": {},
        "banks": {},
        "discrete_differences": [],
        "informational": {},
    }
    discrete = report["discrete_differences"]

    # registration binding
    for section in ("development", "controls", "banks"):
        left = reference.metadata.get(section, {}).get("registration_sha256")
        right = other.metadata.get(section, {}).get("registration_sha256")
        if expected_registration_sha256 is not None and (left != expected_registration_sha256
                                                         or right != expected_registration_sha256):
            discrete.append({"path": f"{section}.registration_sha256", "reference": left, "other": right,
                             "expected": expected_registration_sha256})

    # candidate identifiers
    left_ids = list(reference.metadata.get("banks", {}).get("candidate_ids", []))
    right_ids = list(other.metadata.get("banks", {}).get("candidate_ids", []))
    report["candidate_ids"] = {"reference": left_ids, "other": right_ids, "equal": left_ids == right_ids,
                               "count": len(left_ids)}
    if left_ids != right_ids:
        discrete.append({"path": "banks.candidate_ids", "reference": left_ids, "other": right_ids})

    # arrays: everything except banks is informational; banks and the GR mean carry the tolerance
    for name in sorted(set(reference.arrays) | set(other.arrays)):
        if name not in reference.arrays or name not in other.arrays:
            discrete.append({"path": f"arrays.{name}", "reference": name in reference.arrays,
                             "other": name in other.arrays})
            continue
        is_bank = name.startswith("banks.")
        deviation = array_deviation(reference.arrays[name], other.arrays[name], atol=atol if is_bank else None)
        if not deviation["shape_equal"]:
            discrete.append({"path": f"arrays.{name}.shape", "reference": deviation["shape"],
                             "other": list(np.asarray(other.arrays[name]).shape)})
        if name == "banks.gr_clip_image_mean":
            report["gr_clip_image_mean"] = deviation
        elif is_bank:
            report["banks"][name[len("banks."):]] = deviation
        else:
            report["arrays"][name] = deviation
    if "gr_clip_image_mean" not in report:
        report["gr_clip_image_mean"] = None
        if any(GR_BANK_ID in ids for ids in (left_ids, right_ids)):
            discrete.append({"path": "arrays.banks.gr_clip_image_mean", "reference": "absent", "other": "absent"})

    # metadata (discrete fields, hashes, numeric operator/normalizer statistics)
    metadata_report = compare_metadata(reference.metadata, other.metadata)
    report["metadata"] = metadata_report
    for entry in metadata_report["discrete_mismatches"]:
        discrete.append(entry)
    sidecar_report = compare_metadata(reference.sidecars, other.sidecars)
    report["sidecars"] = sidecar_report

    # implied deployability
    left_deploy = implied_deployability(reference)
    right_deploy = implied_deployability(other)
    report["deployability"] = {"reference": left_deploy, "other": right_deploy}
    for name in ("clean_direction", "noisy_direction"):
        if left_deploy[name]["implied_deployable"] != right_deploy[name]["implied_deployable"] \
                or not left_deploy[name]["implied_deployable"]:
            discrete.append({"path": f"deployability.{name}", "reference": left_deploy[name]["implied_deployable"],
                             "other": right_deploy[name]["implied_deployable"]})

    # tie flags: critical-competitor sets of the development items under the registered tolerance
    if all(f"development.{name}" in prepared.arrays for prepared in (reference, other)
           for name in ("prototypes", "clean_features")) \
            and reference.arrays["development.clean_features"].shape == other.arrays["development.clean_features"].shape \
            and reference.arrays["development.prototypes"].shape == other.arrays["development.prototypes"].shape:
        left_protected, left_critical, left_tie = critical_competitor_sets(
            reference.arrays["development.prototypes"], reference.arrays["development.clean_features"])
        right_protected, right_critical, right_tie = critical_competitor_sets(
            other.arrays["development.prototypes"], other.arrays["development.clean_features"])
        ties = {
            "items": int(len(left_protected)),
            "protected_class_differences": int(np.sum(left_protected != right_protected)),
            "critical_set_differences": int(np.sum(np.any(left_critical != right_critical, axis=1))),
            "tie_flag_differences": int(np.sum(left_tie != right_tie)),
            "tied_items_reference": int(left_tie.sum()),
            "tied_items_other": int(right_tie.sum()),
            "tie_tolerance": TIE_TOLERANCE,
        }
        report["tie_flags"] = ties
        if ties["protected_class_differences"] or ties["critical_set_differences"] or ties["tie_flag_differences"]:
            discrete.append({"path": "tie_flags", "reference": "see tie_flags", "other": ties})
    else:
        report["tie_flags"] = None
        discrete.append({"path": "tie_flags", "reference": "not computable", "other": "shape or array mismatch"})

    # informational: development-item argmax under every bank (the worker never scores development items)
    argmax_report = {}
    for candidate_id in left_ids:
        left = bank_argmax(reference, candidate_id)
        right = bank_argmax(other, candidate_id) if candidate_id in right_ids else None
        if left is None or right is None or left.shape != right.shape:
            argmax_report[candidate_id] = None
        else:
            argmax_report[candidate_id] = int(np.sum(left != right))
    report["informational"]["development_argmax_differences_by_bank"] = argmax_report
    left_dev = reference.sidecars.get("development") or {}
    right_dev = other.sidecars.get("development") or {}
    report["informational"]["clean_zero_shot_accuracy"] = {
        "reference": left_dev.get("clean_zero_shot_accuracy"), "other": right_dev.get("clean_zero_shot_accuracy"),
    }
    left_ctl = reference.sidecars.get("controls") or {}
    right_ctl = other.sidecars.get("controls") or {}
    report["informational"]["control_training_accuracy"] = {
        "reference": {key: left_ctl.get(key) for key in ("shared_train_accuracy", "lowrank_train_accuracy")},
        "other": {key: right_ctl.get(key) for key in ("shared_train_accuracy", "lowrank_train_accuracy")},
    }

    tolerance_entries = dict(report["banks"])
    if report["gr_clip_image_mean"] is not None:
        tolerance_entries["gr_clip_image_mean"] = report["gr_clip_image_mean"]
    bank_deviations = [entry["max_abs"] for entry in tolerance_entries.values() if entry.get("max_abs") is not None]
    report["summary"] = {
        "bank_count": len(report["banks"]),
        "banks_within_tolerance": sum(1 for entry in report["banks"].values() if entry.get("within_tolerance")),
        "banks_beyond_tolerance": sorted(name for name, entry in tolerance_entries.items()
                                         if not entry.get("within_tolerance")),
        "gr_clip_image_mean_within_tolerance": (
            None if report["gr_clip_image_mean"] is None else bool(report["gr_clip_image_mean"].get("within_tolerance"))
        ),
        "max_bank_deviation": max(bank_deviations) if bank_deviations else None,
        "banks_exact_hash_equal": sum(1 for entry in report["banks"].values() if entry.get("exact_hash_equal")),
        "discrete_difference_count": len(discrete),
    }
    return report


# --------------------------------------------------------------------------- manifest


def parse_sha256sum_manifest(path: Path) -> list[tuple[str, str]]:
    entries = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        digest, rest = line[:64], line[64:]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
            raise ValueError(f"malformed manifest line: {line[:80]}")
        rest = rest.lstrip(" ")
        if rest.startswith("*"):
            rest = rest[1:]
        entries.append((digest.lower(), rest.strip()))
    return entries


def verify_manifest(root: Path, manifest_path: Path, *, prefix: str = "results/satml2027/") -> dict[str, Any]:
    entries = parse_sha256sum_manifest(manifest_path)
    matched, mismatched, missing, skipped = [], [], [], []
    for digest, relative in entries:
        if prefix and not relative.startswith(prefix):
            skipped.append(relative)
            continue
        local = root / relative[len(prefix):]
        if not local.is_file():
            missing.append(relative)
            continue
        if sha256_file(local) == digest:
            matched.append(relative)
        else:
            mismatched.append(relative)
    listed = {relative[len(prefix):] for _, relative in entries if relative.startswith(prefix)}
    unlisted = []
    for directory in ("development", "controls", "banks"):
        base = root / directory
        if base.is_dir():
            for item in sorted(base.iterdir()):
                if item.is_file() and f"{directory}/{item.name}" not in listed:
                    unlisted.append(f"{directory}/{item.name}")
    return {
        "manifest_path": str(manifest_path.as_posix()),
        "manifest_sha256": sha256_file(manifest_path),
        "prefix": prefix,
        "entries": len(entries),
        "matched": len(matched),
        "mismatched": mismatched,
        "missing": missing,
        "skipped_outside_prefix": len(skipped),
        "unlisted_files": unlisted,
        "all_listed_files_match": not mismatched and not missing,
    }


# --------------------------------------------------------------------------- adjudication


def adjudicate(cell_reports: dict[str, dict[str, Any]], *, missing_cells: list[str], atol: float) -> dict[str, Any]:
    beyond = {cell: report["summary"]["banks_beyond_tolerance"] for cell, report in cell_reports.items()
              if report["summary"]["banks_beyond_tolerance"]}
    discrete = {cell: report["discrete_differences"] for cell, report in cell_reports.items()
                if report["discrete_differences"]}
    incomplete = {cell: report["summary"]["bank_count"] for cell, report in cell_reports.items()
                  if report["summary"]["bank_count"] != 22}
    deviations = [report["summary"]["max_bank_deviation"] for report in cell_reports.values()
                  if report["summary"]["max_bank_deviation"] is not None]
    immaterial = not beyond and not discrete and not missing_cells and not incomplete and bool(cell_reports)
    return {
        "rule": "immaterial if every bank of every cell agrees within the registered portable tolerance and no "
                "discrete quantity (candidate identifiers, implied deployability, tie flags, item identities, "
                "labels, discrete metadata) differs; otherwise EXP-019 is downgraded to a descriptive extension "
                "with the deviation quantified; no rerun before the SaTML deadline; independent review either way",
        "portable_reproduction_atol": atol,
        "decision": DECISION_IMMATERIAL if immaterial else DECISION_DOWNGRADE,
        "cells_compared": len(cell_reports),
        "cells_missing": missing_cells,
        "cells_with_incomplete_bank_sets": incomplete,
        "banks_beyond_tolerance": beyond,
        "discrete_differences": discrete,
        "max_bank_deviation_over_cells": max(deviations) if deviations else None,
        "independent_review_required": True,
        "rerun_before_deadline": False,
    }


def _environment() -> dict[str, Any]:
    torch = _torch()
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": None if torch is None else torch.__version__,
        "platform": platform.platform(),
    }


def compare_trees(server_root: Path, recompute_root: Path, cells: dict[str, dict[str, Any]], *,
                  atol: float) -> tuple[dict[str, dict[str, Any]], list[str]]:
    reports, missing = {}, []
    for cell_id in sorted(cells):
        try:
            reference = load_cell(server_root, cell_id)
            other = load_cell(recompute_root, cell_id)
        except FileNotFoundError as error:
            missing.append(f"{cell_id}: {error}")
            continue
        reports[cell_id] = compare_cell(reference, other, atol=atol,
                                        expected_registration_sha256=cells[cell_id]["registration_sha256"])
    return reports, missing


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_subparsers(dest="mode", required=True)

    compare = modes.add_parser("compare", help="server tree versus TF32-off recompute tree")
    compare.add_argument("--server-root", required=True, type=Path)
    compare.add_argument("--recompute-root", required=True, type=Path)
    compare.add_argument("--attribution-root", type=Path, default=None,
                         help="optional same-host recompute with the server's flag state (cuDNN TF32 on)")
    compare.add_argument("--registrations", nargs="+", required=True, type=Path)
    compare.add_argument("--cells", nargs="*", default=None, help="restrict to these registered cells")
    compare.add_argument("--server-manifest", type=Path, default=None,
                         help="sha256sum manifest of the server preparation (day1_prepare_attempt2_receipt.txt.manifest.sha256)")
    compare.add_argument("--manifest-prefix", default="results/satml2027/")
    compare.add_argument("--expected-manifest-sha256", default=None)
    compare.add_argument("--atol", type=float, default=PORTABLE_REPRODUCTION_ATOL)
    compare.add_argument("--out", required=True, type=Path)

    check = modes.add_parser("manifest-check", help="verify a preparation tree against a sha256sum manifest")
    check.add_argument("--root", required=True, type=Path)
    check.add_argument("--manifest", required=True, type=Path)
    check.add_argument("--manifest-prefix", default="results/satml2027/")
    check.add_argument("--out", required=True, type=Path)

    export = modes.add_parser("export-npz", help="mirror a .pt preparation tree into torch-free .npz files")
    export.add_argument("--root", required=True, type=Path)
    export.add_argument("--registrations", nargs="+", required=True, type=Path)
    export.add_argument("--cells", nargs="*", default=None)
    export.add_argument("--out", required=True, type=Path)

    args = parser.parse_args(argv)
    assert_not_result_path(args.out, "--out")

    if args.mode == "manifest-check":
        assert_not_result_path(args.root, "--root")
        report = {"schema_version": SCHEMA_VERSION, "mode": "manifest-check", "root": str(args.root.as_posix()),
                  "manifest": verify_manifest(args.root, args.manifest, prefix=args.manifest_prefix),
                  "result_files_touched": False, "final_test_access": False, "environment": _environment()}
        write_json_atomic(args.out, report)
        print(f"manifest: {report['manifest']['matched']}/{report['manifest']['entries']} matched; "
              f"mismatched={len(report['manifest']['mismatched'])} missing={len(report['manifest']['missing'])}")
        return 0 if report["manifest"]["all_listed_files_match"] else 2

    cells = registered_cells(list(args.registrations))
    if args.cells:
        unknown = sorted(set(args.cells) - set(cells))
        if unknown:
            raise RuntimeError(f"unregistered cells requested: {unknown}")
        cells = {cell_id: cells[cell_id] for cell_id in args.cells}

    if args.mode == "export-npz":
        assert_not_result_path(args.root, "--root")
        written = {}
        for cell_id in sorted(cells):
            written[cell_id] = export_cell_npz(load_cell(args.root, cell_id), args.out)
        write_json_atomic(args.out / "export_manifest.json", {
            "schema_version": SCHEMA_VERSION, "mode": "export-npz", "source_root": str(args.root.as_posix()),
            "cells": written, "result_files_touched": False, "final_test_access": False,
            "environment": _environment(),
        })
        print(f"exported {len(written)} cells to {args.out}")
        return 0

    assert_not_result_path(args.server_root, "--server-root")
    assert_not_result_path(args.recompute_root, "--recompute-root")
    if args.attribution_root is not None:
        assert_not_result_path(args.attribution_root, "--attribution-root")
    if args.out.exists():
        raise RuntimeError("adjudication output already exists; overwrite is prohibited")
    if args.atol != PORTABLE_REPRODUCTION_ATOL:
        raise RuntimeError(f"the portable reproduction tolerance is frozen at {PORTABLE_REPRODUCTION_ATOL}")

    manifest_report = None
    if args.server_manifest is not None:
        manifest_report = verify_manifest(args.server_root, args.server_manifest, prefix=args.manifest_prefix)
        if args.expected_manifest_sha256 is not None:
            manifest_report["expected_manifest_sha256"] = args.expected_manifest_sha256
            manifest_report["manifest_sha256_matches_expected"] = (
                manifest_report["manifest_sha256"] == args.expected_manifest_sha256
            )
    cell_reports, missing = compare_trees(args.server_root, args.recompute_root, cells, atol=args.atol)
    attribution = None
    if args.attribution_root is not None:
        attribution_reports, attribution_missing = compare_trees(args.attribution_root, args.recompute_root, cells,
                                                                 atol=args.atol)
        attribution = {
            "description": "same-host recompute with the server's flag state versus the TF32-off recompute; "
                           "isolates the cuDNN-TF32 effect from host/kernel differences; no decision is derived",
            "cells": attribution_reports,
            "cells_missing": attribution_missing,
        }
    decision = adjudicate(cell_reports, missing_cells=missing, atol=args.atol)
    report = {
        "schema_version": SCHEMA_VERSION,
        "mode": "compare",
        "addendum_item": "A6",
        "deviation": DEVIATION_STATEMENT,
        "server_root": str(args.server_root.as_posix()),
        "recompute_root": str(args.recompute_root.as_posix()),
        "attribution_root": None if args.attribution_root is None else str(args.attribution_root.as_posix()),
        "server_manifest": manifest_report,
        "registrations": [
            {"path": str(path.as_posix()), "file_sha256": sha256_file(path)} for path in args.registrations
        ],
        "cells_requested": sorted(cells),
        "cells": cell_reports,
        "attribution_same_host": attribution,
        "decision": decision,
        "adjudication_source_sha256": sha256_file(Path(__file__).resolve()),
        "result_files_touched": False,
        "final_test_access": False,
        "environment": _environment(),
    }
    write_json_atomic(args.out, report)
    print(f"cells compared: {decision['cells_compared']}; missing: {len(missing)}; "
          f"max bank deviation: {decision['max_bank_deviation_over_cells']}; "
          f"banks beyond {args.atol:g}: {sum(len(v) for v in decision['banks_beyond_tolerance'].values())}; "
          f"discrete differences: {sum(len(v) for v in decision['discrete_differences'].values())}")
    print(f"decision: {decision['decision']} (independent review required)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
