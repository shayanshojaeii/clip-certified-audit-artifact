#!/usr/bin/env python3
"""EXP-021 bank manifest (V2): the one file the sampling approval binds.

After the preparation stage has sealed every 021A and 021B payload, this script
validates each payload against its registration (provenance plus recomputation
of every deterministic operator), recomputes the noisy class-mean control from
the sealed control cache, checks that all payloads share one data preflight and
one approval and that all cells of a backbone share one model state, links
every 021C cell to the payload of its 021B parent, and writes
``EXP021_BANK_MANIFEST.json``.  The reviewer inspects this manifest (and may
rerun the preparation to compare) before signing the sampling stage; the worker
refuses any payload whose file hash is not the manifest's.

V3 (internal re-review of V2, 2026-09-26):

* 021A origin is re-checked here, not taken from the preparation's own report
  (re-review M1b): the stored prototypes, development features, directions,
  control features and labels must equal the tensors re-derived from the
  EXP-016 files bound by the registration (same preparation code path), and
  the 18 legacy banks are compared again with the bound EXP-017 banks;
* the data preflight file is an argument: every payload must name its hash,
  and the manifest records it; the sampling approval binds the same file and
  the worker refuses a manifest built from another preflight (M2c).

V4 (reviews of reviewer bundle V3, 2026-09-26):

* the origin check is chosen by the registered ``bank_construction.source_mode``,
  never by the payload, and is mandatory for every 021A cell (audit E1);
* 021A's 18 legacy banks are the saved EXP-017 tensors: the check requires the
  stored copies to equal the bound EXP-017 file bit for bit, the sampled banks
  to equal them, and the image mean to have EXP-017's recorded calibration hash;
  the EXP-016 formula reconstruction is a conformance check within 2e-6 whose
  exact-match count is a runtime-dependent diagnostic (text review item 5).

Usage:
  python satml2027ext/bank_manifest_ext.py --data-preflight results/satml2027ext/EXP021_DATA_PREFLIGHT.json \
      --out results/satml2027ext/EXP021_BANK_MANIFEST.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from satml2027ext._common import PROJECT_ROOT, canonical_json_hash, read_json, sha256_file, write_json_atomic  # noqa: E402
from satml2027ext import banks_ext, candidates_ext  # noqa: E402
from satml2027ext.guard_ext import EXPERIMENT_IDS, load_registration  # noqa: E402

MANIFEST_SCHEMA = "satml2027ext.bank_manifest.v2"
EXP021A_SOURCE_KEYS = ("prototypes", "development_clean_features", "image_mean", "clean_direction", "noisy_direction")
DEFAULT_REGISTRATIONS = [
    PROJECT_ROOT / "configs/satml2027ext/exp-20260921-021a.json",
    PROJECT_ROOT / "configs/satml2027ext/exp-20260921-021b.json",
    PROJECT_ROOT / "configs/satml2027ext/exp-20260921-021c.json",
]


def payload_path(banks_dir: Path, cell_id: str) -> Path:
    return Path(banks_dir) / f"{cell_id}__banks.pt"


def noisy_class_mean_check(payload: Mapping[str, Any], cache_path: Path) -> float:
    """Recompute the noisy class-mean bank from the sealed control cache; return max |delta|."""

    import torch

    from common.banks import noisy_class_mean_bank
    from common.hashing import tensor_scientific_hash

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    recorded = payload["provenance"]["control_cache_tensor_sha256"]
    for key in ("noisy_features", "labels", "clean_features"):
        if tensor_scientific_hash(cache[key]) != recorded.get(key):
            raise RuntimeError(f"control cache {key} differs from the payload provenance")
    bank = payload["candidates"][f"{candidates_ext.CONTROL_PREFIX}noisy_class_mean"]["prototypes"]
    rebuilt = noisy_class_mean_bank(cache["noisy_features"], cache["labels"].to(torch.long), int(bank.shape[0]))
    return float((rebuilt.double() - bank.double()).abs().max())


def exp021a_origin_check(payload: Mapping[str, Any], registration: Mapping[str, Any], cell: Mapping[str, Any], *,
                         items_dir: Path, exp016_root: Path, exp017_root: Path) -> dict[str, Any]:
    """Re-derive a 021A cell's sources from the bound EXP-016 files and re-check the EXP-017 legacy banks."""

    from satml2027ext import items_ext, prepare_banks_ext

    rows = items_ext.load_items(items_dir, cell, "development", registration)
    items_ext.verify_item_list(rows, cell, "development")
    saved = prepare_banks_ext.load_exp016_sources(registration, cell, rows, exp016_root=exp016_root, exp017_root=exp017_root)
    expected = banks_ext.source_hashes(banks_ext._prepare_sources({
        "prototypes": saved["prototypes"],
        "development_clean_features": saved["development_clean_features"],
        "clean_direction": saved["clean_direction"],
        "noisy_direction": saved["noisy_direction"],
        "controls": {"noisy_features": saved["development_noisy_features"], "labels": saved["labels"]},
    }))
    import torch

    stored = payload["source_hashes"]
    differing = [key for key in EXP021A_SOURCE_KEYS if stored.get(key) != expected.get(key)]
    for key in ("noisy_features", "labels"):
        if (stored.get("controls") or {}).get(key) != expected["controls"].get(key):
            differing.append(f"controls.{key}")
    if differing:
        raise RuntimeError(f"{cell['cell_id']}: stored sources differ from the bound EXP-016 tensors: {differing}")
    stored_legacy = (payload.get("sources") or {}).get("legacy_banks") or {}
    bound = saved["exp017_banks"]
    if sorted(stored_legacy) != sorted(bound):
        raise RuntimeError(f"{cell['cell_id']}: the stored legacy tensors are not the 18 banks of the bound EXP-017 file")
    for key, tensor in bound.items():
        if not torch.equal(stored_legacy[key], tensor.to(torch.float32)):
            raise RuntimeError(f"{cell['cell_id']}: stored legacy tensor {key} differs from the bound EXP-017 file")
    reproduction = prepare_banks_ext.legacy_reproduction(payload, bound, registration, cell,
                                                         exp017_image_mean_sha256=saved["exp017_image_mean_sha256"])
    recorded = (payload["provenance"].get("legacy_reproduction") or {}).get("max_abs_delta")
    return {"sources_equal_exp016": True,
            "legacy_banks_equal_exp017_bitwise": reproduction["sampled_banks_equal_saved_exp017_bitwise"],
            "image_mean_equals_exp017_record": reproduction["image_mean_equals_exp017_record"],
            "legacy_formula_conformance_max_delta_recomputed": reproduction["max_abs_delta"],
            "legacy_formula_conformance_max_delta_recorded": recorded,
            "legacy_formula_exact_matches_diagnostic": reproduction["exact_formula_matches_diagnostic"],
            "exp016_files": saved["files"]}


def build_manifest(registrations: Mapping[str, Mapping[str, Any]], *, banks_dir: Path, preparation_dir: Path,
                   data_preflight: Path, items_dir: Path | None = None, exp016_root: Path | None = None,
                   exp017_root: Path | None = None) -> dict[str, Any]:
    preflight_sha256 = sha256_file(data_preflight)
    items_dir = Path(items_dir) if items_dir is not None else PROJECT_ROOT / "results/satml2027ext/items"
    exp016_root = Path(exp016_root) if exp016_root is not None else PROJECT_ROOT / "results/EXP-20260906-016"
    exp017_root = Path(exp017_root) if exp017_root is not None else PROJECT_ROOT / "results/EXP-20260906-017"
    entries: dict[str, Any] = {}
    preflights: set[str] = set()
    approvals: set[str] = set()
    model_states: dict[str, set[str]] = {}
    for experiment_id in ("EXP-20260921-021A", "EXP-20260921-021B"):
        registration = registrations[experiment_id]
        for cell in registration["cells"]:
            path = payload_path(banks_dir, cell["cell_id"])
            payload = banks_ext.load_bank_payload(path)
            banks_ext.validate_registered_bank_payload_v2(payload, registration, cell)
            recomputation = banks_ext.recomputation_report(payload, registration, cell)
            provenance = payload["provenance"]
            cache_path = Path(preparation_dir) / f"{cell['cell_id']}__control_cache.pt"
            class_mean_delta = noisy_class_mean_check(payload, cache_path)
            if class_mean_delta > banks_ext.PORTABLE_RECOMPUTE_ATOL:
                raise RuntimeError(f"{cell['cell_id']}: noisy class-mean bank does not follow from its control cache ({class_mean_delta:.3e})")
            if provenance["data_preflight_sha256"] != preflight_sha256:
                raise RuntimeError(f"{cell['cell_id']}: payload was prepared with another data preflight than {Path(data_preflight).name}")
            origin = None
            registered_mode = banks_ext.registered_source_mode(registration)
            if provenance.get("source_mode") != registered_mode:
                raise RuntimeError(f"{cell['cell_id']}: payload source_mode differs from the registration")
            if registered_mode == banks_ext.SAVED_LEGACY_MODE:
                # V4 (audit E1): mandatory for every 021A cell, selected by the registration
                origin = exp021a_origin_check(payload, registration, cell, items_dir=items_dir,
                                              exp016_root=exp016_root, exp017_root=exp017_root)
            elif experiment_id == "EXP-20260921-021A":
                raise RuntimeError("the 021A registration must use the saved-tensor source mode")
            preflights.add(provenance["data_preflight_sha256"])
            approvals.add(provenance["approval_sha256"])
            model_states.setdefault(cell["model_id"], set()).add(provenance["checkpoint"]["model_state_sha256"])
            entries[cell["cell_id"]] = {
                "experiment_id": experiment_id,
                "registration_sha256": registration["registration_sha256"],
                "bank_file": path.name,
                "bank_file_sha256": sha256_file(path),
                "bank_sidecar_sha256": sha256_file(path.with_suffix(".json")),
                "control_cache_sha256": sha256_file(cache_path),
                "candidate_order": list(payload["candidate_order"]),
                "prototype_sha256": {key: entry["metadata"]["output_prototype_sha256"] for key, entry in payload["candidates"].items()},
                "image_mean_sha256": payload["source_hashes"]["image_mean"],
                "checkpoint_sha256": provenance["checkpoint"]["sha256"],
                "model_state_sha256": provenance["checkpoint"]["model_state_sha256"],
                "source_mode": provenance["source_mode"],
                "roles": provenance["roles"],
                "max_recomputation_delta": max(recomputation.values()),
                "noisy_class_mean_recomputation_delta": class_mean_delta,
                "legacy_banks_equal_exp017_bitwise": None if origin is None else origin["legacy_banks_equal_exp017_bitwise"],
                "legacy_formula_conformance_max_delta": None if origin is None else origin["legacy_formula_conformance_max_delta_recomputed"],
                "exp021a_origin": origin,
                "control_training": provenance["control_training"],
                "prompt_control": provenance["prompt_control"],
            }
    if len(preflights) != 1 or len(approvals) != 1:
        raise RuntimeError(f"payloads disagree on the data preflight or approval: {preflights}, {approvals}")
    for model_id, states in model_states.items():
        if len(states) != 1:
            raise RuntimeError(f"{model_id} was loaded with {len(states)} different model states")
    registration_c = registrations["EXP-20260921-021C"]
    parent = candidates_ext.subset_parent(registration_c)
    if parent is None or parent["registration_sha256"] != registrations["EXP-20260921-021B"]["registration_sha256"]:
        raise RuntimeError("021C does not name the 021B registration as its parent")
    subset = candidates_ext.registered_candidate_ids(registration_c)
    for cell in registration_c["cells"]:
        parent_entry = entries.get(cell["parent_cell_id"])
        if parent_entry is None:
            raise RuntimeError(f"021C cell {cell['cell_id']} has no prepared parent {cell['parent_cell_id']}")
        entries[cell["cell_id"]] = {
            "experiment_id": "EXP-20260921-021C",
            "registration_sha256": registration_c["registration_sha256"],
            "parent_cell_id": cell["parent_cell_id"],
            "bank_file": parent_entry["bank_file"],
            "bank_file_sha256": parent_entry["bank_file_sha256"],
            "scored_candidate_ids": subset,
            "prototype_sha256": {key: parent_entry["prototype_sha256"][key] for key in subset},
            "model_state_sha256": parent_entry["model_state_sha256"],
        }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "registrations": {key: value["registration_sha256"] for key, value in sorted(registrations.items())},
        "data_preflight_sha256": next(iter(preflights)),
        "data_preflight_file_verified": True,
        "preparation_approval_sha256": next(iter(approvals)),
        "model_state_sha256": {key: next(iter(value)) for key, value in sorted(model_states.items())},
        "cell_count": len(entries),
        "cells": dict(sorted(entries.items())),
        "sealed_evaluation_access": False,
    }
    manifest["content_sha256"] = canonical_json_hash({key: value for key, value in manifest.items() if key != "content_sha256"})
    return manifest


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = read_json(path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise RuntimeError(f"{path} is not an EXP-021 bank manifest")
    body = {key: value for key, value in manifest.items() if key != "content_sha256"}
    if canonical_json_hash(body) != manifest.get("content_sha256"):
        raise RuntimeError("bank manifest content hash mismatch")
    return manifest


def manifest_entry(manifest: Mapping[str, Any], registration: Mapping[str, Any], cell_id: str) -> dict[str, Any]:
    entry = manifest["cells"].get(cell_id)
    if entry is None:
        raise RuntimeError(f"bank manifest has no entry for {cell_id}")
    if entry["registration_sha256"] != registration["registration_sha256"]:
        raise RuntimeError(f"bank manifest entry for {cell_id} names a different registration")
    if manifest["registrations"].get(registration["experiment_id"]) != registration["registration_sha256"]:
        raise RuntimeError("bank manifest was built for different registrations")
    return dict(entry)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--registrations", nargs=3, type=Path, default=DEFAULT_REGISTRATIONS)
    parser.add_argument("--banks", type=Path, default=PROJECT_ROOT / "results/satml2027ext/banks")
    parser.add_argument("--preparation", type=Path, default=PROJECT_ROOT / "results/satml2027ext/preparation")
    parser.add_argument("--data-preflight", required=True, type=Path)
    parser.add_argument("--items", type=Path, default=PROJECT_ROOT / "results/satml2027ext/items")
    parser.add_argument("--exp016", type=Path, default=PROJECT_ROOT / "results/EXP-20260906-016")
    parser.add_argument("--exp017", type=Path, default=PROJECT_ROOT / "results/EXP-20260906-017")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    registrations = {}
    for path in args.registrations:
        registration = load_registration(path)
        registrations[registration["experiment_id"]] = registration
    if set(registrations) != set(EXPERIMENT_IDS):
        raise SystemExit(f"need the three registrations {EXPERIMENT_IDS}")
    manifest = build_manifest(registrations, banks_dir=args.banks, preparation_dir=args.preparation,
                              data_preflight=args.data_preflight, items_dir=args.items, exp016_root=args.exp016,
                              exp017_root=args.exp017)
    write_json_atomic(args.out, manifest)
    print(f"bank manifest: {manifest['cell_count']} cells -> {args.out} sha256={sha256_file(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
