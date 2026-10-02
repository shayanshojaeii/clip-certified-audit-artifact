#!/usr/bin/env python3
"""EXP-021 run custody manifest (V2): verify every result file of one experiment before analysis.

V1's analysis read shards without authenticating their sidecars, the approval,
the banks, or the environment (V1 review finding P1-I, "custody").  This script
checks, per registered cell, that:

* the shards tile the registered evaluation order exactly (contiguous, no gap,
  no overlap, every item done), with one shard count per cell;
* every sidecar names this registration, one sampling approval for the whole
  experiment, the approved bank manifest, the manifest's bank file and model
  state, the approved data preflight, and the registered checkpoint;
* the scored candidate ids equal the registered ids, TF32 was off, one GPU
  class ran the whole cell, and ``sealed_evaluation_access`` equals the cell's
  registered ``consumes_sealed_split``;
* the stored selection and confirmation seeds equal the seeds recomputed from
  the item ids, and every count and counter array has the registered shape;

and writes a manifest with the SHA-256 of every result file.  The analysis
approval binds this manifest; the analysis re-verifies every file hash before
reading a single count.

V3 (internal re-review of V2, 2026-09-26):

* every shard's ``ground_truth`` and ``dataset_indices`` must equal the labels
  and dataset indices of the registered list rows it covers (re-review m3);
* the bank manifest must name the same data preflight as the file given here
  (M2c), and every sidecar's forward batch must be the registered
  ``certification.blocks_per_forward`` (n1);
* cells that were not run are listed explicitly (``--not-run``, re-review m4):
  a listed cell must be registered and must have no result file at all; the
  manifest records the list, so the analysis approval binds it and the analysis
  reports those cells as not run instead of refusing the whole experiment.

V4 (reviewer-bundle V3 audit E3): every shard passes the shared count contract
(``satml2027ext/contract_ext.py``: integer nonnegative counts with exact totals,
prefix sums and monotonicity, counter algebra, Boolean applicability flags that
agree across shards and match the registered applicability); the analysis merge
applies the same function.

V4.1 (reviewer-bundle V5 audit, S0 "custody closure"): a cell directory is read
only through ``contract_ext.shard_file_set``, so it must hold exactly one
complete shard set (archives and sidecars for every index, margin stores exactly
when the cell registers them) and nothing else.  ``verify_files_against_custody``
requires the files on disk to equal the listed files of every cell, allows no
unlisted cell directory (a declared not-run cell may only be empty), and returns
the verified ordered file lists that the analysis consumes.

Usage:
  python satml2027ext/custody_ext.py --registration configs/satml2027ext/exp-20260921-021b.json \
      --results results/satml2027ext/EXP-20260921-021B --bank-manifest results/satml2027ext/EXP021_BANK_MANIFEST.json \
      --data-preflight results/satml2027ext/EXP021_DATA_PREFLIGHT.json --out results/satml2027ext/EXP-20260921-021B.custody.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from satml2027ext._common import PROJECT_ROOT, canonical_json_hash, loads_json, read_json, sha256_file, write_json_atomic  # noqa: E402
from satml2027ext import contract_ext, items_ext  # noqa: E402
from satml2027ext.guard_ext import certification_parameters, load_registration, registered_candidate_ids  # noqa: E402
from common.seeds import derive_item_seed  # noqa: E402

CUSTODY_SCHEMA = "satml2027ext.run_custody.v2"
SHARD_META_SCHEMA = "satml2027ext.shard_meta.v3"
COUNTER_NAMES = ("flip_changed_draws", "flip_containment_violations", "flip_loose_violations",
                 "flip_loose_budget_draws", "flip_useful_draws", "flip_harmful_draws")


def _load_npz_bytes(blob: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        return {key: np.array(archive[key]) for key in archive.files}


def verify_cell(registration: Mapping[str, Any], cell: Mapping[str, Any], results: Path, items_dir: Path, *,
                expected: Mapping[str, Any]) -> dict[str, Any]:
    certification = certification_parameters(registration)
    cell_dir = Path(results) / "cells" / cell["cell_id"]
    # V4.1: exactly one complete shard set and nothing else (no glob, no unlisted file)
    shard_set = contract_ext.shard_file_set(cell_dir, cell["cell_id"], sidecars=True,
                                            margins=bool(cell.get("store_identity_margins", False)))
    shards = shard_set["shards"]
    shard_count = shard_set["shard_count"]
    rows = items_ext.load_items(items_dir, cell, "evaluation", registration)
    items_ext.verify_item_list(rows, cell, "evaluation")
    registered_ids = [row["item_id"] for row in rows]
    registered_labels = [int(row["label"]) for row in rows]
    registered_indices = [int(row["dataset_index"]) for row in rows]
    candidate_ids = registered_candidate_ids(registration, cell)
    covered: list[str] = []
    devices: set[str] = set()
    files: dict[str, str] = {}
    first_flags = None
    for path in sorted(shards, key=lambda value: int(value.stem.split("_")[1])):
        # V4.1: each file is read once; the manifest records the hash of exactly the bytes that were checked
        blob = path.read_bytes()
        data = _load_npz_bytes(blob)
        meta_path = path.with_suffix(".meta.json")
        meta_blob = meta_path.read_bytes()
        meta = loads_json(meta_blob.decode("utf-8"))
        if meta.get("schema_version") != SHARD_META_SCHEMA:
            raise RuntimeError(f"{path.name}: sidecar schema {meta.get('schema_version')!r}")
        if meta.get("registration_sha256") != registration["registration_sha256"]:
            raise RuntimeError(f"{path.name}: sidecar names another registration")
        if meta.get("approval_sha256") != expected["approval_sha256"]:
            raise RuntimeError(f"{path.name}: sidecar names another approval")
        for key in ("bank_manifest_sha256", "bank_file_sha256", "data_preflight_sha256", "checkpoint_sha256", "model_state_sha256"):
            if (meta.get("custody") or {}).get(key) != expected["custody"][key]:
                raise RuntimeError(f"{path.name}: custody field {key} differs from the approved value")
        if meta.get("scored_candidate_ids") != candidate_ids or list(data["candidate_ids"]) != candidate_ids:
            raise RuntimeError(f"{path.name}: scored candidates differ from the registration")
        environment = meta.get("environment") or {}
        if environment.get("tf32_matmul_allowed") or environment.get("tf32_cudnn_allowed"):
            raise RuntimeError(f"{path.name}: TF32 was enabled")
        devices.add(str(environment.get("device_name")))
        if bool(meta.get("sealed_evaluation_access")) != bool(cell.get("consumes_sealed_split", False)):
            raise RuntimeError(f"{path.name}: sealed_evaluation_access differs from the registration")
        if not bool(np.all(data["done"])):
            raise RuntimeError(f"{path.name}: not every item is done")
        offset = int(data["item_offset"][0])
        ids = [str(value) for value in data["item_ids"]]
        if offset != len(covered) or registered_ids[offset : offset + len(ids)] != ids:
            raise RuntimeError(f"{path.name}: items do not tile the registered order at offset {offset}")
        if [int(value) for value in data["ground_truth"]] != registered_labels[offset : offset + len(ids)]:
            raise RuntimeError(f"{path.name}: ground_truth differs from the registered labels")
        if [int(value) for value in data["dataset_indices"]] != registered_indices[offset : offset + len(ids)]:
            raise RuntimeError(f"{path.name}: dataset_indices differ from the registered list")
        registered_blocks = certification.get("blocks_per_forward")
        if registered_blocks is not None and environment.get("blocks_per_forward") != registered_blocks:
            raise RuntimeError(f"{path.name}: forward batch {environment.get('blocks_per_forward')} is not the registered {registered_blocks}")
        covered.extend(ids)
        flags = contract_ext.check_shard(data, registration=registration, cell=cell, candidate_ids=candidate_ids,
                                         certification=certification, where=path.name)
        if first_flags is None:
            first_flags = flags
        else:
            contract_ext.check_flags_agree(first_flags, flags, path.name)
        for position, item_id in enumerate(ids):
            for key, base, role in (("selection_seeds", certification["selection_base_seed"], certification["selection_stream_role"]),
                                    ("confirmation_seeds", certification["confirmation_base_seed"], certification["confirmation_stream_role"])):
                seed = derive_item_seed(base_seed=base, model_id=str(cell["model_id"]), dataset_id=str(cell["dataset_id"]),
                                        sigma=float(cell["sigma"]), item_id=item_id, stream_role=role)
                if int(data[key][position]) != seed:
                    raise RuntimeError(f"{path.name}: stored {key} differ from the registered derivation at {item_id}")
        files[path.name] = hashlib.sha256(blob).hexdigest()
        files[meta_path.name] = hashlib.sha256(meta_blob).hexdigest()
        margins = path.with_name(path.stem + "_margins.npz")
        if margins.is_file():
            files[margins.name] = sha256_file(margins)
    if covered != registered_ids:
        raise RuntimeError(f"{cell['cell_id']}: shards cover {len(covered)} of {len(registered_ids)} registered items")
    if len(devices) != 1:
        raise RuntimeError(f"{cell['cell_id']}: shards ran on several GPU classes {sorted(devices)}")
    if sorted(files) != shard_set["names"]:
        raise RuntimeError(f"{cell['cell_id']}: the custody file list differs from the files on disk")
    return {"shard_count": shard_count, "items": len(covered), "device_name": devices.pop(), "files": files}


def not_run_check(registration: Mapping[str, Any], results: Path, not_run: Sequence[str]) -> list[str]:
    """Validate the declared not-run cells: registered, and without any result file."""

    registered = {cell["cell_id"] for cell in registration["cells"]}
    listed = sorted(set(str(value) for value in not_run))
    unknown = [value for value in listed if value not in registered]
    if unknown:
        raise RuntimeError(f"--not-run names unregistered cells {unknown}")
    for cell_id in listed:
        directory = Path(results) / "cells" / cell_id
        # V4.1: a declared not-run cell may only be absent or an empty directory, as the verification requires
        if directory.exists() and (not directory.is_dir() or contract_ext.is_link_like(directory) or any(directory.iterdir())):
            raise RuntimeError(f"{cell_id} is declared not run but has result files; a started cell must be completed and reported")
    if len(listed) == len(registered):
        raise RuntimeError("every registered cell is declared not run")
    return listed


def build_custody(registration: Mapping[str, Any], results: Path, items_dir: Path, *, bank_manifest: Path,
                  data_preflight: Path, not_run: Sequence[str] = ()) -> dict[str, Any]:
    from satml2027ext import bank_manifest_ext

    manifest = bank_manifest_ext.load_manifest(bank_manifest)
    if manifest.get("data_preflight_sha256") != sha256_file(data_preflight):
        raise RuntimeError("the bank manifest names another data preflight than the file given")
    skipped = not_run_check(registration, results, not_run)
    run_cells = [cell for cell in registration["cells"] if cell["cell_id"] not in skipped]
    approvals: set[str] = set()
    cells: dict[str, Any] = {}
    first_meta = None
    for cell in run_cells:
        # V4.1: the sidecar of shard 0 of the cell's one complete shard set, never a glob
        shard_set = contract_ext.shard_file_set(Path(results) / "cells" / cell["cell_id"], cell["cell_id"], sidecars=True,
                                                margins=bool(cell.get("store_identity_margins", False)))
        first_meta = read_json(shard_set["sidecars"][0])
        approvals.add(first_meta.get("approval_sha256"))
    if len(approvals) != 1:
        raise RuntimeError(f"the experiment ran under several approvals: {approvals}")
    approval_sha256 = approvals.pop()
    for cell in run_cells:
        entry = bank_manifest_ext.manifest_entry(manifest, registration, cell["cell_id"])
        expected = {
            "approval_sha256": approval_sha256,
            "custody": {
                "bank_manifest_sha256": sha256_file(bank_manifest),
                "bank_file_sha256": entry["bank_file_sha256"],
                "data_preflight_sha256": sha256_file(data_preflight),
                "checkpoint_sha256": registration["models"][str(cell["model_id"])]["sha256"],
                "model_state_sha256": entry["model_state_sha256"],
            },
        }
        cells[cell["cell_id"]] = verify_cell(registration, cell, results, items_dir, expected=expected)
    custody = {
        "schema_version": CUSTODY_SCHEMA,
        "experiment_id": registration["experiment_id"],
        "registration_sha256": registration["registration_sha256"],
        "sampling_approval_sha256": approval_sha256,
        "bank_manifest_sha256": sha256_file(bank_manifest),
        "data_preflight_sha256": sha256_file(data_preflight),
        "not_run_cells": skipped,
        "cells": cells,
        "file_count": sum(len(value["files"]) for value in cells.values()),
    }
    custody["content_sha256"] = canonical_json_hash({key: value for key, value in custody.items() if key != "content_sha256"})
    verify_files_against_custody(custody, results)  # V4.1: never write a manifest that the analysis would refuse
    return custody


def verify_files_against_custody(custody: Mapping[str, Any], results: Path) -> dict[str, dict[str, Any]]:
    """The result tree must equal the custody manifest exactly (called by the analysis before reading).

    V4.1: besides every listed hash, the files on disk must be exactly the listed
    files of each cell (no unlisted sidecar, archive or margin store), and no
    unlisted cell directory may exist (a declared not-run cell may only be an
    empty directory).  Returns, per cell, the verified ``shard_file_set`` whose
    ordered paths are the only files the analysis reads.
    """

    body = {key: value for key, value in custody.items() if key != "content_sha256"}
    if canonical_json_hash(body) != custody.get("content_sha256"):
        raise RuntimeError("run custody manifest content hash mismatch")
    cells_root = Path(results) / "cells"
    listed_cells = set(custody["cells"])
    not_run = set(custody.get("not_run_cells") or [])
    for name in sorted(listed_cells | not_run):
        if not isinstance(name, str) or name in ("", ".", "..") or Path(name).name != name or "\\" in name:
            raise RuntimeError(f"run custody manifest names an invalid cell directory {name!r}")
    if listed_cells & not_run:
        raise RuntimeError(f"cells both listed and declared not run: {sorted(listed_cells & not_run)}")
    if cells_root.exists():
        for entry in sorted(cells_root.iterdir()):
            if contract_ext.is_link_like(entry) or not entry.is_dir():
                raise RuntimeError(f"unexpected entry {entry.name} in {cells_root}")
            if entry.name in listed_cells:
                continue
            if entry.name in not_run and not any(entry.iterdir()):
                continue
            raise RuntimeError(f"{entry.name}: result directory absent from the run custody manifest")
    verified: dict[str, dict[str, Any]] = {}
    for cell_id, record in custody["cells"].items():
        cell_dir = cells_root / cell_id
        listed = sorted(record["files"])
        shard_set = contract_ext.shard_file_set(cell_dir, cell_id, sidecars=True, margins=None)  # refuses any stray entry
        if shard_set["names"] != listed:
            raise RuntimeError(f"{cell_id}: files on disk differ from the run custody manifest "
                               f"(unlisted {sorted(set(shard_set['names']) - set(listed))}, "
                               f"missing {sorted(set(listed) - set(shard_set['names']))})")
        if shard_set["shard_count"] != int(record["shard_count"]):
            raise RuntimeError(f"{cell_id}: the shard count on disk differs from the run custody manifest")
        for name, digest in record["files"].items():
            if sha256_file(cell_dir / name) != digest:
                raise RuntimeError(f"{cell_id}/{name} differs from the run custody manifest")
        shard_set["digests"] = dict(record["files"])  # the analysis re-checks these on the bytes it parses
        verified[cell_id] = shard_set
    return verified


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--items", type=Path, default=PROJECT_ROOT / "results/satml2027ext/items")
    parser.add_argument("--bank-manifest", required=True, type=Path)
    parser.add_argument("--data-preflight", required=True, type=Path)
    parser.add_argument("--not-run", action="append", default=[],
                        help="repeatable: a registered cell that was not run (scope descent recorded in the decision log)")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    registration = load_registration(args.registration)
    custody = build_custody(registration, args.results, args.items, bank_manifest=args.bank_manifest,
                            data_preflight=args.data_preflight, not_run=args.not_run)
    write_json_atomic(args.out, custody)
    print(f"run custody: {len(custody['cells'])} cells, {custody['file_count']} files -> {args.out} sha256={sha256_file(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
