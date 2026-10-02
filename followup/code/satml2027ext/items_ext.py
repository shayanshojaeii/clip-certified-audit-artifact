"""Registered item lists for every EXP-021 stage (V2; V1 review finding P0-B).

V1's worker sorted each list by item id and then compared the hash of the
sorted list with the registered hash of the *registered order* (Phase-3 file
order or class round-robin order), which rejected 30 of the 36 cells.  Every
stage now loads a list here, in its registered order, and verifies:

* the ordered item-id hash equals the registration's ``<role>_items_sha256``;
* the count equals the registration's count for the role;
* the CSV file hashes to the value the data preflight recorded (when given);
* preparation roles never read an evaluation, test or sealed source split.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

from satml2027ext._common import canonical_json_hash, sha256_file
from satml2027ext.guard_ext import item_list_stems

PREPARATION_ROLES = ("development", "control_train")
FORBIDDEN_PREPARATION_SOURCES = ("cifar100_official_test", "eurosat_final_test_sealed", "imagenette_val")
FORBIDDEN_PREPARATION_PREFIXES = ("phase3_calibration_validation",)
REQUIRED_COLUMNS = ("item_id", "dataset_index", "label")


def item_list_path(items_dir: Path, registration: Mapping[str, Any], cell: Mapping[str, Any], role: str) -> Path:
    candidates = [Path(items_dir) / f"{stem}.csv" for stem in item_list_stems(registration, cell, role)]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"registered {role} list is absent; tried {[c.name for c in candidates]} in {items_dir}")
    return path


def load_items(items_dir: Path, cell: Mapping[str, Any], role: str,
               registration: Mapping[str, Any] | None = None) -> list[dict[str, str]]:
    """The registered list in its registered order (never re-sorted)."""

    path = item_list_path(Path(items_dir), registration or {}, cell, role)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for column in REQUIRED_COLUMNS:
        if any(column not in row for row in rows):
            raise RuntimeError(f"item list {path.name} lacks the {column} column")
    return rows


def expected_count(cell: Mapping[str, Any], role: str) -> int | None:
    if role == "evaluation":
        return int(cell["item_count"])
    value = cell.get(f"{role}_item_count")
    return None if value is None else int(value)


def verify_item_list(rows: list[dict[str, str]], cell: Mapping[str, Any], role: str) -> str:
    """Ordered hash (and count, when registered) must equal the registration's values."""

    count = expected_count(cell, role)
    if count is not None and len(rows) != count:
        raise RuntimeError(f"{role} item count {len(rows)} differs from the registration's {count} for {cell['cell_id']}")
    expected = cell.get(f"{role}_items_sha256")
    actual = canonical_json_hash([row["item_id"] for row in rows])
    if expected is None or actual != expected:
        raise RuntimeError(f"registered {role} item-list hash mismatch for {cell['cell_id']}: expected {expected}, got {actual}")
    return actual


def verify_preparation_sources(rows: list[dict[str, str]], role: str) -> None:
    """A preparation role must never list an evaluation, official-test or sealed item."""

    if role not in PREPARATION_ROLES:
        raise RuntimeError(f"role {role!r} may not be loaded by the preparation stage")
    for row in rows:
        source = row.get("source_split", "")
        if source in FORBIDDEN_PREPARATION_SOURCES or any(source.startswith(prefix) for prefix in FORBIDDEN_PREPARATION_PREFIXES):
            raise RuntimeError(f"{role} list names {row['item_id']} from the forbidden source {source!r}")


def load_verified(items_dir: Path, registration: Mapping[str, Any], cell: Mapping[str, Any], role: str, *,
                  preflight: Mapping[str, Any] | None = None, preparation: bool = False) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Load, verify and describe one registered list; returns ``(rows, record)``."""

    path = item_list_path(Path(items_dir), registration, cell, role)
    rows = load_items(Path(items_dir), cell, role, registration)
    ordered_hash = verify_item_list(rows, cell, role)
    if preparation:
        verify_preparation_sources(rows, role)
    file_hash = sha256_file(path)
    if preflight is not None:
        expected = preflight.get("item_list_file_sha256", {}).get(path.name)
        if expected != file_hash:
            raise RuntimeError(f"item list {path.name} differs from the data preflight record")
    return rows, {"file": path.name, "csv_sha256": file_hash, "items_sha256": ordered_hash, "count": len(rows)}
