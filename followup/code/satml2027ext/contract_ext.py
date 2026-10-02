"""EXP-021 count contract (V4; reviewer-bundle V3 audit, finding E3).

One function checks every count array of one shard against the registered
budgets and against the algebra of the worker's per-draw counters.  The run
custody manifest (``custody_ext.verify_cell``) and the analysis merge
(``analyze_ext.merge_cell_from_shards``) both call it, so a shard the custody
stage accepts is exactly a shard the analysis accepts.

Checked, per shard (``N`` = registered confirmation draws, ``K`` = classes):

* every count array has an integer dtype, the registered shape and no negative
  entry; selection votes sum to the registered selection budget and
  confirmation votes to ``N`` for every candidate and item;
* raw predictions and labels are class ids in ``[0, K)``; seeds are integer
  arrays of one value per item; ``done`` is Boolean and all true;
* budget checkpoints (021C) equal the registered list; each prefix sums to its
  checkpoint, prefixes are elementwise nondecreasing and never exceed the full
  counts; a shard without registered checkpoints carries no prefix;
* counters (changed ``F``, useful ``U``, harmful ``H``, containment
  violations ``Vc``, loose violations ``Vl``, loose budget ``E``):
  ``U + H <= F <= N``, ``Vc <= F``, ``Vl <= F``, ``E <= N``;
* applicability flags are Boolean, one per candidate, loose implies
  containment; where containment does not apply ``Vc = 0``, where the loose
  bound does not apply ``Vl = E = 0``; the identity bank has every counter 0;
* registered applicability: every text-translation (flip-theorem) bank has
  both flags true; the identity, image-side and supervised banks without a
  recorded operator have both false; the learned shared tangent control is
  exploratory and only type-checked.

Flags must also agree across the shards of a cell (``check_flags_agree``).

V4.1 (reviewer-bundle V5 audit, two S0 findings):

* **Integer range before conversion or arithmetic.**  Every integer array is
  range-checked in exact (Python) integer arithmetic before it is converted to
  int64: counts and counters lie in ``[0, budget]``, predictions and labels in
  ``[0, K - 1]``, seeds in the 63-bit domain of ``common.seeds``, checkpoints
  in ``[1, N]``.  A row sum is then at most ``K * budget``, which is checked to
  fit in int64, so no modular wrap-around can produce a registered total.  The
  counter algebra is written without an overflowing addition:
  ``U <= F`` and ``H <= F - U``.
* **One exact shard set per cell** (``shard_file_set``).  Custody and analysis
  read a cell directory only through this function: every entry must be a
  regular file with the worker's exact name ``shard_<i>_of_<n>`` plus
  ``.npz``, ``.meta.json`` or ``_margins.npz``, with one shard count and every
  index in ``[0, n)``.  An extra, temporary, linked or non-canonically named
  file (for example ``shard_-001_of_1.meta.json``) is refused, so no unlisted
  file can reach a registered output.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from satml2027ext import candidates_ext

COUNTER_NAMES = ("flip_changed_draws", "flip_containment_violations", "flip_loose_violations",
                 "flip_loose_budget_draws", "flip_useful_draws", "flip_harmful_draws")
FLAG_NAMES = ("flip_containment_applicable", "flip_loose_applicable")
INT64_MAX = int(np.iinfo(np.int64).max)
SEED_DOMAIN = (0, INT64_MAX)  # common.seeds.derive_item_seed returns 63-bit nonnegative seeds
SHARD_SUFFIXES = (".npz", ".meta.json", "_margins.npz")
SHARD_FILE = re.compile(r"^shard_([0-9]{3,})_of_([0-9]{3,})(\.npz|\.meta\.json|_margins\.npz)$")


class CountContractError(RuntimeError):
    """A shard violates the registered count contract."""


def registered_applicability(registration: Mapping[str, Any], cell: Mapping[str, Any],
                             candidate_ids: Sequence[str]) -> dict[str, tuple[bool, bool] | None]:
    """Required (containment, loose) flags per candidate; ``None`` for the exploratory control."""

    specs = {spec["candidate_id"]: spec for spec in candidates_ext.candidate_specs(registration, cell)}
    exploratory = set(candidates_ext.EXPLORATORY_CONTAINMENT)
    required: dict[str, tuple[bool, bool] | None] = {}
    for candidate_id in candidate_ids:
        spec = specs.get(candidate_id)
        if spec is None:
            raise CountContractError(f"{candidate_id} is not a registered candidate of {cell.get('cell_id')}")
        if candidate_id in exploratory:
            required[candidate_id] = None
        elif spec["operator_class"] == candidates_ext.OPERATOR_TEXT_TRANSLATION:
            required[candidate_id] = (True, True)
        else:
            required[candidate_id] = (False, False)
    listed = (registration.get("candidates") or {}).get("flip_theorem_candidates")
    if isinstance(listed, list):
        theorem = {key for key, value in required.items() if value == (True, True)}
        if set(listed) & set(candidate_ids) != theorem:
            raise CountContractError("the registered flip_theorem_candidates differ from the text-translation banks")
    return required


def _integer_array(data: Mapping[str, np.ndarray], name: str, shape: tuple[int, ...], where: str, *,
                   lower: int = 0, upper: int = INT64_MAX) -> np.ndarray:
    """The array as int64 after proving every entry lies in ``[lower, upper]`` (V4.1: before any conversion)."""

    if name not in data:
        raise CountContractError(f"{where}: {name} is missing")
    array = np.asarray(data[name])
    if array.dtype.kind not in "iu":  # V4.1: plain signed or unsigned integers only (no bool, timedelta, object)
        raise CountContractError(f"{where}: {name} must be an integer array, found {array.dtype}")
    if array.shape != shape:
        raise CountContractError(f"{where}: {name} has shape {array.shape}, expected {shape}")
    if not -INT64_MAX - 1 <= lower <= upper <= INT64_MAX:
        raise ValueError(f"invalid range [{lower}, {upper}] for {name}")
    if array.size:
        smallest, largest = int(array.min()), int(array.max())  # exact for every integer dtype, uint64 included
        if smallest < lower or largest > upper:
            raise CountContractError(f"{where}: {name} has an entry outside [{lower}, {upper}] "
                                     f"(found {smallest} to {largest})")
    return array.astype(np.int64)


def _flags(data: Mapping[str, np.ndarray], name: str, count: int, where: str) -> np.ndarray:
    if name not in data:
        raise CountContractError(f"{where}: {name} is missing")
    flags = np.asarray(data[name])
    if flags.dtype != np.bool_ or flags.shape != (count,):
        raise CountContractError(f"{where}: {name} must be a Boolean array of one flag per candidate")
    return flags


def check_shard(data: Mapping[str, np.ndarray], *, registration: Mapping[str, Any], cell: Mapping[str, Any],
                candidate_ids: Sequence[str], certification: Mapping[str, Any], where: str) -> dict[str, np.ndarray]:
    """Refuse a shard that breaks the count contract; return its applicability flags."""

    item_ids = np.asarray(data.get("item_ids", []))
    if item_ids.ndim != 1:
        raise CountContractError(f"{where}: item_ids must be a one-dimensional array")
    items = int(item_ids.shape[0])
    candidates = len(candidate_ids)
    classes = int(cell["class_count"])
    draws = int(certification["confirmation_draws"])
    selection_budget = int(certification["selection_draws"])
    if items < 1:
        raise CountContractError(f"{where}: the shard holds no items")
    if classes < 1 or draws < 1 or selection_budget < 1:
        raise CountContractError(f"{where}: the class count and both budgets must be positive")
    if classes * max(draws, selection_budget) > INT64_MAX:
        raise CountContractError(f"{where}: a vote total could exceed the int64 range")
    # V4.1: each vote lies in [0, budget], so a row sum is at most K * budget and is computed exactly
    selection = _integer_array(data, "selection_counts", (candidates, items, classes), where, upper=selection_budget)
    confirmation = _integer_array(data, "confirmation_counts", (candidates, items, classes), where, upper=draws)
    if not bool(np.all(selection.sum(axis=2) == selection_budget)):
        raise CountContractError(f"{where}: selection votes do not sum to the registered budget")
    if not bool(np.all(confirmation.sum(axis=2) == draws)):
        raise CountContractError(f"{where}: confirmation votes do not sum to the registered budget")
    # V4.1: class ids are bounded on both sides after a range-checked conversion
    _integer_array(data, "raw_predictions", (candidates, items), where, upper=classes - 1)
    _integer_array(data, "ground_truth", (items,), where, upper=classes - 1)
    for name in ("selection_seeds", "confirmation_seeds"):
        _integer_array(data, name, (items,), where, lower=SEED_DOMAIN[0], upper=SEED_DOMAIN[1])
    # V4.1: the position arrays the worker writes are range-checked too when present
    if "item_offset" in data:
        _integer_array(data, "item_offset", (1,), where)
    if "dataset_indices" in data:
        _integer_array(data, "dataset_indices", (items,), where)
    done = np.asarray(data.get("done"))
    if done.dtype != np.bool_ or done.shape != (items,) or not bool(done.all()):
        raise CountContractError(f"{where}: done must be a Boolean array with every item done")

    checkpoints = [int(value) for value in certification.get("budget_checkpoints") or []]
    if checkpoints:
        stored = _integer_array(data, "budget_checkpoints", (len(checkpoints),), where, lower=1, upper=draws)
        if [int(value) for value in stored] != checkpoints:
            raise CountContractError(f"{where}: budget checkpoints {stored.tolist()} are not the registered {checkpoints}")
        prefix = _integer_array(data, "prefix_confirmation_counts", (len(checkpoints), candidates, items, classes), where,
                                upper=draws)
        for index, checkpoint in enumerate(checkpoints):
            if not bool(np.all(prefix[index].sum(axis=2) == checkpoint)):
                raise CountContractError(f"{where}: prefix {index} does not sum to its checkpoint {checkpoint}")
            if index and not bool(np.all(prefix[index - 1] <= prefix[index])):
                raise CountContractError(f"{where}: prefix counts decrease between checkpoints")
        if not bool(np.all(prefix[-1] <= confirmation)):
            raise CountContractError(f"{where}: a prefix count exceeds the full confirmation count")
    elif "prefix_confirmation_counts" in data or "budget_checkpoints" in data:
        raise CountContractError(f"{where}: the shard carries prefix counts that are not registered")

    counter = {name: _integer_array(data, name, (candidates, items), where, upper=draws) for name in COUNTER_NAMES}
    changed = counter["flip_changed_draws"]
    if bool(np.any(changed > draws)):
        raise CountContractError(f"{where}: changed draws exceed the confirmation budget")
    useful, harmful = counter["flip_useful_draws"], counter["flip_harmful_draws"]
    # V4.1: U + H <= F without an addition that could overflow
    if bool(np.any(useful > changed)) or bool(np.any(harmful > changed - useful)):
        raise CountContractError(f"{where}: useful plus harmful draws exceed the changed draws")
    for name in ("flip_containment_violations", "flip_loose_violations"):
        if bool(np.any(counter[name] > changed)):
            raise CountContractError(f"{where}: {name} exceed the changed draws")
    if bool(np.any(counter["flip_loose_budget_draws"] > draws)):
        raise CountContractError(f"{where}: the loose budget exceeds the confirmation budget")

    containment = _flags(data, "flip_containment_applicable", candidates, where)
    loose = _flags(data, "flip_loose_applicable", candidates, where)
    if bool(np.any(loose & ~containment)):
        raise CountContractError(f"{where}: the loose bound is flagged where containment is not")
    if bool(np.any(counter["flip_containment_violations"][~containment] != 0)):
        raise CountContractError(f"{where}: containment violations recorded where containment does not apply")
    for name in ("flip_loose_violations", "flip_loose_budget_draws"):
        if bool(np.any(counter[name][~loose] != 0)):
            raise CountContractError(f"{where}: {name} recorded where the loose bound does not apply")
    identity = list(candidate_ids).index(candidates_ext.IDENTITY_CANDIDATE_ID)
    if any(bool(np.any(counter[name][identity] != 0)) for name in COUNTER_NAMES) or containment[identity] or loose[identity]:
        raise CountContractError(f"{where}: the identity bank must have no counted change and no applicability")
    required = registered_applicability(registration, cell, candidate_ids)
    for position, candidate_id in enumerate(candidate_ids):
        expected = required[candidate_id]
        if expected is not None and (bool(containment[position]), bool(loose[position])) != expected:
            raise CountContractError(f"{where}: applicability of {candidate_id} is "
                                     f"{(bool(containment[position]), bool(loose[position]))}, registered {expected}")
    return {"flip_containment_applicable": containment.copy(), "flip_loose_applicable": loose.copy()}


def check_flags_agree(first: Mapping[str, np.ndarray], other: Mapping[str, np.ndarray], where: str) -> None:
    """Every shard of a cell must carry the same applicability flags."""

    for name in FLAG_NAMES:
        if not np.array_equal(first[name], other[name]):
            raise CountContractError(f"{where}: {name} differs from the cell's first shard")


def shard_stem(index: int, count: int) -> str:
    """The worker's file stem of shard ``index`` of ``count`` (``run_shard_ext.process_shard``)."""

    return f"shard_{index:03d}_of_{count:03d}"


def is_link_like(path: Path) -> bool:
    """A symbolic link or (Windows) a junction: either could be re-targeted after a verification."""

    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def read_verified_bytes(path: str | Path, digests: Mapping[str, str] | None) -> bytes:
    """The file's bytes, read once; with ``digests`` they must hash to the custody value (V4.1: hash on read).

    Parsing these bytes, rather than re-opening the path, closes the gap between a
    custody verification and the analysis read.
    """

    data = Path(path).read_bytes()
    if digests is not None:
        expected = digests.get(Path(path).name)
        if expected is None or hashlib.sha256(data).hexdigest() != expected:
            raise CountContractError(f"{Path(path).name} differs from its custody-verified bytes")
    return data


def shard_file_set(cell_dir: str | Path, where: str, *, sidecars: bool, margins: bool | None) -> dict[str, Any]:
    """The one complete shard set of a cell directory, or a refusal (V4.1, reviewer-bundle V5 audit S0).

    Every entry must be a regular file whose name is exactly ``shard_stem(i, n)``
    plus one of ``SHARD_SUFFIXES``, with a single shard count ``n`` and every
    index ``i`` in ``[0, n)``.  Every shard needs its ``.npz``.  ``sidecars=True``
    requires every ``.meta.json``; ``False`` allows all of them or none.
    ``margins=True`` requires every ``_margins.npz``, ``False`` forbids them and
    ``None`` allows all or none.  Anything else is refused: an extra or temporary
    file, a sub-directory, a link, or a non-canonical name such as
    ``shard_-001_of_1.meta.json`` or ``shard_0000_of_001.npz``.

    Returns the shard count, the ordered paths per kind and the sorted file names.
    """

    base = Path(cell_dir)
    if not base.is_dir() or is_link_like(base):
        raise CountContractError(f"{where}: result directory {base} is absent or a link")
    parsed: dict[str, tuple[int, int, str]] = {}
    for entry in sorted(base.iterdir()):
        if is_link_like(entry) or not entry.is_file():
            raise CountContractError(f"{where}: {entry.name} is not a regular file")
        if entry.lstat().st_nlink != 1:  # a hard link could be changed from outside the result tree
            raise CountContractError(f"{where}: {entry.name} has other hard links")
        match = SHARD_FILE.fullmatch(entry.name)
        if match is None:
            raise CountContractError(f"{where}: unexpected file {entry.name} in the result directory")
        index, count, suffix = int(match.group(1)), int(match.group(2)), match.group(3)
        if entry.name != shard_stem(index, count) + suffix:
            raise CountContractError(f"{where}: non-canonical shard file name {entry.name}")
        parsed[entry.name] = (index, count, suffix)
    if not parsed:
        raise CountContractError(f"{where}: no shard files in {base}")
    counts = {count for _, count, _ in parsed.values()}
    if len(counts) != 1:
        raise CountContractError(f"{where}: shard files disagree on the shard count {sorted(counts)}")
    count = counts.pop()
    full = list(range(count))
    present = {suffix: sorted(index for index, _, kind in parsed.values() if kind == suffix) for suffix in SHARD_SUFFIXES}
    if count < 1 or present[".npz"] != full:
        raise CountContractError(f"{where}: shard archives {present['.npz']} do not cover 0..{count - 1}")
    if present[".meta.json"] != full and (sidecars or present[".meta.json"]):
        raise CountContractError(f"{where}: shard sidecars {present['.meta.json']} do not match the archives 0..{count - 1}")
    if margins is False and present["_margins.npz"]:
        raise CountContractError(f"{where}: margin stores exist although the cell registers none")
    if present["_margins.npz"] != full and (margins is True or present["_margins.npz"]):
        raise CountContractError(f"{where}: margin stores {present['_margins.npz']} do not match the archives 0..{count - 1}")
    return {
        "shard_count": count,
        "shards": [base / f"{shard_stem(index, count)}.npz" for index in full],
        "sidecars": [base / f"{shard_stem(index, count)}.meta.json" for index in present[".meta.json"]],
        "margins": [base / f"{shard_stem(index, count)}_margins.npz" for index in present["_margins.npz"]],
        "names": sorted(parsed),
    }
