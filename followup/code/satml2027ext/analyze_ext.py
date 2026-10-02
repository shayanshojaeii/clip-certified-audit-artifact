#!/usr/bin/env python3
"""EXP-021 registered analysis (V2). Torch-free.

Consumes the worker's shard archives (after the run custody manifest has been
verified file by file) and writes one JSON report plus CSV tables.  Every
per-item outcome is computed by ``satml2027/common/certify.py::certify_from_counts``
(imported, never copied).  V2 repairs the V1 pre-sampling review findings:

* **Two orthogonal adjudications per primary contrast** (V1 Q6 finding).
  Direction: Holm over the family's two-sided bootstrap p-values against zero
  (``positive`` / ``negative`` / ``unresolved``).  Practical magnitude, each
  with its own Holm procedure over the same family: ``gain_at_least_sesoi``
  (``p = P*(delta* <= +SESOI)``), ``loss_at_least_sesoi``
  (``p = P*(delta* >= -SESOI)``), ``practically_equivalent`` (percentile-bootstrap
  two one-sided tests in interval-inclusion form, ``p = max(P*(delta* <= -SESOI),
  P*(delta* >= +SESOI))``), otherwise ``inconclusive``.  A small but precise
  positive effect is therefore reported as positive in direction *and*
  practically equivalent, never as "superior" alone.
* **Families.**  The primary family holds the five shared operators; the four
  supervised controls form a separate positive-control family (Q5).  Families
  are each dataset x sigma and the pool over datasets at each sigma; Q1 and Q6
  are adjudicated on the pooled families, dataset families are secondary.
* **Intervals.**  Percentile 95% intervals are labelled unadjusted; the
  simultaneous band over all non-identity contrasts is the single-step max-t
  band; Romano-Wolf step-down adjusted p-values use the standard construction
  (the statistic of rank j is compared with the maximum over ranks j..m,
  monotonised), with rejections at alpha.  No "Holm interval" is reported.
* **Per-draw decision-change conformance** (Q3, Q4) from the worker's exact
  counters: zero containment violations for every applicable bank, zero loose-bound
  violations for the step banks, and zero containment violations for the
  projected-gap banks; the aggregate "changed outcomes <= flippable draws"
  comparison of V1 is gone (it compared quantities with different units).
* **Explicit question verdicts** Q1-Q8 here and Q9 in ``analyze_budget`` (021C).
* **Custody before inference.**  ``main`` requires the analysis-stage approval,
  which binds the run custody manifest, and re-hashes every result file.

V3 (internal re-review of V2, 2026-09-26):

* **021A Q1** (re-review B1).  One registered contrast outside the Holm
  families.  Primary estimand: the equal-cell macro over the twelve cells (the
  manuscript's headline); secondary: the item-pooled value.  Two tests with
  disjoint nulls, each at 5%: direction against zero and shortfall against
  the saved-draw EXP-017 value (``q1_classify``).  Verdicts reproduced /
  reproduced_smaller / not_reproduced / inconclusive / reversed, each with a
  descriptive sentence and no causal clause; not evaluable when a cell is not run.
* **No vacuous conformance** (M1d).  Q3/Q4 are ``not_evaluable`` unless every
  registered flip-theorem bank carries containment and loose-bound counters in
  every cell; the learned shared tangent control is reported as exploratory.
* **021C prefix join required and bound** (m2): the parent registration must be
  the one ``subset_of`` names, and ``main`` needs the parent's approved custody.
* **Labels** (m3): shard labels and dataset indices must equal the registered
  lists when ``main`` supplies them.
* **Not-run cells** (m4) come from the approved run custody manifest; families
  are computed from the cells that ran and name the missing ones.
* **Power labels** (m6): each family carries its registered power label
  (for example EuroSAT ``underpowered``).

V4 (reviews of reviewer bundle V3, 2026-09-26):

* **Count contract** (audit E3): every shard passes ``contract_ext.check_shard``,
  the function the run custody stage applies, and the applicability flags must
  agree across the shards of a cell.
* **Budget join** (E4): the 021B join reports confirmation-prefix identity and
  full certificate-input identity (labels, selection seeds, selection counts
  and selected classes, confirmation seeds, raw predictions) separately.
* **Projected-gap counts** (E7): Q4 reports ``changed_projected_gap_draws`` with
  the tie budget each such draw satisfied (identity winner-pair margin at most
  the recorded normalizer difference plus the tolerance), not an explanation.
* **Q1 reporting**: the fresh-minus-saved difference and both p-values are
  reported and in the sentence; each verdict carries its registered meaning.
* **Scope descent**: a family with a not-run cell keeps its registered power
  label only as ``registered``; its ``effective`` label says that the
  registered power statement no longer applies.

V4.1 (reviewer-bundle V5 audit, S0 findings):

* **Read only custody-verified files.**  A cell is read through
  ``cell_shard_set``, never a glob: its directory must hold exactly one complete
  shard set, and in ``main`` exactly the files the custody verification
  returned.  The projected-gap tie budget reads every sidecar of that set and
  requires them to agree.
* **Range-checked integers.**  The contract proves every count, label, seed and
  counter in range before conversion, and the merge carries them as int64.

Usage:
  python satml2027ext/analyze_ext.py --registration configs/satml2027ext/exp-20260921-021b.json \
      --results results/satml2027ext/EXP-20260921-021B --custody results/satml2027ext/EXP-20260921-021B.custody.json \
      --bank-manifest results/satml2027ext/EXP021_BANK_MANIFEST.json \
      --data-preflight results/satml2027ext/EXP021_DATA_PREFLIGHT.json --out analysis/exp021/021b
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from satml2027ext._common import PROJECT_ROOT, canonical_json_hash, loads_json, read_json, sha256_file, write_json_atomic  # noqa: E402
from satml2027ext import candidates_ext, contract_ext  # noqa: E402
from satml2027ext.guard_ext import (  # noqa: E402
    DEFAULT_APPROVAL_PATH,
    certification_parameters,
    identity_candidate_id,
    load_registration,
    registered_candidate_ids,
    verify_approval,
)
from common.certify import certify_from_counts, prediction_concentration  # noqa: E402

SCHEMA_VERSION = "satml2027ext.analysis.v3"
GENERATOR_NAME = "PCG64DXSM"
OUTPUT_FILES = {
    "report": "analysis_ext.json",
    "cells": "cells_ext.csv",
    "contrasts": "contrasts_ext.csv",
    "clean_gate": "clean_gate_ext.csv",
    "flip": "flip_conformance_ext.csv",
}
DEFAULT_STRATA = ("dataset_id", "class")
CLEAN_GATE_MACRO_POINTS = 1
CLEAN_GATE_CELL_POINTS = 2
CLEAN_GATE_CELL_IMAGES = 3
COUNTER_NAMES = ("flip_changed_draws", "flip_containment_violations", "flip_loose_violations",
                 "flip_loose_budget_draws", "flip_useful_draws", "flip_harmful_draws")
DIRECTION = ("positive", "negative", "unresolved")
MAGNITUDE = ("gain_at_least_sesoi", "loss_at_least_sesoi", "practically_equivalent", "inconclusive")


# --------------------------------------------------------------------------- registration


def inference_spec(registration: Mapping[str, Any]) -> dict[str, Any]:
    """Read the registered V2 inference block; every field that decides a verdict is mandatory."""

    block = registration.get("inference")
    if not isinstance(block, Mapping):
        raise RuntimeError("registration lacks an inference block")
    primary = block.get("primary_family") or {}
    controls = block.get("control_family") or {}
    practical = block.get("practical_magnitude") or {}
    uncertainty = block.get("uncertainty") or {}
    for name, value in (("primary_family.contrasts", primary.get("contrasts")),
                        ("control_family.contrasts", controls.get("contrasts"))):
        if not isinstance(value, list) or not value:
            raise RuntimeError(f"inference.{name} is absent")
    seed = uncertainty.get("seed")
    replicates = uncertainty.get("replicates")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RuntimeError("inference.uncertainty.seed is absent or invalid")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise RuntimeError("inference.uncertainty.replicates is absent or invalid")
    if GENERATOR_NAME not in str(uncertainty.get("generator", "")):
        raise RuntimeError(f"registered generator is not {GENERATOR_NAME}")
    sesoi = practical.get("sesoi_points")
    if not isinstance(sesoi, (int, float)) or not sesoi > 0:
        raise RuntimeError("inference.practical_magnitude.sesoi_points is absent")
    return {
        "reference": str(primary.get("reference", identity_candidate_id(registration))),
        "primary_contrasts": [str(value) for value in primary["contrasts"]],
        "control_contrasts": [str(value) for value in controls["contrasts"]],
        "control_comparator": str(controls.get("comparator", "gr_clip_style_two_sided__coefficient_1")),
        "family_alpha": float(primary.get("alpha_two_sided", 0.05)),
        "sesoi_points": float(sesoi),
        "equivalence_level": 0.90,
        "confidence_level": 0.95,
        "replicates": replicates,
        "seed": seed,
        "generator": GENERATOR_NAME,
        "strata": parse_strata(uncertainty.get("strata", {}).get("definition") if isinstance(uncertainty.get("strata"), Mapping) else uncertainty.get("strata")),
        "adjudication_families": str(block.get("adjudication_families", "pooled_over_datasets_per_sigma")),
        "romano_wolf_alpha": float((block.get("secondary_family") or {}).get("alpha", 0.05)),
    }


def parse_strata(value: Any) -> tuple[str, ...]:
    allowed = {"dataset_id", "class", "cell_id", "model_id", "sigma", "fold"}
    aliases = {"dataset": "dataset_id", "label": "class", "ground_truth": "class", "model": "model_id", "cell": "cell_id"}
    if value is None:
        return DEFAULT_STRATA
    if isinstance(value, str):
        tokens = [token for token in re.split(r"[\s,x×+/]+", value.strip()) if token]
    elif isinstance(value, Sequence):
        tokens = [str(token) for token in value]
    else:
        raise RuntimeError(f"unsupported stratum definition {value!r}")
    parsed = tuple(aliases.get(token.lower(), token.lower()) for token in tokens)
    unknown = [token for token in parsed if token not in allowed]
    if unknown or not parsed:
        raise RuntimeError(f"unsupported stratum keys {unknown} in {value!r}")
    return parsed


# --------------------------------------------------------------------------- inputs


def cell_shard_set(root: Path, cell: Mapping[str, Any],
                   verified: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """The cell's one complete shard set (V4.1, reviewer-bundle V5 audit S0: never a glob).

    With ``verified`` (the listings returned by ``custody_ext.verify_files_against_custody``),
    the files on disk must still be exactly the custody-verified files of the cell.
    """

    cell_id = str(cell["cell_id"])
    current = contract_ext.shard_file_set(Path(root) / "cells" / cell_id, cell_id, sidecars=False, margins=None)
    if verified is not None:
        approved = verified.get(cell_id)
        if approved is None:
            raise RuntimeError(f"{cell_id}: absent from the verified run custody manifest")
        if current["names"] != list(approved["names"]):
            raise RuntimeError(f"{cell_id}: the result files changed after the custody verification")
        current["digests"] = dict(approved["digests"])  # every read below re-checks its bytes against these
    return current


def _shard_files(root: Path, cell_id: str) -> list[Path]:
    return cell_shard_set(root, {"cell_id": cell_id})["shards"]


def merge_cell_from_shards(cell: Mapping[str, Any], root: Path, expected_ids: Sequence[str],
                           certification: Mapping[str, Any],
                           registered_rows: Sequence[Mapping[str, Any]] | None = None, *,
                           registration: Mapping[str, Any],
                           shard_set: Mapping[str, Any] | None = None) -> dict[str, np.ndarray]:
    """In-memory merge with tiling checks; carries counts, counters, seeds and budget prefixes.

    With ``registered_rows`` (the registered evaluation list of the cell), the
    merged labels and dataset indices must equal the list's (V3, re-review m3).
    Every shard passes the shared count contract of the custody stage (V4, E3).
    V4.1: only the shard archives of ``shard_set`` (default: the cell's strict
    shard set) are read, in index order, and every integer array is carried as
    int64 once the contract has proven its range.
    """

    listing = shard_set or cell_shard_set(root, cell)
    shards = list(listing["shards"])
    if not shards:
        raise FileNotFoundError(f"no shards for {cell['cell_id']} under {root}")
    loaded = []
    for path in shards:
        blob = contract_ext.read_verified_bytes(path, listing.get("digests"))  # V4.1: parse the checked bytes
        with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
            loaded.append({key: np.array(archive[key]) for key in archive.files})
    counts = {int(path.name.split("_of_")[1].split(".")[0]) for path in shards}
    if len(counts) != 1 or len(shards) != next(iter(counts)):
        raise RuntimeError(f"{cell['cell_id']}: shard files do not form one complete shard set")
    first_flags = None
    for index, data in enumerate(loaded):
        if [str(value) for value in data["candidate_ids"]] != list(expected_ids):
            raise RuntimeError(f"{cell['cell_id']}: shard bank list differs from the registered candidate order")
        if not bool(np.all(data["done"])):
            raise RuntimeError(f"{cell['cell_id']}: a shard is incomplete")
        where = f"{cell['cell_id']} shard {index}"
        flags = contract_ext.check_shard(data, registration=registration, cell=cell, candidate_ids=list(expected_ids),
                                         certification=certification, where=where)
        if first_flags is None:
            first_flags = flags
        else:
            contract_ext.check_flags_agree(first_flags, flags, where)
    loaded.sort(key=lambda data: int(data["item_offset"][0]))
    offset = 0
    for data in loaded:
        if int(data["item_offset"][0]) != offset:
            raise RuntimeError(f"{cell['cell_id']}: shards do not tile the item list")
        offset += len(data["item_ids"])
    def joined(name: str, axis: int = 0) -> np.ndarray:
        # each shard is range-checked by the contract, so its int64 copy is exact; converting before the
        # concatenation avoids numpy's float64 promotion of mixed uint64/int64 shards (V4.1 re-review S1)
        return np.concatenate([np.asarray(data[name]).astype(np.int64) for data in loaded], axis=axis)

    merged = {
        "candidate_ids": np.array(list(expected_ids)),
        "item_ids": np.concatenate([data["item_ids"] for data in loaded]).astype(str),
        "ground_truth": joined("ground_truth"),
        "raw_predictions": joined("raw_predictions", axis=1),
        "selection_counts": joined("selection_counts", axis=1),
        "confirmation_counts": joined("confirmation_counts", axis=1),
        "selection_seeds": joined("selection_seeds"),
        "confirmation_seeds": joined("confirmation_seeds"),
        "flip_containment_applicable": loaded[0]["flip_containment_applicable"],
        "flip_loose_applicable": loaded[0]["flip_loose_applicable"],
    }
    for name in COUNTER_NAMES:
        merged[name] = joined(name, axis=1)
    if certification["budget_checkpoints"]:
        merged["prefix_confirmation_counts"] = joined("prefix_confirmation_counts", axis=2)
        merged["budget_checkpoints"] = loaded[0]["budget_checkpoints"].astype(np.int64)
    if not np.all(merged["selection_counts"].sum(axis=2) == certification["selection_draws"]):
        raise RuntimeError(f"{cell['cell_id']}: selection votes do not sum to the registered budget")
    if not np.all(merged["confirmation_counts"].sum(axis=2) == certification["confirmation_draws"]):
        raise RuntimeError(f"{cell['cell_id']}: confirmation votes do not sum to the registered budget")
    if len(merged["item_ids"]) != int(cell["item_count"]):
        raise RuntimeError(f"{cell['cell_id']}: merged item count differs from the registration")
    if canonical_json_hash(merged["item_ids"].tolist()) != cell.get("evaluation_items_sha256"):
        raise RuntimeError(f"{cell['cell_id']}: merged item ids do not hash to the registered ordered list")
    merged["dataset_indices"] = joined("dataset_indices") if all("dataset_indices" in data for data in loaded) else None
    if registered_rows is not None:
        if [str(row["item_id"]) for row in registered_rows] != merged["item_ids"].tolist():
            raise RuntimeError(f"{cell['cell_id']}: merged items differ from the registered list")
        if [int(row["label"]) for row in registered_rows] != [int(value) for value in merged["ground_truth"]]:
            raise RuntimeError(f"{cell['cell_id']}: shard labels differ from the registered labels")
        if merged["dataset_indices"] is None or [int(row["dataset_index"]) for row in registered_rows] != [
                int(value) for value in merged["dataset_indices"]]:
            raise RuntimeError(f"{cell['cell_id']}: shard dataset indices differ from the registered list")
    return merged


# --------------------------------------------------------------------------- per-cell outcomes


def cell_outcomes(cell: Mapping[str, Any], data: Mapping[str, np.ndarray], *, alpha: float,
                  radii: Sequence[float], confirmation_key: str = "confirmation_counts",
                  prefix_index: int | None = None) -> dict[str, Any]:
    sigma = float(cell["sigma"])
    keys = sorted(set([float(value) for value in radii] + [sigma]))
    candidate_ids = [str(value) for value in data["candidate_ids"]]
    summaries, outcomes = {}, {}
    for position, candidate_id in enumerate(candidate_ids):
        confirmation = data[confirmation_key][position] if prefix_index is None else data[confirmation_key][prefix_index, position]
        summary, outcome = certify_from_counts(
            raw_predictions=data["raw_predictions"][position], ground_truth=data["ground_truth"],
            selection_counts=data["selection_counts"][position], confirmation_counts=confirmation,
            sigma=sigma, alpha=alpha, radii=keys,
        )
        outcome["raw_correct"] = data["raw_predictions"][position] == data["ground_truth"]
        for value in keys:
            key = format(value, ".12g")
            outcome[f"anchored@{key}"] = outcome[f"standard@{key}"] & (outcome["selected_classes"] == data["raw_predictions"][position])
        summaries[candidate_id] = summary
        outcomes[candidate_id] = outcome
    return {"sigma": sigma, "primary_key": format(sigma, ".12g"), "radius_keys": [format(v, ".12g") for v in keys],
            "summaries": summaries, "outcomes": outcomes, "candidate_ids": candidate_ids}


def certified_wrong_share(outcome: Mapping[str, np.ndarray], key: str, class_count: int) -> dict[str, Any]:
    wrong = outcome[f"wrong_stable@{key}"]
    total = int(wrong.sum())
    if total == 0:
        return {"certified_wrong_count": 0, "certified_wrong_top_class": None, "certified_wrong_top_class_share": None}
    counts = np.bincount(outcome["selected_classes"][wrong], minlength=class_count)
    return {"certified_wrong_count": total, "certified_wrong_top_class": int(counts.argmax()),
            "certified_wrong_top_class_share": float(counts.max() / total)}


# --------------------------------------------------------------------------- bootstrap and tests (carried from V1 unchanged)


def rao_wu_factor(size: int) -> float:
    return 1.0 if size <= 1 else math.sqrt(size / (size - 1.0))



def build_images(rows: Sequence[Mapping[str, Any]], strata_keys: Sequence[str]) -> dict[str, Any]:
    """Index the unique evaluation images and their strata over every row (cell x item).

    ``rows`` carry ``dataset_id, item_id, class, cell_id, model_id, sigma, fold``.
    Images are ordered by (dataset_id, item_id); strata by their sorted key tuple.
    """

    image_index: dict[tuple[str, str], int] = {}
    image_class: dict[tuple[str, str], int] = {}
    image_stratum: dict[tuple[str, str], tuple] = {}
    for row in rows:
        identity = (str(row["dataset_id"]), str(row["item_id"]))
        label = int(row["class"])
        if identity in image_class and image_class[identity] != label:
            raise RuntimeError(f"image {identity} carries two labels across cells")
        image_class[identity] = label
        stratum = tuple(row[key] if key != "class" else label for key in strata_keys)
        if identity in image_stratum and image_stratum[identity] != stratum:
            raise RuntimeError(f"image {identity} falls in two strata; stratum keys must be image-level")
        image_stratum[identity] = stratum
    ordered = sorted(image_class)
    for position, identity in enumerate(ordered):
        image_index[identity] = position
    strata: dict[tuple, list[int]] = {}
    for identity in ordered:
        strata.setdefault(image_stratum[identity], []).append(image_index[identity])
    stratum_list = [{"key": [str(v) for v in key], "size": len(members), "members": np.array(members, dtype=np.int64),
                     "factor": rao_wu_factor(len(members))} for key, members in sorted(strata.items(), key=lambda kv: [str(v) for v in kv[0]])]
    row_image = np.array([image_index[(str(row["dataset_id"]), str(row["item_id"]))] for row in rows], dtype=np.int64)
    return {"count": len(ordered), "row_image": row_image, "strata": stratum_list,
            "identities": ordered}



def rao_wu_bootstrap(aggregate: np.ndarray, strata: Sequence[Mapping[str, Any]], *, replicates: int, seed: int,
                     chunk: int = 20000, return_unscaled: bool = False) -> dict[str, np.ndarray]:
    """Rao-Wu rescaled replicate deviations [replicates, width] from one registered draw sequence.

    Stratum ``k`` (in sorted key order) draws its multiplicities from
    ``PCG64DXSM(SeedSequence(seed).spawn(K)[k])`` so that every family reuses
    exactly the same multiplicity of every image regardless of which strata it
    contains.  The result does not depend on ``chunk``.
    """

    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise ValueError("replicates must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    width = int(aggregate.shape[1])
    deviation = np.zeros((replicates, width), dtype=np.float64)
    unscaled = np.zeros((replicates, width), dtype=np.float64) if return_unscaled else None
    children = np.random.SeedSequence(seed).spawn(len(strata))
    for stratum, child in zip(strata, children):
        members = stratum["members"]
        size = int(stratum["size"])
        block = aggregate[members]
        if not np.any(block):
            continue  # this family contains none of the stratum's images
        generator = np.random.Generator(np.random.PCG64DXSM(child))
        draws = generator.integers(0, size, size=(replicates, size), endpoint=False)
        factor = float(stratum["factor"])
        for start in range(0, replicates, chunk):
            rows = draws[start:start + chunk]
            flat = (rows + (np.arange(rows.shape[0], dtype=np.int64) * size)[:, None]).ravel()
            counts = np.bincount(flat, minlength=rows.shape[0] * size).reshape(rows.shape[0], size).astype(np.float64)
            contribution = (counts - 1.0) @ block
            deviation[start:start + rows.shape[0]] += factor * contribution
            if unscaled is not None:
                unscaled[start:start + rows.shape[0]] += contribution
    result = {"deviation": deviation}
    if unscaled is not None:
        result["unscaled"] = unscaled
    return result



def percentile_interval(values: np.ndarray, level: float) -> tuple[float, float]:
    lower = float(np.quantile(values, (1.0 - level) / 2.0, method="lower"))
    upper = float(np.quantile(values, (1.0 + level) / 2.0, method="higher"))
    return lower, upper



def two_sided_p(values: np.ndarray) -> float:
    return float(min(1.0, 2.0 * min(np.mean(values <= 0.0), np.mean(values >= 0.0))))



def tost_p_values(values: np.ndarray, sesoi: float) -> tuple[float, float, float]:
    p_low = float(np.mean(values <= -sesoi))
    p_high = float(np.mean(values >= sesoi))
    return p_low, p_high, max(p_low, p_high)



def holm(p_values: Sequence[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted, running = [0.0] * len(p_values), 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted



def clean_gate(cell_rows: Sequence[Mapping[str, Any]], reference: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Exact integer/rational clean-accuracy gate; flags, never hides."""

    by_cell: dict[str, dict[str, Any]] = {}
    for row in cell_rows:
        by_cell.setdefault(row["cell_id"], {})[row["candidate_id"]] = row
    table: list[dict[str, Any]] = []
    macro_drop: dict[str, Fraction] = {}
    cell_flags: dict[str, list[str]] = {}
    candidates = sorted({row["candidate_id"] for row in cell_rows if row["candidate_id"] != reference})
    for cell_id, entries in sorted(by_cell.items()):
        reference_row = entries[reference]
        n = int(reference_row["example_count"])
        for candidate_id in candidates:
            row = entries[candidate_id]
            drop = int(reference_row["clean_correct_count"]) - int(row["clean_correct_count"])
            exceeds_points = 100 * drop > CLEAN_GATE_CELL_POINTS * n
            exceeds_images = drop > CLEAN_GATE_CELL_IMAGES
            flagged = exceeds_points and exceeds_images
            table.append({"cell_id": cell_id, "candidate_id": candidate_id, "example_count": n,
                          "reference_clean_correct": int(reference_row["clean_correct_count"]),
                          "candidate_clean_correct": int(row["clean_correct_count"]), "drop_images": drop,
                          "drop_points_exact": str(Fraction(100 * drop, n)), "cell_flag": flagged})
            macro_drop[candidate_id] = macro_drop.get(candidate_id, Fraction(0)) + Fraction(drop, n)
            if flagged:
                cell_flags.setdefault(candidate_id, []).append(cell_id)
    cell_count = len(by_cell)
    summary = {}
    for candidate_id in candidates:
        macro = macro_drop[candidate_id] / cell_count
        summary[candidate_id] = {
            "macro_drop_points_exact": str(macro * 100), "macro_drop_points": float(macro * 100),
            "macro_flag": macro * 100 > CLEAN_GATE_MACRO_POINTS,
            "flagged_cells": cell_flags.get(candidate_id, []),
            "flagged": bool(macro * 100 > CLEAN_GATE_MACRO_POINTS or cell_flags.get(candidate_id)),
        }
    return table, {"rule": f"macro drop <= {CLEAN_GATE_MACRO_POINTS} point and per-cell drop <= max({CLEAN_GATE_CELL_POINTS} points, {CLEAN_GATE_CELL_IMAGES} images); exact arithmetic; flags never hide",
                   "candidates": summary}



def gain_p(values: np.ndarray, sesoi: float) -> float:
    """Percentile-bootstrap p-value for H0: delta <= +SESOI (small when the whole distribution exceeds +SESOI)."""

    return float(np.mean(values <= sesoi))


def loss_p(values: np.ndarray, sesoi: float) -> float:
    """Percentile-bootstrap p-value for H0: delta >= -SESOI."""

    return float(np.mean(values >= -sesoi))


def classify_direction(point: float, holm_p: float, alpha: float) -> str:
    if holm_p <= alpha and point > 0:
        return "positive"
    if holm_p <= alpha and point < 0:
        return "negative"
    return "unresolved"


def classify_magnitude(holm_gain: float, holm_loss: float, holm_equivalence: float, alpha: float) -> str:
    if holm_gain <= alpha:
        return "gain_at_least_sesoi"
    if holm_loss <= alpha:
        return "loss_at_least_sesoi"
    if holm_equivalence <= alpha:
        return "practically_equivalent"
    return "inconclusive"


Q1_VERDICTS = ("reproduced", "reproduced_smaller", "not_reproduced", "inconclusive", "reversed")
Q1_VERDICT_MEANING = {
    "reproduced": ("the direction test found a positive gain and the shortfall test did not find the fresh gain below the "
                   "saved value; not finding a shortfall is not evidence that the two values are equal"),
    "reproduced_smaller": "the direction test found a positive gain and the shortfall test found it below the saved value",
    "not_reproduced": "the direction test did not separate the gain from zero and the shortfall test found it below the saved value",
    "inconclusive": "neither test rejected: the data neither separate the gain from zero nor place it below the saved value",
    "reversed": "the direction test found a negative gain",
}


def q1_classify(point: float, values: np.ndarray, reference: float, alpha: float,
                interval: tuple[float, float] | None = None) -> dict[str, Any]:
    """021A Q1 (V3): direction against zero and shortfall against the saved-draw reference.

    Direction: two-sided percentile-bootstrap p = 2 min(P*(d <= 0), P*(d >= 0)).
    Shortfall: one-sided p = P*(d >= reference) for H0: the fresh-draw gain is at
    least the saved-draw value.  The two nulls (gain = 0; gain >= reference > 0)
    are disjoint, so at most one is true and testing each at ``alpha`` keeps the
    family-wise error at ``alpha``.  Verdicts: positive and no shortfall ->
    reproduced; positive and shortfall -> reproduced_smaller; unresolved and
    shortfall -> not_reproduced; unresolved and no shortfall -> inconclusive;
    negative -> reversed.
    """

    if not reference > 0:
        raise ValueError("the Q1 reference must be a positive gain")
    p_direction = two_sided_p(values)
    p_shortfall = float(np.mean(values >= reference))
    direction = classify_direction(point, p_direction, alpha)
    shortfall = p_shortfall <= alpha
    if direction == "negative":
        verdict = "reversed"
    elif direction == "positive":
        verdict = "reproduced_smaller" if shortfall else "reproduced"
    else:
        verdict = "not_reproduced" if shortfall else "inconclusive"
    lower, upper = interval if interval is not None else percentile_interval(values, 0.95)
    return {"point_points": float(point), "reference_points": float(reference),
            "fresh_minus_saved_points": float(point) - float(reference),
            "ci95_unadjusted_lower": float(lower), "ci95_unadjusted_upper": float(upper),
            "one_sided_upper95": float(np.quantile(values, 0.95, method="higher")),
            "p_direction": p_direction, "p_shortfall": p_shortfall,
            "direction": direction, "shortfall": bool(shortfall), "verdict": verdict,
            "verdict_meaning": Q1_VERDICT_MEANING[verdict]}


def romano_wolf_stepdown(points: np.ndarray, deviations: np.ndarray, alpha: float) -> dict[str, Any]:
    """Romano-Wolf step-down max-t with standard adjusted p-values; single-step max-t simultaneous band.

    Hypotheses are ordered by decreasing |t|.  The adjusted p-value of rank j is
    the running maximum over ranks k <= j of ``P*(max_{l >= k} |z_l| >= |t_(k)|)``,
    where ``z`` are the studentized centred bootstrap deviations; a hypothesis
    is rejected when its adjusted p-value is at most ``alpha`` (equivalent to the
    step-down algorithm).  The band uses the single-step critical value (the
    step-down procedure does not define a band).
    """

    sd = deviations.std(axis=0, ddof=1) if deviations.shape[0] > 1 else np.zeros(points.shape[0])
    safe = np.where(sd > 0, sd, 1.0)
    t = np.where(sd > 0, points / safe, np.where(points == 0, 0.0, np.inf))
    z = np.abs(deviations) / safe[None, :]
    z[:, sd <= 0] = 0.0
    order = list(np.argsort(-np.abs(t), kind="stable"))
    adjusted = np.zeros(len(order))
    critical = np.zeros(len(order))
    running = 0.0
    for rank, column in enumerate(order):
        rest = order[rank:]
        max_stat = z[:, rest].max(axis=1)
        critical[column] = float(np.quantile(max_stat, 1.0 - alpha, method="higher"))
        p = float(np.mean(max_stat >= abs(t[column]))) if np.isfinite(t[column]) else 0.0
        running = max(running, p)
        adjusted[column] = running
    single_step_critical = float(np.quantile(z.max(axis=1), 1.0 - alpha, method="higher"))
    return {"t": t, "sd": sd, "critical": critical, "adjusted_p": adjusted, "rejected": adjusted <= alpha,
            "single_step_critical": single_step_critical,
            "band_lower": points - single_step_critical * sd, "band_upper": points + single_step_critical * sd}


# --------------------------------------------------------------------------- per-draw decision-change conformance


def flip_rows(cell: Mapping[str, Any], data: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    """Per candidate: exact per-draw counters summed over the cell's items (from the worker)."""

    candidate_ids = [str(value) for value in data["candidate_ids"]]
    rows = []
    for position, candidate_id in enumerate(candidate_ids):
        row = {
            "cell_id": str(cell["cell_id"]), "candidate_id": candidate_id,
            "step": candidates_ext.step_value(candidate_id),
            "containment_applicable": bool(data["flip_containment_applicable"][position]),
            "loose_applicable": bool(data["flip_loose_applicable"][position]),
            "items": int(data["flip_changed_draws"].shape[1]),
        }
        for name in COUNTER_NAMES:
            row[name] = int(data[name][position].sum())
        row["changed_draw_fraction"] = row["flip_changed_draws"] / (row["items"] * int(data["confirmation_counts"][position].sum(axis=1)[0]))
        row["loose_budget_fraction"] = (row["flip_loose_budget_draws"] / (row["items"] * int(data["confirmation_counts"][position].sum(axis=1)[0]))
                                        if row["loose_applicable"] else None)
        rows.append(row)
    return rows


def projected_gap_tie_budget(results: Path, cell: Mapping[str, Any], candidate_ids: Sequence[str],
                             tolerance: float, shard_set: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Per projected-gap bank: the largest recorded normalizer difference max |n_j - n_k| (V4, audit E7).

    Read from the operator records in the cell's shard sidecars (hashed by the run custody manifest).
    A changed draw without a containment violation had an identity winner-pair margin of at most
    ``|n_j - n_k| + |a_j - a_k| + tolerance`` with every ``a = 1``, so this bound is the tie budget.
    V4.1 (reviewer-bundle V5 audit S0): the sidecars come from ``shard_set`` (default: the cell's
    strict shard set), never from a glob, and every sidecar must carry the same operator record.
    """

    cell_id = str(cell["cell_id"])
    if shard_set is None:
        if not (Path(results) / "cells" / cell_id).is_dir():
            return {}
        shard_set = cell_shard_set(results, cell)
    sidecars = list(shard_set["sidecars"])
    if not sidecars:
        return {}
    records = [loads_json(contract_ext.read_verified_bytes(path, shard_set.get("digests")).decode("utf-8")).get("candidate_metadata")
               or {} for path in sidecars]
    budget = {}
    for candidate_id in candidate_ids:
        if not candidate_id.startswith("chowers_exact_projected_gap__"):
            continue
        operators = [((record.get(candidate_id) or {}).get("metadata") or {}).get("operator") or {} for record in records]
        if any(value != operators[0] for value in operators[1:]):
            raise RuntimeError(f"{cell_id}: the shard sidecars disagree on the operator record of {candidate_id}")
        operator = operators[0]
        norms = [float(value) for value in operator.get("row_norms") or []]
        scales = [float(value) for value in operator.get("row_scales") or []]
        if not norms or not all(math.isfinite(value) for value in norms + scales):
            budget[candidate_id] = None
            continue
        normalizer = max(norms) - min(norms)
        scale = (max(scales) - min(scales)) if scales else 0.0
        budget[candidate_id] = {"max_normalizer_difference": normalizer, "max_scale_difference": scale,
                                "tolerance": float(tolerance), "margin_bound": normalizer + scale + float(tolerance)}
    return budget


def flip_summary(rows: Sequence[Mapping[str, Any]], *, theorem_ids: Sequence[str] | None = None,
                 exploratory_ids: Sequence[str] = (),
                 tie_budgets: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Q3/Q4 conformance over the registered flip-theorem banks (V3: never vacuous).

    Every registered flip-theorem bank (``theorem_ids``; default: every row with a
    step or a projected-gap id) must carry containment and loose-bound counters
    in every cell; otherwise Q3/Q4 are ``not_evaluable`` and the missing
    (cell, bank) pairs are listed.  ``exploratory_ids`` (the learned shared
    tangent control) are summarized separately and never enter a verdict.
    """

    if theorem_ids is None:
        theorem = [row for row in rows if row["step"] is not None or row["candidate_id"].startswith("chowers_exact_projected_gap__")]
    else:
        wanted = set(theorem_ids)
        theorem = [row for row in rows if row["candidate_id"] in wanted]
        present = {row["candidate_id"] for row in theorem}
        if present != wanted:
            missing = sorted(wanted - present)
            theorem_missing = [f"*::{value}" for value in missing]
        else:
            theorem_missing = []
    if theorem_ids is None:
        theorem_missing = []
    not_applicable = theorem_missing + [f"{row['cell_id']}::{row['candidate_id']}" for row in theorem
                                        if not (row["containment_applicable"] and row["loose_applicable"])]
    steps = [row for row in theorem if row["step"] is not None]
    projected = [row for row in theorem if row["candidate_id"].startswith("chowers_exact_projected_gap__")]
    evaluable = not not_applicable
    q3_ok = bool(steps) and all(row["flip_containment_violations"] == 0 and row["flip_loose_violations"] == 0 for row in steps)
    q4_ok = bool(projected) and all(row["flip_containment_violations"] == 0 for row in projected)
    exploratory = [row for row in rows if row["candidate_id"] in set(exploratory_ids)]
    return {
        "evaluable": evaluable,
        "not_applicable_pairs": not_applicable,
        "q3_verdict": "not_evaluable" if not (evaluable and steps) else ("conforms" if q3_ok else "violated"),
        "q3_step_banks_conform": evaluable and q3_ok,
        "q3_exceptions": [f"{row['cell_id']}::{row['candidate_id']}" for row in steps
                          if row["flip_containment_violations"] or row["flip_loose_violations"]],
        "q4_verdict": "not_evaluable" if not (evaluable and projected) else ("conforms" if q4_ok else "violated"),
        "q4_projected_gap_conforms": evaluable and q4_ok,
        "q4_changed_projected_gap_draws": int(sum(row["flip_changed_draws"] for row in projected)),
        "q4_tie_budget": {
            "statement": ("with zero containment violations, every changed projected-gap draw had an identity winner-pair "
                          "margin of at most the recorded max |n_j - n_k| plus the tolerance (every a_k = 1); in exact "
                          "arithmetic all n_k are equal, so such draws are within the numerical tie budget. This bounds the "
                          "margin; it does not identify the cause of the change."),
            "per_cell": {key: value for key, value in sorted((tie_budgets or {}).items())},
            "max_margin_bound": max((entry["margin_bound"] for per_cell in (tie_budgets or {}).values()
                                     for entry in per_cell.values() if entry), default=None),
        },
        "containment_conforms_everywhere": evaluable and all(row["flip_containment_violations"] == 0 for row in theorem),
        "containment_exceptions": [f"{row['cell_id']}::{row['candidate_id']}" for row in theorem if row["flip_containment_violations"]],
        "exploratory_containment": [
            {"cell_id": row["cell_id"], "candidate_id": row["candidate_id"], "applicable": bool(row["containment_applicable"]),
             "changed_draws": row["flip_changed_draws"], "containment_violations": row["flip_containment_violations"]}
            for row in exploratory],
        "note": "exact per-draw counts from the worker; the V1 aggregate comparison of changed-outcome and flippable-draw fractions is not a theorem and is not computed; the exploratory rows never enter Q3/Q4",
    }


# --------------------------------------------------------------------------- analysis


def _family_names(cell: Mapping[str, Any]) -> list[str]:
    sigma = format(float(cell["sigma"]), ".12g")
    return [f"{cell['dataset_id']}__sigma{sigma}", f"pooled__sigma{sigma}"]


def family_power_labels(registration: Mapping[str, Any]) -> dict[str, Any]:
    block = registration.get("inference") or {}
    labels = block.get("family_power") if isinstance(block, Mapping) else None
    return dict(labels) if isinstance(labels, Mapping) else {}


def effective_power(registered: Any, family_cells: Sequence[str], not_run: Sequence[str]) -> dict[str, Any]:
    """The registered power label, downgraded when a cell of the family was not run (V4)."""

    if not not_run:
        return {"registered": registered, "effective": registered, "cells_run": len(family_cells), "cells_not_run": 0}
    return {"registered": registered,
            "effective": (f"below the registered statement: {len(not_run)} registered cell(s) of this family were not run "
                          "(scope descent), so the family has fewer images than the registered power statement assumed "
                          "and that statement no longer applies"),
            "cells_run": len(family_cells), "cells_not_run": len(not_run)}


def analyze(registration: Mapping[str, Any], *, results: Path, replicates_override: int | None = None,
            not_run: Sequence[str] = (), registered_rows: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
            verified_files: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Run the registered V3 analysis of 021A or 021B (cells in ``not_run`` are reported, not analysed).

    ``verified_files`` (V4.1): the per-cell listings returned by the custody verification; ``main``
    always passes them, so the analysis reads exactly the custody-verified files.
    """

    if candidates_ext.is_subset_registration(registration):
        raise RuntimeError("021C is analysed by analyze_budget")
    spec = inference_spec(registration)
    certification = certification_parameters(registration)
    alpha = certification["alpha_per_example"]
    radii = certification["reported_radii"]
    reference = spec["reference"]
    replicates = int(replicates_override) if replicates_override is not None else spec["replicates"]
    sesoi = spec["sesoi_points"]
    family_alpha = spec["family_alpha"]

    skipped = sorted(set(str(value) for value in not_run))
    unknown = [value for value in skipped if value not in {cell["cell_id"] for cell in registration["cells"]}]
    if unknown:
        raise RuntimeError(f"not-run cells {unknown} are not registered")
    cells = [cell for cell in registration["cells"] if cell["cell_id"] not in skipped]
    if not cells:
        raise RuntimeError("no registered cell ran")
    expected_ids = registered_candidate_ids(registration, cells[0])
    if reference not in expected_ids:
        raise RuntimeError(f"reference candidate {reference!r} is not registered")
    for name in ("primary_contrasts", "control_contrasts"):
        missing = [c for c in spec[name] if c not in expected_ids]
        if missing:
            raise RuntimeError(f"registered {name} are absent from the candidate family: {missing}")
    contrast_ids = [candidate for candidate in expected_ids if candidate != reference]

    cell_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    flips: list[dict[str, Any]] = []
    per_cell: dict[str, dict[str, Any]] = {}
    tie_budgets: dict[str, Any] = {}
    flip_tolerance = float(((registration.get("certification") or {}).get("flip_counters") or {}).get("tolerance", 1e-5))
    for cell in cells:
        rows_for_cell = None if registered_rows is None else registered_rows[cell["cell_id"]]
        shard_set = cell_shard_set(results, cell, verified_files)
        data = merge_cell_from_shards(cell, results, expected_ids, certification, registered_rows=rows_for_cell,
                                      registration=registration, shard_set=shard_set)
        tie_budgets[cell["cell_id"]] = projected_gap_tie_budget(results, cell, expected_ids, flip_tolerance, shard_set=shard_set)
        computed = cell_outcomes(cell, data, alpha=alpha, radii=radii)
        per_cell[cell["cell_id"]] = {"data": data, "computed": computed}
        primary_key = computed["primary_key"]
        class_count = int(cell["class_count"])
        for candidate_id in expected_ids:
            summary = dict(computed["summaries"][candidate_id])
            outcome = computed["outcomes"][candidate_id]
            row = {"cell_id": cell["cell_id"], "model_id": cell["model_id"], "dataset_id": cell["dataset_id"],
                   "sigma": float(cell["sigma"]), "fold": cell.get("fold"), "candidate_id": candidate_id,
                   "clean_correct_count": int(outcome["raw_correct"].sum()),
                   **summary, **certified_wrong_share(outcome, primary_key, class_count)}
            concentration = prediction_concentration(outcome["selected_classes"], class_count)
            row["smoothed_top_class"] = concentration["top_class"]
            row["smoothed_top_class_share"] = concentration["top_class_share"]
            row["primary_standard_ca"] = summary[f"standard_certified_accuracy@{primary_key}"]
            row["primary_anchored_ca"] = summary[f"anchored_certified_accuracy@{primary_key}"]
            row["primary_certified_wrong_rate"] = summary[f"certified_wrong_rate@{primary_key}"]
            cell_rows.append(row)
        for index, item_id in enumerate(data["item_ids"]):
            position_rows.append({"cell_id": cell["cell_id"], "model_id": cell["model_id"], "dataset_id": cell["dataset_id"],
                                  "sigma": float(cell["sigma"]), "fold": cell.get("fold"), "item_id": str(item_id),
                                  "class": int(data["ground_truth"][index]), "index": index})
        flips.extend(flip_rows(cell, data))

    radius_keys = [format(float(value), ".12g") for value in radii]
    outcome_keys = list(dict.fromkeys(["standard@sigma", "anchored@sigma"] + [f"standard@{key}" for key in radius_keys]))

    def difference_matrix(cell_id: str, key: str) -> np.ndarray:
        computed = per_cell[cell_id]["computed"]
        actual = key.replace("@sigma", f"@{computed['primary_key']}")
        reference_values = computed["outcomes"][reference][actual].astype(np.float64)
        return np.stack([computed["outcomes"][c][actual].astype(np.float64) - reference_values for c in contrast_ids], axis=1)

    families: dict[str, list[str]] = {}
    for cell in cells:
        for name in _family_names(cell):
            families.setdefault(name, []).append(cell["cell_id"])
    images = build_images(position_rows, spec["strata"])
    rows_of_cell: dict[str, list[int]] = {}
    for global_index, row in enumerate(position_rows):
        rows_of_cell.setdefault(row["cell_id"], []).append(global_index)
    width = len(contrast_ids)
    primary_index = [contrast_ids.index(c) for c in spec["primary_contrasts"]]
    control_index = [contrast_ids.index(c) for c in spec["control_contrasts"]]
    comparator_index = contrast_ids.index(spec["control_comparator"])
    family_reports: dict[str, Any] = {}
    contrast_rows: list[dict[str, Any]] = []
    power_labels = family_power_labels(registration)
    q1_entry = _prediction(registration, "Q1")
    for family_name, family_cells in sorted(families.items()):
        ordered = [index for cell_id in family_cells for index in rows_of_cell[cell_id]]
        n_positions = len(ordered)
        cell_sizes = {cell_id: len(rows_of_cell[cell_id]) for cell_id in family_cells}
        pooled_weights = np.full(n_positions, 1.0 / n_positions)
        macro_weights = np.array([1.0 / (cell_sizes[position_rows[index]["cell_id"]] * len(family_cells)) for index in ordered])
        row_image = images["row_image"][ordered]
        blocks = [np.concatenate([difference_matrix(cell_id, key) for cell_id in family_cells], axis=0) for key in outcome_keys]
        primary_block = blocks[0]
        matrix = np.concatenate(blocks + [primary_block], axis=1)
        weights_matrix = np.concatenate(
            [np.repeat(pooled_weights[:, None], width * len(outcome_keys), axis=1),
             np.repeat(macro_weights[:, None], width, axis=1)], axis=1)
        aggregate = np.zeros((images["count"], matrix.shape[1]), dtype=np.float64)
        np.add.at(aggregate, row_image, weights_matrix * matrix)
        point = aggregate.sum(axis=0) * 100.0
        bootstrap = rao_wu_bootstrap(aggregate, images["strata"], replicates=replicates, seed=spec["seed"])
        values = point[None, :] + bootstrap["deviation"] * 100.0
        discordance = (primary_block != 0).mean(axis=0)
        n_images = int(np.count_nonzero(np.bincount(row_image, minlength=images["count"])))
        primary_points = point[:width]
        primary_values = values[:, :width]

        p_two = [two_sided_p(primary_values[:, j]) for j in range(width)]
        tost = [tost_p_values(primary_values[:, j], sesoi) for j in range(width)]
        gains = [gain_p(primary_values[:, j], sesoi) for j in range(width)]
        losses = [loss_p(primary_values[:, j], sesoi) for j in range(width)]
        holm_primary = {name: dict(zip(primary_index, holm([source[j] for j in primary_index])))
                        for name, source in (("direction", p_two), ("equivalence", [value[2] for value in tost]),
                                             ("gain", gains), ("loss", losses))}
        holm_control = dict(zip(control_index, holm([p_two[j] for j in control_index])))
        rw = romano_wolf_stepdown(primary_points, bootstrap["deviation"][:, :width] * 100.0, spec["romano_wolf_alpha"])
        macro_start = width * len(outcome_keys)

        contrasts_report: dict[str, Any] = {}
        for j, contrast_id in enumerate(contrast_ids):
            ci95 = percentile_interval(primary_values[:, j], spec["confidence_level"])
            ci90 = percentile_interval(primary_values[:, j], spec["equivalence_level"])
            entry: dict[str, Any] = {
                "contrast_id": contrast_id, "family_role": ("primary_shared" if j in holm_primary["direction"] else
                                                            "positive_control" if j in holm_control else "secondary"),
                "n_positions": n_positions, "n_images": n_images, "discordance": float(discordance[j]),
                "point_points": float(primary_points[j]), "bootstrap_sd_points": float(rw["sd"][j]),
                "ci95_unadjusted_lower": ci95[0], "ci95_unadjusted_upper": ci95[1],
                "ci90_unadjusted_lower": ci90[0], "ci90_unadjusted_upper": ci90[1],
                "p_two_sided": p_two[j], "tost_p_low": tost[j][0], "tost_p_high": tost[j][1],
                "tost_p_equivalence": tost[j][2], "p_gain_at_least_sesoi": gains[j], "p_loss_at_least_sesoi": losses[j],
                "romano_wolf_adjusted_p": float(rw["adjusted_p"][j]), "romano_wolf_rejected": bool(rw["rejected"][j]),
                "simultaneous_band_lower": float(rw["band_lower"][j]), "simultaneous_band_upper": float(rw["band_upper"][j]),
            }
            if j in holm_primary["direction"]:
                entry.update({
                    "holm_p_direction": holm_primary["direction"][j], "holm_p_equivalence": holm_primary["equivalence"][j],
                    "holm_p_gain": holm_primary["gain"][j], "holm_p_loss": holm_primary["loss"][j],
                    "direction": classify_direction(primary_points[j], holm_primary["direction"][j], family_alpha),
                    "magnitude": classify_magnitude(holm_primary["gain"][j], holm_primary["loss"][j],
                                                    holm_primary["equivalence"][j], family_alpha),
                })
            elif j in holm_control:
                comparison = primary_values[:, j] - primary_values[:, comparator_index]
                interval = percentile_interval(comparison, spec["confidence_level"])
                entry.update({
                    "holm_p_direction": holm_control[j],
                    "direction": classify_direction(primary_points[j], holm_control[j], family_alpha),
                    "minus_comparator_point": float(primary_points[j] - primary_points[comparator_index]),
                    "minus_comparator_ci95_unadjusted": list(interval),
                })
            macro_column = macro_start + j
            macro_ci = percentile_interval(values[:, macro_column], spec["confidence_level"])
            entry.update({"macro_point_points": float(point[macro_column]),
                          "macro_ci95_unadjusted_lower": macro_ci[0], "macro_ci95_unadjusted_upper": macro_ci[1]})
            secondary = {}
            for k, key in enumerate(outcome_keys[1:], start=1):
                column = k * width + j
                ci = percentile_interval(values[:, column], spec["confidence_level"])
                secondary[key] = {"point_points": float(point[column]), "ci95_unadjusted_lower": ci[0], "ci95_unadjusted_upper": ci[1]}
            entry["secondary_outcomes"] = secondary
            contrasts_report[contrast_id] = entry
            contrast_rows.append({"family": family_name, **{k: v for k, v in entry.items() if k not in ("secondary_outcomes", "minus_comparator_ci95_unadjusted")}})
        decomposition = {}
        for c in candidates_ext.COEFFICIENTS:
            key = candidates_ext.fmt(c)
            names = {part: pattern.format(c=key) for part, pattern in candidates_ext.DECOMPOSITION.items()}
            if not all(name in contrast_ids for name in names.values()):
                continue
            index = {part: contrast_ids.index(name) for part, name in names.items()}
            interaction = primary_values[:, index["two_sided"]] - primary_values[:, index["text_only"]] - primary_values[:, index["image_only"]]
            decomposition[key] = {
                **{part: float(primary_points[index[part]]) for part in index},
                "interaction_point": float(primary_points[index["two_sided"]] - primary_points[index["text_only"]] - primary_points[index["image_only"]]),
                "interaction_ci95_unadjusted": list(percentile_interval(interaction, spec["confidence_level"])),
            }
        not_run_of_family = sorted(cell["cell_id"] for cell in registration["cells"]
                                   if cell["cell_id"] in skipped and family_name in _family_names(cell))
        family_reports[family_name] = {
            "cells": list(family_cells), "n_positions": n_positions, "n_images": n_images,
            "adjudication": family_name.startswith("pooled__"),
            "not_run_cells_of_family": not_run_of_family,
            "power": effective_power(power_labels.get(family_name), family_cells, not_run_of_family),
            "romano_wolf_single_step_critical": rw["single_step_critical"],
            "decomposition": decomposition,
            "contrasts": contrasts_report,
        }
        if q1_entry and family_name == q1_family(q1_entry):
            j = contrast_ids.index(q1_entry["contrast"])
            references = q1_entry["saved_draw_reference_points"]
            q1_alpha = float(q1_entry["tests"]["alpha"])
            family_reports[family_name]["q1"] = {
                "equal_cell_macro": q1_classify(float(point[macro_start + j]), values[:, macro_start + j],
                                                q1_reference(references["equal_cell_macro"]), q1_alpha),
                "item_pooled": q1_classify(float(primary_points[j]), primary_values[:, j],
                                           q1_reference(references["item_pooled"]), q1_alpha),
            }

    clean_table, clean_summary = clean_gate(cell_rows, reference)
    theorem_ids = [value for value in (registration.get("candidates") or {}).get("flip_theorem_candidates", [])
                   if value in expected_ids] or None
    exploratory_ids = [value for value in (registration.get("candidates") or {}).get("exploratory_containment_candidates", [])
                       if value in expected_ids]
    flip = flip_summary(flips, theorem_ids=theorem_ids, exploratory_ids=exploratory_ids, tie_budgets=tie_budgets)
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": registration["experiment_id"],
        "registration_sha256": registration["registration_sha256"],
        "reference": reference,
        "candidate_ids": expected_ids,
        "primary_contrasts": spec["primary_contrasts"],
        "control_contrasts": spec["control_contrasts"],
        "primary_outcome": "item-pooled paired difference in standard certified accuracy at r = sigma (points)",
        "inference": {**{key: value for key, value in spec.items() if key != "strata"}, "strata": list(spec["strata"]),
                      "replicates_used": replicates, "replicates_overridden": replicates_override is not None,
                      "stratum_count": len(images["strata"]),
                      "resampling_unit": "unique evaluation image (dataset_id, item_id); multiplicity shared across every cell, candidate and radius",
                      "interval_method": "unadjusted percentile of point + Rao-Wu deviation ('lower'/'higher' quantiles)",
                      "p_value_method": "percentile bootstrap: direction 2*min(P*(d<=0),P*(d>=0)); equivalence max(P*(d<=-S),P*(d>=S)) (two one-sided tests, interval-inclusion form); gain P*(d<=S); loss P*(d>=-S)",
                      "multiplicity": "Holm within each family, separately for direction, equivalence, gain and loss; primary shared family and positive-control family separate; Romano-Wolf step-down over all non-identity contrasts; single-step max-t simultaneous band"},
        "outcome_keys": outcome_keys,
        "families": family_reports,
        "clean_gate": clean_summary,
        "flip_condition": flip,
        "sealed_evaluation_access": bool(any(cell.get("consumes_sealed_split") for cell in cells)),
        "not_run_cells": skipped,
        "labels_checked_against_registered_lists": registered_rows is not None,
        "tables": {"cells": cell_rows, "contrasts": contrast_rows, "clean_gate": clean_table, "flip": flips},
    }
    report["questions"] = adjudicate(registration, report, cell_rows)
    return report


# --------------------------------------------------------------------------- question verdicts


def _prediction(registration: Mapping[str, Any], question: str) -> Mapping[str, Any] | None:
    for entry in registration.get("predictions", []):
        if entry.get("id") == question:
            return entry
    return None


def q1_reference(block: Mapping[str, Any]) -> float:
    """The saved-draw reference in points from its exact fraction (V4: the planning file rounds floats)."""

    exact = float(Fraction(str(block["exact"])))
    if abs(exact - float(block["value"])) > 1e-8:
        raise RuntimeError(f"Q1 reference value {block['value']} differs from its exact fraction {block['exact']}")
    return exact


def q1_family(entry: Mapping[str, Any]) -> str:
    return str((entry.get("tests") or {}).get("family", "pooled__sigma0.25"))


def q1_sentence(entry: Mapping[str, Any], result: Mapping[str, Any], estimand: str) -> str:
    template = entry["sentences"][estimand][result["verdict"]]
    body = template.format(point=result["point_points"], lower=result["ci95_unadjusted_lower"],
                           upper=result["ci95_unadjusted_upper"], reference=result["reference_points"],
                           difference=result["fresh_minus_saved_points"], p_direction=result["p_direction"],
                           p_shortfall=result["p_shortfall"])
    scope = entry.get("scope_sentence")
    return body if not scope else f"{body} {scope}"


def adjudicate(registration: Mapping[str, Any], report: Mapping[str, Any], cell_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Machine-checkable verdict per registered question, with the registered sentence."""

    verdicts: dict[str, Any] = {}
    families = report["families"]
    adjudication = {name: family for name, family in families.items() if family["adjudication"]}
    if _prediction(registration, "Q1"):
        entry = _prediction(registration, "Q1")
        family = families.get(q1_family(entry))
        if report.get("not_run_cells") or family is None or "q1" not in family:
            verdicts["Q1"] = {"verdict": "not_evaluable",
                              "reason": "Q1 needs all twelve 021A cells; not run: " + ", ".join(report.get("not_run_cells") or [])}
        else:
            primary = dict(family["q1"]["equal_cell_macro"])
            secondary = dict(family["q1"]["item_pooled"])
            registered_meaning = entry.get("verdict_meaning") or {}
            for result in (primary, secondary):
                result["verdict_meaning"] = registered_meaning.get(result["verdict"], result["verdict_meaning"])
            by_dataset = {name: {"item_pooled_point": other["contrasts"][entry["contrast"]]["point_points"],
                                 "equal_cell_macro_point": other["contrasts"][entry["contrast"]]["macro_point_points"],
                                 "power": other.get("power")}
                          for name, other in families.items() if not other["adjudication"]}
            verdicts["Q1"] = {"verdict": primary["verdict"], "estimand": "equal_cell_macro",
                              "sentence": q1_sentence(entry, primary, "equal_cell_macro"),
                              "primary_equal_cell_macro": primary,
                              "secondary_item_pooled": {**secondary, "sentence": q1_sentence(entry, secondary, "item_pooled")},
                              "estimands_agree": primary["verdict"] == secondary["verdict"],
                              "by_dataset_descriptive": by_dataset,
                              "registered_prediction": entry.get("registered_prediction")}
    if _prediction(registration, "Q2"):
        verdicts["Q2"] = {"verdict": "descriptive", "decomposition": {name: family["decomposition"] for name, family in adjudication.items()}}
    if _prediction(registration, "Q3"):
        flip = report["flip_condition"]
        verdicts["Q3"] = {"verdict": flip["q3_verdict"], "exceptions": flip["q3_exceptions"],
                          "not_applicable_pairs": flip["not_applicable_pairs"],
                          "step_response": [row for row in report["tables"]["flip"] if row["step"] is not None]}
    if _prediction(registration, "Q4"):
        flip = report["flip_condition"]
        verdicts["Q4"] = {"verdict": flip["q4_verdict"], "not_applicable_pairs": flip["not_applicable_pairs"],
                          "changed_projected_gap_draws": flip["q4_changed_projected_gap_draws"],
                          "tie_budget": flip["q4_tie_budget"]}
    if _prediction(registration, "Q5"):
        verdicts["Q5"] = {name: {contrast: {key: family["contrasts"][contrast][key] for key in
                                            ("point_points", "holm_p_direction", "direction", "minus_comparator_point",
                                             "minus_comparator_ci95_unadjusted")}
                                 for contrast in report["control_contrasts"]} for name, family in adjudication.items()}
    if _prediction(registration, "Q6"):
        entry = _prediction(registration, "Q6")
        table = {}
        for name, family in families.items():
            table[name] = {}
            for contrast in report["primary_contrasts"]:
                row = family["contrasts"][contrast]
                table[name][contrast] = {
                    "direction": row["direction"], "magnitude": row["magnitude"], "point": row["point_points"],
                    "ci95_unadjusted": [row["ci95_unadjusted_lower"], row["ci95_unadjusted_upper"]],
                    "sentence_direction": entry["direction_sentences"][row["direction"]],
                    "sentence_magnitude": entry["magnitude_sentences"][row["magnitude"]],
                    "role": "adjudication" if family["adjudication"] else "secondary (dataset-specific)",
                    "power": family.get("power"),
                    "not_run_cells_of_family": family.get("not_run_cells_of_family", []),
                }
        verdicts["Q6"] = table
    if _prediction(registration, "Q7"):
        identity = [row for row in cell_rows if row["candidate_id"] == report["reference"]]
        verdicts["Q7"] = {"verdict": "descriptive", "identity_regime": [
            {key: row[key] for key in ("cell_id", "smoothed_accuracy", "smoothed_top_class_share", "primary_standard_ca", "abstention_rate")}
            for row in identity]}
    if _prediction(registration, "Q8"):
        rows = [row for row in cell_rows if row["candidate_id"] == report["reference"]
                and row["dataset_id"] == "imagenette" and float(row["sigma"]) == 0.25]
        verdicts["Q8"] = {"verdict": "orientation only", "imagenette_identity": [
            {"cell_id": row["cell_id"], **{f"standard_ca@{r}": row.get(f"standard_certified_accuracy@{r}") for r in ("0.25", "0.5")}}
            for row in rows]}
    return verdicts


# --------------------------------------------------------------------------- 021C budget analysis (Q9)


def analyze_budget(registration: Mapping[str, Any], *, results: Path, parent_registration: Mapping[str, Any] | None = None,
                   parent_results: Path | None = None, verified_files: Mapping[str, Mapping[str, Any]] | None = None,
                   parent_verified_files: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Q9: certificates at r = sigma from the 4,096-draw prefix versus all 100,000 draws, same items and draws.

    V3 (re-review m2): the 021B join is required and the parent registration
    must be the one ``subset_of`` names.  V4.1: with the custody-verified
    listings, both merges read exactly the verified files.
    """

    if not candidates_ext.is_subset_registration(registration):
        raise RuntimeError("analyze_budget applies to the 021C subset registration only")
    if parent_registration is None or parent_results is None:
        raise RuntimeError("the registered Q9 prefix join needs the 021B parent registration and results")
    named = candidates_ext.subset_parent(registration) or {}
    if parent_registration.get("registration_sha256") != named.get("registration_sha256"):
        raise RuntimeError("the parent registration is not the one the 021C subset block names")
    certification = certification_parameters(registration)
    checkpoints = certification["budget_checkpoints"]
    if checkpoints != [4096]:
        raise RuntimeError(f"021C registers the 4,096-draw prefix; found {checkpoints}")
    alpha = certification["alpha_per_example"]
    ids = registered_candidate_ids(registration)
    entry = _prediction(registration, "Q9") or {}
    rows: list[dict[str, Any]] = []
    joins: list[dict[str, Any]] = []
    for cell in registration["cells"]:
        data = merge_cell_from_shards(cell, results, ids, certification, registration=registration,
                                      shard_set=cell_shard_set(results, cell, verified_files))
        sigma_key = format(float(cell["sigma"]), ".12g")
        full = cell_outcomes(cell, data, alpha=alpha, radii=[float(cell["sigma"])])
        prefix = cell_outcomes(cell, data, alpha=alpha, radii=[float(cell["sigma"])],
                               confirmation_key="prefix_confirmation_counts", prefix_index=0)
        for candidate_id in ids:
            certified_full = full["outcomes"][candidate_id][f"standard@{sigma_key}"]
            certified_prefix = prefix["outcomes"][candidate_id][f"standard@{sigma_key}"]
            changed = certified_full != certified_prefix
            rows.append({
                "cell_id": cell["cell_id"], "candidate_id": candidate_id, "items": int(len(changed)),
                "certified_correct_4096": int(certified_prefix.sum()), "certified_correct_100000": int(certified_full.sum()),
                "changed": int(changed.sum()), "gained_with_100000": int((certified_full & ~certified_prefix).sum()),
                "lost_with_100000": int((~certified_full & certified_prefix).sum()),
                "abstained_4096": int(prefix["outcomes"][candidate_id]["abstained"].sum()),
                "abstained_100000": int(full["outcomes"][candidate_id]["abstained"].sum()),
                "sentence": str(entry.get("outcome_template", "")).replace("N", str(int(changed.sum()))).replace("B", candidate_id).replace("C", cell["cell_id"]) if entry else None,
            })
        if True:  # the join is registered (V3): always performed
            parent_cell = next(value for value in parent_registration["cells"] if value["cell_id"] == cell["parent_cell_id"])
            parent_ids = registered_candidate_ids(parent_registration, parent_cell)
            parent = merge_cell_from_shards(parent_cell, parent_results, parent_ids, certification_parameters(parent_registration),
                                            registration=parent_registration,
                                            shard_set=cell_shard_set(parent_results, parent_cell, parent_verified_files))
            n = len(data["item_ids"])
            same_items = list(parent["item_ids"][:n]) == list(data["item_ids"])
            same_labels = same_items and bool(np.array_equal(parent["ground_truth"][:n], data["ground_truth"]))
            same_confirmation_seeds = same_items and bool(np.array_equal(parent["confirmation_seeds"][:n], data["confirmation_seeds"]))
            same_selection_seeds = same_items and bool(np.array_equal(parent["selection_seeds"][:n], data["selection_seeds"]))
            prefix_matches, selection_matches, class_matches, raw_matches = {}, {}, {}, {}
            for position, candidate_id in enumerate(ids):
                parent_position = parent_ids.index(candidate_id)
                prefix_equal = np.all(parent["confirmation_counts"][parent_position, :n] == data["prefix_confirmation_counts"][0, position], axis=1)
                selection_equal = np.all(parent["selection_counts"][parent_position, :n] == data["selection_counts"][position], axis=1)
                classes_equal = (np.argmax(parent["selection_counts"][parent_position, :n], axis=1)
                                 == np.argmax(data["selection_counts"][position], axis=1))
                raw_equal = parent["raw_predictions"][parent_position, :n] == data["raw_predictions"][position]
                prefix_matches[candidate_id] = int(prefix_equal.sum())
                selection_matches[candidate_id] = int(selection_equal.sum())
                class_matches[candidate_id] = int(classes_equal.sum())
                raw_matches[candidate_id] = int(raw_equal.sum())
            prefix_identity = (same_items and same_confirmation_seeds
                               and all(value == n for value in prefix_matches.values()))
            input_identity = (same_items and same_labels and same_selection_seeds and same_confirmation_seeds
                              and all(value == n for value in selection_matches.values())
                              and all(value == n for value in class_matches.values())
                              and all(value == n for value in raw_matches.values()))
            joins.append({"cell_id": cell["cell_id"], "parent_cell_id": parent_cell["cell_id"], "items": n,
                          "same_items": same_items, "same_labels": same_labels,
                          "same_selection_seeds": same_selection_seeds, "same_confirmation_seeds": same_confirmation_seeds,
                          "prefix_equal_to_021b_counts": prefix_matches,
                          "selection_counts_equal_to_021b": selection_matches,
                          "selected_classes_equal_to_021b": class_matches,
                          "raw_predictions_equal_to_021b": raw_matches,
                          "confirmation_prefix_identity": prefix_identity,
                          "certificate_input_identity": input_identity,
                          "exact": prefix_identity and input_identity})
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": registration["experiment_id"],
        "registration_sha256": registration["registration_sha256"],
        "questions": {"Q9": {"verdict": "descriptive", "rows": rows,
                             "prefix_identity_with_021b": joins,
                             "confirmation_prefix_identity_everywhere": all(join["confirmation_prefix_identity"] for join in joins),
                             "certificate_input_identity_everywhere": all(join["certificate_input_identity"] for join in joins),
                             "prefix_identity_exact_everywhere": all(join["exact"] for join in joins),
                             "prefix_identity_rule": ("two identities are reported separately (V4, audit E4): the confirmation "
                                                      "prefix (items, confirmation seeds, first 4,096 confirmation counts) and "
                                                      "the full certificate input (labels, selection seeds, selection counts and "
                                                      "selected classes, confirmation seeds, raw predictions); a disagreement is "
                                                      "reported as a determinism finding (registered), never hidden")}},
        "note": "a 100,000-draw Clopper-Pearson bound is still a finite-sample lower confidence bound, not the population probability",
        "tables": {"budget": rows},
    }


# --------------------------------------------------------------------------- output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if value is None else value) for key, value in row.items()})


def write_outputs(out_dir: Path, report: Mapping[str, Any]) -> list[Path]:
    out_dir = Path(out_dir)
    tables = report["tables"]
    targets = {"report": out_dir / OUTPUT_FILES["report"]}
    for name, rows in tables.items():
        targets[name] = out_dir / f"{name}_ext.csv"
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing analysis outputs: {existing}")
    out_dir.mkdir(parents=True, exist_ok=True)
    body = {key: value for key, value in report.items() if key != "tables"}
    body["source_sha256"] = sha256_file(Path(__file__))
    write_json_atomic(targets["report"], body)
    for name, rows in tables.items():
        _write_csv(targets[name], rows)
    return list(targets.values())


def main(argv: list[str] | None = None, *, approval_path: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--custody", required=True, type=Path)
    parser.add_argument("--bank-manifest", required=True, type=Path)
    parser.add_argument("--data-preflight", required=True, type=Path)
    parser.add_argument("--parent-registration", type=Path, default=None, help="021C: the 021B registration (prefix join)")
    parser.add_argument("--parent-results", type=Path, default=None)
    parser.add_argument("--parent-custody", type=Path, default=None)
    parser.add_argument("--items", type=Path, default=PROJECT_ROOT / "results/satml2027ext/items")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    from satml2027ext.custody_ext import verify_files_against_custody

    from satml2027ext import items_ext

    registration = load_registration(args.registration)
    subset = candidates_ext.is_subset_registration(registration)
    if subset and (args.parent_registration is None or args.parent_results is None or args.parent_custody is None):
        raise SystemExit("021C needs --parent-registration, --parent-results and --parent-custody (the registered prefix join)")
    stage_files = {"data_preflight": args.data_preflight, "bank_manifest": args.bank_manifest,
                   "run_custody_manifest": args.custody}
    if subset:
        stage_files["parent_run_custody_manifest"] = args.parent_custody
    approval = verify_approval(registration, stage="analysis", approval_path=approval_path or DEFAULT_APPROVAL_PATH,
                               stage_files=stage_files)
    custody = read_json(args.custody)
    if custody.get("registration_sha256") != registration["registration_sha256"]:
        raise RuntimeError("run custody manifest names another registration")
    verified = verify_files_against_custody(custody, args.results)
    not_run = list(custody.get("not_run_cells") or [])
    if set(custody["cells"]) | set(not_run) != {cell["cell_id"] for cell in registration["cells"]}:
        raise RuntimeError("the run custody manifest does not cover exactly the registered cells")
    if subset:
        parent = load_registration(args.parent_registration)
        parent_custody = read_json(args.parent_custody)
        if parent_custody.get("registration_sha256") != parent["registration_sha256"]:
            raise RuntimeError("the parent run custody manifest names another registration")
        parent_verified = verify_files_against_custody(parent_custody, args.parent_results)
        if set(parent_custody["cells"]) | set(parent_custody.get("not_run_cells") or []) != {
                cell["cell_id"] for cell in parent["cells"]}:
            raise RuntimeError("the parent run custody manifest does not cover exactly the registered parent cells")
        if not_run:
            raise RuntimeError("021C cells cannot be declared not run in an analysed 021C experiment")
        report = analyze_budget(registration, results=args.results, parent_registration=parent, parent_results=args.parent_results,
                                verified_files=verified, parent_verified_files=parent_verified)
    else:
        rows = {}
        for cell in registration["cells"]:
            if cell["cell_id"] in not_run:
                continue
            listed = items_ext.load_items(args.items, cell, "evaluation", registration)
            items_ext.verify_item_list(listed, cell, "evaluation")
            rows[cell["cell_id"]] = listed
        report = analyze(registration, results=args.results, not_run=not_run, registered_rows=rows, verified_files=verified)
    report["approval_sha256"] = approval["_approval_sha256"]
    report["run_custody_sha256"] = sha256_file(args.custody)
    written = write_outputs(args.out, report)
    print(f"wrote {', '.join(path.name for path in written)} to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
