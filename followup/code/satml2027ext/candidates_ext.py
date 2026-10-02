"""Canonical EXP-021 candidate family (V2): the single source of truth. Torch-free.

``make_registrations_ext.py`` writes the registration ``candidates`` block from
this module, ``banks_ext.py`` builds tensors from the specs expanded here,
``guard_ext.py``, ``run_shard_ext.py`` and ``analyze_ext.py`` enumerate
candidate ids from here, and every formula printed in a registration is
generated from here.  The V1 registrations carried a hand-written flat block
that the bank parser could not read and that misstated two operators (V1 review
findings P0-A and item 11); that cannot recur while all consumers share this file.

Notation used in the generated formulas (every row is unit-normalised before use):

* ``T = [t_1 .. t_K]``: the source prototype bank (prompt-ensemble text prototypes);
* ``mean(t)``: ``T.mean(dim=0)``; ``mean_w(t)``: ``(1/K) 1^T T``, the frozen EXP-017
  "weighted" text mean (numerically equal to ``mean(t)`` up to the last float bits);
* ``mu_I``: mean of the unit clean development image features of the cell;
* ``V``: orthonormal basis of the row space of ``T - 1 mean_w(t)^T``;
* ``u_clean``, ``u_noisy``: unit development directions (EXP-016 objectives);
* ``z``: the unit image feature of one noisy draw; ``z'`` its transformed version.

V2 changes relative to V1 (see research/EXP021_V2_AMENDMENT_2026-09-26.md):

* ``text_only_centering`` is new.  V1 used ``global_mean_centering`` as the
  "text-only" half of the two-sided operator, but that bank translates the
  prototypes by ``-c (mean_w(t) - mu_I)``, not by ``-c mean(t)``; the true text
  half of ``gr_clip_style_two_sided`` is ``normalize(t_k - c mean(t))`` with the
  image feature unchanged, and its prototypes are identical to the two-sided
  bank's at the same ``c``.  The decomposition triple is now exact.
* the rank-8 control is ``lowrank_tangent_r8`` trained with the EXP-019 noisy
  plus clean cross-entropy recipe; V1 labelled it "tail objective" although no
  tail objective was implemented.
* 36 candidates: 1 + 3 + 3 + 3 + 3 + 3 + 8 + 8 + 4.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

CANDIDATES_SCHEMA_VERSION = "satml2027ext.candidates.v2"
IDENTITY_CANDIDATE_ID = "no_correction"
CONTROL_PREFIX = "control__"
IMAGE_TRANSFORM_CENTER = "center_and_renormalize"

COEFFICIENTS: tuple[float, ...] = (0.25, 0.5, 1.0)
LEGACY_STEPS: tuple[float, ...] = (0.0025, 0.005, 0.01, 0.02, 0.04)
NEW_STEPS: tuple[float, ...] = (0.08, 0.16, 0.32)

# Operator classes.  The per-draw decision-change theorem (research/DECISION_CHANGE_CONDITION.md)
# applies to TEXT_TRANSLATION banks: t'_k = (t_k + v)/n_k with a shared vector v and an unchanged
# image feature.  IMAGE_SIDE banks transform z and are outside the theorem.
OPERATOR_IDENTITY = "identity"
OPERATOR_TEXT_TRANSLATION = "text_common_translation"
OPERATOR_IMAGE_SIDE = "image_side_centering"
OPERATOR_SUPERVISED = "supervised_control"

STEP_FAMILIES = ("clean_boundary_active", "cohen_aligned_noisy_margin")
COEFFICIENT_FAMILIES = (
    "global_mean_centering",
    "chowers_exact_projected_gap",
    "gr_clip_style_two_sided",
    "text_only_centering",
    "image_only_centering",
)
IMAGE_SIDE_FAMILIES = ("gr_clip_style_two_sided", "image_only_centering")
CONTROL_IDS = (
    "learned_shared_translation_tangent",
    "lowrank_tangent_r8",
    "noisy_class_mean",
    "fewshot_prompt_bank",
)

_FAMILIES: tuple[dict[str, Any], ...] = (
    {
        "family": "no_correction",
        "operator_class": OPERATOR_IDENTITY,
        "form": "t'_k = t_k; z' = z",
        "legacy_exp017": True,
    },
    {
        "family": "global_mean_centering",
        "parameter": "coefficient",
        "values": list(COEFFICIENTS),
        "legacy_exp017_values": list(COEFFICIENTS),
        "operator_class": OPERATOR_TEXT_TRANSLATION,
        "form": "t'_k = normalize(t_k - {c} (mean_w(t) - mu_I)); z' = z",
        "shared_vector": "-{c} (mean_w(t) - mu_I)",
        "note": "historical EXP-017 text mean-centering: moves the prototypes toward the image mean; NOT the text half of the two-sided operator",
    },
    {
        "family": "chowers_exact_projected_gap",
        "parameter": "coefficient",
        "values": list(COEFFICIENTS),
        "legacy_exp017_values": list(COEFFICIENTS),
        "operator_class": OPERATOR_TEXT_TRANSLATION,
        "form": "t'_k = normalize(t_k + {c} (I - V V^T)(mu_I - mean_w(t))); z' = z",
        "shared_vector": "{c} (I - V V^T)(mu_I - mean_w(t))",
        "note": "exact decision-invariance control (Q4): the shared vector is orthogonal to every t_j - t_k, so all normalizers are equal",
    },
    {
        "family": "gr_clip_style_two_sided",
        "parameter": "coefficient",
        "values": list(COEFFICIENTS),
        "legacy_exp017_values": [1.0],
        "operator_class": OPERATOR_IMAGE_SIDE,
        "form": "t'_k = normalize(t_k - {c} mean(t)); z' = normalize(z - {c} mu_I)",
        "shared_vector": "-{c} mean(t)",
        "decomposition_triple": "two_sided",
    },
    {
        "family": "text_only_centering",
        "parameter": "coefficient",
        "values": list(COEFFICIENTS),
        "legacy_exp017_values": [],
        "operator_class": OPERATOR_TEXT_TRANSLATION,
        "form": "t'_k = normalize(t_k - {c} mean(t)); z' = z",
        "shared_vector": "-{c} mean(t)",
        "decomposition_triple": "text_only",
        "note": "the text half of gr_clip_style_two_sided at the same coefficient (identical prototype tensor, identity image transform); new in V2",
    },
    {
        "family": "image_only_centering",
        "parameter": "coefficient",
        "values": list(COEFFICIENTS),
        "legacy_exp017_values": [],
        "operator_class": OPERATOR_IMAGE_SIDE,
        "form": "t'_k = t_k; z' = normalize(z - {c} mu_I)",
        "shared_vector": "0",
        "decomposition_triple": "image_only",
        "note": "the image half of gr_clip_style_two_sided at the same coefficient; acts as a class-dependent score shift -{c}<mu_I, t_k> before renormalization",
    },
    {
        "family": "clean_boundary_active",
        "parameter": "step",
        "values": list(LEGACY_STEPS + NEW_STEPS),
        "legacy_exp017_values": list(LEGACY_STEPS),
        "operator_class": OPERATOR_TEXT_TRANSLATION,
        "form": "t'_k = normalize(t_k + {c} u_clean); z' = z",
        "shared_vector": "{c} u_clean",
    },
    {
        "family": "cohen_aligned_noisy_margin",
        "parameter": "step",
        "values": list(LEGACY_STEPS + NEW_STEPS),
        "legacy_exp017_values": list(LEGACY_STEPS),
        "operator_class": OPERATOR_TEXT_TRANSLATION,
        "form": "t'_k = normalize(t_k + {c} u_noisy); z' = z",
        "shared_vector": "{c} u_noisy",
    },
    {
        "family": "registered_control",
        "operator_class": OPERATOR_SUPERVISED,
        "controls": [
            {
                "control_id": "learned_shared_translation_tangent",
                "form": "t'_k = normalize(t_k + v - <v, t_k> t_k); z' = z",
                "training": "one shared vector v fitted by noisy+clean cross-entropy on the registered control role (bank_construction.control_training)",
            },
            {
                "control_id": "lowrank_tangent_r8",
                "form": "t'_k = normalize(t_k + (L R)_k - <(L R)_k, t_k> t_k), L in R^{K x 8}, R in R^{8 x d}; z' = z",
                "training": "rank-8 tangent adapter fitted by noisy+clean cross-entropy on the registered control role (EXP-019 recipe)",
            },
            {
                "control_id": "noisy_class_mean",
                "form": "t'_k = normalize(mean over the unit noisy control-role draws of class k); z' = z",
                "training": "optimization-free",
            },
            {
                "control_id": "fewshot_prompt_bank",
                "form": "t'_k = normalize(TextEncoder([ctx_1 .. ctx_4] + 'a photo of a {class_k}.')); z' = z",
                "training": "PromptSmooth-inspired: four shared context vectors fitted by noisy+clean cross-entropy through the frozen text encoder on the registered control role; not a reproduction of PromptSmooth",
                "context_tokens": 4,
            },
        ],
    },
)

# Registered contrast families (research/EXP021_V2_AMENDMENT_2026-09-26.md, Section 4).
PRIMARY_SHARED_CONTRASTS: tuple[str, ...] = (
    "gr_clip_style_two_sided__coefficient_1",
    "text_only_centering__coefficient_1",
    "image_only_centering__coefficient_1",
    "clean_boundary_active__step_0.16",
    "cohen_aligned_noisy_margin__step_0.16",
)
CONTROL_CONTRASTS: tuple[str, ...] = tuple(f"{CONTROL_PREFIX}{control_id}" for control_id in CONTROL_IDS)
DECOMPOSITION = {
    "two_sided": "gr_clip_style_two_sided__coefficient_{c}",
    "text_only": "text_only_centering__coefficient_{c}",
    "image_only": "image_only_centering__coefficient_{c}",
}
# V3 (re-review n2): the learned shared tangent control has the exact form
# t'_k = normalize((1 - <v, t_k>) t_k + v), so the worker's containment counter can
# be evaluated for it whenever every a_k = 1 - <v, t_k> is positive.  It is not a
# registered flip-theorem bank: its counts are exploratory, reported outside Q3/Q4.
EXPLORATORY_CONTAINMENT: tuple[str, ...] = (f"{CONTROL_PREFIX}learned_shared_translation_tangent",)
IMAGE_MEAN_ALGORITHM = (
    "satml2027ext/banks_ext.py::portable_column_mean: column sums in a fixed cascade order (blocks of 16 rows, "
    "4 accumulator levels, the order of torch's CPU cascade sum), one IEEE-754 addition per element per step in the "
    "feature dtype, then one correctly rounded division by the row count; every operation is fixed, so the bits do "
    "not depend on the platform; on the 12 EXP-016 development feature tensors it equals torch 2.11 CPU "
    "mean(dim=0) and the calibration image-mean hash recorded by EXP-017 bit for bit"
)
EXPLORATORY_CONTAINMENT_NOTE = ("exploratory: containment counted when every a_k = 1 - <v, t_k> > 0; "
                                "reported outside Q3/Q4, never part of a registered verdict")
BUDGET_SUBSET_021C: tuple[str, ...] = (
    IDENTITY_CANDIDATE_ID,
    "gr_clip_style_two_sided__coefficient_1",
    "cohen_aligned_noisy_margin__step_0.16",
)


def fmt(value: float) -> str:
    """Canonical number formatting used in every candidate id (``%.12g``)."""

    return f"{float(value):.12g}"


def family_table() -> list[dict[str, Any]]:
    return copy.deepcopy(list(_FAMILIES))


def _expand(families: Sequence[Mapping[str, Any]], cell_id: str | None) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    seen_families: set[str] = set()

    def add(candidate_id: str, **fields: Any) -> None:
        if any(spec["candidate_id"] == candidate_id for spec in specs):
            raise ValueError(f"duplicate candidate id {candidate_id}")
        spec = {"candidate_id": candidate_id, "cell_id": cell_id, "position": len(specs)}
        spec.update(fields)
        specs.append(spec)

    for entry in families:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("family"), str):
            raise ValueError("every family entry must be a mapping with a family name")
        family = entry["family"]
        if family in seen_families:
            raise ValueError(f"family {family} is listed twice")
        seen_families.add(family)
        legacy = {float(value) for value in entry.get("legacy_exp017_values", [])}
        operator_class = entry.get("operator_class")
        if family == IDENTITY_CANDIDATE_ID:
            add(IDENTITY_CANDIDATE_ID, family=family, objective_id=None, parameter=None, value=None,
                image_transform=None, role="identity", legacy_exp017=bool(entry.get("legacy_exp017", False)),
                operator_class=OPERATOR_IDENTITY, form=entry.get("form"))
        elif family in COEFFICIENT_FAMILIES:
            for value in _values(entry, family):
                add(
                    f"{family}__coefficient_{fmt(value)}",
                    family=family,
                    objective_id=None,
                    parameter="coefficient",
                    value=value,
                    image_transform=(
                        {"type": IMAGE_TRANSFORM_CENTER, "coefficient": value} if family in IMAGE_SIDE_FAMILIES else None
                    ),
                    role="registered_grid",
                    legacy_exp017=value in legacy,
                    operator_class=operator_class,
                    form=str(entry.get("form", "")).replace("{c}", fmt(value)),
                )
        elif family in STEP_FAMILIES:
            for value in _values(entry, family):
                add(
                    f"{family}__step_{fmt(value)}",
                    family="proposal",  # the frozen EXP-017 metadata name of both step families
                    objective_id=family,
                    parameter="step",
                    value=value,
                    image_transform=None,
                    role="registered_grid",
                    legacy_exp017=value in legacy,
                    operator_class=operator_class,
                    form=str(entry.get("form", "")).replace("{c}", fmt(value)),
                )
        elif family == "registered_control":
            controls = entry.get("controls")
            if not isinstance(controls, Sequence) or isinstance(controls, (str, bytes)) or not controls:
                raise ValueError("registered_control family must list controls")
            for control in controls:
                control_id = control["control_id"] if isinstance(control, Mapping) else control
                if control_id not in CONTROL_IDS:
                    raise ValueError(f"unsupported control {control_id!r}; supported: {CONTROL_IDS}")
                fields: dict[str, Any] = {}
                if isinstance(control, Mapping) and control_id == "fewshot_prompt_bank":
                    fields["context_tokens"] = int(control.get("context_tokens", 4))
                add(
                    f"{CONTROL_PREFIX}{control_id}",
                    family="registered_control",
                    objective_id=control_id,
                    parameter=None,
                    value=None,
                    image_transform=None,
                    role="registered_control",
                    legacy_exp017=False,
                    operator_class=OPERATOR_SUPERVISED,
                    form=control.get("form") if isinstance(control, Mapping) else None,
                    **fields,
                )
        else:
            raise ValueError(f"unsupported candidate family {family!r}")
    return specs


def _values(entry: Mapping[str, Any], family: str) -> list[float]:
    values = entry.get("values")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError(f"family {family} must list values")
    parsed = [float(value) for value in values]
    if any(not math.isfinite(value) for value in parsed):
        raise ValueError(f"family {family} has a non-finite value")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"family {family} repeats a value")
    return parsed


def canonical_specs(cell_id: str | None = None) -> list[dict[str, Any]]:
    return _expand(_FAMILIES, cell_id)


def canonical_candidate_ids() -> list[str]:
    return [spec["candidate_id"] for spec in canonical_specs()]


def _flat_entries(specs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    decomposition = {}
    for key, pattern in DECOMPOSITION.items():
        for c in COEFFICIENTS:
            decomposition[pattern.format(c=fmt(c))] = key
    entries = []
    for spec in specs:
        entry = {
            "candidate_id": spec["candidate_id"],
            "family": spec["objective_id"] if spec["family"] == "proposal" else spec["family"],
            "parameter": spec["parameter"],
            "value": spec["value"],
            "form": spec.get("form"),
            "operator_class": spec["operator_class"],
            "legacy_exp017": spec["legacy_exp017"],
            "image_transform": spec["image_transform"],
            "flip_theorem": ("applies" if spec["operator_class"] == OPERATOR_TEXT_TRANSLATION
                             else EXPLORATORY_CONTAINMENT_NOTE if spec["candidate_id"] in EXPLORATORY_CONTAINMENT
                             else "not applicable"),
        }
        if spec["candidate_id"] in decomposition:
            entry["decomposition_triple"] = decomposition[spec["candidate_id"]]
        entries.append(entry)
    return entries


def canonical_candidates_block() -> dict[str, Any]:
    """The registration ``candidates`` block for 021A and 021B (36 banks)."""

    specs = canonical_specs()
    ids = [spec["candidate_id"] for spec in specs]
    if len(ids) != 36 or len(set(ids)) != 36:
        raise AssertionError(f"canonical family must have 36 unique ids, got {len(ids)}")
    return {
        "schema_version": CANDIDATES_SCHEMA_VERSION,
        "expected_bank_count": len(ids),
        "count_rule": "1 identity + 3 global mean-centering + 3 projected gap + 3 two-sided + 3 text-only + 3 image-only + 8 clean steps + 8 noisy steps + 4 controls = 36",
        "identity_candidate_id": IDENTITY_CANDIDATE_ID,
        "control_prefix": CONTROL_PREFIX,
        "notation": {
            "T": "source prototype bank [K, d], unit rows",
            "mean(t)": "T.mean(dim=0)",
            "mean_w(t)": "(1/K) 1^T T (frozen EXP-017 weighted text mean; equals mean(t) up to float bits)",
            "mu_I": "mean of the unit clean development image features of the cell",
            "V": "orthonormal basis of the row space of T - 1 mean_w(t)^T (SVD, frozen EXP-017 threshold)",
            "u_clean": "unit EXP-016 clean boundary-active development direction",
            "u_noisy": "unit EXP-016 Cohen-aligned noisy-margin development direction",
            "normalize": "row-wise division by the Euclidean norm",
        },
        "families": family_table(),
        "candidate_ids": ids,
        "candidates": _flat_entries(specs),
        "decomposition": {key: [pattern.format(c=fmt(c)) for c in COEFFICIENTS] for key, pattern in DECOMPOSITION.items()},
        "decomposition_note": "text_only_centering and gr_clip_style_two_sided share the prototype tensor at each coefficient; two_sided = text_only prototypes + image_only transform, so the triple is an exact decomposition of the operator (effects need not be additive; the interaction is reported)",
        "flip_theorem_candidates": [spec["candidate_id"] for spec in specs if spec["operator_class"] == OPERATOR_TEXT_TRANSLATION],
        "flip_theorem_requirement": "every flip-theorem bank must record its exact operator with every a_k = 1 (a shared translation); the bank validator refuses a payload otherwise, and the analysis reports Q3/Q4 as not evaluable if any cell lacks the counters",
        "exploratory_containment_candidates": list(EXPLORATORY_CONTAINMENT),
        "exploratory_containment_rule": EXPLORATORY_CONTAINMENT_NOTE,
        "projected_gap_candidates": [spec["candidate_id"] for spec in specs if spec["family"] == "chowers_exact_projected_gap"],
        "builder": "satml2027ext/banks_ext.py::build_registered_banks from satml2027ext/candidates_ext.py::candidate_specs (this block is generated from the same module)",
        "frozen_after_hash": True,
    }


def subset_candidates_block(candidate_ids: Sequence[str], *, parent_experiment_id: str,
                            parent_registration_sha256: str) -> dict[str, Any]:
    """A registration block that scores a registered subset of a parent's banks (021C)."""

    parent = canonical_candidates_block()
    ids = [str(value) for value in candidate_ids]
    unknown = [value for value in ids if value not in parent["candidate_ids"]]
    if unknown:
        raise ValueError(f"subset ids are not in the parent family: {unknown}")
    if IDENTITY_CANDIDATE_ID not in ids:
        raise ValueError("a scored subset must contain the identity candidate")
    ordered = [value for value in parent["candidate_ids"] if value in set(ids)]
    if ordered != ids:
        raise ValueError("subset ids must be listed in the parent's registered order")
    return {
        "schema_version": CANDIDATES_SCHEMA_VERSION,
        "expected_bank_count": len(ids),
        "identity_candidate_id": IDENTITY_CANDIDATE_ID,
        "control_prefix": CONTROL_PREFIX,
        "subset_of": {
            "experiment_id": parent_experiment_id,
            "registration_sha256": parent_registration_sha256,
            "parent_expected_bank_count": parent["expected_bank_count"],
            "rule": "the worker loads the parent cell's approved bank payload (bank manifest entry of the parent cell), validates it against the parent registration, and scores exactly these candidates in the parent's order",
        },
        "families": parent["families"],
        "candidate_ids": ids,
        "candidates": [entry for entry in parent["candidates"] if entry["candidate_id"] in set(ids)],
        "frozen_after_hash": True,
    }


def _block(registration: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(registration, Mapping) or not isinstance(registration.get("candidates"), Mapping):
        raise ValueError("registration has no candidates block")
    block = registration["candidates"]
    if block.get("schema_version") != CANDIDATES_SCHEMA_VERSION:
        raise ValueError(f"candidates block must declare schema {CANDIDATES_SCHEMA_VERSION}")
    return block


def candidate_specs(registration: Mapping[str, Any], cell: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Expand the registration's block into ordered specs and check them against its explicit id list.

    For a subset block (021C) the parent family is expanded and filtered; each
    returned spec keeps its parent position so the bank payload order is the
    parent's.
    """

    block = _block(registration)
    cell_id = None
    if cell is not None:
        if not isinstance(cell.get("cell_id"), str) or not cell["cell_id"]:
            raise ValueError("cell must carry a nonempty cell_id")
        cell_id = cell["cell_id"]
    families = block.get("families")
    if not isinstance(families, Sequence) or isinstance(families, (str, bytes)) or not families:
        raise ValueError("candidates block must list families")
    specs = _expand(families, cell_id)
    listed = block.get("candidate_ids")
    if not isinstance(listed, list) or not listed or len(set(listed)) != len(listed):
        raise ValueError("candidates block must carry a nonempty, repeat-free candidate_ids list")
    if "subset_of" in block:
        by_id = {spec["candidate_id"]: spec for spec in specs}
        missing = [value for value in listed if value not in by_id]
        if missing:
            raise ValueError(f"subset ids absent from the parent family: {missing}")
        subset = [spec for spec in specs if spec["candidate_id"] in set(listed)]
        if [spec["candidate_id"] for spec in subset] != list(listed):
            raise ValueError("subset candidate_ids are not in the parent's registered order")
        specs = subset
    else:
        expanded = [spec["candidate_id"] for spec in specs]
        if expanded != list(listed):
            raise ValueError("expanded families differ from the registered candidate_ids list")
    expected = block.get("expected_bank_count")
    if expected is not None and len(specs) != int(expected):
        raise ValueError(f"candidates block expands to {len(specs)} specs, registration expects {expected}")
    if IDENTITY_CANDIDATE_ID not in {spec["candidate_id"] for spec in specs}:
        raise ValueError("the identity candidate is not registered")
    return specs


def registered_candidate_ids(registration: Mapping[str, Any], cell: Mapping[str, Any] | None = None) -> list[str]:
    return [spec["candidate_id"] for spec in candidate_specs(registration, cell)]


def is_subset_registration(registration: Mapping[str, Any]) -> bool:
    return "subset_of" in _block(registration)


def subset_parent(registration: Mapping[str, Any]) -> dict[str, Any] | None:
    block = _block(registration)
    return dict(block["subset_of"]) if "subset_of" in block else None


def flip_theorem_ids(candidate_ids: Sequence[str]) -> list[str]:
    classes = {spec["candidate_id"]: spec["operator_class"] for spec in canonical_specs()}
    return [value for value in candidate_ids if classes.get(value) == OPERATOR_TEXT_TRANSLATION]


def step_value(candidate_id: str) -> float | None:
    for family in STEP_FAMILIES:
        prefix = f"{family}__step_"
        if candidate_id.startswith(prefix):
            return float(candidate_id[len(prefix):])
    return None
