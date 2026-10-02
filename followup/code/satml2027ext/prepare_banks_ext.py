#!/usr/bin/env python3
"""EXP-021 preparation stage (V2): build, validate and seal one cell's bank payload.

V1 had bank library code but no executable that produced the payloads the
worker consumes, and a payload could validate without proving where it came
from (V1 review findings P0-C and P0-D).  This stage closes that gap.  For one
registered 021A or 021B cell it:

1. refuses to run unless ``research/EXP021_APPROVAL.json`` authorizes the
   ``preparation`` stage (exact registration and dependency binding), the data
   preflight passed, the runtime environment equals the preflight's, and the
   Hugging Face hub is offline;
2. loads only the registered ``development`` and control-role lists, in
   registered order, verifying ordered hashes, CSV hashes, image-file hashes and
   that no evaluation, official-test or sealed item is named (no evaluation
   pixel is ever decoded here);
3. builds the sources:
   * 021A (``source_mode = exp016_saved_tensors``): the EXP-016 text prototypes,
     development clean features, 16 proposal noisy draws per item and both
     directions of the same fold, each file verified against the EXP-016
     artifact manifest bound in the registration; the 18 legacy banks are the
     saved EXP-017 tensors, bit for bit (V4), and the image mean has EXP-017's
     recorded hash, so 021A refreshes the certification draws and the execution
     pipeline on the same images and legacy bank tensors;
   * 021B (``source_mode = encode_registered_roles``): prompt-ensemble
     prototypes, clean and 64 noisy development draws, the two directions with
     the approved EXP-016/EXP-019 builders, and a 16-draw control-role cache,
     all through the checkpoint-bound model;
4. trains the controls with the registered recipe (EXP-019 ``fit``, imported
   unchanged) and the few-shot prompt control (``prompt_bank``) on the
   registered control role;
5. builds all registered banks, attaches a provenance block (ordered role
   hashes, prompt and recipe hashes, checkpoint bytes and model state, data
   preflight, approval, dependency closure, environment, feature/direction/
   control hashes), validates the payload with recomputation, and writes it
   without overwriting.

021C cells have no payload of their own: they score the approved payload of
their 021B parent cell.

V3 (internal re-review of V2, 2026-09-26):

* the provenance fields that describe source tensors (development features,
  directions, control features and labels, prompt prototypes and context
  vectors) are written from the stored, prepared tensors and the bank
  validator checks them against the payload (re-review M1a);
* the few-shot prompt control receives the registered per-class budget
  (``prompt_control.shots_per_class``: 5 in 021A, 20 in 021B) instead of the
  observed maximum, so the learner's refusal is a real check (m1);
* deterministic algorithms are requested (warn-only) and any
  nondeterministic-operation warning raised while training the controls is
  recorded in the provenance; trained controls are compared descriptively on a
  re-run, never within a claimed tolerance (n4);
* a stage-1 approval that names a data preflight binds this stage to that
  exact file (M2b); in any case the bank manifest records the preflight used
  and the sampling stage refuses a manifest built from another preflight (M2c).

V4 (reviews of reviewer bundle V3, 2026-09-26): 021A passes the saved EXP-017
tensors (``candidate_prototype_banks.pt``) to the bank builder, which stores
them and uses them as the 18 legacy banks bit for bit; EXP-017's
``candidate_metadata.json`` is bound and its calibration image-mean hash must
equal the portable image mean; ``legacy_reproduction`` records the bitwise
identity and, separately, the formula conformance within 2e-6 with its
runtime-dependent exact-match count (text review item 5, audit E6).

Usage (one GPU per process):
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python satml2027ext/prepare_banks_ext.py \
      --registration configs/satml2027ext/exp-20260921-021b.json --cell-id <cell> \
      --data-root data --data-preflight results/satml2027ext/EXP021_DATA_PREFLIGHT.json
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib
import platform
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from satml2027ext._common import PROJECT_ROOT, canonical_json_hash, read_json, sha256_file, write_json_atomic  # noqa: E402
from satml2027ext import banks_ext, candidates_ext, items_ext  # noqa: E402
from satml2027ext.guard_ext import (  # noqa: E402
    DEFAULT_APPROVAL_PATH,
    dependency_closure,
    find_cell,
    load_registration,
    verify_approval,
)
from common.hashing import tensor_scientific_hash  # noqa: E402
from common.seeds import (  # noqa: E402
    CONTROL_TRAIN_STREAM,
    DEVELOPMENT_STREAM,
    NOISE_BLOCK,
    block_plan,
    derive_block_seed,
    derive_item_seed,
)

PREPARATION_SCHEMA = "satml2027ext.preparation.v1"
SOURCE_EXP016 = "exp016_saved_tensors"
SOURCE_ENCODE = "encode_registered_roles"
LEGACY_REPRODUCTION_ATOL = 2e-6
LEGACY_BANK_COUNT = 18
EXP017_FILES = ("candidate_prototype_banks.pt", "candidate_metadata.json")
EXP016_FILES = ("text_prototypes.pt", "clean_features.pt", "proposal_noisy_features.pt", "candidate_directions.pt",
                "cell_summary.json", "ground_truth_evaluation_only.pt")
SOURCE_LOADERS = {
    "cifar100_reserved_not_accessed": "cifar100",
    "eurosat_reserved_not_accessed": "eurosat",
    "imagenette_train": "imagenette_train",
}


# --------------------------------------------------------------------------- recipe


def construction(registration: Mapping[str, Any]) -> dict[str, Any]:
    block = registration.get("bank_construction")
    if not isinstance(block, Mapping):
        raise RuntimeError("registration lacks a bank_construction block")
    for key in ("source_mode", "control_training", "prompt_control"):
        if key not in block:
            raise RuntimeError(f"bank_construction lacks {key}")
    if block["source_mode"] not in (SOURCE_EXP016, SOURCE_ENCODE):
        raise RuntimeError(f"unsupported source_mode {block['source_mode']!r}")
    return dict(block)


def fit_function() -> Callable[..., Any]:
    """The approved EXP-019 control fitter, imported unchanged."""

    return importlib.import_module("day1.d1_03b_train_controls").fit


def project_directions_function() -> Callable[..., Any]:
    """The approved EXP-019 direction builder (EXP-016 interfaces), imported unchanged."""

    return importlib.import_module("day1.d1_02_build_development").project_directions


# --------------------------------------------------------------------------- backend (model-facing)


@dataclass
class Backend:
    """Everything that needs a model or pixels; tests inject fakes."""

    device: str
    mean: tuple[float, ...]
    std: tuple[float, ...]
    encode_images: Callable[[Any], Any]
    load_pixels: Callable[[str, list[dict[str, str]]], Any]
    class_names: Callable[[str], list[str]]
    text_prototypes: Callable[[str], Any]
    learn_prompt: Callable[..., Any]
    checkpoint: dict[str, Any] = field(default_factory=dict)


def unit_rows(tensor):
    import torch

    return tensor / torch.linalg.vector_norm(tensor, dim=-1, keepdim=True)


def clean_features(backend: Backend, pixels, *, batch: int = 64):
    import torch

    mean = torch.tensor(backend.mean, dtype=torch.float32, device=backend.device).view(1, 3, 1, 1)
    std = torch.tensor(backend.std, dtype=torch.float32, device=backend.device).view(1, 3, 1, 1)
    parts = []
    with torch.inference_mode():
        for start in range(0, pixels.shape[0], batch):
            chunk = (pixels[start : start + batch].to(backend.device) - mean) / std
            parts.append(unit_rows(backend.encode_images(chunk).to(torch.float32)).cpu())
    return torch.cat(parts)


def noisy_features(backend: Backend, pixels, rows: Sequence[Mapping[str, str]], *, draws: int, base_seed: int,
                   stream_role: str, model_id: str, dataset_id: str, sigma: float):
    """``[items, draws, d]`` unit noisy features with the frozen block seeding (EXP-019 d1_02/d1_03b)."""

    import torch

    mean = torch.tensor(backend.mean, dtype=torch.float32, device=backend.device).view(1, 3, 1, 1)
    std = torch.tensor(backend.std, dtype=torch.float32, device=backend.device).view(1, 3, 1, 1)
    output = None
    for position, row in enumerate(rows):
        item_seed = derive_item_seed(base_seed=base_seed, model_id=model_id, dataset_id=dataset_id, sigma=float(sigma),
                                     item_id=row["item_id"], stream_role=stream_role)
        pixel = pixels[position].to(backend.device)
        collected = []
        for block_index, block_draws in block_plan(draws):
            generator = torch.Generator(device=backend.device)
            generator.manual_seed(derive_block_seed(item_seed, block_index))
            noise = torch.randn((NOISE_BLOCK, *pixel.shape), generator=generator, dtype=torch.float32,
                                device=backend.device)[:block_draws]
            standardized = ((pixel.unsqueeze(0) + float(sigma) * noise) - mean) / std
            with torch.inference_mode():
                collected.append(unit_rows(backend.encode_images(standardized).to(torch.float32)).cpu())
        features = torch.cat(collected)
        if output is None:
            output = torch.zeros((len(rows), draws, features.shape[1]), dtype=torch.float32)
        output[position] = features
    return output


# --------------------------------------------------------------------------- 021A sources


def load_exp016_sources(registration: Mapping[str, Any], cell: Mapping[str, Any], dev_rows: Sequence[Mapping[str, str]],
                        *, exp016_root: Path, exp017_root: Path) -> dict[str, Any]:
    """EXP-016 development tensors of the same fold, hash-verified against the bound artifact manifests."""

    import torch

    bound = construction(registration).get("exp016_sources") or {}
    manifests = {}
    for key, root in (("exp016", exp016_root), ("exp017", exp017_root)):
        path = Path(root) / "artifact_manifest.json"
        if sha256_file(path) != bound.get(f"{key}_artifact_manifest_sha256"):
            raise RuntimeError(f"{key.upper()} artifact manifest differs from the registered hash")
        manifests[key] = {entry["path"]: entry["sha256"] for entry in read_json(path)["artifacts"]}
    legacy_cell = f"{cell['model_id']}__{cell['dataset_id']}__fold{int(cell['fold'])}"
    files: dict[str, str] = {}
    for name in EXP016_FILES:
        relative = f"cells/{legacy_cell}/{name}"
        observed = sha256_file(Path(exp016_root) / relative)
        if manifests["exp016"].get(relative) != observed:
            raise RuntimeError(f"EXP-016 file {relative} differs from its artifact manifest")
        files[relative] = observed
    for name in EXP017_FILES:
        relative = f"cells/{legacy_cell}/{name}"
        observed = sha256_file(Path(exp017_root) / relative)
        if manifests["exp017"].get(relative) != observed:
            raise RuntimeError(f"EXP-017 file {relative} differs from its artifact manifest")
        files[f"exp017/{relative}"] = observed
    bank_relative = f"cells/{legacy_cell}/candidate_prototype_banks.pt"
    exp017_metadata = read_json(Path(exp017_root) / f"cells/{legacy_cell}/candidate_metadata.json")
    recorded_means = {entry.get("calibration_image_mean_sha256") for entry in exp017_metadata.values()
                      if isinstance(entry, Mapping) and entry.get("calibration_image_mean_sha256")}
    if len(recorded_means) != 1:
        raise RuntimeError(f"EXP-017 {legacy_cell} records {len(recorded_means)} calibration image means, expected one")

    base = Path(exp016_root) / "cells" / legacy_cell
    summary = read_json(base / "cell_summary.json")
    sample_ids = [str(value) for value in summary["sample_ids"]]
    registered = {row["item_id"]: int(row["label"]) for row in dev_rows}
    if sorted(sample_ids) != sorted(registered) or len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError("EXP-016 development items differ from the registered development list")
    labels = torch.tensor([registered[item_id] for item_id in sample_ids], dtype=torch.long)
    saved_labels = torch.load(base / "ground_truth_evaluation_only.pt", map_location="cpu", weights_only=True).to(torch.long)
    if not torch.equal(saved_labels, labels):
        raise RuntimeError("EXP-016 development labels differ from the registered labels")
    directions = torch.load(base / "candidate_directions.pt", map_location="cpu", weights_only=True)
    sources = {
        "prototypes": torch.load(base / "text_prototypes.pt", map_location="cpu", weights_only=True).to(torch.float32).contiguous(),
        "development_clean_features": torch.load(base / "clean_features.pt", map_location="cpu", weights_only=True).to(torch.float32).contiguous(),
        "development_noisy_features": torch.load(base / "proposal_noisy_features.pt", map_location="cpu", weights_only=True).to(torch.float32).contiguous(),
        "clean_direction": directions["clean_direction"],
        "noisy_direction": directions["noisy_direction"],
        "item_ids": sample_ids,
        "labels": labels,
        "exp017_banks": torch.load(Path(exp017_root) / bank_relative, map_location="cpu", weights_only=True),
        "exp017_image_mean_sha256": next(iter(recorded_means)),
        "files": files,
        "exp016_order_sha256": canonical_json_hash(sample_ids),
    }
    for key, summary_key in (("prototypes", "prototype_bank_sha256"), ("development_clean_features", "clean_features_sha256"),
                             ("development_noisy_features", "noisy_features_sha256")):
        if tensor_scientific_hash(sources[key]) != summary.get(summary_key):
            raise RuntimeError(f"EXP-016 {key} differ from the {summary_key} recorded in its cell summary")
    return sources


# --------------------------------------------------------------------------- controls


def train_controls(prototypes, clean, noisy, labels, recipe: Mapping[str, Any], *, device: str, fit=None) -> dict[str, Any]:
    """Shared tangent translation and rank-8 tangent adapter with the registered EXP-019 recipe."""

    fit = fit or fit_function()
    training = recipe["control_training"]
    common = dict(rank=int(training["rank"]), epochs=int(training["epochs"]), lr=float(training["learning_rate"]),
                  temperature=float(training["classification_temperature"]), clean_weight=float(training["clean_loss_weight"]),
                  device=device, optimizer_seed=int(training["optimizer_seed"]), minibatch_seed=int(training["minibatch_seed"]),
                  batch_size=int(training["batch_size"]), weight_decay=float(training["adamw_weight_decay"]))
    shared, shared_stats = fit(prototypes, clean, noisy, labels, mode="shared", **common)
    lowrank, lowrank_stats = fit(prototypes, clean, noisy, labels, mode="lowrank", **common)
    return {
        "shared_delta": shared[0].to("cpu").contiguous(),
        "lowrank_left": lowrank[0].to("cpu").contiguous(),
        "lowrank_right": lowrank[1].to("cpu").contiguous(),
        "stats": {"shared": shared_stats, "lowrank": lowrank_stats,
                  "shared_delta_norm": float(shared[0].norm())},
    }


def observed_shots(labels) -> int:
    """The largest per-class count among the supervision labels (reported, never used as the budget)."""

    import torch

    return int(torch.bincount(labels).max())


def registered_shots(recipe: Mapping[str, Any]) -> int:
    """The registered few-shot budget per class (V3, re-review m1); the prompt learner refuses more."""

    value = (recipe.get("prompt_control") or {}).get("shots_per_class")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError("bank_construction.prompt_control.shots_per_class must be a registered positive integer")
    return int(value)


def deterministic_attention():
    """The math SDPA kernel for the prompt learner (V3).

    The local V3 smoke (2026-09-26) recorded that the memory-efficient attention
    kernel's backward pass through the text encoder is nondeterministic; the math
    kernel is deterministic, so the prompt control is reproducible on one GPU class.
    """

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:  # pragma: no cover - older torch
        return contextlib.nullcontext()
    return sdpa_kernel(SDPBackend.MATH)


def _nondeterminism(caught: Sequence[warnings.WarningMessage]) -> list[str]:
    messages = sorted({str(item.message).splitlines()[0][:240] for item in caught
                       if "determinis" in str(item.message).lower()})
    return messages


# --------------------------------------------------------------------------- assembly


def build_payload(registration: Mapping[str, Any], cell: Mapping[str, Any], *, prototypes, development_clean,
                  clean_direction, noisy_direction, control_clean, control_noisy, control_labels, controls: Mapping[str, Any],
                  prompt, provenance: Mapping[str, Any], legacy_banks: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build every registered bank, attach provenance, and validate with recomputation (fail closed).

    ``legacy_banks`` (021A only): the saved EXP-017 tensors, which become the 18 legacy banks bit for bit.
    """

    import torch

    specs = candidates_ext.candidate_specs(registration, cell)
    sources: dict[str, Any] = {
        "prototypes": prototypes,
        "development_clean_features": development_clean,
        "clean_direction": clean_direction,
        "noisy_direction": noisy_direction,
        "controls": {
            "shared_delta": controls["shared_delta"],
            "lowrank_left": controls["lowrank_left"],
            "lowrank_right": controls["lowrank_right"],
            "noisy_features": control_noisy,
            "labels": control_labels,
            "fewshot_prompt_prototypes": prompt.prototypes,
            "fewshot_context_vectors": prompt.context_vectors,
        },
    }
    if legacy_banks is not None:
        sources["legacy_banks"] = {key: value.to(torch.float32).contiguous() for key, value in legacy_banks.items()}
    banks = banks_ext.build_registered_banks(specs, sources)
    # V3 (re-review M1a): the source-describing provenance fields are the hashes of the stored
    # (prepared) tensors, so the validator can hold the payload to them; the hashes of the tensors
    # as loaded are kept beside them for transparency.
    provenance = copy.deepcopy(dict(provenance))
    bindings = banks_ext.provenance_source_bindings(sources)
    provenance.setdefault("feature_hashes", {}).update(bindings["feature_hashes"])
    if "direction_hashes" in provenance:
        provenance["direction_hashes_as_loaded"] = dict(provenance["direction_hashes"])
    provenance["direction_hashes"] = dict(bindings["direction_hashes"])
    provenance.setdefault("prompt_control", {}).update(bindings["prompt_control"])
    payload = banks_ext.make_bank_payload(banks, registration=registration, cell=cell, sources=sources, provenance=provenance)
    banks_ext.validate_registered_bank_payload_v2(payload, registration, cell)
    return payload


def legacy_reproduction(payload: Mapping[str, Any], saved: Mapping[str, Any], registration: Mapping[str, Any],
                        cell: Mapping[str, Any], *, exp017_image_mean_sha256: str) -> dict[str, Any]:
    """021A legacy record (V4; text review of reviewer bundle V3, item 5; audit E6).

    Two separate checks, both fail closed:

    * identity: every one of the 18 sampled legacy banks is the saved EXP-017 tensor bit for bit, and the
      image mean used by the image-side transforms has the hash EXP-017 recorded for its calibration mean;
    * conformance: the registered formulas applied to the stored EXP-016 sources reproduce the saved tensors
      within 2e-6.  Exact equality of the reconstruction is reported as a diagnostic only, because its last
      bits depend on the runtime (12 of 18 exact on the auditors' Linux runtime, 18 of 18 here).
    """

    import torch

    if len(saved) != LEGACY_BANK_COUNT:
        raise RuntimeError(f"the bound EXP-017 file holds {len(saved)} banks, expected {LEGACY_BANK_COUNT}")
    identical = []
    for candidate_id, tensor in sorted(saved.items()):
        if candidate_id not in payload["candidates"]:
            raise RuntimeError(f"legacy bank {candidate_id} is not registered")
        ours = payload["candidates"][candidate_id]["prototypes"]
        if not torch.equal(ours, tensor.to(torch.float32)):
            raise RuntimeError(f"sampled legacy bank {candidate_id} is not the saved EXP-017 tensor bit for bit")
        identical.append(candidate_id)
    image_mean_sha256 = payload["source_hashes"]["image_mean"]
    if image_mean_sha256 != exp017_image_mean_sha256:
        raise RuntimeError("the 021A image mean differs from the calibration image mean recorded by EXP-017 "
                           f"({image_mean_sha256} != {exp017_image_mean_sha256})")
    conformance = banks_ext.legacy_formula_conformance(payload, registration, cell)
    if conformance is None or conformance["banks"] != LEGACY_BANK_COUNT or not conformance["within_tolerance"]:
        raise RuntimeError(f"the registered formulas do not reproduce the saved EXP-017 banks within "
                           f"{LEGACY_REPRODUCTION_ATOL} ({None if conformance is None else conformance['max_abs_delta']})")
    return {
        "rule": "021A samples the saved EXP-017 tensors bit for bit; the EXP-016 formula reconstruction is a conformance check within 2e-6; exact reconstruction is a runtime-dependent diagnostic",
        "sampled_banks_equal_saved_exp017_bitwise": len(identical),
        "image_mean_equals_exp017_record": True,
        "image_mean_sha256": image_mean_sha256,
        "formula_conformance": conformance,
        "atol": LEGACY_REPRODUCTION_ATOL,
        "max_abs_delta": conformance["max_abs_delta"],
        "exact_formula_matches_diagnostic": conformance["exact_matches_diagnostic"],
    }


def environment_record(device: str) -> dict[str, Any]:
    import torch

    record = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" and torch.cuda.is_available() else "cpu",
        "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn_allowed": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }
    try:
        import open_clip

        record["open_clip"] = str(open_clip.__version__)
    except ImportError:
        record["open_clip"] = None
    return record


def prepare_cell(registration: Mapping[str, Any], cell: Mapping[str, Any], *, backend: Backend, items_dir: Path,
                 preflight: Mapping[str, Any], preflight_sha256: str, approval_sha256: str,
                 exp016_root: Path, exp017_root: Path, fit=None, directions_fn=None,
                 log: Callable[[str], None] = print) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(payload, control_cache)`` for one 021A/021B cell; nothing is written here."""

    import torch

    if candidates_ext.is_subset_registration(registration):
        raise RuntimeError("021C cells score their 021B parent's approved payload; they are not prepared")
    recipe = construction(registration)
    mode = recipe["source_mode"]
    control_role = banks_ext.control_role(registration)
    started = time.time()
    dev_rows, dev_record = items_ext.load_verified(items_dir, registration, cell, "development", preflight=preflight, preparation=True)
    roles = {"development": dev_record}
    if control_role != "development":
        ctrl_rows, ctrl_record = items_ext.load_verified(items_dir, registration, cell, control_role, preflight=preflight, preparation=True)
        roles[control_role] = ctrl_record
    else:
        ctrl_rows = dev_rows
    extra: dict[str, Any] = {}

    if mode == SOURCE_EXP016:
        if control_role != "development":
            raise RuntimeError("the EXP-016 source mode trains controls on the development role")
        saved = load_exp016_sources(registration, cell, dev_rows, exp016_root=exp016_root, exp017_root=exp017_root)
        prototypes = saved["prototypes"]
        development_clean = saved["development_clean_features"]
        clean_direction, noisy_direction = saved["clean_direction"], saved["noisy_direction"]
        control_clean, control_noisy, control_labels = development_clean, saved["development_noisy_features"], saved["labels"]
        control_ids = saved["item_ids"]
        extra["exp016_sources"] = {"files": saved["files"], "exp016_item_order_sha256": saved["exp016_order_sha256"],
                                   "control_noisy_draws_per_item": int(control_noisy.shape[1])}
        dev_noisy = None
    else:
        direction = recipe["direction_construction"]
        training = recipe["control_training"]
        prototypes = backend.text_prototypes(str(cell["dataset_id"])).to(torch.float32).contiguous()
        dev_pixels = backend.load_pixels(SOURCE_LOADERS[dev_rows[0]["source_split"]], dev_rows)
        development_clean = clean_features(backend, dev_pixels)
        dev_noisy = noisy_features(backend, dev_pixels, dev_rows, draws=int(direction["development_draws_per_item"]),
                                   base_seed=int(direction["noise_base_seed"]), stream_role=DEVELOPMENT_STREAM,
                                   model_id=str(cell["model_id"]), dataset_id=str(cell["dataset_id"]), sigma=float(cell["sigma"]))
        for key in ("split_manifest", "exclusion_manifest"):
            if sha256_file(PROJECT_ROOT / direction[f"{key}_path"]) != direction[f"{key}_sha256"]:
                raise RuntimeError(f"{key} differs from the registered hash")
        builder = directions_fn or project_directions_function()
        clean_direction, noisy_direction = builder(
            development_clean, dev_noisy, prototypes, float(direction["noisy_margin_temperature"]), PROJECT_ROOT,
            dataset_id=str(cell["dataset_id"]), sample_ids=[row["item_id"] for row in dev_rows],
            split_manifest=PROJECT_ROOT / direction["split_manifest_path"],
            exclusion_manifest=PROJECT_ROOT / direction["exclusion_manifest_path"], sigma=float(cell["sigma"]),
        )
        ctrl_pixels = backend.load_pixels(SOURCE_LOADERS[ctrl_rows[0]["source_split"]], ctrl_rows)
        control_clean = clean_features(backend, ctrl_pixels)
        control_noisy = noisy_features(backend, ctrl_pixels, ctrl_rows, draws=int(training["train_draws_per_item"]),
                                       base_seed=int(training["noise_base_seed"]), stream_role=CONTROL_TRAIN_STREAM,
                                       model_id=str(cell["model_id"]), dataset_id=str(cell["dataset_id"]), sigma=float(cell["sigma"]))
        control_labels = torch.tensor([int(row["label"]) for row in ctrl_rows], dtype=torch.long)
        control_ids = [row["item_id"] for row in ctrl_rows]
    log(f"{cell['cell_id']}: sources ready ({time.time() - started:.1f} s)")

    shots = registered_shots(recipe)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        controls = train_controls(prototypes, control_clean, control_noisy, control_labels, recipe, device=backend.device, fit=fit)
        prompt_recipe = recipe["prompt_control"]
        names = backend.class_names(str(cell["dataset_id"]))
        with deterministic_attention():
            prompt = backend.learn_prompt(names, control_clean, control_noisy, control_labels, prompt_recipe,
                                          shots_per_class=shots)
    nondeterministic = _nondeterminism(caught)
    log(f"{cell['cell_id']}: controls trained ({time.time() - started:.1f} s)")

    control_cache = {
        "cell_id": cell["cell_id"], "role": control_role, "item_ids": control_ids,
        "clean_features": control_clean.contiguous(), "noisy_features": control_noisy.contiguous(),
        "labels": control_labels.contiguous(), "sealed_evaluation_access": False,
    }
    provenance = {
        "schema_version": PREPARATION_SCHEMA,
        "experiment_id": registration["experiment_id"],
        "registration_sha256": registration["registration_sha256"],
        "cell_id": cell["cell_id"],
        "model_id": cell["model_id"],
        "dataset_id": cell["dataset_id"],
        "sigma": float(cell["sigma"]),
        "source_mode": mode,
        "roles": roles,
        "roles_loaded": sorted(roles),
        "prompt_config": dict(registration["prompt_configs"][cell["dataset_id"]]),
        "checkpoint": dict(backend.checkpoint),
        "recipe_sha256": canonical_json_hash(registration["bank_construction"]),
        "data_preflight_sha256": preflight_sha256,
        "approval_sha256": approval_sha256,
        "code_sha256": dependency_closure(),
        "environment": environment_record(backend.device),
        "feature_hashes": {
            "development_clean": tensor_scientific_hash(development_clean),
            "development_noisy": None if dev_noisy is None else tensor_scientific_hash(dev_noisy),
            "control_clean": tensor_scientific_hash(control_clean),
            "control_noisy": tensor_scientific_hash(control_noisy),
            "control_labels": tensor_scientific_hash(control_labels),
            "control_item_order_sha256": canonical_json_hash(control_ids),
        },
        "direction_hashes": {"clean": tensor_scientific_hash(clean_direction), "noisy": tensor_scientific_hash(noisy_direction)},
        "control_training": controls["stats"],
        "prompt_control": {
            "initial_loss": float(prompt.initial_loss), "final_loss": float(prompt.final_loss),
            "hook_max_abs_difference": float(getattr(prompt, "hook_difference", float("nan"))),
            "context_vectors_sha256": tensor_scientific_hash(prompt.context_vectors),
            "prototypes_sha256": tensor_scientific_hash(prompt.prototypes),
            "shots_per_class": shots,
            "observed_max_per_class": observed_shots(control_labels),
        },
        "trained_control_reproducibility": {
            "deterministic_algorithms_requested": True,
            "prompt_attention_kernel": "math (deterministic scaled-dot-product attention) while learning the prompt control",
            "nondeterministic_operation_warnings": nondeterministic,
            "rule": "trained controls are compared descriptively on a re-run (max |delta| reported); no tolerance is claimed; the tensors sealed in the approved bank manifest are the ones scored",
        },
        "control_cache_tensor_sha256": {key: tensor_scientific_hash(value) for key, value in control_cache.items()
                                        if hasattr(value, "shape")},
        "sealed_evaluation_access": False,
        "evaluation_items_loaded": 0,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.time() - started, 2),
        **extra,
    }
    payload = build_payload(
        registration, cell, prototypes=prototypes, development_clean=development_clean,
        clean_direction=clean_direction, noisy_direction=noisy_direction, control_clean=control_clean,
        control_noisy=control_noisy, control_labels=control_labels, controls=controls, prompt=prompt,
        provenance=provenance, legacy_banks=saved["exp017_banks"] if mode == SOURCE_EXP016 else None,
    )
    if mode == SOURCE_EXP016:
        reproduction = legacy_reproduction(payload, saved["exp017_banks"], registration, cell,
                                           exp017_image_mean_sha256=saved["exp017_image_mean_sha256"])
        payload["provenance"]["legacy_reproduction"] = reproduction
        banks_ext.validate_registered_bank_payload_v2(payload, registration, cell)
    return payload, control_cache


def write_cell(payload: Mapping[str, Any], control_cache: Mapping[str, Any], *, banks_dir: Path, preparation_dir: Path) -> dict[str, Any]:
    import torch

    cell_id = payload["cell_id"]
    cache_path = Path(preparation_dir) / f"{cell_id}__control_cache.pt"
    if cache_path.exists():
        raise FileExistsError(f"refusing to overwrite {cache_path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(control_cache), cache_path)
    pt_path, json_path = banks_ext.save_bank_payload(payload, Path(banks_dir) / f"{cell_id}__banks.pt")
    summary = {
        "schema_version": PREPARATION_SCHEMA,
        "cell_id": cell_id,
        "bank_file": pt_path.name,
        "bank_file_sha256": sha256_file(pt_path),
        "bank_sidecar_sha256": sha256_file(json_path),
        "control_cache_file": cache_path.name,
        "control_cache_sha256": sha256_file(cache_path),
        "provenance": payload["provenance"],
    }
    write_json_atomic(Path(preparation_dir) / f"{cell_id}__preparation.json", summary)
    return summary


# --------------------------------------------------------------------------- real backend


def real_backend(registration: Mapping[str, Any], cell: Mapping[str, Any], *, device: str, data_root: Path,
                 preflight: Mapping[str, Any], cache_dir: str | None) -> Backend:
    """Checkpoint-bound model, verified pixel loaders and the real prompt learner."""

    import torch

    from satml2027ext import datasets_ext, models_ext, prompt_bank
    from satml2027ext.preflight_data_ext import verify_image_file
    from common.datasets import class_names as frozen_class_names, load_dataset, recover_pixels
    from common.models import build_text_prototypes
    from common.prompts import render_prompts

    loaded, record = models_ext.load_bound_model(str(cell["model_id"]), registration["models"][str(cell["model_id"])],
                                                 device=device, cache_dir=cache_dir)
    datasets: dict[str, Any] = {}

    def dataset_for(key: str):
        if key not in datasets:
            if key == "imagenette_train":
                datasets[key] = datasets_ext.load_dataset_ext("imagenette", data_root, transform=loaded.preprocess, split="train")
            elif key == "imagenette_val":
                datasets[key] = datasets_ext.load_dataset_ext("imagenette", data_root, transform=loaded.preprocess, split="val")
            else:
                datasets[key] = load_dataset(key, data_root, transform=loaded.preprocess)
        return datasets[key]

    def load_pixels(key: str, rows):
        dataset, item_ids, labels = dataset_for(key)
        for row in rows:
            index = int(row["dataset_index"])
            if item_ids[index] != row["item_id"] or int(labels[index]) != int(row["label"]):
                raise RuntimeError(f"dataset identity differs from the registered list at {row['item_id']}")
            verify_image_file(preflight, data_root, row["item_id"])
        return recover_pixels(dataset, [int(row["dataset_index"]) for row in rows], loaded.mean, loaded.std)

    def prompt_config(dataset_id: str) -> dict[str, Any]:
        entry = registration["prompt_configs"][dataset_id]
        path = PROJECT_ROOT / entry["path"]
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"prompt configuration for {dataset_id} differs from the registered hash")
        return read_json(path)

    def class_names(dataset_id: str) -> list[str]:
        config = prompt_config(dataset_id)
        if dataset_id == "imagenette":
            return datasets_ext.class_names_ext(dataset_for("imagenette_train")[0], config)
        source = {"cifar100": "cifar100", "cifar100_test": "cifar100", "eurosat": "eurosat", "eurosat_sealed": "eurosat"}[dataset_id]
        return frozen_class_names(dataset_for(source)[0], config)

    def text_prototypes(dataset_id: str):
        config = prompt_config(dataset_id)
        return build_text_prototypes(loaded, render_prompts(class_names(dataset_id), config["templates"]))

    def learn_prompt(names, clean, noisy, labels, recipe, *, shots_per_class):
        text_encoder_fn, token_embedding_fn = prompt_bank.open_clip_prompt_hooks(loaded.model, loaded.tokenizer, device=device)
        rendered = prompt_bank.render_prompts(names, recipe["template"], context_tokens=int(recipe["context_tokens"]),
                                              placeholder=recipe["placeholder"])
        difference = prompt_bank.verify_hooks_against_encode_text(loaded.model, loaded.tokenizer, rendered, device=device,
                                                                   atol=float(recipe["hook_verification_atol"]))
        learned = prompt_bank.learn_context_prompts(
            text_encoder_fn, token_embedding_fn, names, recipe["template"], noisy, clean, labels,
            context_tokens=int(recipe["context_tokens"]), steps=int(recipe["steps"]), lr=float(recipe["learning_rate"]),
            seed=int(recipe["seed"]), temperature=float(recipe["temperature"]),
            clean_loss_weight=float(recipe["clean_loss_weight"]), shots_per_class=shots_per_class,
            placeholder=recipe["placeholder"], init_std=float(recipe["init_std"]),
        )
        learned.hook_difference = difference
        return learned

    return Backend(device=device, mean=tuple(loaded.mean), std=tuple(loaded.std),
                   encode_images=loaded.model.encode_image, load_pixels=load_pixels, class_names=class_names,
                   text_prototypes=text_prototypes, learn_prompt=learn_prompt, checkpoint=record)


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None, *, approval_path: Path | None = None, backend_factory=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--cell-id", action="append", default=None, help="repeatable; default: every cell")
    parser.add_argument("--items", type=Path, default=PROJECT_ROOT / "results/satml2027ext/items")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--data-preflight", required=True, type=Path)
    parser.add_argument("--banks-out", type=Path, default=PROJECT_ROOT / "results/satml2027ext/banks")
    parser.add_argument("--preparation-out", type=Path, default=PROJECT_ROOT / "results/satml2027ext/preparation")
    parser.add_argument("--exp016", type=Path, default=PROJECT_ROOT / "results/EXP-20260906-016")
    parser.add_argument("--exp017", type=Path, default=PROJECT_ROOT / "results/EXP-20260906-017")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args(argv)

    from satml2027ext import models_ext
    from satml2027ext.preflight_data_ext import load_preflight, verify_environment
    from satml2027ext.run_shard_ext import configure_precision

    registration = load_registration(args.registration)
    approval = verify_approval(registration, stage="preparation", approval_path=approval_path or DEFAULT_APPROVAL_PATH,
                               stage_files={"data_preflight": args.data_preflight})
    preflight = load_preflight(args.data_preflight, registrations=approval["registrations"])
    verify_environment(preflight)
    models_ext.require_offline()
    configure_precision()
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cell_ids = args.cell_id or [cell["cell_id"] for cell in registration["cells"]]
    for cell_id in cell_ids:
        cell = find_cell(registration, cell_id)
        factory = backend_factory or real_backend
        backend = factory(registration, cell, device=device, data_root=args.data_root, preflight=preflight,
                          cache_dir=args.cache_dir)
        payload, cache = prepare_cell(registration, cell, backend=backend, items_dir=args.items, preflight=preflight,
                                      preflight_sha256=sha256_file(args.data_preflight),
                                      approval_sha256=approval["_approval_sha256"], exp016_root=args.exp016,
                                      exp017_root=args.exp017)
        summary = write_cell(payload, cache, banks_dir=args.banks_out, preparation_dir=args.preparation_out)
        print(f"{cell_id}: {len(payload['candidate_order'])} banks sealed, sha256 {summary['bank_file_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
