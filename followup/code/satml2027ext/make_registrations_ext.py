#!/usr/bin/env python3
"""Emit and hash the EXP-021 V4 registrations (schema satml2027ext.registration.v4).

Reads ``results/satml2027ext/items/item_manifest_ext.json`` (unchanged since
2026-09-26; its item lists passed both independent item audits) and writes

  configs/satml2027ext/exp-20260921-021a.json   same-item fresh-noise replication and decomposition
  configs/satml2027ext/exp-20260921-021b.json   powered fresh-item study (three backbones)
  configs/satml2027ext/exp-20260921-021c.json   4,096 versus 100,000 draws on 100 items per dataset

plus a ``.sha256`` sidecar per file.  V4 answers the two reviews of reviewer
bundle V3 (research/EXP021_V4_AMENDMENT_2026-09-26.md): 021A samples the saved
EXP-017 tensors of its 18 legacy banks bit for bit (the EXP-016 reconstruction
becomes a conformance check) and pins the image mean by a portable algorithm;
the source mode, the count contract, the budget join, the projected-gap
counts, the Q1 planning file (sharp variance bound, rounded floats, portable
comparison), the Q1 sentences and the power labels under scope descent are
corrected; every stage approval names a git commit holding the approved files.
V3 answered the internal adversarial
re-review of V2 (research/EXP021_V3_AMENDMENT_2026-09-26.md): 021A's Q1 is one
registered contrast with an equal-cell primary estimand, two disjoint tests
against zero and against the saved-draw EXP-017 value (numbers from
``results/satml2027ext/planning/EXP021A_Q1_PLANNING.json``, bound by hash) and
descriptive verdict sentences; the families carry power labels; the forward
batch, the determinism settings, the few-shot budget and the checkpoint bytes
are registered; the preflight, access and not-run rules are stated as the code
enforces them.  V2 answered the two independent pre-sampling reviews of V1
(research/EXP021_V2_AMENDMENT_2026-09-26.md): the
candidate block, every formula and the analysis families come from
``satml2027ext/candidates_ext.py``; the registrations bind the bank-construction
recipe, the checkpoint bytes, the data archives, stage-specific access, the
sharding rule, per-draw decision-change counters, the 021C budget design, the
corrected power statement and machine-checkable question rules.

A registration whose item lists are still ``PENDING_ITEM_LIST`` cannot be
finalized (none is pending since 2026-09-26).  A finalized file is
byte-identical on regeneration when ``--registered-at`` is fixed.

Usage:
  python satml2027ext/make_registrations_ext.py --items results/satml2027ext/items \
      --out configs/satml2027ext --registered-at 2026-09-26T12:00:00+00:00
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _common import PROJECT_ROOT, canonical_json_hash, read_json, sha256_file, write_json_atomic  # noqa: E402
from common.certify import min_successes_for_radius  # noqa: E402
from common.seeds import CONFIRMATION_STREAM, CONTROL_TRAIN_STREAM, DEVELOPMENT_STREAM, NOISE_BLOCK, SELECTION_STREAM  # noqa: E402
from draw_items_ext import EXPERIMENT_IDS, PENDING, PHASE3_FOLDS, PLAN_021A, SUBSET_021C  # noqa: E402
from satml2027ext import candidates_ext  # noqa: E402

SCHEMA_VERSION = "satml2027ext.registration.v4"
REGISTRATION_VERSION = "V4"
PLAN_PATH = "research/EXP021_POWERED_FOLLOWUP_PLAN_2026-09-25.md"
AMENDMENT_PATH = "research/EXP021_V4_AMENDMENT_2026-09-26.md"
PREVIOUS_AMENDMENTS = ("research/EXP021_V3_AMENDMENT_2026-09-26.md", "research/EXP021_V2_AMENDMENT_2026-09-26.md")
Q1_PLANNING_PATH = "results/satml2027ext/planning/EXP021A_Q1_PLANNING.json"
REREVIEW_PATH = "research/independent_review/exp021_V2_internal_adversarial_rereview_2026-09-26.md"
REVIEWS_V3 = (
    "research/independent_review/reviewer_bundle_V3_critical_audit_2026-09-26.pdf",
    "research/independent_review/reviewer_bundle_V3_critical_audit_evidence_2026-09-26.zip",
    "research/independent_review/reviewer_bundle_V3_text_review_2026-09-26.md",
)
SUPERSEDED_V3 = {
    "EXP-20260921-021A": "f47daa7cce41f82d4ef816e8be37c757ee1acf35f38ccce4cfcf0158b14b2cde",
    "EXP-20260921-021B": "419b7fed404208de94a60a8c531473c50c387c37449c78c21d69924afc9efcc7",
    "EXP-20260921-021C": "1b7b9a7327fe71a5d5f5ce7b53589e24efec3b0c4b51d8d92fa3f0a959e02763",
}
SUPERSEDED_V2 = {
    "EXP-20260921-021A": "e9efc3115195b4e69e71fb9863fcd8c9b4d0df0f337096f68ea05c484091c5c6",
    "EXP-20260921-021B": "e12d0a8971b6fabdc261db7657956675540ec16147f8d7afee015bec3be9fb9c",
    "EXP-20260921-021C": "8fdf50b93e2ee1b56db6a8048fdf322ae6f807394c5946aae77796693888ce58",
}
BLOCKS_PER_FORWARD = 4
PROMPT_SHOTS_PER_CLASS = {"021a": 5, "021b": 20}
APPROVAL_FILE = "research/EXP021_APPROVAL.json"
SUPERSEDED_V1 = {
    "EXP-20260921-021A": "eae3040a1abeb9a76fb81a18150ae01b3e59225c1aa217c52f371befe7fa56a3",
    "EXP-20260921-021B": "a258319205c586aeb3379245c37405df9e35d657cfa01b2bfba992060cc361fb",
    "EXP-20260921-021C": "6a6000230f2cf5860420080a323b1ab64bcb1ce026f41dc0a31d1f32a12fc8ba",
}
SUPERSEDED_EARLIER = {"EXP-20260921-021A": "d3c4a826e8144644db13207171a7a94382d813f8f32e6ff531b14b6499e99a8e"}

MODELS_021A = ["openai-clip-vit-b32-quickgelu", "openai-clip-vit-l14-quickgelu"]
MODELS_021B = ["openai-clip-vit-b32-quickgelu", "openai-clip-vit-l14-quickgelu", "openclip-vit-b32-laion2b"]
DATASETS_021B = ["cifar100_test", "eurosat_sealed", "imagenette"]
SEALED_DATASETS = {"cifar100_test", "eurosat_sealed"}
SIGMAS_021B = [0.12, 0.25]
SIGMA_021A = 0.25
SIGMA_021C = 0.25
RETIRED_021B_LADDER = {
    "models": ["openai-clip-vit-b16-quickgelu", "openai-clip-rn50-quickgelu"],
    "sigma": 0.5,
    "retired_on": "2026-09-26",
    "reason": "covered by EXP-019A/B (five backbones, sigma 0.12/0.25/0.5, fresh items; D-128); low value per GPU-hour",
}

RADII = [0.0, 0.1, 0.25, 0.5]
ALPHA_PER_EXAMPLE = 0.001
SELECTION_DRAWS = 128
CONFIRMATION_DRAWS = 4096
CONFIRMATION_DRAWS_021C = 100000
BUDGET_CHECKPOINTS_021C = [4096]
SESOI_POINTS = 2.0
BOOTSTRAP_REPLICATES = 100000
BOOTSTRAP_SEED = 2026092101
FAMILY_ALPHA = 0.05
FLIP_TOLERANCE = 1e-5
MAX_ITEMS_PER_SHARD = 1000

SEEDS = {
    "selection_base_seed": 20260921001,
    "confirmation_base_seed": 20260921002,
    "development_noise_base_seed": 20260921004,
    "control_noise_base_seed": 20260921005,
    "control_optimizer_seed": 20260921006,
    "control_minibatch_seed": 20260921007,
    "prompt_learner_seed": 20260921009,
}

PROMPT_CONFIGS = {
    "cifar100": "configs/prompts/cifar100_openai_readme.json",
    "cifar100_test": "configs/prompts/cifar100_openai_readme.json",
    "eurosat": "configs/prompts/eurosat_openai_ensemble_v1.json",
    "eurosat_sealed": "configs/prompts/eurosat_openai_ensemble_v1.json",
    "imagenette": "configs/satml2027ext/prompts/imagenette_openai_v1.json",
}

# Checkpoint bytes (blob SHA-256 recorded for EXP-019 on 2026-09-25, D-123; the files are open_clip_model.safetensors).
MODEL_BINDINGS = {
    "openai-clip-vit-b32-quickgelu": {
        "open_clip_name": "ViT-B-32-quickgelu", "pretrained": "openai",
        "hf_repository": "timm/vit_base_patch32_clip_224.openai", "snapshot": "a6f597a3",
        "file": "open_clip_model.safetensors", "bytes": 605143284,
        "sha256": "e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31",
    },
    "openai-clip-vit-l14-quickgelu": {
        "open_clip_name": "ViT-L-14-quickgelu", "pretrained": "openai",
        "hf_repository": "timm/vit_large_patch14_clip_224.openai", "snapshot": "18d05354",
        "file": "open_clip_model.safetensors", "bytes": 1710517724,
        "sha256": "9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e",
    },
    "openclip-vit-b32-laion2b": {
        "open_clip_name": "ViT-B-32", "pretrained": "laion2b_s34b_b79k",
        "hf_repository": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K", "snapshot": "1a25a446",
        "file": "open_clip_model.safetensors", "bytes": 605143316,
        "bytes_source": "Hugging Face LFS metadata of the repository (2026-09-26): the file's LFS oid equals the registered SHA-256",
        "sha256": "ac4f8c4b88af6d963118cbf40ad93176d092abbedfcb752601ae1866352656e6",
    },
}

DATA_ROOTS = {
    "cifar100": {
        "archive": "cifar-100-python.tar.gz",
        "source": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
        "archive_md5": "eb9058c3a382ffc7106e4002c42a8d85",
        "archive_sha256": "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7",
        "members_md5": {
            "cifar-100-python/train": "16019d7e3df5f24257cddd939b257f8d",
            "cifar-100-python/test": "f0ef6b0ae62326f3e7ffdfab6717acfc",
            "cifar-100-python/meta": "7973b15100ade9c7d40fb424638fde48",
        },
    },
    "imagenette": {
        "archive": "imagenette2-320.tgz",
        "source": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
        "bytes": 341663724,
        "archive_sha256": "569b4497c98db6dd29f335d1f109cf315fe127053cedf69010d047f0188e158c",
        "official_counts": {"train": 9469, "val": 3925},
        "resolution_note": "320-pixel shorter side; the model preprocessing resizes the shorter side to 224 (bicubic) and centre-crops 224 x 224, so the noise lives in a downsampled model-input space, not an upsampled one as for CIFAR-100 and EuroSAT (V1 called this 'native 224 resolution', which was imprecise)",
    },
    "eurosat": {
        "archive": "EuroSAT.zip",
        "archive_sha256": "8ebea626349354c5328b142b96d0430e647051f26efc2dc974c843f25ecf70bd",
        "folder": "eurosat/2750",
    },
    "item_lists": {"created_at": "2026-09-26T00:00:00+00:00",
                   "generator": "satml2027ext/draw_items_ext.py (with --imagenette-archive)"},
    "rule": "the sampling host runs satml2027ext/preflight_data_ext.py before the preparation stage; it must reproduce every item list byte for byte, tie every registered EuroSAT and Imagenette file to its archive member by SHA-256, and record every file hash; the bank manifest records the preflight the preparation used, the sampling approval binds that same file (the worker refuses a manifest that names another), a stage-1 approval that names a preflight binds the preparation to it, and the preparation stage and the worker check each file they load against the record",
    "preflight_inputs": "regenerating the lists needs, on the host, the Phase-3 and Phase-4 assignment files and the EXP-019 item lists (satml2027ext/draw_items_ext.py defaults); a missing input is a failed preflight check",
}


class PendingItemListError(RuntimeError):
    """The registration still contains PENDING_ITEM_LIST and cannot be hashed."""


# --------------------------------------------------------------------------- shared blocks


def certification_block(confirmation_draws: int, *, budget_checkpoints: list[int], store_margins: bool) -> dict[str, Any]:
    return {
        "selection_draws": SELECTION_DRAWS,
        "confirmation_draws": confirmation_draws,
        "alpha_per_example": ALPHA_PER_EXAMPLE,
        "alpha_form": "one-sided Clopper-Pearson lower bound per example on the selected class (CERTIFY); abstain if it is at most 0.5",
        "reported_radii": RADII,
        "primary_radius_rule": "r = sigma (k_min identical at every sigma)",
        "k_min_at_r_equals_sigma": min_successes_for_radius(confirmation_draws, ALPHA_PER_EXAMPLE, 0.25, 0.25),
        "radius_coordinates": "model-input coordinates: Gaussian noise added to the 224 x 224 tensor after the model's deterministic resize and centre crop, unclipped",
        "tf32": "both torch TF32 switches off and cuDNN deterministic in the preparation stage and the worker",
        "determinism": "torch.use_deterministic_algorithms(True, warn_only=True) and CUBLAS_WORKSPACE_CONFIG=:4096:8 in the preparation stage and the worker (satml2027ext/run_shard_ext.py::configure_precision); bit identity is expected only within one GPU class and software stack",
        "blocks_per_forward": BLOCKS_PER_FORWARD,
        "forward_batch_draws": BLOCKS_PER_FORWARD * NOISE_BLOCK,
        "forward_batch_rule": "every item's draws are encoded in forward batches of blocks_per_forward x noise_block_draws; the worker refuses any other value and the run custody manifest checks every sidecar",
        "streams": {
            "selection": SELECTION_STREAM,
            "confirmation": CONFIRMATION_STREAM,
            "development": DEVELOPMENT_STREAM,
            "control_train": CONTROL_TRAIN_STREAM,
            "independence": "selection and confirmation streams independent; all banks scored on identical draws",
            "seed_derivation": "satml2027/common/seeds.py::derive_item_seed(base, model_id, dataset_id, sigma, item_id, stream): a function of the cell's model, dataset and sigma, the stream and the item id and of nothing else (not of the shard split or the run order), so draws are shard-invariant, and a 021C item's confirmation stream (same base seed, model, dataset, sigma and item) begins with its 021B stream",
        },
        "seeds": SEEDS,
        "noise_block_draws": NOISE_BLOCK,
        "budget_checkpoints": budget_checkpoints,
        "margin_storage": ({
            "bank": candidates_ext.IDENTITY_CANDIDATE_ID,
            "stream": "confirmation",
            "per_draw": ["top1_class_id int16", "top2_class_id int16", "top1_margin float32 (cosine-score gap)"],
            "purpose": "descriptive record of the identity-bank margin distribution for Q3",
        } if store_margins else None),
        "flip_counters": {
            "tolerance": FLIP_TOLERANCE,
            "stream": "confirmation",
            "counters": ["changed draws (candidate argmax differs from the identity argmax)",
                         "containment violations: a changed draw j -> k with a_j (s_j - s_k) > |n_j - n_k| + |a_j - a_k| + tolerance (the approved N3C P1 form of the decision-change proposition)",
                         "loose-bound violations: a changed draw whose identity top-1 margin exceeds 2 min(1, ||v||) + tolerance (unit-scale translations)",
                         "loose flippable budget: draws whose identity top-1 margin is at most 2 min(1, ||v||) + tolerance",
                         "useful changes (identity wrong, candidate right) and harmful changes (identity right, candidate wrong)"],
            "applicability": "registered: every flip-theorem bank (candidates.flip_theorem_candidates), which the bank validator requires to record its exact operator t'_k = (t_k + v)/n_k (every a_k = 1) with an identity image transform, so containment and the loose bound both apply; Q3/Q4 are not evaluable if any cell lacks them. Exploratory: the learned shared tangent control (candidates.exploratory_containment_candidates, a_k = 1 - <v, t_k>) when every a_k > 0, reported outside Q3/Q4",
            "storage": "per candidate and item, int32, in every shard",
            "count_contract": ("satml2027ext/contract_ext.py, applied by the run custody stage and by the analysis merge: integer "
                               "nonnegative counts with exact totals; prefixes that sum to their checkpoints, never decrease and never "
                               "exceed the full counts; useful + harmful <= changed <= N, violations <= changed, loose budget <= N; "
                               "Boolean flags, equal across the shards of a cell, loose only with containment, zero counters where a "
                               "bound does not apply, none for the identity; the registered applicability per operator class"),
        },
    }


def sharding_block() -> dict[str, Any]:
    return {
        "max_items_per_shard": MAX_ITEMS_PER_SHARD,
        "rule": "a cell may be split into contiguous shards of at most 1,000 items of its registered order (satml2027ext/run_shard_ext.py::shard_slice); item seeds do not depend on the split (certification.streams.seed_derivation), so the realized draws do not either; every shard of a cell runs on one GPU class with the registered forward batch; the run custody manifest requires the shards to tile the registered order, with the registered labels and dataset indices, before any analysis",
    }


def access_block(sealed: bool, sealed_datasets: list[str]) -> dict[str, Any]:
    return {
        "list_construction": "item ids and labels of every role were read to draw the lists on 2026-09-26 (CIFAR-100 official test labels, EuroSAT final_test_sealed ids and labels from the Phase-3 assignments, the Imagenette folder listing); no image was decoded",
        "preparation": "only the development and registered control roles are loaded; evaluation, official-test and sealed items are refused by satml2027ext/items_ext.py; the payload records sealed_evaluation_access=false",
        "sampling": ("evaluation pixels of the sealed or official-test datasets " + ", ".join(sealed_datasets) + " are decoded by the worker only, once; no later selection may use them") if sealed
        else "evaluation pixels of the Phase-3 calibration_validation items (already used by EXP-017) are decoded by the worker only",
        "analysis": "outcomes are read only after the analysis-stage approval, which binds the run custody manifest",
        "sealed_access_definition": "decoding a sealed or official-test image, or computing any model output on it; reading list labels and hashing file bytes (the data preflight) are custody, not access",
        "sealed_evaluation_access": sealed,
    }


def stages_block() -> dict[str, Any]:
    return {
        "approval_file": APPROVAL_FILE,
        "approval_schema": "satml2027ext.approval.v3 (satml2027ext/guard_ext.py)",
        "source_commit": ("every approval names the git commit that holds exactly the approved dependency closure "
                          "(source_commit); every stage recomputes the git blob id of each approved file on disk and compares "
                          "it with that commit (guard_ext.verify_source_commit), so the host runs from a git checkout containing "
                          "the commit; uncommitted or differing files are a refusal"),
        "data_preflight": "a custody step that may run before the stage-1 approval: satml2027ext/preflight_data_ext.py reads labels and file bytes, decodes no image and computes no model output (a local smoke preflight ran on 2026-09-26 and is disclosed in research/DECISION_LOG.md)",
        "stage_1_preparation": "authorized after the independent pre-sampling review of this package; binds the three registration hashes and the complete dependency closure (and the host preflight's hash if it exists when the approval is written); runs satml2027ext/prepare_banks_ext.py and satml2027ext/bank_manifest_ext.py --data-preflight",
        "stage_2_sampling": "authorized after the reviewer inspects EXP021_DATA_PREFLIGHT.json and EXP021_BANK_MANIFEST.json; the approval adds both SHA-256 values; the worker refuses a bank manifest that names another preflight; runs satml2027ext/run_shard_ext.py",
        "stage_3_analysis": "authorized after the reviewer inspects the run custody manifests (satml2027ext/custody_ext.py, which lists any not-run cell); the 021C analysis also needs the approved 021B custody manifest for the registered prefix join; runs satml2027ext/analyze_ext.py",
        "planning": "satml2027ext/plan_ext.py reads registrations only and is not gated (it touches no data or outcome)",
        "rule": "every flag must be present and boolean; a missing flag, an unbound registration, any difference between the approved and current dependency files, or a source_commit that does not hold those files is a refusal",
    }


def stop_rules() -> list[str]:
    return [
        "nothing is authorized until the approval file authorizes the stage; the guard enforces it",
        "no evaluation item is loaded before the sampling stage; the preparation stage refuses evaluation, official-test and sealed items",
        "every registered cell that finishes is reported regardless of sign; a started cell is completed; a cell that never started is declared in the run custody manifest (custody_ext.py --not-run, which refuses any result file for it) and reported as not run",
        "no candidate, control, recipe, step, sigma, dataset, item list or checkpoint changes after the registration hash is recorded; a change is a new registration",
        "the sealed EuroSAT holdout and the CIFAR-100 test split are consumed once; no later selection may use them",
        "partial outcomes are not inspected; operational metadata only, as for EXP-019",
        "a per-draw containment or loose-bound violation is reported as a failed prediction and investigated; it never stops a run or removes a cell",
        "if any 021B result contradicts the EXP-017 narrative, the paper presents the boundary condition, not a uniform negative",
        "EXP-021 results enter no manuscript before custody verification and independent reconstruction",
    ]


def scope_ladder_block() -> dict[str, Any]:
    return {
        "levels": ["core", "minimal"],
        "definitions": {
            "core": "021A + 021B (three backbones, sigma 0.12 and 0.25) + 021C",
            "minimal": "021A + 021B at sigma 0.25 only, three backbones",
        },
        "rule": "descend only before the affected cells run, with a timestamped reason recorded in research/DECISION_LOG.md; the descended cells are declared not run in the run custody manifest",
        "retired_level": dict(RETIRED_021B_LADDER, level="full",
                              note="removed before registration; no cell of this level was ever sampled; adding it back would be a new registration"),
    }


def prompt_config_block(dataset_ids: list[str]) -> dict[str, Any]:
    return {dataset_id: {"path": PROMPT_CONFIGS[dataset_id], "sha256": sha256_file(PROJECT_ROOT / PROMPT_CONFIGS[dataset_id])}
            for dataset_id in dataset_ids}


def provenance_block(item_manifest: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "plan": PLAN_PATH,
        "plan_sha256": sha256_file(PROJECT_ROOT / PLAN_PATH),
        "amendment": AMENDMENT_PATH,
        "amendment_sha256": sha256_file(PROJECT_ROOT / AMENDMENT_PATH),
        "previous_amendments": [{"path": value, "sha256": sha256_file(PROJECT_ROOT / value)} for value in PREVIOUS_AMENDMENTS],
        "reviews": [{"path": value, "sha256": sha256_file(PROJECT_ROOT / value)} for value in REVIEWS_V3],
        "previous_reviews": [{"path": REREVIEW_PATH, "sha256": sha256_file(PROJECT_ROOT / REREVIEW_PATH)}],
        "item_manifest": "results/satml2027ext/items/item_manifest_ext.json",
        "item_manifest_sha256": canonical_json_hash(item_manifest),
        "phase3_assignments_sha256": item_manifest["inputs"]["phase3_assignments"]["sha256"],
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "exclusion_manifest_sha256": sha256_file(args.exclusion_manifest),
        "synthetic_inputs": bool(item_manifest.get("synthetic_inputs_allowed", False)),
    }


def supersedes_block(experiment_id: str) -> dict[str, Any]:
    earlier: list[dict[str, Any]] = [{
        "version": "V1",
        "registration_sha256": SUPERSEDED_V1[experiment_id],
        "status": "failed two independent pre-sampling reviews on 2026-09-26; never approved; never sampled",
        "reviews": "research/independent_review/ (2026-09-26 text review and 'Critical Audit' PDF); findings mapped in research/EXP021_V2_AMENDMENT_2026-09-26.md",
    }]
    if experiment_id in SUPERSEDED_EARLIER:
        earlier.append({"version": "V0", "registration_sha256": SUPERSEDED_EARLIER[experiment_id],
                        "status": "retired 2026-09-26 (D-130) before approval"})
    earlier.insert(0, {
        "version": "V2",
        "registration_sha256": SUPERSEDED_V2[experiment_id],
        "status": "superseded on 2026-09-26 before any external verdict: an internal adversarial re-review (a separate agent instance, not the team member's independent review) found one blocker (021A Q1), two major and six minor defects and five notes; never approved; never sampled",
        "review": REREVIEW_PATH,
        "findings_mapped_in": "research/EXP021_V3_AMENDMENT_2026-09-26.md",
    })
    return {
        "version": "V3",
        "registration_sha256": SUPERSEDED_V3[experiment_id],
        "status": ("superseded on 2026-09-26: two reviews of reviewer bundle V3 (a self-described AI-assisted critical audit and "
                   "a text review) held the stage-1 approval; they found that 021A reconstructed rather than loaded the legacy "
                   "banks, seven implementation gaps (source mode, non-finite values, count contract, budget join, variance "
                   "bound, planning portability, projected-gap labels) and that the source was not committed; never approved; "
                   "never sampled"),
        "reviews": list(REVIEWS_V3),
        "findings_mapped_in": AMENDMENT_PATH,
        "earlier": earlier,
    }


def hardware_block() -> dict[str, Any]:
    return {
        "assumed": "six RTX 5090 (32 GB) on one rented host reachable by SSH with one key; the university RTX 4090 is not part of this plan",
        "assignment_rule": "satml2027ext/plan_ext.py balances estimated seconds across the six GPUs (longest jobs first) after splitting cells into registered shards; device model, driver and library versions are recorded in every shard sidecar; all shards of a cell run on one GPU class",
    }


# --------------------------------------------------------------------------- bank construction recipes


CONTROL_REPRODUCIBILITY = ("deterministic algorithms are requested (warn-only) and any nondeterministic-operation warning is "
                           "recorded in the payload provenance; a re-run is compared descriptively (max |delta| reported) and no "
                           "tolerance is claimed; the tensors sealed in the approved bank manifest are the ones scored")


def control_training_block(role: str, *, exp016: bool) -> dict[str, Any]:
    block: dict[str, Any] = {
        "role": role,
        "reproducibility": CONTROL_REPRODUCIBILITY,
        "fit": "satml2027/day1/d1_03b_train_controls.py::fit (the EXP-019 function, imported unchanged)",
        "epochs": 80,
        "batch_size": 1024,
        "learning_rate": 0.005,
        "adamw_weight_decay": 0.0001,
        "classification_temperature": 0.05,
        "clean_loss_weight": 0.5,
        "rank": 8,
        "optimizer_seed": SEEDS["control_optimizer_seed"],
        "minibatch_seed": SEEDS["control_minibatch_seed"],
        "objective": "noisy + clean cross-entropy of the unit bank at temperature 0.05 (EXP-019 recipe); shared tangent translation and rank-8 tangent adapter",
    }
    if exp016:
        block["features"] = "the EXP-016 development clean features and the 16 EXP-016 proposal noisy draws per development item of the same fold (saved tensors, hash-verified); labels from the registered development list"
        block["train_draws_per_item"] = 16
    else:
        block["features"] = "clean features and 16 noisy draws per control_train item encoded at the cell's sigma with the control stream"
        block["train_draws_per_item"] = 16
        block["noise_base_seed"] = SEEDS["control_noise_base_seed"]
        block["stream_role"] = CONTROL_TRAIN_STREAM
    return block


def prompt_control_block(role: str, shots_per_class: int) -> dict[str, Any]:
    return {
        "role": role,
        "reproducibility": CONTROL_REPRODUCIBILITY,
        "learner": "satml2027ext/prompt_bank.py::learn_context_prompts with open_clip_prompt_hooks (verified against model.encode_text before use)",
        "context_tokens": 4,
        "template": "{context} a photo of a {name}.",
        "placeholder": "X",
        "class_names": "from the registered prompt configuration of the cell's dataset",
        "steps": 300,
        "optimizer": "adam_full_batch",
        "learning_rate": 0.002,
        "temperature": 0.05,
        "clean_loss_weight": 0.5,
        "init_std": 0.02,
        "seed": SEEDS["prompt_learner_seed"],
        "hook_verification_atol": 0.0001,
        "supervision": "the same clean features, noisy draws and labels as the other supervised controls",
        "shots_per_class": int(shots_per_class),
        "shots_rule": "the registered per-class budget is passed to the learner, which refuses supervision with more items in any class (satml2027ext/prepare_banks_ext.py::registered_shots)",
        "naming": "PromptSmooth-inspired few-shot context-prompt control; not a reproduction of PromptSmooth",
    }


def bank_construction_021a() -> dict[str, Any]:
    return {
        "executable": "satml2027ext/prepare_banks_ext.py",
        "source_mode": "exp016_saved_tensors",
        "exp016_sources": {
            "exp016_artifact_manifest_sha256": sha256_file(PROJECT_ROOT / "results/EXP-20260906-016/artifact_manifest.json"),
            "exp017_artifact_manifest_sha256": sha256_file(PROJECT_ROOT / "results/EXP-20260906-017/artifact_manifest.json"),
            "files_per_cell": ["text_prototypes.pt", "clean_features.pt", "proposal_noisy_features.pt", "candidate_directions.pt",
                               "cell_summary.json", "ground_truth_evaluation_only.pt", "EXP-017 candidate_prototype_banks.pt",
                               "EXP-017 candidate_metadata.json"],
            "legacy_reproduction_atol": 2e-6,
            "rule": ("every file must hash to its entry in the bound artifact manifest. The 18 legacy banks are the saved EXP-017 "
                     "tensors, loaded and stored bit for bit (V4); the registered formulas applied to the EXP-016 sources must "
                     "reproduce them within 2e-6 (a conformance check; exact reconstruction depends on the runtime and is reported "
                     "as a diagnostic). The image mean must have the calibration image-mean hash that EXP-017 recorded. 021A therefore "
                     "refreshes the certification draws and the execution pipeline on the same images and the same bank tensors"),
        },
        "legacy_banks": ("the saved EXP-017 tensors of candidate_prototype_banks.pt (18 banks), bit for bit; their operator records "
                         "(shared vector, row scales, row norms) follow the registered formulas and must reproduce the saved tensors "
                         "within the portable tolerance"),
        "text_prototypes": "the EXP-016 saved prompt-ensemble prototypes of the fold",
        "direction_construction": "the EXP-016 saved clean boundary-active and noisy-margin directions of the fold",
        "image_mean": ("satml2027ext/banks_ext.py::portable_column_mean of the EXP-016 saved unit clean development features "
                       "(in EXP-016 row order); its hash must equal the calibration_image_mean_sha256 of EXP-017's candidate metadata"),
        "image_mean_algorithm": candidates_ext.IMAGE_MEAN_ALGORITHM,
        "control_training": control_training_block("development", exp016=True),
        "noisy_class_mean": "normalize(mean of the unit noisy control-role draws of each class) from the same features",
        "prompt_control": prompt_control_block("development", PROMPT_SHOTS_PER_CLASS["021a"]),
        "origin_check": ("mandatory for every 021A cell and selected by this registration's source_mode (never by the payload): "
                         "satml2027ext/bank_manifest_ext.py re-derives every stored source tensor from the bound EXP-016 files, requires "
                         "the stored and sampled legacy banks to equal the bound EXP-017 file bit for bit and the image mean to have "
                         "EXP-017's recorded hash, and recomputes the formula conformance; it does not copy the preparation's own report"),
        "outputs": "results/satml2027ext/banks/<cell_id>__banks.pt (+ .json); results/satml2027ext/preparation/<cell_id>__control_cache.pt and __preparation.json",
    }


def bank_construction_021b(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "executable": "satml2027ext/prepare_banks_ext.py",
        "source_mode": "encode_registered_roles",
        "text_prototypes": "satml2027/common/models.py::build_text_prototypes over the registered prompt configuration (mean of unit prompt features, renormalized)",
        "direction_construction": {
            "builder": "satml2027/day1/d1_02_build_development.py::project_directions (EXP-016 interfaces, imported unchanged)",
            "role": "development",
            "development_draws_per_item": 64,
            "noise_base_seed": SEEDS["development_noise_base_seed"],
            "stream_role": DEVELOPMENT_STREAM,
            "noisy_margin_temperature": 0.05,
            "tie_tolerance": 1e-10,
            "split_manifest_path": "artifacts/day14/phase3_splits_v1/manifest.json",
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "exclusion_manifest_path": "artifacts/day14/phase3_splits_v1/exclusion_manifest.json",
            "exclusion_manifest_sha256": sha256_file(args.exclusion_manifest),
        },
        "image_mean": "satml2027ext/banks_ext.py::portable_column_mean of the unit clean development features",
        "image_mean_algorithm": candidates_ext.IMAGE_MEAN_ALGORITHM,
        "control_training": control_training_block("control_train", exp016=False),
        "noisy_class_mean": "normalize(mean of the unit noisy control_train draws of each class) from the control cache",
        "prompt_control": prompt_control_block("control_train", PROMPT_SHOTS_PER_CLASS["021b"]),
        "outputs": "results/satml2027ext/banks/<cell_id>__banks.pt (+ .json); results/satml2027ext/preparation/<cell_id>__control_cache.pt and __preparation.json",
    }


def bank_construction_021c() -> dict[str, Any]:
    return {
        "source_mode": "parent_payload",
        "rule": "021C builds no bank: each cell scores the approved payload of its 021B parent cell (the bank manifest links them); the worker scores the parent's complete 36-bank stack in the parent's order, so the logits of the first 4,096 draws come from the same GEMM shapes as in 021B, and stores exactly the three registered candidates",
    }


# --------------------------------------------------------------------------- inference and power


def family_power_021a() -> dict[str, Any]:
    return {
        "cifar100__sigma0.25": {"label": "underpowered for the 2-point smallest effect of interest", "unique_images": 1500,
                                "basis": "five-contrast family, conservative unique-image exact McNemar at 10% discordance: 2,993 images needed for 80% power"},
        "eurosat__sigma0.25": {"label": "underpowered", "unique_images": 150,
                               "basis": "150 unique images; descriptive only"},
        "pooled__sigma0.25": {"label": "underpowered for the 2-point smallest effect of interest (power about 0.47)", "unique_images": 1650,
                              "basis": "power block; Q1 is planned separately (power.q1_planning)"},
    }


def family_power_021b() -> dict[str, Any]:
    labels: dict[str, Any] = {}
    for sigma in SIGMAS_021B:
        key = f"{sigma:.12g}"
        labels[f"cifar100_test__sigma{key}"] = {"label": "powered (about 0.80 for 2 points at 10% discordance)", "unique_images": 3000}
        labels[f"imagenette__sigma{key}"] = {"label": "powered (about 0.80 for 2 points at 10% discordance)", "unique_images": 3000}
        labels[f"eurosat_sealed__sigma{key}"] = {"label": "underpowered (power about 0.25 for 2 points)", "unique_images": 1000}
        labels[f"pooled__sigma{key}"] = {"label": "well powered (about 0.997 for 2 points)", "unique_images": 7000}
    return labels


def inference_block_primary(family_power: dict[str, Any]) -> dict[str, Any]:
    return {
        "primary_estimand": "item-pooled paired difference in standard certified accuracy at r = sigma, candidate minus no_correction, in points",
        "families": {
            "definition": ["each dataset x sigma", "pooled over datasets at each sigma"],
            "rule": "multiplicity is controlled within each family; Q6 is adjudicated on the pooled families; dataset-specific families are secondary; every family carries its registered power label (family_power), and the analysis prints it beside each family and each Q6 row; 021A's Q1 is a single registered contrast outside these families (predictions, Q1)",
            "power_under_scope_descent": ("a family with a not-run cell reports its registered power label as 'registered' and an "
                                          "'effective' label stating that the registered power statement no longer applies"),
        },
        "family_power": family_power,
        "adjudication_families": "pooled_over_datasets_per_sigma",
        "primary_family": {
            "contrasts": list(candidates_ext.PRIMARY_SHARED_CONTRASTS),
            "reference": candidates_ext.IDENTITY_CANDIDATE_ID,
            "alpha_two_sided": FAMILY_ALPHA,
            "correction": "Holm within each family, applied separately to the direction, equivalence, gain and loss tests",
            "note": "shared operators only; the supervised controls are in control_family",
        },
        "control_family": {
            "contrasts": list(candidates_ext.CONTROL_CONTRASTS),
            "reference": candidates_ext.IDENTITY_CANDIDATE_ID,
            "comparator": "gr_clip_style_two_sided__coefficient_1",
            "correction": "Holm over the four controls within each family (direction test); the difference to the comparator is descriptive",
        },
        "direction": {
            "test": "two-sided percentile-bootstrap p-value 2 min(P*(d <= 0), P*(d >= 0))",
            "classification": ["positive", "negative", "unresolved"],
        },
        "practical_magnitude": {
            "sesoi_points": SESOI_POINTS,
            "tests": {
                "gain_at_least_sesoi": "H0: delta <= +SESOI; p = P*(d <= +SESOI)",
                "loss_at_least_sesoi": "H0: delta >= -SESOI; p = P*(d >= -SESOI)",
                "practically_equivalent": "percentile-bootstrap two one-sided tests in interval-inclusion form; p = max(P*(d <= -SESOI), P*(d >= +SESOI))",
            },
            "classification": ["gain_at_least_sesoi", "loss_at_least_sesoi", "practically_equivalent", "inconclusive"],
            "precedence": "gain, then loss, then equivalence, else inconclusive",
            "reporting": "direction and magnitude are reported side by side; a precise small positive effect is 'positive' and 'practically_equivalent'",
            "fixed_before_sampling": True,
        },
        "uncertainty": {
            "method": "class-stratified shared-item paired bootstrap with Rao-Wu rescaling",
            "resampling_unit": "the unique evaluation image; its bootstrap multiplicity is drawn once per replicate within its dataset-class stratum and shared across every backbone and sigma cell in which the image appears and across every candidate and radius",
            "strata": {"definition": ["dataset", "class"], "stratum_key": "(dataset_id, label)"},
            "replicates": BOOTSTRAP_REPLICATES,
            "generator": "numpy PCG64DXSM",
            "seed": BOOTSTRAP_SEED,
        },
        "intervals": {
            "unadjusted": "percentile 95% (90% as the description of the equivalence test); labelled unadjusted",
            "simultaneous": "single-step max-t band over all non-identity contrasts",
            "holm": "Holm adjusts decisions only; no Holm confidence interval is reported",
        },
        "secondary_family": {
            "method": "Romano-Wolf step-down max-t over every non-identity contrast; adjusted p of rank j = running maximum of P*(max over ranks >= k of |z| >= |t_(k)|)",
            "size": len(candidates_ext.canonical_candidate_ids()) - 1,
            "alpha": FAMILY_ALPHA,
        },
        "secondary_outcomes": [
            "anchored certified accuracy", "abstention", "certified-wrong rate and its top-class share",
            "raw/smoothed agreement", "clean accuracy (exact counts)", "equal-cell macros", "standard CA at the other radii",
        ],
        "clean_accuracy_reporting_gate": "macro drop <= 1 point and per-cell drop <= max(2 points, 3 images), exact integer arithmetic; flags, never hides",
        "no_selection_among_secondary_outcomes": True,
    }


def q1_planning() -> tuple[dict[str, Any], str]:
    path = PROJECT_ROOT / Q1_PLANNING_PATH
    record = read_json(path)
    body = {key: value for key, value in record.items() if key != "content_sha256"}
    if canonical_json_hash(body) != record.get("content_sha256"):
        raise RuntimeError("Q1 planning file content hash mismatch")
    return record, sha256_file(path)


def power_block_021a() -> dict[str, Any]:
    planning, planning_sha = q1_planning()
    saved = planning["saved_draws_under_registered_q1_analysis"]
    models = planning["fresh_draw_models"]
    q1 = {
        "file": Q1_PLANNING_PATH,
        "sha256": planning_sha,
        "content_sha256": planning["content_sha256"],
        "generator": ("satml2027ext/plan_q1_ext.py (deterministic; floats rounded to 10 significant digits; a regeneration must "
                      "agree under plan_q1_ext.compare_planning on any runtime, and is expected to be byte-identical on the reference "
                      "runtime recorded in the V4 amendment)"),
        "item_bootstrap_se_points": {name: saved[name]["bootstrap_se_points"] for name in ("macro", "pooled")},
        "saved_draw_z": {name: saved[name]["z"] for name in ("macro", "pooled")},
        "saved_draws_classified_as": {name: saved[name]["verdict"] for name in ("macro", "pooled")},
        "monte_carlo_sd_points": {model: models[model]["monte_carlo_sd_points"] for model in ("plug_in", "posterior_predictive")},
        "sharp_bound_normal_verdict_probabilities": {model: models[model]["sharp_bound_normal_verdict_probabilities"]
                                                     for model in ("plug_in", "posterior_predictive")},
        "expected_fresh_points": {model: models[model]["expected_points"] for model in ("plug_in", "posterior_predictive")},
        "verdict_probabilities": {model: models[model]["verdict_probabilities"] for model in ("plug_in", "posterior_predictive")},
        "verdict_probabilities_basis": "model-based: simulated with the candidate and identity outcomes coupled independently",
        "fixed_gain_scenarios_normal_approximation": planning["fixed_gain_scenarios_normal_approximation"],
        "reading": planning["reading"],
    }
    return {
        "q1_planning": q1,
        "method": "exact conditional McNemar, two-sided, Bonferroni-sized planning at 0.05/5, 80% power, 2-point paired difference (scripted as in research/LITERATURE_AND_POWER_RECHECK_2026-09-25.md Appendix A.5)",
        "inferential_unit": "the unique evaluation image; the two backbones score the same image, so the conservative count is 1,650 images, not 3,300 positions",
        "unique_images": 1650,
        "positions": 3300,
        "items_needed_for_5_contrast_family": {"0.05": 1502, "0.10": 2993, "0.20": 5925},
        "at_discordance_0.10": {"power_for_2_points": 0.47, "minimum_detectable_points": 2.70},
        "upper_bound_if_backbones_were_independent": {"power_for_2_points": 0.85, "minimum_detectable_points": 1.90},
        "statement": "021A contains 1,650 unique evaluation images in 3,300 backbone-image positions. Under the conservative unique-image calculation at 10% discordance its five-contrast family is underpowered for a 2-point effect (power about 0.47; minimum detectable effect about 2.7 points); that family is secondary for 021A. Q1 is planned separately (q1_planning): because the images and banks are EXP-017's, only the Monte-Carlo draws are fresh, and the planning file puts their standard deviation for the equal-cell gain near 0.06-0.09 points against an item-bootstrap standard error of about 0.72, so Q1 is expected to classify the outcome decisively; it measures the Monte-Carlo part of the post-hoc gain and checks the EXP-021 pipeline end to end, not item-level replication (021B, Q6). V1's '3,300 positions resolve about 2 points' was wrong; V2's Q1 rule (Holm-5 direction test of the item-pooled value, with a causal 'otherwise' sentence) is replaced.",
        "observed_discordance_rule": "the observed paired discordance is reported next to every interval",
    }


def power_block_021b() -> dict[str, Any]:
    return {
        "method": "exact conditional McNemar, two-sided, Bonferroni-sized planning at 0.05/5, 80% power, 2-point paired difference",
        "inferential_unit": "the unique evaluation image (each appears in three backbone cells per sigma; counted once)",
        "items_needed_for_5_contrast_family": {"0.05": 1502, "0.10": 2993, "0.20": 5925},
        "per_dataset_sigma_at_discordance_0.10": {
            "cifar100_test": {"unique_images": 3000, "power_for_2_points": 0.80, "minimum_detectable_points": 2.0},
            "imagenette": {"unique_images": 3000, "power_for_2_points": 0.80, "minimum_detectable_points": 2.0},
            "eurosat_sealed": {"unique_images": 1000, "power_for_2_points": 0.25, "minimum_detectable_points": 3.47, "label": "underpowered"},
        },
        "pooled_over_datasets_per_sigma_at_discordance_0.10": {"unique_images": 7000, "power_for_2_points": 0.997, "minimum_detectable_points": 1.30},
        "sensitivity_pooled": {"0.05": {"minimum_detectable_points": 0.92}, "0.20": {"power_for_2_points": 0.87, "minimum_detectable_points": 1.84}},
        "statement": "CIFAR-100 and Imagenette (3,000 unique images each) approximately meet the 2,993-image target of the five-contrast family at 10% discordance (power about 0.80); EuroSAT (1,000) does not (power about 0.25) and is reported as underpowered; the pooled family per sigma (7,000) is well powered (about 0.997). V1 said '3,000 meet 3,096', which was false for the six-contrast family it registered.",
        "observed_discordance_rule": "the observed paired discordance is reported next to every interval",
    }


# --------------------------------------------------------------------------- predictions


Q6_DIRECTION = {
    "positive": "the operator increased standard certified accuracy at r = sigma on fresh items (Holm-adjusted two-sided direction test).",
    "negative": "the operator decreased standard certified accuracy at r = sigma on fresh items (Holm-adjusted two-sided direction test).",
    "unresolved": "the direction of the effect was not resolved at the family-wise 5% level.",
}
Q6_MAGNITUDE = {
    "gain_at_least_sesoi": "the increase is at least the 2-point smallest effect of interest (Holm-adjusted one-sided test against +2 points).",
    "loss_at_least_sesoi": "the decrease is at least 2 points (Holm-adjusted one-sided test against -2 points).",
    "practically_equivalent": "the effect lies within +-2 points: practically equivalent to no correction at the registered smallest effect of interest (percentile-bootstrap two one-sided tests, Holm-adjusted).",
    "inconclusive": "the data neither establish an effect of at least 2 points nor exclude one.",
}


def q3_block() -> dict[str, Any]:
    steps = [f"{family}__step_{candidates_ext.fmt(step)}" for family in candidates_ext.STEP_FAMILIES
             for step in candidates_ext.LEGACY_STEPS + candidates_ext.NEW_STEPS]
    return {"id": "Q3", "question": "do text-side steps act only on draws the decision-change proposition allows, and how does the changed-draw fraction grow with the step",
            "contrasts": steps, "type": "per-draw conformance + step-response curve",
            "rule": "conforms iff, summed over every cell, every step bank has zero containment violations and zero loose-bound violations (worker counters, tolerance 1e-5); the changed-draw fraction and the loose flippable budget are reported per step whatever their shape",
            "not_a_rule": "V1 compared the item-level fraction of changed certified outcomes with the draw-level fraction of flippable draws; those quantities have different units and no theorem orders them, so the comparison is not computed"}


def q4_block() -> dict[str, Any]:
    return {"id": "Q4", "question": "is the projected-gap bank decision-invariant under fresh noise, draw by draw",
            "contrasts": [f"chowers_exact_projected_gap__coefficient_{candidates_ext.fmt(c)}" for c in candidates_ext.COEFFICIENTS],
            "type": "per-draw conformance",
            "rule": ("conforms iff every projected-gap bank has zero containment violations; the changed projected-gap draws are "
                     "reported with their tie budget (each had an identity winner-pair margin of at most the recorded max |n_j - n_k| "
                     "plus the tolerance); the count bounds the margin and does not identify a cause")}


def q5_block() -> dict[str, Any]:
    return {"id": "Q5", "question": "do the supervised class-specific and few-shot prompt controls improve standard CA beyond the identity, and by how much relative to the two-sided operator",
            "contrasts": list(candidates_ext.CONTROL_CONTRASTS), "type": "positive control",
            "rule": "per pooled family: Holm over the four controls (direction test); control minus gr_clip_style_two_sided__coefficient_1 reported with an unadjusted interval (descriptive)"}


def q2_block(scope: str) -> dict[str, Any]:
    return {"id": "Q2", "question": f"how much of the two-sided operator's effect is carried by its text half and by its image half ({scope})",
            "contrasts": {key: [pattern.format(c=candidates_ext.fmt(c)) for c in candidates_ext.COEFFICIENTS] for key, pattern in candidates_ext.DECOMPOSITION.items()},
            "type": "decomposition, reported whatever the signs",
            "rule": "per pooled family and coefficient: the three effects and the interaction (two-sided minus text-only minus image-only) with an unadjusted interval; no additivity is assumed",
            "outcome_template": "at c = X the two-sided operator changed standard CA by Y points; its text half by T and its image half by I (interaction Z)."}


def q1_sentences(label: str) -> dict[str, str]:
    lead = ("With fresh certification draws and a fresh execution pipeline on the same 1,650 images and the same legacy bank tensors, the "
            + label + " of the two-sided operator was {point:+.2f} points (unadjusted 95% interval {lower:+.2f} to {upper:+.2f}; "
            "saved-draw value {reference:+.2f}, fresh minus saved {difference:+.2f}; direction p = {p_direction:.3f}, "
            "shortfall p = {p_shortfall:.3f}): ")
    return {
        "reproduced": lead + ("positive, and not significantly below the saved-draw value (not finding a shortfall does not show "
                              "that the two values are equal)."),
        "reproduced_smaller": lead + "positive, but significantly below the saved-draw value (one-sided test at 5%).",
        "not_reproduced": lead + "not distinguishable from zero, and significantly below the saved-draw value.",
        "inconclusive": lead + "neither distinguishable from zero nor significantly below the saved-draw value.",
        "reversed": lead + "significantly negative.",
    }


Q1_VERDICT_MEANING = {
    "reproduced": ("the direction test found a positive gain and the shortfall test did not find the fresh gain below the saved "
                   "value; not finding a shortfall is not evidence that the two values are equal"),
    "reproduced_smaller": "the direction test found a positive gain and the shortfall test found it below the saved value",
    "not_reproduced": "the direction test did not separate the gain from zero and the shortfall test found it below the saved value",
    "inconclusive": "neither test rejected: the data neither separate the gain from zero nor place it below the saved value",
    "reversed": "the direction test found a negative gain",
}


def q1_block() -> dict[str, Any]:
    planning, planning_sha = q1_planning()
    references = planning["reference_points"]
    models = planning["fresh_draw_models"]
    predictive = models["posterior_predictive"]
    basis = (
        "planning file: under the posterior-predictive fresh-draw model the Monte-Carlo SD of the equal-cell gain is at most "
        f"{predictive['monte_carlo_sd_points']['sharp_upper_bound_any_coupling']['macro']:.2f} points (sharp bound over every "
        "coupling of the candidate and identity outcomes) against an item-bootstrap SE of "
        f"{planning['saved_draws_under_registered_q1_analysis']['macro']['bootstrap_se_points']:.2f}; with that SD the normal "
        f"approximation gives 'reproduced' with probability {predictive['sharp_bound_normal_verdict_probabilities']['macro']['reproduced']:.2f} "
        f"for the primary and {predictive['sharp_bound_normal_verdict_probabilities']['pooled']['reproduced']:.2f} for the secondary "
        f"estimand (simulated under an independent coupling: {predictive['verdict_probabilities']['macro']['reproduced']:.2f} and "
        f"{predictive['verdict_probabilities']['pooled']['reproduced']:.2f})"
    )
    return {
        "id": "Q1",
        "question": ("does the historical post-hoc two-sided gain that the saved EXP-017 draws gave on these 1,650 images "
                     "(equal-cell +2.97, item-pooled +1.03 points) persist when the certification draws and the execution pipeline "
                     "are refreshed on the same registered images and the same legacy bank tensors"),
        "contrast": "gr_clip_style_two_sided__coefficient_1",
        "reference_candidate": candidates_ext.IDENTITY_CANDIDATE_ID,
        "type": "fresh-draw, fresh-pipeline persistence check on the same images and bank tensors (not an item-level replication)",
        "primary_estimand": "equal_cell_macro: mean over the 12 cells of the cell-mean paired difference in standard certified accuracy at r = sigma = 0.25 (the manuscript's headline estimand)",
        "secondary_estimand": "item_pooled: paired difference over the 3,300 positions",
        "saved_draw_reference_points": {
            "equal_cell_macro": {"exact": references["equal_cell_macro"]["exact_points"], "value": references["equal_cell_macro"]["value"]},
            "item_pooled": {"exact": references["item_pooled"]["exact_points"], "value": references["item_pooled"]["value"]},
        },
        "reference_source": {
            "path": Q1_PLANNING_PATH, "sha256": planning_sha, "content_sha256": planning["content_sha256"],
            "derivation": "recomputed with common.certify.certify_from_counts from the EXP-017 sufficient statistics bound by bank_construction.exp016_sources.exp017_artifact_manifest_sha256 (the recomputed outcomes equal the saved ones); equal to the Gate-N2 point estimate",
        },
        "tests": {
            "family": "pooled__sigma0.25",
            "alpha": 0.05,
            "direction": "two-sided percentile-bootstrap p = 2 min(P*(d <= 0), P*(d >= 0)); positive or negative when p <= 0.05, else unresolved",
            "shortfall": "one-sided percentile-bootstrap p = P*(d >= reference) for H0: the fresh-draw gain is at least its saved-draw value; shortfall when p <= 0.05",
            "multiplicity": "none: Q1 is one registered contrast outside the Holm families; its two nulls (gain = 0; gain >= reference > 0) are disjoint, so at most one is true and testing each at 5% keeps the family-wise error at 5%",
            "bootstrap": "the registered Rao-Wu shared-image bootstrap of inference.uncertainty (same seed and replicates)",
            "implementation": "satml2027ext/analyze_ext.py::q1_classify",
        },
        "verdict_rules": {
            "reproduced": "direction positive and no shortfall",
            "reproduced_smaller": "direction positive and shortfall",
            "not_reproduced": "direction unresolved and shortfall",
            "inconclusive": "direction unresolved and no shortfall",
            "reversed": "direction negative",
        },
        "sentences": {"equal_cell_macro": q1_sentences("equal-cell gain"), "item_pooled": q1_sentences("item-pooled gain")},
        "sentence_fields": "point, interval, saved-draw reference, fresh minus saved, and both p-values (satml2027ext/analyze_ext.py::q1_sentence)",
        "verdict_meaning": Q1_VERDICT_MEANING,
        "scope_sentence": ("These are the images on which the contrast was selected post hoc, so no Q1 verdict is an item-level "
                           "replication; 021B (Q6) tests fresh images. The two banks of the contrast are saved EXP-017 legacy tensors "
                           "(021A's 18 new banks and trained controls are not part of Q1), so a difference "
                           "between the fresh and saved values comes from the certification draws or from the execution pipeline "
                           "(image encoding under the EXP-021 software stack and hardware), which 021A does not separate."),
        "rule": "the verdict is the primary estimand's; the secondary estimand's class and the dataset-specific values are reported beside it; Q1 is not evaluable if any of the twelve cells is not run; the contrast still appears in the secondary five-contrast family table",
        "registered_prediction": "reproduced",
        "prediction_basis": basis,
        "if_not_reproduced": "custody, the software stack and the pipeline are examined first (021A shares images and banks with EXP-017, so a large difference points to a numerical or pipeline difference before a scientific one); the registered sentence is reported regardless",
        "replaces": "V2's rule (Holm-5 direction test of the item-pooled value; 'otherwise' sentence asserting non-reproduction and a cause) failed the internal re-review (blocker B1)",
    }


def predictions_021a() -> list[dict[str, Any]]:
    return [
        q1_block(),
        q2_block("same items, fresh noise"),
        q3_block(),
        q4_block(),
        q5_block(),
    ]


def predictions_021b() -> list[dict[str, Any]]:
    return [
        q2_block("fresh items"),
        q3_block(),
        q4_block(),
        q5_block(),
        {"id": "Q6", "question": "on fresh items at sigma 0.12 and 0.25, does any shared operator change standard CA, and is any change at least the 2-point smallest effect of interest",
         "contrasts": list(candidates_ext.PRIMARY_SHARED_CONTRASTS), "type": "falsifiable, two orthogonal adjudications per contrast",
         "rule": "per family and contrast: direction class and magnitude class from their separate Holm procedures; the pooled family at each sigma adjudicates, dataset families are secondary",
         "direction_sentences": Q6_DIRECTION,
         "magnitude_sentences": Q6_MAGNITUDE},
        {"id": "Q7", "question": "does the negative result persist where the smoothed identity classifier is not collapsed",
         "type": "descriptive", "rule": "identity smoothed accuracy, top-class share, abstention and standard CA per cell reported beside Q6"},
        {"id": "Q8", "question": "are radii comparable to the literature",
         "type": "orientation only", "rule": "identity standard CA on Imagenette at r in {0.25, 0.5}, sigma 0.25, per backbone, reported beside published CLIP numbers; no claim"},
    ]


def predictions_021c() -> list[dict[str, Any]]:
    return [
        {"id": "Q9", "question": "for 100 fixed items per dataset and two backbones, how many certificates at r = sigma differ between the 4,096-draw prefix and all 100,000 draws of the same confirmation stream",
         "banks": list(candidates_ext.BUDGET_SUBSET_021C), "type": "descriptive",
         "rule": ("per bank and cell: certified-correct counts at r = sigma from the first 4,096 and from all 100,000 draws (exact "
                  "Clopper-Pearson), changed certificates split into gained and lost; the join with 021B reports two identities "
                  "separately: the confirmation prefix (items, confirmation seeds, first 4,096 counts) and the full certificate input "
                  "(labels, selection seeds, selection counts and selected classes, confirmation seeds, raw predictions)"),
         "outcome_template": "of 100 items, N certificates at r = sigma changed between 4,096 and 100,000 draws for bank B in cell C."},
    ]


# --------------------------------------------------------------------------- cells


def _split_hash(split: dict[str, Any]) -> str:
    return split["item_ids_sha256"] if split["status"] == "written" else PENDING


def _split_count(split: dict[str, Any]) -> int:
    return int(split["count"] if split["status"] == "written" else split["planned_count"])


def cells_021a(item_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    study = item_manifest["studies"][EXPERIMENT_IDS["021a"]]
    cells = []
    for model_id in MODELS_021A:
        for dataset_id, plan in PLAN_021A.items():
            entry = study["datasets"][dataset_id]
            for fold in PHASE3_FOLDS:
                roles = entry["folds"][fold]
                cells.append({
                    "cell_id": f"{model_id}__{dataset_id}__fold{fold}__sigma{SIGMA_021A:.12g}",
                    "model_id": model_id,
                    "dataset_id": dataset_id,
                    "fold": int(fold),
                    "sigma": SIGMA_021A,
                    "data_role": entry["data_role"],
                    "item_count": int(roles["evaluation"]["count"]),
                    "class_count": int(plan["class_count"]),
                    "evaluation_items_sha256": roles["evaluation"]["item_ids_sha256"],
                    "development_items_sha256": roles["development"]["item_ids_sha256"],
                    "development_item_count": int(roles["development"]["count"]),
                    "development_per_class": int(roles["development"]["per_class"]),
                    "consumes_sealed_split": False,
                    "store_identity_margins": True,
                    "scope_tier": "minimal",
                })
    cells.sort(key=lambda cell: cell["cell_id"])
    if len(cells) != 12:
        raise RuntimeError(f"021A must have 12 cells, got {len(cells)}")
    return cells


def cells_021b(item_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    study = item_manifest["studies"][EXPERIMENT_IDS["021b"]]
    cells = []
    for model_id in MODELS_021B:
        for dataset_id in DATASETS_021B:
            entry = study["datasets"][dataset_id]
            splits = entry["splits"]
            for sigma in SIGMAS_021B:
                cell = {
                    "cell_id": f"{model_id}__{dataset_id}__sigma{sigma:.12g}",
                    "model_id": model_id,
                    "dataset_id": dataset_id,
                    "sigma": float(sigma),
                    "data_role": entry["data_role"],
                    "evaluation_source": entry["evaluation_source"],
                    "item_count": _split_count(splits["evaluation"]),
                    "class_count": int(entry["class_count"]),
                    "evaluation_items_sha256": _split_hash(splits["evaluation"]),
                    "development_items_sha256": _split_hash(splits["development"]),
                    "development_item_count": _split_count(splits["development"]),
                    "control_train_items_sha256": _split_hash(splits["control_train"]),
                    "control_train_item_count": _split_count(splits["control_train"]),
                    "consumes_sealed_split": dataset_id in SEALED_DATASETS,
                    "store_identity_margins": False,
                    "scope_tier": "minimal" if sigma == 0.25 else "core",
                }
                if dataset_id == "imagenette":
                    cell["evaluation_split"] = "val"
                cells.append(cell)
    cells.sort(key=lambda cell: cell["cell_id"])
    if len(cells) != 18:
        raise RuntimeError(f"021B must have 18 cells, got {len(cells)}")
    return cells


def cells_021c(item_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    study = item_manifest["studies"][EXPERIMENT_IDS["021c"]]
    study_b = item_manifest["studies"][EXPERIMENT_IDS["021b"]]
    cells = []
    for model_id in MODELS_021A:
        for dataset_id in DATASETS_021B:
            entry = study["datasets"][dataset_id]
            parent = study_b["datasets"][dataset_id]
            written = entry["status"] == "written"
            cells.append({
                "cell_id": f"{model_id}__{dataset_id}__sigma{SIGMA_021C:.12g}__subset{SUBSET_021C}",
                "model_id": model_id,
                "dataset_id": dataset_id,
                "sigma": SIGMA_021C,
                "data_role": parent["data_role"],
                "item_count": int(entry["count"]) if written else SUBSET_021C,
                "class_count": int(parent["class_count"]),
                "evaluation_items_sha256": entry["item_ids_sha256"] if written else PENDING,
                "parent_evaluation_items_sha256": entry.get("parent_item_ids_sha256", PENDING),
                "development_items_sha256": _split_hash(parent["splits"]["development"]),
                "control_train_items_sha256": _split_hash(parent["splits"]["control_train"]),
                "parent_cell_id": f"{model_id}__{dataset_id}__sigma{SIGMA_021C:.12g}",
                "consumes_sealed_split": dataset_id in SEALED_DATASETS,
                "store_identity_margins": False,
                "scope_tier": "core",
            })
    cells.sort(key=lambda cell: cell["cell_id"])
    return cells


# --------------------------------------------------------------------------- finalization


def pending_paths(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            found.extend(pending_paths(child, f"{prefix}/{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(pending_paths(child, f"{prefix}[{index}]"))
    elif value == PENDING:
        found.append(prefix or "/")
    return found


def finalize_registration(registration: dict[str, Any]) -> dict[str, Any]:
    pending = pending_paths(registration)
    if pending:
        raise PendingItemListError(f"{registration['experiment_id']} has pending item lists: {pending}")
    if registration.get("status") == "draft" or registration.get("registration_sha256") is not None:
        raise RuntimeError("registration must be unhashed and not marked draft before finalization")
    final = dict(registration)
    final.pop("registration_sha256", None)
    final["status"] = "preregistered_pending_independent_approval"
    final["registration_sha256"] = canonical_json_hash(final)
    return final


def verify_registration_hash(registration: dict[str, Any]) -> bool:
    stored = registration.get("registration_sha256")
    if registration.get("status") == "draft" or not isinstance(stored, str):
        return False
    content = {key: value for key, value in registration.items() if key != "registration_sha256"}
    return canonical_json_hash(content) == stored


def draft_registration(registration: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    draft = dict(registration)
    draft["status"] = "draft"
    draft["registration_sha256"] = None
    draft["draft_reasons"] = reasons
    draft["draft_content_sha256"] = canonical_json_hash(draft)
    return draft


def write_registration(out: Path, registration: dict[str, Any]) -> tuple[Path, str]:
    stem = registration["experiment_id"].lower()
    path = write_json_atomic(out / f"{stem}.json", registration)
    if registration["status"] == "draft":
        line = f"DRAFT_NOT_FINALIZED PENDING_ITEM_LIST draft_content_sha256={registration['draft_content_sha256']}\n"
    else:
        line = registration["registration_sha256"] + "\n"
    (out / f"{stem}.sha256").write_bytes(line.encode("ascii"))
    return path, line.strip()


# --------------------------------------------------------------------------- registrations


def common_blocks(item_manifest: dict[str, Any], args: argparse.Namespace, registered_at: str, experiment_id: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "registration_version": REGISTRATION_VERSION,
        "registered_at": registered_at,
        "supersedes": supersedes_block(experiment_id),
        "method_selection": False,
        "models": MODEL_BINDINGS,
        "data_roots": DATA_ROOTS,
        "sharding": sharding_block(),
        "stages": stages_block(),
        "scope_ladder": scope_ladder_block(),
        "priority_if_short": ["021B", "021A", "021C"],
        "stop_rules": stop_rules(),
        "provenance": provenance_block(item_manifest, args),
        "hardware": hardware_block(),
    }


def build_registrations(item_manifest: dict[str, Any], args: argparse.Namespace, registered_at: str) -> list[dict[str, Any]]:
    id_a, id_b, id_c = EXPERIMENT_IDS["021a"], EXPERIMENT_IDS["021b"], EXPERIMENT_IDS["021c"]
    reg_a = dict(common_blocks(item_manifest, args, registered_at, id_a), **{
        "experiment_id": id_a,
        "title": "same-item re-certification of the 18 saved EXP-017 legacy bank tensors with fresh certification draws and a fresh execution pipeline, 18 new banks, and decomposition of the two-sided operator (not an independent replication)",
        "data_role": "Phase-3 calibration_validation items per fold (evaluation; already used by EXP-017) and the calibration_development items of the same fold (the EXP-016 development tensors); fresh evaluation noise; no new items",
        "sigmas": [SIGMA_021A],
        "candidates": candidates_ext.canonical_candidates_block(),
        "bank_construction": bank_construction_021a(),
        "cells": cells_021a(item_manifest),
        "certification": certification_block(CONFIRMATION_DRAWS, budget_checkpoints=[], store_margins=True),
        "prompt_configs": prompt_config_block(["cifar100", "eurosat"]),
        "access": access_block(False, []),
        "sealed_evaluation_access": False,
        "inference": inference_block_primary(family_power_021a()),
        "power": power_block_021a(),
        "primary_outcome": "Q1: equal-cell macro over the 12 cells of the paired change in standard certified accuracy at r = 0.25 of the two-sided c=1 bank vs no_correction under fresh draws, classified against zero and against its saved-draw EXP-017 value (+89/30 = +2.97 points); the item-pooled value (+34/33 = +1.03 points on the saved draws) is secondary",
        "predictions": predictions_021a(),
        "position_count": 3300,
        "unique_image_count": 1650,
        "compute_estimate_4090_hours": 24,
    })

    reg_b = dict(common_blocks(item_manifest, args, registered_at, id_b), **{
        "experiment_id": id_b,
        "title": "powered fresh-item study: three datasets, three backbones, sigma 0.12 and 0.25",
        "data_role": "fresh never-touched evaluation items (CIFAR-100 official test 3,000; EuroSAT sealed holdout 1,000; Imagenette2-320 validation 3,000); development and control-train from the Phase-3 reserve minus Phase-4 and EXP-019 items, or from the Imagenette training split",
        "sigmas": SIGMAS_021B,
        "candidates": candidates_ext.canonical_candidates_block(),
        "bank_construction": bank_construction_021b(args),
        "cells": cells_021b(item_manifest),
        "certification": certification_block(CONFIRMATION_DRAWS, budget_checkpoints=[], store_margins=False),
        "prompt_configs": prompt_config_block(DATASETS_021B),
        "access": access_block(True, ["cifar100_test", "eurosat_sealed"]),
        "sealed_evaluation_access": True,
        "consumed_once": ["cifar100 official test split", "eurosat final_test_sealed"],
        "inference": inference_block_primary(family_power_021b()),
        "power": power_block_021b(),
        "primary_outcome": "Q6: per pooled family (each sigma) and contrast, the direction and practical-magnitude classes of the item-pooled paired change in standard certified accuracy at r = sigma vs no_correction",
        "predictions": predictions_021b(),
        "position_count_per_backbone": 14000,
        "retired_scope": RETIRED_021B_LADDER,
        "compute_estimate_4090_hours": 105,
    })
    final_b = finalize_registration(reg_b) if not pending_paths(reg_b) else None

    reg_c = dict(common_blocks(item_manifest, args, registered_at, id_c), **{
        "experiment_id": id_c,
        "title": "budget subset: exact Clopper-Pearson certificates from the 4,096-draw prefix versus all 100,000 draws of the same confirmation stream, 100 fixed items per dataset",
        "data_role": f"first {SUBSET_021C} EXP-021B evaluation items per dataset in registered order; the parent 021B cell's approved bank payload",
        "sigmas": [SIGMA_021C],
        "candidates": candidates_ext.subset_candidates_block(
            candidates_ext.BUDGET_SUBSET_021C, parent_experiment_id=id_b,
            parent_registration_sha256=final_b["registration_sha256"] if final_b else PENDING),
        "bank_construction": bank_construction_021c(),
        "cells": cells_021c(item_manifest),
        "certification": certification_block(CONFIRMATION_DRAWS_021C, budget_checkpoints=BUDGET_CHECKPOINTS_021C, store_margins=False),
        "prompt_configs": prompt_config_block(DATASETS_021B),
        "access": access_block(True, ["cifar100_test", "eurosat_sealed"]),
        "sealed_evaluation_access": True,
        "inference": {
            "design": "descriptive budget comparison; no hypothesis test, no Holm, no equivalence",
            "estimand": "per bank and cell: certified-correct count at r = sigma from the 4,096-draw prefix and from all 100,000 draws of the same confirmation stream; changed certificates (gained, lost)",
            "prefix_identity_check": "the 4,096-draw prefix counts are joined with the 021B confirmation counts of the same items and banks (required; the analysis-stage approval binds the 021B custody manifest); with the full parent stack scored, the registered forward batch and the determinism settings, items, seeds and counts are expected to agree exactly on one GPU class; a disagreement is reported as a determinism finding, never hidden, and never stops the analysis",
            "analyzer": "satml2027ext/analyze_ext.py::analyze_budget",
        },
        "primary_outcome": "Q9 (descriptive)",
        "predictions": predictions_021c(),
        "inherits_from": id_b,
        "position_count": 600,
        "draws_per_position": SELECTION_DRAWS + CONFIRMATION_DRAWS_021C,
        "compute_estimate_4090_hours": 33,
    })
    return [reg_a, reg_b, reg_c]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", type=Path, default=PROJECT_ROOT / "results/satml2027ext/items")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "configs/satml2027ext")
    parser.add_argument("--registered-at", default=None, help="frozen UTC ISO timestamp; required for byte-identical regeneration")
    parser.add_argument("--split-manifest", type=Path, default=PROJECT_ROOT / "artifacts/day14/phase3_splits_v1/manifest.json")
    parser.add_argument("--exclusion-manifest", type=Path, default=PROJECT_ROOT / "artifacts/day14/phase3_splits_v1/exclusion_manifest.json")
    parser.add_argument("--allow-synthetic-manifest", action="store_true",
                        help="TESTS ONLY: finalize registrations from a manifest that used synthetic inputs")
    args = parser.parse_args(argv)

    item_manifest = read_json(args.items / "item_manifest_ext.json")
    if item_manifest.get("synthetic_inputs_allowed") and not args.allow_synthetic_manifest:
        raise SystemExit("item manifest was produced with --allow-synthetic; refusing to build registrations from it")
    registered_at = args.registered_at or datetime.now(timezone.utc).isoformat(timespec="seconds")

    written = []
    for registration in build_registrations(item_manifest, args, registered_at):
        try:
            final = finalize_registration(registration)
        except PendingItemListError as error:
            final = draft_registration(registration, [str(error)] + pending_paths(registration))
        path, line = write_registration(args.out, final)
        written.append((final["experiment_id"], final["status"], len(final["cells"]), line, path))
    for experiment_id, status, cell_count, line, path in written:
        print(f"{experiment_id:20s} {status:40s} cells={cell_count:3d}  {line}  {path}")
    print("\nDrafts cannot be approved; finalized hashes must be committed or timestamped BEFORE any sampling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
