"""EXP-021 fail-closed guards (V2), shared by preparation, sampling and analysis.

Nothing in ``satml2027ext`` builds a bank from registered data, samples, or
analyses a real result unless ``research/EXP021_APPROVAL.json`` authorizes that
exact stage.  V2 closes the gaps the V1 pre-sampling reviews found (P0-F):

* every stage flag (``preparation_authorized``, ``sampling_authorized``,
  ``analysis_authorized``) must be present and a JSON boolean; the requested
  stage must be ``true``; a missing flag is a refusal, never a pass;
* ``registrations`` must bind exactly the three EXP-021 experiment ids, and the
  registration being acted on must hash to its bound value;
* ``source_sha256`` must equal the complete dependency closure computed from
  disk (``dependency_closure``): the same set of files and the same SHA-256 for
  each; a missing, extra or changed file is a refusal (no silent skip);
* sampling additionally requires the approved data-preflight and bank-manifest
  hashes and the files themselves; analysis additionally requires the approved
  run-custody manifest of the experiment (021C: also its 021B parent's);
* V3 (internal re-review, 2026-09-26): approval files with a repeated JSON key
  are refused (``_common.read_json``); ``approved_on`` must be an ISO date
  (YYYY-MM-DD); a stage-1 approval that names a data preflight binds the
  preparation stage to that exact file.
* V4 (reviews of reviewer bundle V3, 2026-09-26): the approval names a git
  commit (``source_commit``, 40 hexadecimal digits) and every file of the
  approved dependency closure must be committed there with exactly the approved
  bytes (``verify_source_commit``: the git blob id of each file on disk must
  equal the blob of that path at the commit; the repository stores bytes
  unconverted, ``* -text``).  Every stage checks it, so the host must run from a
  git checkout that contains the commit.  Approval schema v3; registration
  schema v4 (V3 registrations are superseded and refused).

Planning (``plan_ext.py``) is deliberately *not* gated: it reads registrations
only, never touches data or outcomes, and a dry-run plan is needed before
approval.  Importing this module and unit-testing its functions authorizes nothing.

Approval layout (``satml2027ext.approval.v2``)::

    {
      "approval_schema": "satml2027ext.approval.v3",
      "status": "INDEPENDENTLY_APPROVED_FOR_EXP021",
      "registrations": {"EXP-20260921-021A": "<sha256>", "EXP-20260921-021B": "...", "EXP-20260921-021C": "..."},
      "preparation_authorized": true, "sampling_authorized": false, "analysis_authorized": false,
      "source_sha256": {"<project-relative path>": "<sha256>", ...},   (exactly dependency_closure())
      "source_commit": "<40-hex git commit holding exactly those files>",
      "data_preflight_sha256": null | "<sha256>",
      "bank_manifest_sha256": null | "<sha256>",
      "run_custody_manifest_sha256": null | {"EXP-20260921-021A": "<sha256>", ...},
      "approved_by": "<name>", "approved_on": "<ISO date>",
      "notes": "...", "verdicts": {...}                                  (optional)
    }
"""

from __future__ import annotations

import glob
import hashlib
import re
import subprocess
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from satml2027ext._common import (  # puts satml2027/ on sys.path
    DuplicateKeyError,
    PROJECT_ROOT,
    canonical_json_hash,
    read_json,
    sha256_file,
)
from satml2027ext import candidates_ext

REGISTRATION_SCHEMA = "satml2027ext.registration.v4"
APPROVAL_SCHEMA = "satml2027ext.approval.v3"
APPROVAL_RELATIVE_PATH = Path("research") / "EXP021_APPROVAL.json"
DEFAULT_APPROVAL_PATH = PROJECT_ROOT / APPROVAL_RELATIVE_PATH
REQUIRED_APPROVAL_STATUS = "INDEPENDENTLY_APPROVED_FOR_EXP021"
DEFAULT_IDENTITY_CANDIDATE_ID = candidates_ext.IDENTITY_CANDIDATE_ID
STAGES = ("preparation", "sampling", "analysis")
EXPERIMENT_IDS = ("EXP-20260921-021A", "EXP-20260921-021B", "EXP-20260921-021C")
REQUIRED_APPROVAL_KEYS = (
    "approval_schema", "status", "registrations", "preparation_authorized", "sampling_authorized",
    "analysis_authorized", "source_sha256", "source_commit", "data_preflight_sha256", "bank_manifest_sha256",
    "run_custody_manifest_sha256", "approved_by", "approved_on",
)
OPTIONAL_APPROVAL_KEYS = ("notes", "verdicts", "package", "instructions")

# The files every EXP-021 stage executes or reads as registered inputs.  Tests are
# excluded (they never run on a sampling host); third-party packages are bound by
# the data preflight's environment record instead.
DEPENDENCY_PATTERNS: tuple[str, ...] = (
    "satml2027ext/*.py",
    "satml2027/common/*.py",
    "satml2027/day1/__init__.py",
    "satml2027/day1/d1_02_build_development.py",
    "satml2027/day1/d1_03b_train_controls.py",
    "interventions/*.py",
    "certification/*.py",
    "configs/prompts/cifar100_openai_readme.json",
    "configs/prompts/eurosat_openai_ensemble_v1.json",
    "configs/satml2027ext/prompts/*.json",
    "configs/satml2027ext/exp-20260921-021a.json",
    "configs/satml2027ext/exp-20260921-021b.json",
    "configs/satml2027ext/exp-20260921-021c.json",
    "results/satml2027ext/items/*.csv",
    "results/satml2027ext/items/item_manifest_ext.json",
    "results/satml2027ext/planning/*.json",
)
_ISO_DATE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")

_HEX = frozenset("0123456789abcdef")


class ApprovalError(RuntimeError):
    """Raised when the EXP-021 approval file does not authorize the requested stage."""


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


# --------------------------------------------------------------------------- registrations


def verify_registration_hash(registration: Mapping[str, Any]) -> str:
    """Recompute the canonical registration hash exactly as the approved worker does."""

    recorded = registration.get("registration_sha256")
    body = {key: value for key, value in registration.items() if key != "registration_sha256"}
    recomputed = canonical_json_hash(body)
    if not isinstance(recorded, str) or recorded != recomputed:
        raise RuntimeError(f"registration hash mismatch: file says {recorded}, content hashes to {recomputed}")
    return recorded


def load_registration(path: str | Path) -> dict[str, Any]:
    """Load a V4 (``satml2027ext.registration.v4``) registration and verify its self-hash."""

    registration = read_json(path)
    if not isinstance(registration, dict):
        raise RuntimeError("registration is not a JSON object")
    if registration.get("schema_version") != REGISTRATION_SCHEMA:
        raise RuntimeError(
            f"registration schema {registration.get('schema_version')!r} is not {REGISTRATION_SCHEMA} "
            "(V1-V3 registrations were superseded before any approval and cannot be executed)"
        )
    verify_registration_hash(registration)
    if registration.get("status") != "preregistered_pending_independent_approval":
        raise RuntimeError(f"registration status {registration.get('status')!r} is not executable")
    if not isinstance(registration.get("experiment_id"), str) or registration["experiment_id"] not in EXPERIMENT_IDS:
        raise RuntimeError("registration lacks a registered EXP-021 experiment_id")
    if not isinstance(registration.get("cells"), list) or not registration["cells"]:
        raise RuntimeError("registration lists no cells")
    candidates_ext.candidate_specs(registration)  # the candidate block must parse (V1 finding P0-A)
    return registration


load_registration_v2 = load_registration  # name used by V1 callers


def identity_candidate_id(registration: Mapping[str, Any]) -> str:
    candidates = registration.get("candidates")
    if isinstance(candidates, Mapping):
        value = candidates.get("identity_candidate_id")
        if isinstance(value, str) and value:
            return value
    return DEFAULT_IDENTITY_CANDIDATE_ID


def registered_candidate_ids(registration: Mapping[str, Any], cell: Mapping[str, Any] | None = None) -> list[str]:
    """Registered candidate ids in registered order (torch-free; from ``candidates_ext``)."""

    return candidates_ext.registered_candidate_ids(registration, cell)


def certification_parameters(registration: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten the ``certification`` block."""

    certification = registration.get("certification")
    if not isinstance(certification, Mapping):
        raise RuntimeError("registration lacks a certification block")
    seeds = certification.get("seeds") if isinstance(certification.get("seeds"), Mapping) else {}
    streams = certification.get("streams") if isinstance(certification.get("streams"), Mapping) else {}
    selection_seed = seeds.get("selection_base_seed")
    confirmation_seed = seeds.get("confirmation_base_seed")
    if not isinstance(selection_seed, int) or not isinstance(confirmation_seed, int):
        raise RuntimeError("registration certification block lacks the selection/confirmation base seeds")
    budget = certification.get("budget_checkpoints", [])
    if not isinstance(budget, list) or any(not isinstance(value, int) or value <= 0 for value in budget):
        raise RuntimeError("budget_checkpoints must be a list of positive integers")
    flip = certification.get("flip_counters") if isinstance(certification.get("flip_counters"), Mapping) else {}
    parameters = {
        "selection_draws": int(certification["selection_draws"]),
        "confirmation_draws": int(certification["confirmation_draws"]),
        "alpha_per_example": float(certification["alpha_per_example"]),
        "reported_radii": [float(value) for value in certification.get("reported_radii", [])],
        "selection_base_seed": int(selection_seed),
        "confirmation_base_seed": int(confirmation_seed),
        "selection_stream_role": str(streams.get("selection", "")),
        "confirmation_stream_role": str(streams.get("confirmation", "")),
        "noise_block_draws": int(certification.get("noise_block_draws", 64)),
        "margin_storage": certification.get("margin_storage"),
        "budget_checkpoints": sorted(int(value) for value in budget),
        "flip_tolerance": float(flip.get("tolerance", 1e-5)),
        "blocks_per_forward": certification.get("blocks_per_forward"),
    }
    if parameters["blocks_per_forward"] is not None and (isinstance(parameters["blocks_per_forward"], bool)
                                                         or not isinstance(parameters["blocks_per_forward"], int)
                                                         or parameters["blocks_per_forward"] <= 0):
        raise RuntimeError("certification.blocks_per_forward must be a positive integer when registered")
    if parameters["selection_base_seed"] == parameters["confirmation_base_seed"]:
        raise RuntimeError("selection and confirmation base seeds coincide")
    if any(value >= parameters["confirmation_draws"] for value in parameters["budget_checkpoints"]):
        raise RuntimeError("every budget checkpoint must be smaller than the confirmation budget")
    return parameters


def margins_registered(registration: Mapping[str, Any]) -> bool:
    """True when the registration asks for the identity bank's per-draw confirmation margins."""

    storage = certification_parameters(registration)["margin_storage"]
    return isinstance(storage, Mapping) and storage.get("bank", identity_candidate_id(registration)) == identity_candidate_id(registration)


def item_list_stems(registration: Mapping[str, Any], cell: Mapping[str, Any], role: str) -> list[str]:
    """Candidate CSV stems for a cell's item list, most specific first.

    ``draw_items_ext.py`` writes ``exp021a__<dataset>__fold<f>__<role>.csv``,
    ``exp021b__<dataset>__<role>.csv`` and ``exp021c__<dataset>__subset.csv``;
    a cell may also name its list explicitly (``<role>_item_list``).
    """

    stems: list[str] = []
    explicit = cell.get(f"{role}_item_list")
    if isinstance(explicit, str) and explicit:
        stems.append(explicit.removesuffix(".csv"))
    dataset_id = str(cell["dataset_id"])
    experiment = str(registration.get("experiment_id", "")).lower()
    tag = None
    for token in ("021a", "021b", "021c"):
        if token in experiment:
            tag = f"exp{token}"
    if tag is not None:
        if cell.get("fold") is not None:
            stems.append(f"{tag}__{dataset_id}__fold{int(cell['fold'])}__{role}")
        if tag == "exp021c" and role == "evaluation":
            stems.append(f"{tag}__{dataset_id}__subset")
        stems.append(f"{tag}__{dataset_id}__{role}")
    return list(dict.fromkeys(stems))


def find_cell(registration: Mapping[str, Any], cell_id: str) -> dict[str, Any]:
    matches = [cell for cell in registration["cells"] if cell.get("cell_id") == cell_id]
    if len(matches) != 1:
        raise RuntimeError(f"cell {cell_id!r} occurs {len(matches)} times in the registration")
    return dict(matches[0])


# --------------------------------------------------------------------------- dependency closure


def dependency_closure(root: str | Path = PROJECT_ROOT) -> dict[str, str]:
    """``{project-relative posix path: sha256}`` of every file in ``DEPENDENCY_PATTERNS``.

    A pattern that matches nothing is an error (a missing dependency must never
    shrink the closure silently).
    """

    base = Path(root)
    closure: dict[str, str] = {}
    for pattern in DEPENDENCY_PATTERNS:
        matches = sorted(Path(path) for path in glob.glob(str(base / pattern)))
        files = [path for path in matches if path.is_file() and "__pycache__" not in path.parts]
        if not files:
            raise RuntimeError(f"dependency pattern {pattern!r} matches no file under {base}")
        for path in files:
            closure[path.relative_to(base).as_posix()] = sha256_file(path)
    return dict(sorted(closure.items()))


def verify_dependency_map(recorded: Any, root: str | Path = PROJECT_ROOT) -> dict[str, str]:
    """Exact comparison of an approved ``source_sha256`` map with the closure on disk."""

    if not isinstance(recorded, Mapping) or not recorded:
        raise ApprovalError("approval source_sha256 must be a nonempty mapping")
    current = dependency_closure(root)
    missing = sorted(set(current) - set(recorded))
    extra = sorted(set(recorded) - set(current))
    changed = sorted(key for key in set(current) & set(recorded) if recorded[key] != current[key])
    if missing or extra or changed:
        raise ApprovalError(
            "approved dependency map differs from the files on disk: "
            f"unapproved={missing[:5]}{'...' if len(missing) > 5 else ''} ({len(missing)}), "
            f"absent={extra[:5]}{'...' if len(extra) > 5 else ''} ({len(extra)}), "
            f"changed={changed[:5]}{'...' if len(changed) > 5 else ''} ({len(changed)})"
        )
    return current


# --------------------------------------------------------------------------- source commit


def git_blob_id(path: str | Path) -> str:
    """The git blob id (SHA-1 of ``blob <size>\\0<bytes>``) of a file's bytes as stored on disk."""

    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def _git(root: str | Path, *arguments: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, check=False)
    except OSError as error:
        raise ApprovalError(f"git is not available to verify the approved source commit ({error})") from error


def verify_source_commit(commit: Any, closure: Mapping[str, str], root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    """Every approved dependency is committed at ``commit`` with exactly its bytes on disk (V4, text review item 22)."""

    if not isinstance(commit, str) or not _GIT_COMMIT.match(commit):
        raise ApprovalError("approval source_commit must be a 40-digit lowercase hexadecimal git commit id")
    kind = _git(root, "cat-file", "-t", commit)
    if kind.returncode != 0 or kind.stdout.decode("ascii", "replace").strip() != "commit":
        raise ApprovalError(f"source_commit {commit} is not a commit of the repository at {root}")
    paths = sorted(closure)
    listing = _git(root, "ls-tree", "-r", "--full-tree", "-z", commit, "--", *paths)
    if listing.returncode != 0:
        raise ApprovalError(f"git cannot list source_commit {commit}: {listing.stderr.decode('utf-8', 'replace').strip()}")
    committed: dict[str, str] = {}
    for record in listing.stdout.split(b"\0"):
        if not record:
            continue
        meta, _, name = record.partition(b"\t")
        fields = meta.decode("ascii").split()
        if len(fields) == 3 and fields[1] == "blob":
            committed[name.decode("utf-8")] = fields[2]
    base = Path(root)
    missing = [value for value in paths if value not in committed]
    different = [value for value in paths if value in committed and committed[value] != git_blob_id(base / value)]
    if missing or different:
        raise ApprovalError(
            f"the approved dependency closure is not committed at {commit}: "
            f"not committed={missing[:5]}{'...' if len(missing) > 5 else ''} ({len(missing)}), "
            f"different bytes={different[:5]}{'...' if len(different) > 5 else ''} ({len(different)})"
        )
    return {"source_commit": commit, "files": len(paths)}


# --------------------------------------------------------------------------- approval


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ApprovalError(message)


def verify_approval(
    registration: Mapping[str, Any],
    *,
    stage: str,
    approval_path: str | Path | None = None,
    root: str | Path = PROJECT_ROOT,
    stage_files: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Fail closed unless the approval authorizes ``stage`` for this registration.

    ``stage_files`` supplies the paths of the stage artifacts whose hashes the
    approval binds: ``data_preflight`` and ``bank_manifest`` for sampling and
    analysis, and ``run_custody_manifest`` for analysis.
    """

    _require(stage in STAGES, f"unknown stage {stage!r}; stages are {STAGES}")
    path = Path(approval_path) if approval_path is not None else DEFAULT_APPROVAL_PATH
    _require(path.is_file(), f"EXP-021 approval file is absent ({path}); nothing is authorized")
    try:
        approval = read_json(path)
    except DuplicateKeyError as error:
        raise ApprovalError(f"EXP-021 approval file repeats a JSON key ({error}); refused") from error
    _require(isinstance(approval, Mapping), "EXP-021 approval file is not a JSON object")
    missing = [key for key in REQUIRED_APPROVAL_KEYS if key not in approval]
    _require(not missing, f"approval lacks required keys {missing}")
    unknown = sorted(set(approval) - set(REQUIRED_APPROVAL_KEYS) - set(OPTIONAL_APPROVAL_KEYS))
    _require(not unknown, f"approval carries unknown keys {unknown}")
    _require(approval["approval_schema"] == APPROVAL_SCHEMA, f"approval schema is not {APPROVAL_SCHEMA}")
    _require(approval["status"] == REQUIRED_APPROVAL_STATUS,
             f"EXP-021 approval status {approval['status']!r} is not {REQUIRED_APPROVAL_STATUS!r}")
    for flag in ("preparation_authorized", "sampling_authorized", "analysis_authorized"):
        _require(type(approval[flag]) is bool, f"approval flag {flag} must be a JSON boolean")
    _require(approval[f"{stage}_authorized"] is True, f"approval does not authorize the {stage} stage")
    for key in ("approved_by", "approved_on"):
        _require(isinstance(approval[key], str) and approval[key].strip() != "", f"approval {key} must be a nonempty string")
    _require(bool(_ISO_DATE.match(approval["approved_on"])), "approval approved_on must be an ISO date YYYY-MM-DD")
    try:
        date.fromisoformat(approval["approved_on"])
    except ValueError:
        _require(False, "approval approved_on is not a valid calendar date")

    bound = approval["registrations"]
    _require(isinstance(bound, Mapping) and set(bound) == set(EXPERIMENT_IDS),
             f"approval must bind exactly the registrations {EXPERIMENT_IDS}")
    _require(all(is_sha256(value) for value in bound.values()), "approved registration hashes must be SHA-256 strings")
    recorded_hash = verify_registration_hash(registration)
    experiment_id = str(registration.get("experiment_id"))
    _require(bound.get(experiment_id) == recorded_hash,
             f"approval binds {experiment_id} to {bound.get(experiment_id)}, but the registration hashes to {recorded_hash}")

    closure = verify_dependency_map(approval["source_sha256"], root)
    verify_source_commit(approval["source_commit"], closure, root)

    files = {key: Path(value) for key, value in (stage_files or {}).items()}
    if stage == "preparation" and approval["data_preflight_sha256"] is not None:
        approved = approval["data_preflight_sha256"]
        _require(is_sha256(approved), "the approved data_preflight_sha256 must be null or a SHA-256 string")
        _require("data_preflight" in files and files["data_preflight"].is_file(),
                 "this approval binds a data preflight; the preparation stage needs that file")
        _require(sha256_file(files["data_preflight"]) == approved, "data_preflight file differs from the approved hash")
    if stage in ("sampling", "analysis"):
        for key in ("data_preflight", "bank_manifest"):
            approved = approval[f"{key}_sha256"]
            _require(is_sha256(approved), f"the {stage} stage requires an approved {key}_sha256")
            _require(key in files and files[key].is_file(), f"the {stage} stage needs the {key} file")
            _require(sha256_file(files[key]) == approved, f"{key} file differs from the approved hash")
    if stage == "analysis":
        custody = approval["run_custody_manifest_sha256"]
        _require(isinstance(custody, Mapping) and is_sha256(custody.get(experiment_id)),
                 f"the analysis stage requires an approved run custody manifest for {experiment_id}")
        _require("run_custody_manifest" in files and files["run_custody_manifest"].is_file(),
                 "the analysis stage needs the run custody manifest file")
        _require(sha256_file(files["run_custody_manifest"]) == custody[experiment_id],
                 "run custody manifest differs from the approved hash")
        parent = candidates_ext.subset_parent(registration)
        if parent is not None:
            parent_id = str(parent["experiment_id"])
            _require(is_sha256(custody.get(parent_id)),
                     f"the 021C analysis requires the approved run custody manifest of its parent {parent_id}")
            _require("parent_run_custody_manifest" in files and files["parent_run_custody_manifest"].is_file(),
                     "the 021C analysis needs the parent's run custody manifest file")
            _require(sha256_file(files["parent_run_custody_manifest"]) == custody[parent_id],
                     "the parent run custody manifest differs from the approved hash")
    output = dict(approval)
    output["_approval_sha256"] = sha256_file(path)
    output["_approval_path"] = str(path)
    return output


def approval_template(registrations: Mapping[str, str], *, root: str | Path = PROJECT_ROOT, package: str = "") -> dict[str, Any]:
    """A template that authorizes nothing (status is not the required value; every flag false)."""

    return {
        "approval_schema": APPROVAL_SCHEMA,
        "status": "TEMPLATE_NOT_AN_APPROVAL",
        "registrations": {key: registrations[key] for key in EXPERIMENT_IDS},
        "preparation_authorized": False,
        "sampling_authorized": False,
        "analysis_authorized": False,
        "source_sha256": dependency_closure(root),
        "source_commit": "",
        "data_preflight_sha256": None,
        "bank_manifest_sha256": None,
        "run_custody_manifest_sha256": None,
        "approved_by": "",
        "approved_on": "",
        "package": package,
        "instructions": (
            f"Stage 1 (after this package passes review): set status to {REQUIRED_APPROVAL_STATUS}, "
            "preparation_authorized to true, fill approved_by/approved_on and source_commit (the 40-digit git commit that "
            "holds exactly the files of source_sha256; every stage verifies it), keep every hash. "
            "Stage 2 (after reviewing EXP021_DATA_PREFLIGHT.json and EXP021_BANK_MANIFEST.json): add their SHA-256 "
            "and set sampling_authorized to true. Stage 3 (after reviewing the run custody manifests): add them and set "
            "analysis_authorized to true. Any code or input change invalidates the source map and requires a new approval."
        ),
    }


def project_relative(path: str | Path) -> str:
    """Posix path of ``path`` relative to the project root."""

    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).name
