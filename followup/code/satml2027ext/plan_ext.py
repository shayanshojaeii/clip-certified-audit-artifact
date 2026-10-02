#!/usr/bin/env python3
"""EXP-021 compute plan: registered shards balanced across GPUs, longest first.

Inputs are v2 registrations and a device list.  Two device syntaxes are
accepted:

  ``5090:6``          one rented host with six RTX 5090 (CUDA_VISIBLE_DEVICES 0-5)
  ``A:1:4090 B:5:5090``   named hosts, ``NAME:GPU_COUNT:GPU_CLASS``

Throughput comes from the plan's Section 2.5 calibration (seconds per position
at 4,224 draws on a 4090: ViT-L/14 24.2; B/16 4.9; RN50 2.5; B/32 1.2; LAION
B/32 1.1) with an explicit 5090 row at ``--rate-factor-5090`` (default 1.4x the
4090 rate).  A ``--preflight`` directory of ``preflight_*.json`` reports
(``{"throughput": {gpu_class: {model_id: encodes_per_second}}}``) overrides the
assumed rows.

Assignment (plan Section 2.5, coordinator update 2026-09-25; V3 wording): every
cell is split into contiguous shards of at most ``--max-shard-items`` items (the
default equals the registered ``sharding.max_items_per_shard`` of 1,000, and a
larger value is refused); jobs are ordered longest first and each goes to the
GPU whose projected finish is earliest; ViT-L/14 shards go preferentially to
5090s when several GPU classes exist; every shard of a cell stays on the same
GPU class and carries a fixed ``(shard_index, shard_count)`` so a resume is
unambiguous.  The forward batch is the registered
``certification.blocks_per_forward`` (a different ``--blocks-per-forward`` is
refused here and by the worker); ``--cache-dir`` is passed to every job; the
scripts export ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` with the TF32 overrides.

Outputs (never overwritten): ``plan.json`` with its own SHA-256, one sequential
``run_gpu<i>.sh`` per GPU (``run_<host>_gpu<i>.sh`` for named hosts), a
single-host ``launch_all.sh`` (nohup/setsid, per-GPU logs and pid files) and a
``status.sh`` that reads only logs and shard metadata sidecars.

Usage:
  python satml2027ext/plan_ext.py --registrations configs/satml2027ext/*.json \
      --devices 5090:6 --out results/satml2027ext/plan
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from satml2027ext._common import PROJECT_ROOT, canonical_json_hash, read_json, write_json_atomic  # noqa: E402
from satml2027ext.guard_ext import certification_parameters, load_registration_v2, margins_registered  # noqa: E402

PLAN_SCHEMA = "satml2027ext.plan.v1"
CALIBRATION_DRAWS = 4224
REFERENCE_GPU_CLASS = "4090"
SECONDS_PER_POSITION_4090: dict[str, float] = {
    "openai-clip-vit-l14-quickgelu": 24.2,
    "openai-clip-vit-b16-quickgelu": 4.9,
    "openai-clip-rn50-quickgelu": 2.5,
    "openai-clip-vit-b32-quickgelu": 1.2,
    "openclip-vit-b32-laion2b": 1.1,
}
GPU_CLASS_RELATIVE_RATE: dict[str, float] = {"4090": 1.0, "5090": 1.4, "3090": 0.45}
GPU_CLASS_MODEL: dict[str, str] = {
    "5090": "NVIDIA GeForce RTX 5090 (32 GB)",
    "4090": "NVIDIA GeForce RTX 4090 (24 GB)",
    "3090": "NVIDIA GeForce RTX 3090 (24 GB)",
}
PREFERRED_CLASS_BY_MODEL: dict[str, str] = {"openai-clip-vit-l14-quickgelu": "5090"}
BALANCE_TARGET_RATIO = 1.15
SINGLE_HOST_NAME = "host"


@dataclass(frozen=True)
class Device:
    server: str
    gpu_index: int
    gpu_class: str

    @property
    def key(self) -> str:
        return f"{self.server}:{self.gpu_index}"


@dataclass
class Job:
    experiment_id: str
    registration_path: str
    registration_sha256: str
    cell_id: str
    model_id: str
    dataset_id: str
    sigma: float
    shard_index: int
    shard_count: int
    item_count: int
    draws_per_item: int
    encodes: int
    store_margins: bool
    device_key: str = ""
    gpu_class: str = ""
    estimated_seconds: float = 0.0
    extra_args: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- inputs


def parse_devices(specs: Sequence[str]) -> list[Device]:
    devices: list[Device] = []
    for spec in specs:
        parts = spec.split(":")
        if len(parts) == 2:
            gpu_class, count = parts[0].strip(), parts[1].strip()
            server = SINGLE_HOST_NAME
        elif len(parts) == 3:
            server, count, gpu_class = parts[0].strip(), parts[1].strip(), parts[2].strip()
        else:
            raise ValueError(f"device spec {spec!r} is neither CLASS:COUNT nor NAME:COUNT:CLASS")
        if not gpu_class or not server or not count.isdigit() or int(count) <= 0:
            raise ValueError(f"device spec {spec!r} has an empty field or a non-positive GPU count")
        start = sum(1 for device in devices if device.server == server)
        devices.extend(Device(server=server, gpu_index=start + index, gpu_class=gpu_class) for index in range(int(count)))
    if not devices:
        raise ValueError("at least one device is required")
    if len({device.key for device in devices}) != len(devices):
        raise ValueError("duplicate device keys")
    return devices


def default_throughput_table(rate_factor_5090: float = GPU_CLASS_RELATIVE_RATE["5090"]) -> dict[str, dict[str, float]]:
    """Encodes per second per (GPU class, model) from the Section 2.5 calibration."""

    if not math.isfinite(rate_factor_5090) or rate_factor_5090 <= 0:
        raise ValueError("the 5090 rate factor must be positive")
    factors = dict(GPU_CLASS_RELATIVE_RATE)
    factors["5090"] = float(rate_factor_5090)
    table: dict[str, dict[str, float]] = {}
    for gpu_class, factor in factors.items():
        table[gpu_class] = {
            model_id: round(CALIBRATION_DRAWS / seconds * factor, 4)
            for model_id, seconds in SECONDS_PER_POSITION_4090.items()
        }
    return table


def load_preflight_throughput(preflight_dir: Path | None) -> dict[str, dict[str, float]]:
    overrides: dict[str, dict[str, float]] = {}
    if preflight_dir is None:
        return overrides
    for path in sorted(Path(preflight_dir).glob("preflight_*.json")):
        report = read_json(path)
        for gpu_class, table in (report.get("throughput") or {}).items():
            for model_id, value in table.items():
                if value:
                    overrides.setdefault(str(gpu_class), {})[str(model_id)] = float(value)
    return overrides


def merge_throughput(base: Mapping[str, Mapping[str, float]], overrides: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    table = {gpu_class: dict(models) for gpu_class, models in base.items()}
    for gpu_class, models in overrides.items():
        table.setdefault(gpu_class, {}).update({model_id: float(value) for model_id, value in models.items()})
    return table


def rate(table: Mapping[str, Mapping[str, float]], gpu_class: str, model_id: str) -> float:
    try:
        value = float(table[gpu_class][model_id])
    except KeyError as error:
        raise KeyError(f"no throughput entry for GPU class {gpu_class!r} and model {model_id!r}") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"throughput for {gpu_class}/{model_id} must be positive")
    return value


def registered_shard_limit(registration: Mapping[str, Any]) -> int:
    sharding = registration.get("sharding")
    if not isinstance(sharding, Mapping) or not isinstance(sharding.get("max_items_per_shard"), int):
        raise ValueError("registration lacks sharding.max_items_per_shard")
    return int(sharding["max_items_per_shard"])


def cells_from_registration(registration: Mapping[str, Any], path: Path, *, store_margins: bool = True,
                            parent_path: Path | None = None) -> list[dict[str, Any]]:
    certification = certification_parameters(registration)
    draws = certification["selection_draws"] + certification["confirmation_draws"]
    subset = "subset_of" in (registration.get("candidates") or {})
    if subset and parent_path is None:
        raise ValueError(f"{registration['experiment_id']} scores its parent's banks; pass the parent registration too")
    cells = []
    for cell in registration["cells"]:
        cells.append({
            **cell,
            "experiment_id": registration["experiment_id"],
            "registration_path": Path(path).as_posix(),
            "registration_sha256": registration["registration_sha256"],
            "draws_per_item": draws,
            "store_margins": bool(store_margins and cell.get("store_identity_margins", False)),
            "parent_registration_path": Path(parent_path).as_posix() if subset else None,
            "shard_limit": registered_shard_limit(registration),
        })
    return cells


# --------------------------------------------------------------------------- jobs and assignment


def expand_jobs(cells: Sequence[Mapping[str, Any]], *, max_shard_items: int = 0) -> list[Job]:
    """One job per cell, or ``ceil(items / max_shard_items)`` shards when splitting is enabled."""

    jobs: list[Job] = []
    seen: set[str] = set()
    for cell in cells:
        cell_id = str(cell["cell_id"])
        if cell_id in seen:
            raise ValueError(f"cell {cell_id} is registered twice")
        seen.add(cell_id)
        items = int(cell["item_count"])
        draws = int(cell["draws_per_item"])
        limit = int(cell.get("shard_limit") or 0)
        if limit and (max_shard_items <= 0 or max_shard_items > limit):
            raise ValueError(f"shards of {cell_id} must hold at most the registered {limit} items (--max-shard-items)")
        shard_count = 1 if max_shard_items <= 0 else max(1, math.ceil(items / max_shard_items))
        base, extra = divmod(items, shard_count)
        for shard_index in range(shard_count):
            shard_items = base + (1 if shard_index < extra else 0)
            if shard_items == 0:
                continue
            extra_args = []
            if cell.get("dataset_id") == "eurosat_sealed" and cell.get("phase3_assignments"):
                extra_args += ["--phase3-assignments", str(cell["phase3_assignments"])]
            if cell.get("parent_registration_path"):
                extra_args += ["--parent-registration", str(cell["parent_registration_path"])]
            jobs.append(Job(
                experiment_id=str(cell["experiment_id"]), registration_path=str(cell["registration_path"]),
                registration_sha256=str(cell["registration_sha256"]), cell_id=cell_id,
                model_id=str(cell["model_id"]), dataset_id=str(cell["dataset_id"]), sigma=float(cell["sigma"]),
                shard_index=shard_index, shard_count=shard_count, item_count=shard_items, draws_per_item=draws,
                encodes=shard_items * draws, store_margins=bool(cell.get("store_margins", True)),
                extra_args=extra_args,
            ))
    return jobs


def assign_jobs(
    jobs: Sequence[Job], devices: Sequence[Device], table: Mapping[str, Mapping[str, float]], *,
    preferred_class_by_model: Mapping[str, str] = PREFERRED_CLASS_BY_MODEL,
) -> list[Job]:
    """Longest-first greedy to the earliest-finishing eligible GPU; split cells keep one GPU class."""

    if not devices:
        raise ValueError("no devices to plan on")
    classes = {device.gpu_class for device in devices}
    for job in jobs:
        for gpu_class in classes:
            rate(table, gpu_class, job.model_id)  # fail closed on an unknown (class, model)

    def fastest_seconds(job: Job) -> float:
        return min(job.encodes / rate(table, gpu_class, job.model_id) for gpu_class in classes)

    ordered = sorted(jobs, key=lambda job: (-fastest_seconds(job), job.cell_id, job.shard_index))
    loads: dict[str, float] = {device.key: 0.0 for device in devices}
    class_of_cell: dict[str, str] = {}
    assigned: list[Job] = []
    for job in ordered:
        eligible = list(devices)
        if job.cell_id in class_of_cell:
            eligible = [device for device in eligible if device.gpu_class == class_of_cell[job.cell_id]]
        else:
            preferred = preferred_class_by_model.get(job.model_id)
            if preferred is not None and len(classes) > 1:
                preferred_devices = [device for device in eligible if device.gpu_class == preferred]
                if preferred_devices:
                    eligible = preferred_devices
        best = min(
            eligible,
            key=lambda device: (loads[device.key] + job.encodes / rate(table, device.gpu_class, job.model_id),
                                device.server, device.gpu_index),
        )
        seconds = job.encodes / rate(table, best.gpu_class, job.model_id)
        placed = Job(**{**asdict(job), "device_key": best.key, "gpu_class": best.gpu_class, "estimated_seconds": seconds})
        loads[best.key] += seconds
        class_of_cell[job.cell_id] = best.gpu_class
        assigned.append(placed)
    return assigned


def summarize(jobs: Sequence[Job], devices: Sequence[Device]) -> dict[str, Any]:
    per_device = {device.key: 0.0 for device in devices}
    per_class_hours: dict[str, float] = {}
    for job in jobs:
        per_device[job.device_key] += job.estimated_seconds
        per_class_hours[job.gpu_class] = per_class_hours.get(job.gpu_class, 0.0) + job.estimated_seconds / 3600.0
    loads = list(per_device.values())
    wall = max(loads) if loads else 0.0
    least = min(loads) if loads else 0.0
    ratio = (wall / least) if least > 0 else (math.inf if wall > 0 else 1.0)
    cells = {job.cell_id for job in jobs}
    split_cells = sorted({job.cell_id for job in jobs if job.shard_count > 1})
    return {
        "job_count": len(jobs),
        "cell_count": len(cells),
        "split_cells": split_cells,
        "total_gpu_hours": round(sum(job.estimated_seconds for job in jobs) / 3600.0, 3),
        "wall_clock_hours": round(wall / 3600.0, 3),
        "wall_clock_seconds": round(wall, 1),
        "per_device_seconds": {key: round(value, 1) for key, value in sorted(per_device.items())},
        "per_device_hours": {key: round(value / 3600.0, 3) for key, value in sorted(per_device.items())},
        "per_class_gpu_hours": {key: round(value, 3) for key, value in sorted(per_class_hours.items())},
        "max_min_load_ratio": None if math.isinf(ratio) else round(ratio, 4),
        "balance_target_ratio": BALANCE_TARGET_RATIO,
        "balanced_within_target": bool(ratio <= BALANCE_TARGET_RATIO),
    }


def plan_document(jobs: Sequence[Job], devices: Sequence[Device], table: Mapping[str, Mapping[str, float]],
                  registrations: Sequence[Mapping[str, Any]], *, assumptions: Mapping[str, Any]) -> dict[str, Any]:
    per_gpu: dict[str, dict[str, Any]] = {}
    for device in devices:
        device_jobs = sorted((job for job in jobs if job.device_key == device.key),
                             key=lambda job: (-job.estimated_seconds, job.cell_id, job.shard_index))
        per_gpu[device.key] = {
            "server": device.server, "gpu_index": device.gpu_index, "gpu_class": device.gpu_class,
            "device_model": GPU_CLASS_MODEL.get(device.gpu_class, f"GPU class {device.gpu_class}"),
            "jobs": [f"{job.cell_id}#{job.shard_index}/{job.shard_count}" for job in device_jobs],
            "estimated_seconds": round(sum(job.estimated_seconds for job in device_jobs), 1),
        }
    body = {
        "schema_version": PLAN_SCHEMA,
        "assumptions": dict(assumptions),
        "devices": [{"key": device.key, "server": device.server, "gpu_index": device.gpu_index,
                     "gpu_class": device.gpu_class,
                     "device_model": GPU_CLASS_MODEL.get(device.gpu_class, f"GPU class {device.gpu_class}")}
                    for device in devices],
        "throughput_table_encodes_per_second": {gpu_class: dict(models) for gpu_class, models in sorted(table.items())},
        "registrations": [dict(entry) for entry in registrations],
        "jobs": [asdict(job) for job in jobs],
        "per_gpu": per_gpu,
        "summary": summarize(jobs, devices),
        "sealed_evaluation_access": any(bool(entry.get("sealed_evaluation_access")) for entry in registrations),
        "note": "planning reads registrations only; it touches no data and no outcome and is not gated by the approval",
    }
    body["plan_sha256"] = canonical_json_hash(body)
    return body


# --------------------------------------------------------------------------- scripts


RUN_TEMPLATE = """#!/usr/bin/env bash
# Auto-generated by satml2027ext/plan_ext.py - host {server}, GPU {gpu} ({gpu_class}); plan sha256 {plan_sha256}
# Jobs are sequential on this GPU; run one script per GPU, all in parallel (see launch_all.sh).
set -euo pipefail
cd "$(dirname "$0")/{to_root}"
export CUDA_VISIBLE_DEVICES={gpu}
export PYTHONUNBUFFERED=1
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
mkdir -p {logs}
{lines}
echo "[$(date -Is)] GPU {gpu} on host {server}: all jobs finished"
"""

JOB_LINE = (
    'echo "[$(date -Is)] start {cell_id} shard {shard_index}/{shard_count}"\n'
    "{python} satml2027ext/run_shard_ext.py \\\n"
    "  --registration {registration} --cell-id {cell_id} \\\n"
    "  --shard-index {shard_index} --shard-count {shard_count} \\\n"
    "  --items {items} --banks {banks} --data-root {data_root} --out {out} \\\n"
    "  --bank-manifest {bank_manifest} --data-preflight {data_preflight} \\\n"
    "  --blocks-per-forward {blocks}{extra} \\\n"
    "  2>&1 | tee -a {logs}/{cell_id}__shard{shard_index}.log"
)

LAUNCH_ALL_TEMPLATE = """#!/usr/bin/env bash
# Auto-generated by satml2027ext/plan_ext.py - start every per-GPU script on this host under nohup/setsid.
set -euo pipefail
cd "$(dirname "$0")/{to_root}"
mkdir -p {logs} {pids}
for gpu in {gpus}; do
  if [ -f {pids}/gpu$gpu.pid ] && kill -0 "$(cat {pids}/gpu$gpu.pid)" 2>/dev/null; then
    echo "GPU $gpu: already running (pid $(cat {pids}/gpu$gpu.pid))"
    continue
  fi
  setsid nohup bash {launch}/run_gpu$gpu.sh > {logs}/gpu$gpu.log 2>&1 < /dev/null &
  echo $! > {pids}/gpu$gpu.pid
  echo "GPU $gpu: started pid $!"
done
echo "logs: {logs}/gpu<i>.log   pids: {pids}/gpu<i>.pid   status: bash {launch}/status.sh"
"""

STATUS_TEMPLATE = """#!/usr/bin/env bash
# Auto-generated by satml2027ext/plan_ext.py - per-GPU progress from logs and shard metadata sidecars only.
# Reads {logs}/gpu<i>.log, {pids}/gpu<i>.pid and .../cells/<cell>/shard_XXX_of_YYY.meta.json.
# It never opens a count or margin archive, so no scientific outcome is inspected.
cd "$(dirname "$0")/{to_root}"
progress() {{
  local meta="$1"
  if [ -f "$meta" ]; then
    local done total
    done=$(grep -o '"items_done": *[0-9]*' "$meta" | grep -o '[0-9]*$' || true)
    total=$(grep -o '"items_total": *[0-9]*' "$meta" | grep -o '[0-9]*$' || true)
    echo "${{done:-?}}/${{total:-?}}"
  else
    echo "-"
  fi
}}
{blocks}
"""

STATUS_BLOCK = """echo "GPU {gpu} ({key}, {gpu_class})"
if [ -f {pids}/gpu{gpu}.pid ] && kill -0 "$(cat {pids}/gpu{gpu}.pid)" 2>/dev/null; then
  echo "  process: running (pid $(cat {pids}/gpu{gpu}.pid))"
else
  echo "  process: not running"
fi
cells_done=0; current=""
for entry in {entries}; do
  cell="${{entry%%|*}}"; meta="${{entry##*|}}"
  p=$(progress "$meta")
  if [ "$p" != "-" ] && [ "${{p%%/*}}" = "${{p##*/}}" ]; then
    cells_done=$((cells_done + 1))
  elif [ -z "$current" ]; then
    current="$cell ($p items)"
  fi
done
echo "  jobs done: $cells_done/{job_count}   current: ${{current:-none}}"
echo "  last log line: $(tail -n 1 {logs}/gpu{gpu}.log 2>/dev/null || echo '(no log yet)')"
"""


def _relative_to_root(directory: Path) -> str:
    try:
        return Path(os.path.relpath(PROJECT_ROOT.resolve(), directory.resolve())).as_posix()
    except ValueError:  # different drive on Windows
        return PROJECT_ROOT.resolve().as_posix()


def write_scripts(
    out_dir: Path, jobs: Sequence[Job], devices: Sequence[Device], plan: Mapping[str, Any], *,
    items: Path, banks: Path, data_root: Path, results_root: Path, blocks_per_forward: int, python: str = "python",
    bank_manifest: Path = Path("results/satml2027ext/EXP021_BANK_MANIFEST.json"),
    data_preflight: Path = Path("results/satml2027ext/EXP021_DATA_PREFLIGHT.json"),
    cache_dir: str | None = None,
) -> list[Path]:
    launch = out_dir / "launch"
    launch.mkdir(parents=True, exist_ok=True)
    logs = (results_root / "logs").as_posix()
    pids = (results_root / "pids").as_posix()
    to_root = _relative_to_root(launch)
    written: list[Path] = []
    single_host = len({device.server for device in devices}) == 1 and devices[0].server == SINGLE_HOST_NAME
    status_blocks = []
    for device in devices:
        device_jobs = sorted((job for job in jobs if job.device_key == device.key),
                             key=lambda job: (-job.estimated_seconds, job.cell_id, job.shard_index))
        lines = []
        entries = []
        for job in device_jobs:
            extra = ""
            if job.extra_args:
                extra += " " + " ".join(job.extra_args)
            if cache_dir:
                extra += f" --cache-dir {cache_dir}"
            out = (results_root / job.experiment_id).as_posix()
            lines.append(JOB_LINE.format(
                python=python, registration=job.registration_path, cell_id=job.cell_id, shard_index=job.shard_index,
                shard_count=job.shard_count, items=items.as_posix(), banks=banks.as_posix(),
                data_root=data_root.as_posix(), out=out, blocks=blocks_per_forward, extra=extra, logs=logs,
                bank_manifest=bank_manifest.as_posix(), data_preflight=data_preflight.as_posix(),
            ))
            meta = f"{out}/cells/{job.cell_id}/shard_{job.shard_index:03d}_of_{job.shard_count:03d}.meta.json"
            entries.append(f'"{job.cell_id}#{job.shard_index}|{meta}"')
        name = f"run_gpu{device.gpu_index}.sh" if single_host else f"run_{device.server}_gpu{device.gpu_index}.sh"
        path = launch / name
        _write_new(path, RUN_TEMPLATE.format(
            server=device.server, gpu=device.gpu_index, gpu_class=device.gpu_class, plan_sha256=plan["plan_sha256"],
            to_root=to_root, logs=logs, lines="\n".join(lines) if lines else 'echo "no jobs assigned"',
        ))
        written.append(path)
        status_blocks.append(STATUS_BLOCK.format(
            gpu=device.gpu_index, key=device.key, gpu_class=device.gpu_class, pids=pids, logs=logs,
            entries=" ".join(entries) if entries else '""', job_count=len(device_jobs),
        ))
    if single_host:
        launch_rel = Path(os.path.relpath(launch.resolve(), PROJECT_ROOT.resolve())).as_posix() \
            if _relative_to_root(launch) != PROJECT_ROOT.resolve().as_posix() else launch.resolve().as_posix()
        gpus = " ".join(str(device.gpu_index) for device in devices)
        path = launch / "launch_all.sh"
        _write_new(path, LAUNCH_ALL_TEMPLATE.format(to_root=to_root, logs=logs, pids=pids, gpus=gpus, launch=launch_rel))
        written.append(path)
    path = launch / "status.sh"
    _write_new(path, STATUS_TEMPLATE.format(to_root=to_root, logs=logs, pids=pids, blocks="\n".join(status_blocks)))
    written.append(path)
    for path in written:
        try:
            path.chmod(0o755)
        except OSError:  # pragma: no cover - Windows
            pass
    return written


def _write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(text, encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- preparation stage


PREP_EXP016_ENCODES = 20000  # 021A: no image encoding (saved EXP-016 tensors); about the prompt-learning time
PREP_LINE = (
    'echo "[$(date -Is)] prepare {cell_id}"\n'
    "{python} satml2027ext/prepare_banks_ext.py --registration {registration} --cell-id {cell_id} \\\n"
    "  --items {items} --data-root {data_root} --data-preflight {data_preflight} \\\n"
    "  --banks-out {banks} --preparation-out {preparation}{extra} \\\n"
    "  2>&1 | tee -a {logs}/prepare__{cell_id}.log"
)


def preparation_jobs(registration_paths: Sequence[Path]) -> list[Job]:
    """One job per 021A/021B cell (021C scores its parent's payload); encodes estimated from the registered recipe."""

    jobs: list[Job] = []
    for path in registration_paths:
        registration = load_registration_v2(path)
        if "subset_of" in (registration.get("candidates") or {}):
            continue
        construction = registration["bank_construction"]
        for cell in registration["cells"]:
            if construction["source_mode"] == "exp016_saved_tensors":
                encodes = PREP_EXP016_ENCODES
            else:
                development = int(cell["development_item_count"])
                control = int(cell["control_train_item_count"])
                encodes = (development * (1 + int(construction["direction_construction"]["development_draws_per_item"]))
                           + control * (1 + int(construction["control_training"]["train_draws_per_item"])))
            jobs.append(Job(
                experiment_id=registration["experiment_id"], registration_path=Path(path).as_posix(),
                registration_sha256=registration["registration_sha256"], cell_id=cell["cell_id"],
                model_id=str(cell["model_id"]), dataset_id=str(cell["dataset_id"]), sigma=float(cell["sigma"]),
                shard_index=0, shard_count=1, item_count=0, draws_per_item=0, encodes=encodes, store_margins=False,
            ))
    return jobs


def write_preparation_scripts(out_dir: Path, jobs: Sequence[Job], devices: Sequence[Device], *, items: Path,
                              data_root: Path, data_preflight: Path, banks: Path, preparation: Path,
                              results_root: Path, python: str = "python", cache_dir: str | None = None) -> list[Path]:
    launch = out_dir / "launch_preparation"
    launch.mkdir(parents=True, exist_ok=True)
    logs = (results_root / "logs").as_posix()
    to_root = _relative_to_root(launch)
    written: list[Path] = []
    for device in devices:
        device_jobs = sorted((job for job in jobs if job.device_key == device.key), key=lambda job: (-job.estimated_seconds, job.cell_id))
        lines = [PREP_LINE.format(python=python, registration=job.registration_path, cell_id=job.cell_id,
                                  items=items.as_posix(), data_root=data_root.as_posix(),
                                  data_preflight=data_preflight.as_posix(), banks=banks.as_posix(),
                                  preparation=preparation.as_posix(), logs=logs,
                                  extra=f" --cache-dir {cache_dir}" if cache_dir else "") for job in device_jobs]
        path = launch / f"prepare_gpu{device.gpu_index}.sh"
        _write_new(path, RUN_TEMPLATE.format(
            server=device.server, gpu=device.gpu_index, gpu_class=device.gpu_class, plan_sha256="preparation stage",
            to_root=to_root, logs=logs, lines="\n".join(lines) if lines else 'echo "no jobs assigned"',
        ))
        written.append(path)
    path = launch / "after_all_gpus.txt"
    _write_new(path, "When every prepare_gpu<i>.sh has finished, seal the manifest (one process, CPU is enough):\n"
               f"  {python} satml2027ext/bank_manifest_ext.py --data-preflight {data_preflight.as_posix()} --out results/satml2027ext/EXP021_BANK_MANIFEST.json\n"
               "Then send EXP021_DATA_PREFLIGHT.json and EXP021_BANK_MANIFEST.json for the stage-2 (sampling) approval.\n")
    written.append(path)
    return written


# --------------------------------------------------------------------------- main


def build_plan(
    registration_paths: Sequence[Path], devices: Sequence[Device], table: Mapping[str, Mapping[str, float]], *,
    max_shard_items: int = 0, store_margins: bool = True, assumptions: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[Job]]:
    cells: list[dict[str, Any]] = []
    registrations = []
    loaded = [(Path(path), load_registration_v2(path)) for path in registration_paths]
    by_hash = {registration["registration_sha256"]: path for path, registration in loaded}
    for path, registration in loaded:
        registrations.append({
            "path": Path(path).as_posix(), "experiment_id": registration["experiment_id"],
            "registration_sha256": registration["registration_sha256"],
            "cell_count": len(registration["cells"]),
            "draws_per_item": certification_parameters(registration)["selection_draws"] + certification_parameters(registration)["confirmation_draws"],
            "store_margins": bool(store_margins and margins_registered(registration)),
            "sealed_evaluation_access": bool(registration.get("sealed_evaluation_access", False)),
        })
        parent = (registration.get("candidates") or {}).get("subset_of")
        parent_path = by_hash.get(parent["registration_sha256"]) if parent else None
        cells.extend(cells_from_registration(registration, Path(path), store_margins=store_margins, parent_path=parent_path))
    jobs = assign_jobs(expand_jobs(cells, max_shard_items=max_shard_items), devices, table)
    plan = plan_document(jobs, devices, table, registrations,
                         assumptions={"max_shard_items": max_shard_items, **(assumptions or {})})
    return plan, jobs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--registrations", nargs="+", required=True, type=Path)
    parser.add_argument("--devices", nargs="+", required=True, help="CLASS:COUNT (single host) or NAME:COUNT:CLASS")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--preflight", type=Path, default=None, help="directory with preflight_*.json throughput reports")
    parser.add_argument("--rate-factor-5090", type=float, default=GPU_CLASS_RELATIVE_RATE["5090"])
    parser.add_argument("--items", type=Path, default=Path("results/satml2027ext/items"))
    parser.add_argument("--banks", type=Path, default=Path("results/satml2027ext/banks"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--results-root", type=Path, default=Path("results/satml2027ext"))
    parser.add_argument("--blocks-per-forward", type=int, default=None,
                        help="must equal the registered certification.blocks_per_forward (default: the registered value)")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache directory passed to every job")
    parser.add_argument("--max-shard-items", type=int, default=1000,
                        help="contiguous shard size; must not exceed the registered sharding.max_items_per_shard (1,000)")
    parser.add_argument("--bank-manifest", type=Path, default=Path("results/satml2027ext/EXP021_BANK_MANIFEST.json"))
    parser.add_argument("--data-preflight", type=Path, default=Path("results/satml2027ext/EXP021_DATA_PREFLIGHT.json"))
    parser.add_argument("--no-store-margins", action="store_true")
    parser.add_argument("--python", default="python")
    parser.add_argument("--stage", choices=["sampling", "preparation"], default="sampling")
    parser.add_argument("--preparation-out", type=Path, default=Path("results/satml2027ext/preparation"))
    args = parser.parse_args(argv)
    registered_blocks = {certification_parameters(load_registration_v2(path)).get("blocks_per_forward") for path in args.registrations}
    registered_blocks.discard(None)
    if len(registered_blocks) > 1:
        raise SystemExit(f"registrations disagree on blocks_per_forward: {sorted(registered_blocks)}")
    if registered_blocks:
        registered = next(iter(registered_blocks))
        if args.blocks_per_forward is not None and args.blocks_per_forward != registered:
            raise SystemExit(f"--blocks-per-forward {args.blocks_per_forward} differs from the registered {registered}")
        args.blocks_per_forward = registered
    elif args.blocks_per_forward is None:
        args.blocks_per_forward = 4

    if args.stage == "preparation":
        devices = parse_devices(args.devices)
        table = merge_throughput(default_throughput_table(args.rate_factor_5090), load_preflight_throughput(args.preflight))
        jobs = assign_jobs(preparation_jobs(args.registrations), devices, table)
        summary = summarize(jobs, devices)
        scripts = write_preparation_scripts(args.out, jobs, devices, items=args.items, data_root=args.data_root,
                                            data_preflight=args.data_preflight, banks=args.banks,
                                            preparation=args.preparation_out, results_root=args.results_root,
                                            python=args.python, cache_dir=args.cache_dir)
        print(json.dumps(summary, indent=2))
        print(f"preparation: {len(jobs)} cells; wall-clock estimate {summary['wall_clock_hours']:.2f} h; scripts in {args.out / 'launch_preparation'}")
        return 0

    plan_path = args.out / "plan.json"
    if plan_path.exists():
        raise FileExistsError(f"refusing to overwrite {plan_path}")
    devices = parse_devices(args.devices)
    overrides = load_preflight_throughput(args.preflight)
    table = merge_throughput(default_throughput_table(args.rate_factor_5090), overrides)
    assumptions = {
        "calibration": "plan Section 2.5: seconds per position at 4,224 draws on a 4090",
        "seconds_per_position_4090": SECONDS_PER_POSITION_4090,
        "rate_factor_5090_vs_4090": args.rate_factor_5090,
        "preflight_overrides": overrides,
        "preferred_class_by_model": PREFERRED_CLASS_BY_MODEL,
        "blocks_per_forward": args.blocks_per_forward,
        "tf32": "both switches off in the worker; NVIDIA_TF32_OVERRIDE=0 and TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 exported",
        "bit_identity_across_gpu_classes": "not expected; disclosed",
    }
    plan, jobs = build_plan(args.registrations, devices, table, max_shard_items=args.max_shard_items,
                            store_margins=not args.no_store_margins, assumptions=assumptions)
    write_json_atomic(plan_path, plan)
    (args.out / "plan.sha256").write_text(f"{plan['plan_sha256']}  plan.json\n", encoding="utf-8", newline="\n")
    scripts = write_scripts(args.out, jobs, devices, plan, items=args.items, banks=args.banks,
                            data_root=args.data_root, results_root=args.results_root,
                            blocks_per_forward=args.blocks_per_forward, python=args.python,
                            bank_manifest=args.bank_manifest, data_preflight=args.data_preflight,
                            cache_dir=args.cache_dir)
    summary = plan["summary"]
    print(json.dumps(summary, indent=2))
    print(f"\nplan sha256 {plan['plan_sha256']}")
    print(f"wall-clock estimate: {summary['wall_clock_hours']:.2f} h on {len(devices)} GPU(s) "
          f"({summary['total_gpu_hours']:.2f} GPU-hours; max/min load ratio "
          f"{summary['max_min_load_ratio']}, target <= {BALANCE_TARGET_RATIO})")
    print(f"scripts: {', '.join(path.name for path in scripts)} in {args.out / 'launch'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
