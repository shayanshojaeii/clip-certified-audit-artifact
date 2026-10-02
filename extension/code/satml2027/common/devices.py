"""GPU inventory and cost-balanced shard planning across heterogeneous servers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from typing import Any, Sequence

# Fallback relative speeds, used only until d1_01_preflight.py measures the real
# throughput of each (server, gpu, model) pair.  Encodes per second, float32.
FALLBACK_THROUGHPUT: dict[str, dict[str, float]] = {
    "4090": {"openai-clip-vit-b32-quickgelu": 620.0, "openai-clip-vit-b16-quickgelu": 185.0,
             "openai-clip-vit-l14-quickgelu": 95.0, "openclip-vit-b32-laion2b": 620.0,
             "openai-clip-rn50-quickgelu": 780.0},
    "3090": {"openai-clip-vit-b32-quickgelu": 280.0, "openai-clip-vit-b16-quickgelu": 84.0,
             "openai-clip-vit-l14-quickgelu": 43.0, "openclip-vit-b32-laion2b": 280.0,
             "openai-clip-rn50-quickgelu": 350.0},
}


@dataclass(frozen=True)
class Device:
    server: str
    gpu_index: int
    gpu_class: str

    @property
    def key(self) -> str:
        return f"{self.server}:{self.gpu_index}"


@dataclass
class ShardJob:
    cell_id: str
    model_id: str
    dataset_id: str
    sigma: float
    shard_index: int
    shard_count: int
    item_count: int
    encodes: int
    estimated_seconds: float
    device_key: str


def classify_gpu(name: str) -> str:
    upper = name.upper()
    for token in ("4090", "3090", "A100", "H100", "6000", "4080", "3080"):
        if token in upper:
            return token
    return "other"


def local_inventory(server: str) -> list[Device]:
    import torch

    if not torch.cuda.is_available():
        return []
    return [
        Device(server=server, gpu_index=index, gpu_class=classify_gpu(torch.cuda.get_device_name(index)))
        for index in range(torch.cuda.device_count())
    ]


def throughput(table: dict[str, Any], gpu_class: str, model_id: str) -> float:
    measured = table.get(gpu_class, {}).get(model_id)
    if measured:
        return float(measured)
    fallback = FALLBACK_THROUGHPUT.get(gpu_class, {}).get(model_id)
    if fallback:
        return float(fallback)
    slowest = min(FALLBACK_THROUGHPUT["3090"].values())
    return slowest


def plan_shards(
    *,
    cells: Sequence[dict[str, Any]],
    devices: Sequence[Device],
    throughput_table: dict[str, Any],
    draws_per_item: int | dict[str, int],
    max_shard_items: int = 0,
) -> list[ShardJob]:
    """Split every cell into shards and greedily balance them over all devices.

    Cells are split so that no single shard is longer than the slowest device's
    time for ``ceil(items / devices)`` items; longest jobs are placed first.
    """

    if not devices:
        raise RuntimeError("no CUDA devices available for planning")
    jobs: list[ShardJob] = []
    by_server: dict[str, list[Device]] = {}
    for device in devices:
        by_server.setdefault(device.server, []).append(device)
    device_loads = {device.key: 0.0 for device in devices}
    server_loads = {server: 0.0 for server in by_server}
    # Assign a whole scientific cell to one server first. Shards may use several
    # GPUs on that server, but never cross a server boundary.
    def draws_for(cell: dict[str, Any]) -> int:
        if isinstance(draws_per_item, dict):
            try:
                return int(draws_per_item[cell["cell_id"]])
            except KeyError as error:
                raise KeyError(f"missing draw budget for cell {cell['cell_id']}") from error
        return int(draws_per_item)

    ordered_cells = sorted(cells, key=lambda cell: -int(cell["item_count"]) * draws_for(cell))
    for cell in ordered_cells:
        items = int(cell["item_count"])
        cell_draws = draws_for(cell)
        def projected_finish(server: str) -> float:
            rates = sum(throughput(throughput_table, d.gpu_class, cell["model_id"]) for d in by_server[server])
            return server_loads[server] + items * cell_draws / max(rates, 1e-12)
        server = min(by_server, key=projected_finish)
        server_devices = by_server[server]
        shard_count = max(1, min(items, len(server_devices)))
        if max_shard_items:
            shard_count = max(shard_count, math.ceil(items / max_shard_items))
        base, extra = divmod(items, shard_count)
        for shard_index in range(shard_count):
            shard_items = base + (1 if shard_index < extra else 0)
            if shard_items == 0:
                continue
            best = min(
                server_devices,
                key=lambda device: device_loads[device.key]
                + shard_items * cell_draws / throughput(throughput_table, device.gpu_class, cell["model_id"]),
            )
            seconds = shard_items * cell_draws / throughput(
                throughput_table, best.gpu_class, cell["model_id"]
            )
            job = ShardJob(
                    cell_id=cell["cell_id"],
                    model_id=cell["model_id"],
                    dataset_id=cell["dataset_id"],
                    sigma=float(cell["sigma"]),
                    shard_index=shard_index,
                    shard_count=shard_count,
                    item_count=shard_items,
                    encodes=shard_items * cell_draws,
                    estimated_seconds=seconds,
                    device_key=best.key,
                )
            jobs.append(job)
            device_loads[best.key] += seconds
        server_loads[server] = max(device_loads[d.key] for d in server_devices)
    return jobs


def summarize(jobs: Sequence[ShardJob]) -> dict[str, Any]:
    per_device: dict[str, float] = {}
    for job in jobs:
        per_device[job.device_key] = per_device.get(job.device_key, 0.0) + job.estimated_seconds
    return {
        "job_count": len(jobs),
        "total_gpu_hours": round(sum(job.estimated_seconds for job in jobs) / 3600.0, 2),
        "wall_clock_hours": round(max(per_device.values()) / 3600.0, 2) if per_device else 0.0,
        "per_device_hours": {key: round(value / 3600.0, 2) for key, value in sorted(per_device.items())},
    }


def jobs_to_json(jobs: Sequence[ShardJob]) -> list[dict[str, Any]]:
    return [asdict(job) for job in jobs]
