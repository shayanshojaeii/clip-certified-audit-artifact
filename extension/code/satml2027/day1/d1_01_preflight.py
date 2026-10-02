#!/usr/bin/env python3
"""Day 1 - preflight: environment, checkpoints, RNG agreement, throughput, precision.

Run this on BOTH servers before anything else.  It writes
`preflight_<server>.json`, which `d1_04_plan.py` consumes to balance the shard
plan over heterogeneous GPUs, and it produces two custody facts:

  * `rng_digests` - a digest of the first noise block for a fixed seed at each
    input resolution.  The two servers must agree, otherwise the sharded run is
    not reproducible and the plan must be changed to keep every cell on one
    machine.
  * `precision_equivalence` - vote-count differences between float32 (the
    registered setting) and TF32 / autocast on a handful of items, so the paper
    can state what reduced precision would have cost.

Usage:
  python day1/d1_01_preflight.py --server A --models openai-clip-vit-b32-quickgelu ... \
      --out results/satml2027/preflight [--benchmark-draws 512] [--skip-precision]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.devices import classify_gpu  # noqa: E402
from common.hashing import write_json_atomic  # noqa: E402
from common.models import MODEL_REGISTRY, load_model, resolve  # noqa: E402
from common.seeds import NOISE_BLOCK, derive_block_seed  # noqa: E402
from day1.d1_05_run_shard import rng_selftest  # noqa: E402


def benchmark(loaded, *, draws: int, blocks_per_forward: int, device: str) -> dict:
    size = loaded.input_size
    pixel = torch.rand((3, size, size), device=device)
    mean = torch.tensor(loaded.mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(loaded.std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    batch = NOISE_BLOCK * blocks_per_forward
    generator = torch.Generator(device=device)
    generator.manual_seed(1234)
    # warm-up
    for _ in range(2):
        noise = torch.randn((batch, 3, size, size), generator=generator, dtype=torch.float32, device=device)
        with torch.inference_mode():
            loaded.model.encode_image(((pixel.unsqueeze(0) + 0.25 * noise) - mean) / std)
    torch.cuda.synchronize() if device == "cuda" else None
    started = time.time()
    encoded = 0
    while encoded < draws:
        noise = torch.randn((batch, 3, size, size), generator=generator, dtype=torch.float32, device=device)
        with torch.inference_mode():
            loaded.model.encode_image(((pixel.unsqueeze(0) + 0.25 * noise) - mean) / std)
        encoded += batch
    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = time.time() - started
    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
    return {
        "encodes_per_second": round(encoded / elapsed, 1),
        "forward_batch": batch,
        "peak_memory_gib": round(peak, 2),
    }


def precision_equivalence(loaded, *, device: str, draws: int = 512) -> dict:
    """Synthetic precision-disagreement diagnostic, not an equivalence proof."""

    size = loaded.input_size
    pixel = torch.rand((3, size, size), device=device)
    mean = torch.tensor(loaded.mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(loaded.std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    bank = torch.randn((10, loaded.model.visual.output_dim if hasattr(loaded.model.visual, "output_dim") else 512), device=device)
    bank = bank / torch.linalg.vector_norm(bank, dim=1, keepdim=True)
    generator = torch.Generator(device=device)
    generator.manual_seed(derive_block_seed(99, 0))
    noise = torch.randn((draws, 3, size, size), generator=generator, dtype=torch.float32, device=device)
    standardized = ((pixel.unsqueeze(0) + 0.25 * noise) - mean) / std

    def labels(allow_tf32: bool, autocast_dtype):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        out = []
        for start in range(0, draws, 128):
            chunk = standardized[start : start + 128]
            with torch.inference_mode():
                if autocast_dtype is None:
                    features = loaded.model.encode_image(chunk)
                else:
                    with torch.autocast("cuda", dtype=autocast_dtype):
                        features = loaded.model.encode_image(chunk)
            features = features.to(dtype=torch.float32)
            features = features / torch.linalg.vector_norm(features, dim=1, keepdim=True)
            out.append(torch.argmax(features @ bank.T, dim=1))
        return torch.cat(out)

    reference = labels(False, None)
    result = {}
    for name, (tf32, dtype) in {
        "tf32": (True, None),
        "autocast_fp16": (False, torch.float16),
        "autocast_bf16": (False, torch.bfloat16),
    }.items():
        try:
            other = labels(tf32, dtype)
            result[name] = {
                "label_disagreement_rate": float((other != reference).double().mean()),
                "draws": int(draws),
            }
        except Exception as error:  # pragma: no cover - hardware dependent
            result[name] = {"error": str(error)}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result["note"] = (
        "random bank; a disagreement rate of a few tenths of a percent is the "
        "scale at which reduced precision perturbs near-tie draws. The registered "
        "runs use float32 with TF32 disabled."
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--models", nargs="*", default=sorted(MODEL_REGISTRY))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--benchmark-draws", type=int, default=1024)
    parser.add_argument("--blocks-per-forward", type=int, default=4)
    parser.add_argument("--skip-precision", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report: dict = {
        "schema_version": "satml2027.preflight.v1",
        "server": args.server,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "devices": [],
        "throughput": {},
        "rng_digests": {},
        "synthetic_precision_disagreement": {},
        "failures": [],
    }
    try:
        import open_clip

        report["open_clip"] = open_clip.__version__
    except Exception as error:
        report["failures"].append(f"open_clip import failed: {error}")

    for index in range(torch.cuda.device_count() if device == "cuda" else 0):
        properties = torch.cuda.get_device_properties(index)
        report["devices"].append(
            {
                "index": index,
                "name": properties.name,
                "gpu_class": classify_gpu(properties.name),
                "total_memory_gib": round(properties.total_memory / 2**30, 1),
                "capability": f"{properties.major}.{properties.minor}",
            }
        )
    if not report["devices"]:
        report["failures"].append("no CUDA device visible")

    for shape in ((NOISE_BLOCK, 3, 224, 224), (NOISE_BLOCK, 3, 32, 32)):
        report["rng_digests"][str(list(shape))] = rng_selftest(device, shape)

    gpu_class = report["devices"][0]["gpu_class"] if report["devices"] else "cpu"
    for model_id in args.models:
        try:
            loaded = load_model(model_id, device=device, cache_dir=args.cache_dir)
        except Exception as error:
            report["failures"].append(f"{model_id}: load failed: {error}")
            continue
        spec = resolve(model_id)
        entry = {
            "input_size": loaded.input_size,
            "mean": loaded.mean,
            "std": loaded.std,
            "open_clip_name": spec["open_clip_name"],
            "pretrained_tag": spec["pretrained"],
            "preprocess_repr": repr(loaded.preprocess),
            "model_dtype": loaded.dtype,
        }
        entry.update(benchmark(loaded, draws=args.benchmark_draws, blocks_per_forward=args.blocks_per_forward, device=device))
        report["throughput"].setdefault(gpu_class, {})[model_id] = entry["encodes_per_second"]
        report.setdefault("model_details", {})[model_id] = entry
        if not args.skip_precision and device == "cuda":
            report["synthetic_precision_disagreement"][model_id] = precision_equivalence(loaded, device=device)
        del loaded
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    report["status"] = "PASS" if not report["failures"] else "FAIL"
    path = write_json_atomic(args.out / f"preflight_{args.server}.json", report)
    print(json.dumps({key: report[key] for key in ("status", "devices", "throughput", "rng_digests", "failures")}, indent=2))
    print(f"report: {path}")
    print("\nCompare rng_digests across servers - they must be identical.")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
