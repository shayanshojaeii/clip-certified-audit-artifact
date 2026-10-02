"""Deterministic, shard-independent noise seeding.

Two properties are required and are both preregistered:

1.  **Per-item independence.**  Every (cell, item, stream) pair gets its own
    seed derived by SHA-256 from the registration's base seed, so an item can be
    computed on any GPU on any server without changing its noise, and shards are
    exactly the same as a single-process run.

2.  **Block-invariance.**  Noise is generated in fixed-size blocks of
    ``NOISE_BLOCK`` draws, each block with its own generator seeded from the item
    seed and the block index.  The forward batch size may therefore be tuned per
    GPU without changing a single realized draw.

The payload includes the model, dataset, sigma and stream role, so the sigma
sweep uses independent noise streams rather than a common one scaled by sigma.
"""

from __future__ import annotations

import hashlib

NOISE_BLOCK = 64  # frozen: draws generated per generator seeding
MASK63 = (1 << 63) - 1

SELECTION_STREAM = "cohen_class_selection"
CONFIRMATION_STREAM = "cohen_class_confirmation"
DIAGNOSTIC_STREAM = "n3c_fresh_noise_diagnostic"
DEVELOPMENT_STREAM = "development_direction_estimation"
CONTROL_TRAIN_STREAM = "control_bank_training_noise"


def derive_item_seed(
    *,
    base_seed: int,
    model_id: str,
    dataset_id: str,
    sigma: float,
    item_id: str,
    stream_role: str,
) -> int:
    """Return the 63-bit seed for one item's noise stream."""

    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer")
    if not isinstance(sigma, float) or not (sigma > 0.0):
        raise ValueError("sigma must be a positive float")
    for value, name in ((model_id, "model_id"), (dataset_id, "dataset_id"), (item_id, "item_id"), (stream_role, "stream_role")):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a nonempty string")
    payload = f"{base_seed}:{model_id}:{dataset_id}:{sigma:.12g}:{item_id}:{stream_role}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & MASK63


def derive_block_seed(item_seed: int, block_index: int) -> int:
    """Return the seed of one fixed-size noise block inside an item's stream."""

    if isinstance(item_seed, bool) or not isinstance(item_seed, int) or item_seed < 0:
        raise ValueError("item_seed must be a nonnegative integer")
    if isinstance(block_index, bool) or not isinstance(block_index, int) or block_index < 0:
        raise ValueError("block_index must be a nonnegative integer")
    payload = f"{item_seed}:{block_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & MASK63


def block_plan(draws: int) -> list[tuple[int, int]]:
    """Return [(block_index, draws_in_block)] covering ``draws`` in fixed blocks."""

    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    plan = []
    remaining, index = draws, 0
    while remaining > 0:
        take = min(NOISE_BLOCK, remaining)
        plan.append((index, take))
        remaining -= take
        index += 1
    return plan
