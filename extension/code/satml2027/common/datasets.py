"""Dataset access with frozen item identities.

Item identifiers follow the Phase-3 convention already in the assignments table:
  cifar100  ->  "cifar100-train-{index}"      (torchvision CIFAR100 train split)
  cifar10   ->  "cifar10-train-{index}"       (torchvision CIFAR10 train split)
  eurosat   ->  "eurosat/2750/{Class}/{file}" (path relative to the dataset root)

Only training splits are used.  Official test splits are never read by any script
in this bundle; the EuroSAT sealed test rows live in the Phase-3 assignments file
and are excluded by role.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "cifar100": {"class_count": 100, "torchvision": "CIFAR100", "split": "train"},
    "cifar10": {"class_count": 10, "torchvision": "CIFAR10", "split": "train"},
    "eurosat": {"class_count": 10, "torchvision": "EuroSAT", "split": "all"},
}


def load_dataset(dataset_id: str, root: str | Path, transform=None):
    """Return (dataset, item_ids, labels) with download disabled."""

    from torchvision.datasets import CIFAR10, CIFAR100, EuroSAT

    base = Path(root)
    if dataset_id == "cifar100":
        dataset = CIFAR100(root=base, train=True, download=False, transform=transform)
        item_ids = [f"cifar100-train-{index}" for index in range(len(dataset))]
        labels = [int(value) for value in dataset.targets]
    elif dataset_id == "cifar10":
        dataset = CIFAR10(root=base, train=True, download=False, transform=transform)
        item_ids = [f"cifar10-train-{index}" for index in range(len(dataset))]
        labels = [int(value) for value in dataset.targets]
    elif dataset_id == "eurosat":
        dataset = EuroSAT(root=base, download=False, transform=transform)
        item_ids = [Path(path).resolve().relative_to(base.resolve()).as_posix() for path, _ in dataset.samples]
        labels = [int(label) for _, label in dataset.samples]
    else:
        raise ValueError(f"unsupported dataset {dataset_id}")
    return dataset, item_ids, labels


def class_names(dataset, prompt_config: dict[str, Any]) -> list[str]:
    """Map torchvision class folders to the prompt config's class names."""

    configured = prompt_config.get("class_names")
    mapping = prompt_config.get("class_name_by_torchvision_folder")
    if mapping is not None:
        if set(mapping) != set(dataset.classes):
            raise RuntimeError("prompt mapping does not match dataset classes")
        return [mapping[name] for name in dataset.classes]
    if configured is not None:
        if len(configured) != len(dataset.classes):
            raise RuntimeError("prompt class_names length differs from the dataset")
        return list(configured)
    return list(dataset.classes)


def recover_pixels(dataset, dataset_indices, mean, std):
    """Undo the preprocessing Normalize to recover pixels in [0, 1]."""

    import torch

    normalized = torch.stack([dataset[int(index)][0] for index in dataset_indices])
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    pixels = normalized.to(dtype=torch.float32) * std_tensor + mean_tensor
    if not torch.isfinite(pixels).all() or float(pixels.min()) < -1e-6 or float(pixels.max()) > 1.0 + 1e-6:
        raise RuntimeError("recovered pixels escaped [0,1]")
    return pixels.clamp_(0.0, 1.0)
