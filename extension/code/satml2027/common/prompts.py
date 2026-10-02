"""Prompt ensembles.

Existing datasets keep the project's frozen prompt configuration files.  CIFAR-10
is new to this project, so its prompt ensemble is registered here explicitly: it
is the same 18-template OpenAI ensemble already used for CIFAR-100, applied to
the ten CIFAR-10 class names in torchvision order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OPENAI_CIFAR_TEMPLATES: tuple[str, ...] = (
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a low contrast photo of a {}.",
    "a high contrast photo of a {}.",
    "a bad photo of a {}.",
    "a good photo of a {}.",
    "a photo of a small {}.",
    "a photo of a big {}.",
    "a photo of the {}.",
    "a blurry photo of the {}.",
    "a black and white photo of the {}.",
    "a low contrast photo of the {}.",
    "a high contrast photo of the {}.",
    "a bad photo of the {}.",
    "a good photo of the {}.",
    "a photo of the small {}.",
    "a photo of the big {}.",
)

CIFAR10_CLASS_NAMES: tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def cifar10_prompt_config() -> dict[str, Any]:
    return {
        "schema_version": "satml2027.prompt_config.v1",
        "dataset_id": "cifar10",
        "class_count": 10,
        "class_names": list(CIFAR10_CLASS_NAMES),
        "templates": list(OPENAI_CIFAR_TEMPLATES),
        "source": "openai/CLIP prompt-engineering notebook, CIFAR ensemble (same set as the frozen cifar100 config)",
        "class_order": "torchvision.datasets.CIFAR10.classes",
    }


def render_prompts(class_names, templates) -> list[list[str]]:
    rendered = []
    for name in class_names:
        row = []
        for template in templates:
            if "{class_name}" in template:
                row.append(template.format(class_name=name))
            else:
                row.append(template.format(name))
        rendered.append(row)
    return rendered


def load_prompt_config(project_root: Path, dataset_id: str, configured: str | None) -> dict[str, Any]:
    """Prefer the project's frozen prompt file; fall back to the registered CIFAR-10 set."""

    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = project_root / path
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            data.setdefault("dataset_id", dataset_id)
            # Persist only a project-relative registered identity; absolute local
            # paths can deanonymize binary artifacts and are not scientific data.
            try:
                data["prompt_config_path"] = path.resolve().relative_to(project_root.resolve()).as_posix()
            except ValueError:
                raise RuntimeError("prompt config must be inside the project root")
            return data
        raise FileNotFoundError(f"configured prompt file is missing: {path}")
    if dataset_id == "cifar10":
        return cifar10_prompt_config()
    raise ValueError(f"dataset {dataset_id} requires an explicit frozen prompt config")
