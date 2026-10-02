"""Backbone registry and loading.

The five backbones are chosen so that every comparison a reviewer may want is
available on the page: OVC (SaTML 2024) certified CLIP RN50, CLIP ViT-B/32 and
OpenCLIP ViT-B-32 (LAION); Chowers (CVPR 2026) used ViT-B/16 and ViT-L/14; Sato
(EMNLP 2026) used ViT-B/32 with B/16 and L/14 ablations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "openai-clip-vit-b32-quickgelu": {
        "open_clip_name": "ViT-B-32-quickgelu",
        "pretrained": "openai",
        "embedding_dim": 512,
        "in_exp017": True,
        "comparability": ["OVC 2024", "Sato 2026 (main)", "Liang 2022"],
    },
    "openai-clip-vit-l14-quickgelu": {
        "open_clip_name": "ViT-L-14-quickgelu",
        "pretrained": "openai",
        "embedding_dim": 768,
        "in_exp017": True,
        "comparability": ["Chowers 2026", "Sato 2026 (ablation)"],
    },
    "openai-clip-vit-b16-quickgelu": {
        "open_clip_name": "ViT-B-16-quickgelu",
        "pretrained": "openai",
        "embedding_dim": 512,
        "in_exp017": False,
        "comparability": ["Chowers 2026", "Sato 2026 (ablation)", "Liang 2022", "OVC 2024 (timing)"],
    },
    "openclip-vit-b32-laion2b": {
        "open_clip_name": "ViT-B-32",
        "pretrained": "laion2b_s34b_b79k",
        "embedding_dim": 512,
        "in_exp017": False,
        "comparability": ["OVC 2024 (OpenCLIP ViT-B-32)"],
    },
    "openai-clip-rn50-quickgelu": {
        "open_clip_name": "RN50-quickgelu",
        "pretrained": "openai",
        "embedding_dim": 1024,
        "in_exp017": False,
        "comparability": ["OVC 2024 (main backbone)"],
    },
}


@dataclass
class LoadedModel:
    model_id: str
    model: Any
    preprocess: Any
    tokenizer: Any
    mean: tuple[float, ...]
    std: tuple[float, ...]
    input_size: int
    device: str
    dtype: str


def resolve(model_id: str) -> dict[str, Any]:
    if model_id not in MODEL_REGISTRY:
        raise KeyError(f"unknown model_id {model_id}; registered: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_id]


def load_model(model_id: str, *, device: str = "cuda", cache_dir: str | None = None) -> LoadedModel:
    """Load a frozen backbone in float32 and put it in eval/inference mode."""

    import open_clip
    import torch

    spec = resolve(model_id)
    model, _, preprocess = open_clip.create_model_and_transforms(
        spec["open_clip_name"], pretrained=spec["pretrained"], cache_dir=cache_dir
    )
    model = model.to(device=device, dtype=torch.float32).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(spec["open_clip_name"])
    normalize = _find_normalize(preprocess)
    resize = _find_input_size(preprocess)
    return LoadedModel(
        model_id=model_id,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        mean=tuple(float(value) for value in normalize.mean),
        std=tuple(float(value) for value in normalize.std),
        input_size=resize,
        device=device,
        dtype="float32",
    )


def _find_normalize(transform):
    from torchvision.transforms import Normalize

    for step in getattr(transform, "transforms", []):
        if isinstance(step, Normalize):
            return step
    raise RuntimeError("preprocessing pipeline has no Normalize step")


def _find_input_size(transform) -> int:
    from torchvision.transforms import CenterCrop

    for step in getattr(transform, "transforms", []):
        if isinstance(step, CenterCrop):
            size = step.size
            return int(size[0] if isinstance(size, (tuple, list)) else size)
    raise RuntimeError("preprocessing pipeline has no CenterCrop step")


def encode_images(model, pixels, *, normalize_rows: bool = True):
    """Encode standardized pixels and return unit-norm float32 features."""

    import torch

    with torch.inference_mode():
        features = model.encode_image(pixels)
    features = features.to(dtype=torch.float32)
    if normalize_rows:
        features = features / torch.linalg.vector_norm(features, dim=-1, keepdim=True)
    return features


def build_text_prototypes(loaded: LoadedModel, rendered_prompts, *, batch_size: int = 256):
    """Prompt-ensemble prototypes: mean of unit-norm text features, renormalized."""

    import torch

    rows = []
    for prompts in rendered_prompts:
        tokens = loaded.tokenizer(list(prompts)).to(loaded.device)
        with torch.inference_mode():
            features = loaded.model.encode_text(tokens).to(dtype=torch.float32)
        features = features / torch.linalg.vector_norm(features, dim=-1, keepdim=True)
        mean = features.mean(dim=0)
        rows.append(mean / torch.linalg.vector_norm(mean))
    return torch.stack(rows).cpu().contiguous()
