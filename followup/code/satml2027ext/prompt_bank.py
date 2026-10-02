"""Few-shot context-prompt bank (PromptSmooth-style) for EXP-021 control 4.

The bank is ``t_k = Normalize(TextEncoder([ctx_1 .. ctx_C] + template(class_k)))``
where the ``C`` context vectors live in the token-embedding space of the frozen
text encoder and are the only trainable parameters.  They are fitted by
noisy-plus-clean cross-entropy on the items of the cell's registered control
role (``development`` in 021A, ``control_train`` in 021B: the same supervision
every other supervised control receives), never on evaluation items; the
registered per-class budget ``shots_per_class`` is enforced.

The encoder is injected as two callables so that the learner is testable with
a tiny deterministic fake and so that this module never imports ``open_clip``:

``token_embedding_fn(prompts) -> TokenizedPrompts``
    embeds the rendered prompts (placeholder context words included) and
    reports where the context slots sit and where each prompt's EOT token is;
``text_encoder_fn(embeddings, eot_index) -> features [K, d]``
    runs the rest of the frozen encoder from the token-embedding layer and
    must be differentiable with respect to ``embeddings``.

``open_clip_prompt_hooks`` builds the two callables for a loaded open_clip CLIP
model from its own attributes (``token_embedding``, ``positional_embedding``,
``transformer``, ``ln_final``, ``text_projection``, ``attn_mask``); it takes the
model object, so open_clip is still not imported here.  Before trusting the
real path, ``verify_hooks_against_encode_text`` checks that the hooks reproduce
``model.encode_text`` on the unmodified placeholder prompts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch

from satml2027ext._common import PROJECT_ROOT  # noqa: F401  (puts satml2027/ on sys.path)
from common.hashing import tensor_scientific_hash  # noqa: E402

DEFAULT_TEMPLATE = "{context} a photo of a {name}."
DEFAULT_PLACEHOLDER = "X"
CONTEXT_INIT_STD = 0.02


@dataclass
class TokenizedPrompts:
    """Token embeddings of the rendered prompts and the positions the learner needs."""

    embeddings: torch.Tensor  # [K, L, D], the frozen token-embedding output
    context_start: int  # first context slot (the slot after SOT)
    eot_index: torch.Tensor  # [K] long, position of each prompt's EOT token


@dataclass
class LearnedPromptBank:
    prototypes: torch.Tensor  # [K, d] unit rows, float32, CPU
    context_vectors: torch.Tensor  # [C, D] learned context embeddings
    prompts: list[str]
    initial_loss: float
    final_loss: float
    loss_history: list[float] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)


def render_prompts(class_names: Sequence[str], template: str, *, context_tokens: int,
                   placeholder: str = DEFAULT_PLACEHOLDER) -> list[str]:
    if "{context}" not in template or "{name}" not in template:
        raise ValueError("template must contain the {context} and {name} placeholders")
    if not isinstance(context_tokens, int) or context_tokens < 1:
        raise ValueError("context_tokens must be a positive integer")
    context = " ".join([placeholder] * context_tokens)
    return [template.format(context=context, name=str(name)) for name in class_names]


def _validate_supervision(noisy_features, clean_features, labels, class_count, shots_per_class):
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1 or labels.dtype != torch.long:
        raise TypeError("labels must be a one-dimensional long tensor")
    if not isinstance(clean_features, torch.Tensor) or clean_features.ndim != 2:
        raise TypeError("clean_features must be [items, dim]")
    if not isinstance(noisy_features, torch.Tensor) or noisy_features.ndim != 3:
        raise TypeError("noisy_features must be [items, draws, dim]")
    items = labels.shape[0]
    if clean_features.shape[0] != items or noisy_features.shape[0] != items:
        raise ValueError("noisy_features, clean_features and labels must align on items")
    if noisy_features.shape[2] != clean_features.shape[1]:
        raise ValueError("noisy and clean features must share a feature dimension")
    if items == 0 or noisy_features.shape[1] == 0:
        raise ValueError("supervision must be nonempty")
    if not torch.isfinite(clean_features).all() or not torch.isfinite(noisy_features).all():
        raise ValueError("features must be finite")
    if int(labels.min()) < 0 or int(labels.max()) >= class_count:
        raise ValueError("labels must lie in [0, class_count)")
    counts = torch.bincount(labels, minlength=class_count)
    if bool((counts == 0).any()):
        raise ValueError("every class needs at least one supervision item")
    if shots_per_class is not None and int(counts.max()) > int(shots_per_class):
        raise ValueError(
            f"a class has {int(counts.max())} items but the few-shot budget is {shots_per_class}; "
            "evaluation items must never reach the prompt learner"
        )


def learn_context_prompts(
    text_encoder_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    token_embedding_fn: Callable[[Sequence[str]], TokenizedPrompts],
    class_names: Sequence[str],
    template: str,
    noisy_features: torch.Tensor,
    clean_features: torch.Tensor,
    labels: torch.Tensor,
    *,
    context_tokens: int = 4,
    steps: int,
    lr: float,
    seed: int,
    temperature: float = 0.05,
    clean_loss_weight: float = 0.5,
    shots_per_class: int | None = None,
    placeholder: str = DEFAULT_PLACEHOLDER,
    init_std: float = CONTEXT_INIT_STD,
) -> LearnedPromptBank:
    """Fit ``context_tokens`` shared context vectors and return the unit prototype bank.

    Loss at every step (full batch; registered control-role items only):
    ``CE(noisy @ T^T / temperature, labels) + clean_loss_weight * CE(clean @ T^T / temperature, labels)``
    where ``T`` is the current unit bank.  Deterministic for a fixed seed on a
    fixed device (context initialisation uses a seeded ``torch.Generator``;
    the optimiser is Adam, which has no randomness).
    """

    class_count = len(class_names)
    if class_count < 2:
        raise ValueError("at least two classes are required")
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not math.isfinite(float(lr)) or float(lr) <= 0.0:
        raise ValueError("lr must be finite and positive")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(float(clean_loss_weight)) or float(clean_loss_weight) < 0.0:
        raise ValueError("clean_loss_weight must be finite and nonnegative")
    _validate_supervision(noisy_features, clean_features, labels, class_count, shots_per_class)

    prompts = render_prompts(class_names, template, context_tokens=context_tokens, placeholder=placeholder)
    tokenized = token_embedding_fn(prompts)
    if not isinstance(tokenized, TokenizedPrompts):
        raise TypeError("token_embedding_fn must return TokenizedPrompts")
    embeddings = tokenized.embeddings.detach()
    if embeddings.ndim != 3 or embeddings.shape[0] != class_count:
        raise ValueError("token embeddings must be [classes, length, embed_dim]")
    start = int(tokenized.context_start)
    stop = start + context_tokens
    if start < 1 or stop > embeddings.shape[1] - 1:
        raise ValueError("context slots must lie strictly between SOT and EOT")
    eot_index = tokenized.eot_index.to(torch.long)
    if eot_index.shape != (class_count,) or int(eot_index.min()) < stop:
        raise ValueError("every EOT position must come after the context slots")
    slots = embeddings[:, start:stop]
    if not torch.equal(slots, slots[:1].expand_as(slots)):
        raise ValueError("context slots must hold the same placeholder embedding in every prompt")

    device, dtype = embeddings.device, embeddings.dtype
    prefix = embeddings[:, :start]
    suffix = embeddings[:, stop:]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    context = (torch.randn((context_tokens, embeddings.shape[2]), generator=generator, dtype=torch.float32)
               * float(init_std)).to(device=device, dtype=dtype)
    context.requires_grad_(True)
    optimizer = torch.optim.Adam([context], lr=float(lr))

    clean = clean_features.to(device=device, dtype=torch.float32)
    noisy = noisy_features.reshape(-1, noisy_features.shape[-1]).to(device=device, dtype=torch.float32)
    clean_labels = labels.to(device)
    noisy_labels = labels.repeat_interleave(int(noisy_features.shape[1])).to(device)
    scale = 1.0 / float(temperature)

    def bank() -> torch.Tensor:
        assembled = torch.cat([prefix, context.unsqueeze(0).expand(class_count, -1, -1), suffix], dim=1)
        features = text_encoder_fn(assembled, eot_index).to(torch.float32)
        if features.shape != (class_count, clean.shape[1]):
            raise ValueError("text_encoder_fn must return one feature row per class with the image dimension")
        return features / torch.linalg.vector_norm(features, dim=1, keepdim=True)

    def loss_of(current: torch.Tensor) -> torch.Tensor:
        value = torch.nn.functional.cross_entropy(noisy @ current.T * scale, noisy_labels)
        if float(clean_loss_weight) > 0.0:
            value = value + float(clean_loss_weight) * torch.nn.functional.cross_entropy(
                clean @ current.T * scale, clean_labels
            )
        return value

    history: list[float] = []
    for _ in range(steps):
        loss = loss_of(bank())
        if not torch.isfinite(loss):
            raise RuntimeError("prompt learning diverged")
        history.append(float(loss.detach()))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_bank = bank().detach().cpu().contiguous()
        final_loss = float(loss_of(final_bank.to(device)))

    return LearnedPromptBank(
        prototypes=final_bank,
        context_vectors=context.detach().cpu().clone(),
        prompts=prompts,
        initial_loss=history[0],
        final_loss=final_loss,
        loss_history=history,
        config={
            "context_tokens": int(context_tokens),
            "steps": int(steps),
            "lr": float(lr),
            "seed": int(seed),
            "temperature": float(temperature),
            "clean_loss_weight": float(clean_loss_weight),
            "shots_per_class": shots_per_class,
            "template": template,
            "placeholder": placeholder,
            "init_std": float(init_std),
            "optimizer": "adam_full_batch",
            "supervision": "registered_control_role_items_only",
            "final_test_access": False,
            "noisy_features_sha256": tensor_scientific_hash(noisy_features.detach().cpu().contiguous()),
            "clean_features_sha256": tensor_scientific_hash(clean_features.detach().cpu().contiguous()),
            "labels_sha256": tensor_scientific_hash(labels.detach().cpu().contiguous()),
            "prototypes_sha256": tensor_scientific_hash(final_bank),
        },
    )


# --------------------------------------------------------------------------- real path


def open_clip_prompt_hooks(model, tokenizer, *, device: str = "cpu"):
    """Build ``(text_encoder_fn, token_embedding_fn)`` for an open_clip CLIP model.

    The caller loads the model (``satml2027/common/models.py::load_model``) and
    passes ``loaded.model`` and ``loaded.tokenizer``; nothing is imported here.
    ``token_embedding_fn`` tokenizes with the open_clip tokenizer, takes the
    token embeddings and locates SOT (position 0), the context slots (1..C) and
    EOT (``argmax`` of the token ids, the open_clip pooling convention).
    ``text_encoder_fn`` mirrors ``CLIP.encode_text`` from the embedding layer on:
    positional embedding, transformer with the causal ``attn_mask``,
    ``ln_final``, EOT pooling, ``text_projection``.
    """

    def token_embedding_fn(prompts: Sequence[str]) -> TokenizedPrompts:
        tokens = tokenizer(list(prompts)).to(device)
        eot_index = tokens.argmax(dim=-1)
        with torch.no_grad():
            embeddings = model.token_embedding(tokens).to(torch.float32)
        return TokenizedPrompts(embeddings=embeddings, context_start=1, eot_index=eot_index)

    def text_encoder_fn(embeddings: torch.Tensor, eot_index: torch.Tensor) -> torch.Tensor:
        x = embeddings + model.positional_embedding.to(embeddings.dtype)
        attn_mask = getattr(model, "attn_mask", None)
        transformer = model.transformer
        if getattr(transformer, "batch_first", True):
            x = transformer(x, attn_mask=attn_mask)
        else:  # older open_clip: sequence-first transformer
            x = transformer(x.permute(1, 0, 2), attn_mask=attn_mask).permute(1, 0, 2)
        x = model.ln_final(x)
        pooled = x[torch.arange(x.shape[0], device=x.device), eot_index]
        projection = model.text_projection
        if projection is None:
            return pooled
        if isinstance(projection, torch.nn.Module):
            return projection(pooled)
        return pooled @ projection

    return text_encoder_fn, token_embedding_fn


def verify_hooks_against_encode_text(model, tokenizer, prompts: Sequence[str], *, device: str = "cpu",
                                     atol: float = 1e-4) -> float:
    """Return max |hooks(prompts) - model.encode_text(prompts)| and fail above ``atol``."""

    text_encoder_fn, token_embedding_fn = open_clip_prompt_hooks(model, tokenizer, device=device)
    tokenized = token_embedding_fn(prompts)
    with torch.no_grad():
        ours = text_encoder_fn(tokenized.embeddings, tokenized.eot_index).to(torch.float32)
        reference = model.encode_text(tokenizer(list(prompts)).to(device)).to(torch.float32)
    difference = float((ours - reference).abs().max())
    if difference > atol:
        raise RuntimeError(f"prompt hooks differ from model.encode_text by {difference:.3e} > {atol:.1e}")
    return difference
