"""Tests for satml2027ext.prompt_bank with a tiny deterministic fake encoder."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from satml2027ext.prompt_bank import (  # noqa: E402
    DEFAULT_TEMPLATE,
    LearnedPromptBank,
    TokenizedPrompts,
    learn_context_prompts,
    render_prompts,
)

CLASSES = ["apple", "bicycle", "cloud", "dolphin", "elephant", "forest"]
EMBED, FEATURE, LENGTH = 16, 24, 12


class FakeEncoder:
    """Word-hash token embeddings and a fixed nonlinear pooling head."""

    def __init__(self, seed: int = 11):
        generator = torch.Generator().manual_seed(seed)
        self.table = torch.randn(512, EMBED, generator=generator)
        self.projection = torch.randn(EMBED, FEATURE, generator=generator) / EMBED ** 0.5
        self.calls = 0

    @staticmethod
    def _word_id(word: str) -> int:
        return 3 + sum((index + 1) * ord(character) for index, character in enumerate(word)) % 509

    def token_embedding_fn(self, prompts):
        rows, eot = [], []
        for prompt in prompts:
            ids = [1] + [self._word_id(word) for word in prompt.split()] + [2]
            if len(ids) > LENGTH:
                raise ValueError("prompt too long for the fake encoder")
            eot.append(len(ids) - 1)
            ids = ids + [0] * (LENGTH - len(ids))
            rows.append(self.table[torch.tensor(ids)])
        return TokenizedPrompts(embeddings=torch.stack(rows), context_start=1, eot_index=torch.tensor(eot))

    def text_encoder_fn(self, embeddings, eot_index):
        self.calls += 1
        positions = torch.arange(embeddings.shape[1])[None, :]
        mask = (positions <= eot_index[:, None]).to(embeddings.dtype)[:, :, None]
        pooled = (torch.tanh(embeddings) * mask).sum(dim=1) / mask.sum(dim=1)
        return torch.tanh(pooled @ self.projection)


def supervision(seed: int = 5, shots: int = 5, draws: int = 4):
    generator = torch.Generator().manual_seed(seed)
    centers = torch.randn(len(CLASSES), FEATURE, generator=generator)
    labels = torch.arange(len(CLASSES)).repeat_interleave(shots)
    clean = centers[labels] + 0.3 * torch.randn(labels.shape[0], FEATURE, generator=generator)
    noisy = centers[labels][:, None, :] + 0.8 * torch.randn(labels.shape[0], draws, FEATURE, generator=generator)
    clean = clean / torch.linalg.vector_norm(clean, dim=-1, keepdim=True)
    noisy = noisy / torch.linalg.vector_norm(noisy, dim=-1, keepdim=True)
    return noisy, clean, labels


def learn(seed: int = 3, steps: int = 60, **overrides) -> LearnedPromptBank:
    encoder = FakeEncoder()
    noisy, clean, labels = supervision()
    settings = dict(context_tokens=4, steps=steps, lr=0.05, seed=seed, temperature=0.05,
                    clean_loss_weight=0.5, shots_per_class=5)
    settings.update(overrides)
    return learn_context_prompts(
        encoder.text_encoder_fn, encoder.token_embedding_fn, CLASSES, DEFAULT_TEMPLATE,
        noisy, clean, labels, **settings,
    )


def test_render_prompts_places_four_context_placeholders_before_the_class_name():
    prompts = render_prompts(CLASSES, DEFAULT_TEMPLATE, context_tokens=4)
    assert prompts[0] == "X X X X a photo of a apple."
    with pytest.raises(ValueError):
        render_prompts(CLASSES, "a photo of a {name}.", context_tokens=4)


def test_prompt_learner_returns_unit_rows_of_the_right_shape():
    result = learn()
    assert result.prototypes.shape == (len(CLASSES), FEATURE)
    assert result.prototypes.dtype == torch.float32
    norms = torch.linalg.vector_norm(result.prototypes.double(), dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6, rtol=0.0)
    assert result.context_vectors.shape == (4, EMBED)
    assert len(result.loss_history) == 60
    assert result.config["final_test_access"] is False


def test_prompt_learner_is_deterministic_for_a_seed_and_differs_across_seeds():
    first, second, other = learn(seed=3), learn(seed=3), learn(seed=4)
    assert torch.equal(first.prototypes, second.prototypes)
    assert torch.equal(first.context_vectors, second.context_vectors)
    assert first.loss_history == second.loss_history
    assert not torch.equal(first.prototypes, other.prototypes)


def test_prompt_learner_reduces_the_training_loss():
    result = learn(steps=80)
    assert result.final_loss < result.initial_loss
    assert result.final_loss < 0.9 * result.initial_loss
    assert min(result.loss_history) <= result.loss_history[0]


def test_prompt_learner_refuses_more_than_the_few_shot_budget():
    encoder = FakeEncoder()
    noisy, clean, labels = supervision(shots=6)
    with pytest.raises(ValueError, match="few-shot budget"):
        learn_context_prompts(
            encoder.text_encoder_fn, encoder.token_embedding_fn, CLASSES, DEFAULT_TEMPLATE,
            noisy, clean, labels, steps=2, lr=0.05, seed=1, shots_per_class=5,
        )


def test_prompt_learner_validates_encoder_contract_and_labels():
    encoder = FakeEncoder()
    noisy, clean, labels = supervision()
    with pytest.raises(ValueError, match="labels"):
        learn_context_prompts(encoder.text_encoder_fn, encoder.token_embedding_fn, CLASSES, DEFAULT_TEMPLATE,
                              noisy, clean, labels + 10, steps=2, lr=0.05, seed=1)

    def wrong_slots(prompts):
        tokenized = encoder.token_embedding_fn(prompts)
        tokenized.embeddings[0, 1] += 1.0  # context slot differs between prompts
        return tokenized

    with pytest.raises(ValueError, match="placeholder"):
        learn_context_prompts(encoder.text_encoder_fn, wrong_slots, CLASSES, DEFAULT_TEMPLATE,
                              noisy, clean, labels, steps=2, lr=0.05, seed=1)
