"""Trainer and losses on CPU with the tiny GPT-2 host from conftest.py. No download.

One optimizer step must update the transform only: the frozen host keeps every weight bit for
bit. Padded answer positions must not change the loss. At identical logits the forward-KL
utility term is checked in float64, where only rounding residue is left.
"""

import math
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from privacy_transform import PrivacyTransform, TransformConfig
from training.losses import distillation
from training.trainer import Trainer, TrainingConfig

FP64_RESIDUE = 1e-15
PAPER_WEIGHTS = {'utility_weight': 1.0, 'cosine_weight': 0.3, 'norm_weight': 0.1, 'variance_weight': 0.001}
LOGGED_FLOATS = (
    'utility_kl',
    'cosine',
    'norm',
    'variance',
    'grad_norm_pre_clip',
    'kl_logical',
    'kl_ema',
    'total',
)


def small_transform(dim: int) -> PrivacyTransform:
    """The paper architecture with reduced dimensions, moved off the identity so the loss has a gradient."""
    torch.manual_seed(0)
    config = TransformConfig(embedding_dim=dim, hidden_dim=16, num_layers=2, num_heads=2, feedforward_dim=32)
    transform = PrivacyTransform(config)
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for parameter in transform.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.05)
    return transform


def prompt_batch(
    tiny_llm: nn.Module, seq: int = 8, answer_len: int = 4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Equal-length prompts and right-padded answers: the second answer has two padding positions."""
    torch.manual_seed(0)
    vocab = tiny_llm.config.vocab_size
    ids = torch.randint(0, vocab, (2, seq))
    attention = torch.ones_like(ids, dtype=torch.bool)
    content = torch.ones_like(ids, dtype=torch.bool)
    content[:, 0] = False  # a pass-through position
    answer = torch.randint(0, vocab, (2, answer_len))
    answer_mask = torch.ones_like(answer, dtype=torch.bool)
    answer_mask[1, -2:] = False
    e = tiny_llm.get_input_embeddings()(ids).detach().float()
    return e, ids, attention, content, answer, answer_mask


def snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    """A detached copy of every tensor in the module's state dict."""
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def test_one_step_updates_the_transform_and_not_the_host(tiny_llm: nn.Module) -> None:
    e, ids, attention, content, answer, answer_mask = prompt_batch(tiny_llm)
    transform = small_transform(e.shape[-1])
    trainer = Trainer(tiny_llm, transform, config=TrainingConfig(learning_rate=1e-3, **PAPER_WEIGHTS))

    trained = {id(p) for group in trainer.optimizer.param_groups for p in group['params']}
    assert trained == {id(p) for p in transform.parameters()}, 'the optimizer must hold only the transform'
    assert not any(p.requires_grad for p in tiny_llm.parameters())

    host_before, transform_before = snapshot(tiny_llm), snapshot(transform)
    logged = trainer.step(e, ids, attention, content, answer, answer_mask=answer_mask)

    assert set(logged) == set(LOGGED_FLOATS)
    assert all(math.isfinite(logged[name]) for name in LOGGED_FLOATS)
    assert logged['utility_kl'] > 0 and logged['grad_norm_pre_clip'] > 0
    assert logged['kl_logical'] == logged['utility_kl']

    assert all(p.grad is None for p in tiny_llm.parameters()), 'the host received a gradient'
    host_after = snapshot(tiny_llm)
    assert host_after.keys() == host_before.keys()
    for name, value in host_before.items():
        assert torch.equal(host_after[name], value), f'host weight {name} changed'

    grads = [p.grad for p in transform.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads), 'the transform received no gradient'
    after = snapshot(transform)
    changed = [name for name, value in transform_before.items() if not torch.equal(after[name], value)]
    assert 'mean_projection.weight' in changed, f'W2 did not move; changed: {changed}'


def test_padded_answer_logits_do_not_change_the_distillation_loss() -> None:
    """Arbitrary logits at masked-out answer positions leave the loss bit for bit unchanged."""
    generator = torch.Generator().manual_seed(2)
    clean = torch.randn(2, 5, 64, generator=generator)
    protected = torch.randn(2, 5, 64, generator=generator)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[0, -1] = False
    mask[1, -3:] = False
    reference = distillation(clean, protected, mask)

    clean_changed, protected_changed = clean.clone(), protected.clone()
    clean_changed[~mask] = 100.0 * torch.randn(int((~mask).sum()), 64, generator=generator)
    protected_changed[~mask] = -100.0 * torch.randn(int((~mask).sum()), 64, generator=generator)
    assert torch.equal(distillation(clean_changed, protected_changed, mask), reference)
    assert not torch.equal(distillation(clean_changed, protected_changed, torch.ones_like(mask)), reference)


def test_padded_answer_tokens_do_not_change_a_trainer_step(tiny_llm: nn.Module) -> None:
    """Different token ids at masked-out answer positions give the same step, bit for bit."""
    e, ids, attention, content, answer, answer_mask = prompt_batch(tiny_llm)
    other_answer = answer.clone()
    other_answer[~answer_mask] = (answer[~answer_mask] + 1) % tiny_llm.config.vocab_size

    results = []
    for answer_ids in (answer, other_answer):
        transform = small_transform(e.shape[-1])
        trainer = Trainer(tiny_llm, transform, config=TrainingConfig(learning_rate=1e-3, **PAPER_WEIGHTS))
        torch.manual_seed(3)  # the same sampled noise in both runs
        logged = trainer.step(e, ids, attention, content, answer_ids, answer_mask=answer_mask)
        results.append((logged, snapshot(transform)))

    (logged_a, state_a), (logged_b, state_b) = results
    assert logged_a == logged_b
    assert all(torch.equal(state_a[name], state_b[name]) for name in state_a)


def test_unequal_prompt_lengths_are_rejected(tiny_llm: nn.Module) -> None:
    """A right-padded shorter prompt would predict its first answer token from padding."""
    e, ids, attention, content, answer, answer_mask = prompt_batch(tiny_llm)
    attention[1, -1] = False
    trainer = Trainer(tiny_llm, small_transform(e.shape[-1]), config=TrainingConfig(**PAPER_WEIGHTS))
    with pytest.raises(ValueError, match='unequal length'):
        trainer.step(e, ids, attention, content, answer, answer_mask=answer_mask)


def test_forward_kl_at_identity_in_float64(tiny_llm: nn.Module) -> None:
    """Identical clean and protected logits: the forward KL is only fp64 rounding residue.

    In float64 softmax and log-softmax do not cancel to an exact zero, so the check is a
    bound far below float32 resolution, not equality with 0.0.
    """
    e, ids, attention, content, answer, answer_mask = prompt_batch(tiny_llm)
    with torch.no_grad():
        logits = tiny_llm(inputs_embeds=e, attention_mask=attention).logits
    mask = torch.ones(logits.shape[:2], dtype=torch.bool)
    kl = distillation(logits, logits.clone(), mask)
    assert math.isfinite(float(kl)) and abs(float(kl)) < FP64_RESIDUE
