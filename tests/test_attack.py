"""Offline check of the chosen-input subspace attack (CPU, tiny tensors, no Qwen weights).

With stochastic=False the transform moves each content embedding by W g(h) + b, where W is the
output projection (Eq. 2): the displacement lies in a rank-hidden_dim subspace plus a shift.
Nearest-neighbour decoding of Z fails, but estimating that subspace from the attacker's own deltas
Z - E and removing it from Z and from the vocabulary table, as scripts/attack.py does, recovers every
token.
"""

import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO))
import attack  # noqa: E402  (scripts/attack.py; importing it needs no GPU and no model)

import common  # noqa: E402
from privacy_transform.model import PrivacyTransform, TransformConfig  # noqa: E402

VOCAB, DIM, RANK = 64, 32, 4
Setup = tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor], torch.Tensor, list[torch.Tensor]]


@pytest.fixture(scope='module')
def setup() -> Setup:
    """Vocabulary table, the protect function, victim token ids and the attacker's own prompts."""
    generator = torch.Generator().manual_seed(0)
    table = torch.randn(VOCAB, DIM, generator=generator)
    torch.manual_seed(0)
    config = TransformConfig(
        embedding_dim=DIM,
        hidden_dim=RANK,
        num_layers=2,
        num_heads=2,
        feedforward_dim=16,
        sampling_space='output',
    )
    transform = PrivacyTransform(config).eval()
    with torch.no_grad():  # a trained-looking displacement: large, inside span(W), plus a shift
        transform.mean_projection.weight.normal_(0.0, 3.0, generator=generator)
        transform.mean_projection.bias.normal_(0.0, 3.0, generator=generator)

    @torch.no_grad()
    def protect(ids: torch.Tensor) -> torch.Tensor:
        e = table[ids][None]
        mask = torch.ones(e.shape[:2], dtype=torch.bool)
        return transform(e, mask, mask, stochastic=False)[0]

    victim = torch.randperm(VOCAB, generator=generator)[:48]
    attacker = [torch.randint(0, VOCAB, (16,), generator=generator) for _ in range(4)]
    return table, protect, victim, attacker


def test_nearest_neighbour_decoding_fails_on_protected_embeddings(setup: Setup) -> None:
    table, protect, victim, _ = setup
    assert torch.equal(common.nearest(table[victim], table), victim)  # clean input decodes exactly
    assert (common.nearest(protect(victim), table) != victim).any()
    zero = torch.zeros(DIM)
    assert not torch.equal(attack.project_nearest(protect(victim), table, None, zero), victim)


def test_chosen_input_subspace_recovers_all_tokens(setup: Setup) -> None:
    table, protect, victim, attacker = setup
    deltas = torch.cat([protect(ids) - table[ids] for ids in attacker])  # attacker knows E, sees Z
    basis, shift, energy = attack.displacement_subspace(deltas, RANK)
    assert energy > 1 - 1e-6  # all delta energy lies in RANK dimensions
    assert torch.equal(attack.project_nearest(protect(victim), table, basis, shift), victim)
