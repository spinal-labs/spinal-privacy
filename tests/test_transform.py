"""PrivacyTransform on CPU: identity at initialisation, pass-through positions, the strength
endpoints the package supplies, checkpoint round trips and the parameter count of the paper config.

Small configs use the tiny GPT-2 embedding table from conftest.py. No download.
"""

import io
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from privacy_transform import PrivacyTransform, TransformConfig

PAPER_PARAMETERS = 55_627_776


def small_config(dim: int) -> TransformConfig:
    """Same architecture as the paper, reduced so it runs in milliseconds."""
    return TransformConfig(embedding_dim=dim, hidden_dim=16, num_layers=2, num_heads=2, feedforward_dim=32)


def small_transform(dim: int) -> PrivacyTransform:
    """A seeded, untrained small transform in eval mode."""
    torch.manual_seed(0)
    return PrivacyTransform(small_config(dim)).eval()


def perturb(transform: PrivacyTransform, std: float = 0.05) -> PrivacyTransform:
    """Move every weight off its initial value, so the transform is no longer the identity."""
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for parameter in transform.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * std)
    return transform


def embeddings(embed_table: nn.Embedding, ids: torch.Tensor) -> torch.Tensor:
    """Host input embeddings E of the token ids, detached, in float32."""
    return embed_table(ids).detach().float()


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """torch.equal treats -0.0 and +0.0 as equal; this compares the bit patterns too."""
    return (
        a.dtype == b.dtype == torch.float32
        and torch.equal(a, b)
        and torch.equal(a.view(torch.int32), b.view(torch.int32))
    )


def test_untrained_transform_is_exactly_the_identity(
    embed_table: nn.Embedding, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """At initialisation W2 is zero, so Delta = 0 and the deterministic output is E, bit for bit."""
    ids, attention = batch
    e = embeddings(embed_table, ids)
    transform = small_transform(e.shape[-1])
    assert not transform.mean_projection.weight.any(), 'W2 must start at zero'
    content = attention.bool()
    z = transform(e, attention, content, stochastic=False)
    assert bitwise_equal(z, e)
    mean, _ = transform.distribution(e, attention)
    assert bitwise_equal(mean[content], e[content]), 'Delta must be exactly zero on content rows'


def test_only_output_space_sampling_is_supported() -> None:
    config = TransformConfig(
        embedding_dim=32, hidden_dim=16, num_layers=2, num_heads=2, sampling_space='latent'
    )
    with pytest.raises(ValueError, match='sampling_space'):
        PrivacyTransform(config)
    with pytest.raises(ValueError, match='sampling_space'):
        TransformConfig(sampling_space='latent').validate_target()


def test_strength_endpoints_supplied_by_the_package(
    embed_table: nn.Embedding, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """The strength scalar a is not part of this package.

    scripts/evaluate.py applies the blend (1 - a) * E + a * T(E). The package supplies
    the two endpoints, and this test checks what they need:
      a = 0: E itself. The transform must not change its input in place.
      a = 1: T(E), the deterministic output. It must be a fixed function of E, equal to
             the distribution mean on content rows, and not E itself.
    """
    ids, attention = batch
    e = embeddings(embed_table, ids)
    original = e.clone()
    transform = perturb(small_transform(e.shape[-1]))
    content = attention.bool()

    first = transform(e, attention, content, stochastic=False)
    second = transform(e, attention, content, stochastic=False)
    assert bitwise_equal(e, original), 'the transform modified E in place'
    assert bitwise_equal(first, second), 'the deterministic output must repeat bit for bit'
    mean, _ = transform.distribution(e, attention)
    assert bitwise_equal(first[content], mean[content])
    assert not torch.equal(first[content], e[content]), 'a perturbed transform must move E'

    torch.manual_seed(0)
    sampled = transform(e, attention, content, stochastic=True)
    assert not torch.equal(sampled[content], first[content]), 'sampling must add noise'


@pytest.mark.parametrize('stochastic', [False, True])
def test_pass_through_positions_are_unchanged(
    stochastic: bool, embed_table: nn.Embedding, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Positions outside the content mask, and padding, are returned as E, bit for bit."""
    ids, attention = batch  # row 1 ends in two padding positions
    e = embeddings(embed_table, ids)
    transform = perturb(small_transform(e.shape[-1]))
    content = torch.ones_like(attention, dtype=torch.bool)
    content[:, 0] = False  # e.g. a special token at the start
    content[0, 4] = False  # and one inside the prompt
    torch.manual_seed(0)
    z = transform(e, attention, content, stochastic=stochastic)

    passed = ~(content & attention.bool())
    assert passed.sum() == 5
    assert bitwise_equal(z[passed], e[passed]), 'pass-through positions were changed'
    changed = ~passed
    assert (z[changed] != e[changed]).any(-1).all(), 'every content position must move'


def test_content_mask_shape_is_checked(
    embed_table: nn.Embedding, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    ids, attention = batch
    e = embeddings(embed_table, ids)
    transform = small_transform(e.shape[-1])
    with pytest.raises(ValueError, match='content mask shape mismatch'):
        transform(e, attention, torch.ones(2, 3, dtype=torch.bool), stochastic=False)


def test_checkpoint_round_trip(embed_table: nn.Embedding, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
    """config_dict() and state_dict(), saved as scripts/train.py saves them, rebuild the same transform."""
    ids, attention = batch
    e = embeddings(embed_table, ids)
    transform = perturb(small_transform(e.shape[-1]))
    buffer = io.BytesIO()
    torch.save({'config': transform.config_dict(), 'state_dict': transform.state_dict()}, buffer)
    buffer.seek(0)
    state = torch.load(buffer, weights_only=True)

    torch.manual_seed(1)  # a different initialisation, fully overwritten by the state dict
    restored = PrivacyTransform(TransformConfig(**state['config'])).eval()
    restored.load_state_dict(state['state_dict'])
    assert restored.config == transform.config

    content = attention.bool()
    expected = transform(e, attention, content, stochastic=False)
    assert bitwise_equal(restored(e, attention, content, stochastic=False), expected)
    torch.manual_seed(0)
    expected_sample = transform(e, attention, content, stochastic=True)
    torch.manual_seed(0)
    assert bitwise_equal(restored(e, attention, content, stochastic=True), expected_sample)


def test_paper_config_parameter_count_and_identity() -> None:
    """d = 5120, r = 1024, 3 layers, 16 heads, feed-forward 4096, output-space sampling.

    About 220 MB of fp32 on CPU. The forward pass uses four positions only.
    """
    config = TransformConfig()
    assert (
        config.embedding_dim,
        config.hidden_dim,
        config.num_layers,
        config.num_heads,
        config.feedforward_dim,
        config.sampling_space,
    ) == (5120, 1024, 3, 16, 4096, 'output')
    config.validate_target()
    torch.manual_seed(0)
    transform = PrivacyTransform(config).eval()
    assert sum(p.numel() for p in transform.parameters()) == PAPER_PARAMETERS

    e = torch.randn(1, 4, 5120) * 0.02
    with torch.no_grad():
        z = transform(e, torch.ones(1, 4, dtype=torch.long), stochastic=False)
    assert bitwise_equal(z, e)
