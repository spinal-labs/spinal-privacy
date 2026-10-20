"""Shared fixtures: a tiny seeded GPT-2 host, so the suite runs offline on CPU in seconds."""

import pytest
import torch
from torch import nn

EMBED_DIM = 32
VOCAB = 256
SEQ = 8


@pytest.fixture(scope='session')
def tiny_llm() -> nn.Module:
    """A two-layer GPT-2 with seeded random weights, frozen and in eval mode."""
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=VOCAB,
        n_positions=64,
        n_embd=EMBED_DIM,
        n_layer=2,
        n_head=2,
        n_inner=64,
        bos_token_id=VOCAB - 1,
        eos_token_id=VOCAB - 1,
    )
    torch.manual_seed(0)
    model = GPT2LMHeadModel(config).eval()
    model.requires_grad_(False)
    return model


@pytest.fixture(scope='session')
def embed_table(tiny_llm: nn.Module) -> nn.Embedding:
    """The tiny host's input embedding table."""
    return tiny_llm.get_input_embeddings()


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Two prompts of SEQ token ids; the second ends in two padding positions."""
    torch.manual_seed(0)
    ids = torch.randint(0, VOCAB, (2, SEQ))
    mask = torch.ones_like(ids)
    mask[1, -2:] = 0
    return ids, mask
