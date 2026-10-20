"""Training losses: forward-KL distillation on answer tokens and the auxiliary representation terms."""

import torch
from torch.nn import functional as F


def distillation(
    clean_logits: torch.Tensor,
    protected_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 16,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Forward KL(clean || protected), averaged over the masked positions and returned in float32.

    clean_logits and protected_logits are [batch, positions, vocab]; mask is [batch, positions]
    and selects the real answer tokens, so padded positions never enter the loss. Positions are
    processed in chunks of chunk_size to bound memory over a large vocabulary.

    The KL is evaluated in FP64 by default. In FP32, softmax and log-softmax over the full
    vocabulary do not cancel exactly for bitwise-identical logits: the loss reads about 1.4e-8
    instead of 0 and has a nonzero gradient. AdamW normalises that residue into a full-size step,
    walking the transform off an exact optimum. In FP64 the same quantity is about 3.6e-17.
    """
    if temperature <= 0 or not mask.any():
        raise ValueError('positive temperature and nonempty utility mask required')
    clean = clean_logits.reshape(-1, clean_logits.shape[-1])
    protected = protected_logits.reshape_as(clean)
    selected = mask.flatten().nonzero().flatten()
    total = protected.flatten()[0].to(dtype) * 0
    for indexes in selected.split(chunk_size):
        target = F.softmax(clean[indexes].detach().to(dtype) / temperature, -1)
        logp = F.log_softmax(protected[indexes].to(dtype) / temperature, -1)
        total = total + F.kl_div(logp, target, reduction='sum') * temperature**2
    return (total / selected.numel()).float()


def representation_losses(
    e: torch.Tensor, z: torch.Tensor, sigma: torch.Tensor, mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Auxiliary terms, averaged over content positions. None measures whether E is recoverable.

    cosine    |cos(E, Z)|: pushes Z away from E's direction. A rotation scores perfectly and
              leaks everything, so a low value is not evidence of protection.
    norm      (|Z| - |E|)^2: holds the vector length. Says nothing about content.
    variance  -log(sigma). Minimising this maximises sigma: it rewards more noise and acts as a
              stochasticity term, not a variance penalty despite the name. The paper weights it
              at 0.001 against a utility weight of 1.0.
    """
    if not mask.any():
        raise ValueError('nonempty content mask required')
    clean, sent = e[mask].float(), z[mask].float()
    return {
        'cosine': F.cosine_similarity(clean, sent, dim=-1).abs().mean(),
        'norm': ((sent.norm(dim=-1) - clean.norm(dim=-1)) ** 2).mean(),
        'variance': -sigma[mask].float().clamp_min(1e-12).log().mean(),
    }
