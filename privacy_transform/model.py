"""The input-embedding transform: a bottleneck context encoder that adds a learned
displacement and diagonal Gaussian noise to the host model's prompt embeddings."""

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TransformConfig:
    """Architecture of the transform. The defaults are the Qwen3.8-27B configuration.

    preserve_special_tokens is recorded in checkpoints only; the transform does not read it.
    The caller keeps special tokens unchanged by leaving them out of the content mask.

    sampling_space is kept so saved configs load unchanged. Only 'output' is supported: the
    noise is a diagonal Gaussian in the host's embedding space.
    """

    embedding_dim: int = 5120
    hidden_dim: int = 1024
    num_layers: int = 3
    num_heads: int = 16
    feedforward_dim: int = 4096
    dropout: float = 0.0
    initial_sigma: float = 0.001
    preserve_special_tokens: bool = True
    sampling_space: str = 'output'

    def validate_target(self) -> None:
        """Raise ValueError unless this config fits Qwen3.8-27B (5120 -> 1024 -> 5120, 2-4 blocks)."""
        if self.embedding_dim != 5120 or self.hidden_dim != 1024:
            raise ValueError('Qwen3.8-27B requires the 5120 -> 1024 -> 5120 architecture')
        if not 2 <= self.num_layers <= 4:
            raise ValueError('The transform requires 2-4 contextual blocks')
        if self.sampling_space != 'output':
            raise ValueError("sampling_space must be 'output'")


class PrivacyTransform(nn.Module):
    """Maps prompt embeddings E to Z = E + Delta + sigma * noise on content positions.

    E is projected to hidden_dim, given sinusoidal positions and passed through a pre-norm
    Transformer encoder (h). The displacement is Delta = W2 gelu(mean_head(h)) and the noise
    scale is sigma = softplus(noise_projection(gelu(noise_head(h)))). W2 and the noise
    projection weights start at zero, so an untrained transform returns E exactly and its
    noise scale starts at initial_sigma. Reduced dimensions are for unit tests only.
    """

    def __init__(self, config: TransformConfig | None = None) -> None:
        """Build the transform; config defaults to TransformConfig(), the Qwen3.8-27B configuration."""
        super().__init__()
        config = TransformConfig() if config is None else config
        if config.initial_sigma <= 0:
            raise ValueError('initial_sigma must be positive')
        if config.sampling_space != 'output':
            raise ValueError("sampling_space must be 'output'")
        self.config = config
        d, h = config.embedding_dim, config.hidden_dim
        # Construction order fixes the random initialisation; keep it unchanged.
        self.input_projection = nn.Linear(d, h)
        block = nn.TransformerEncoderLayer(
            h, config.num_heads, config.feedforward_dim, config.dropout, batch_first=True, norm_first=True
        )
        self.context = nn.TransformerEncoder(block, config.num_layers, enable_nested_tensor=False)
        self.mean_head = nn.Linear(h, h)
        self.noise_head = nn.Linear(h, h)
        self.mean_projection = nn.Linear(h, d)
        nn.init.zeros_(self.mean_projection.bias)
        self.noise_projection = nn.Linear(h, d)
        nn.init.zeros_(self.mean_projection.weight)
        nn.init.zeros_(self.noise_projection.weight)
        nn.init.constant_(self.noise_projection.bias, math.log(math.expm1(config.initial_sigma)))

    def distribution(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the mean E + Delta and the noise scale sigma for every position, in float32.

        embeddings is [batch, sequence, embedding_dim]; attention_mask marks real tokens.
        No mask is applied to the result: forward() keeps non-content positions as E.
        """
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.config.embedding_dim:
            raise ValueError('expected [batch, sequence, embedding_dim]')
        h = self.input_projection(embeddings.to(self.input_projection.weight.dtype))
        # Position information is required for contextual, order-sensitive changes.
        width = h.shape[-1]
        positions = torch.arange(h.shape[1], device=h.device, dtype=torch.float32)[:, None]
        frequencies = torch.exp(
            torch.arange(0, width, 2, device=h.device).float() * (-math.log(10000.0) / width)
        )
        encoding = torch.zeros(h.shape[1:], device=h.device, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[-1]])
        padding = None if attention_mask is None else ~attention_mask.bool()
        if padding is not None and (padding.all(-1).any() or padding.shape != h.shape[:2]):
            raise ValueError('attention mask must match and contain a token in each sequence')
        h = self.context(h + encoding.to(h.dtype), src_key_padding_mask=padding)
        delta = self.mean_projection(F.gelu(self.mean_head(h)))
        sigma = F.softplus(self.noise_projection(F.gelu(self.noise_head(h))).float())
        return embeddings.float() + delta.float(), sigma

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        content_mask: torch.Tensor | None = None,
        stochastic: bool = True,
    ) -> torch.Tensor:
        """Return Z in float32. Positions outside attention_mask or content_mask are E unchanged.

        stochastic=True samples Z = mean + sigma * noise (training). stochastic=False returns the
        mean (evaluation and the attack).
        """
        mean, sigma = self.distribution(embeddings, attention_mask)
        z = mean + sigma * torch.randn_like(sigma) if stochastic else mean
        transformed = torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.bool)
        if attention_mask is not None:
            transformed &= attention_mask.bool()
        if content_mask is not None:
            if content_mask.shape != transformed.shape:
                raise ValueError('content mask shape mismatch')
            transformed &= content_mask.bool()
        return torch.where(transformed[..., None], z, embeddings.float())

    def config_dict(self) -> dict[str, int | float | bool | str]:
        """The config as a plain dict, as stored in checkpoints next to the state dict."""
        return asdict(self.config)
