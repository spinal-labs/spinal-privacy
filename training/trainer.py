"""Trainer for the input-embedding transform against a frozen host model.

Each step runs the host twice on [prompt, reference answer]: once on the clean prompt
embeddings E (no gradient) and once on the transform's sampled output Z. The loss is

    utility_weight * KL(clean || protected)                     on real answer tokens
    + cosine_weight * |cos(E, Z)| + norm_weight * (|Z| - |E|)^2  on content positions
    + variance_weight * -log(sigma)                             on content positions

and only the transform's parameters are optimised (AdamW, gradient-norm clipping).
"""

from dataclasses import dataclass

import torch
from torch import nn

from privacy_transform import PrivacyTransform

from .losses import distillation, representation_losses


@dataclass
class TrainingConfig:
    """Optimiser settings and loss weights.

    The paper run sets learning_rate, utility_weight=1.0, cosine_weight=0.3, norm_weight=0.1 and
    variance_weight=0.001, and leaves the other fields at their defaults. kl_ema_decay sets the
    running average of the utility KL that step() reports as kl_ema; it does not affect training.
    """

    learning_rate: float = 3e-5
    temperature: float = 1.0
    utility_weight: float = 1.0
    cosine_weight: float = 0.1
    norm_weight: float = 0.1
    variance_weight: float = 0.001
    grad_clip: float = 1.0
    kl_ema_decay: float = 0.9


class Trainer:
    """Optimises the transform; the host model is frozen and kept in eval mode.

    The transform sees only the prompt. The reference answer is appended after the prompt as
    teacher-forced input for the utility loss and is never transformed.
    """

    def __init__(
        self, model: nn.Module, transform: PrivacyTransform, config: TrainingConfig | None = None
    ) -> None:
        self.config = config or TrainingConfig()
        self.model = model.eval().requires_grad_(False)
        self.answer_embeddings = model.get_input_embeddings()
        self.transform = transform.train()
        self.optimizer = torch.optim.AdamW(transform.parameters(), lr=self.config.learning_rate)
        decay = self.config.kl_ema_decay
        if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not 0 < decay < 1:
            raise ValueError('kl_ema_decay must be in (0, 1)')
        # Running average of the utility KL, a logged diagnostic. Python float: FP64.
        self.kl_ema = 0.0

    def step(
        self,
        prompt_embeddings: torch.Tensor,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        content_mask: torch.Tensor,
        answer_ids: torch.Tensor,
        answer_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Take one optimizer step on one batch and return the loss terms as floats.

        prompt_embeddings  [batch, prompt, dim] host input embeddings E of the prompts.
        prompt_ids         [batch, prompt] prompt token ids; the loss does not read them.
        attention_mask     [batch, prompt] real prompt tokens. All rows must have the same
                           length: the answer is appended after the full prompt width, so a
                           shorter right-padded row would take its first answer prediction
                           from a padding position. A ValueError is raised otherwise.
        content_mask       [batch, prompt] positions the transform may change.
        answer_ids         [batch, answer] reference-answer tokens.
        answer_mask        [batch, answer] real answer tokens. Callers that pad answers must
                           pass it: padded positions are then neither attended to nor scored.

        Returns utility_kl, cosine, norm, variance (unweighted loss terms), grad_norm_pre_clip,
        kl_logical (the utility KL that entered the running average), kl_ema and total.
        """
        cfg = self.config
        e = prompt_embeddings.detach().float()
        if answer_ids.ndim != 2 or answer_ids.shape[1] == 0:
            raise ValueError('nonempty reference-answer tokens required')
        self.transform.train()
        lengths = attention_mask.sum(1)
        if int(lengths.min()) != int(lengths.max()):
            raise ValueError(
                f'right-padded prompts of unequal length ({int(lengths.min())}..'
                f'{int(lengths.max())}) place the first answer prediction on padding for '
                'shorter rows. Left-pad or batch equal lengths.'
            )
        z = self.transform(e, attention_mask, content_mask, stochastic=True)
        with torch.no_grad():
            response = self.answer_embeddings(answer_ids).detach()
        model_dtype = next(self.model.parameters()).dtype
        valid = torch.ones_like(answer_ids, dtype=torch.bool) if answer_mask is None else answer_mask.bool()
        full_mask = torch.cat([attention_mask, valid], 1)
        # Logits from the last prompt position up to the second-to-last answer position
        # predict the answer tokens.
        start = e.shape[1] - 1
        with torch.no_grad():
            clean = self.model(
                inputs_embeds=torch.cat([e, response], 1).to(model_dtype),
                attention_mask=full_mask,
                use_cache=False,
            ).logits[:, start:-1]
        protected = self.model(
            inputs_embeds=torch.cat([z, response], 1).to(model_dtype),
            attention_mask=full_mask,
            use_cache=False,
        ).logits[:, start:-1]
        utility = distillation(clean, protected, valid, cfg.temperature)
        total = cfg.utility_weight * utility
        logged = {'utility_kl': float(utility.detach())}
        _, sigma = self.transform.distribution(e, attention_mask)
        aux = representation_losses(e, z, sigma, content_mask)
        total = total + (cfg.cosine_weight * aux['cosine'] + cfg.norm_weight * aux['norm'])
        total = total + cfg.variance_weight * aux['variance']
        logged.update({name: float(value.detach()) for name, value in aux.items()})
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.transform.parameters(), cfg.grad_clip, error_if_nonfinite=True
        )
        logged['grad_norm_pre_clip'] = float(norm)  # clip_grad_norm_ returns the pre-clip norm
        self.optimizer.step()
        kl_logical = logged['utility_kl']
        self.kl_ema = cfg.kl_ema_decay * self.kl_ema + (1.0 - cfg.kl_ema_decay) * kl_logical
        logged['kl_logical'] = kl_logical
        logged['kl_ema'] = self.kl_ema
        logged['total'] = float(total.detach())
        return logged
