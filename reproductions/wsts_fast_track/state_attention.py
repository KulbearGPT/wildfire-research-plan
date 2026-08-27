"""Prior-token temporal attention for missing active-fire state inference."""

from __future__ import annotations

import math

import torch

from .latent_state_common import FrozenP00BeliefModel, PROCESSED_FEATURES


class TemporalAttentionBeliefState(FrozenP00BeliefModel):
    """Infer the latest fire state from temporal evidence and a context prior."""

    def __init__(self, default_model: torch.nn.Module, history: int) -> None:
        super().__init__(default_model, history)
        self.context_projection = torch.nn.Conv2d(34, 16, kernel_size=1)
        self.fire_projection = torch.nn.Conv2d(3, 16, kernel_size=1)
        self.prior_projection = torch.nn.Conv2d(34, 16, kernel_size=1)
        self.spatial_operator = torch.nn.Sequential(
            torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16),
            torch.nn.GroupNorm(4, 16),
            torch.nn.SiLU(),
        )
        self.state_head = torch.nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def trainable_parameters(self):
        return (
            *self.context_projection.parameters(),
            *self.fire_projection.parameters(),
            *self.prior_projection.parameters(),
            *self.spatial_operator.parameters(),
            *self.state_head.parameters(),
        )

    def _validate_inputs(
        self, features: torch.Tensor, reliability: torch.Tensor
    ) -> None:
        if (
            not isinstance(features, torch.Tensor)
            or features.ndim != 5
            or features.shape[1] != self.history
            or features.shape[2] != PROCESSED_FEATURES
            or features.shape[-2] <= 0
            or features.shape[-1] <= 0
        ):
            raise ValueError("features must have shape [B,T,40,H,W] for this history")
        if (
            not isinstance(reliability, torch.Tensor)
            or reliability.ndim != 5
            or reliability.shape[2] != 1
            or reliability.shape[:2] != features.shape[:2]
            or reliability.shape[-2:] != features.shape[-2:]
            or reliability.device != features.device
        ):
            raise ValueError("reliability must have shape [B,T,1,H,W] matching features")

    def _tokens_and_weights(
        self, features: torch.Tensor, reliability: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(features, reliability)
        batch, steps, _, height, width = features.shape
        reliability = reliability.to(dtype=features.dtype)
        lag = torch.arange(steps, device=features.device, dtype=features.dtype)
        lag = (lag - (steps - 1)) / 4.0
        lag = lag.view(1, steps, 1, 1, 1).expand(batch, steps, 1, height, width)

        context = torch.cat((features[:, :, :33], lag), dim=2)
        context_tokens = self.context_projection(context.reshape(batch * steps, 34, height, width))
        context_tokens = context_tokens.reshape(batch, steps, 16, height, width)
        fire_evidence = torch.cat((features[:, :, 38:40], reliability), dim=2)
        fire_tokens = self.fire_projection(
            fire_evidence.reshape(batch * steps, 3, height, width)
        ).reshape(batch, steps, 16, height, width)
        evidence_tokens = context_tokens + fire_tokens * reliability
        evidence_tokens = self.spatial_operator(
            evidence_tokens.reshape(batch * steps, 16, height, width)
        ).reshape(batch, steps, 16, height, width)

        prior_lag = features.new_zeros((batch, 1, height, width))
        prior = self.prior_projection(torch.cat((features[:, -1, :33], prior_lag), dim=1))
        prior = self.spatial_operator(prior)
        tokens = torch.cat((evidence_tokens, prior[:, None]), dim=1)
        scores = (tokens * prior[:, None]).sum(dim=2) / math.sqrt(16)
        weights = torch.softmax(scores, dim=1)
        attended = (weights[:, :, None] * tokens).sum(dim=1)
        return attended, weights

    def attention_weights(
        self, features: torch.Tensor, reliability: torch.Tensor
    ) -> torch.Tensor:
        """Recompute pixelwise attention over all evidence tokens and the prior."""

        _, weights = self._tokens_and_weights(features, reliability)
        return weights

    def infer_state_logits(
        self, features: torch.Tensor, reliability: torch.Tensor
    ) -> torch.Tensor:
        attended, _ = self._tokens_and_weights(features, reliability)
        return self.state_head(attended)
