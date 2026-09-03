"""Small validity-conditioned temporal primitives for the gated D3 idea."""

from __future__ import annotations

import torch


def masked_temporal_softmax(
    logits: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Normalize head-wise temporal logits over valid observations only."""

    if logits.ndim != 5:
        raise ValueError("temporal logits must have shape (heads, B, T, H, W)")
    expected = (logits.shape[1], logits.shape[2], logits.shape[3], logits.shape[4])
    if valid.shape != expected or valid.dtype is not torch.bool:
        raise ValueError("validity must be boolean with shape (B, T, H, W)")
    mask = valid.unsqueeze(0)
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(masked_logits, dim=2)
    weights = torch.where(mask, weights, torch.zeros_like(weights))
    weights = torch.nan_to_num(weights, nan=0.0)
    total = weights.sum(dim=2, keepdim=True)
    return torch.where(total > 0, weights / total.clamp_min(1e-12), weights)
