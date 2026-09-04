"""Counterfactual impact weighting for clean-corrupt consistency."""

from __future__ import annotations

import torch
import torch.nn.functional as F


IMPACT_WEIGHTING = "per-sample-absolute-probability-change"


def counterfactual_impact_kl_from_logits(
    clean_logits: torch.Tensor, corrupt_logits: torch.Tensor
) -> torch.Tensor:
    """Focus clean-to-corrupt Bernoulli KL on affected forecast pixels."""

    if (
        clean_logits.shape != corrupt_logits.shape
        or clean_logits.numel() == 0
        or clean_logits.ndim < 2
    ):
        raise ValueError(
            "clean and corrupt logits must have the same nonempty batched shape"
        )
    teacher = clean_logits.detach()
    student = corrupt_logits
    teacher_probability = torch.sigmoid(teacher)
    impact = (
        teacher_probability - torch.sigmoid(student.detach())
    ).abs()
    reduce_dims = tuple(range(1, impact.ndim))
    mean_impact = impact.mean(dim=reduce_dims, keepdim=True)
    weight = torch.where(
        mean_impact > 0.0,
        impact / mean_impact.clamp_min(1e-6),
        torch.zeros_like(impact),
    )
    per_pixel = teacher_probability * (
        F.logsigmoid(teacher) - F.logsigmoid(student)
    ) + (1.0 - teacher_probability) * (
        F.logsigmoid(-teacher) - F.logsigmoid(-student)
    )
    return (weight * per_pixel).mean()
