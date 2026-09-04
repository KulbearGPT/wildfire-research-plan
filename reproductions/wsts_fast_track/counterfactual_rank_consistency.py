"""Listwise spatial-risk consistency for clean-corrupt forecasts."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


RANK_FORMULATION = "normalized-spatial-jensen-shannon"
RANK_TEMPERATURE = 1.0


def normalized_spatial_js_from_logits(
    clean_logits: torch.Tensor, corrupt_logits: torch.Tensor
) -> torch.Tensor:
    """Return bounded JS divergence between per-sample spatial risk ranks."""

    if (
        clean_logits.shape != corrupt_logits.shape
        or clean_logits.numel() == 0
        or clean_logits.ndim < 2
    ):
        raise ValueError(
            "clean and corrupt logits must have the same nonempty batched shape"
        )
    teacher_log_probability = F.log_softmax(
        clean_logits.detach().flatten(start_dim=1) / RANK_TEMPERATURE,
        dim=1,
    )
    student_log_probability = F.log_softmax(
        corrupt_logits.flatten(start_dim=1) / RANK_TEMPERATURE,
        dim=1,
    )
    teacher_probability = teacher_log_probability.exp()
    student_probability = student_log_probability.exp()
    mixture = 0.5 * (teacher_probability + student_probability)
    mixture_log_probability = mixture.clamp_min(
        torch.finfo(mixture.dtype).tiny
    ).log()
    teacher_kl = (
        teacher_probability
        * (teacher_log_probability - mixture_log_probability)
    ).sum(dim=1)
    student_kl = (
        student_probability
        * (student_log_probability - mixture_log_probability)
    ).sum(dim=1)
    return (0.5 * (teacher_kl + student_kl)).mean() / math.log(2.0)
