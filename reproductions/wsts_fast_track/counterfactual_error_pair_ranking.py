"""Rank label-confirmed errors selected by a clean-corrupt intervention."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


PAIR_FORMULATION = "counterfactual-error-softplus"
PAIR_TOP_K = 64


def counterfactual_error_pair_ranking_from_logits(
    clean_logits: torch.Tensor,
    corrupt_logits: torch.Tensor,
    target: torch.Tensor,
    corrupted_sample: torch.Tensor,
    *,
    top_k: int = PAIR_TOP_K,
) -> torch.Tensor:
    """Rank corruption-depressed positives above corruption-inflated negatives."""

    if (
        clean_logits.shape != corrupt_logits.shape
        or clean_logits.shape != target.shape
        or clean_logits.numel() == 0
        or clean_logits.ndim < 2
        or corrupted_sample.shape != (clean_logits.shape[0],)
        or top_k <= 0
    ):
        raise ValueError("CEPR inputs do not satisfy the fixed batched contract")
    if not torch.all((target == 0) | (target == 1)):
        raise ValueError("CEPR target must be binary")

    clean_probability = torch.sigmoid(clean_logits.detach())
    corrupt_probability = torch.sigmoid(corrupt_logits.detach())
    flat_clean_probability = clean_probability.flatten(start_dim=1)
    flat_corrupt_probability = corrupt_probability.flatten(start_dim=1)
    flat_corrupt_logits = corrupt_logits.flatten(start_dim=1)
    flat_target = target.flatten(start_dim=1).bool()
    sample_losses: list[torch.Tensor] = []
    for sample_index in range(clean_logits.shape[0]):
        if not bool(corrupted_sample[sample_index]):
            continue
        positive_mask = flat_target[sample_index]
        negative_mask = ~positive_mask
        positive_drop = (
            flat_clean_probability[sample_index]
            - flat_corrupt_probability[sample_index]
        ).clamp_min(0.0)
        negative_rise = (
            flat_corrupt_probability[sample_index]
            - flat_clean_probability[sample_index]
        ).clamp_min(0.0)
        positive_indices = torch.nonzero(
            positive_mask & (positive_drop > 0.0), as_tuple=False
        ).flatten()
        negative_indices = torch.nonzero(
            negative_mask & (negative_rise > 0.0), as_tuple=False
        ).flatten()
        if positive_indices.numel() == 0 or negative_indices.numel() == 0:
            continue
        positive_count = min(top_k, positive_indices.numel())
        negative_count = min(top_k, negative_indices.numel())
        positive_selection = positive_indices[
            torch.topk(
                positive_drop[positive_indices], k=positive_count, sorted=False
            ).indices
        ]
        negative_selection = negative_indices[
            torch.topk(
                negative_rise[negative_indices], k=negative_count, sorted=False
            ).indices
        ]
        positive_logits = flat_corrupt_logits[sample_index, positive_selection]
        negative_logits = flat_corrupt_logits[sample_index, negative_selection]
        pair_loss = F.softplus(
            negative_logits[:, None] - positive_logits[None, :]
        ).mean() / math.log(2.0)
        sample_losses.append(pair_loss)

    if not sample_losses:
        return corrupt_logits.sum() * 0.0
    return torch.stack(sample_losses).mean()
