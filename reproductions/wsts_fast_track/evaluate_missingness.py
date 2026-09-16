"""Shared checkpoint loading and metrics for retained T=1 evaluations."""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch



def evaluate_batches(
    model: torch.nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
) -> dict[str, int | float]:
    """Compute exact AP and streaming threshold metrics for one data population."""

    model = model.to(device)
    model.eval()
    score_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    true_positive = false_positive = false_negative = 0
    sample_count = pixel_count = 0
    weighted_loss = 0.0
    squared_probability_error = 0.0
    predictive_variance_total = 0.0
    has_predictive_variance = False

    with torch.inference_mode():
        for x, target in batches:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).long()
            uncertainty_forward = getattr(model, "forward_with_uncertainty", None)
            if callable(uncertainty_forward):
                logits, predictive_variance = uncertainty_forward(x)
                has_predictive_variance = True
            else:
                logits = model(x)
                predictive_variance = None
            if logits.ndim == target.ndim + 1 and logits.shape[1] == 1:
                logits = logits.squeeze(1)
            if (
                predictive_variance is not None
                and predictive_variance.ndim == target.ndim + 1
                and predictive_variance.shape[1] == 1
            ):
                predictive_variance = predictive_variance.squeeze(1)
            if logits.shape != target.shape:
                raise ValueError(
                    f"model output shape {tuple(logits.shape)} differs from target "
                    f"{tuple(target.shape)}"
                )
            if (
                predictive_variance is not None
                and predictive_variance.shape != target.shape
            ):
                raise ValueError("predictive variance shape differs from target")
            if not torch.isfinite(logits).all():
                raise ValueError("evaluation produced non-finite logits")
            if predictive_variance is not None and not torch.isfinite(predictive_variance).all():
                raise ValueError("evaluation produced non-finite predictive variance")
            loss = model.compute_loss(logits, target)
            if not torch.isfinite(loss).all():
                raise ValueError("evaluation produced non-finite loss")
            pixels = target.numel()
            weighted_loss += float(loss.detach().cpu()) * pixels
            probabilities = torch.sigmoid(logits)
            squared_probability_error += float(
                (probabilities - target.float()).square().sum().detach().cpu()
            )
            if predictive_variance is not None:
                predictive_variance_total += float(
                    predictive_variance.sum().detach().cpu()
                )
            pixel_count += pixels
            sample_count += target.shape[0]

            score_chunks.append(logits.detach().float().cpu().flatten())
            target_chunks.append(target.detach().bool().cpu().flatten())
            prediction = probabilities >= 0.5
            positive = target.bool()
            true_positive += int((prediction & positive).sum().item())
            false_positive += int((prediction & ~positive).sum().item())
            false_negative += int((~prediction & positive).sum().item())

    if sample_count == 0 or pixel_count == 0:
        raise ValueError("evaluation dataloader produced no samples")
    scores = torch.cat(score_chunks)
    targets = torch.cat(target_chunks)
    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    sorted_targets = targets[order].to(torch.float64)
    positive_count = int(sorted_targets.sum().item())
    if positive_count:
        cumulative_positive = torch.cumsum(sorted_targets, dim=0)
        threshold_ends = torch.ones_like(sorted_scores, dtype=torch.bool)
        threshold_ends[:-1] = sorted_scores[:-1] != sorted_scores[1:]
        end_indices = torch.nonzero(threshold_ends, as_tuple=False).flatten()
        true_positive_at_threshold = cumulative_positive[end_indices]
        precision_at_threshold = true_positive_at_threshold / (end_indices + 1)
        recall_at_threshold = true_positive_at_threshold / positive_count
        previous_recall = torch.cat(
            [torch.zeros(1, dtype=torch.float64), recall_at_threshold[:-1]]
        )
        ap = float(
            ((recall_at_threshold - previous_recall) * precision_at_threshold)
            .sum()
            .item()
        )
    else:
        ap = 0.0
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = (
        true_positive / precision_denominator if precision_denominator else 0.0
    )
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1_denominator = 2 * true_positive + false_positive + false_negative
    iou_denominator = true_positive + false_positive + false_negative
    metrics: dict[str, int | float] = {
        "sample_count": sample_count,
        "pixel_count": pixel_count,
        "avg_precision": ap,
        "f1": 2 * true_positive / f1_denominator if f1_denominator else 0.0,
        "iou": true_positive / iou_denominator if iou_denominator else 0.0,
        "precision": precision,
        "recall": recall,
        "loss": weighted_loss / pixel_count,
        "brier": squared_probability_error / pixel_count,
    }
    if has_predictive_variance:
        metrics["mean_predictive_variance"] = (
            predictive_variance_total / pixel_count
        )
    return metrics


def checkpoint_init_args(
    hyperparameters: Mapping[str, Any],
    *,
    experiment_id: str,
) -> dict[str, Any]:
    """Build constructor arguments compatible with the declared architecture."""

    init_args = dict(hyperparameters)
    init_args["encoder_weights"] = None
    if experiment_id not in {"C00", "C02"}:
        raise ValueError("retained checkpoint loader supports C00 and C02")
    if experiment_id == "C02":
        init_args.pop("use_doy", None)
    return init_args


def load_checkpoint_model(
    checkpoint: Path,
    *,
    experiment_id: str,
    upstream_root: Path,
    device: torch.device,
) -> torch.nn.Module:
    """Strict-load the retained C00 architecture from a checkpoint."""

    upstream = Path(upstream_root).resolve()
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "src"))
    if experiment_id not in {"C00", "C02"}:
        raise ValueError("retained checkpoint loader supports C00 and C02")
    class_name = "SMPModel" if experiment_id == "C00" else "SMPTempModel"
    model_class = getattr(importlib.import_module("models." + class_name), class_name)

    payload = torch.load(
        Path(checkpoint).resolve(strict=True), map_location="cpu", weights_only=False
    )
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    hyperparameters = payload.get("hyper_parameters")
    state_dict = payload.get("state_dict")
    if not isinstance(hyperparameters, dict) or not isinstance(state_dict, dict):
        raise ValueError("checkpoint lacks hyperparameters or state_dict")
    init_args = checkpoint_init_args(
        hyperparameters,
        experiment_id=experiment_id,
    )
    model = model_class(**init_args)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device)


def write_result_new(path: Path, payload: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
