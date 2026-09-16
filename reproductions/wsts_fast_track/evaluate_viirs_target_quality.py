"""Audit target-label censoring on the fixed 24-sample VIIRS gate."""

from __future__ import annotations

import argparse
import importlib
import math
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .diagnostic_artifacts import verified_reliability_path
from .contract import validate_inventory
from .runtime import _install_runtime_contract, load_training_stats
from .diagnostic_support import resolve_dataset_index
from .evaluation import _center_crop_last_two
from .evaluate_belief_state import (
    LatestDayP00,
    _build_evaluation_dataset,
    _load_validated_checkpoint,
)
from .evaluate_missingness import load_checkpoint_model, write_result_new
from .diagnostic_support import _checkpoint_from_record
from .latent_state_common import masked_unweighted_focal
from .latent_state_common import load_trainable_state_dict
from .diagnostic_support import FIRE_DROPOUT_PROBABILITY, PROTOTYPE_ID as P00_ID
from .train_belief_state import build_model
from .viirs_reliability import FIXED_2021_MODEL_GATE, FIXED_2021_TARGET_GATE
from .evaluate_viirs_reliability import NaturalReliabilityDataset


def target_observation_mask(
    target: torch.Tensor, reliability: torch.Tensor
) -> torch.Tensor:
    """Keep every positive and only reliability-supported zero labels."""

    if target.shape != reliability.shape:
        raise ValueError("target and target reliability must have equal shapes")
    return target.bool() | reliability.bool()


def binary_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    include: torch.Tensor | None = None,
) -> dict[str, int | float]:
    """Compute binary forecast metrics over one explicit pixel population."""

    if logits.shape != target.shape:
        raise ValueError("logits and target must have equal shapes")
    if include is None:
        include = torch.ones_like(target, dtype=torch.bool)
    if include.shape != target.shape:
        raise ValueError("include mask must have the same shape as target")
    selected = include.bool()
    pixel_count = int(selected.sum().item())
    if pixel_count == 0:
        raise ValueError("binary metrics require at least one included pixel")

    scores = logits[selected].detach().to(dtype=torch.float64, device="cpu")
    labels = target[selected].detach().bool().to(device="cpu")
    probabilities = torch.sigmoid(scores)
    predictions = probabilities >= 0.5
    true_positive = int((predictions & labels).sum().item())
    false_positive = int((predictions & ~labels).sum().item())
    false_negative = int((~predictions & labels).sum().item())

    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    sorted_targets = labels[order].to(torch.float64)
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
            (torch.zeros(1, dtype=torch.float64), recall_at_threshold[:-1])
        )
        average_precision = float(
            ((recall_at_threshold - previous_recall) * precision_at_threshold)
            .sum()
            .item()
        )
    else:
        average_precision = 0.0

    f1_denominator = 2 * true_positive + false_positive + false_negative
    iou_denominator = true_positive + false_positive + false_negative
    focal = masked_unweighted_focal(logits, target, selected)
    return {
        "pixel_count": pixel_count,
        "positive_count": positive_count,
        "avg_precision": average_precision,
        "f1": 2 * true_positive / f1_denominator if f1_denominator else 0.0,
        "iou": true_positive / iou_denominator if iou_denominator else 0.0,
        "precision": (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        ),
        "recall": (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        ),
        "loss": float(focal.detach().cpu()),
        "brier": float(((probabilities - labels.float()).square()).mean().item()),
    }


def risk_coverage(
    records: Sequence[Mapping[str, float]],
    *,
    quality_key: str,
    error_key: str,
) -> list[dict[str, int | float]]:
    """Report mean error after retaining the highest-quality event fractions."""

    if not records:
        raise ValueError("risk coverage requires at least one event")
    ordered = sorted(records, key=lambda record: float(record[quality_key]), reverse=True)
    points: list[dict[str, int | float]] = []
    for coverage in (0.25, 0.5, 0.75, 1.0):
        count = max(1, math.ceil(len(ordered) * coverage))
        errors = [float(record[error_key]) for record in ordered[:count]]
        if not all(math.isfinite(error) for error in errors):
            raise ValueError("risk coverage errors must be finite")
        points.append(
            {
                "coverage": coverage,
                "sample_count": count,
                "mean_error": sum(errors) / count,
            }
        )
    return points


class TargetQualityDataset(Dataset[Any]):
    """Attach the next-day VIIRS reliability crop to the fixed input gate."""

    def __init__(self, input_dataset: Any, target_root: Path) -> None:
        if len(input_dataset) != len(FIXED_2021_MODEL_GATE):
            raise ValueError("target-quality audit requires the fixed 24 input samples")
        input_identities = tuple(
            (event, day) for event, day, _index, _record in input_dataset.samples
        )
        if input_identities != FIXED_2021_MODEL_GATE:
            raise ValueError("input identities differ from the fixed model gate")

        self.input = input_dataset
        self.root = Path(target_root).resolve()
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        records = manifest.get("samples")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("gate") != "target"
            or not isinstance(records, list)
            or len(records) != len(FIXED_2021_TARGET_GATE)
        ):
            raise ValueError("target reliability manifest is not the fixed target gate")
        by_identity = {
            (record.get("event"), record.get("day")): record
            for record in records
            if isinstance(record, dict)
        }
        if set(by_identity) != set(FIXED_2021_TARGET_GATE):
            raise ValueError("target reliability identities differ from the fixed target gate")
        self.samples = tuple(
            (
                event,
                model_day,
                target_day,
                input_dataset.samples[position][3],
                by_identity[(event, target_day)],
            )
            for position, ((event, model_day), (_, target_day)) in enumerate(
                zip(FIXED_2021_MODEL_GATE, FIXED_2021_TARGET_GATE, strict=True)
            )
        )
        self._reliability_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Any, torch.Tensor, int]:
        packed, target = self.input[index]
        record = self.samples[index][4]
        relative = str(record["output"])
        if relative not in self._reliability_cache:
            path = verified_reliability_path(self.root, record)
            with np.load(path, allow_pickle=False) as payload:
                self._reliability_cache[relative] = payload["reliability"].copy()
        full = self._reliability_cache[relative]
        if target.shape[-2] != target.shape[-1]:
            raise ValueError("target-quality audit requires a square target crop")
        cropped = _center_crop_last_two(full, target.shape[-1])
        reliability = torch.as_tensor(cropped, dtype=torch.uint8)
        return packed, target, reliability, index


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p00-record", type=Path, required=True)
    parser.add_argument("--artifact-map", type=Path, help="checksum-verified relocation map for historical artifact paths")
    parser.add_argument("--attention-checkpoint", type=Path, required=True)
    parser.add_argument("--input-reliability-root", type=Path, required=True)
    parser.add_argument("--target-reliability-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    p00_record = args.p00_record.resolve(strict=True)
    p00_checkpoint = _checkpoint_from_record(
        p00_record,
        prototype_id=P00_ID,
        artifact_map=args.artifact_map,
        training_policy={
            "active_fire_dropout_probability": FIRE_DROPOUT_PROBABILITY,
            "training_only": True,
        },
    )
    attention_checkpoint = args.attention_checkpoint.resolve(strict=True)
    checkpoint = _load_validated_checkpoint(
        attention_checkpoint,
        p00_record=p00_record,
        p00_checkpoint=p00_checkpoint,
        artifact_map=args.artifact_map,
    )
    if checkpoint["method"] != "attention" or checkpoint["history"] != 1:
        raise ValueError("target-quality audit requires attention T=1")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA evaluation requested but unavailable")

    upstream = args.upstream_root.resolve()
    data_root = args.data_root.resolve()
    validate_inventory(data_root)
    stats = load_training_stats(args.stats_path)
    _install_runtime_contract(upstream, "C00", stats)
    dataset_class = importlib.import_module(
        "dataloader.FireSpreadDataset"
    ).FireSpreadDataset
    dataset_class.find_image_index_from_dataset_index = resolve_dataset_index
    standard = _build_evaluation_dataset(
        dataset_class, data_root=data_root, history=1, scenario_id="M00"
    )
    input_dataset = NaturalReliabilityDataset(
        standard, args.input_reliability_root
    )
    dataset = TargetQualityDataset(input_dataset, args.target_reliability_root)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    default_model = load_checkpoint_model(
        p00_checkpoint,
        experiment_id="C00",
        upstream_root=upstream,
        device=device,
    )
    attention = build_model("attention", default_model, 1).to(device)
    load_trainable_state_dict(attention, checkpoint["model_state"])
    if (
        sum(parameter.numel() for parameter in attention.trainable_parameters())
        != checkpoint["trainable_parameters"]
    ):
        raise ValueError("attention parameter count differs from checkpoint")
    models = {
        "p00": LatestDayP00(default_model).to(device),
        "attention": attention,
    }

    model_logits: dict[str, torch.Tensor] = {}
    reference_targets: torch.Tensor | None = None
    reference_reliability: torch.Tensor | None = None
    reference_indices: torch.Tensor | None = None
    for name, model in models.items():
        model.eval()
        logits_chunks: list[torch.Tensor] = []
        target_chunks: list[torch.Tensor] = []
        reliability_chunks: list[torch.Tensor] = []
        index_chunks: list[torch.Tensor] = []
        with torch.inference_mode():
            for packed, target, target_reliability, sample_index in loader:
                logits = model(packed.to(device, non_blocking=True))
                if logits.ndim == target.ndim + 1 and logits.shape[1] == 1:
                    logits = logits.squeeze(1)
                if logits.shape != target.shape:
                    raise ValueError("model output shape differs from target")
                logits_chunks.append(logits.detach().float().cpu())
                target_chunks.append(target.long().cpu())
                reliability_chunks.append(target_reliability.to(torch.uint8).cpu())
                index_chunks.append(sample_index.long().cpu())
        combined_targets = torch.cat(target_chunks)
        combined_reliability = torch.cat(reliability_chunks)
        combined_indices = torch.cat(index_chunks)
        if reference_targets is None:
            reference_targets = combined_targets
            reference_reliability = combined_reliability
            reference_indices = combined_indices
        elif not (
            torch.equal(reference_targets, combined_targets)
            and torch.equal(reference_reliability, combined_reliability)
            and torch.equal(reference_indices, combined_indices)
        ):
            raise ValueError("model passes saw different target-quality populations")
        model_logits[name] = torch.cat(logits_chunks)

    if (
        reference_targets is None
        or reference_reliability is None
        or reference_indices is None
        or reference_indices.tolist() != list(range(len(dataset)))
    ):
        raise ValueError("target-quality dataloader population is incomplete")
    include = target_observation_mask(reference_targets, reference_reliability)
    metrics = {
        name: {
            "standard": binary_metrics(logits, reference_targets),
            "qa_censored": binary_metrics(logits, reference_targets, include),
        }
        for name, logits in model_logits.items()
    }

    events: list[dict[str, Any]] = []
    for index, sample in enumerate(dataset.samples):
        event, input_day, target_day, input_record, target_record = sample
        standard_mask = torch.ones_like(reference_targets[index], dtype=torch.bool)
        qa_mask = include[index]
        record: dict[str, Any] = {
            "event": event,
            "input_day": input_day,
            "target_day": target_day,
            "input_reliable_fraction": float(input_record["reliable_fraction"]),
            "input_mean_reliable_age_hours": input_record.get(
                "mean_reliable_age_hours"
            ),
            "target_reliable_fraction": float(target_record["reliable_fraction"]),
            "target_mean_reliable_age_hours": target_record.get(
                "mean_reliable_age_hours"
            ),
            "positive_pixels": int(reference_targets[index].bool().sum().item()),
            "qa_included_fraction": float(qa_mask.float().mean().item()),
        }
        for name, logits in model_logits.items():
            standard_event = binary_metrics(
                logits[index], reference_targets[index], standard_mask
            )
            censored_event = binary_metrics(
                logits[index], reference_targets[index], qa_mask
            )
            record[f"{name}_standard_loss"] = standard_event["loss"]
            record[f"{name}_standard_brier"] = standard_event["brier"]
            record[f"{name}_qa_censored_loss"] = censored_event["loss"]
            record[f"{name}_qa_censored_brier"] = censored_event["brier"]
        events.append(record)

    risk = {
        name: {
            metric_name: risk_coverage(
                events,
                quality_key="input_reliable_fraction",
                error_key=f"{name}_{metric_name}",
            )
            for metric_name in (
                "standard_loss",
                "standard_brier",
                "qa_censored_loss",
                "qa_censored_brier",
            )
        }
        for name in models
    }
    standard_ap_delta = float(metrics["attention"]["standard"]["avg_precision"]) - float(
        metrics["p00"]["standard"]["avg_precision"]
    )
    censored_ap_delta = float(
        metrics["attention"]["qa_censored"]["avg_precision"]
    ) - float(metrics["p00"]["qa_censored"]["avg_precision"])
    result = {
        "schema_version": 1,
        "status": "pass",
        "scientific_claim": False,
        "purpose": "fixed 24-sample target-observation feasibility audit",
        "year": 2021,
        "sample_count": len(dataset),
        "input_gate": "model",
        "target_gate": "target",
        "target_mask_policy": "positive-or-reliably-observed",
        "p00_record": str(p00_record),
        "attention_checkpoint": str(attention_checkpoint),
        "label_population": {
            "total_pixels": int(reference_targets.numel()),
            "positive_pixels": int(reference_targets.bool().sum().item()),
            "qa_included_pixels": int(include.sum().item()),
            "unknown_zero_pixels": int(
                ((~reference_targets.bool()) & (~reference_reliability.bool())).sum().item()
            ),
        },
        "models": metrics,
        "comparison": {
            "standard_attention_minus_p00_ap": standard_ap_delta,
            "qa_censored_attention_minus_p00_ap": censored_ap_delta,
            "ap_delta_change_after_censoring": censored_ap_delta - standard_ap_delta,
            "ap_ranking_changed": (standard_ap_delta > 0) != (censored_ap_delta > 0),
        },
        "risk_coverage_by_input_reliability": risk,
        "events": events,
    }
    write_result_new(output, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
