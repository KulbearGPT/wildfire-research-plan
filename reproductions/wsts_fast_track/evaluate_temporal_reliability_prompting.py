"""Evaluate matched standard/SARP C02 T=5 continuations."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .evaluate_corrected_baseline import SCREEN_SCENARIOS, evaluation_boundary
from .evaluate_missingness import (
    checkpoint_init_args,
    evaluate_batches,
    write_result_new,
)
from .evaluation import build_controlled_dataset
from .reliability_prompt_pyramid import (
    PROMPT_CHANNELS,
    PROMPT_PARAMETER_COUNT,
    PROMPT_POOLING,
    SEVERITY_THRESHOLD,
)
from .temporal_reliability_prompting import (
    TEMPORAL_FEATURE_COUNT,
    TEMPORAL_STEPS,
    TemporalSeverityAdaptiveReliabilityPrompting,
)


def validate_t5_reliability_checkpoint(payload: Mapping[str, object]) -> str:
    """Return the variant after enforcing the fixed D13 T=5 contract."""

    variant = str(payload.get("variant"))
    candidate = {
        "standard": "D13-STD-T5",
        "sarp": "D13-SARP-T5",
    }.get(variant)
    common = (
        payload.get("schema_version") == 1
        and payload.get("status") == "pass"
        and candidate is not None
        and payload.get("candidate_id") == candidate
        and payload.get("matched_pair") == "D13-T5"
        and payload.get("base_control") == "B5"
        and payload.get("experiment") == "C02"
        and payload.get("steps") == 3_000
        and payload.get("seed") == 0
        and payload.get("processed_space_matched_corruption") is True
        and payload.get("temporal_steps") == TEMPORAL_STEPS
        and payload.get("feature_count") == TEMPORAL_FEATURE_COUNT
        and isinstance(payload.get("hyper_parameters"), Mapping)
        and isinstance(payload.get("state_dict"), Mapping)
    )
    if variant == "sarp":
        specific = (
            payload.get("prompt_channels") == list(PROMPT_CHANNELS)
            and payload.get("prompt_parameter_count")
            == PROMPT_PARAMETER_COUNT + PROMPT_CHANNELS[0]
            and payload.get("prompt_pooling") == PROMPT_POOLING
            and payload.get("severity_threshold") == SEVERITY_THRESHOLD
            and payload.get("mild_prompt")
            == "hierarchical-before-temporal-fusion"
            and payload.get("severe_prompt") == "per-step-input"
        )
    else:
        specific = (
            variant == "standard"
            and payload.get("prompt_channels") is None
            and payload.get("prompt_parameter_count") == 0
            and payload.get("prompt_pooling") is None
            and payload.get("severity_threshold") is None
            and payload.get("mild_prompt") is None
            and payload.get("severe_prompt") is None
        )
    if not common or not specific:
        raise ValueError("T5 reliability checkpoint contract is invalid")
    return variant


class T5ReliabilityEvaluationModel(torch.nn.Module):
    """Retain the invalidity channel only for the SARP branch."""

    def __init__(self, model: torch.nn.Module, variant: str) -> None:
        super().__init__()
        self.model = model
        self.variant = variant

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        model_input = packed if self.variant == "sarp" else packed[:, :, :-1]
        return self.model(model_input)

    def compute_loss(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return self.model.compute_loss(logits, target)


def load_t5_reliability_model(
    payload: Mapping[str, Any], *, upstream_root: Path, device: torch.device
) -> T5ReliabilityEvaluationModel:
    """Reconstruct one fixed D13 C02 checkpoint."""

    variant = validate_t5_reliability_checkpoint(payload)
    upstream = Path(upstream_root).resolve()
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "src"))
    model_class = getattr(
        importlib.import_module("models.SMPTempModel"), "SMPTempModel"
    )
    init_args = checkpoint_init_args(payload["hyper_parameters"], experiment_id="C02")
    base_model = model_class(**init_args)
    model: torch.nn.Module
    if variant == "sarp":
        model = TemporalSeverityAdaptiveReliabilityPrompting(base_model)
    else:
        model = base_model
    model.load_state_dict(payload["state_dict"], strict=True)
    return T5ReliabilityEvaluationModel(model, variant).to(device)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--year", type=int, choices=(2021, 2022, 2023), default=2021)
    parser.add_argument("--heldout-authorized", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint.resolve(strict=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("T5 reliability checkpoint must be a mapping")
    variant = validate_t5_reliability_checkpoint(payload)
    candidate_id = str(payload["candidate_id"])
    mode, scientific_claim = evaluation_boundary(
        args.year, heldout_authorized=args.heldout_authorized
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA evaluation requested but unavailable")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers nonnegative")
    model = load_t5_reliability_model(
        payload, upstream_root=args.upstream_root, device=device
    )

    results: dict[str, dict[str, int | float]] = {}
    for scenario_id in SCREEN_SCENARIOS:
        dataset = build_controlled_dataset(
            upstream_root=args.upstream_root,
            data_root=args.data_root,
            stats_path=args.stats_path,
            experiment_id="C02",
            scenario_id=scenario_id,
            evaluation_year=args.year,
            heldout_authorized=scientific_claim,
            routing_mask_channel=True,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        metrics = evaluate_batches(model, loader, device=device)
        result = {
            "schema_version": 1,
            "status": "pass",
            "mode": mode,
            "scientific_claim": scientific_claim,
            "candidate_id": candidate_id,
            "variant": variant,
            "checkpoint": str(checkpoint),
            "scenario_id": scenario_id,
            "year": args.year,
            "metrics": metrics,
        }
        write_result_new(output_root / f"{scenario_id}.json", result)
        results[scenario_id] = metrics
        print(json.dumps(result, sort_keys=True), flush=True)

    clean_ap = float(results["M00"]["avg_precision"])
    summary = {
        "schema_version": 1,
        "status": "pass",
        "mode": mode,
        "scientific_claim": scientific_claim,
        "candidate_id": candidate_id,
        "variant": variant,
        "checkpoint": str(checkpoint),
        "year": args.year,
        "results": {
            scenario_id: {
                "metrics": metrics,
                "delta_avg_precision_from_m00": float(metrics["avg_precision"])
                - clean_ap,
            }
            for scenario_id, metrics in results.items()
        },
    }
    write_result_new(output_root / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
