"""Evaluate the matched D2 standard/RNC checkpoints on 2021."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .evaluate_corrected_baseline import SCREEN_SCENARIOS, evaluation_boundary
from .evaluate_missingness import (
    checkpoint_init_args,
    evaluate_batches,
    write_result_new,
)
from .evaluation import build_controlled_dataset
from .reliability_normalized_conv import InputReliabilityNormalizedConv2d


def validate_d2_checkpoint(payload: Mapping[str, object]) -> str:
    """Return the variant after enforcing the exact matched D2 contract."""

    variant = str(payload.get("variant"))
    expected_id = {"standard": "D2-STD", "rnc": "D2-RNC"}.get(variant)
    valid = (
        payload.get("schema_version") == 1
        and payload.get("status") == "pass"
        and payload.get("candidate_id") == expected_id
        and payload.get("matched_pair") == "D2"
        and payload.get("experiment") == "C00"
        and payload.get("steps") == 3_000
        and payload.get("seed") == 0
        and payload.get("processed_space_matched_corruption") is True
        and isinstance(payload.get("hyper_parameters"), Mapping)
        and isinstance(payload.get("state_dict"), Mapping)
    )
    if expected_id is None or not valid:
        raise ValueError("D2 checkpoint contract is invalid")
    return variant


class D2EvaluationModel(nn.Module):
    """Expose one evaluation interface while retaining the invalidity channel."""

    def __init__(self, model: nn.Module, variant: str) -> None:
        super().__init__()
        self.model = model
        self.variant = variant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        model_input = x if self.variant == "rnc" else x[:, :, :-1]
        return self.model(model_input)

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.model.compute_loss(logits, target)


def load_d2_model(
    payload: Mapping[str, Any],
    *,
    upstream_root: Path,
    device: torch.device,
) -> D2EvaluationModel:
    """Reconstruct the standard or RNC C00 model from a D2 payload."""

    variant = validate_d2_checkpoint(payload)
    upstream = Path(upstream_root).resolve()
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "src"))
    model_class = getattr(importlib.import_module("models.SMPModel"), "SMPModel")
    init_args = checkpoint_init_args(payload["hyper_parameters"], experiment_id="C00")
    model = model_class(**init_args)
    if variant == "rnc":
        model.model.encoder.conv1 = InputReliabilityNormalizedConv2d(
            model.model.encoder.conv1
        )
    model.load_state_dict(payload["state_dict"], strict=True)
    return D2EvaluationModel(model, variant).to(device)


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
        raise ValueError("D2 checkpoint must be a mapping")
    variant = validate_d2_checkpoint(payload)
    candidate_id = str(payload["candidate_id"])
    mode, scientific_claim = evaluation_boundary(
        args.year, heldout_authorized=args.heldout_authorized
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA evaluation requested but unavailable")
    model = load_d2_model(payload, upstream_root=args.upstream_root, device=device)

    results: dict[str, dict[str, int | float]] = {}
    for scenario_id in SCREEN_SCENARIOS:
        dataset = build_controlled_dataset(
            upstream_root=args.upstream_root,
            data_root=args.data_root,
            stats_path=args.stats_path,
            experiment_id="C00",
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
