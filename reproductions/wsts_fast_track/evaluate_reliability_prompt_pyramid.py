"""Evaluate the fixed D10 reliability prompt pyramid."""

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
    INPUT_PROMPT,
    PROMPT_CHANNELS,
    PROMPT_PARAMETER_COUNT,
    PROMPT_POOLING,
    CompleteReliabilityPromptPyramid,
    ReliabilityPromptPyramid,
)


def validate_rpp_checkpoint(payload: Mapping[str, object]) -> str:
    """Require the exact fixed D10-RPP checkpoint contract."""

    valid = (
        payload.get("schema_version") == 1
        and payload.get("status") == "pass"
        and payload.get("candidate_id") == "D10-RPP"
        and payload.get("matched_pair") == "D10"
        and payload.get("base_control") == "D2-STD"
        and payload.get("closest_ablation") == "D4-TOKEN"
        and payload.get("experiment") == "C00"
        and payload.get("steps") == 3_000
        and payload.get("seed") == 0
        and payload.get("variant") == "prompt-pyramid"
        and payload.get("processed_space_matched_corruption") is True
        and payload.get("prompt_channels") == list(PROMPT_CHANNELS)
        and payload.get("prompt_parameter_count") == PROMPT_PARAMETER_COUNT
        and payload.get("prompt_pooling") == PROMPT_POOLING
        and isinstance(payload.get("hyper_parameters"), Mapping)
        and isinstance(payload.get("state_dict"), Mapping)
    )
    if not valid:
        raise ValueError("RPP checkpoint contract is invalid")
    return "D10-RPP"


def validate_crpp_checkpoint(payload: Mapping[str, object]) -> str:
    """Require the exact fixed D11 complete-pyramid contract."""

    valid = (
        payload.get("schema_version") == 1
        and payload.get("status") == "pass"
        and payload.get("candidate_id") == "D11-CRPP"
        and payload.get("matched_pair") == "D11"
        and payload.get("base_control") == "D2-STD"
        and payload.get("closest_ablations") == ["D4-TOKEN", "D10-RPP"]
        and payload.get("experiment") == "C00"
        and payload.get("steps") == 3_000
        and payload.get("seed") == 0
        and payload.get("variant") == "complete-prompt-pyramid"
        and payload.get("processed_space_matched_corruption") is True
        and payload.get("prompt_channels") == list(PROMPT_CHANNELS)
        and payload.get("prompt_parameter_count")
        == PROMPT_PARAMETER_COUNT + PROMPT_CHANNELS[0]
        and payload.get("prompt_pooling") == PROMPT_POOLING
        and payload.get("input_prompt") == INPUT_PROMPT
        and isinstance(payload.get("hyper_parameters"), Mapping)
        and isinstance(payload.get("state_dict"), Mapping)
    )
    if not valid:
        raise ValueError("CRPP checkpoint contract is invalid")
    return "D11-CRPP"


def validate_prompt_checkpoint(payload: Mapping[str, object]) -> str:
    if payload.get("candidate_id") == "D11-CRPP":
        return validate_crpp_checkpoint(payload)
    return validate_rpp_checkpoint(payload)


def load_rpp_model(
    payload: Mapping[str, Any], *, upstream_root: Path, device: torch.device
) -> ReliabilityPromptPyramid:
    """Reconstruct the fixed ResNet-18 base and prompt pyramid."""

    upstream = Path(upstream_root).resolve()
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "src"))
    model_class = getattr(importlib.import_module("models.SMPModel"), "SMPModel")
    init_args = checkpoint_init_args(payload["hyper_parameters"], experiment_id="C00")
    base_model = model_class(**init_args)
    model_class = (
        CompleteReliabilityPromptPyramid
        if payload.get("candidate_id") == "D11-CRPP"
        else ReliabilityPromptPyramid
    )
    model = model_class(base_model)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument(
        "--year", type=int, choices=(2021, 2022, 2023), default=2021
    )
    parser.add_argument("--heldout-authorized", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint.resolve(strict=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("RPP checkpoint must be a mapping")
    candidate_id = validate_prompt_checkpoint(payload)
    mode, scientific_claim = evaluation_boundary(
        args.year, heldout_authorized=args.heldout_authorized
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA evaluation requested but unavailable")
    model = load_rpp_model(
        payload, upstream_root=args.upstream_root, device=device
    )

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
        "checkpoint": str(checkpoint),
        "year": args.year,
        "results": {
            scenario_id: {
                "metrics": metrics,
                "delta_avg_precision_from_m00": float(
                    metrics["avg_precision"]
                )
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
