"""Evaluate one matched D1 checkpoint on the 2021 reliability screen."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from .evaluate_corrected_baseline import SCREEN_SCENARIOS
from .evaluate_missingness import (
    evaluate_batches,
    load_checkpoint_model,
    write_result_new,
)
from .evaluation import build_controlled_dataset


def validate_d1_checkpoint(payload: Mapping[str, object]) -> str:
    """Require one of the exact matched D1 control/method checkpoints."""

    candidate_id = payload.get("candidate_id")
    expected_lambda = {"D1-ERM": 0.0, "D1-KL": 0.1}.get(str(candidate_id))
    valid = (
        payload.get("schema_version") == 1
        and payload.get("status") == "pass"
        and payload.get("matched_pair") == "D1"
        and payload.get("experiment") == "C00"
        and payload.get("steps") == 3_000
        and payload.get("seed") == 0
        and payload.get("lambda_consistency") == expected_lambda
        and isinstance(payload.get("hyper_parameters"), Mapping)
        and isinstance(payload.get("state_dict"), Mapping)
    )
    if expected_lambda is None or not valid:
        raise ValueError("D1 checkpoint contract is invalid")
    return str(candidate_id)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint.resolve(strict=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("D1 checkpoint must be a mapping")
    candidate_id = validate_d1_checkpoint(payload)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA evaluation requested but unavailable")

    model = load_checkpoint_model(
        checkpoint,
        experiment_id="C00",
        upstream_root=args.upstream_root,
        device=device,
    )
    results: dict[str, dict[str, int | float]] = {}
    for scenario_id in SCREEN_SCENARIOS:
        dataset = build_controlled_dataset(
            upstream_root=args.upstream_root,
            data_root=args.data_root,
            stats_path=args.stats_path,
            experiment_id="C00",
            scenario_id=scenario_id,
            evaluation_year=2021,
            heldout_authorized=False,
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
            "mode": "D1-validation",
            "scientific_claim": False,
            "candidate_id": candidate_id,
            "checkpoint": str(checkpoint),
            "scenario_id": scenario_id,
            "year": 2021,
            "metrics": metrics,
        }
        write_result_new(output_root / f"{scenario_id}.json", result)
        results[scenario_id] = metrics
        print(json.dumps(result, sort_keys=True), flush=True)

    clean_ap = float(results["M00"]["avg_precision"])
    summary = {
        "schema_version": 1,
        "status": "pass",
        "mode": "D1-validation",
        "scientific_claim": False,
        "candidate_id": candidate_id,
        "checkpoint": str(checkpoint),
        "year": 2021,
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
