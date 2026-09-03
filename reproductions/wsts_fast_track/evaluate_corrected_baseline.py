"""Evaluate one corrected B0--B3 baseline on the frozen rapid screen."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from .corrected_baselines import CorrectedBaselineSpec, corrected_baseline_spec
from .evaluate_missingness import (
    evaluate_batches,
    load_checkpoint_model,
    write_result_new,
)
from .evaluation import build_controlled_dataset


SCREEN_SCENARIOS = ("M00", "M01", "M06", "M07")


def validate_evaluation_record(
    record: Mapping[str, object],
) -> CorrectedBaselineSpec:
    """Require the exact corrected-index, from-scratch screening contract."""

    try:
        baseline = corrected_baseline_spec(str(record.get("baseline_id")))
    except ValueError as error:
        raise ValueError("corrected baseline completion record is invalid") from error
    expected = {
        "schema_version": 1,
        "status": "pass",
        "experiment": baseline.experiment_id,
        "training_policy": baseline.training_policy,
        "seed": baseline.seed,
        "max_steps": baseline.max_steps,
        "corrected_index": True,
        "initialization": "from_scratch",
        "validation_years": [2021],
        "test_enabled": False,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("corrected baseline completion record is invalid")
    checkpoint = record.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("corrected baseline completion record is invalid")
    return baseline


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    record_path = args.record.resolve(strict=True)
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("corrected baseline completion record must be an object")
    baseline = validate_evaluation_record(payload)
    checkpoint = Path(str(payload["checkpoint"])).resolve(strict=True)
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers nonnegative")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA evaluation requested but unavailable")

    model = load_checkpoint_model(
        checkpoint,
        experiment_id=baseline.experiment_id,
        upstream_root=args.upstream_root,
        device=device,
    )
    results: dict[str, dict[str, int | float]] = {}
    for scenario_id in SCREEN_SCENARIOS:
        dataset = build_controlled_dataset(
            upstream_root=args.upstream_root,
            data_root=args.data_root,
            stats_path=args.stats_path,
            experiment_id=baseline.experiment_id,
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
            "mode": "corrected-baseline-validation",
            "scientific_claim": False,
            "baseline_id": baseline.baseline_id,
            "record": str(record_path),
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
        "mode": "corrected-baseline-validation",
        "scientific_claim": False,
        "baseline_id": baseline.baseline_id,
        "record": str(record_path),
        "year": 2021,
        "scenario_count": len(SCREEN_SCENARIOS),
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
