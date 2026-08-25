"""Evaluate the frozen-backbone P04 residual gate on 2021."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from .evaluate_missingness import evaluate_batches, load_checkpoint_model, write_result_new
from .evaluate_spatial_router import SCENARIOS, _checkpoint_from_record
from .evaluation import build_controlled_dataset
from .prototype import FIRE_DROPOUT_PROBABILITY, PROTOTYPE_ID as P00_ID
from .residual_gate import (
    GATE_TRAINING_STEPS,
    PROTOTYPE_ID,
    SPATIAL_PROTOTYPE_ID,
    FrozenSpatialResidualGate,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p00-record", type=Path, required=True)
    parser.add_argument("--gate-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--residual-kernel-size", type=int, choices=(1, 3), default=1
    )
    args = parser.parse_args(argv)

    p00_checkpoint = _checkpoint_from_record(
        args.p00_record,
        prototype_id=P00_ID,
        training_policy={
            "active_fire_dropout_probability": FIRE_DROPOUT_PROBABILITY,
            "training_only": True,
        },
    )
    gate_payload = torch.load(
        args.gate_checkpoint.resolve(strict=True),
        map_location="cpu",
        weights_only=False,
    )
    expected_parameters = 17 if args.residual_kernel_size == 1 else 145
    prototype_id = (
        PROTOTYPE_ID
        if args.residual_kernel_size == 1
        else SPATIAL_PROTOTYPE_ID
    )
    if (
        not isinstance(gate_payload, dict)
        or gate_payload.get("status") != "pass"
        or gate_payload.get("prototype_id") != prototype_id
        or gate_payload.get("steps") != GATE_TRAINING_STEPS
        or gate_payload.get("residual_kernel_size") != args.residual_kernel_size
        or gate_payload.get("trainable_parameters") != expected_parameters
        or not isinstance(gate_payload.get("residual_head"), dict)
    ):
        raise ValueError("invalid residual-gate checkpoint")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA evaluation requested but unavailable")
    default_model = load_checkpoint_model(
        p00_checkpoint,
        experiment_id="C00",
        upstream_root=args.upstream_root,
        device=device,
    )
    gate = FrozenSpatialResidualGate(
        default_model, residual_kernel_size=args.residual_kernel_size
    )
    gate.residual_head.load_state_dict(gate_payload["residual_head"], strict=True)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)

    results: dict[str, dict[str, int | float]] = {}
    for scenario_id in SCENARIOS:
        dataset = build_controlled_dataset(
            upstream_root=args.upstream_root,
            data_root=args.data_root,
            stats_path=args.stats_path,
            experiment_id="C00",
            scenario_id=scenario_id,
            evaluation_year=2021,
            heldout_authorized=False,
            routing_mask_channel=True,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        metrics = evaluate_batches(gate, loader, device=device)
        result = {
            "schema_version": 1,
            "status": "pass",
            "mode": "prototype-validation",
            "scientific_claim": False,
            "prototype_id": prototype_id,
            "scenario_id": scenario_id,
            "year": 2021,
            "metrics": metrics,
        }
        write_result_new(output_root / f"{scenario_id}.json", result)
        results[scenario_id] = metrics
        print(json.dumps(result, sort_keys=True), flush=True)

    summary = {
        "schema_version": 1,
        "status": "pass",
        "mode": "prototype-validation",
        "scientific_claim": False,
        "prototype_id": prototype_id,
        "p00_record": str(args.p00_record.resolve(strict=True)),
        "gate_checkpoint": str(args.gate_checkpoint.resolve(strict=True)),
        "year": 2021,
        "scenario_count": len(SCENARIOS),
        "results": {
            scenario_id: {"metrics": metrics}
            for scenario_id, metrics in results.items()
        },
    }
    write_result_new(output_root / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
