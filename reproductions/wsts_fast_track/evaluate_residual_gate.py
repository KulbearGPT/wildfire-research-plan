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
    BELIEF_PROTOTYPE_ID,
    BELIEF_SAMPLE_COUNT,
    BELIEF_TRAINING_STEPS,
    GATE_TRAINING_STEPS,
    LAST_BLOCK_PROTOTYPE_ID,
    PROTOTYPE_ID,
    SPATIAL_PROTOTYPE_ID,
    FrozenLastBlockRouter,
    FrozenSpatialResidualGate,
    FrozenStochasticBeliefResidual,
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
    parser.add_argument("--adapt-last-block", action="store_true")
    parser.add_argument("--stochastic-belief", action="store_true")
    args = parser.parse_args(argv)
    if args.adapt_last_block and args.stochastic_belief:
        raise ValueError("choose either last-block or stochastic-belief adaptation")

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
    if args.stochastic_belief:
        prototype_id = BELIEF_PROTOTYPE_ID
        gate_type = "stochastic-belief"
        training_steps = BELIEF_TRAINING_STEPS
    elif args.adapt_last_block:
        prototype_id = LAST_BLOCK_PROTOTYPE_ID
        gate_type = "last-decoder-block"
        training_steps = GATE_TRAINING_STEPS
    else:
        prototype_id = (
            PROTOTYPE_ID
            if args.residual_kernel_size == 1
            else SPATIAL_PROTOTYPE_ID
        )
        gate_type = f"residual-{args.residual_kernel_size}x{args.residual_kernel_size}"
        training_steps = GATE_TRAINING_STEPS
    if (
        not isinstance(gate_payload, dict)
        or gate_payload.get("status") != "pass"
        or gate_payload.get("prototype_id") != prototype_id
        or gate_payload.get("steps") != training_steps
        or gate_payload.get("gate_type") != gate_type
        or gate_payload.get("residual_kernel_size") != args.residual_kernel_size
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
    if args.stochastic_belief:
        gate = FrozenStochasticBeliefResidual(
            default_model, sample_count=BELIEF_SAMPLE_COUNT
        )
        if (
            gate_payload.get("sample_count") != BELIEF_SAMPLE_COUNT
            or gate_payload.get("learning_rate") != 1e-3
            or not isinstance(gate_payload.get("belief_head"), dict)
            or not isinstance(gate_payload.get("output_head"), dict)
        ):
            raise ValueError("P07 checkpoint lacks its stochastic heads")
        gate.belief_head.load_state_dict(gate_payload["belief_head"], strict=True)
        gate.output_head.load_state_dict(gate_payload["output_head"], strict=True)
    elif args.adapt_last_block:
        gate = FrozenLastBlockRouter(default_model)
        if (
            not isinstance(gate_payload.get("adapted_block"), dict)
            or not isinstance(gate_payload.get("adapted_head"), dict)
        ):
            raise ValueError("P06 checkpoint lacks its adapted branch")
        gate.adapted_block.load_state_dict(gate_payload["adapted_block"], strict=True)
        gate.adapted_head.load_state_dict(gate_payload["adapted_head"], strict=True)
    else:
        gate = FrozenSpatialResidualGate(
            default_model, residual_kernel_size=args.residual_kernel_size
        )
        if not isinstance(gate_payload.get("residual_head"), dict):
            raise ValueError("residual checkpoint lacks its head")
        gate.residual_head.load_state_dict(gate_payload["residual_head"], strict=True)
    expected_parameters = sum(
        parameter.numel() for parameter in gate.trainable_parameters()
    )
    if gate_payload.get("trainable_parameters") != expected_parameters:
        raise ValueError("gate checkpoint parameter count differs from model")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)

    results: dict[str, dict[str, int | float]] = {}
    for scenario_id in SCENARIOS:
        torch.manual_seed(0)
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
