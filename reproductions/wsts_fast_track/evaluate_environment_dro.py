"""Evaluate P09 and its optional matched P10 ERM control."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from .environment_dro import (
    ERM_PROTOTYPE_ID,
    GROUP_COUNT,
    GROUP_DRO_STEP_SIZE,
    LEARNING_RATE,
    PROTOTYPE_ID,
    TRAINING_STEPS,
)
from .evaluate_missingness import evaluate_batches, load_checkpoint_model, write_result_new
from .evaluate_spatial_router import SCENARIOS, _checkpoint_from_record, router_evaluation_boundary
from .evaluation import build_controlled_dataset
from .prototype import (
    BLOCK_DROPOUT_PROBABILITY,
    BLOCK_PROTOTYPE_ID,
    FIRE_DROPOUT_PROBABILITY,
    PROTOTYPE_ID as P00_ID,
)
from .spatial_router import (
    RELIABILITY_ROUTER_ID,
    ROUTER_ID,
    ReliabilityExpertRouter,
    RoutingInputModel,
    SpatialExpertRouter,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p00-record", type=Path, required=True)
    parser.add_argument("--p02-record", type=Path, required=True)
    parser.add_argument("--dro-checkpoint", type=Path, required=True)
    parser.add_argument("--erm-checkpoint", type=Path)
    parser.add_argument("--reliability-router", action="store_true")
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
    mode, scientific_claim = router_evaluation_boundary(
        args.year, heldout_authorized=args.heldout_authorized
    )

    p00_checkpoint = _checkpoint_from_record(
        args.p00_record,
        prototype_id=P00_ID,
        training_policy={
            "active_fire_dropout_probability": FIRE_DROPOUT_PROBABILITY,
            "training_only": True,
        },
    )
    p02_checkpoint = _checkpoint_from_record(
        args.p02_record,
        prototype_id=BLOCK_PROTOTYPE_ID,
        training_policy={
            "active_fire_dropout_probability": FIRE_DROPOUT_PROBABILITY,
            "block_dropout_probability": BLOCK_DROPOUT_PROBABILITY,
            "block_fractions": [0.25, 0.5],
            "training_only": True,
        },
    )
    payload = torch.load(
        args.dro_checkpoint.resolve(strict=True),
        map_location="cpu",
        weights_only=False,
    )
    if (
        not isinstance(payload, dict)
        or payload.get("status") != "pass"
        or payload.get("prototype_id") != PROTOTYPE_ID
        or payload.get("steps") != TRAINING_STEPS
        or payload.get("group_count") != GROUP_COUNT
        or payload.get("learning_rate") != LEARNING_RATE
        or payload.get("group_dro_step_size") != GROUP_DRO_STEP_SIZE
        or payload.get("base_p02_checkpoint") != str(p02_checkpoint)
        or not isinstance(payload.get("model_state"), dict)
    ):
        raise ValueError("invalid P09 GroupDRO checkpoint")
    erm_payload = None
    if args.erm_checkpoint is not None:
        erm_payload = torch.load(
            args.erm_checkpoint.resolve(strict=True),
            map_location="cpu",
            weights_only=False,
        )
        if (
            not isinstance(erm_payload, dict)
            or erm_payload.get("status") != "pass"
            or erm_payload.get("prototype_id") != ERM_PROTOTYPE_ID
            or erm_payload.get("objective") != "erm"
            or erm_payload.get("steps") != TRAINING_STEPS
            or erm_payload.get("group_count") != GROUP_COUNT
            or erm_payload.get("learning_rate") != LEARNING_RATE
            or erm_payload.get("group_dro_step_size") is not None
            or erm_payload.get("base_p02_checkpoint") != str(p02_checkpoint)
            or not isinstance(erm_payload.get("model_state"), dict)
        ):
            raise ValueError("invalid P10 matched ERM checkpoint")
    if args.reliability_router and erm_payload is None:
        raise ValueError("P11 reliability routing requires --erm-checkpoint")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA evaluation requested but unavailable")
    default_model = load_checkpoint_model(
        p00_checkpoint, experiment_id="C00", upstream_root=args.upstream_root, device=device
    )
    original_expert = load_checkpoint_model(
        p02_checkpoint, experiment_id="C00", upstream_root=args.upstream_root, device=device
    )
    dro_expert = load_checkpoint_model(
        p02_checkpoint, experiment_id="C00", upstream_root=args.upstream_root, device=device
    )
    dro_expert.load_state_dict(payload["model_state"], strict=True)
    baseline = RoutingInputModel(default_model)
    p03 = SpatialExpertRouter(default_model, original_expert)
    p09 = SpatialExpertRouter(default_model, dro_expert)
    models = [
        ("P00", P00_ID, baseline),
        ("P03", ROUTER_ID, p03),
        ("P09", PROTOTYPE_ID, p09),
    ]
    if erm_payload is not None:
        erm_expert = load_checkpoint_model(
            p02_checkpoint, experiment_id="C00", upstream_root=args.upstream_root, device=device
        )
        erm_expert.load_state_dict(erm_payload["model_state"], strict=True)
        models.append(
            ("P10", ERM_PROTOTYPE_ID, SpatialExpertRouter(default_model, erm_expert))
        )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)

    all_results: dict[str, dict[str, dict[str, int | float]]] = {
        label: {} for label, _, _ in models
    }
    if args.reliability_router:
        all_results["P11"] = {}
    for scenario_id in SCENARIOS:
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
        scenario_models = list(models)
        if args.reliability_router:
            scenario_models.append(
                (
                    "P11",
                    RELIABILITY_ROUTER_ID,
                    ReliabilityExpertRouter(
                        default_model,
                        erm_expert,
                        route_all=scenario_id == "M01",
                    ),
                )
            )
        for label, prototype_id, model in scenario_models:
            metrics = evaluate_batches(model, loader, device=device)
            result = {
                "schema_version": 1,
                "status": "pass",
                "mode": mode,
                "scientific_claim": scientific_claim,
                "prototype_id": prototype_id,
                "scenario_id": scenario_id,
                "year": args.year,
                "metrics": metrics,
            }
            write_result_new(output_root / f"{label}-{scenario_id}.json", result)
            all_results[label][scenario_id] = metrics
            print(json.dumps(result, sort_keys=True), flush=True)

    summary = {
        "schema_version": 1,
        "status": "pass",
        "mode": mode,
        "scientific_claim": scientific_claim,
        "prototype_id": (
            RELIABILITY_ROUTER_ID
            if args.reliability_router
            else ERM_PROTOTYPE_ID if erm_payload is not None else PROTOTYPE_ID
        ),
        "p00_record": str(args.p00_record.resolve(strict=True)),
        "p02_record": str(args.p02_record.resolve(strict=True)),
        "dro_checkpoint": str(args.dro_checkpoint.resolve(strict=True)),
        "erm_checkpoint": (
            str(args.erm_checkpoint.resolve(strict=True))
            if args.erm_checkpoint is not None
            else None
        ),
        "year": args.year,
        "scenario_count": len(SCENARIOS),
        "results": all_results,
    }
    write_result_new(output_root / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
