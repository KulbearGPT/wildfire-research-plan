"""Evaluate the fixed D9-CEPR checkpoint on reliability scenarios."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from .counterfactual_error_pair_ranking import PAIR_FORMULATION, PAIR_TOP_K
from .evaluate_corrected_baseline import SCREEN_SCENARIOS, evaluation_boundary
from .evaluate_missingness import (
    evaluate_batches,
    load_checkpoint_model,
    write_result_new,
)
from .evaluation import build_controlled_dataset


def validate_cepr_checkpoint(payload: Mapping[str, object]) -> str:
    """Require the exact fixed D9-CEPR checkpoint contract."""

    valid = (
        payload.get("schema_version") == 1
        and payload.get("status") == "pass"
        and payload.get("candidate_id") == "D9-CEPR"
        and payload.get("matched_pair") == "D9"
        and payload.get("base_control") == "D1-ERM"
        and payload.get("closest_ablations") == ["D5-CIWC", "D6-CIRC"]
        and payload.get("experiment") == "C00"
        and payload.get("steps") == 3_000
        and payload.get("seed") == 0
        and payload.get("lambda_ciwc") == 0.1
        and payload.get("lambda_pair") == 0.01
        and payload.get("pair_top_k") == PAIR_TOP_K
        and payload.get("pair_formulation") == PAIR_FORMULATION
        and isinstance(payload.get("hyper_parameters"), Mapping)
        and isinstance(payload.get("state_dict"), Mapping)
    )
    if not valid:
        raise ValueError("CEPR checkpoint contract is invalid")
    return "D9-CEPR"


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
        raise ValueError("CEPR checkpoint must be a mapping")
    candidate_id = validate_cepr_checkpoint(payload)
    mode, scientific_claim = evaluation_boundary(
        args.year, heldout_authorized=args.heldout_authorized
    )
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
            evaluation_year=args.year,
            heldout_authorized=scientific_claim,
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
