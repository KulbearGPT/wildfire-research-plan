"""Evaluate P03 against P00 on validation or authorized held-out years."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from .evaluate_missingness import evaluate_batches, load_checkpoint_model, write_result_new
from .evaluate_prototype import evaluation_boundary
from .evaluation import build_controlled_dataset
from .prototype import (
    BLOCK_DROPOUT_PROBABILITY,
    BLOCK_PROTOTYPE_ID,
    FIRE_DROPOUT_PROBABILITY,
    PROTOTYPE_ID,
)
from .spatial_router import ROUTER_ID, SpatialExpertRouter


SCENARIOS = ("M00", "M01", "M06", "M07")


def router_evaluation_boundary(
    year: int, *, heldout_authorized: bool
) -> tuple[str, bool]:
    """Apply the shared one-time held-out authorization boundary."""

    return evaluation_boundary(year, heldout_authorized=heldout_authorized)


def _checkpoint_from_record(
    path: Path,
    *,
    prototype_id: str,
    training_policy: Mapping[str, object],
) -> Path:
    record = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if (
        not isinstance(record, dict)
        or record.get("status") != "pass"
        or record.get("prototype_id") != prototype_id
        or record.get("experiment") != "C00"
        or record.get("seed") != 0
        or record.get("max_steps") != 10_000
        or record.get("training_policy") != dict(training_policy)
        or record.get("validation_years") != [2021]
        or record.get("test_enabled") is not False
    ):
        raise ValueError(f"invalid frozen prototype record: {prototype_id}")
    return Path(str(record.get("checkpoint", ""))).resolve(strict=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p00-record", type=Path, required=True)
    parser.add_argument("--p02-record", type=Path, required=True)
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
        prototype_id=PROTOTYPE_ID,
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
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA evaluation requested but unavailable")
    default_model = load_checkpoint_model(
        p00_checkpoint,
        experiment_id="C00",
        upstream_root=args.upstream_root,
        device=device,
    )
    block_model = load_checkpoint_model(
        p02_checkpoint,
        experiment_id="C00",
        upstream_root=args.upstream_root,
        device=device,
    )
    router = SpatialExpertRouter(default_model, block_model)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)

    results: dict[str, dict[str, int | float]] = {}
    baseline_results: dict[str, dict[str, int | float]] = {}
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
        baseline_metrics = evaluate_batches(default_model, loader, device=device)
        metrics = evaluate_batches(router, loader, device=device)
        baseline_result = {
            "schema_version": 1,
            "status": "pass",
            "mode": mode,
            "scientific_claim": scientific_claim,
            "prototype_id": PROTOTYPE_ID,
            "scenario_id": scenario_id,
            "year": args.year,
            "metrics": baseline_metrics,
        }
        result = {
            "schema_version": 1,
            "status": "pass",
            "mode": mode,
            "scientific_claim": scientific_claim,
            "prototype_id": ROUTER_ID,
            "scenario_id": scenario_id,
            "year": args.year,
            "metrics": metrics,
        }
        write_result_new(output_root / f"P00-{scenario_id}.json", baseline_result)
        write_result_new(output_root / f"P03-{scenario_id}.json", result)
        baseline_results[scenario_id] = baseline_metrics
        results[scenario_id] = metrics
        print(json.dumps(baseline_result, sort_keys=True), flush=True)
        print(json.dumps(result, sort_keys=True), flush=True)

    summary = {
        "schema_version": 1,
        "status": "pass",
        "mode": mode,
        "scientific_claim": scientific_claim,
        "prototype_id": ROUTER_ID,
        "p00_record": str(args.p00_record.resolve(strict=True)),
        "p02_record": str(args.p02_record.resolve(strict=True)),
        "year": args.year,
        "scenario_count": len(SCENARIOS),
        "results": {
            scenario_id: {"metrics": metrics}
            for scenario_id, metrics in results.items()
        },
        "baseline_results": {
            scenario_id: {"metrics": metrics}
            for scenario_id, metrics in baseline_results.items()
        },
    }
    write_result_new(output_root / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
