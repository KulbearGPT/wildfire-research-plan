"""Evaluate P00 on the four validation scenarios used for rapid triage."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from .evaluate_missingness import (
    evaluate_batches,
    load_checkpoint_model,
    write_result_new,
)
from .evaluation import build_controlled_dataset
from .prototype import FIRE_DROPOUT_PROBABILITY, PROTOTYPE_ID


SCENARIOS = ("M00", "M01", "M02", "M07")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    record_path = args.record.resolve(strict=True)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (
        not isinstance(record, dict)
        or record.get("status") != "pass"
        or record.get("prototype_id") != PROTOTYPE_ID
        or record.get("experiment") != "C00"
        or record.get("seed") != 0
        or record.get("max_steps") != 10_000
        or record.get("validation_years") != [2021]
        or record.get("test_enabled") is not False
        or record.get("training_policy")
        != {
            "active_fire_dropout_probability": FIRE_DROPOUT_PROBABILITY,
            "training_only": True,
        }
    ):
        raise ValueError("P00 completion record is invalid")
    checkpoint = Path(str(record.get("checkpoint", ""))).resolve(strict=True)
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
    results: dict[str, object] = {}
    for scenario_id in SCENARIOS:
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
            "mode": "prototype-validation",
            "scientific_claim": False,
            "prototype_id": PROTOTYPE_ID,
            "record": str(record_path),
            "scenario_id": scenario_id,
            "year": 2021,
            "metrics": metrics,
        }
        write_result_new(output_root / f"{scenario_id}.json", result)
        results[scenario_id] = metrics
        print(json.dumps(result, sort_keys=True), flush=True)

    clean = results["M00"]
    assert isinstance(clean, dict)
    summary = {
        "schema_version": 1,
        "status": "pass",
        "mode": "prototype-validation",
        "scientific_claim": False,
        "prototype_id": PROTOTYPE_ID,
        "record": str(record_path),
        "year": 2021,
        "scenario_count": len(SCENARIOS),
        "results": {
            scenario_id: {
                "metrics": metrics,
                "delta_avg_precision_from_m00": float(metrics["avg_precision"])
                - float(clean["avg_precision"]),
            }
            for scenario_id, metrics in results.items()
            if isinstance(metrics, dict)
        },
    }
    write_result_new(output_root / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
