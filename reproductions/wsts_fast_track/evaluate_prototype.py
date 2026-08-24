"""Evaluate P00/P01 on the four scenarios used for rapid triage."""

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
from .prototype import (
    FIRE_DROPOUT_PROBABILITY,
    PROTOTYPE_ID,
    RELIABILITY_PROTOTYPE_ID,
)


SCENARIOS = ("M00", "M01", "M02", "M07")


def evaluation_boundary(
    year: int, *, heldout_authorized: bool
) -> tuple[str, bool]:
    """Resolve validation or one-time held-out prototype evaluation."""

    if year == 2021 and heldout_authorized is False:
        return "prototype-validation", False
    if year in {2022, 2023} and heldout_authorized is True:
        return "prototype-formal", True
    if year in {2022, 2023}:
        raise ValueError("held-out evaluation requires explicit authorization")
    raise ValueError("prototype evaluation year must be 2021, 2022, or 2023")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--year", type=int, choices=(2021, 2022, 2023), default=2021)
    parser.add_argument("--heldout-authorized", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    record_path = args.record.resolve(strict=True)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    prototype_id = record.get("prototype_id") if isinstance(record, dict) else None
    expected_policy = (
        {
            "active_fire_dropout_probability": FIRE_DROPOUT_PROBABILITY,
            "training_only": True,
        }
        if prototype_id == PROTOTYPE_ID
        else {
            "active_fire_dropout_probability": FIRE_DROPOUT_PROBABILITY,
            "active_fire_validity_channel": True,
            "training_only_dropout": True,
        }
    )
    if (
        not isinstance(record, dict)
        or record.get("status") != "pass"
        or prototype_id not in {PROTOTYPE_ID, RELIABILITY_PROTOTYPE_ID}
        or record.get("experiment") != "C00"
        or record.get("seed") != 0
        or record.get("max_steps") != 10_000
        or record.get("validation_years") != [2021]
        or record.get("test_enabled") is not False
        or record.get("training_policy") != expected_policy
    ):
        raise ValueError("prototype completion record is invalid")
    checkpoint = Path(str(record.get("checkpoint", ""))).resolve(strict=True)
    mode, scientific_claim = evaluation_boundary(
        args.year,
        heldout_authorized=args.heldout_authorized,
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
    results: dict[str, object] = {}
    for scenario_id in SCENARIOS:
        dataset = build_controlled_dataset(
            upstream_root=args.upstream_root,
            data_root=args.data_root,
            stats_path=args.stats_path,
            experiment_id="C00",
            scenario_id=scenario_id,
            evaluation_year=args.year,
            heldout_authorized=scientific_claim,
            active_fire_validity_channel=(
                prototype_id == RELIABILITY_PROTOTYPE_ID
            ),
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
            "prototype_id": prototype_id,
            "record": str(record_path),
            "scenario_id": scenario_id,
            "year": args.year,
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
        "mode": mode,
        "scientific_claim": scientific_claim,
        "prototype_id": prototype_id,
        "record": str(record_path),
        "year": args.year,
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
