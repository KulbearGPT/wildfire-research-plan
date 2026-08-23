"""Render immutable engineering or formal M00--M07 evaluation matrices."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .matrix import CORRUPTIONS, run_spec
from .missingness import SCHEMA_VERSION
from .promotion import validate_completed_for_run


DECLARED_10K_RUNS = (
    "C00-S0-10K",
    "C00-S1-10K",
    "C00-S2-10K",
    "C02-S0-10K",
    "C02-S1-10K",
    "C02-S2-10K",
)
FORMAL_YEARS = (2022, 2023)
ENGINEERING_YEARS = (2021,)


def _validated_records(
    record_paths: Mapping[str, Path], *, require_all: bool
) -> list[dict[str, object]]:
    supplied = set(record_paths)
    expected = set(DECLARED_10K_RUNS)
    if require_all and supplied != expected:
        raise ValueError("formal missingness manifest requires all six clean 10K records")
    if not supplied or not supplied <= expected:
        raise ValueError("records must identify declared clean 10K runs")

    records: list[dict[str, object]] = []
    for run_id in DECLARED_10K_RUNS:
        if run_id not in record_paths:
            continue
        record_path = Path(record_paths[run_id]).resolve(strict=True)
        payload = validate_completed_for_run(record_path, run_id)
        checkpoint = Path(str(payload["checkpoint"])).resolve()
        if not checkpoint.is_file():
            raise ValueError(f"checkpoint file is missing for {run_id}: {checkpoint}")
        spec = run_spec(run_id)
        records.append(
            {
                "run_id": run_id,
                "experiment_id": spec.experiment_id,
                "seed": spec.seed,
                "record": str(record_path),
                "slurm_job_id": payload["slurm_job_id"],
                "checkpoint": str(checkpoint),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "checkpoint_global_step": payload["checkpoint_global_step"],
                "clean_validation_metrics": payload["metrics"],
            }
        )
    return records


def _manifest(
    record_paths: Mapping[str, Path],
    *,
    output_root: Path,
    mode: str,
) -> dict[str, object]:
    if mode not in {"engineering", "formal"}:
        raise ValueError(f"unknown missingness manifest mode: {mode}")
    formal = mode == "formal"
    records = _validated_records(record_paths, require_all=formal)
    years = FORMAL_YEARS if formal else ENGINEERING_YEARS
    root = Path(output_root).resolve()
    tasks: list[dict[str, object]] = []
    records_by_run = {str(record["run_id"]): record for record in records}
    for run_id in DECLARED_10K_RUNS:
        if run_id not in records_by_run:
            continue
        record = records_by_run[run_id]
        for scenario_id, scenario in CORRUPTIONS.items():
            for year in years:
                evaluation_id = f"{run_id}-{scenario_id}-Y{year}"
                tasks.append(
                    {
                        "evaluation_id": evaluation_id,
                        "run_id": run_id,
                        "experiment_id": record["experiment_id"],
                        "seed": record["seed"],
                        "checkpoint": record["checkpoint"],
                        "scenario_id": scenario_id,
                        "matrix_seed": scenario.matrix_seed,
                        "year": year,
                        "output": str(root / evaluation_id / "result.json"),
                    }
                )
    return {
        "schema_version": 1,
        "corruption_schema_version": SCHEMA_VERSION,
        "mode": mode,
        "scientific_claim": formal,
        "heldout_access": formal,
        "years": list(years),
        "boundary": {
            "train_statistics_years": [2016, 2017, 2018, 2019, 2020],
            "engineering_years": [2021],
            "heldout_years": [2022, 2023],
            "effective_history": 6,
            "is_train": False,
            "retraining": False,
        },
        "records": records,
        "tasks": tasks,
    }


def engineering_manifest(
    record_paths: Mapping[str, Path], *, output_root: Path
) -> dict[str, object]:
    """Render a non-scientific 2021 engineering matrix for available checkpoints."""

    return _manifest(record_paths, output_root=output_root, mode="engineering")


def formal_manifest(
    record_paths: Mapping[str, Path], *, output_root: Path
) -> dict[str, object]:
    """Render the full held-out matrix only after all clean replications pass."""

    return _manifest(record_paths, output_root=output_root, mode="formal")


def write_manifest_new(path: Path, manifest: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _record_argument(value: str) -> tuple[str, Path]:
    run_id, separator, path = value.partition("=")
    if not separator or run_id not in DECLARED_10K_RUNS or not path:
        raise argparse.ArgumentTypeError("record must use declared RUN_ID=PATH form")
    return run_id, Path(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("engineering", "formal"), required=True)
    parser.add_argument("--record", type=_record_argument, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    records = dict(args.record)
    if len(records) != len(args.record):
        raise ValueError("duplicate record run ID")
    manifest = (
        formal_manifest(records, output_root=args.output_root)
        if args.mode == "formal"
        else engineering_manifest(records, output_root=args.output_root)
    )
    write_manifest_new(args.output, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
