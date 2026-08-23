"""Validate screening evidence and render immutable 10K promotion manifests."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

from .matrix import RunSpec, matrix_payload, run_spec


_COMPLETED_KEYS = {
    "status",
    "purpose",
    "experiment",
    "slurm_job_id",
    "max_steps",
    "seed",
    "train_years",
    "validation_years",
    "test_enabled",
    "withheld_years",
    "checkpoint",
    "checkpoint_global_step",
    "peak_cuda_allocated_bytes",
    "wall_seconds",
    "metrics",
}
_METRIC_KEYS = {"val_avg_precision", "val_f1", "val_loss"}
_TRAIN_YEARS = [2016, 2017, 2018, 2019, 2020]
_VALIDATION_YEARS = [2021]
_WITHHELD_YEARS = [2022, 2023]


def load_completed(path: Path) -> dict[str, object]:
    """Load one strict external fast-track completion record."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"completed record is unreadable: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("completed record must be a JSON object")
    if set(payload) != _COMPLETED_KEYS:
        raise ValueError("completed record fields differ from the required schema")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != _METRIC_KEYS:
        raise ValueError("metrics fields differ from the required schema")
    return payload


def _require_string(payload: Mapping[str, object], field: str) -> str:
    value = payload[field]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _require_exact(
    payload: Mapping[str, object], field: str, expected: object
) -> None:
    if payload[field] != expected:
        raise ValueError(f"{field} must equal {expected!r}")


def _metric(payload: Mapping[str, object], field: str) -> float:
    metrics = payload["metrics"]
    assert isinstance(metrics, dict)
    value = metrics[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _validate_completed(
    payload: dict[str, object], expected: RunSpec
) -> dict[str, object]:
    _require_exact(payload, "status", "pass")
    _require_string(payload, "purpose")
    _require_exact(payload, "experiment", expected.experiment_id)
    _require_exact(payload, "max_steps", expected.max_steps)
    _require_exact(payload, "seed", expected.seed)
    _require_exact(payload, "train_years", _TRAIN_YEARS)
    _require_exact(payload, "validation_years", _VALIDATION_YEARS)
    _require_exact(payload, "test_enabled", False)
    _require_exact(payload, "withheld_years", _WITHHELD_YEARS)
    _require_string(payload, "checkpoint")
    _require_string(payload, "slurm_job_id")

    checkpoint_step = payload["checkpoint_global_step"]
    if (
        type(checkpoint_step) is not int
        or checkpoint_step <= 0
        or checkpoint_step > expected.max_steps
    ):
        raise ValueError("checkpoint_global_step is outside the screening budget")
    peak = payload["peak_cuda_allocated_bytes"]
    if type(peak) is not int or peak <= 0:
        raise ValueError("peak_cuda_allocated_bytes must be a positive integer")

    average_precision = _metric(payload, "val_avg_precision")
    f1 = _metric(payload, "val_f1")
    _metric(payload, "val_loss")
    if not 0.0 <= average_precision <= 1.0:
        raise ValueError("val_avg_precision must be within [0, 1]")
    if not 0.0 <= f1 <= 1.0:
        raise ValueError("val_f1 must be within [0, 1]")
    return payload


def validate_prerequisites(
    target: RunSpec, paths: Sequence[Path]
) -> tuple[dict[str, object], ...]:
    """Require exactly one passing record for every declared prerequisite."""

    if len(paths) != len(target.prerequisites):
        raise ValueError("prerequisite result count does not match the target")
    expected_by_experiment = {
        run_spec(run_id).experiment_id: run_spec(run_id)
        for run_id in target.prerequisites
    }
    loaded: dict[str, dict[str, object]] = {}
    for path in paths:
        payload = load_completed(path)
        experiment = payload["experiment"]
        if not isinstance(experiment, str) or experiment not in expected_by_experiment:
            raise ValueError("experiment is not a required prerequisite")
        if experiment in loaded:
            raise ValueError(f"duplicate prerequisite experiment: {experiment}")
        loaded[experiment] = _validate_completed(
            payload, expected_by_experiment[experiment]
        )
    expected_order = [run_spec(item).experiment_id for item in target.prerequisites]
    if set(loaded) != set(expected_order):
        raise ValueError("prerequisite experiment identities are incomplete")
    return tuple(loaded[experiment] for experiment in expected_order)


def promotion_manifest(
    run_id: str,
    result_paths: Sequence[Path],
    *,
    upstream_root: Path,
    data_root: Path,
    run_root: Path,
    stats_path: Path,
) -> dict[str, object]:
    """Build one deterministic seed-0 promotion manifest."""

    target = run_spec(run_id)
    if target.launch_state != "promotable":
        raise ValueError(f"fast-track run is not promotable: {run_id}")
    prerequisites = validate_prerequisites(target, result_paths)
    prerequisite_payload = []
    for prerequisite_id, payload in zip(target.prerequisites, prerequisites):
        prerequisite_payload.append(
            {
                "run_id": prerequisite_id,
                "experiment": payload["experiment"],
                "slurm_job_id": payload["slurm_job_id"],
                "checkpoint": payload["checkpoint"],
                "metrics": payload["metrics"],
            }
        )
    return {
        "schema_version": 1,
        "run": asdict(target),
        "prerequisites": prerequisite_payload,
        "boundary": {
            "train_years": _TRAIN_YEARS,
            "validation_years": _VALIDATION_YEARS,
            "test_enabled": False,
            "withheld_years": _WITHHELD_YEARS,
        },
        "command": [
            "python",
            "-m",
            "reproductions.wsts_fast_track.entrypoint",
            "--upstream-root",
            str(Path(upstream_root).resolve()),
            "--run-id",
            target.run_id,
            "--data-root",
            str(Path(data_root).resolve()),
            "--run-root",
            str(Path(run_root).resolve()),
            "--stats-path",
            str(Path(stats_path).resolve()),
        ],
    }


def write_json_new(path: Path, payload: Mapping[str, object]) -> None:
    """Write one new JSON artifact without replacing an existing path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    matrix_command = commands.add_parser("matrix")
    matrix_command.add_argument("--output", type=Path, required=True)

    render = commands.add_parser("render")
    render.add_argument("--run-id", required=True)
    render.add_argument("--result", type=Path, action="append", required=True)
    render.add_argument("--upstream-root", type=Path, required=True)
    render.add_argument("--data-root", type=Path, required=True)
    render.add_argument("--run-root", type=Path, required=True)
    render.add_argument("--stats-path", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "matrix":
        payload = matrix_payload()
    else:
        payload = promotion_manifest(
            args.run_id,
            args.result,
            upstream_root=args.upstream_root,
            data_root=args.data_root,
            run_root=args.run_root,
            stats_path=args.stats_path,
        )
    write_json_new(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
