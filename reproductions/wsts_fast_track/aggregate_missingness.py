"""Aggregate M00--M07 results as same-year, same-seed deltas."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path

from .matrix import CORRUPTIONS, run_spec


METRIC_NAMES = ("avg_precision", "f1", "iou", "precision", "recall", "loss")
FORMAL_RUNS = (
    "C00-S0-10K",
    "C00-S1-10K",
    "C00-S2-10K",
    "C02-S0-10K",
    "C02-S1-10K",
    "C02-S2-10K",
)


def _load_result(path: Path) -> dict[str, object]:
    resolved = Path(path).resolve(strict=True)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"result is not valid JSON: {resolved}") from error
    if not isinstance(payload, dict):
        raise ValueError("result must be a JSON object")
    if set(payload) != {
        "schema_version",
        "status",
        "mode",
        "scientific_claim",
        "manifest",
        "task",
        "metrics",
        "boundary",
    }:
        raise ValueError("result fields differ from the required schema")
    if payload["schema_version"] != 1 or payload["status"] != "pass":
        raise ValueError("result schema or status is invalid")
    task = payload["task"]
    metrics = payload["metrics"]
    if not isinstance(task, dict) or not isinstance(metrics, dict):
        raise ValueError("result task and metrics must be objects")
    if set(metrics) != {"sample_count", "pixel_count", *METRIC_NAMES}:
        raise ValueError("result metrics differ from the required schema")
    if type(metrics["sample_count"]) is not int or metrics["sample_count"] <= 0:
        raise ValueError("sample_count must be a positive integer")
    if type(metrics["pixel_count"]) is not int or metrics["pixel_count"] <= 0:
        raise ValueError("pixel_count must be a positive integer")
    for name in METRIC_NAMES:
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if any(not 0.0 <= float(metrics[name]) <= 1.0 for name in METRIC_NAMES[:-1]):
        raise ValueError("bounded classification metric is outside [0, 1]")
    if float(metrics["loss"]) < 0.0:
        raise ValueError("loss must be nonnegative")
    if task.get("output") != str(resolved):
        raise ValueError("result path differs from its declared task output")
    run = run_spec(str(task.get("run_id")))
    if (
        run.max_steps != 10_000
        or task.get("experiment_id") != run.experiment_id
        or task.get("seed") != run.seed
        or task.get("scenario_id") not in CORRUPTIONS
    ):
        raise ValueError("result task identity is inconsistent")
    return payload


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_results(paths: Sequence[Path]) -> dict[str, object]:
    """Validate a complete matrix and aggregate within each evaluation year."""

    if not paths:
        raise ValueError("at least one missingness result is required")
    results = [_load_result(path) for path in paths]
    modes = {str(result["mode"]) for result in results}
    claims = {result["scientific_claim"] for result in results}
    manifests = {str(result["manifest"]) for result in results}
    if len(modes) != 1 or len(claims) != 1 or len(manifests) != 1:
        raise ValueError("results must share one mode, claim boundary, and manifest")
    mode = modes.pop()
    if mode not in {"engineering", "formal"}:
        raise ValueError("result mode is invalid")
    if claims != {mode == "formal"}:
        raise ValueError("result scientific claim differs from its mode")

    by_key: dict[tuple[str, str, int], dict[str, object]] = {}
    for result in results:
        task = result["task"]
        assert isinstance(task, dict)
        key = (str(task["run_id"]), str(task["scenario_id"]), int(task["year"]))
        if key in by_key:
            raise ValueError(f"duplicate missingness result: {key}")
        by_key[key] = result

    if mode == "formal":
        expected = {
            (run_id, scenario_id, year)
            for run_id in FORMAL_RUNS
            for scenario_id in CORRUPTIONS
            for year in (2022, 2023)
        }
        if set(by_key) != expected:
            raise ValueError("formal aggregation requires the complete 96-task matrix")
    else:
        if any(year != 2021 for _, _, year in by_key):
            raise ValueError("engineering aggregation is 2021-only")
        run_ids = {run_id for run_id, _, _ in by_key}
        expected = {
            (run_id, scenario_id, 2021)
            for run_id in run_ids
            for scenario_id in CORRUPTIONS
        }
        if set(by_key) != expected:
            raise ValueError("engineering aggregation requires eight scenarios per run")

    populations: dict[tuple[str, int], tuple[int, int]] = {}
    for (run_id, _scenario_id, year), result in by_key.items():
        metrics = result["metrics"]
        assert isinstance(metrics, dict)
        population = (int(metrics["sample_count"]), int(metrics["pixel_count"]))
        population_key = (run_id, year)
        if population_key in populations and populations[population_key] != population:
            raise ValueError("evaluation population drifted between scenarios")
        populations[population_key] = population

    baselines = {
        (run_id, year): result["metrics"]
        for (run_id, scenario_id, year), result in by_key.items()
        if scenario_id == "M00"
    }
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for (_run_id, scenario_id, year), result in by_key.items():
        task = result["task"]
        assert isinstance(task, dict)
        group_key = (str(task["experiment_id"]), scenario_id, year)
        grouped.setdefault(group_key, []).append(result)

    rows: list[dict[str, object]] = []
    for experiment_id in ("C00", "C02"):
        for year in ((2021,) if mode == "engineering" else (2022, 2023)):
            for scenario_id in CORRUPTIONS:
                group = grouped.get((experiment_id, scenario_id, year), [])
                if not group:
                    continue
                group.sort(key=lambda item: int(item["task"]["seed"]))  # type: ignore[index]
                metric_summary: dict[str, object] = {}
                for name in METRIC_NAMES:
                    values: list[float] = []
                    deltas: list[float] = []
                    for result in group:
                        task = result["task"]
                        metrics = result["metrics"]
                        assert isinstance(task, dict) and isinstance(metrics, dict)
                        value = float(metrics[name])
                        baseline = baselines[(str(task["run_id"]), year)]
                        assert isinstance(baseline, dict)
                        values.append(value)
                        deltas.append(value - float(baseline[name]))
                    mean, std = _mean_std(values)
                    delta_mean, delta_std = _mean_std(deltas)
                    metric_summary[name] = {
                        "mean": mean,
                        "std": std,
                        "delta_from_m00_mean": delta_mean,
                        "delta_from_m00_std": delta_std,
                    }
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "scenario_id": scenario_id,
                        "year": year,
                        "seed_count": len(group),
                        "metrics": metric_summary,
                    }
                )

    return {
        "schema_version": 1,
        "status": "pass",
        "mode": mode,
        "scientific_claim": mode == "formal",
        "manifest": manifests.pop(),
        "result_count": len(results),
        "cross_year_comparison": False,
        "rows": rows,
    }


def write_summary_new(path: Path, payload: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = aggregate_results(args.result)
    write_summary_new(args.output, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
