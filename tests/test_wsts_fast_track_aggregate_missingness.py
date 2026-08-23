from __future__ import annotations

import json
from pathlib import Path

import pytest

from reproductions.wsts_fast_track import aggregate_missingness, matrix


RUNS = (
    "C00-S0-10K",
    "C00-S1-10K",
    "C00-S2-10K",
    "C02-S0-10K",
    "C02-S1-10K",
    "C02-S2-10K",
)


def _result(
    tmp_path: Path,
    run_id: str,
    scenario_id: str,
    year: int,
    *,
    mode: str,
) -> Path:
    spec = matrix.run_spec(run_id)
    scenario_index = int(scenario_id[1:])
    base_ap = 0.60 + 0.01 * spec.seed + (0.02 if spec.experiment_id == "C02" else 0)
    metrics = {
        "sample_count": 100,
        "pixel_count": 1000,
        "avg_precision": base_ap - 0.01 * scenario_index,
        "f1": 0.5 - 0.01 * scenario_index,
        "iou": 0.4 - 0.01 * scenario_index,
        "precision": 0.6 - 0.01 * scenario_index,
        "recall": 0.5 - 0.01 * scenario_index,
        "loss": 0.1 + 0.01 * scenario_index,
    }
    path = tmp_path / f"{run_id}-{scenario_id}-Y{year}.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "pass",
                "mode": mode,
                "scientific_claim": mode == "formal",
                "manifest": "/manifest.json",
                "task": {
                    "evaluation_id": path.stem,
                    "run_id": run_id,
                    "experiment_id": spec.experiment_id,
                    "seed": spec.seed,
                    "checkpoint": f"/{run_id}.ckpt",
                    "scenario_id": scenario_id,
                    "matrix_seed": 0,
                    "year": year,
                    "output": str(path),
                },
                "metrics": metrics,
                "boundary": {
                    "is_train": False,
                    "retraining": False,
                    "checkpoint_selection": False,
                    "natural_missingness_claim": False,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_formal_aggregation_requires_and_summarizes_complete_96_results(
    tmp_path: Path,
) -> None:
    paths = [
        _result(tmp_path, run_id, scenario_id, year, mode="formal")
        for run_id in RUNS
        for scenario_id in matrix.CORRUPTIONS
        for year in (2022, 2023)
    ]

    summary = aggregate_missingness.aggregate_results(paths)

    assert summary["mode"] == "formal"
    assert summary["result_count"] == 96
    assert len(summary["rows"]) == 32
    row = next(
        item
        for item in summary["rows"]
        if item["experiment_id"] == "C00"
        and item["scenario_id"] == "M03"
        and item["year"] == 2022
    )
    assert row["seed_count"] == 3
    assert row["metrics"]["avg_precision"]["mean"] == pytest.approx(0.58)
    assert row["metrics"]["avg_precision"]["delta_from_m00_mean"] == pytest.approx(
        -0.03
    )
    assert row["metrics"]["avg_precision"]["std"] == pytest.approx(0.01)


def test_formal_aggregation_rejects_missing_task(tmp_path: Path) -> None:
    paths = [
        _result(tmp_path, run_id, scenario_id, year, mode="formal")
        for run_id in RUNS
        for scenario_id in matrix.CORRUPTIONS
        for year in (2022, 2023)
    ]

    with pytest.raises(ValueError, match="96"):
        aggregate_missingness.aggregate_results(paths[:-1])


def test_engineering_aggregation_accepts_one_complete_checkpoint_matrix(
    tmp_path: Path,
) -> None:
    paths = [
        _result(tmp_path, "C00-S0-10K", scenario_id, 2021, mode="engineering")
        for scenario_id in matrix.CORRUPTIONS
    ]

    summary = aggregate_missingness.aggregate_results(paths)

    assert summary["mode"] == "engineering"
    assert summary["scientific_claim"] is False
    assert summary["result_count"] == 8
    assert len(summary["rows"]) == 8


def test_aggregation_rejects_population_drift_between_scenarios(
    tmp_path: Path,
) -> None:
    paths = [
        _result(tmp_path, "C00-S0-10K", scenario_id, 2021, mode="engineering")
        for scenario_id in matrix.CORRUPTIONS
    ]
    payload = json.loads(paths[-1].read_text())
    payload["metrics"]["sample_count"] = 99
    paths[-1].write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="population"):
        aggregate_missingness.aggregate_results(paths)


def test_summary_writer_never_replaces_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    aggregate_missingness.write_summary_new(output, {"status": "pass"})
    with pytest.raises(FileExistsError):
        aggregate_missingness.write_summary_new(output, {"status": "pass"})
