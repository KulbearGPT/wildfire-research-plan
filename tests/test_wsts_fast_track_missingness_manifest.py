from __future__ import annotations

import json
from pathlib import Path

import pytest

from reproductions.wsts_fast_track import matrix, missingness_manifest


TEN_K_RUNS = (
    "C00-S0-10K",
    "C00-S1-10K",
    "C00-S2-10K",
    "C02-S0-10K",
    "C02-S1-10K",
    "C02-S2-10K",
)


def _record(tmp_path: Path, run_id: str) -> Path:
    spec = matrix.run_spec(run_id)
    checkpoint = tmp_path / "checkpoints" / f"{run_id}.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(run_id.encode("ascii"))
    path = tmp_path / "records" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "purpose": "10000-step clean run; not a test-performance claim",
                "experiment": spec.experiment_id,
                "slurm_job_id": f"job-{run_id}",
                "max_steps": 10_000,
                "seed": spec.seed,
                "train_years": [2016, 2017, 2018, 2019, 2020],
                "validation_years": [2021],
                "test_enabled": False,
                "withheld_years": [2022, 2023],
                "checkpoint": str(checkpoint),
                "checkpoint_global_step": 9_500,
                "peak_cuda_allocated_bytes": 1,
                "wall_seconds": 100,
                "metrics": {
                    "val_avg_precision": 0.5,
                    "val_f1": 0.4,
                    "val_loss": 0.1,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_engineering_manifest_allows_one_10k_checkpoint_and_only_2021(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path, "C00-S0-10K")

    manifest = missingness_manifest.engineering_manifest(
        {"C00-S0-10K": record}, output_root=tmp_path / "results"
    )

    assert manifest["mode"] == "engineering"
    assert manifest["scientific_claim"] is False
    assert manifest["years"] == [2021]
    assert manifest["heldout_access"] is False
    assert [task["scenario_id"] for task in manifest["tasks"]] == list(
        matrix.CORRUPTIONS
    )
    assert len(manifest["tasks"]) == 8
    assert all(task["year"] == 2021 for task in manifest["tasks"])
    assert "sbatch" not in json.dumps(manifest)


def test_formal_manifest_requires_all_six_clean_replications(tmp_path: Path) -> None:
    incomplete = {run_id: _record(tmp_path, run_id) for run_id in TEN_K_RUNS[:-1]}

    with pytest.raises(ValueError, match="six"):
        missingness_manifest.formal_manifest(
            incomplete, output_root=tmp_path / "results"
        )


def test_formal_manifest_renders_ordered_96_task_matrix(tmp_path: Path) -> None:
    records = {run_id: _record(tmp_path, run_id) for run_id in TEN_K_RUNS}

    manifest = missingness_manifest.formal_manifest(
        records, output_root=tmp_path / "results"
    )

    assert manifest["mode"] == "formal"
    assert manifest["scientific_claim"] is True
    assert manifest["years"] == [2022, 2023]
    assert manifest["heldout_access"] is True
    assert len(manifest["records"]) == 6
    assert [item["experiment_id"] for item in manifest["clean_validation_summary"]] == [
        "C00",
        "C02",
    ]
    assert all(item["seed_count"] == 3 for item in manifest["clean_validation_summary"])
    assert len(manifest["tasks"]) == 96
    expected = [
        (run_id, scenario_id, year)
        for run_id in TEN_K_RUNS
        for scenario_id in matrix.CORRUPTIONS
        for year in (2022, 2023)
    ]
    actual = [
        (task["run_id"], task["scenario_id"], task["year"])
        for task in manifest["tasks"]
    ]
    assert actual == expected
    assert len({task["output"] for task in manifest["tasks"]}) == 96


def test_manifest_rejects_record_identity_or_missing_checkpoint(tmp_path: Path) -> None:
    path = _record(tmp_path, "C00-S1-10K")
    payload = json.loads(path.read_text())
    payload["seed"] = 2
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="seed"):
        missingness_manifest.engineering_manifest(
            {"C00-S1-10K": path}, output_root=tmp_path / "results"
        )

    path = _record(tmp_path, "C00-S1-10K")
    payload = json.loads(path.read_text())
    Path(payload["checkpoint"]).unlink()
    with pytest.raises(ValueError, match="checkpoint file"):
        missingness_manifest.engineering_manifest(
            {"C00-S1-10K": path}, output_root=tmp_path / "results"
        )


def test_manifest_writer_never_replaces_existing_output(tmp_path: Path) -> None:
    manifest = missingness_manifest.engineering_manifest(
        {"C00-S0-10K": _record(tmp_path, "C00-S0-10K")},
        output_root=tmp_path / "results",
    )
    output = tmp_path / "manifest.json"
    missingness_manifest.write_manifest_new(output, manifest)

    with pytest.raises(FileExistsError):
        missingness_manifest.write_manifest_new(output, manifest)
