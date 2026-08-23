from __future__ import annotations

import json
from pathlib import Path

import pytest

from reproductions.wsts_fast_track import matrix, promotion


def _completed(experiment: str, job_id: str) -> dict[str, object]:
    return {
        "status": "pass",
        "purpose": "3000-step screening; not a clean-performance claim",
        "experiment": experiment,
        "slurm_job_id": job_id,
        "max_steps": 3_000,
        "seed": 0,
        "train_years": [2016, 2017, 2018, 2019, 2020],
        "validation_years": [2021],
        "test_enabled": False,
        "withheld_years": [2022, 2023],
        "checkpoint": f"/runs/{experiment}/best.ckpt",
        "checkpoint_global_step": 2_800,
        "peak_cuda_allocated_bytes": 1_234,
        "wall_seconds": 100,
        "metrics": {
            "val_avg_precision": 0.3,
            "val_f1": 0.2,
            "val_loss": 0.8,
        },
    }


def _write_results(tmp_path: Path) -> tuple[Path, Path]:
    paths: list[Path] = []
    for experiment, job_id in (("C00", "100"), ("C02", "102")):
        path = tmp_path / f"{experiment}.json"
        path.write_text(
            json.dumps(_completed(experiment, job_id)), encoding="utf-8"
        )
        paths.append(path)
    return paths[0], paths[1]


def _manifest(tmp_path: Path, paths: tuple[Path, ...]) -> dict[str, object]:
    return promotion.promotion_manifest(
        "C00-S0-10K",
        paths,
        upstream_root=tmp_path / "upstream",
        data_root=tmp_path / "data",
        run_root=tmp_path / "run",
        stats_path=tmp_path / "stats.npz",
    )


def test_passing_screening_pair_unlocks_seed_zero_10k_manifest(
    tmp_path: Path,
) -> None:
    c00, c02 = _write_results(tmp_path)

    payload = _manifest(tmp_path, (c02, c00))

    assert payload["schema_version"] == 1
    assert payload["run"]["run_id"] == "C00-S0-10K"
    assert payload["run"]["seed"] == 0
    assert payload["run"]["max_steps"] == 10_000
    assert [item["experiment"] for item in payload["prerequisites"]] == [
        "C00",
        "C02",
    ]
    assert payload["command"] == [
        "python",
        "-m",
        "reproductions.wsts_fast_track.entrypoint",
        "--upstream-root",
        str((tmp_path / "upstream").resolve()),
        "--run-id",
        "C00-S0-10K",
        "--data-root",
        str((tmp_path / "data").resolve()),
        "--run-root",
        str((tmp_path / "run").resolve()),
        "--stats-path",
        str((tmp_path / "stats.npz").resolve()),
    ]
    assert payload["boundary"] == {
        "train_years": [2016, 2017, 2018, 2019, 2020],
        "validation_years": [2021],
        "test_enabled": False,
        "withheld_years": [2022, 2023],
    }


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (("status",), "status"),
        (("experiment",), "experiment"),
        (("max_steps",), "max_steps"),
        (("seed",), "seed"),
        (("train_years",), "train_years"),
        (("validation_years",), "validation_years"),
        (("test_enabled",), "test_enabled"),
        (("withheld_years",), "withheld_years"),
        (("metrics", "val_avg_precision"), "val_avg_precision"),
        (("metrics", "val_f1"), "val_f1"),
        (("metrics", "val_loss"), "val_loss"),
        (("peak_cuda_allocated_bytes",), "peak_cuda_allocated_bytes"),
        (("checkpoint",), "checkpoint"),
        (("slurm_job_id",), "slurm_job_id"),
        (("checkpoint_global_step",), "checkpoint_global_step"),
    ],
)
def test_screening_gate_rejects_invalid_fields(
    tmp_path: Path, mutation: tuple[str, ...], match: str
) -> None:
    c00, c02 = _write_results(tmp_path)
    payload = _completed("C00", "100")
    bad_values: dict[tuple[str, ...], object] = {
        ("status",): "failed",
        ("experiment",): "C02",
        ("max_steps",): 2_999,
        ("seed",): 1,
        ("train_years",): [2017, 2018, 2019, 2020],
        ("validation_years",): [2020],
        ("test_enabled",): True,
        ("withheld_years",): [2023],
        ("metrics", "val_avg_precision"): float("nan"),
        ("metrics", "val_f1"): 1.1,
        ("metrics", "val_loss"): float("inf"),
        ("peak_cuda_allocated_bytes",): 0,
        ("checkpoint",): "",
        ("slurm_job_id",): "",
        ("checkpoint_global_step",): 3_001,
    }
    cursor: dict[str, object] = payload
    for key in mutation[:-1]:
        cursor = cursor[key]  # type: ignore[assignment]
    cursor[mutation[-1]] = bad_values[mutation]
    c00.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        _manifest(tmp_path, (c00, c02))


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra"])
def test_screening_gate_requires_exactly_one_result_per_prerequisite(
    tmp_path: Path, change: str
) -> None:
    c00, c02 = _write_results(tmp_path)
    if change == "missing":
        paths = (c00,)
    elif change == "duplicate":
        duplicate = tmp_path / "duplicate.json"
        duplicate.write_text(c00.read_text(encoding="utf-8"), encoding="utf-8")
        paths = (c00, duplicate)
    else:
        extra = tmp_path / "extra.json"
        extra.write_text(
            json.dumps(_completed("C99", "199")), encoding="utf-8"
        )
        paths = (c00, c02, extra)

    with pytest.raises(ValueError, match="prerequisite"):
        _manifest(tmp_path, paths)


@pytest.mark.parametrize("schema_change", ["missing", "extra"])
def test_screening_gate_rejects_completed_schema_drift(
    tmp_path: Path, schema_change: str
) -> None:
    c00, c02 = _write_results(tmp_path)
    payload = _completed("C00", "100")
    if schema_change == "missing":
        del payload["purpose"]
    else:
        payload["unexpected"] = True
    c00.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fields"):
        _manifest(tmp_path, (c00, c02))


@pytest.mark.parametrize(
    "run_id", ["C00-S1-10K", "C00-S2-10K", "C02-S1-10K", "C02-S2-10K"]
)
def test_gated_replications_cannot_render(run_id: str, tmp_path: Path) -> None:
    c00, c02 = _write_results(tmp_path)

    with pytest.raises(ValueError, match="not promotable"):
        promotion.promotion_manifest(
            run_id,
            (c00, c02),
            upstream_root=tmp_path / "upstream",
            data_root=tmp_path / "data",
            run_root=tmp_path / "run",
            stats_path=tmp_path / "stats.npz",
        )


def test_unknown_run_cannot_render(tmp_path: Path) -> None:
    c00, c02 = _write_results(tmp_path)

    with pytest.raises(ValueError, match="unknown fast-track run"):
        promotion.promotion_manifest(
            "C99-S0-10K",
            (c00, c02),
            upstream_root=tmp_path / "upstream",
            data_root=tmp_path / "data",
            run_root=tmp_path / "run",
            stats_path=tmp_path / "stats.npz",
        )


def test_matrix_cli_writes_once_without_mutating_existing_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "matrix.json"

    assert promotion.main(["matrix", "--output", str(output)]) == 0
    first = output.read_bytes()
    assert json.loads(first)["schema_version"] == 1

    with pytest.raises(FileExistsError):
        promotion.main(["matrix", "--output", str(output)])
    assert output.read_bytes() == first
    assert json.loads(first) == matrix.matrix_payload()


def test_render_cli_writes_reviewable_manifest(tmp_path: Path) -> None:
    c00, c02 = _write_results(tmp_path)
    output = tmp_path / "promotion.json"

    result = promotion.main(
        [
            "render",
            "--run-id",
            "C02-S0-10K",
            "--result",
            str(c00),
            "--result",
            str(c02),
            "--upstream-root",
            str(tmp_path / "upstream"),
            "--data-root",
            str(tmp_path / "data"),
            "--run-root",
            str(tmp_path / "run"),
            "--stats-path",
            str(tmp_path / "stats.npz"),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["run"]["run_id"] == "C02-S0-10K"
    assert payload["boundary"]["test_enabled"] is False
