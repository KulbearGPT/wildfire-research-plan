from __future__ import annotations

import json
from pathlib import Path

import pytest

from reproductions.wsts_fast_track import completion


@pytest.mark.parametrize(
    ("run_id", "experiment", "seed"),
    [
        ("C00-S0-10K", "C00", 0),
        ("C00-S1-10K", "C00", 1),
        ("C00-S2-10K", "C00", 2),
        ("C02-S0-10K", "C02", 0),
        ("C02-S1-10K", "C02", 1),
        ("C02-S2-10K", "C02", 2),
    ],
)
def test_completion_identity_accepts_only_declared_10k_runs(
    run_id: str, experiment: str, seed: int
) -> None:
    assert completion.run_identity(run_id) == (experiment, seed)


@pytest.mark.parametrize("run_id", ["C00-S0-3K", "C02-S0-3K", "unknown"])
def test_completion_identity_rejects_screening_or_unknown_runs(run_id: str) -> None:
    with pytest.raises(ValueError, match="10K"):
        completion.run_identity(run_id)


def _synthetic_run(root: Path) -> None:
    checkpoint = root / "work" / "checkpoints" / "best.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    (root / "training.log").write_text(
        "val_loss=0.0061, val_avg_precision=0.61, val_f1=0.47\n"
        "WSTS_OBSERVER_PEAK_ALLOCATED_BYTES=456789\n"
        "`Trainer.fit` stopped: `max_steps=10000` reached.\n",
        encoding="utf-8",
    )


def test_finalize_run_seals_replication_evidence_without_test_claim(
    tmp_path: Path,
) -> None:
    _synthetic_run(tmp_path)

    record = completion.finalize_run(
        "C02-S2-10K",
        tmp_path,
        started_at_epoch=100,
        slurm_job_id="12345",
        now_epoch=223,
        checkpoint_loader=lambda _path: {"global_step": 9876},
    )

    assert record == json.loads((tmp_path / "completed.json").read_text())
    assert record["status"] == "pass"
    assert record["experiment"] == "C02"
    assert record["seed"] == 2
    assert record["max_steps"] == 10_000
    assert record["test_enabled"] is False
    assert record["withheld_years"] == [2022, 2023]
    assert record["checkpoint_global_step"] == 9876
    assert record["peak_cuda_allocated_bytes"] == 456789
    assert record["wall_seconds"] == 123
    assert record["metrics"] == {
        "val_avg_precision": 0.61,
        "val_f1": 0.47,
        "val_loss": 0.0061,
    }
    assert "test-performance" in record["purpose"]


def test_finalize_run_refuses_to_replace_existing_record(tmp_path: Path) -> None:
    _synthetic_run(tmp_path)
    kwargs = {
        "started_at_epoch": 100,
        "slurm_job_id": "12345",
        "now_epoch": 223,
        "checkpoint_loader": lambda _path: {"global_step": 9876},
    }
    completion.finalize_run("C00-S1-10K", tmp_path, **kwargs)

    with pytest.raises(FileExistsError):
        completion.finalize_run("C00-S1-10K", tmp_path, **kwargs)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("marker", "completion marker"),
        ("peak", "CUDA peak"),
        ("checkpoint", "exactly one"),
    ],
)
def test_finalize_run_rejects_incomplete_training_evidence(
    tmp_path: Path, mutation: str, message: str
) -> None:
    _synthetic_run(tmp_path)
    log_path = tmp_path / "training.log"
    if mutation == "marker":
        log_path.write_text("WSTS_OBSERVER_PEAK_ALLOCATED_BYTES=1\n")
    elif mutation == "peak":
        log_path.write_text(
            "`Trainer.fit` stopped: `max_steps=10000` reached.\n"
        )
    else:
        (tmp_path / "work" / "second.ckpt").touch()

    with pytest.raises(ValueError, match=message):
        completion.finalize_run(
            "C00-S2-10K",
            tmp_path,
            started_at_epoch=100,
            slurm_job_id="12345",
            now_epoch=223,
            checkpoint_loader=lambda _path: {"global_step": 9876},
        )
