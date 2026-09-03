from __future__ import annotations

import json
from pathlib import Path

import pytest

from reproductions.wsts_fast_track.complete_corrected_baseline import (
    finalize_corrected_baseline,
)


def _run_root(tmp_path: Path, *, complete: bool = True) -> Path:
    root = tmp_path / "run"
    checkpoint = root / "work" / "logs" / "best.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    marker = "`Trainer.fit` stopped: `max_steps=3000` reached." if complete else ""
    (root / "training.log").write_text(
        "\n".join(
            (
                marker,
                "val_avg_precision 0.51",
                "val_f1 0.42",
                "val_loss 0.07",
                "WSTS_OBSERVER_PEAK_ALLOCATED_BYTES=123456",
            )
        ),
        encoding="utf-8",
    )
    return root


def test_finalize_corrected_baseline_records_from_scratch_contract(
    tmp_path: Path,
) -> None:
    root = _run_root(tmp_path)

    record = finalize_corrected_baseline(
        "B0",
        root,
        started_at_epoch=100,
        slurm_job_id="123",
        now_epoch=160,
        checkpoint_loader=lambda _path: {"global_step": 2999},
    )

    assert record["baseline_id"] == "B0"
    assert record["experiment"] == "C00"
    assert record["training_policy"] == "clean"
    assert record["max_steps"] == 3_000
    assert record["seed"] == 0
    assert record["corrected_index"] is True
    assert record["initialization"] == "from_scratch"
    assert record["validation_years"] == [2021]
    assert record["test_enabled"] is False
    assert record["metrics"] == {
        "val_avg_precision": 0.51,
        "val_f1": 0.42,
        "val_loss": 0.07,
    }
    assert json.loads((root / "completed.json").read_text()) == record


def test_finalize_corrected_baseline_rejects_incomplete_or_existing_output(
    tmp_path: Path,
) -> None:
    incomplete = _run_root(tmp_path / "incomplete", complete=False)
    with pytest.raises(ValueError, match="3000-step completion marker"):
        finalize_corrected_baseline(
            "B2",
            incomplete,
            started_at_epoch=100,
            slurm_job_id="123",
            now_epoch=160,
            checkpoint_loader=lambda _path: {"global_step": 2999},
        )

    complete = _run_root(tmp_path / "existing")
    (complete / "completed.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        finalize_corrected_baseline(
            "B3",
            complete,
            started_at_epoch=100,
            slurm_job_id="123",
            now_epoch=160,
            checkpoint_loader=lambda _path: {"global_step": 2999},
        )
