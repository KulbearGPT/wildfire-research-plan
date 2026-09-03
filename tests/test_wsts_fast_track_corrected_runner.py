from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from reproductions.wsts_fast_track.evaluate_corrected_baseline import (
    SCREEN_SCENARIOS,
    validate_evaluation_record,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "reproductions"
    / "wsts_fast_track"
    / "run_corrected_baseline_on_nibi.sh"
)


def _record(baseline_id: str = "B0") -> dict[str, object]:
    experiment = "C02" if baseline_id == "B1" else "C00"
    policy = {
        "B0": "clean",
        "B1": "clean",
        "B2": "fire",
        "B3": "fire-block",
        "B4": "year-balanced-fire-block",
    }[baseline_id]
    return {
        "schema_version": 1,
        "status": "pass",
        "baseline_id": baseline_id,
        "experiment": experiment,
        "training_policy": policy,
        "seed": 0,
        "max_steps": 3_000,
        "corrected_index": True,
        "initialization": "from_scratch",
        "validation_years": [2021],
        "test_enabled": False,
        "checkpoint": "/checkpoint.ckpt",
    }


def test_evaluation_record_requires_exact_corrected_from_scratch_contract() -> None:
    spec = validate_evaluation_record(_record("B3"))
    assert spec.baseline_id == "B3"
    assert spec.training_policy == "fire-block"
    assert SCREEN_SCENARIOS == ("M00", "M01", "M06", "M07")

    for key, wrong in (
        ("corrected_index", False),
        ("initialization", "fine_tune"),
        ("max_steps", 10_000),
        ("validation_years", [2022]),
        ("test_enabled", True),
    ):
        record = _record("B0")
        record[key] = wrong
        with pytest.raises(ValueError, match="completion record"):
            validate_evaluation_record(record)


def test_evaluation_record_rejects_mismatched_baseline_identity() -> None:
    record = _record("B1")
    record["experiment"] = "C00"
    with pytest.raises(ValueError, match="completion record"):
        validate_evaluation_record(record)


def test_corrected_runner_is_safe_and_self_contained() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(RUNNER)], text=True, capture_output=True, check=False
    )
    assert syntax.returncode == 0, syntax.stderr
    text = RUNNER.read_text(encoding="utf-8")
    assert "sbatch" not in text
    assert "B0|B1|B2|B3|B4" in text
    assert "archive --format=tar HEAD" in text
    assert "train_corrected_baseline" in text
    assert "complete_corrected_baseline" in text
    assert "evaluate_corrected_baseline" in text
    assert 'SLURM_SUBMIT_DIR:-$PWD' in text
