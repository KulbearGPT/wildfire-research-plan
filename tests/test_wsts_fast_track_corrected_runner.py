from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from reproductions.wsts_fast_track.evaluate_corrected_baseline import (
    SCREEN_SCENARIOS,
    evaluation_boundary,
    validate_evaluation_record,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "reproductions"
    / "wsts_fast_track"
    / "run_corrected_baseline_on_nibi.sh"
)
HELDOUT_RUNNER = (
    ROOT
    / "reproductions"
    / "wsts_fast_track"
    / "run_reliability_evaluation_on_nibi.sh"
)


def _record(baseline_id: str = "B0") -> dict[str, object]:
    policy = {
        "B0": "clean",
        "B2": "fire",
        "B3": "fire-block",
    }[baseline_id]
    return {
        "schema_version": 1,
        "status": "pass",
        "baseline_id": baseline_id,
        "experiment": "C00",
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
    record = _record("B2")
    record["experiment"] = "C02"
    with pytest.raises(ValueError, match="completion record"):
        validate_evaluation_record(record)


def test_corrected_runner_is_safe_and_self_contained() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(RUNNER)], text=True, capture_output=True, check=False
    )
    assert syntax.returncode == 0, syntax.stderr
    text = RUNNER.read_text(encoding="utf-8")
    assert "sbatch" not in text
    assert "B0|B2|B3" in text
    assert "archive --format=tar HEAD" in text
    assert "train_corrected_baseline" in text
    assert "complete_corrected_baseline" in text
    assert "evaluate_corrected_baseline" in text
    assert 'SLURM_SUBMIT_DIR:-$PWD' in text


def test_reliability_evaluation_boundary_requires_explicit_heldout_authorization() -> None:
    assert evaluation_boundary(2021, heldout_authorized=False) == (
        "reliability-validation",
        False,
    )
    assert evaluation_boundary(2022, heldout_authorized=True) == (
        "reliability-formal",
        True,
    )
    with pytest.raises(ValueError, match="explicit authorization"):
        evaluation_boundary(2023, heldout_authorized=False)
    with pytest.raises(ValueError, match="2021, 2022, or 2023"):
        evaluation_boundary(2020, heldout_authorized=True)


def test_reliability_evaluation_runner_is_small_and_safe() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(HELDOUT_RUNNER)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr
    text = HELDOUT_RUNNER.read_text(encoding="utf-8")
    assert "sbatch" not in text
    assert "--heldout-authorized" in text
    assert "evaluate_corrected_baseline" in text
    assert "evaluate_predictive_consistency" in text
    assert "evaluate_standard_reliability_control" in text
    assert "evaluate_severity_adaptive_reliability_prompting" in text
