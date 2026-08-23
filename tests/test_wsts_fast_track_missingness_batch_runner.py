from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "reproductions" / "wsts_fast_track" / "run_missingness_batch_on_nibi.sh"


def test_batch_runner_has_valid_syntax_and_no_submission() -> None:
    result = subprocess.run(
        ["bash", "-n", str(RUNNER)], text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    text = RUNNER.read_text(encoding="utf-8")
    assert "sbatch" not in text
    assert 'SLURM_SUBMIT_DIR:-$PWD' in text
    assert "run_missingness_on_nibi.sh" in text


def test_batch_runner_selects_manifest_tasks_for_exact_run_id() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "MISSINGNESS_MANIFEST RUN_ID" in text
    assert "C00-S0-10K" in text
    assert "C02-S2-10K" in text
    assert 'task["run_id"] == run_id' in text
    assert "evaluation_id" in text
