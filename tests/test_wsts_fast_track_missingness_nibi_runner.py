from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "reproductions" / "wsts_fast_track" / "run_missingness_on_nibi.sh"


def test_missingness_runner_has_valid_shell_syntax_and_no_submission() -> None:
    result = subprocess.run(
        ["bash", "-n", str(RUNNER)], text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    text = RUNNER.read_text(encoding="utf-8")
    assert "sbatch" not in text
    assert "reproductions.wsts_fast_track.evaluate_missingness" in text
    assert "archive --format=tar HEAD" in text
    assert "WANDB_MODE=disabled" in text


def test_missingness_runner_requires_manifest_and_evaluation_id() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "MISSINGNESS_MANIFEST EVALUATION_ID" in text
    assert "--manifest" in text
    assert "--evaluation-id" in text
    assert "--device cuda" in text
