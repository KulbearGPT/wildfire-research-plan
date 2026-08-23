from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "reproductions" / "wsts_fast_track" / "run_replication_on_nibi.sh"


def test_replication_runner_has_valid_shell_syntax_and_no_submission() -> None:
    result = subprocess.run(
        ["bash", "-n", str(RUNNER)], text=True, capture_output=True, check=False
    )

    assert result.returncode == 0, result.stderr
    text = RUNNER.read_text(encoding="utf-8")
    assert "sbatch" not in text
    assert "reproductions.wsts_fast_track.completion" in text
    assert "archive --format=tar HEAD" in text


def test_replication_runner_is_limited_to_declared_seed_one_and_two_runs() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    for run_id in (
        "C00-S1-10K",
        "C00-S2-10K",
        "C02-S1-10K",
        "C02-S2-10K",
    ):
        assert run_id in text
    assert "C00-S0-10K|C02-S0-10K" not in text
    assert "manifest" in text
    assert '"test_enabled": false' in text
