from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch

from reproductions.wsts_fast_track.evaluate_standard_reliability_control import (
    validate_d2_checkpoint,
)
from reproductions.wsts_fast_track.processed_reliability import (
    PROCESSED_ACTIVE_FIRE_BINARY,
    PROCESSED_ACTIVE_FIRE_VALUE,
    apply_processed_reliability_corruption,
)


def test_processed_corruption_appends_aligned_invalidity_channel() -> None:
    x = torch.ones(1, 40, 4, 4)
    result = apply_processed_reliability_corruption(
        x,
        fire_drop=False,
        block_fraction=0.5,
        key_digest="00" * 32,
        active_fire_missing_value=-0.25,
    )

    assert result.shape == (1, 41, 4, 4)
    invalid = result[0, 40].bool()
    assert int(invalid.sum()) == 8
    assert torch.all(result[:, PROCESSED_ACTIVE_FIRE_VALUE, invalid] == -0.25)
    assert torch.all(result[:, PROCESSED_ACTIVE_FIRE_BINARY, invalid] == 0.0)
    assert torch.all(result[:, 12:15] == 1.0)


def _payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "pass",
        "candidate_id": "D2-STD",
        "matched_pair": "D2",
        "variant": "standard",
        "experiment": "C00",
        "steps": 3_000,
        "seed": 0,
        "processed_space_matched_corruption": True,
        "hyper_parameters": {"n_channels": 40},
        "state_dict": {"weight": torch.tensor(1.0)},
    }


def test_d2_checkpoint_accepts_only_the_standard_control() -> None:
    payload = _payload()
    assert validate_d2_checkpoint(payload) == "D2-STD"

    payload["candidate_id"] = "D2-RNC"
    with pytest.raises(ValueError, match="D2-STD checkpoint"):
        validate_d2_checkpoint(payload)


def test_d2_runner_has_one_focused_entrypoint(tmp_path: Path) -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "reproductions"
        / "wsts_fast_track"
        / "run_standard_reliability_control_on_nibi.sh"
    )
    syntax = subprocess.run(
        ["bash", "-n", str(runner)], text=True, capture_output=True, check=False
    )
    assert syntax.returncode == 0, syntax.stderr

    record = tmp_path / "b3.json"
    record.write_text("{}\n", encoding="utf-8")
    invalid = subprocess.run(
        [str(runner), str(record), "extra"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert invalid.returncode == 2
    text = runner.read_text(encoding="utf-8")
    assert "train_standard_reliability_control" in text
    assert "evaluate_standard_reliability_control" in text
    assert "rnc" not in text.lower()
