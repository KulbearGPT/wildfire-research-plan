import json
import sys
from pathlib import Path

import pytest


SCRIPTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "reproductions"
    / "wsts_res18_unet_t1"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

from verify_calibration import (  # noqa: E402
    independent_epoch_boundaries,
    independent_epoch_projection,
    independent_effective_config,
    independent_progress,
    independent_timing,
)


def _event(seconds: float, text: str) -> str:
    return json.dumps({"seconds": seconds, "stream": "stdout", "text": text})


def test_independent_progress_handles_lightning_completed_epoch_reset() -> None:
    events = "\n".join(
        [
            _event(1.0, "Epoch 0: 0%| | 0/2"),
            _event(2.0, "Epoch 0: 50%| | 1/2"),
            _event(3.0, "Epoch 0: 100%| | 2/2"),
            _event(4.0, "Epoch 0: 0%| | 0/2"),
            _event(5.0, "Epoch 1: 0%| | 0/2"),
            _event(6.0, "Epoch 1: 50%| | 1/2"),
        ]
    )

    assert independent_progress(events) == [(0, 1.0), (1, 2.0), (2, 3.0), (3, 6.0)]


def test_independent_timing_excludes_warmup_through_step49() -> None:
    result = independent_timing(
        [(0, 0.0), (49, 4.9), (50, 5.2), (51, 5.6), (52, 6.0)],
        wall_seconds=10.0,
        validation_seconds=1.0,
    )

    assert result["median_step_seconds"] == pytest.approx(0.4)
    assert result["instantaneous_samples_per_second"] == pytest.approx(160.0)
    assert result["compute_only_10000_seconds"] == pytest.approx(4000.0)


def test_independent_epoch_projection_recomputes_boundaries_cycles_and_wall_model() -> None:
    events = "\n".join(
        [
            _event(1.0, "Epoch 0: 0%| | 0/2"),
            _event(2.0, "Epoch 0: 50%| | 1/2"),
            _event(3.0, "Epoch 0: 100%| | 2/2"),
            _event(4.0, "Epoch 0: 0%| | 0/2"),
            _event(5.0, "Epoch 1: 0%| | 0/2"),
            _event(6.0, "Epoch 1: 50%| | 1/2"),
            _event(7.0, "Epoch 1: 100%| | 2/2"),
            _event(8.0, "Epoch 1: 0%| | 0/2"),
            _event(9.0, "Epoch 2: 0%| | 0/2"),
            _event(10.0, "Epoch 2: 50%| | 1/2"),
        ]
    )
    boundaries = independent_epoch_boundaries(events)
    result = independent_epoch_projection(
        boundaries,
        startup_seconds=1.0,
        median_step_seconds=0.1,
        p25_step_seconds=0.08,
        p75_step_seconds=0.12,
        wall_seconds=10.0,
        observed_steps=5,
    )

    assert boundaries == [(0, 1, 2.0, 2), (1, 3, 6.0, 2), (2, 5, 10.0, 2)]
    assert result["epoch_cycle_seconds"] == [4.0, 4.0]
    assert result["epoch_aware_10000_central_seconds"] == pytest.approx(20_001.0)
    assert result["naive_wall_linear_10000_seconds"] == pytest.approx(20_000.0)


def test_independent_effective_config_reads_dynamic_positive_weight(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "model:\n  init_args:\n    pos_class_weight: 608.4653828020165\n",
        encoding="utf-8",
    )

    assert independent_effective_config(config) == 608.4653828020165
