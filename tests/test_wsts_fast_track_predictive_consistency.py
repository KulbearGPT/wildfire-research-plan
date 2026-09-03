from __future__ import annotations

import numpy as np
import pytest
import subprocess
import torch
from pathlib import Path

from reproductions.wsts_fast_track.evaluate_predictive_consistency import (
    validate_d1_checkpoint,
)

from reproductions.wsts_fast_track.predictive_consistency import (
    CleanCorruptPairDataset,
    bernoulli_kl_from_logits,
)
from reproductions.wsts_fast_track.train_predictive_consistency import (
    paired_training_objective,
    validate_b3_record,
)


def test_bernoulli_kl_is_stable_and_only_updates_corrupt_prediction() -> None:
    clean = torch.tensor([-20.0, -1.0, 1.0, 20.0], requires_grad=True)
    same = clean.detach().clone().requires_grad_(True)
    identical = bernoulli_kl_from_logits(clean, same)
    assert float(identical.detach()) == pytest.approx(0.0, abs=1e-7)

    corrupt = torch.tensor([20.0, 1.0, -1.0, -20.0], requires_grad=True)
    loss = bernoulli_kl_from_logits(clean, corrupt)
    assert torch.isfinite(loss)
    assert float(loss.detach()) > 0.0
    loss.backward()
    assert clean.grad is None
    assert corrupt.grad is not None
    assert torch.count_nonzero(corrupt.grad) == corrupt.numel()


class _FakeBaseDataset:
    is_train = True
    remove_duplicate_features = False
    features_to_keep = None
    return_doy = False

    def __len__(self) -> int:
        return 1

    def find_image_index_from_dataset_index(self, index: int):
        assert index == 0
        return 2016, "fire", 0

    def load_imgs(self, _year: int, _name: str, _index: int):
        x = np.ones((1, 23, 4, 4), dtype=np.float32)
        y = np.arange(16, dtype=np.float32).reshape(4, 4)
        return x, y

    def preprocess_and_augment(self, x: np.ndarray, y: np.ndarray):
        offset = float(np.random.random())
        return torch.from_numpy(x.copy()) + offset, torch.from_numpy(y.copy()) + offset


def test_clean_corrupt_pair_reuses_exact_augmentation_and_target() -> None:
    dataset = CleanCorruptPairDataset(
        _FakeBaseDataset(),
        fire_probability=1.0,
        block_probability=0.0,
    )

    clean, corrupt, target = dataset[0]

    torch.testing.assert_close(clean[:, :22], corrupt[:, :22])
    assert torch.count_nonzero(clean[:, 22]) == clean[:, 22].numel()
    torch.testing.assert_close(clean[:, 22] - corrupt[:, 22], torch.ones_like(clean[:, 22]))
    expected_offset = target[0, 0]
    torch.testing.assert_close(clean[0, 0, 0, 0], expected_offset + 1.0)
    torch.testing.assert_close(corrupt[0, 0, 0, 0], expected_offset + 1.0)


def test_paired_objective_changes_only_by_declared_consistency_term() -> None:
    class SquaredLoss:
        @staticmethod
        def compute_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return (logits - target).square().mean()

    clean = torch.tensor([0.0, 1.0])
    corrupt = torch.tensor([1.0, -1.0])
    target = torch.tensor([0.0, 0.0])
    control, control_parts = paired_training_objective(
        SquaredLoss(), clean, corrupt, target, lambda_consistency=0.0
    )
    method, method_parts = paired_training_objective(
        SquaredLoss(), clean, corrupt, target, lambda_consistency=0.1
    )

    assert control_parts["supervised"] == pytest.approx(0.75)
    assert control_parts["consistency"] > 0.0
    assert float(method - control) == pytest.approx(
        0.1 * method_parts["consistency"]
    )


def test_d1_requires_the_exact_corrected_b3_record() -> None:
    record = {
        "schema_version": 1,
        "status": "pass",
        "baseline_id": "B3",
        "experiment": "C00",
        "training_policy": "fire-block",
        "seed": 0,
        "max_steps": 3_000,
        "corrected_index": True,
        "initialization": "from_scratch",
        "validation_years": [2021],
        "test_enabled": False,
        "checkpoint": "/b3.ckpt",
    }
    assert validate_b3_record(record) == "/b3.ckpt"

    record["initialization"] = "fine_tune"
    with pytest.raises(ValueError, match="B3 completion record"):
        validate_b3_record(record)


def test_d1_evaluation_requires_a_matched_checkpoint() -> None:
    payload = {
        "schema_version": 1,
        "status": "pass",
        "candidate_id": "D1-KL",
        "matched_pair": "D1",
        "experiment": "C00",
        "steps": 3_000,
        "seed": 0,
        "lambda_consistency": 0.1,
        "hyper_parameters": {"n_channels": 40},
        "state_dict": {"weight": torch.tensor(1.0)},
    }
    assert validate_d1_checkpoint(payload) == "D1-KL"

    payload["steps"] = 10_000
    with pytest.raises(ValueError, match="D1 checkpoint"):
        validate_d1_checkpoint(payload)


def test_d1_runner_has_valid_shell_syntax_and_no_nested_submission(
    tmp_path: Path,
) -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "reproductions"
        / "wsts_fast_track"
        / "run_predictive_consistency_on_nibi.sh"
    )
    result = subprocess.run(
        ["bash", "-n", str(runner)], text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    text = runner.read_text(encoding="utf-8")
    assert "sbatch" not in text
    assert "train_predictive_consistency" in text
    assert "evaluate_predictive_consistency" in text
    assert "archive --format=tar HEAD" in text

    record = tmp_path / "b3.json"
    record.write_text("{}\n", encoding="utf-8")
    invalid = subprocess.run(
        [str(runner), str(record), "other"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert invalid.returncode == 2
    assert "variant must be erm or kl" in invalid.stderr
