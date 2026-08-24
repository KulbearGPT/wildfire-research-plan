from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from reproductions.wsts_fast_track import evaluate_missingness


class PerfectModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1]

    @staticmethod
    def compute_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, target.float())


def test_checkpoint_init_args_do_not_forward_c02_fixed_use_doy() -> None:
    hyperparameters = {
        "encoder_name": "resnet18",
        "encoder_weights": "imagenet",
        "n_channels": 5,
        "use_doy": False,
    }

    result = evaluate_missingness.checkpoint_init_args(
        hyperparameters,
        experiment_id="C02",
    )

    assert result == {
        "encoder_name": "resnet18",
        "encoder_weights": None,
        "n_channels": 5,
    }


def test_evaluate_batches_computes_exact_streaming_binary_metrics() -> None:
    logits = torch.tensor(
        [
            [[[-10.0, 10.0], [10.0, -10.0]]],
            [[[10.0, -10.0], [-10.0, 10.0]]],
        ]
    )
    target = torch.tensor(
        [
            [[0, 1], [1, 0]],
            [[1, 0], [0, 1]],
        ]
    )

    result = evaluate_missingness.evaluate_batches(
        PerfectModel(), [(logits, target)], device=torch.device("cpu")
    )

    assert result["sample_count"] == 2
    assert result["pixel_count"] == 8
    assert result["avg_precision"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)
    assert result["iou"] == pytest.approx(1.0)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["loss"] < 0.001


def _manifest(mode: str = "engineering") -> dict[str, object]:
    formal = mode == "formal"
    run_ids = (
        "C00-S0-10K",
        "C00-S1-10K",
        "C00-S2-10K",
        "C02-S0-10K",
        "C02-S1-10K",
        "C02-S2-10K",
    )
    records = [
        {
            "run_id": run_id,
            "checkpoint": "/checkpoint" if index == 0 else f"/checkpoint-{index}",
        }
        for index, run_id in enumerate(run_ids[: 6 if formal else 1])
    ]
    year = 2022 if formal else 2021
    return {
        "schema_version": 1,
        "corruption_schema_version": 1,
        "mode": mode,
        "scientific_claim": formal,
        "heldout_access": formal,
        "years": [2022, 2023] if formal else [2021],
        "records": records,
        "tasks": [
            {
                "evaluation_id": "C00-S0-10K-M03-Y2021"
                if not formal
                else "C00-S0-10K-M03-Y2022",
                "run_id": "C00-S0-10K",
                "experiment_id": "C00",
                "seed": 0,
                "checkpoint": "/checkpoint",
                "scenario_id": "M03",
                "matrix_seed": 0,
                "year": year,
                "output": "/result.json",
            }
        ],
    }


def test_select_task_keeps_engineering_on_2021() -> None:
    selected = evaluate_missingness.select_task(
        _manifest(), "C00-S0-10K-M03-Y2021"
    )

    assert selected.heldout_authorized is False
    assert selected.task["year"] == 2021


def test_select_task_requires_complete_formal_gate_for_heldout() -> None:
    manifest = _manifest("formal")
    selected = evaluate_missingness.select_task(
        manifest, "C00-S0-10K-M03-Y2022"
    )
    assert selected.heldout_authorized is True

    manifest["records"] = list(manifest["records"])[:-1]
    with pytest.raises(ValueError, match="six"):
        evaluate_missingness.select_task(
            manifest, "C00-S0-10K-M03-Y2022"
        )


def test_select_task_rejects_unknown_duplicate_or_year_escape() -> None:
    manifest = _manifest()
    with pytest.raises(ValueError, match="not found"):
        evaluate_missingness.select_task(manifest, "unknown")

    manifest["tasks"] = list(manifest["tasks"]) * 2
    with pytest.raises(ValueError, match="exactly once"):
        evaluate_missingness.select_task(
            manifest, "C00-S0-10K-M03-Y2021"
        )

    manifest = _manifest()
    manifest["tasks"][0]["year"] = 2022
    with pytest.raises(ValueError, match="2021"):
        evaluate_missingness.select_task(
            manifest, "C00-S0-10K-M03-Y2021"
        )


def test_write_result_never_replaces_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    evaluate_missingness.write_result_new(output, {"status": "pass"})
    assert json.loads(output.read_text()) == {"status": "pass"}

    with pytest.raises(FileExistsError):
        evaluate_missingness.write_result_new(output, {"status": "pass"})
