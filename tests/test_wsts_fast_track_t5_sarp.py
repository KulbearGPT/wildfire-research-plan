from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from reproductions.wsts_fast_track.contract import MULTI_FEATURES
from reproductions.wsts_fast_track.evaluate_temporal_reliability_prompting import (
    validate_t5_reliability_checkpoint,
)
from reproductions.wsts_fast_track.temporal_reliability_prompting import (
    TemporalSeverityAdaptiveReliabilityPrompting,
    apply_processed_temporal_reliability_corruption,
)


class _Encoder(torch.nn.Module):
    out_channels = (33, 2, 4)

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(33, 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        shallow = self.conv1(x)
        deep = F.avg_pool2d(torch.cat((shallow, shallow), dim=1), kernel_size=2)
        return x, shallow, deep


class _Decoder(torch.nn.Module):
    def forward(self, _input, shallow: torch.Tensor, deep: torch.Tensor):
        return F.interpolate(deep, scale_factor=2, mode="nearest") + torch.cat(
            (shallow, shallow), dim=1
        )


class _Core(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()
        self.decoder = _Decoder()
        self.segmentation_head = torch.nn.Conv2d(4, 1, 1)


class _LTAE(torch.nn.Module):
    def forward(self, x: torch.Tensor, *, batch_positions: torch.Tensor):
        del batch_positions
        batch, time, _channels, height, width = x.shape
        attention = torch.full(
            (batch, 1, time, height, width),
            1.0 / time,
            dtype=x.dtype,
            device=x.device,
        )
        return x.mean(dim=1), attention


class _Aggregator(torch.nn.Module):
    def forward(self, x: torch.Tensor, *, attn_mask: torch.Tensor):
        del attn_mask
        return x.mean(dim=1)


class _Base(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Core()
        self.ltae = _LTAE()
        self.temporal_aggregator = _Aggregator()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = [self.model.encoder(x[:, step]) for step in range(x.shape[1])]
        shallow = torch.stack([item[1] for item in encoded], dim=1).mean(dim=1)
        deep = torch.stack([item[2] for item in encoded], dim=1).mean(dim=1)
        decoded = self.model.decoder(x[:, 0], shallow, deep)
        return self.model.segmentation_head(decoded)

    @staticmethod
    def compute_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (logits - target).square().mean()


def test_t5_corruption_uses_c02_feature_positions_and_appends_invalidity() -> None:
    x = torch.ones(5, len(MULTI_FEATURES), 8, 8)
    output = apply_processed_temporal_reliability_corruption(
        x,
        fire_drop=True,
        block_fraction=0.25,
        key_digest="0" * 64,
        active_fire_missing_value=-3.0,
    )

    invalid = output[:, -1]
    active_value = MULTI_FEATURES.index(38)
    active_binary = MULTI_FEATURES.index(39)
    dynamic = MULTI_FEATURES.index(0)
    static = MULTI_FEATURES.index(12)
    mask = invalid[0].bool()

    assert output.shape == (5, 34, 8, 8)
    assert torch.equal(invalid[0], invalid[-1])
    assert int(mask.sum()) == 16
    assert torch.all(output[:, active_value] == -3.0)
    assert torch.all(output[:, active_binary] == 0.0)
    assert torch.all(output[:, dynamic, mask] == 0.0)
    assert torch.all(output[:, static] == 1.0)


def test_t5_sarp_is_exact_on_valid_input_for_any_tokens() -> None:
    base = _Base()
    reference = copy.deepcopy(base)
    model = TemporalSeverityAdaptiveReliabilityPrompting(base)
    with torch.no_grad():
        model.input_token.invalid_token.fill_(2.0)
        for token in model.prompt_tokens:
            token.fill_(3.0)
    features = torch.randn(2, 5, 33, 4, 4)
    valid = torch.cat((features, torch.zeros(2, 5, 1, 4, 4)), dim=2)

    torch.testing.assert_close(model(valid), reference(features))
    assert model.prompt_parameter_count == 8


@pytest.mark.parametrize(
    ("missing_rows", "input_active", "deep_active"),
    ((1, False, True), (2, True, False)),
)
def test_t5_severity_routes_gradient_to_one_prompt_family(
    missing_rows: int, input_active: bool, deep_active: bool
) -> None:
    model = TemporalSeverityAdaptiveReliabilityPrompting(_Base())
    features = torch.randn(2, 5, 33, 4, 4)
    invalid = torch.zeros(2, 5, 1, 4, 4)
    invalid[:, :, :, :missing_rows] = 1.0

    model(torch.cat((features, invalid), dim=2)).sum().backward()

    input_nonzero = bool(torch.count_nonzero(model.input_token.invalid_token.grad))
    deep_nonzero = any(
        bool(torch.count_nonzero(token.grad)) for token in model.prompt_tokens
    )
    assert input_nonzero is input_active
    assert deep_nonzero is deep_active


def _checkpoint_payload(variant: str) -> dict[str, object]:
    sarp = variant == "sarp"
    return {
        "schema_version": 1,
        "status": "pass",
        "candidate_id": "D13-SARP-T5" if sarp else "D13-STD-T5",
        "matched_pair": "D13-T5",
        "base_control": "B5",
        "experiment": "C02",
        "steps": 3_000,
        "seed": 0,
        "variant": variant,
        "processed_space_matched_corruption": True,
        "temporal_steps": 5,
        "feature_count": 33,
        "prompt_channels": [64, 64, 128, 256, 512] if sarp else None,
        "prompt_parameter_count": 1_088 if sarp else 0,
        "prompt_pooling": "adaptive-area-average" if sarp else None,
        "severity_threshold": 0.375 if sarp else None,
        "mild_prompt": "hierarchical-before-temporal-fusion" if sarp else None,
        "severe_prompt": "per-step-input" if sarp else None,
        "hyper_parameters": {"n_channels": 33},
        "state_dict": {"weight": torch.tensor(1.0)},
    }


def test_t5_checkpoint_contract_distinguishes_standard_and_sarp() -> None:
    assert validate_t5_reliability_checkpoint(_checkpoint_payload("standard")) == (
        "standard"
    )
    assert validate_t5_reliability_checkpoint(_checkpoint_payload("sarp")) == "sarp"

    invalid = _checkpoint_payload("sarp")
    invalid["severity_threshold"] = 0.4
    with pytest.raises(ValueError, match="T5 reliability checkpoint"):
        validate_t5_reliability_checkpoint(invalid)


def test_t5_runners_have_minimal_shell_contract(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    training = (
        root
        / "reproductions"
        / "wsts_fast_track"
        / "run_t5_reliability_on_nibi.sh"
    )
    heldout = (
        root
        / "reproductions"
        / "wsts_fast_track"
        / "run_t5_sarp_heldout_on_nibi.sh"
    )
    for runner in (training, heldout):
        syntax = subprocess.run(
            ["bash", "-n", str(runner)], text=True, capture_output=True, check=False
        )
        assert syntax.returncode == 0, syntax.stderr

    record = tmp_path / "b5.json"
    record.write_text("{}\n", encoding="utf-8")
    rejected = subprocess.run(
        [str(training), "unknown", str(record)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode == 2
    assert "variant must be standard or sarp" in rejected.stderr
