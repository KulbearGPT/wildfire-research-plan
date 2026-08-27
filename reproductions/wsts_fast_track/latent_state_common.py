"""Shared processed-space contract for fire belief-state prototypes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset

try:
    from torchvision.ops import sigmoid_focal_loss
except ModuleNotFoundError:  # The audit environment intentionally omits torchvision.
    def sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "none",
    ) -> torch.Tensor:
        """Compatibility implementation of torchvision's sigmoid focal loss."""

        probabilities = torch.sigmoid(inputs)
        cross_entropy = functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        probability_target = probabilities * targets + (1 - probabilities) * (1 - targets)
        losses = cross_entropy * ((1 - probability_target) ** gamma)
        if alpha >= 0:
            alpha_target = alpha * targets + (1 - alpha) * (1 - targets)
            losses = alpha_target * losses
        if reduction == "mean":
            return losses.mean()
        if reduction == "sum":
            return losses.sum()
        if reduction == "none":
            return losses
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

from .missingness import structured_block_mask


PROCESSED_FEATURES = 40
ACTIVE_HOUR = 38
ACTIVE_BINARY = 39
RELIABILITY = 40
STATE_CONTEXT = slice(0, 33)
PROCESSED_DYNAMIC_NON_FIRE = tuple(range(12)) + (15,) + tuple(range(33, 38))
SCENARIOS = ("M00", "M01", "M06", "M07")
TRAIN_SCENARIOS = ("M01", "M06", "M07")


def _require_processed_features(features: torch.Tensor, *, batched: bool) -> None:
    expected_dimensions = 5 if batched else 4
    if (
        not isinstance(features, torch.Tensor)
        or features.ndim != expected_dimensions
        or features.shape[-3] != PROCESSED_FEATURES
        or features.shape[-2] <= 0
        or features.shape[-1] <= 0
    ):
        prefix = "[B,T,40,H,W]" if batched else "[T,40,H,W]"
        raise ValueError(f"processed features must have shape {prefix}")


def pack_observations(
    features: torch.Tensor, reliability: torch.Tensor
) -> torch.Tensor:
    """Pack processed features and their observation-validity map."""

    _require_processed_features(features, batched=True)
    if (
        not isinstance(reliability, torch.Tensor)
        or reliability.ndim != 5
        or reliability.shape[2] != 1
        or reliability.shape[:2] != features.shape[:2]
        or reliability.shape[-2:] != features.shape[-2:]
    ):
        raise ValueError("reliability must have shape [B,T,1,H,W] matching features")
    if reliability.device != features.device:
        raise ValueError("features and reliability must be on the same device")
    return torch.cat((features, reliability), dim=2)


def split_observations(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a `[B,T,41,H,W]` observation tensor into features and validity."""

    if (
        not isinstance(packed, torch.Tensor)
        or packed.ndim != 5
        or packed.shape[2] != RELIABILITY + 1
        or packed.shape[-2] <= 0
        or packed.shape[-1] <= 0
    ):
        raise ValueError("packed observations must have shape [B,T,41,H,W]")
    return packed[:, :, :PROCESSED_FEATURES], packed[:, :, RELIABILITY:]


def observation_consistent_state(
    observed: torch.Tensor,
    proposal_logits: torch.Tensor,
    reliability: torch.Tensor,
) -> torch.Tensor:
    """Preserve valid observations and infer only unavailable active-fire pixels."""

    if observed.shape != proposal_logits.shape or observed.shape != reliability.shape:
        raise ValueError("observed state, proposals, and reliability must have equal shapes")
    return torch.where(reliability.bool(), observed, torch.sigmoid(proposal_logits))


def masked_unweighted_focal(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return alpha-disabled focal loss over all pixels or one nonempty mask."""

    if logits.shape != target.shape:
        raise ValueError("logits and target must have equal shapes")
    pixel_losses = sigmoid_focal_loss(
        logits,
        target.to(dtype=logits.dtype),
        alpha=-1.0,
        gamma=2.0,
        reduction="none",
    )
    if mask is None:
        return pixel_losses.mean()
    if mask.shape != logits.shape:
        raise ValueError("mask must have the same shape as logits")
    selected = pixel_losses[mask.bool()]
    if selected.numel() == 0:
        raise ValueError("masked focal loss requires at least one selected pixel")
    return selected.mean()


def apply_processed_corruption(
    clean: torch.Tensor,
    scenario_id: str,
    *,
    normalized_active_fire_zero: float,
    key_digest: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Corrupt one already aligned processed crop and emit its validity map."""

    _require_processed_features(clean, batched=False)
    if scenario_id not in SCENARIOS:
        raise ValueError(f"unknown belief-state scenario: {scenario_id}")

    corrupted = clean.clone()
    reliability = clean.new_ones((clean.shape[0], 1, clean.shape[-2], clean.shape[-1]))
    if scenario_id == "M00":
        return corrupted, reliability

    if scenario_id == "M01":
        corrupted[:, ACTIVE_HOUR] = normalized_active_fire_zero
        corrupted[:, ACTIVE_BINARY] = 0.0
        reliability.zero_()
        return corrupted, reliability

    fraction = 0.25 if scenario_id == "M06" else 0.50
    mask_array = structured_block_mask(
        clean.shape[-2], clean.shape[-1], fraction, key_digest=key_digest
    )
    mask = torch.as_tensor(mask_array, dtype=torch.bool, device=clean.device)
    for channel in (*PROCESSED_DYNAMIC_NON_FIRE, ACTIVE_BINARY):
        corrupted[:, channel].masked_fill_(mask, 0.0)
    corrupted[:, ACTIVE_HOUR].masked_fill_(mask, normalized_active_fire_zero)
    reliability[:, 0].masked_fill_(mask, 0.0)
    return corrupted, reliability


def normalized_active_fire_zero(base: Any) -> float:
    """Derive processed raw-zero from the installed active-fire statistics."""

    return float(-base.means[0, 22, 0, 0] / base.stds[0, 22, 0, 0])


class BeliefStateTrainingDataset(Dataset[Any]):
    """Apply one uniformly sampled processed-space corruption per training item."""

    def __init__(self, base: Any, normalized_active_fire_zero: float) -> None:
        self.base = base
        self.normalized_active_fire_zero = float(normalized_active_fire_zero)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, Any, int]:
        clean, target = self.base[index]
        _require_processed_features(clean, batched=False)
        scenario_index = int(np.random.randint(len(TRAIN_SCENARIOS)))
        scenario_id = TRAIN_SCENARIOS[scenario_index]
        corrupted, reliability = apply_processed_corruption(
            clean,
            scenario_id,
            normalized_active_fire_zero=self.normalized_active_fire_zero,
            key_digest=np.random.bytes(32).hex(),
        )
        return (
            pack_observations(corrupted[None], reliability[None])[0],
            clean[-1, ACTIVE_BINARY : ACTIVE_BINARY + 1],
            target,
            scenario_index,
        )


class BeliefStateEvaluationDataset(Dataset[Any]):
    """Translate controlled raw-space corruption masks to reliability `R`."""

    def __init__(self, controlled: Any) -> None:
        if getattr(controlled, "routing_mask_channel", False) is not True:
            raise ValueError("belief-state evaluation requires routing_mask_channel=True")
        if getattr(controlled, "scenario_id", None) not in SCENARIOS:
            raise ValueError("belief-state evaluation requires an M00/M01/M06/M07 dataset")
        self.controlled = controlled

    def __len__(self) -> int:
        return len(self.controlled)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Any]:
        routed, target = self.controlled[index]
        if (
            not isinstance(routed, torch.Tensor)
            or routed.ndim != 4
            or routed.shape[1] != RELIABILITY + 1
        ):
            raise ValueError("controlled sample must have shape [T,41,H,W]")
        features = routed[:, :PROCESSED_FEATURES]
        missing = routed[:, RELIABILITY : RELIABILITY + 1]
        if self.controlled.scenario_id == "M01":
            missing = torch.ones_like(missing)
        reliability = 1.0 - missing
        return pack_observations(features[None], reliability[None])[0], target


class FrozenP00BeliefModel(torch.nn.Module):
    """Base class that hard-corrects fire state before frozen P00 forecasting."""

    def __init__(self, default_model: torch.nn.Module, history: int) -> None:
        super().__init__()
        if history not in {1, 5}:
            raise ValueError("history must be 1 or 5")
        self.default_model = default_model.requires_grad_(False)
        self.default_model.eval()
        self.history = history

    def train(self, mode: bool = True):
        super().train(mode)
        self.default_model.eval()
        return self

    def infer_state_logits(
        self, features: torch.Tensor, reliability: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward_state(
        self, packed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, reliability = split_observations(packed)
        if features.shape[1] != self.history:
            raise ValueError("packed history differs from model history")
        state_logits = self.infer_state_logits(features, reliability)
        state = observation_consistent_state(
            features[:, -1, ACTIVE_BINARY : ACTIVE_BINARY + 1],
            state_logits,
            reliability[:, -1],
        )
        latest = features[:, -1].clone()
        latest[:, ACTIVE_BINARY : ACTIVE_BINARY + 1] = state
        forecast = self.default_model(latest[:, None])
        return forecast, state_logits, state

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        return self.forward_state(packed)[0]

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return masked_unweighted_focal(logits.squeeze(1), target)


def trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy checkpoint values for the belief-state module, excluding frozen P00."""

    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("default_model.")
    }


def load_trainable_state_dict(
    model: torch.nn.Module, state: Mapping[str, torch.Tensor]
) -> None:
    """Strictly restore only the non-P00 state saved by `trainable_state_dict`."""

    expected = trainable_state_dict(model)
    if set(state) != set(expected):
        raise ValueError("trainable checkpoint keys do not exactly match the model")
    complete = model.state_dict()
    complete.update(state)
    model.load_state_dict(complete, strict=True)


def screen_2021(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the frozen 2021 AP screen to M01, M06, and M07 only."""

    if set(results) != set(TRAIN_SCENARIOS):
        raise ValueError("screen_2021 requires exactly M01, M06, and M07 results")
    deltas: dict[str, float] = {}
    for scenario_id in TRAIN_SCENARIOS:
        try:
            model_ap = float(results[scenario_id]["model"]["metrics"]["avg_precision"])
            p00_ap = float(results[scenario_id]["p00"]["metrics"]["avg_precision"])
        except (KeyError, TypeError) as error:
            raise ValueError("screen results must contain nested model/p00 AP metrics") from error
        deltas[scenario_id] = model_ap - p00_ap
    mean_delta = sum(deltas.values()) / len(deltas)
    improved_scenarios = sum(delta > 0.0 for delta in deltas.values())
    worst_delta = min(deltas.values())
    return {
        "ap_deltas": deltas,
        "mean_delta": mean_delta,
        "improved_scenarios": improved_scenarios,
        "worst_delta": worst_delta,
        "pass": (
            mean_delta > 0.0
            and improved_scenarios >= 2
            and worst_delta >= -0.01
        ),
    }
