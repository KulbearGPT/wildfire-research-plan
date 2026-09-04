"""Reliability-conditioned counterfactual forecast-logit correction."""

from __future__ import annotations

from typing import Any

import torch


ADAPTER_PARAMETER_COUNT = 2_769
RELIABILITY_MAPS = ("fire-drop", "block-drop")


def pack_reliability_input(
    features: torch.Tensor,
    fire_missing: torch.Tensor,
    block_missing: torch.Tensor,
) -> torch.Tensor:
    """Append aligned FireDrop and BlockDrop reliability maps."""

    if (
        features.ndim != 5
        or features.shape[2] != 40
        or fire_missing.shape != (
            features.shape[0],
            features.shape[1],
            1,
            features.shape[3],
            features.shape[4],
        )
        or block_missing.shape != fire_missing.shape
    ):
        raise ValueError(
            "features and reliability must have shapes [B,T,40,H,W] and [B,T,1,H,W]"
        )
    return torch.cat((features, fire_missing, block_missing), dim=2)


class CounterfactualReliabilityAdapter(torch.nn.Module):
    """Joint base forecast plus a small reliability-conditioned logit adapter."""

    def __init__(self, base_model: torch.nn.Module) -> None:
        super().__init__()
        self.base_model = base_model
        self.adapter = torch.nn.Sequential(
            torch.nn.Conv2d(19, 16, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(16, 1, kernel_size=1),
        )
        torch.nn.init.zeros_(self.adapter[-1].weight)
        torch.nn.init.zeros_(self.adapter[-1].bias)
        self.adapter_parameter_count = sum(
            parameter.numel() for parameter in self.adapter.parameters()
        )
        if self.adapter_parameter_count != ADAPTER_PARAMETER_COUNT:
            raise RuntimeError("unexpected CRA parameter count")

    def _base_forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.base_model.model.encoder(features)
        decoder_features = self.base_model.model.decoder(*encoded)
        logits = self.base_model.model.segmentation_head(decoder_features)
        return logits, decoder_features

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        if packed.ndim != 5 or packed.shape[1] != 1 or packed.shape[2] != 42:
            raise ValueError("CRA input must have shape [B,1,42,H,W]")
        features = packed[:, 0, :40]
        fire_missing = packed[:, 0, 40:41].clamp(0.0, 1.0)
        block_missing = packed[:, 0, 41:42].clamp(0.0, 1.0)
        logits, decoder_features = self._base_forward(features)
        block_fraction = block_missing.mean(dim=(2, 3), keepdim=True).expand_as(
            block_missing
        )
        residual = self.adapter(
            torch.cat(
                (decoder_features, fire_missing, block_missing, block_fraction),
                dim=1,
            )
        )
        any_missing = (
            (fire_missing.amax(dim=(1, 2, 3), keepdim=True) > 0)
            | (block_missing.amax(dim=(1, 2, 3), keepdim=True) > 0)
        ).to(logits.dtype)
        return logits + any_missing * residual

    def clean_forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run the exact base path without constructing or applying an adapter."""

        if features.ndim != 5 or features.shape[1] != 1 or features.shape[2] != 40:
            raise ValueError("clean CRA input must have shape [B,1,40,H,W]")
        logits, _ = self._base_forward(features[:, 0])
        return logits

    def compute_loss(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return self.base_model.compute_loss(logits, target)


class TwoRegimeReliabilityDataset:
    """Translate a routed controlled sample into FireDrop/BlockDrop maps."""

    def __init__(self, controlled_dataset: Any, scenario_id: str) -> None:
        if scenario_id not in {"M00", "M01", "M06", "M07"}:
            raise ValueError("CRA supports only M00/M01/M06/M07")
        self.base = controlled_dataset
        self.scenario_id = scenario_id

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        routed, target = self.base[index]
        if routed.ndim != 4 or routed.shape[1] != 41:
            raise ValueError("controlled CRA input must have shape [T,41,H,W]")
        features = routed[:, :40]
        block_missing = routed[:, 40:41]
        fire_missing = torch.zeros_like(block_missing)
        if self.scenario_id == "M01":
            fire_missing.fill_(1.0)
            block_missing = torch.zeros_like(block_missing)
        elif self.scenario_id == "M00":
            block_missing = torch.zeros_like(block_missing)
        return torch.cat((features, fire_missing, block_missing), dim=1), target
