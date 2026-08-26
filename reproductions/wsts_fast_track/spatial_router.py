"""P03 spatial reliability routing between frozen P00 and P02 experts."""

from __future__ import annotations

import torch


ROUTER_ID = "P03-SpatialExpertRouter-P00-P02"
RELIABILITY_ROUTER_ID = "P11-MissingnessRouter-P00-P10"


class RoutingInputModel(torch.nn.Module):
    """Expose a 40-channel model through the routed 41-channel input contract."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, routed_input: torch.Tensor) -> torch.Tensor:
        if routed_input.ndim != 5 or routed_input.shape[2] != 41:
            raise ValueError("routed input must have shape (B, T, 41, H, W)")
        return self.model(routed_input[:, :, :40])

    def compute_loss(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return self.model.compute_loss(logits, target)


def route_spatial_logits(
    default_logits: torch.Tensor,
    block_logits: torch.Tensor,
    missing_mask: torch.Tensor,
) -> torch.Tensor:
    """Use the block expert only at pixels marked spatially unavailable."""

    if (
        default_logits.shape != block_logits.shape
        or missing_mask.shape != default_logits.shape
    ):
        raise ValueError("expert logits and spatial mask must have identical shapes")
    return torch.where(missing_mask.bool(), block_logits, default_logits)


class SpatialExpertRouter(torch.nn.Module):
    """Route each pixel between frozen P00 and P02 logits using known validity."""

    def __init__(self, default_model: torch.nn.Module, block_model: torch.nn.Module):
        super().__init__()
        self.default_model = default_model
        self.block_model = block_model

    def forward(self, routed_input: torch.Tensor) -> torch.Tensor:
        if routed_input.ndim != 5 or routed_input.shape[2] != 41:
            raise ValueError("router input must have shape (B, T, 41, H, W)")
        features = routed_input[:, :, :40]
        missing_mask = routed_input[:, -1, 40:41].bool()
        default_logits = self.default_model(features)
        block_logits = self.block_model(features)
        return route_spatial_logits(default_logits, block_logits, missing_mask)

    def compute_loss(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return self.default_model.compute_loss(logits, target)


class ReliabilityExpertRouter(SpatialExpertRouter):
    """Route global fire loss or spatially missing pixels to one expert."""

    def __init__(
        self,
        default_model: torch.nn.Module,
        expert_model: torch.nn.Module,
        *,
        route_all: bool,
    ) -> None:
        super().__init__(default_model, expert_model)
        self.route_all = route_all

    def forward(self, routed_input: torch.Tensor) -> torch.Tensor:
        if routed_input.ndim != 5 or routed_input.shape[2] != 41:
            raise ValueError("router input must have shape (B, T, 41, H, W)")
        if self.route_all:
            return self.block_model(routed_input[:, :, :40])
        return super().forward(routed_input)
