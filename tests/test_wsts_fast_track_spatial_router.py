from __future__ import annotations

import torch
import pytest


class _ConstantExpert(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (features.shape[0], 1, features.shape[-2], features.shape[-1]),
            self.value,
            dtype=features.dtype,
            device=features.device,
        )


class _FortyChannelExpert(torch.nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        assert tuple(features.shape[1:3]) == (1, 40)
        return features[:, -1, :1]

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (logits - target).square().mean()


def test_spatial_router_uses_block_expert_only_inside_missing_area() -> None:
    from reproductions.wsts_fast_track import spatial_router

    default_logits = torch.zeros((1, 1, 2, 3))
    block_logits = torch.ones((1, 1, 2, 3))
    missing = torch.tensor(
        [[[[False, True, True], [False, False, True]]]], dtype=torch.bool
    )

    routed = spatial_router.route_spatial_logits(
        default_logits, block_logits, missing
    )

    torch.testing.assert_close(routed, missing.float())


def test_spatial_router_consumes_mask_channel_without_passing_it_to_experts() -> None:
    from reproductions.wsts_fast_track.spatial_router import SpatialExpertRouter

    routed_input = torch.zeros((2, 1, 41, 2, 3))
    routed_input[:, :, 40, 0, 1:] = 1.0
    router = SpatialExpertRouter(_ConstantExpert(2.0), _ConstantExpert(5.0))

    routed = router(routed_input)

    expected = torch.tensor(
        [[[[2.0, 5.0, 5.0], [2.0, 2.0, 2.0]]]]
    ).expand(2, -1, -1, -1)
    torch.testing.assert_close(routed, expected)


def test_spatial_router_heldout_requires_explicit_authorization() -> None:
    from reproductions.wsts_fast_track.evaluate_spatial_router import (
        router_evaluation_boundary,
    )

    assert router_evaluation_boundary(2021, heldout_authorized=False) == (
        "prototype-validation",
        False,
    )
    with pytest.raises(ValueError, match="authorization"):
        router_evaluation_boundary(2022, heldout_authorized=False)
    assert router_evaluation_boundary(2023, heldout_authorized=True) == (
        "prototype-formal",
        True,
    )


def test_default_baseline_strips_routing_mask_channel() -> None:
    from reproductions.wsts_fast_track.spatial_router import RoutingInputModel

    routed_input = torch.zeros((2, 1, 41, 3, 3))
    routed_input[:, :, 40] = 1.0

    logits = RoutingInputModel(_FortyChannelExpert())(routed_input)

    assert tuple(logits.shape) == (2, 1, 3, 3)
    assert torch.count_nonzero(logits) == 0
