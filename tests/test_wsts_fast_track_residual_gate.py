from __future__ import annotations

import torch


class _Encoder(torch.nn.Module):
    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor]:
        return (features[:, :16],)


class _TwoFeatureEncoder(torch.nn.Module):
    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        selected = features[:, :16]
        return selected, selected


class _Decoder(torch.nn.Module):
    def forward(
        self, feature: torch.Tensor, _skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        return feature


class _Head(torch.nn.Module):
    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return feature[:, :1] * 2.0


class _TinyUnet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()
        self.decoder = _Decoder()
        self.segmentation_head = _Head()


class _TinyDefault(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _TinyUnet()

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (logits - target).square().mean()


def test_masked_residual_is_zero_outside_missing_block() -> None:
    from reproductions.wsts_fast_track.residual_gate import apply_masked_residual

    default_logits = torch.full((1, 1, 2, 3), 2.0)
    residual_logits = torch.full((1, 1, 2, 3), 3.0)
    missing = torch.tensor(
        [[[[False, True, True], [False, False, True]]]], dtype=torch.bool
    )

    routed = apply_masked_residual(default_logits, residual_logits, missing)

    expected = torch.tensor([[[[2.0, 5.0, 5.0], [2.0, 2.0, 5.0]]]])
    torch.testing.assert_close(routed, expected)


def test_frozen_gate_reuses_decoder_features_and_preserves_unmasked_logits() -> None:
    from reproductions.wsts_fast_track.residual_gate import FrozenSpatialResidualGate

    routed_input = torch.zeros((1, 1, 41, 2, 2))
    routed_input[:, :, 0] = 2.0
    routed_input[:, :, 40, 0, 1] = 1.0
    gate = FrozenSpatialResidualGate(_TinyDefault())
    with torch.no_grad():
        gate.residual_head.weight.fill_(0.0)
        gate.residual_head.weight[:, 0].fill_(1.0)
        gate.residual_head.bias.fill_(1.0)

    routed = gate(routed_input)

    expected = torch.tensor([[[[4.0, 7.0], [4.0, 4.0]]]])
    torch.testing.assert_close(routed, expected)
    assert sum(parameter.numel() for parameter in gate.trainable_parameters()) == 17


def test_processed_block_dropout_appends_mask_and_hides_dynamic_channels() -> None:
    from reproductions.wsts_fast_track.residual_gate import (
        apply_processed_block_dropout,
    )

    processed = torch.ones((1, 40, 4, 4))

    routed, mask = apply_processed_block_dropout(
        processed,
        fraction=0.5,
        key_digest="00" * 32,
        normalized_active_fire_zero=-2.0,
    )

    assert tuple(routed.shape) == (1, 41, 4, 4)
    assert int(mask.sum()) == 8
    dynamic = (*range(12), 15, *range(33, 38))
    assert torch.count_nonzero(routed[:, dynamic][:, :, mask]) == 0
    assert torch.all(routed[:, 38][:, mask] == -2.0)
    assert torch.count_nonzero(routed[:, 39][:, mask]) == 0
    assert torch.all(routed[:, 40][:, mask] == 1.0)
    assert torch.all(routed[:, 12:15] == 1.0)
    assert torch.all(routed[:, 16:33] == 1.0)


def test_spatial_residual_gate_supports_minimal_three_by_three_head() -> None:
    from reproductions.wsts_fast_track.residual_gate import FrozenSpatialResidualGate

    gate = FrozenSpatialResidualGate(_TinyDefault(), residual_kernel_size=3)

    assert tuple(gate.residual_head.weight.shape) == (1, 16, 3, 3)
    assert sum(parameter.numel() for parameter in gate.trainable_parameters()) == 145


def test_last_block_router_preserves_default_outside_mask() -> None:
    from reproductions.wsts_fast_track.residual_gate import FrozenLastBlockRouter

    model = _TinyDefault()
    model.model.encoder = _TwoFeatureEncoder()
    model.model.decoder.center = torch.nn.Identity()
    model.model.decoder.blocks = torch.nn.ModuleList([_Decoder()])
    model.model.segmentation_head = torch.nn.Conv2d(16, 1, 1)
    with torch.no_grad():
        model.model.segmentation_head.weight.zero_()
        model.model.segmentation_head.bias.fill_(2.0)
    routed_input = torch.zeros((1, 1, 41, 2, 2))
    routed_input[:, :, 40, 0, 1] = 1.0
    router = FrozenLastBlockRouter(model)
    assert all(parameter.requires_grad for parameter in router.trainable_parameters())
    with torch.no_grad():
        router.adapted_head.bias.fill_(5.0)

    routed = router(routed_input)

    expected = torch.tensor([[[[2.0, 5.0], [2.0, 2.0]]]])
    torch.testing.assert_close(routed, expected)


def test_stochastic_belief_preserves_unmasked_logits_and_round_trips() -> None:
    from reproductions.wsts_fast_track.residual_gate import (
        FrozenStochasticBeliefResidual,
    )

    routed_input = torch.zeros((1, 1, 41, 2, 2))
    routed_input[:, :, 0] = 2.0
    routed_input[:, :, 40, 0, 1] = 1.0
    gate = FrozenStochasticBeliefResidual(_TinyDefault(), sample_count=4)
    with torch.no_grad():
        gate.belief_head.weight.zero_()
        gate.belief_head.bias.zero_()
        gate.output_head.weight.fill_(1.0 / 16.0)
        gate.output_head.bias.zero_()

    torch.manual_seed(7)
    logits, variance = gate.forward_with_uncertainty(routed_input)
    missing = routed_input[:, 0, 40:41].bool()

    torch.testing.assert_close(logits[~missing], torch.full((3,), 4.0))
    assert torch.count_nonzero(variance[~missing]) == 0
    assert torch.isfinite(variance).all()
    assert torch.all(variance >= 0)
    assert float(variance[missing].item()) > 0.0
    assert all(
        parameter.requires_grad for parameter in gate.trainable_parameters()
    )

    clone = FrozenStochasticBeliefResidual(_TinyDefault(), sample_count=4)
    clone.belief_head.load_state_dict(gate.belief_head.state_dict())
    clone.output_head.load_state_dict(gate.output_head.state_dict())
    torch.manual_seed(7)
    cloned_logits, cloned_variance = clone.forward_with_uncertainty(routed_input)
    torch.testing.assert_close(cloned_logits, logits)
    torch.testing.assert_close(cloned_variance, variance)

    loss = gate.compute_loss(logits.squeeze(1), torch.zeros((1, 2, 2)))
    loss.backward()
    assert any(
        parameter.grad is not None for parameter in gate.trainable_parameters()
    )


def test_teacher_posterior_trains_all_heads_and_prior_preserves_routing() -> None:
    from reproductions.wsts_fast_track.residual_gate import (
        FrozenTeacherPosteriorBelief,
    )

    training_input = torch.zeros((1, 1, 81, 2, 2))
    training_input[:, :, 0] = 2.0
    training_input[:, :, 40, 0, 1] = 1.0
    training_input[:, :, 41] = 3.0
    target = torch.zeros((1, 2, 2))
    model = FrozenTeacherPosteriorBelief(_TinyDefault(), sample_count=4)

    torch.manual_seed(11)
    loss, components, training_logits = model.training_objective(
        training_input, target
    )

    assert set(components) == {"forecast", "kl", "reconstruction", "total"}
    assert all(torch.isfinite(value) for value in components.values())
    assert torch.isfinite(loss)
    missing = training_input[:, 0, 40:41].bool()
    torch.testing.assert_close(training_logits[~missing], torch.full((3,), 4.0))
    loss.backward()
    for head in (
        model.posterior_head,
        model.prior_head,
        model.reconstruction_head,
        model.output_head,
    ):
        assert any(parameter.grad is not None for parameter in head.parameters())

    torch.manual_seed(11)
    inference_logits, variance = model.forward_with_uncertainty(
        training_input[:, :, :41]
    )
    torch.testing.assert_close(inference_logits[~missing], torch.full((3,), 4.0))
    assert torch.count_nonzero(variance[~missing]) == 0
    assert torch.isfinite(variance).all()
    assert torch.all(variance >= 0)
