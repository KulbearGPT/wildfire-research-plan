import torch


class _TinyDefault(torch.nn.Module):
    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        return packed[:, 0, 39:40] * 2.0

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (logits - target).square().mean()


def test_common_contract_preserves_observations_and_masks_state_loss() -> None:
    from reproductions.wsts_fast_track.latent_state_common import (
        apply_processed_corruption,
        masked_unweighted_focal,
        observation_consistent_state,
        pack_observations,
        split_observations,
    )

    for history in (1, 5):
        clean = torch.ones((history, 40, 8, 8))
        m01, r01 = apply_processed_corruption(
            clean, "M01", normalized_active_fire_zero=-2.0,
            key_digest="00" * 32,
        )
        assert torch.count_nonzero(r01) == 0
        assert torch.count_nonzero(m01[:, 39]) == 0
        m06, r06 = apply_processed_corruption(
            clean, "M06", normalized_active_fire_zero=-2.0,
            key_digest="00" * 32,
        )
        assert int((r06 == 0).sum()) == history * 16
        packed = pack_observations(m06[None], r06[None])
        x, reliability = split_observations(packed)
        proposal = torch.zeros((1, 1, 8, 8))
        corrected = observation_consistent_state(
            x[:, -1, 39:40], proposal, reliability[:, -1]
        )
        observed = reliability[:, -1].bool()
        assert torch.equal(corrected[observed], x[:, -1, 39:40][observed])
        loss = masked_unweighted_focal(
            proposal, clean[None, -1, 39:40], ~observed
        )
        assert torch.isfinite(loss)


# Recurrent filter
def test_recurrent_filter_supports_both_histories_and_keeps_m00_exact() -> None:
    from reproductions.wsts_fast_track.latent_state_common import pack_observations
    from reproductions.wsts_fast_track.state_filter import RecurrentBeliefFilter

    for history in (1, 5):
        features = torch.randn((2, history, 40, 16, 16))
        reliability = torch.ones((2, history, 1, 16, 16))
        model = RecurrentBeliefFilter(_TinyDefault(), history)
        forecast, state_logits, state = model.forward_state(
            pack_observations(features, reliability)
        )
        assert state_logits.shape == state.shape == (2, 1, 16, 16)
        assert torch.equal(forecast, model.default_model(features[:, -1:]))
        reliability[:, -1, :, :8] = 0
        model(pack_observations(features, reliability)).mean().backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.trainable_parameters()
        )


# Temporal attention
def test_attention_state_is_history_matched_and_missing_fire_invariant() -> None:
    from reproductions.wsts_fast_track.latent_state_common import pack_observations
    from reproductions.wsts_fast_track.state_attention import TemporalAttentionBeliefState

    one = TemporalAttentionBeliefState(_TinyDefault(), 1)
    five = TemporalAttentionBeliefState(_TinyDefault(), 5)
    assert sum(p.numel() for p in one.trainable_parameters()) == sum(
        p.numel() for p in five.trainable_parameters()
    )
    features = torch.randn((2, 5, 40, 12, 12))
    reliability = torch.zeros((2, 5, 1, 12, 12))
    changed = features.clone()
    changed[:, :, 38:40] = 1000.0
    torch.testing.assert_close(
        five.infer_state_logits(features, reliability),
        five.infer_state_logits(changed, reliability),
    )
    weights = five.attention_weights(features, reliability)
    assert weights.shape == (2, 6, 12, 12)
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights.sum(1), torch.ones_like(weights[:, 0]))
    reliability[:, :-1] = 1
    forecast = five(pack_observations(features, reliability))
    forecast.mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in five.trainable_parameters()
    )

    reliability.fill_(1)
    assert torch.equal(
        five(pack_observations(features, reliability)),
        five.default_model(features[:, -1:]),
    )


# Reconstruction baseline
def test_reconstruction_baseline_uses_state_loss_and_preserves_m00() -> None:
    from reproductions.wsts_fast_track.latent_state_common import (
        masked_unweighted_focal,
        pack_observations,
    )
    from reproductions.wsts_fast_track.reconstruction_baseline import (
        ReconstructionFirstBeliefState,
    )

    for history in (1, 5):
        features = torch.randn((2, history, 40, 16, 16))
        reliability = torch.ones((2, history, 1, 16, 16))
        model = ReconstructionFirstBeliefState(_TinyDefault(), history)
        assert torch.equal(
            model(pack_observations(features, reliability)),
            model.default_model(features[:, -1:]),
        )
        reliability[:, -1, :, :8] = 0
        _, state_logits, _ = model.forward_state(pack_observations(features, reliability))
        loss = masked_unweighted_focal(
            state_logits, features[:, -1, 39:40], ~reliability[:, -1].bool()
        )
        loss.backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.trainable_parameters()
        )


# Unified integration
def test_objective_and_screen_are_frozen() -> None:
    from reproductions.wsts_fast_track.evaluate_belief_state import screen_payload
    from reproductions.wsts_fast_track.train_belief_state import combine_losses

    forecast, state = torch.tensor(2.0), torch.tensor(3.0)
    torch.testing.assert_close(combine_losses("filter", forecast, state), torch.tensor(2.3))
    torch.testing.assert_close(combine_losses("attention", forecast, state), torch.tensor(2.3))
    torch.testing.assert_close(combine_losses("reconstruction", forecast, state), state)
    results = {
        scenario: {
            "model": {"metrics": {"avg_precision": a}},
            "p00": {"metrics": {"avg_precision": b}},
        }
        for scenario, a, b in (
            ("M01", 0.11, 0.10),
            ("M06", 0.31, 0.30),
            ("M07", 0.19, 0.20),
        )
    }
    screen = screen_payload(results)
    assert screen["pass"] is True
    assert screen["improved_scenarios"] == 2
