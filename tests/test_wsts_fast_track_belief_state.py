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
# Unified integration
