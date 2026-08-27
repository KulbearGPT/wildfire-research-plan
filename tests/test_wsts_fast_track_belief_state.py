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
# Reconstruction baseline
# Unified integration
