import torch

from reproductions.cross_history.models import DeepForecastSupervision


def test_deep_supervision_starts_with_zero_full_resolution_logits() -> None:
    heads = DeepForecastSupervision((8, 16, 32))
    features = (
        torch.randn(2, 8, 16, 16),
        torch.randn(2, 16, 8, 8),
        torch.randn(2, 32, 4, 4),
    )

    logits = heads(features, (32, 32))

    assert len(logits) == 3
    for prediction in logits:
        assert prediction.shape == (2, 1, 32, 32)
        torch.testing.assert_close(prediction, torch.zeros_like(prediction))
