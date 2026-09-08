import pytest
import torch
from torch import nn

from reproductions.cross_history.architectures import (
    DirectForecaster,
    architecture_config,
    canonical_architecture,
    validate_architecture,
)


class _ShapeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_shape = None

    def forward(self, inputs):
        self.seen_shape = tuple(inputs.shape)
        return inputs[:, -1, :1]

    @staticmethod
    def compute_loss(logits, target):
        return (logits - target).abs().mean()


def test_architecture_contract_preserves_canonical_models_and_history_limits():
    assert canonical_architecture(1) == "res18_unet"
    assert canonical_architecture(5) == "res18_utae"
    validate_architecture("swin_unet", 1)
    validate_architecture("swin_unet", 5)
    with pytest.raises(ValueError, match="convlstm.*T=1"):
        validate_architecture("convlstm", 1)


def test_direct_forecaster_removes_routing_masks_before_backbone_forward():
    base = _ShapeBase()
    model = DirectForecaster(base, history=5, channels=33)
    packed = torch.randn(2, 5, 35, 8, 8)

    logits = model(packed)

    assert base.seen_shape == (2, 5, 33, 8, 8)
    assert logits.shape == (2, 1, 8, 8)
    assert model.compute_loss(logits.squeeze(1), torch.zeros(2, 8, 8)) >= 0


def test_published_backbone_configs_preserve_temporal_representation():
    swin_t1 = architecture_config("swin_unet", 1)
    swin_t5 = architecture_config("swin_unet", 5)
    convlstm = architecture_config("convlstm", 5)

    assert (swin_t1.module, swin_t1.class_name) == (
        "models.SwinUnetLightning", "SwinUnetLightning")
    assert swin_t1.kwargs["n_channels"] == 40
    assert swin_t5.kwargs["n_channels"] == 165
    assert swin_t5.kwargs["flatten_temporal_dimension"] is True
    assert convlstm.kwargs["n_channels"] == 33
    assert convlstm.kwargs["flatten_temporal_dimension"] is False
    assert convlstm.learning_rate == 0.01
