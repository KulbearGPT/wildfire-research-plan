import pytest
import torch
from torch import nn

from reproductions.cross_history.architectures import (
    DirectForecaster,
    architecture_config,
    canonical_architecture,
    checkpoint_architecture,
    resolve_architecture,
    swin_pretrained_asset,
    swin_runtime_config,
    validate_run_source,
    verify_asset,
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


def test_checkpoint_architecture_is_backward_compatible_but_rejects_mismatch():
    assert checkpoint_architecture({"history": 1}) == "res18_unet"
    assert checkpoint_architecture({"history": 5}) == "res18_utae"
    assert checkpoint_architecture({"history": 1, "architecture": "swin_unet"}) == "swin_unet"
    assert resolve_architecture(5, None) == "res18_utae"
    with pytest.raises(ValueError, match="checkpoint architecture"):
        checkpoint_architecture(
            {"history": 1, "architecture": "convlstm"}, expected="swin_unet")


def test_noncanonical_training_requires_one_explicit_initialization_source():
    validate_run_source("res18_unet", 1, "control",
                        bootstrap=False, initial_checkpoint=None,
                        evaluate_only=None)
    validate_run_source("swin_unet", 1, "control",
                        bootstrap=True, initial_checkpoint=None,
                        evaluate_only=None)
    validate_run_source("swin_unet", 5, "cosine_erm",
                        bootstrap=False, initial_checkpoint="base.pt",
                        evaluate_only=None)
    with pytest.raises(ValueError, match="exactly one"):
        validate_run_source("swin_unet", 1, "control",
                            bootstrap=False, initial_checkpoint=None,
                            evaluate_only=None)
    with pytest.raises(ValueError, match="exactly one"):
        validate_run_source("swin_unet", 1, "control",
                            bootstrap=True, initial_checkpoint="base.pt",
                            evaluate_only=None)
    with pytest.raises(ValueError, match="method"):
        validate_run_source("swin_unet", 1, "context",
                            bootstrap=True, initial_checkpoint=None,
                            evaluate_only=None)


def test_canonical_backbones_cannot_enter_hparam_free_bootstrap_path():
    with pytest.raises(ValueError, match="canonical.*bootstrap"):
        validate_run_source("res18_unet", 1, "control",
                            bootstrap=True, initial_checkpoint=None,
                            evaluate_only=None)


def test_swin_pretraining_is_project_rooted_and_checksum_guarded(tmp_path):
    asset = swin_pretrained_asset()
    assert str(asset.path) == (
        "/project/6085198/kulbear/wildfire/cache/swin/"
        "swin_tiny_patch4_window7_224.pth")
    assert asset.sha256 == "9f71c168d837d1b99dd1dc29e14990a7a9e8bdc5f673d46b04fe36fe15590ad3"
    assert "/home/" not in str(asset.path)

    fixture = tmp_path / "asset.bin"
    fixture.write_bytes(b"abc")
    verify_asset(fixture, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    with pytest.raises(ValueError, match="checksum"):
        verify_asset(fixture, "0" * 64)


def test_swin_runtime_config_matches_wsts_crop_and_selected_channels():
    config = swin_runtime_config(165, "/project/cache/swin.pth")

    assert config.DATA.IMG_SIZE == 128
    assert config.MODEL.SWIN.IN_CHANS == 165
    assert config.MODEL.PRETRAIN_CKPT == "/project/cache/swin.pth"
    assert config.MODEL.SWIN.DEPTHS == [2, 2, 2, 2]
    assert config.MODEL.SWIN.NUM_HEADS == [3, 6, 12, 24]
