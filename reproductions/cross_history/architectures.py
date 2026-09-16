"""Minimal backbone boundary for matched cross-history experiments."""
from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace

from reproductions.paths import load_paths

from torch import nn
from torch.nn import functional as F


_HISTORIES = {
    "res18_unet": frozenset((1,)),
    "res18_utae": frozenset((5,)),
    "swin_unet": frozenset((1, 5)),
    "segformer_b2": frozenset((1, 5)),
    "convlstm": frozenset((5,)),
}
ARCHITECTURES = tuple(_HISTORIES)
DIRECT_METHODS = frozenset(("control", "cosine_erm", "block_specialist"))


@dataclass(frozen=True)
class ArchitectureConfig:
    module: str
    class_name: str
    kwargs: dict
    learning_rate: float


@dataclass(frozen=True)
class PretrainedAsset:
    path: Path
    sha256: str


def swin_pretrained_asset():
    return PretrainedAsset(
        load_paths().root / "cache/swin/swin_tiny_patch4_window7_224.pth",
        "9f71c168d837d1b99dd1dc29e14990a7a9e8bdc5f673d46b04fe36fe15590ad3",
    )


def segformer_pretrained_assets():
    root = load_paths().root / "cache/huggingface/nvidia-mit-b2-3bb39e87"
    return (
        PretrainedAsset(root / "config.json",
            "d9a879499e7d73e2b33af0638cee320b1070c8f0dadb620eac8907df3d18caa9"),
        PretrainedAsset(root / "pytorch_model.bin",
            "4500b5665471b593e6757e15bcca5034f433fe3902fe8ec2b7230774a57f264f"),
    )


def verify_asset(path, expected_sha256):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"pretrained asset is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"pretrained asset checksum mismatch: {path}")


def swin_runtime_config(input_channels, pretrained_path):
    swin = SimpleNamespace(PATCH_SIZE=4, IN_CHANS=input_channels, EMBED_DIM=96,
        DEPTHS=[2, 2, 2, 2], NUM_HEADS=[3, 6, 12, 24], WINDOW_SIZE=7,
        MLP_RATIO=4.0, QKV_BIAS=True, QK_SCALE=None, APE=False, PATCH_NORM=True)
    model = SimpleNamespace(DROP_RATE=0.0, DROP_PATH_RATE=0.2,
        LABEL_SMOOTHING=0.1, NAME="swin_tiny_patch4_window7_224",
        PRETRAIN_CKPT=str(pretrained_path), SWIN=swin)
    return SimpleNamespace(DATA=SimpleNamespace(IMG_SIZE=224), MODEL=model,
                           TRAIN=SimpleNamespace(USE_CHECKPOINT=False))


def canonical_architecture(history):
    if history == 1:
        return "res18_unet"
    if history == 5:
        return "res18_utae"
    raise ValueError(f"unsupported history T={history}")


def validate_architecture(architecture, history):
    if architecture not in _HISTORIES:
        raise ValueError(f"unknown architecture {architecture!r}")
    if history not in _HISTORIES[architecture]:
        raise ValueError(f"{architecture} does not support T={history}")


def resolve_architecture(history, requested):
    architecture = requested or canonical_architecture(history)
    validate_architecture(architecture, history)
    return architecture


def checkpoint_architecture(payload, expected=None):
    architecture = payload.get("architecture") or canonical_architecture(payload["history"])
    if expected is not None and architecture != expected:
        raise ValueError(
            f"checkpoint architecture {architecture!r} does not match {expected!r}")
    validate_architecture(architecture, payload["history"])
    return architecture


def validate_run_source(architecture, history, method, *, bootstrap,
                        initial_checkpoint, evaluate_only):
    validate_architecture(architecture, history)
    canonical = architecture == canonical_architecture(history)
    if not canonical and method not in DIRECT_METHODS:
        raise ValueError(f"method {method!r} does not support architecture {architecture!r}")
    if evaluate_only is not None:
        if bootstrap or initial_checkpoint is not None:
            raise ValueError("evaluate-only supplies its own checkpoint")
        return
    if bootstrap and method != "control":
        raise ValueError("bootstrap requires method 'control'")
    if canonical and bootstrap:
        raise ValueError("canonical architecture does not support bootstrap")
    if not canonical and bool(bootstrap) == bool(initial_checkpoint):
        raise ValueError(
            "noncanonical training requires exactly one of bootstrap or initial checkpoint")


def architecture_config(architecture, history, hparams=None):
    validate_architecture(architecture, history)
    channels = 40 if history == 1 else 33
    if architecture in ("res18_unet", "res18_utae"):
        if hparams is None:
            raise ValueError("canonical architecture requires checkpoint hyperparameters")
        kwargs = dict(hparams)
        kwargs["encoder_weights"] = None
        if architecture == "res18_utae":
            kwargs.pop("use_doy", None)
        class_name = "SMPModel" if architecture == "res18_unet" else "SMPTempModel"
        return ArchitectureConfig(f"models.{class_name}", class_name, kwargs, 0.001)
    if architecture == "swin_unet":
        kwargs = dict(n_channels=channels * history,
                      flatten_temporal_dimension=True,
                      pos_class_weight=236,
                      loss_function="Focal", crop_before_eval=True,
                      encoder_weights="imagenet")
        return ArchitectureConfig("models.SwinUnetLightning", "SwinUnetLightning",
                                  kwargs, 0.001)
    if architecture == "segformer_b2":
        pretrained_root = str(segformer_pretrained_assets()[0].path.parent)
        kwargs = dict(model_name="segformer-b2", n_channels=channels * history,
                      flatten_temporal_dimension=True,
                      pos_class_weight=236, loss_function="Focal",
                      crop_before_eval=True, encoder_weights=pretrained_root)
        return ArchitectureConfig("models.SegFormerLightning", "SegFormerLightning",
                                  kwargs, 0.001)
    kwargs = dict(n_channels=channels, flatten_temporal_dimension=False,
                  pos_class_weight=236, loss_function="Jaccard",
                  img_height_width=(128, 128), kernel_size=(3, 3), num_layers=1)
    return ArchitectureConfig("models.ConvLSTMLightning", "ConvLSTMLightning",
                              kwargs, 0.01)


def make_architecture(architecture, history, hparams=None):
    config = architecture_config(architecture, history, hparams)
    cls = getattr(importlib.import_module(config.module), config.class_name)
    if architecture == "swin_unet":
        kwargs = dict(config.kwargs)
        kwargs["encoder_weights"] = None
        base = cls(**kwargs)
        asset = swin_pretrained_asset()
        verify_asset(asset.path, asset.sha256)
        base.model.load_from(swin_runtime_config(config.kwargs["n_channels"], asset.path))
        return base
    if architecture == "segformer_b2":
        for asset in segformer_pretrained_assets():
            verify_asset(asset.path, asset.sha256)
    return cls(**config.kwargs)


class DirectForecaster(nn.Module):
    """Expose a common packed-input interface without backbone internals."""
    def __init__(self, base, history, channels, input_size=None):
        super().__init__()
        self.base = base
        self.history = history
        self.channels = channels
        self.input_size = input_size

    def forward_base(self, packed):
        inputs = packed[:, :, :self.channels]
        height, width = inputs.shape[-2:]
        if self.input_size is not None:
            if height > self.input_size or width > self.input_size:
                raise ValueError("input exceeds fixed backbone size")
            pad_h, pad_w = self.input_size - height, self.input_size - width
            inputs = F.pad(inputs, (pad_w // 2, pad_w - pad_w // 2,
                                    pad_h // 2, pad_h - pad_h // 2))
        logits = self.base(inputs)
        if logits.shape[-2:] != (height, width):
            top = (logits.shape[-2] - height) // 2
            left = (logits.shape[-1] - width) // 2
            logits = logits[..., top:top + height, left:left + width]
        return logits

    def forward(self, packed, details=False):
        logits = self.forward_base(packed)
        if details:
            return logits, (), None, None
        return logits

    def compute_loss(self, logits, target):
        return self.base.compute_loss(logits, target)
