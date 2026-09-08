"""Minimal backbone boundary for matched cross-history experiments."""
from dataclasses import dataclass
import importlib

from torch import nn


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
        kwargs = dict(model_name="segformer-b2", n_channels=channels * history,
                      flatten_temporal_dimension=True,
                      pos_class_weight=236, loss_function="Focal",
                      crop_before_eval=True, encoder_weights="imagenet")
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
    return cls(**config.kwargs)


class DirectForecaster(nn.Module):
    """Expose a common packed-input interface without backbone internals."""
    def __init__(self, base, history, channels):
        super().__init__()
        self.base = base
        self.history = history
        self.channels = channels

    def forward(self, packed, details=False):
        logits = self.base(packed[:, :, :self.channels])
        if details:
            return logits, (), None, None
        return logits

    def compute_loss(self, logits, target):
        return self.base.compute_loss(logits, target)
