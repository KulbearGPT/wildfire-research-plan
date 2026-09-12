"""Small mechanisms for the three cross-history reliability experiments."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


_BatchNorm = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


class _Moments:
    def __init__(self, channels: int):
        self.sum = torch.zeros(channels, dtype=torch.float64)
        self.square_sum = torch.zeros(channels, dtype=torch.float64)
        self.count = 0

    @staticmethod
    def from_tensor(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Reduce one activation on its device, then transfer channel summaries."""
        value = value.detach()
        axes = tuple(axis for axis in range(value.ndim) if axis != 1)
        channel_sum = value.sum(dim=axes, dtype=torch.float64).cpu()
        square_sum = value.square().sum(dim=axes, dtype=torch.float64).cpu()
        count = value.numel() // value.shape[1]
        return channel_sum, square_sum, count

    def add_summary(self, summary: tuple[torch.Tensor, torch.Tensor, int]) -> None:
        channel_sum, square_sum, count = summary
        self.sum += channel_sum
        self.square_sum += square_sum
        self.count += count

    def result(self) -> dict[str, object]:
        if self.count < 2:
            raise ValueError("a BN moment bank needs at least two values per channel")
        mean = self.sum / self.count
        variance = (self.square_sum - self.sum.square() / self.count) / (self.count - 1)
        return {"mean": mean.float(), "var": variance.clamp_min(0).float(), "count": self.count}


class MomentCollector:
    """Collect BN moments into common and two routed banks.

    Set ``bank`` to 0 (block absent) or 1 (block present) before each
    homogeneous forward. The model remains in evaluation mode. With
    ``batch_statistics=True``, only BN layers normalize with current-batch
    statistics; dropout stays inactive and BN running buffers stay untouched.
    """

    def __init__(self, model: nn.Module, *, batch_statistics: bool = False):
        self.model = model
        self._bank = 0
        self._common: dict[str, _Moments] = {}
        self._banks: dict[int, dict[str, _Moments]] = {0: {}, 1: {}}
        self._handles = []
        self._bn_flags = []
        model.eval()
        for name, module in model.named_modules():
            if isinstance(module, _BatchNorm):
                self._bn_flags.append((module, module.training, module.track_running_stats))
                if batch_statistics:
                    module.training = True
                    module.track_running_stats = False
                self._common[name] = _Moments(module.num_features)
                self._banks[0][name] = _Moments(module.num_features)
                self._banks[1][name] = _Moments(module.num_features)
                self._handles.append(module.register_forward_pre_hook(self._hook(name)))
        if not self._common:
            raise ValueError("model contains no BatchNorm layers")

    @property
    def bank(self) -> int:
        return self._bank

    @bank.setter
    def bank(self, value: int) -> None:
        if value not in (0, 1):
            raise ValueError("bank must be 0 (block absent) or 1 (block present)")
        self._bank = value

    def _hook(self, name: str):
        def collect(_module, inputs):
            summary = _Moments.from_tensor(inputs[0])
            self._common[name].add_summary(summary)
            self._banks[self.bank][name].add_summary(summary)
        return collect

    def finalize(self):
        """Remove hooks and return ``(common_bank, {0: bank, 1: bank})``."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        try:
            try:
                common = {name: moments.result() for name, moments in self._common.items()}
                banks = {
                    route: {name: moments.result() for name, moments in values.items()}
                    for route, values in self._banks.items()
                }
            except ValueError as exc:
                raise ValueError("cannot finalize an empty or undersized BN bank") from exc
        finally:
            for module, training, track_running_stats in self._bn_flags:
                module.training = training
                module.track_running_stats = track_running_stats
        return common, banks


def apply_bn_bank(model: nn.Module, bank: Mapping[str, Mapping[str, object]]) -> None:
    """Copy named means and variances into BN running buffers only."""
    modules = dict(model.named_modules())
    expected = {name for name, module in modules.items() if isinstance(module, _BatchNorm)}
    if set(bank) != expected:
        raise ValueError("bank names do not exactly match model BatchNorm names")
    with torch.no_grad():
        for name in sorted(expected):
            module = modules[name]
            entry = bank[name]
            mean = torch.as_tensor(entry["mean"])
            variance = torch.as_tensor(entry["var"])
            if mean.shape != module.running_mean.shape or variance.shape != module.running_var.shape:
                raise ValueError(f"BN bank shape mismatch for {name}")
            if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
                raise ValueError(f"non-finite BN bank values for {name}")
            module.running_mean.copy_(mean.to(module.running_mean))
            module.running_var.copy_(variance.to(module.running_var))


def _residual_path(input_channels: int, output_channels: int) -> nn.Sequential:
    path = nn.Sequential(
        nn.Conv2d(input_channels, 16, 3, padding=1, bias=False),
        nn.SiLU(),
        nn.Conv2d(16, output_channels, 3, padding=1, bias=False),
    )
    nn.init.zeros_(path[-1].weight)
    return path


class ShallowInputStem(nn.Module):
    """Identity-initialized mixed or static/dynamic residual input stem."""

    def __init__(self, channels: int, static_indices, mode: str):
        super().__init__()
        static = tuple(sorted(set(static_indices)))
        if channels < 1 or any(index < 0 or index >= channels for index in static):
            raise ValueError("static indices must be unique valid channel indices")
        dynamic = tuple(index for index in range(channels) if index not in static)
        if mode not in ("mixed", "typed"):
            raise ValueError("mode must be 'mixed' or 'typed'")
        if mode == "typed" and (not static or not dynamic):
            raise ValueError("typed mode needs non-empty static and dynamic groups")
        self.channels = channels
        self.static_indices = static
        self.dynamic_indices = dynamic
        self.mode = mode
        if mode == "mixed":
            self.paths = nn.ModuleList([_residual_path(channels, channels)])
        else:
            self.paths = nn.ModuleList([
                _residual_path(len(static), len(static)),
                _residual_path(len(dynamic), len(dynamic)),
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim not in (4, 5) or x.shape[-3] != self.channels:
            raise ValueError("input must be BCHW or BTCHW with the configured channels")
        original_shape = x.shape
        flat = x.reshape(-1, self.channels, *x.shape[-2:])
        if self.mode == "mixed":
            result = flat + self.paths[0](flat)
        else:
            result = flat.clone()
            for indices, path in zip((self.static_indices, self.dynamic_indices), self.paths):
                values = flat[:, indices]
                result[:, indices] = values + path(values)
        return result.reshape(original_shape)


def observable_routes(packed: torch.Tensor) -> torch.Tensor:
    """Return clean/fire-only/small-block/large-block route IDs per sample."""
    if packed.ndim not in (4, 5) or packed.shape[-3] < 2:
        raise ValueError("packed input must end in channel, height, width dimensions")
    spatial = packed[..., -2, :, :]
    fire = packed[..., -1, :, :]
    reduce_axes = tuple(range(1, spatial.ndim))
    block_fraction = spatial.float().mean(dim=reduce_axes)
    block = block_fraction > 0
    fire_missing = fire.bool().any(dim=reduce_axes)
    routes = fire_missing.long()
    routes = torch.where(block & (block_fraction <= 0.375), torch.full_like(routes, 2), routes)
    routes = torch.where(block & (block_fraction > 0.375), torch.full_like(routes, 3), routes)
    return routes


def bernoulli_teacher_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Mean Bernoulli KL(teacher || student), with the teacher detached."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have identical shapes")
    if not torch.isfinite(student_logits).all() or not torch.isfinite(teacher_logits).all():
        raise ValueError("student and teacher logits must be finite")
    teacher = teacher_logits.detach()
    probability = teacher.sigmoid()
    kl = probability * (F.logsigmoid(teacher) - F.logsigmoid(student_logits))
    kl += (1 - probability) * (F.logsigmoid(-teacher) - F.logsigmoid(-student_logits))
    result = kl.mean()
    if not torch.isfinite(result):
        raise ValueError("Bernoulli KL produced a non-finite value")
    return result
