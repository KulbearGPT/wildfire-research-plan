"""Fixed mechanisms for the three-direction experiment; no dataset dependencies."""
import torch
from torch import nn
from torch.nn import functional as F


def condition_ids(packed):
    spatial = packed[:, -1, -2].flatten(1).any(1)
    global_fire = packed[:, -1, -1].flatten(1).all(1)
    return 2 * spatial.long() + global_fire.long()


def teacher_ids(packed):
    groups = condition_ids(packed)
    severity = packed[:, -1, -2].mean(dim=(-2, -1))
    # 0 ERM, 1 X22 (also compound fallback), 2 mild, 3 severe.
    return torch.where(groups == 2, 2 + (severity > .375).long(),
                       torch.where(groups == 0, 0, 1))


class StatsBatchNorm(nn.Module):
    """Shared affine, condition-specific moments; calibration never fits weights."""
    def __init__(self, original, banks):
        super().__init__()
        self.eps = original.eps
        self.banks = banks
        self.weight = nn.Parameter(original.weight.detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(original.bias.detach().clone(), requires_grad=False)
        self.register_buffer('running_mean', original.running_mean.repeat(banks, 1))
        self.register_buffer('running_var', original.running_var.repeat(banks, 1))
        self.register_buffer('count', torch.zeros(banks, dtype=torch.long))
        self.register_buffer('total', torch.zeros(banks, original.num_features, dtype=torch.float64))
        self.register_buffer('squared_total', torch.zeros_like(self.total))
        self.groups = None
        self.calibrating = False

    def begin_calibration(self):
        self.count.zero_(); self.total.zero_(); self.squared_total.zero_()
        self.calibrating = True

    def finish_calibration(self):
        if (self.count < 2).any():
            raise ValueError('insufficient calibration support for a BN condition')
        count = self.count[:, None]
        mean = self.total / count
        variance = (self.squared_total - self.total.square() / count) / (count - 1)
        self.running_mean.copy_(mean)
        self.running_var.copy_(variance.clamp_min(0))
        self.calibrating = False

    def forward(self, x, groups=None):
        groups = self.groups if groups is None else groups
        if self.banks == 1:
            groups = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        if groups is None or len(groups) != x.shape[0]:
            raise ValueError('BN routing does not match model batch')
        output = torch.empty_like(x)
        for group in range(self.banks):
            indices = torch.where(groups == group)[0]
            if not len(indices):
                continue
            values = x[indices]
            if self.calibrating:
                with torch.no_grad():
                    self.count[group] += values.numel() // values.shape[1]
                    self.total[group] += values.sum(dim=(0, 2, 3), dtype=torch.float64)
                    self.squared_total[group] += values.double().square().sum(dim=(0, 2, 3))
                output[indices] = F.batch_norm(values, None, None, self.weight,
                                               self.bias, True, 0., self.eps)
            else:
                output[indices] = F.batch_norm(values, self.running_mean[group],
                    self.running_var[group], self.weight, self.bias, False, 0., self.eps)
        return output


class StatisticsModel(nn.Module):
    def __init__(self, forecaster, banks):
        super().__init__()
        self.forecaster = forecaster
        self.banks = banks
        def replace(module):
            for name, child in list(module.named_children()):
                if isinstance(child, nn.BatchNorm2d):
                    setattr(module, name, StatsBatchNorm(child, banks))
                else:
                    replace(child)
        replace(forecaster)
        self.requires_grad_(False)

    def statistic_layers(self):
        return [m for m in self.modules() if isinstance(m, StatsBatchNorm)]

    def forward(self, packed):
        groups = condition_ids(packed)
        for layer in self.statistic_layers():
            layer.groups = groups
        return self.forecaster(packed)

    def compute_loss(self, logits, target):
        return self.forecaster.compute_loss(logits, target)


def midpoint_state(left, right):
    if left.keys() != right.keys():
        raise ValueError('checkpoint keys differ')
    result = {}
    for key, value in left.items():
        other = right[key]
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError(f'incompatible checkpoint tensor: {key}')
        if value.is_floating_point():
            if not torch.isfinite(value).all() or not torch.isfinite(other).all():
                raise ValueError(f'non-finite checkpoint tensor: {key}')
            result[key] = .5 * value + .5 * other
        elif key.endswith('num_batches_tracked'):
            result[key] = value.clone()
        elif torch.equal(value, other):
            result[key] = value.clone()
        else:
            raise ValueError(f'incompatible non-BN buffer: {key}')
    return result


def bernoulli_kl(student, teacher):
    teacher = teacher.detach()
    p = teacher.sigmoid()
    return (p * (F.logsigmoid(teacher) - F.logsigmoid(student))
            + (1 - p) * (F.logsigmoid(-teacher) - F.logsigmoid(-student))).mean()
