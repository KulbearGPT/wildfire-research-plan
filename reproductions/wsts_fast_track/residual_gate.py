"""P04 single-forward spatial residual gate on frozen P00 features."""

from __future__ import annotations

import importlib
import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .missingness import structured_block_mask


PROTOTYPE_ID = "P04-FrozenP00-SpatialResidualGate"
SPATIAL_PROTOTYPE_ID = "P05-FrozenP00-SpatialResidualGate3x3"
LAST_BLOCK_PROTOTYPE_ID = "P06-FrozenP00-LastBlockRouter"
BELIEF_PROTOTYPE_ID = "P07-StochasticBeliefResidual"
TEACHER_BELIEF_PROTOTYPE_ID = "P08-TeacherPosteriorBelief"
GATE_TRAINING_STEPS = 1_000
BELIEF_TRAINING_STEPS = 3_000
BELIEF_SAMPLE_COUNT = 4
TEACHER_BELIEF_KL_WEIGHT = 1e-3
TEACHER_BELIEF_RECONSTRUCTION_WEIGHT = 1e-2
PROCESSED_DYNAMIC_NON_FIRE = tuple(range(12)) + (15,) + tuple(range(33, 38))


def apply_masked_residual(
    default_logits: torch.Tensor,
    residual_logits: torch.Tensor,
    missing_mask: torch.Tensor,
) -> torch.Tensor:
    """Add a learned correction only at pixels declared spatially missing."""

    if (
        default_logits.shape != residual_logits.shape
        or missing_mask.shape != default_logits.shape
    ):
        raise ValueError("default logits, residual, and mask must have identical shapes")
    return default_logits + residual_logits * missing_mask.to(default_logits.dtype)


def apply_processed_block_dropout(
    processed: torch.Tensor,
    *,
    fraction: float,
    key_digest: str,
    normalized_active_fire_zero: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply M06/M07-equivalent corruption after synchronized crop/rotation."""

    if processed.ndim != 4 or processed.shape[1] != 40:
        raise ValueError("processed C00 input must have shape (T, 40, H, W)")
    mask_array = structured_block_mask(
        processed.shape[-2],
        processed.shape[-1],
        fraction,
        key_digest=key_digest,
    )
    mask = torch.as_tensor(mask_array, dtype=torch.bool, device=processed.device)
    blocked = processed.clone()
    for channel in PROCESSED_DYNAMIC_NON_FIRE:
        blocked[:, channel, mask] = 0.0
    blocked[:, 38, mask] = normalized_active_fire_zero
    blocked[:, 39, mask] = 0.0
    mask_channel = mask.to(processed.dtype)[None, None].expand(
        processed.shape[0], 1, -1, -1
    )
    return torch.cat((blocked, mask_channel), dim=1), mask


def install_training_processed_block_dropout(upstream_root: Path) -> None:
    """Patch C00 preprocessing to emit always-blocked inputs plus routing mask."""

    upstream = Path(upstream_root).resolve()
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "src"))
    dataset_module = importlib.import_module("dataloader.FireSpreadDataset")
    dataset_class = dataset_module.FireSpreadDataset
    original_preprocess = dataset_class.preprocess_and_augment

    def preprocess_with_block(self: Any, *args: Any, **kwargs: Any):
        processed, target = original_preprocess(self, *args, **kwargs)
        if getattr(self, "is_train", False) is not True:
            mask = processed.new_zeros(
                (processed.shape[0], 1, processed.shape[-2], processed.shape[-1])
            )
            return torch.cat((processed, mask), dim=1), target
        fraction = 0.25 if float(np.random.random()) < 0.5 else 0.50
        normalized_zero = float(-self.means[0, 22, 0, 0] / self.stds[0, 22, 0, 0])
        routed, _ = apply_processed_block_dropout(
            processed,
            fraction=fraction,
            key_digest=np.random.bytes(32).hex(),
            normalized_active_fire_zero=normalized_zero,
        )
        return routed, target

    dataset_class.preprocess_and_augment = preprocess_with_block


def install_training_teacher_belief_dropout(upstream_root: Path) -> None:
    """Emit corrupted+mask and the aligned clean teacher during P08 training."""

    upstream = Path(upstream_root).resolve()
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "src"))
    dataset_module = importlib.import_module("dataloader.FireSpreadDataset")
    dataset_class = dataset_module.FireSpreadDataset
    original_preprocess = dataset_class.preprocess_and_augment

    def preprocess_with_teacher(self: Any, *args: Any, **kwargs: Any):
        processed, target = original_preprocess(self, *args, **kwargs)
        if getattr(self, "is_train", False) is not True:
            mask = processed.new_zeros(
                (processed.shape[0], 1, processed.shape[-2], processed.shape[-1])
            )
            return torch.cat((processed, mask), dim=1), target
        fraction = 0.25 if float(np.random.random()) < 0.5 else 0.50
        normalized_zero = float(-self.means[0, 22, 0, 0] / self.stds[0, 22, 0, 0])
        routed, _ = apply_processed_block_dropout(
            processed,
            fraction=fraction,
            key_digest=np.random.bytes(32).hex(),
            normalized_active_fire_zero=normalized_zero,
        )
        return torch.cat((routed, processed), dim=1), target

    dataset_class.preprocess_and_augment = preprocess_with_teacher


class FrozenSpatialResidualGate(torch.nn.Module):
    """Add a minimal spatial correction to frozen P00 decoder features."""

    def __init__(
        self,
        default_model: torch.nn.Module,
        *,
        residual_kernel_size: int = 1,
    ) -> None:
        super().__init__()
        if residual_kernel_size not in {1, 3}:
            raise ValueError("residual kernel size must be 1 or 3")
        self.default_model = default_model
        self.default_model.requires_grad_(False)
        self.default_model.eval()
        self.residual_head = torch.nn.Conv2d(
            16,
            1,
            kernel_size=residual_kernel_size,
            padding=residual_kernel_size // 2,
        )
        torch.nn.init.zeros_(self.residual_head.weight)
        torch.nn.init.zeros_(self.residual_head.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.default_model.eval()
        return self

    def trainable_parameters(self):
        return self.residual_head.parameters()

    def forward(self, routed_input: torch.Tensor) -> torch.Tensor:
        if (
            routed_input.ndim != 5
            or routed_input.shape[1] != 1
            or routed_input.shape[2] != 41
        ):
            raise ValueError("gate input must have shape (B, 1, 41, H, W)")
        features = routed_input[:, 0, :40]
        missing_mask = routed_input[:, 0, 40:41].bool()
        with torch.no_grad():
            encoded = self.default_model.model.encoder(features)
            decoder_features = self.default_model.model.decoder(*encoded)
            default_logits = self.default_model.model.segmentation_head(
                decoder_features
            )
        residual_logits = self.residual_head(decoder_features.detach())
        return apply_masked_residual(
            default_logits, residual_logits, missing_mask
        )

    def compute_loss(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return self.default_model.compute_loss(logits, target)


class FrozenStochasticBeliefResidual(torch.nn.Module):
    """Marginalize sampled missing-region residuals over frozen P00 features."""

    def __init__(
        self,
        default_model: torch.nn.Module,
        *,
        sample_count: int = BELIEF_SAMPLE_COUNT,
    ) -> None:
        super().__init__()
        if sample_count < 2:
            raise ValueError("belief residual requires at least two samples")
        self.default_model = default_model
        self.default_model.requires_grad_(False)
        self.default_model.eval()
        self.sample_count = sample_count
        self.belief_head = torch.nn.Conv2d(17, 32, kernel_size=3, padding=1)
        self.output_head = torch.nn.Conv2d(16, 1, kernel_size=1)
        torch.nn.init.zeros_(self.belief_head.weight)
        torch.nn.init.zeros_(self.belief_head.bias)
        with torch.no_grad():
            self.belief_head.bias[16:].fill_(-2.0)
        torch.nn.init.zeros_(self.output_head.weight)
        torch.nn.init.zeros_(self.output_head.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.default_model.eval()
        return self

    def trainable_parameters(self):
        return (*self.belief_head.parameters(), *self.output_head.parameters())

    def forward_with_uncertainty(
        self, routed_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            routed_input.ndim != 5
            or routed_input.shape[1] != 1
            or routed_input.shape[2] != 41
        ):
            raise ValueError("belief input must have shape (B, 1, 41, H, W)")
        features = routed_input[:, 0, :40]
        missing_mask = routed_input[:, 0, 40:41].bool()
        mask_float = missing_mask.to(features.dtype)
        with torch.no_grad():
            encoded = self.default_model.model.encoder(features)
            decoder_features = self.default_model.model.decoder(*encoded)
            default_logits = self.default_model.model.segmentation_head(
                decoder_features
            )

        belief_parameters = self.belief_head(
            torch.cat((decoder_features.detach(), mask_float), dim=1)
        )
        mean, log_sigma = belief_parameters.chunk(2, dim=1)
        sigma = torch.exp(log_sigma.clamp(min=-5.0, max=2.0))
        sampled_probabilities = []
        for _ in range(self.sample_count):
            latent = mean + sigma * torch.randn_like(mean)
            residual = self.output_head(latent)
            sampled_logits = default_logits + residual * mask_float
            sampled_probabilities.append(torch.sigmoid(sampled_logits))
        probabilities = torch.stack(sampled_probabilities, dim=0)
        mean_probability = probabilities.mean(dim=0)
        belief_logits = torch.logit(
            mean_probability.clamp(min=1e-6, max=1.0 - 1e-6)
        )
        mean_logits = torch.where(missing_mask, belief_logits, default_logits)
        variance = probabilities.var(dim=0, unbiased=False) * mask_float
        return mean_logits, variance

    def forward(self, routed_input: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_uncertainty(routed_input)
        return logits

    def compute_loss(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return self.default_model.compute_loss(logits, target)


class FrozenTeacherPosteriorBelief(torch.nn.Module):
    """Train from a clean teacher posterior and infer from a corrupted prior."""

    def __init__(
        self,
        default_model: torch.nn.Module,
        *,
        sample_count: int = BELIEF_SAMPLE_COUNT,
    ) -> None:
        super().__init__()
        if sample_count < 2:
            raise ValueError("teacher belief requires at least two samples")
        self.default_model = default_model
        self.default_model.requires_grad_(False)
        self.default_model.eval()
        self.sample_count = sample_count
        self.prior_head = torch.nn.Conv2d(17, 32, kernel_size=3, padding=1)
        self.posterior_head = torch.nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.reconstruction_head = torch.nn.Conv2d(16, 16, kernel_size=1)
        self.output_head = torch.nn.Conv2d(16, 1, kernel_size=1)
        torch.nn.init.zeros_(self.prior_head.weight)
        torch.nn.init.zeros_(self.prior_head.bias)
        torch.nn.init.zeros_(self.posterior_head.weight)
        torch.nn.init.zeros_(self.posterior_head.bias)
        with torch.no_grad():
            self.prior_head.bias[16:].fill_(-2.0)
            self.posterior_head.bias[16:].fill_(-2.0)
        torch.nn.init.dirac_(self.reconstruction_head.weight)
        torch.nn.init.zeros_(self.reconstruction_head.bias)
        torch.nn.init.zeros_(self.output_head.weight)
        torch.nn.init.zeros_(self.output_head.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.default_model.eval()
        return self

    def trainable_parameters(self):
        return (
            *self.posterior_head.parameters(),
            *self.prior_head.parameters(),
            *self.reconstruction_head.parameters(),
            *self.output_head.parameters(),
        )

    def _frozen_features(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            encoded = self.default_model.model.encoder(inputs)
            decoder_features = self.default_model.model.decoder(*encoded)
            logits = self.default_model.model.segmentation_head(decoder_features)
        return decoder_features.detach(), logits

    @staticmethod
    def _distribution_parameters(
        parameters: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_sigma = parameters.chunk(2, dim=1)
        return mean, log_sigma.clamp(min=-2.0, max=2.0)

    def _sample_forecasts(
        self,
        default_logits: torch.Tensor,
        missing_mask: torch.Tensor,
        mean: torch.Tensor,
        log_sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        mask_float = missing_mask.to(default_logits.dtype)
        sigma = torch.exp(log_sigma)
        sampled_probabilities = []
        latents = []
        for _ in range(self.sample_count):
            latent = mean + sigma * torch.randn_like(mean)
            latents.append(latent)
            residual = self.output_head(latent)
            sampled_logits = default_logits + residual * mask_float
            sampled_probabilities.append(torch.sigmoid(sampled_logits))
        probabilities = torch.stack(sampled_probabilities, dim=0)
        mean_probability = probabilities.mean(dim=0)
        belief_logits = torch.logit(
            mean_probability.clamp(min=1e-6, max=1.0 - 1e-6)
        )
        mean_logits = torch.where(missing_mask, belief_logits, default_logits)
        variance = probabilities.var(dim=0, unbiased=False) * mask_float
        return mean_logits, variance, latents

    def training_objective(
        self, teacher_input: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        if (
            teacher_input.ndim != 5
            or teacher_input.shape[1] != 1
            or teacher_input.shape[2] != 81
        ):
            raise ValueError("teacher-belief training input must have 81 channels")
        corrupted = teacher_input[:, 0, :40]
        missing_mask = teacher_input[:, 0, 40:41].bool()
        clean = teacher_input[:, 0, 41:81]
        mask_float = missing_mask.to(corrupted.dtype)
        corrupted_features, default_logits = self._frozen_features(corrupted)
        clean_features, _ = self._frozen_features(clean)
        prior_mean, prior_log_sigma = self._distribution_parameters(
            self.prior_head(torch.cat((corrupted_features, mask_float), dim=1))
        )
        posterior_mean, posterior_log_sigma = self._distribution_parameters(
            self.posterior_head(
                torch.cat((clean_features, corrupted_features), dim=1)
            )
        )
        logits, _, latents = self._sample_forecasts(
            default_logits,
            missing_mask,
            posterior_mean,
            posterior_log_sigma,
        )
        forecast_loss = self.default_model.compute_loss(logits.squeeze(1), target)
        prior_variance = torch.exp(2.0 * prior_log_sigma)
        posterior_variance = torch.exp(2.0 * posterior_log_sigma)
        kl_loss = 0.5 * (
            2.0 * (prior_log_sigma - posterior_log_sigma)
            + (
                posterior_variance
                + (posterior_mean - prior_mean).square()
            )
            / prior_variance
            - 1.0
        ).mean()
        feature_residual = (clean_features - corrupted_features) * mask_float
        reconstruction_error = torch.stack(
            [
                (self.reconstruction_head(latent) - feature_residual).square()
                * mask_float
                for latent in latents
            ],
            dim=0,
        )
        denominator = (
            mask_float.sum() * feature_residual.shape[1] * self.sample_count
        ).clamp_min(1.0)
        reconstruction_loss = reconstruction_error.sum() / denominator
        total = (
            forecast_loss
            + TEACHER_BELIEF_KL_WEIGHT * kl_loss
            + TEACHER_BELIEF_RECONSTRUCTION_WEIGHT * reconstruction_loss
        )
        components = {
            "forecast": forecast_loss,
            "kl": kl_loss,
            "reconstruction": reconstruction_loss,
            "total": total,
        }
        return total, components, logits

    def forward_with_uncertainty(
        self, routed_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            routed_input.ndim != 5
            or routed_input.shape[1] != 1
            or routed_input.shape[2] != 41
        ):
            raise ValueError("teacher-belief inference input must have 41 channels")
        corrupted = routed_input[:, 0, :40]
        missing_mask = routed_input[:, 0, 40:41].bool()
        mask_float = missing_mask.to(corrupted.dtype)
        corrupted_features, default_logits = self._frozen_features(corrupted)
        prior_mean, prior_log_sigma = self._distribution_parameters(
            self.prior_head(torch.cat((corrupted_features, mask_float), dim=1))
        )
        logits, variance, _ = self._sample_forecasts(
            default_logits, missing_mask, prior_mean, prior_log_sigma
        )
        return logits, variance

    def forward(self, routed_input: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_uncertainty(routed_input)
        return logits

    def compute_loss(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return self.default_model.compute_loss(logits, target)


class FrozenLastBlockRouter(torch.nn.Module):
    """Share P00 through decoder block four and adapt only its final block."""

    def __init__(self, default_model: torch.nn.Module) -> None:
        super().__init__()
        self.default_model = default_model
        self.default_model.requires_grad_(False)
        self.default_model.eval()
        decoder = self.default_model.model.decoder
        if len(decoder.blocks) < 1:
            raise ValueError("P00 decoder must expose at least one block")
        self.adapted_block = copy.deepcopy(decoder.blocks[-1])
        self.adapted_head = copy.deepcopy(
            self.default_model.model.segmentation_head
        )
        self.adapted_block.requires_grad_(True)
        self.adapted_head.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.default_model.eval()
        return self

    def trainable_parameters(self):
        return (
            *self.adapted_block.parameters(),
            *self.adapted_head.parameters(),
        )

    def forward(self, routed_input: torch.Tensor) -> torch.Tensor:
        if (
            routed_input.ndim != 5
            or routed_input.shape[1] != 1
            or routed_input.shape[2] != 41
        ):
            raise ValueError("router input must have shape (B, 1, 41, H, W)")
        inputs = routed_input[:, 0, :40]
        missing_mask = routed_input[:, 0, 40:41].bool()
        decoder = self.default_model.model.decoder
        with torch.no_grad():
            encoded = self.default_model.model.encoder(inputs)
            features = encoded[1:][::-1]
            if not features:
                raise ValueError("P00 encoder did not expose decoder features")
            x = decoder.center(features[0])
            skips = features[1:]
            for index, block in enumerate(decoder.blocks[:-1]):
                skip = skips[index] if index < len(skips) else None
                x = block(x, skip)
            final_index = len(decoder.blocks) - 1
            final_skip = skips[final_index] if final_index < len(skips) else None
            default_features = decoder.blocks[-1](x, final_skip)
            default_logits = self.default_model.model.segmentation_head(
                default_features
            )
        adapted_features = self.adapted_block(
            x.detach(), None if final_skip is None else final_skip.detach()
        )
        adapted_logits = self.adapted_head(adapted_features)
        return torch.where(missing_mask, adapted_logits, default_logits)

    def compute_loss(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return self.default_model.compute_loss(logits, target)
