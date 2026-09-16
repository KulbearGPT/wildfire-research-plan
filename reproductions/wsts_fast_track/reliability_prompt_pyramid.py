"""Scale-matched spatial reliability prompts for a U-Net encoder pyramid."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .reliability_token_conv import InputReliabilityTokenConv2d


PROMPT_CHANNELS = (64, 64, 128, 256, 512)
PROMPT_PARAMETER_COUNT = sum(PROMPT_CHANNELS)
PROMPT_POOLING = "adaptive-area-average"
INPUT_PROMPT = "local-invalid-coverage-token"
SEVERITY_THRESHOLD = 0.375


def apply_reliability_prompts(
    features: Sequence[torch.Tensor],
    invalidity: torch.Tensor,
    tokens: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Add channel tokens scaled by invalid-cell coverage at each resolution."""

    if (
        invalidity.ndim != 4
        or invalidity.shape[1] != 1
        or len(features) == 0
        or len(features) != len(tokens)
    ):
        raise ValueError("prompt features, invalidity, and tokens are incompatible")
    prompted: list[torch.Tensor] = []
    for feature, token in zip(features, tokens, strict=True):
        if (
            feature.ndim != 4
            or feature.shape[0] != invalidity.shape[0]
            or token.ndim != 1
            or token.shape[0] != feature.shape[1]
        ):
            raise ValueError("one reliability prompt scale is incompatible")
        coverage = F.adaptive_avg_pool2d(invalidity, feature.shape[-2:])
        prompted.append(
            feature + coverage.to(feature.dtype) * token[None, :, None, None]
        )
    return tuple(prompted)


class ReliabilityPromptPyramid(torch.nn.Module):
    """Inject the spatial invalidity map into every post-input encoder scale."""

    def __init__(self, base_model: torch.nn.Module) -> None:
        super().__init__()
        self.base_model = base_model
        encoder_channels = tuple(self.base_model.model.encoder.out_channels)
        if len(encoder_channels) < 2:
            raise ValueError("RPP requires a multi-scale encoder")
        self.prompt_channels = encoder_channels[1:]
        self.prompt_tokens = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(channels)) for channels in self.prompt_channels]
        )
        self.prompt_parameter_count = sum(
            parameter.numel() for parameter in self.prompt_tokens
        )

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        if packed.ndim != 5 or packed.shape[1] != 1 or packed.shape[2] != 41:
            raise ValueError("RPP input must have shape [B,1,41,H,W]")
        features = packed[:, 0, :40]
        invalidity = packed[:, 0, 40:41].clamp(0.0, 1.0)
        encoded = tuple(self.base_model.model.encoder(features))
        if tuple(feature.shape[1] for feature in encoded[1:]) != self.prompt_channels:
            raise ValueError("runtime encoder channels differ from RPP initialization")
        prompted = apply_reliability_prompts(
            encoded[1:], invalidity, self.prompt_tokens
        )
        decoder_features = self.base_model.model.decoder(encoded[0], *prompted)
        return self.base_model.model.segmentation_head(decoder_features)

    def compute_loss(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return self.base_model.compute_loss(logits, target)


class CompleteReliabilityPromptPyramid(ReliabilityPromptPyramid):
    """Combine D4's input token with prompts at every encoder scale."""

    def __init__(self, base_model: torch.nn.Module) -> None:
        base_model.model.encoder.conv1 = InputReliabilityTokenConv2d(
            base_model.model.encoder.conv1
        )
        super().__init__(base_model)
        self.prompt_parameter_count += self.input_token.invalid_token.numel()

    @property
    def input_token(self) -> InputReliabilityTokenConv2d:
        return self.base_model.model.encoder.conv1

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        if packed.ndim != 5 or packed.shape[1] != 1 or packed.shape[2] != 41:
            raise ValueError("complete RPP input must have shape [B,1,41,H,W]")
        features_and_invalidity = packed[:, 0]
        invalidity = features_and_invalidity[:, 40:41].clamp(0.0, 1.0)
        encoded = tuple(
            self.base_model.model.encoder(features_and_invalidity)
        )
        if tuple(feature.shape[1] for feature in encoded[1:]) != self.prompt_channels:
            raise ValueError("runtime encoder channels differ from CRPP initialization")
        prompted = apply_reliability_prompts(
            encoded[1:], invalidity, self.prompt_tokens
        )
        decoder_features = self.base_model.model.decoder(encoded[0], *prompted)
        return self.base_model.model.segmentation_head(decoder_features)


class SeverityAdaptiveReliabilityPrompting(CompleteReliabilityPromptPyramid):
    """Route mild blocks to deep prompts and severe blocks to the input token."""

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        if packed.ndim != 5 or packed.shape[1] != 1 or packed.shape[2] != 41:
            raise ValueError("SARP input must have shape [B,1,41,H,W]")
        features = packed[:, 0, :40]
        invalidity = packed[:, 0, 40:41].clamp(0.0, 1.0)
        severity = invalidity.mean(dim=(2, 3), keepdim=True)
        severe = severity > SEVERITY_THRESHOLD
        mild = (severity > 0.0) & ~severe
        input_invalidity = invalidity * severe.to(invalidity.dtype)
        deep_invalidity = invalidity * mild.to(invalidity.dtype)
        encoded = tuple(
            self.base_model.model.encoder(
                torch.cat((features, input_invalidity), dim=1)
            )
        )
        if tuple(feature.shape[1] for feature in encoded[1:]) != self.prompt_channels:
            raise ValueError("runtime encoder channels differ from SARP initialization")
        prompted = apply_reliability_prompts(
            encoded[1:], deep_invalidity, self.prompt_tokens
        )
        decoder_features = self.base_model.model.decoder(encoded[0], *prompted)
        return self.base_model.model.segmentation_head(decoder_features)
