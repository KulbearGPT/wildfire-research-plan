"""Corrected-index, from-scratch baseline definitions for the active track."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import torch

from .environment_dro import balanced_year_sampling_weights
from .environment_dro import resolve_dataset_index
from .prototype import (
    install_training_fire_and_block_dropout,
    install_training_fire_dropout,
)


TrainingPolicy = Literal[
    "clean", "fire", "fire-block", "year-balanced-fire-block"
]


@dataclass(frozen=True)
class CorrectedBaselineSpec:
    """One minimal corrected-index baseline screen."""

    baseline_id: str
    experiment_id: Literal["C00", "C02"]
    training_policy: TrainingPolicy
    seed: int = 0
    max_steps: int = 3_000


CORRECTED_BASELINES: Final[dict[str, CorrectedBaselineSpec]] = {
    "B0": CorrectedBaselineSpec("B0", "C00", "clean"),
    "B1": CorrectedBaselineSpec("B1", "C02", "clean"),
    "B2": CorrectedBaselineSpec("B2", "C00", "fire"),
    "B3": CorrectedBaselineSpec("B3", "C00", "fire-block"),
    "B4": CorrectedBaselineSpec("B4", "C00", "year-balanced-fire-block"),
    "B5": CorrectedBaselineSpec("B5", "C02", "fire-block"),
}


def corrected_baseline_spec(baseline_id: str) -> CorrectedBaselineSpec:
    """Return one registered corrected-index baseline."""

    try:
        return CORRECTED_BASELINES[baseline_id]
    except KeyError as error:
        raise ValueError(f"unknown corrected baseline: {baseline_id}") from error


def install_balanced_year_training_loader(upstream_root: Path | str) -> None:
    """Give each training year equal sampler mass without changing examples."""

    upstream = Path(upstream_root).resolve()
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "src"))
    datamodule = importlib.import_module("dataloader.FireSpreadDataModule")
    datamodule_class = datamodule.FireSpreadDataModule

    def train_dataloader_with_balanced_years(self):
        weights = balanced_year_sampling_weights(self.train_dataset)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights,
            num_samples=len(self.train_dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(0),
        )
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    datamodule_class.train_dataloader = train_dataloader_with_balanced_years


def install_corrected_baseline(upstream_root: Path | str, baseline_id: str) -> None:
    """Install first-match resolution and only the selected train corruption."""

    spec = corrected_baseline_spec(baseline_id)
    upstream = Path(upstream_root).resolve()
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "src"))
    dataset_module = importlib.import_module("dataloader.FireSpreadDataset")
    dataset_class = dataset_module.FireSpreadDataset
    dataset_class.find_image_index_from_dataset_index = resolve_dataset_index

    if spec.training_policy == "fire":
        install_training_fire_dropout(upstream)
    elif spec.training_policy in {"fire-block", "year-balanced-fire-block"}:
        install_training_fire_and_block_dropout(upstream)
        if spec.training_policy == "year-balanced-fire-block":
            install_balanced_year_training_loader(upstream)
