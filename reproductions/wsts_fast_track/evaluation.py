"""Deterministic 2021-only dataset path for M00--M07 engineering evaluation."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .contract import ExperimentSpec, experiment_spec
from .entrypoint import TRAIN_YEARS, _install_runtime_contract, load_training_stats
from .matrix import CORRUPTIONS
from .missingness import CorruptionResult, apply_corruption


ENGINEERING_YEARS = (2021,)
EFFECTIVE_HISTORY = 6


@dataclass(frozen=True)
class SampleDescriptor:
    dataset_index: int
    event_relative_path: str
    raw_start_index: int
    target_index: int
    target_date: str


def _normalized_date(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _center_crop_last_two(values: np.ndarray, side: int) -> np.ndarray:
    """Match torchvision center-crop geometry on NumPy raw arrays."""

    array = np.asarray(values)
    height, width = array.shape[-2:]
    pad_height = max(0, side - height)
    pad_width = max(0, side - width)
    if pad_height or pad_width:
        padding = [(0, 0)] * (array.ndim - 2) + [
            (pad_height // 2, (pad_height + 1) // 2),
            (pad_width // 2, (pad_width + 1) // 2),
        ]
        array = np.pad(array, padding, mode="constant", constant_values=0)
        height, width = array.shape[-2:]
    top = int(round((height - side) / 2.0))
    left = int(round((width - side) / 2.0))
    return np.asarray(array[..., top : top + side, left : left + side]).copy()


def _read_center_crop(
    dataset: h5py.Dataset,
    selection: tuple[object, ...],
    side: int,
) -> np.ndarray:
    """Read only the model-visible center crop from one HDF5 dataset."""

    height, width = dataset.shape[-2:]
    if height < side or width < side:
        return _center_crop_last_two(
            np.asarray(dataset[selection], dtype=np.float32), side
        )
    top = int(round((height - side) / 2.0))
    left = int(round((width - side) / 2.0))
    spatial_selection = (
        slice(top, top + side),
        slice(left, left + side),
    )
    return np.asarray(
        dataset[selection + spatial_selection], dtype=np.float32
    ).copy()


class ControlledMissingnessDataset:
    """Wrap the pinned upstream dataset and corrupt raw inputs before preprocessing."""

    def __init__(
        self,
        base_dataset: Any,
        scenario_id: str,
        *,
        evaluation_year: int = 2021,
        heldout_authorized: bool = False,
        active_fire_validity_channel: bool = False,
        routing_mask_channel: bool = False,
    ) -> None:
        if scenario_id not in CORRUPTIONS:
            raise ValueError(f"unknown controlled-missingness scenario: {scenario_id}")
        if evaluation_year not in {2021, 2022, 2023}:
            raise ValueError("controlled evaluation year must be 2021, 2022, or 2023")
        if evaluation_year in {2022, 2023} and heldout_authorized is not True:
            raise ValueError("held-out authorization is required for 2022--2023")
        if getattr(base_dataset, "is_train", None) is not False:
            raise ValueError("controlled evaluation requires is_train=false")
        if tuple(getattr(base_dataset, "included_fire_years", ())) != (
            evaluation_year,
        ):
            if evaluation_year == 2021:
                raise ValueError("engineering evaluation is 2021-only")
            raise ValueError("held-out dataset year differs from its authorization")
        if getattr(base_dataset, "load_from_hdf5", None) is not True:
            raise ValueError("controlled evaluation requires HDF5 input")
        if (
            getattr(base_dataset, "n_leading_observations_test_adjustment", None)
            != EFFECTIVE_HISTORY
        ):
            raise ValueError("controlled evaluation requires history adjustment six")
        history = getattr(base_dataset, "n_leading_observations", None)
        if type(history) is not int or history not in {1, 5}:
            raise ValueError("controlled evaluation supports history one or five")
        if getattr(base_dataset, "skip_initial_samples", None) != EFFECTIVE_HISTORY - history:
            raise ValueError("upstream skip count differs from history adjustment")
        if tuple(getattr(base_dataset, "stats_years", ())) != TRAIN_YEARS:
            raise ValueError("controlled evaluation requires train-only statistics years")

        self.base = base_dataset
        self.scenario_id = scenario_id
        self.evaluation_year = evaluation_year
        self.heldout_authorized = heldout_authorized
        self.active_fire_validity_channel = active_fire_validity_channel
        self.routing_mask_channel = routing_mask_channel
        self.data_root = Path(base_dataset.data_dir).resolve()

    def __len__(self) -> int:
        return len(self.base)

    def describe(self, index: int) -> SampleDescriptor:
        year, fire_name, in_fire_index = self.base.find_image_index_from_dataset_index(
            index
        )
        if year != self.evaluation_year:
            raise ValueError("sample descriptor escaped its authorized year boundary")
        hdf5_items = self.base.imgs_per_fire[year][fire_name]
        if len(hdf5_items) != 1:
            raise ValueError("event must resolve to exactly one HDF5 file")
        path = Path(hdf5_items[0]).resolve()
        try:
            relative = path.relative_to(self.data_root).as_posix()
        except ValueError as error:
            raise ValueError("event HDF5 is outside the controlled data root") from error

        raw_start = in_fire_index + self.base.skip_initial_samples
        target_index = raw_start + self.base.n_leading_observations
        with h5py.File(path, "r") as handle:
            dataset = handle["data"]
            dates = dataset.attrs.get("img_dates")
            if dates is None or target_index >= len(dates):
                raise ValueError("target date metadata is incomplete")
            target_date = _normalized_date(dates[target_index])
        return SampleDescriptor(index, relative, raw_start, target_index, target_date)

    def _raw_corruption(
        self, descriptor: SampleDescriptor
    ) -> tuple[CorruptionResult, np.ndarray]:
        path = self.data_root / descriptor.event_relative_path
        history = self.base.n_leading_observations
        crop_side = self.base.crop_side_length
        with h5py.File(path, "r") as handle:
            data = handle["data"]
            raw = _read_center_crop(
                data,
                (
                    slice(descriptor.raw_start_index, descriptor.target_index),
                    slice(None),
                ),
                crop_side,
            )
            target = _read_center_crop(
                data,
                (descriptor.target_index, -1),
                crop_side,
            )
            stale = None
            if self.scenario_id == "M02":
                stale_start = descriptor.raw_start_index - 1
                if stale_start < 0:
                    raise ValueError("M02 predecessor is unavailable")
                stale = _read_center_crop(
                    data,
                    (slice(stale_start, stale_start + history), -1),
                    crop_side,
                )
        corruption = apply_corruption(
            raw,
            self.scenario_id,
            event_relative_path=descriptor.event_relative_path,
            target_date=descriptor.target_date,
            stale_active_fire=stale,
        )
        return corruption, target

    def corruption_for_index(self, index: int) -> CorruptionResult:
        descriptor = self.describe(index)
        corruption, _ = self._raw_corruption(descriptor)
        return corruption

    def __getitem__(self, index: int):
        descriptor = self.describe(index)
        corruption, target = self._raw_corruption(descriptor)
        x, y = self.base.preprocess_and_augment(corruption.values, target)

        if self.active_fire_validity_channel:
            from .prototype import append_fire_validity_channel

            validity = 0.0 if self.scenario_id == "M01" else 1.0
            x = append_fire_validity_channel(x, validity)

        routing_mask_tensor = None
        if self.routing_mask_channel:
            import torch

            mask = corruption.spatial_mask
            if mask is None:
                mask = np.zeros(x.shape[-2:], dtype=bool)
            routing_mask_tensor = torch.as_tensor(
                mask, dtype=x.dtype, device=x.device
            )
            routing_mask_tensor = routing_mask_tensor[None, None].expand(
                x.shape[0], 1, -1, -1
            )
            if self.base.features_to_keep is None:
                x = torch.cat((x, routing_mask_tensor), dim=1)

        if self.base.remove_duplicate_features and self.base.n_leading_observations > 1:
            x = self.base.flatten_and_remove_duplicate_features_(x)
        elif self.base.features_to_keep is not None:
            if len(x.shape) != 4:
                raise ValueError("feature selection requires a four-dimensional tensor")
            x = x[:, self.base.features_to_keep, ...]

        if routing_mask_tensor is not None and self.base.features_to_keep is not None:
            import torch

            x = torch.cat((x, routing_mask_tensor), dim=1)

        if self.base.return_doy:
            raise ValueError("controlled evaluation does not permit day-of-year output")
        return x, y


def engineering_dataset_kwargs(
    spec: ExperimentSpec, data_root: Path
) -> dict[str, object]:
    """Return the literal upstream constructor arguments for 2021 engineering."""

    return {
        "data_dir": str(Path(data_root).resolve()),
        "included_fire_years": list(ENGINEERING_YEARS),
        "n_leading_observations": spec.n_leading_observations,
        "n_leading_observations_test_adjustment": EFFECTIVE_HISTORY,
        "crop_side_length": 128,
        "load_from_hdf5": True,
        "is_train": False,
        "remove_duplicate_features": spec.remove_duplicate_features,
        "features_to_keep": (
            None if spec.features_to_keep is None else list(spec.features_to_keep)
        ),
        "return_doy": False,
        "stats_years": list(TRAIN_YEARS),
        "is_pad": False,
    }


def controlled_dataset_kwargs(
    spec: ExperimentSpec,
    data_root: Path,
    *,
    evaluation_year: int,
    heldout_authorized: bool,
) -> dict[str, object]:
    """Return dataset arguments after enforcing the year authorization."""

    if evaluation_year == 2021:
        return engineering_dataset_kwargs(spec, data_root)
    if evaluation_year not in {2022, 2023} or heldout_authorized is not True:
        raise ValueError("held-out authorization is required for 2022--2023")
    result = engineering_dataset_kwargs(spec, data_root)
    result["included_fire_years"] = [evaluation_year]
    return result


def build_engineering_dataset(
    *,
    upstream_root: Path,
    data_root: Path,
    stats_path: Path,
    experiment_id: str,
    scenario_id: str,
) -> ControlledMissingnessDataset:
    """Build one pinned 2021-only upstream dataset without validation augmentation."""

    return build_controlled_dataset(
        upstream_root=upstream_root,
        data_root=data_root,
        stats_path=stats_path,
        experiment_id=experiment_id,
        scenario_id=scenario_id,
        evaluation_year=2021,
        heldout_authorized=False,
    )


def build_controlled_dataset(
    *,
    upstream_root: Path,
    data_root: Path,
    stats_path: Path,
    experiment_id: str,
    scenario_id: str,
    evaluation_year: int,
    heldout_authorized: bool,
    active_fire_validity_channel: bool = False,
    routing_mask_channel: bool = False,
) -> ControlledMissingnessDataset:
    """Build a pinned dataset after an explicit engineering/formal year gate."""

    upstream = Path(upstream_root).resolve()
    if not (upstream / "src" / "dataloader" / "FireSpreadDataset.py").is_file():
        raise ValueError("pinned upstream FireSpreadDataset.py is missing")
    if scenario_id not in CORRUPTIONS:
        raise ValueError(f"unknown controlled-missingness scenario: {scenario_id}")
    spec = experiment_spec(experiment_id)
    if active_fire_validity_channel and experiment_id != "C00":
        raise ValueError("active-fire validity prototype requires C00")
    stats = load_training_stats(stats_path)
    _install_runtime_contract(upstream, experiment_id, stats)
    dataset_module = importlib.import_module("dataloader.FireSpreadDataset")
    base = dataset_module.FireSpreadDataset(
        **controlled_dataset_kwargs(
            spec,
            data_root,
            evaluation_year=evaluation_year,
            heldout_authorized=heldout_authorized,
        )
    )
    return ControlledMissingnessDataset(
        base,
        scenario_id,
        evaluation_year=evaluation_year,
        heldout_authorized=heldout_authorized,
        active_fire_validity_channel=active_fire_validity_channel,
        routing_mask_channel=routing_mask_channel,
    )
