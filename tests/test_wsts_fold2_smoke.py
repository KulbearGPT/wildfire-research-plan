import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "reproductions"
    / "wsts_res18_unet_t1"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

from smoke_fold2 import assert_fold_mapping, inspect_batch, main  # noqa: E402
from control import verify_inventory  # noqa: E402


def _valid_batch() -> tuple[np.ndarray, np.ndarray]:
    inputs = np.zeros((2, 1, 40, 128, 128), dtype=np.float32)
    inputs[:, :, -1, 10:20, 10:20] = 1.0
    targets = np.zeros((2, 128, 128), dtype=np.int64)
    targets[:, 30:40, 30:40] = 1
    return inputs, targets


def test_inspect_batch_accepts_finite_t1_all_feature_crop_and_records_boundaries() -> None:
    report = inspect_batch(_valid_batch())

    assert report == {
        "loader_input_shape": [2, 1, 40, 128, 128],
        "loader_target_shape": [2, 128, 128],
        "model_input_shape": [2, 40, 128, 128],
        "model_target_shape": [2, 1, 128, 128],
        "temporal_steps": 1,
        "model_channels": 40,
        "crop_side_length": 128,
        "inputs_finite": True,
        "targets_binary": True,
        "next_day_target_distinct": True,
    }


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_inspect_batch_rejects_non_finite_inputs(bad_value: float) -> None:
    inputs, targets = _valid_batch()
    inputs[0, 0, 0, 0, 0] = bad_value

    with pytest.raises(ValueError, match="finite"):
        inspect_batch((inputs, targets))


def test_inspect_batch_rejects_non_binary_targets() -> None:
    inputs, targets = _valid_batch()
    targets[0, 0, 0] = 2

    with pytest.raises(ValueError, match="binary"):
        inspect_batch((inputs, targets))


@pytest.mark.parametrize(
    ("input_shape", "target_shape", "message"),
    [
        ((2, 2, 40, 128, 128), (2, 128, 128), "temporal"),
        ((2, 1, 39, 128, 128), (2, 128, 128), "channel"),
        ((2, 1, 40, 127, 128), (2, 127, 128), "spatial"),
        ((2, 1, 40, 128, 128), (2, 128, 127), "spatial"),
        ((2, 40, 128, 128), (2, 128, 128), "5D"),
        ((2, 1, 40, 128, 128), (2, 1, 128, 128), "3D"),
    ],
)
def test_inspect_batch_rejects_wrong_loader_dimensions(
    input_shape: tuple[int, ...], target_shape: tuple[int, ...], message: str
) -> None:
    inputs = np.zeros(input_shape, dtype=np.float32)
    targets = np.zeros(target_shape, dtype=np.int64)

    with pytest.raises(ValueError, match=message):
        inspect_batch((inputs, targets))


def test_inspect_batch_rejects_same_day_target_masquerading_as_next_day() -> None:
    inputs, targets = _valid_batch()
    targets[:] = inputs[:, 0, -1]

    with pytest.raises(ValueError, match="next-day|distinct"):
        inspect_batch((inputs, targets))


def test_inspect_batch_does_not_require_torch_to_numpy_conversion() -> None:
    class NoNumpyTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, value: np.ndarray) -> "NoNumpyTensor":
            return torch.Tensor._make_subclass(cls, torch.as_tensor(value), False)

        def numpy(self) -> np.ndarray:
            raise RuntimeError("pinned torch reports NumPy is not available")

    inputs, targets = _valid_batch()

    report = inspect_batch((NoNumpyTensor(inputs), NoNumpyTensor(targets)))

    assert report["model_input_shape"] == [2, 40, 128, 128]
    assert report["next_day_target_distinct"] is True


class _Fold2DataModule:
    @staticmethod
    def split_fires(data_fold_id: int, additional_data: bool) -> tuple[list[int], ...]:
        assert data_fold_id == 2
        assert additional_data is False
        return [2018, 2020], [2019], [2021]


def test_assert_fold_mapping_accepts_exact_official_fold_2_years() -> None:
    assert assert_fold_mapping(_Fold2DataModule()) == {
        "train": [2018, 2020],
        "validation": [2019],
        "test": [2021],
    }


def test_assert_fold_mapping_rejects_any_other_year_assignment() -> None:
    class WrongFoldDataModule:
        @staticmethod
        def split_fires(data_fold_id: int, additional_data: bool) -> tuple[list[int], ...]:
            return [2018, 2019], [2020], [2021]

    with pytest.raises(ValueError, match="fold 2"):
        assert_fold_mapping(WrongFoldDataModule())


def test_inventory_ignores_unselected_years_but_rejects_selecting_them(
    tmp_path: Path,
) -> None:
    frozen_counts = {2018: 176, 2019: 74, 2020: 201, 2021: 156}
    for year, count in frozen_counts.items():
        year_dir = tmp_path / str(year)
        year_dir.mkdir()
        for index in range(count):
            (year_dir / f"fire_{index}.hdf5").touch()
    extra_year = tmp_path / "2022"
    extra_year.mkdir()
    (extra_year / "not-selected.hdf5").touch()

    assert verify_inventory(
        tmp_path, selected_years=[2018, 2020, 2019, 2021]
    ) == {
        "2018": 176,
        "2019": 74,
        "2020": 201,
        "2021": 156,
        "total": 607,
    }
    with pytest.raises(ValueError, match="selected years"):
        verify_inventory(
            tmp_path, selected_years=[2018, 2020, 2019, 2021, 2022]
        )


def test_smoke_cli_preserves_full_traceback_on_gate_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    return_code = main(
        [
            "--upstream-root",
            str(tmp_path / "missing-upstream"),
            "--data-root",
            str(tmp_path / "missing-data"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    captured = capsys.readouterr()
    assert return_code == 2
    assert "Traceback (most recent call last)" in captured.err
    assert "cannot determine upstream commit" in captured.err
