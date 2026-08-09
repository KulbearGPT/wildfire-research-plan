from pathlib import Path

import pytest

from wildfire_phase0.schema import EventInventory, validate_inventory_row


def test_validate_inventory_row_rejects_non_monotonic_dates() -> None:
    row = {
        "year": 2021,
        "fire_name": "demo_fire",
        "path": "2021/demo_fire.hdf5",
        "n_days": 3,
        "n_channels": 23,
        "height": 8,
        "width": 8,
        "dates": ["2021-08-03", "2021-08-01", "2021-08-02"],
        "nan_fraction": 0.1,
        "target_days": 2,
        "zero_target_days": 1,
        "positive_target_pixels": 1,
        "active_fire_min_positive": 9,
        "active_fire_max_positive": 9,
    }
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_inventory_row(row)


def test_event_inventory_keeps_relative_path() -> None:
    item = EventInventory(2021, "demo_fire", Path("2021/demo_fire.hdf5"), 3, 23, 8, 8,
                          ("2021-08-01", "2021-08-02", "2021-08-03"), 0.1,
                          2, 1, 1, 9, 9)
    assert item.path == Path("2021/demo_fire.hdf5")


def _valid_row() -> dict[str, object]:
    return {
        "year": 2021,
        "fire_name": "demo_fire",
        "path": "2021/demo_fire.hdf5",
        "n_days": 3,
        "n_channels": 23,
        "height": 8,
        "width": 8,
        "dates": ["2021-08-01", "2021-08-02", "2021-08-03"],
        "nan_fraction": 0.1,
        "target_days": 2,
        "zero_target_days": 1,
        "positive_target_pixels": 1,
        "active_fire_min_positive": 9,
        "active_fire_max_positive": 9,
    }


@pytest.mark.parametrize("field", ["target_days", "zero_target_days", "positive_target_pixels"])
def test_validate_inventory_row_rejects_negative_target_counts(field: str) -> None:
    row = _valid_row()
    row[field] = -1

    with pytest.raises(ValueError, match="target counts must be nonnegative"):
        validate_inventory_row(row)


def test_validate_inventory_row_rejects_more_zero_days_than_target_days() -> None:
    row = _valid_row()
    row["zero_target_days"] = 3

    with pytest.raises(ValueError, match="zero_target_days must not exceed target_days"):
        validate_inventory_row(row)


@pytest.mark.parametrize("field", ["active_fire_min_positive", "active_fire_max_positive"])
def test_validate_inventory_row_rejects_positive_extrema_without_positive_pixels(
    field: str,
) -> None:
    row = _valid_row()
    row["positive_target_pixels"] = 0
    row["active_fire_min_positive"] = None
    row["active_fire_max_positive"] = None
    row[field] = 9

    with pytest.raises(ValueError, match="positive active-fire extrema must be None"):
        validate_inventory_row(row)
