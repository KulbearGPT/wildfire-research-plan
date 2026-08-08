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
    }
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_inventory_row(row)


def test_event_inventory_keeps_relative_path() -> None:
    item = EventInventory(2021, "demo_fire", Path("2021/demo_fire.hdf5"), 3, 23, 8, 8,
                          ("2021-08-01", "2021-08-02", "2021-08-03"), 0.1)
    assert item.path == Path("2021/demo_fire.hdf5")
