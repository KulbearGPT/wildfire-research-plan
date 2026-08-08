from pathlib import Path

import pytest

from wildfire_phase0.schema import EventInventory
from wildfire_phase0.splits import build_forward_split


def _event(year: int, name: str) -> EventInventory:
    return EventInventory(
        year,
        name,
        Path(f"{year}/{name}.hdf5"),
        2,
        23,
        8,
        8,
        (f"{year}-08-01", f"{year}-08-02"),
        0.0,
    )


def test_forward_split_is_event_disjoint_and_temporal() -> None:
    frame = build_forward_split([
        _event(2020, "train_fire"),
        _event(2021, "val_fire"),
        _event(2022, "test_fire"),
    ])

    assert frame.set_index("fire_name")["split"].to_dict() == {
        "train_fire": "train",
        "val_fire": "validation",
        "test_fire": "test",
    }
    assert frame["event_id"].is_unique


@pytest.mark.parametrize(
    ("year", "expected_split"),
    [
        (2016, "train"),
        (2020, "train"),
        (2021, "validation"),
        (2022, "test"),
        (2023, "test"),
    ],
)
def test_forward_split_maps_each_boundary_year(year: int, expected_split: str) -> None:
    frame = build_forward_split([_event(year, "boundary_fire")])

    assert frame.loc[0, "split"] == expected_split


@pytest.mark.parametrize("year", [2015, 2024])
def test_forward_split_rejects_years_outside_protocol(year: int) -> None:
    with pytest.raises(ValueError, match="2016"):
        build_forward_split([_event(year, "outside_fire")])


def test_forward_split_rejects_duplicate_event_identifiers() -> None:
    with pytest.raises(ValueError, match="unique"):
        build_forward_split([_event(2020, "same_fire"), _event(2020, "same_fire")])


def test_forward_split_sorts_events_and_serializes_paths() -> None:
    frame = build_forward_split([
        _event(2021, "zulu_fire"),
        _event(2020, "bravo_fire"),
        _event(2020, "alpha_fire"),
    ])

    assert frame[["year", "fire_name"]].values.tolist() == [
        [2020, "alpha_fire"],
        [2020, "bravo_fire"],
        [2021, "zulu_fire"],
    ]
    assert frame.loc[0, "path"] == "2020/alpha_fire.hdf5"


def test_forward_split_has_required_columns_for_empty_input() -> None:
    frame = build_forward_split([])

    assert frame.columns.tolist() == ["event_id", "year", "fire_name", "path", "split"]
