from pathlib import Path

import pytest

from wildfire_phase0.schema import EventInventory
from wildfire_phase0.splits import build_forward_split
from wildfire_phase0.target_gate import target_integrity_errors


def _event(year: int, name: str, *, positives: int) -> EventInventory:
    positive_extreme = 9.0 if positives else None
    return EventInventory(
        year=year,
        fire_name=name,
        path=Path(f"{year}/{name}.hdf5"),
        n_days=2,
        n_channels=23,
        height=8,
        width=8,
        dates=(f"{year}-08-01", f"{year}-08-02"),
        nan_fraction=0.0,
        target_days=1,
        zero_target_days=int(positives == 0),
        positive_target_pixels=positives,
        active_fire_min_positive=positive_extreme,
        active_fire_max_positive=positive_extreme,
    )


def test_gate_blocks_present_year_and_split_with_zero_positive_targets() -> None:
    inventory = [
        _event(2016, "train_positive", positives=1),
        _event(2021, "validation_positive", positives=1),
        _event(2022, "test_zero", positives=0),
    ]
    split = build_forward_split(inventory)

    errors = target_integrity_errors(inventory, split)

    assert errors == (
        "split test has zero positive target pixels",
        "year 2022 has zero positive target pixels",
    )


def test_gate_allows_zero_positive_events_when_year_and_split_are_positive() -> None:
    inventory = [
        _event(2017, "zero_event", positives=0),
        _event(2017, "positive_event", positives=2),
    ]

    assert target_integrity_errors(inventory, build_forward_split(inventory)) == ()


def test_gate_returns_no_target_error_for_empty_inventory() -> None:
    inventory: list[EventInventory] = []

    assert target_integrity_errors(inventory, build_forward_split(inventory)) == ()


def test_gate_rejects_split_manifest_event_id_mismatch() -> None:
    inventory = [_event(2021, "validation_positive", positives=1)]
    split = build_forward_split(inventory)
    split.loc[0, "event_id"] = "2021:different"

    with pytest.raises(ValueError, match="event IDs must exactly match"):
        target_integrity_errors(inventory, split)
