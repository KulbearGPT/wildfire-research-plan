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


def _complete_inventory(
    *, positive_overrides: dict[int, int] | None = None
) -> list[EventInventory]:
    overrides = positive_overrides or {}
    return [
        _event(year, f"fire_{year}", positives=overrides.get(year, 1))
        for year in range(2016, 2024)
    ]


def test_gate_blocks_present_year_and_split_with_zero_positive_targets() -> None:
    inventory = _complete_inventory(positive_overrides={2022: 0, 2023: 0})
    split = build_forward_split(inventory)

    errors = target_integrity_errors(inventory, split)

    assert errors == (
        "split test has zero positive target pixels",
        "year 2022 has zero positive target pixels",
        "year 2023 has zero positive target pixels",
    )


def test_gate_allows_zero_positive_events_when_year_and_split_are_positive() -> None:
    inventory = [
        *_complete_inventory(),
        _event(2017, "zero_event", positives=0),
    ]

    assert target_integrity_errors(inventory, build_forward_split(inventory)) == ()


def test_gate_reports_every_missing_year_and_split_for_empty_inventory() -> None:
    inventory: list[EventInventory] = []

    assert target_integrity_errors(inventory, build_forward_split(inventory)) == (
        "benchmark year 2016 is missing from inventory",
        "benchmark year 2017 is missing from inventory",
        "benchmark year 2018 is missing from inventory",
        "benchmark year 2019 is missing from inventory",
        "benchmark year 2020 is missing from inventory",
        "benchmark year 2021 is missing from inventory",
        "benchmark year 2022 is missing from inventory",
        "benchmark year 2023 is missing from inventory",
        "canonical split test is missing from split manifest",
        "canonical split train is missing from split manifest",
        "canonical split validation is missing from split manifest",
    )


def test_gate_reports_missing_benchmark_year_and_split() -> None:
    inventory = [_event(year, f"fire_{year}", positives=1) for year in range(2016, 2022)]

    assert target_integrity_errors(inventory, build_forward_split(inventory)) == (
        "benchmark year 2022 is missing from inventory",
        "benchmark year 2023 is missing from inventory",
        "canonical split test is missing from split manifest",
    )


def test_gate_rejects_unknown_split_label() -> None:
    inventory = _complete_inventory()
    split = build_forward_split(inventory)
    split.loc[0, "split"] = "holdout"

    with pytest.raises(
        ValueError,
        match="split manifest contains unknown split labels: holdout",
    ):
        target_integrity_errors(inventory, split)


def test_gate_rejects_missing_canonical_split() -> None:
    inventory = _complete_inventory()
    split = build_forward_split(inventory)
    split.loc[split["split"] == "test", "split"] = "train"

    with pytest.raises(
        ValueError,
        match="split manifest is missing canonical splits: test",
    ):
        target_integrity_errors(inventory, split)


def test_gate_rejects_relabelled_frozen_year_mapping() -> None:
    inventory = _complete_inventory()
    split = build_forward_split(inventory)
    split.loc[split["year"] == 2020, "split"] = "validation"
    split.loc[split["year"] == 2021, "split"] = "train"

    with pytest.raises(
        ValueError,
        match="split manifest maps year 2020 to validation; expected train",
    ):
        target_integrity_errors(inventory, split)


def test_gate_rejects_split_manifest_event_id_mismatch() -> None:
    inventory = _complete_inventory()
    split = build_forward_split(inventory)
    split.loc[0, "event_id"] = "2021:different"

    with pytest.raises(ValueError, match="event IDs must exactly match"):
        target_integrity_errors(inventory, split)
