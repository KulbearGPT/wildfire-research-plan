"""Target-evidence integrity checks for the Phase 0 data gate."""

from collections import defaultdict
from collections.abc import Sequence

import pandas as pd

from wildfire_phase0.schema import EventInventory


_BENCHMARK_YEARS = frozenset(range(2016, 2024))
_CANONICAL_SPLITS = frozenset({"train", "validation", "test"})


def _expected_split(year: int) -> str:
    if year <= 2020:
        return "train"
    if year == 2021:
        return "validation"
    return "test"


def target_integrity_errors(
    inventory: Sequence[EventInventory],
    split_manifest: pd.DataFrame,
) -> tuple[str, ...]:
    """Return deterministic blockers for missing positive target evidence."""
    inventory_by_id = {
        f"{event.year}:{event.fire_name}": event for event in inventory
    }
    if len(inventory_by_id) != len(inventory):
        raise ValueError("inventory event IDs must be unique")
    if not {"event_id", "split"}.issubset(split_manifest.columns):
        raise ValueError("split manifest must contain event_id and split columns")

    manifest_ids = [str(value) for value in split_manifest["event_id"]]
    if len(set(manifest_ids)) != len(manifest_ids) or set(manifest_ids) != set(
        inventory_by_id
    ):
        raise ValueError("split manifest event IDs must exactly match inventory event IDs")

    expected_metadata = {
        "year": lambda event: event.year,
        "fire_name": lambda event: event.fire_name,
        "path": lambda event: event.path.as_posix(),
    }
    manifest_records = split_manifest.to_dict("records")
    for row in manifest_records:
        event = inventory_by_id[str(row["event_id"])]
        if any(
            column in row and row[column] != getter(event)
            for column, getter in expected_metadata.items()
        ):
            raise ValueError("split manifest metadata must match inventory events")

    split_names = {str(row["split"]) for row in manifest_records}
    unknown_splits = sorted(split_names - _CANONICAL_SPLITS)
    if unknown_splits:
        raise ValueError(
            "split manifest contains unknown split labels: "
            + ", ".join(unknown_splits)
        )

    observed_years = {event.year for event in inventory}
    unexpected_years = sorted(observed_years - _BENCHMARK_YEARS)
    if unexpected_years:
        raise ValueError(
            "inventory contains years outside benchmark protocol: "
            + ", ".join(str(year) for year in unexpected_years)
        )
    missing_splits = sorted(_CANONICAL_SPLITS - split_names)
    if observed_years == _BENCHMARK_YEARS and missing_splits:
        raise ValueError(
            "split manifest is missing canonical splits: " + ", ".join(missing_splits)
        )
    for row in sorted(manifest_records, key=lambda item: str(item["event_id"])):
        event = inventory_by_id[str(row["event_id"])]
        actual_split = str(row["split"])
        expected_split = _expected_split(event.year)
        if actual_split != expected_split:
            raise ValueError(
                f"split manifest maps year {event.year} to {actual_split}; "
                f"expected {expected_split}"
            )

    year_totals: dict[int, int] = defaultdict(int)
    for event in inventory:
        year_totals[event.year] += event.positive_target_pixels

    split_totals: dict[str, int] = defaultdict(int)
    for row in manifest_records:
        split_name = str(row["split"])
        split_totals[split_name] += inventory_by_id[
            str(row["event_id"])
        ].positive_target_pixels

    errors = [
        f"benchmark year {year} is missing from inventory"
        for year in sorted(_BENCHMARK_YEARS - observed_years)
    ]
    errors.extend(
        f"canonical split {split_name} is missing from split manifest"
        for split_name in missing_splits
    )
    errors.extend(
        f"year {year} has zero positive target pixels"
        for year, positives in year_totals.items()
        if positives == 0
    )
    errors.extend(
        f"split {split_name} has zero positive target pixels"
        for split_name, positives in split_totals.items()
        if positives == 0
    )
    return tuple(sorted(errors))
