"""Target-evidence integrity checks for the Phase 0 data gate."""

from collections import defaultdict
from collections.abc import Sequence

import pandas as pd

from wildfire_phase0.schema import EventInventory


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
    for row in split_manifest.to_dict("records"):
        event = inventory_by_id[str(row["event_id"])]
        if any(
            column in row and row[column] != getter(event)
            for column, getter in expected_metadata.items()
        ):
            raise ValueError("split manifest metadata must match inventory events")

    year_totals: dict[int, int] = defaultdict(int)
    for event in inventory:
        year_totals[event.year] += event.positive_target_pixels

    split_totals: dict[str, int] = defaultdict(int)
    for row in split_manifest.to_dict("records"):
        split_name = str(row["split"])
        split_totals[split_name] += inventory_by_id[
            str(row["event_id"])
        ].positive_target_pixels

    errors = [
        f"year {year} has zero positive target pixels"
        for year, positives in year_totals.items()
        if positives == 0
    ]
    errors.extend(
        f"split {split_name} has zero positive target pixels"
        for split_name, positives in split_totals.items()
        if positives == 0
    )
    return tuple(sorted(errors))
