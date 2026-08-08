"""Frozen forward split manifest for confirmatory wildfire evaluation."""

from collections.abc import Sequence

import pandas as pd

from wildfire_phase0.schema import EventInventory


_COLUMNS = ["event_id", "year", "fire_name", "path", "split"]


def build_forward_split(events: Sequence[EventInventory]) -> pd.DataFrame:
    """Build the fixed temporal event-disjoint split manifest."""
    records: list[dict[str, object]] = []
    for event in events:
        if not 2016 <= event.year <= 2023:
            raise ValueError("event year must be within the 2016-2023 protocol")
        split = "train" if event.year <= 2020 else "validation" if event.year == 2021 else "test"
        records.append({
            "event_id": f"{event.year}:{event.fire_name}",
            "year": event.year,
            "fire_name": event.fire_name,
            "path": event.path.as_posix(),
            "split": split,
        })

    frame = pd.DataFrame(records, columns=_COLUMNS)
    frame = frame.sort_values(["year", "fire_name"], kind="stable").reset_index(drop=True)
    if not frame["event_id"].is_unique:
        raise ValueError("event_id values must be unique")
    return frame
