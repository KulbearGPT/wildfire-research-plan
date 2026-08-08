"""Validation schema for derived wildfire event inventories."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class EventInventory:
    year: int
    fire_name: str
    path: Path
    n_days: int
    n_channels: int
    height: int
    width: int
    dates: tuple[str, ...]
    nan_fraction: float


def validate_inventory_row(row: Mapping[str, object]) -> EventInventory:
    """Validate one derived inventory row and return its typed representation."""
    year = int(row["year"])
    fire_name = str(row["fire_name"])
    path = Path(str(row["path"]))
    n_days = int(row["n_days"])
    n_channels = int(row["n_channels"])
    height = int(row["height"])
    width = int(row["width"])
    dates = tuple(str(value) for value in row["dates"])
    nan_fraction = float(row["nan_fraction"])

    if path.is_absolute():
        raise ValueError("path must be relative")
    if n_channels != 23:
        raise ValueError("n_channels must equal 23")
    if n_days <= 0 or height <= 0 or width <= 0:
        raise ValueError("n_days, height, and width must be positive")
    if len(dates) != n_days:
        raise ValueError("dates count must equal n_days")
    if not 0 <= nan_fraction <= 1:
        raise ValueError("nan_fraction must be between 0 and 1")

    parsed_dates = tuple(date.fromisoformat(value) for value in dates)
    if any(left >= right for left, right in zip(parsed_dates, parsed_dates[1:])):
        raise ValueError("dates must be strictly increasing")

    return EventInventory(
        year=year,
        fire_name=fire_name,
        path=path,
        n_days=n_days,
        n_channels=n_channels,
        height=height,
        width=width,
        dates=dates,
        nan_fraction=nan_fraction,
    )
