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
    target_days: int = 0
    zero_target_days: int = 0
    positive_target_pixels: int = 0
    active_fire_min_positive: float | None = None
    active_fire_max_positive: float | None = None


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
    target_days = int(row["target_days"])
    zero_target_days = int(row["zero_target_days"])
    positive_target_pixels = int(row["positive_target_pixels"])
    active_fire_min_positive = (
        None
        if row["active_fire_min_positive"] is None
        else float(row["active_fire_min_positive"])
    )
    active_fire_max_positive = (
        None
        if row["active_fire_max_positive"] is None
        else float(row["active_fire_max_positive"])
    )

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
    if target_days < 0 or zero_target_days < 0 or positive_target_pixels < 0:
        raise ValueError("target counts must be nonnegative")
    if zero_target_days > target_days:
        raise ValueError("zero_target_days must not exceed target_days")
    extrema_present = (
        active_fire_min_positive is not None,
        active_fire_max_positive is not None,
    )
    if extrema_present[0] != extrema_present[1]:
        raise ValueError("positive active-fire extrema must both be present or both be None")
    if active_fire_min_positive is not None and active_fire_max_positive is not None:
        if not (
            active_fire_min_positive.is_integer()
            and active_fire_max_positive.is_integer()
            and 1
            <= active_fire_min_positive
            <= active_fire_max_positive
            <= 23
        ):
            raise ValueError(
                "positive active-fire extrema must be ordered integer hours within 1-23"
            )
    elif positive_target_pixels > 0:
        raise ValueError(
            "positive active-fire extrema are required when positive_target_pixels is positive"
        )

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
        target_days=target_days,
        zero_target_days=zero_target_days,
        positive_target_pixels=positive_target_pixels,
        active_fire_min_positive=active_fire_min_positive,
        active_fire_max_positive=active_fire_max_positive,
    )
