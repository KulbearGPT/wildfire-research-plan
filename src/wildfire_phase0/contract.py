"""Machine-readable WSTS+ field contract and deterministic audit."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


_REQUIRED_COLUMNS = (
    "modality",
    "observation_time",
    "availability_time",
    "qa",
    "coverage",
    "target_validity",
)
_FIELD_COLUMNS = _REQUIRED_COLUMNS[1:]
_MISSING_VALUES = {
    "unavailable",
    "filtered_not_retained",
    "nan_unknown_cause",
    "positive_pixel_hour_only",
    "daily_aggregate_only",
    "daily_nominal_date_only",
    "composite_nominal_date_only",
    "annual_nominal_date_only",
    "historical_version_not_retained",
}
_TRACEABLE_VALUES = {"traceable", "static"}
_AVAILABILITY_BLOCKERS = {"unavailable", "historical_version_not_retained"}


@dataclass(frozen=True)
class ContractDecision:
    status: str
    missing_required: tuple[str, ...]
    operational_blockers: tuple[str, ...]
    notes: tuple[str, ...]


def _validate_columns(frame: pd.DataFrame) -> None:
    for column in _REQUIRED_COLUMNS:
        if column not in frame.columns:
            raise ValueError(f"missing required column: {column}")


def audit_contract(frame: pd.DataFrame) -> ContractDecision:
    """Classify whether the field contract supports natural missingness."""
    _validate_columns(frame)
    if frame.empty:
        return ContractDecision(
            status="blocked",
            missing_required=(),
            operational_blockers=("active_fire",),
            notes=("Contract registry is empty.",),
        )
    if "active_fire" not in set(frame["modality"]):
        return ContractDecision(
            status="blocked",
            missing_required=(),
            operational_blockers=("active_fire",),
            notes=("Required active_fire modality is absent.",),
        )

    missing_required = tuple(
        column
        for column in _FIELD_COLUMNS
        if frame[column].isin(_MISSING_VALUES).any()
    )
    operational_blockers: list[str] = []
    if frame["availability_time"].isin(_AVAILABILITY_BLOCKERS).any():
        operational_blockers.append("availability_time")
    if set(frame["modality"]) == {
        "active_fire", "viirs_reflectance", "ndvi_evi", "gridmet",
        "gfs_forecast", "terrain", "land_cover",
    }:
        operational_blockers.append("event_roi_provenance")
    if frame["target_validity"].eq("unavailable").any():
        operational_blockers.append("target_validity")

    natural_values = frame.loc[:, _FIELD_COLUMNS].stack()
    is_natural = natural_values[natural_values.ne("not_applicable")].isin(_TRACEABLE_VALUES).all()
    if is_natural:
        return ContractDecision(
            status="continue_natural",
            missing_required=missing_required,
            operational_blockers=tuple(operational_blockers),
            notes=("All applicable fields are traceable or static.",),
        )
    return ContractDecision(
        status="continue_controlled",
        missing_required=missing_required,
        operational_blockers=tuple(operational_blockers),
        notes=("Natural missingness is not recoverable from this contract.",),
    )


def load_contract(path: Path) -> pd.DataFrame:
    """Read, validate, and sort a field-contract registry without changing it."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    audit_contract(frame)
    return frame.sort_values("modality", kind="stable").reset_index(drop=True)
