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
_TRACEABLE_VALUES = {"traceable", "static"}
_AVAILABILITY_BLOCKERS = {"unavailable", "historical_version_not_retained"}
_APPLICABILITY_EXPECTATIONS = {
    "active_fire": ("traceable", "traceable", "traceable", "traceable", "traceable"),
    "viirs_reflectance": (
        "traceable", "traceable", "traceable", "traceable", "not_applicable",
    ),
    "ndvi_evi": ("traceable", "traceable", "traceable", "traceable", "not_applicable"),
    "gridmet": ("traceable", "traceable", "not_applicable", "traceable", "not_applicable"),
    "gfs_forecast": (
        "traceable", "traceable", "not_applicable", "traceable", "not_applicable",
    ),
    "terrain": ("static", "static", "not_applicable", "traceable", "not_applicable"),
    "land_cover": (
        "traceable", "traceable", "not_applicable", "traceable", "not_applicable",
    ),
}
_PUBLIC_COLUMNS = (
    *_REQUIRED_COLUMNS,
    "source_url",
    "audit_note",
)
_PUBLIC_IDENTITY_COLUMNS = (*_REQUIRED_COLUMNS, "source_url")
_PUBLIC_CONTRACT_RECORDS = {
    (
        "active_fire",
        "positive_pixel_hour_only",
        "unavailable",
        "filtered_not_retained",
        "unavailable",
        "unavailable",
        "https://github.com/SebastianGer/WildfireSpreadTS",
        "Only positive pixels retain within-day detection hour; "
        "no-detection and no-valid-observation are conflated",
    ),
    (
        "viirs_reflectance",
        "daily_aggregate_only",
        "unavailable",
        "unavailable",
        "nan_unknown_cause",
        "not_applicable",
        "https://proceedings.neurips.cc/paper_files/paper/2023/file/"
        "ebd545176bdaa9cd5d45954947bd74b7-Paper-Datasets_and_Benchmarks.pdf",
        "Daily median discards per-acquisition time and original QA",
    ),
    (
        "ndvi_evi",
        "composite_nominal_date_only",
        "unavailable",
        "unavailable",
        "nan_unknown_cause",
        "not_applicable",
        "https://developers.google.com/earth-engine/datasets/catalog/"
        "NASA_VIIRS_002_VNP13A1",
        "Composite product requires source-period lineage",
    ),
    (
        "gridmet",
        "daily_nominal_date_only",
        "historical_version_not_retained",
        "not_applicable",
        "nan_unknown_cause",
        "not_applicable",
        "https://developers.google.com/earth-engine/datasets/catalog/"
        "IDAHO_EPSCOR_GRIDMET",
        "Retrospective file does not retain issue-time product version",
    ),
    (
        "gfs_forecast",
        "daily_aggregate_only",
        "unavailable",
        "not_applicable",
        "nan_unknown_cause",
        "not_applicable",
        "https://github.com/SebastianGer/WildfireSpreadTSCreateDataset",
        "Wind direction and product timing require audit",
    ),
    (
        "terrain",
        "static",
        "static",
        "not_applicable",
        "traceable",
        "not_applicable",
        "https://github.com/SebastianGer/WildfireSpreadTSCreateDataset",
        "Static covariate; record version and resampling",
    ),
    (
        "land_cover",
        "annual_nominal_date_only",
        "unavailable",
        "not_applicable",
        "traceable",
        "not_applicable",
        "https://developers.google.com/earth-engine/datasets/catalog/"
        "MODIS_061_MCD12Q1",
        "Annual product may include post-event information",
    ),
}
_PUBLIC_CONTRACT_IDENTITIES = {record[:-1] for record in _PUBLIC_CONTRACT_RECORDS}


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


def _is_public_registry(frame: pd.DataFrame) -> bool:
    if set(frame.columns) != set(_PUBLIC_COLUMNS) or len(frame) != len(_PUBLIC_CONTRACT_RECORDS):
        return False
    identities = set(frame.loc[:, _PUBLIC_IDENTITY_COLUMNS].itertuples(index=False, name=None))
    return identities == _PUBLIC_CONTRACT_IDENTITIES


def _row_mismatches(row: pd.Series) -> tuple[str, ...]:
    expected = _APPLICABILITY_EXPECTATIONS.get(row["modality"])
    if expected is not None:
        return tuple(
            field
            for field, expected_value in zip(_FIELD_COLUMNS, expected)
            if row[field] != expected_value
        )

    if all(row[field] == "not_applicable" for field in _FIELD_COLUMNS):
        return _FIELD_COLUMNS
    generic_values = {*_TRACEABLE_VALUES, "not_applicable"}
    return tuple(
        field
        for field in _FIELD_COLUMNS
        if not isinstance(row[field], str) or row[field] not in generic_values
    )


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

    row_mismatches = [
        (str(row["modality"]), _row_mismatches(row))
        for _, row in frame.iterrows()
    ]
    missing_required = tuple(
        field
        for field in _FIELD_COLUMNS
        if any(field in mismatches for _, mismatches in row_mismatches)
    )
    active_fire_mismatches = frozenset(
        field
        for modality, mismatches in row_mismatches
        if modality == "active_fire"
        for field in mismatches
    )
    operational_blockers: list[str] = []
    if (
        frame["availability_time"].isin(_AVAILABILITY_BLOCKERS).any()
        or "availability_time" in active_fire_mismatches
    ):
        operational_blockers.append("availability_time")
    if _is_public_registry(frame):
        operational_blockers.append("event_roi_provenance")
    if (
        frame["target_validity"].eq("unavailable").any()
        or "target_validity" in active_fire_mismatches
    ):
        operational_blockers.append("target_validity")

    is_natural = not missing_required
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
