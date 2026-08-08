from pathlib import Path

import pandas as pd
import pytest

from wildfire_phase0.contract import audit_contract, load_contract


def test_public_wsts_contract_downgrades_to_controlled_missingness() -> None:
    frame = pd.DataFrame([
        {"modality": "active_fire", "observation_time": "positive_pixel_hour_only",
         "availability_time": "unavailable", "qa": "filtered_not_retained",
         "coverage": "unavailable", "target_validity": "unavailable"},
        {"modality": "viirs_reflectance", "observation_time": "daily_aggregate_only",
         "availability_time": "unavailable", "qa": "unavailable",
         "coverage": "nan_unknown_cause", "target_validity": "not_applicable"},
    ])

    decision = audit_contract(frame)

    assert decision.status == "continue_controlled"
    assert "target_validity" in decision.missing_required


def test_traceable_contract_allows_natural_missingness() -> None:
    frame = pd.DataFrame([
        {"modality": "active_fire", "observation_time": "traceable",
         "availability_time": "traceable", "qa": "traceable",
         "coverage": "traceable", "target_validity": "traceable"},
    ])

    assert audit_contract(frame).status == "continue_natural"


def test_empty_contract_is_blocked() -> None:
    frame = pd.DataFrame(columns=[
        "modality", "observation_time", "availability_time", "qa", "coverage", "target_validity",
    ])

    assert audit_contract(frame).status == "blocked"


def test_missing_required_schema_column_raises_value_error() -> None:
    frame = pd.DataFrame([{"modality": "active_fire"}])

    with pytest.raises(ValueError, match="missing required column: observation_time"):
        audit_contract(frame)


def test_load_contract_returns_seven_expected_modalities() -> None:
    path = Path("configs/wstsplus_field_contract.csv")

    frame = load_contract(path)

    assert frame["modality"].tolist() == [
        "active_fire",
        "gfs_forecast",
        "gridmet",
        "land_cover",
        "ndvi_evi",
        "terrain",
        "viirs_reflectance",
    ]
