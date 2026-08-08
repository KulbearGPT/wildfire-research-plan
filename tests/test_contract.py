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


def test_not_applicable_active_fire_fields_prevent_natural_missingness() -> None:
    frame = pd.DataFrame([
        {"modality": "active_fire", "observation_time": "not_applicable",
         "availability_time": "not_applicable", "qa": "not_applicable",
         "coverage": "not_applicable", "target_validity": "not_applicable"},
    ])

    assert audit_contract(frame).status == "continue_controlled"


@pytest.mark.parametrize("untraceable_value", [None, 7, "unexpected"])
def test_applicable_untraceable_value_prevents_natural_missingness(
    untraceable_value: object,
) -> None:
    frame = pd.DataFrame([
        {"modality": "active_fire", "observation_time": untraceable_value,
         "availability_time": "traceable", "qa": "traceable",
         "coverage": "traceable", "target_validity": "traceable"},
    ])

    assert audit_contract(frame).status == "continue_controlled"


def test_custom_traceable_registry_does_not_report_public_roi_blocker() -> None:
    modalities = [
        "active_fire",
        "viirs_reflectance",
        "ndvi_evi",
        "gridmet",
        "gfs_forecast",
        "terrain",
        "land_cover",
    ]
    frame = pd.DataFrame([
        {"modality": modality, "observation_time": "traceable",
         "availability_time": "traceable", "qa": "traceable",
         "coverage": "traceable", "target_validity": "traceable"}
        for modality in modalities
    ])

    decision = audit_contract(frame)

    assert decision.status == "continue_natural"
    assert "event_roi_provenance" not in decision.operational_blockers


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


def test_committed_public_registry_reports_event_roi_provenance_blocker() -> None:
    frame = load_contract(Path("configs/wstsplus_field_contract.csv"))

    decision = audit_contract(frame)

    assert "event_roi_provenance" in decision.operational_blockers


def test_public_registry_note_edit_preserves_event_roi_provenance_blocker() -> None:
    frame = load_contract(Path("configs/wstsplus_field_contract.csv"))
    frame.loc[frame["modality"] == "active_fire", "audit_note"] = "Clarified prose only."

    decision = audit_contract(frame)

    assert "event_roi_provenance" in decision.operational_blockers


def test_public_registry_source_edit_does_not_match_public_fingerprint() -> None:
    frame = load_contract(Path("configs/wstsplus_field_contract.csv"))
    frame.loc[frame["modality"] == "active_fire", "source_url"] = "https://example.invalid"

    decision = audit_contract(frame)

    assert "event_roi_provenance" not in decision.operational_blockers
