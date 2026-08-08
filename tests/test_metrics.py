import numpy as np
import pandas as pd
import pytest

from wildfire_phase0.metrics import (
    average_precision_safe,
    event_macro_ap,
    zero_target_event_false_alarm_rate,
    zero_target_false_alarm_rate,
)


def test_average_precision_safe_marks_zero_positive_target() -> None:
    value, has_positive = average_precision_safe(
        np.zeros(4), np.array([0.0, 0.2, 0.7, 0.1])
    )

    assert np.isnan(value)
    assert has_positive is False
    assert zero_target_false_alarm_rate(
        np.zeros(4), np.array([0.0, 0.2, 0.7, 0.1]), 0.5
    ) == 0.25


def test_average_precision_safe_returns_ordinary_average_precision() -> None:
    value, has_positive = average_precision_safe(
        np.array([[1, 0], [1, 0]]), np.array([[0.9, 0.8], [0.7, 0.1]])
    )

    assert value == pytest.approx(5 / 6)
    assert has_positive is True


@pytest.mark.parametrize(
    ("y_true", "y_score"),
    [
        (np.array([]), np.array([])),
        (np.array([0, 1]), np.array([0.1])),
        (np.array([0, 2]), np.array([0.1, 0.2])),
        (np.array([0, 1]), np.array([0.1, np.inf])),
    ],
)
def test_average_precision_safe_rejects_invalid_targets_or_scores(
    y_true: np.ndarray, y_score: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        average_precision_safe(y_true, y_score)


def test_zero_target_false_alarm_rate_includes_scores_at_threshold() -> None:
    assert zero_target_false_alarm_rate(
        np.zeros(4), np.array([0.49, 0.5, 0.7, 0.1]), 0.5
    ) == 0.5


@pytest.mark.parametrize("threshold", [np.nan, np.inf])
def test_zero_target_false_alarm_rate_rejects_non_finite_threshold(threshold: float) -> None:
    with pytest.raises(ValueError):
        zero_target_false_alarm_rate(np.zeros(2), np.zeros(2), threshold)


def test_zero_target_false_alarm_rate_rejects_positive_target() -> None:
    with pytest.raises(ValueError, match="zero-positive"):
        zero_target_false_alarm_rate(np.array([0, 1]), np.array([0.1, 0.2]), 0.5)


def test_event_macro_ap_concatenates_days_within_each_event_equally() -> None:
    records = pd.DataFrame({
        "event_id": ["alpha", "alpha", "bravo"],
        "y_true": [np.array([1, 0]), np.array([0, 1]), np.array([1, 0])],
        "y_score": [np.array([0.9, 0.2]), np.array([0.1, 0.8]), np.array([0.2, 0.3])],
    })

    assert event_macro_ap(records) == (1.0 + 0.5) / 2


def test_event_macro_ap_returns_nan_when_any_event_has_zero_positives_without_mutation() -> None:
    records = pd.DataFrame({
        "event_id": ["zero", "valid"],
        "y_true": [np.array([0, 0]), np.array([1, 0])],
        "y_score": [np.array([0.8, 0.1]), np.array([0.9, 0.2])],
    })
    original = records.copy(deep=True)

    assert np.isnan(event_macro_ap(records))
    assert records.equals(original)


def test_event_macro_ap_returns_nan_when_every_event_has_zero_positives() -> None:
    records = pd.DataFrame({
        "event_id": ["alpha", "bravo"],
        "y_true": [np.array([0, 0]), np.array([0])],
        "y_score": [np.array([0.2, 0.1]), np.array([0.9])],
    })

    assert np.isnan(event_macro_ap(records))


def test_event_macro_ap_requires_all_columns() -> None:
    with pytest.raises(ValueError, match="event_id"):
        event_macro_ap(pd.DataFrame({"y_true": [np.array([0])], "y_score": [np.array([0.0])]}))


def test_zero_target_event_false_alarm_rate_aggregates_entire_zero_positive_events() -> None:
    records = pd.DataFrame({
        "event_id": ["zero_a", "valid", "zero_b", "zero_a"],
        "y_true": [
            np.array([0, 0]),
            np.array([1, 0]),
            np.array([0]),
            np.array([0, 0]),
        ],
        "y_score": [
            np.array([0.6, 0.1]),
            np.array([0.9, 0.8]),
            np.array([0.7]),
            np.array([0.5, 0.2]),
        ],
    })

    assert zero_target_event_false_alarm_rate(records, 0.5) == pytest.approx(3 / 5)


def test_zero_target_event_false_alarm_rate_returns_nan_without_zero_positive_event() -> None:
    records = pd.DataFrame({
        "event_id": ["alpha", "bravo"],
        "y_true": [np.array([1, 0]), np.array([0, 1])],
        "y_score": [np.array([0.9, 0.1]), np.array([0.2, 0.8])],
    })

    assert np.isnan(zero_target_event_false_alarm_rate(records, 0.5))


def test_zero_target_event_false_alarm_rate_uses_record_validation() -> None:
    with pytest.raises(ValueError, match="event_id"):
        zero_target_event_false_alarm_rate(
            pd.DataFrame({"y_true": [np.array([0])], "y_score": [np.array([0.1])]}),
            0.5,
        )
