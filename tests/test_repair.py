from pathlib import Path

import numpy as np
import pytest

from wildfire_phase0.repair import (
    classify_active_fire_encoding,
    normalize_active_fire,
    source_fingerprint,
)


def test_normalize_preserves_hour_encoding_and_replaces_nan() -> None:
    values = np.array([[np.nan, 0.0, 6.0, 22.0]], dtype=np.float32)
    normalized, encoding = normalize_active_fire(values)
    assert encoding == "hour"
    assert normalized.tolist() == [[0.0, 0.0, 6.0, 22.0]]


def test_normalize_converts_hhmm_exactly_once() -> None:
    values = np.array([[0.0, 806.0, 1359.0, 2200.0]], dtype=np.float32)
    normalized, encoding = normalize_active_fire(values)
    assert encoding == "hhmm"
    assert normalized.tolist() == [[0.0, 8.0, 13.0, 22.0]]


def test_normalize_marks_zero_only_source() -> None:
    normalized, encoding = normalize_active_fire(
        np.array([[0.0, np.nan]], dtype=np.float32)
    )
    assert encoding == "no_positive_values"
    assert normalized.tolist() == [[0.0, 0.0]]


@pytest.mark.parametrize(
    "values",
    [
        np.array([[-1.0]]),
        np.array([[6.5]]),
        np.array([[6.0, 806.0]]),
        np.array([[2360.0]]),
        np.array([[np.inf]]),
    ],
)
def test_classification_rejects_invalid_or_mixed_encoding(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="active-fire encoding"):
        classify_active_fire_encoding(values)


def test_source_fingerprint_is_order_independent_and_metadata_sensitive(
    tmp_path: Path,
) -> None:
    first = tmp_path / "2016" / "fire_a" / "2016-01-01.tif"
    second = tmp_path / "2016" / "fire_a" / "2016-01-02.tif"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    forward = source_fingerprint([first, second], tmp_path)
    reverse = source_fingerprint([second, first], tmp_path)
    assert forward == reverse
    second.write_bytes(b"changed-size")
    assert source_fingerprint([first, second], tmp_path) != forward
