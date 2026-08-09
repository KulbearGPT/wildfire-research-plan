"""Safe event-level metrics for wildfire predictions."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


_REQUIRED_COLUMNS = {"event_id", "y_true", "y_score"}


@dataclass(frozen=True)
class BinaryScoreCounts:
    """Confusion-matrix cells for binary target and binary prediction scores."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def __post_init__(self) -> None:
        if any(value < 0 for value in (self.tp, self.fp, self.fn, self.tn)):
            raise ValueError("binary score counts must be nonnegative")

    def __add__(self, other: "BinaryScoreCounts") -> "BinaryScoreCounts":
        if not isinstance(other, BinaryScoreCounts):
            return NotImplemented
        return BinaryScoreCounts(
            self.tp + other.tp,
            self.fp + other.fp,
            self.fn + other.fn,
            self.tn + other.tn,
        )

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def positives(self) -> int:
        return self.tp + self.fn

    @property
    def predicted_positives(self) -> int:
        return self.tp + self.fp

    @property
    def prevalence(self) -> float:
        return self.positives / self.total if self.total else float("nan")


def _validated_arrays(y_true: object, y_score: object) -> tuple[np.ndarray, np.ndarray]:
    """Flatten and validate a binary target and finite prediction scores."""
    target = np.asarray(y_true).ravel()
    try:
        scores = np.asarray(y_score, dtype=float).ravel()
    except (TypeError, ValueError) as error:
        raise ValueError("y_score must contain finite numeric values") from error

    if target.size == 0 or scores.size == 0 or target.size != scores.size:
        raise ValueError("y_true and y_score must have matching non-empty lengths")
    if not np.all((target == 0) | (target == 1)):
        raise ValueError("y_true must be binary")
    if not np.all(np.isfinite(scores)):
        raise ValueError("y_score must contain finite values")
    return target, scores


def binary_score_counts(y_true: object, y_score: object) -> BinaryScoreCounts:
    """Return confusion-matrix counts without retaining individual pixels."""
    target, scores = _validated_arrays(y_true, y_score)
    if not np.all((scores == 0) | (scores == 1)):
        raise ValueError("y_score must be binary")

    target_is_one = target == 1
    score_is_one = scores == 1
    return BinaryScoreCounts(
        tp=int(np.count_nonzero(target_is_one & score_is_one)),
        fp=int(np.count_nonzero(~target_is_one & score_is_one)),
        fn=int(np.count_nonzero(target_is_one & ~score_is_one)),
        tn=int(np.count_nonzero(~target_is_one & ~score_is_one)),
    )


def binary_average_precision(counts: BinaryScoreCounts) -> float:
    """Return non-interpolated AP for a binary-valued score distribution."""
    if counts.total == 0:
        raise ValueError("binary score counts must not be empty")
    if counts.positives == 0:
        return float("nan")
    recall_at_one = counts.tp / counts.positives
    precision_at_one = (
        counts.tp / counts.predicted_positives if counts.predicted_positives else 0.0
    )
    return recall_at_one * precision_at_one + (1.0 - recall_at_one) * counts.prevalence


def binary_false_alarm_rate(counts: BinaryScoreCounts) -> float:
    """Return the positive-score rate for a zero-positive target subset."""
    if counts.total == 0:
        raise ValueError("binary score counts must not be empty")
    if counts.positives:
        raise ValueError("false-alarm rate requires a zero-positive target")
    return counts.fp / counts.total


def average_precision_safe(y_true: object, y_score: object) -> tuple[float, bool]:
    """Return AP only when the target contains at least one positive pixel."""
    target, scores = _validated_arrays(y_true, y_score)
    if not np.any(target == 1):
        return float("nan"), False
    return float(average_precision_score(target, scores)), True


def zero_target_false_alarm_rate(y_true: object, y_score: object, threshold: float) -> float:
    """Return the fraction of zero-target scores at or above ``threshold``."""
    target, scores = _validated_arrays(y_true, y_score)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if np.any(target == 1):
        raise ValueError("false-alarm rate requires a zero-positive target")
    return float(np.mean(scores >= threshold))


def _validate_records(records: pd.DataFrame) -> None:
    missing_columns = _REQUIRED_COLUMNS.difference(records.columns)
    if missing_columns:
        raise ValueError(f"records missing required columns: {sorted(missing_columns)}")
    if records["event_id"].isna().any():
        raise ValueError("event_id values must not be missing")


def _validated_event_arrays(records: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    _validate_records(records)
    events: list[tuple[np.ndarray, np.ndarray]] = []
    for _, event_records in records.groupby("event_id", sort=False):
        target = np.concatenate([np.asarray(values).ravel() for values in event_records["y_true"]])
        scores = np.concatenate([np.asarray(values).ravel() for values in event_records["y_score"]])
        events.append(_validated_arrays(target, scores))
    return events


def event_macro_ap(records: pd.DataFrame) -> float:
    """Average per-event AP, returning NaN when any event AP is undefined."""
    event_scores: list[float] = []
    for target, scores in _validated_event_arrays(records):
        value, has_positive = average_precision_safe(target, scores)
        if not has_positive:
            return float("nan")
        event_scores.append(value)

    if not event_scores:
        return float("nan")
    return float(np.mean(event_scores))


def zero_target_event_false_alarm_rate(records: pd.DataFrame, threshold: float) -> float:
    """Aggregate false alarms across rows/days that contain no positive targets."""
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    _validate_records(records)
    zero_row_arrays = [
        arrays
        for arrays in (
            _validated_arrays(y_true, y_score)
            for y_true, y_score in zip(records["y_true"], records["y_score"])
        )
        if not np.any(arrays[0] == 1)
    ]
    if not zero_row_arrays:
        return float("nan")

    target = np.concatenate([arrays[0] for arrays in zero_row_arrays])
    scores = np.concatenate([arrays[1] for arrays in zero_row_arrays])
    return zero_target_false_alarm_rate(target, scores, threshold)
