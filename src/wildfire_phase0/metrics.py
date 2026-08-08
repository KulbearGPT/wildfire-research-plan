"""Safe event-level metrics for wildfire predictions."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


_REQUIRED_COLUMNS = {"event_id", "y_true", "y_score"}


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


def event_macro_ap(records: pd.DataFrame) -> float:
    """Average defined AP values after concatenating all rows for each event."""
    missing_columns = _REQUIRED_COLUMNS.difference(records.columns)
    if missing_columns:
        raise ValueError(f"records missing required columns: {sorted(missing_columns)}")
    if records["event_id"].isna().any():
        raise ValueError("event_id values must not be missing")

    event_scores: list[float] = []
    for _, event_records in records.groupby("event_id", sort=False):
        target = np.concatenate([np.asarray(values).ravel() for values in event_records["y_true"]])
        scores = np.concatenate([np.asarray(values).ravel() for values in event_records["y_score"]])
        value, has_positive = average_precision_safe(target, scores)
        if has_positive:
            event_scores.append(value)

    if not event_scores:
        return float("nan")
    return float(np.mean(event_scores))
