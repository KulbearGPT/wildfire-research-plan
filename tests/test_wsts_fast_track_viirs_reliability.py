from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np


def test_latest_reliable_viirs_observation_ignores_invalid_and_low_fire() -> None:
    from reproductions.wsts_fast_track.viirs_reliability import (
        update_latest_valid_minutes,
    )

    latest = np.full(4, -1, dtype=np.int16)
    update_latest_valid_minutes(
        latest,
        np.array([0, 1, 2, 3]),
        np.array([5, 4, 8, 7], dtype=np.uint8),
        600,
    )
    update_latest_valid_minutes(
        latest,
        np.array([0, 1]),
        np.array([4, 5], dtype=np.uint8),
        800,
    )

    assert latest.tolist() == [600, 800, 600, -1]


def test_finalize_reliability_preserves_wsts_positive_time_and_finite_age() -> None:
    from reproductions.wsts_fast_track.viirs_reliability import finalize_reliability

    latest = np.array([[600, -1], [-1, 1320]], dtype=np.int16)
    active_fire = np.array([[0.0, 2124.0], [np.nan, 0.0]], dtype=np.float32)

    reliability, age_hours = finalize_reliability(latest, active_fire)

    assert reliability.dtype == np.uint8
    assert reliability.tolist() == [[1, 1], [0, 1]]
    assert np.all(np.isfinite(age_hours))
    np.testing.assert_allclose(age_hours, [[14.0, 2.6], [24.0, 2.0]])


def test_cmr_selection_is_causal_sorted_and_deduplicated() -> None:
    from reproductions.wsts_fast_track.viirs_reliability import (
        select_causal_unique_granules,
    )

    entries = [
        {
            "producer_granule_id": "VNP14IMG.A2021158.2024.002.abc",
            "time_start": "2021-06-07T20:24:00.000Z",
            "time_end": "2021-06-07T20:30:00.000Z",
        },
        {
            "producer_granule_id": "VNP14IMG.A2021158.0900.002.def.nc",
            "time_start": "2021-06-07T09:00:00.000Z",
            "time_end": "2021-06-07T09:06:00.000Z",
        },
        {
            "producer_granule_id": "VNP14IMG.A2021158.2024.002.abc",
            "time_start": "2021-06-07T20:24:00.000Z",
            "time_end": "2021-06-07T20:30:00.000Z",
        },
        {
            "producer_granule_id": "VNP14IMG.A2021159.0100.002.future",
            "time_start": "2021-06-08T01:00:00.000Z",
            "time_end": "2021-06-08T01:06:00.000Z",
        },
    ]

    selected = select_causal_unique_granules(
        entries, datetime(2021, 6, 8, tzinfo=timezone.utc)
    )

    assert [granule.producer_id for granule in selected] == [
        "VNP14IMG.A2021158.0900.002.def.nc",
        "VNP14IMG.A2021158.2024.002.abc.nc",
    ]


def test_cmr_json_retries_transient_connection_resets_only_until_success() -> None:
    from reproductions.wsts_fast_track.viirs_reliability import (
        request_json_with_retries,
    )

    class TemporaryNetworkError(Exception):
        pass

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, bool]:
            return {"ok": True}

    calls = 0
    delays: list[float] = []

    def request():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise TemporaryNetworkError("connection reset")
        return Response()

    payload = request_json_with_retries(
        request,
        (TemporaryNetworkError,),
        sleeper=delays.append,
    )

    assert payload == {"ok": True}
    assert calls == 3
    assert delays == [1.0, 4.0]


def test_model_gate_shifts_every_provenance_sample_by_exactly_one_day() -> None:
    from reproductions.wsts_fast_track.viirs_reliability import (
        FIXED_2021_GATE,
        FIXED_2021_MODEL_GATE,
    )

    assert len(FIXED_2021_GATE) == len(FIXED_2021_MODEL_GATE) == 24
    for provenance, model in zip(FIXED_2021_GATE, FIXED_2021_MODEL_GATE, strict=True):
        assert provenance[0] == model[0]
        assert datetime.fromisoformat(model[1]) == (
            datetime.fromisoformat(provenance[1]) + timedelta(days=1)
        )


def test_target_gate_shifts_every_model_sample_by_exactly_one_day() -> None:
    from reproductions.wsts_fast_track.viirs_reliability import GATES

    model_gate = GATES["model"]
    target_gate = GATES["target"]

    assert len(model_gate) == len(target_gate) == 24
    for model, target in zip(model_gate, target_gate, strict=True):
        assert model[0] == target[0]
        assert datetime.fromisoformat(target[1]) == (
            datetime.fromisoformat(model[1]) + timedelta(days=1)
        )
