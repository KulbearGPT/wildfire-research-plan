from __future__ import annotations

import pytest

from reproductions.wsts_fast_track.quantitative_evidence import (
    compare_2021,
    compare_three_years,
)


def _summary(identifier: str, values: dict[str, float]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "pass",
        "year": 2021,
        "baseline_id": identifier,
        "results": {
            scenario: {"metrics": {"avg_precision": value}}
            for scenario, value in values.items()
        },
    }


def test_compare_2021_applies_predeclared_screen_gate() -> None:
    baseline = _summary(
        "B0", {"M00": 0.50, "M01": 0.10, "M06": 0.20, "M07": 0.15}
    )
    candidate = _summary(
        "B2", {"M00": 0.495, "M01": 0.12, "M06": 0.20, "M07": 0.15}
    )

    comparison = compare_2021(
        baseline, candidate, primary_scenarios=("M01",)
    )

    assert comparison["scenario_ap_delta"]["M01"] == pytest.approx(0.02)
    assert comparison["primary_ap_delta"] == pytest.approx(0.02)
    assert comparison["clean_ap_delta"] == pytest.approx(-0.005)
    assert comparison["baseline_scenario_ap"]["M01"] == pytest.approx(0.10)
    assert comparison["candidate_scenario_ap"]["M01"] == pytest.approx(0.12)
    assert comparison["screen_positive"] is True


def test_compare_2021_rejects_clean_or_primary_regression() -> None:
    baseline = _summary(
        "B2", {"M00": 0.50, "M01": 0.30, "M06": 0.20, "M07": 0.15}
    )
    clean_regression = _summary(
        "B3", {"M00": 0.48, "M01": 0.30, "M06": 0.22, "M07": 0.17}
    )
    weak_primary = _summary(
        "B3", {"M00": 0.50, "M01": 0.30, "M06": 0.202, "M07": 0.152}
    )

    assert (
        compare_2021(
            baseline, clean_regression, primary_scenarios=("M06", "M07")
        )["screen_positive"]
        is False
    )
    assert (
        compare_2021(
            baseline, weak_primary, primary_scenarios=("M06", "M07")
        )["screen_positive"]
        is False
    )


def test_compare_2021_rejects_incomplete_summary() -> None:
    baseline = _summary("B0", {"M00": 0.5, "M01": 0.1})
    candidate = _summary("B2", {"M00": 0.5, "M01": 0.2})
    candidate["year"] = 2022

    with pytest.raises(ValueError, match="2021 summary"):
        compare_2021(baseline, candidate, primary_scenarios=("M01",))


def test_compare_three_years_separates_supported_from_reliable() -> None:
    pairs = []
    for year, primary_delta, clean_delta in (
        (2021, 0.010, -0.002),
        (2022, 0.008, -0.003),
        (2023, -0.001, -0.004),
    ):
        baseline = _summary("control", {"M00": 0.5, "M01": 0.2})
        candidate = _summary(
            "method", {"M00": 0.5 + clean_delta, "M01": 0.2 + primary_delta}
        )
        baseline["year"] = year
        candidate["year"] = year
        pairs.append((baseline, candidate))

    comparison = compare_three_years(pairs, primary_scenarios=("M01",))

    assert comparison["positive_year_count"] == 2
    assert comparison["mean_primary_ap_delta"] == pytest.approx(0.017 / 3)
    assert comparison["years"]["2022"]["baseline_primary_ap"] == pytest.approx(
        0.2
    )
    assert comparison["years"]["2022"]["candidate_primary_ap"] == pytest.approx(
        0.208
    )
    assert comparison["quantitatively_supported"] is True
    assert comparison["reliable_contribution_candidate"] is False


def test_compare_three_years_requires_exact_heldout_years() -> None:
    baseline = _summary("control", {"M00": 0.5, "M01": 0.2})
    candidate = _summary("method", {"M00": 0.5, "M01": 0.21})
    with pytest.raises(ValueError, match="exactly 2021, 2022, and 2023"):
        compare_three_years(
            [(baseline, candidate)] * 3, primary_scenarios=("M01",)
        )
