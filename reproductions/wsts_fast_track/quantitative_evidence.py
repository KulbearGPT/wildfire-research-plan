"""Predeclared AP comparisons for rapid quantitative screening."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path


SCREEN_PRIMARY_DELTA = 0.005
SCREEN_CLEAN_FLOOR = -0.010
SCREEN_SCENARIO_FLOOR = -0.005


def _scenario_ap(
    summary: Mapping[str, object], scenario_id: str, *, expected_year: int
) -> float:
    if (
        summary.get("schema_version") != 1
        or summary.get("status") != "pass"
        or summary.get("year") != expected_year
    ):
        raise ValueError(f"comparison requires a passing {expected_year} summary")
    results = summary.get("results")
    if not isinstance(results, Mapping):
        raise ValueError(f"comparison requires a passing {expected_year} summary")
    scenario = results.get(scenario_id)
    if not isinstance(scenario, Mapping):
        raise ValueError(f"{expected_year} summary lacks scenario {scenario_id}")
    metrics = scenario.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{expected_year} summary lacks scenario {scenario_id} metrics")
    value = metrics.get("avg_precision")
    if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{expected_year} summary has invalid {scenario_id} AP")
    return float(value)


def compare_2021(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    primary_scenarios: tuple[str, ...],
) -> dict[str, object]:
    """Compare a candidate with its matched baseline under the frozen gate."""

    if not primary_scenarios or len(set(primary_scenarios)) != len(primary_scenarios):
        raise ValueError("primary scenarios must be nonempty and unique")
    scenarios = ("M00", *primary_scenarios)
    baseline_ap = {
        scenario: _scenario_ap(baseline, scenario, expected_year=2021)
        for scenario in scenarios
    }
    candidate_ap = {
        scenario: _scenario_ap(candidate, scenario, expected_year=2021)
        for scenario in scenarios
    }
    deltas = {
        scenario: candidate_ap[scenario] - baseline_ap[scenario]
        for scenario in scenarios
    }
    primary_delta = sum(deltas[item] for item in primary_scenarios) / len(
        primary_scenarios
    )
    screen_positive = (
        primary_delta >= SCREEN_PRIMARY_DELTA
        and deltas["M00"] >= SCREEN_CLEAN_FLOOR
        and min(deltas[item] for item in primary_scenarios)
        >= SCREEN_SCENARIO_FLOOR
    )
    return {
        "schema_version": 1,
        "year": 2021,
        "primary_scenarios": list(primary_scenarios),
        "baseline_scenario_ap": baseline_ap,
        "candidate_scenario_ap": candidate_ap,
        "scenario_ap_delta": deltas,
        "primary_ap_delta": primary_delta,
        "clean_ap_delta": deltas["M00"],
        "thresholds": {
            "primary_ap_delta_min": SCREEN_PRIMARY_DELTA,
            "clean_ap_delta_min": SCREEN_CLEAN_FLOOR,
            "per_primary_scenario_delta_min": SCREEN_SCENARIO_FLOOR,
        },
        "screen_positive": screen_positive,
    }


def compare_three_years(
    pairs: Sequence[tuple[Mapping[str, object], Mapping[str, object]]],
    *,
    primary_scenarios: tuple[str, ...],
) -> dict[str, object]:
    """Apply the frozen level-3 and reliable-candidate rules across test years."""

    if not primary_scenarios or len(set(primary_scenarios)) != len(primary_scenarios):
        raise ValueError("primary scenarios must be nonempty and unique")
    by_year: dict[int, dict[str, object]] = {}
    for baseline, candidate in pairs:
        year = baseline.get("year")
        if (
            not isinstance(year, int)
            or candidate.get("year") != year
            or year in by_year
        ):
            raise ValueError("comparison requires exactly 2021, 2022, and 2023")
        scenarios = ("M00", *primary_scenarios)
        baseline_ap = {
            scenario: _scenario_ap(baseline, scenario, expected_year=year)
            for scenario in scenarios
        }
        candidate_ap = {
            scenario: _scenario_ap(candidate, scenario, expected_year=year)
            for scenario in scenarios
        }
        deltas = {
            scenario: candidate_ap[scenario] - baseline_ap[scenario]
            for scenario in scenarios
        }
        by_year[year] = {
            "baseline_scenario_ap": baseline_ap,
            "candidate_scenario_ap": candidate_ap,
            "baseline_primary_ap": sum(baseline_ap[item] for item in primary_scenarios)
            / len(primary_scenarios),
            "candidate_primary_ap": sum(
                candidate_ap[item] for item in primary_scenarios
            )
            / len(primary_scenarios),
            "scenario_ap_delta": deltas,
            "primary_ap_delta": sum(deltas[item] for item in primary_scenarios)
            / len(primary_scenarios),
            "clean_ap_delta": deltas["M00"],
        }
    if set(by_year) != {2021, 2022, 2023}:
        raise ValueError("comparison requires exactly 2021, 2022, and 2023")
    primary_deltas = [
        float(by_year[year]["primary_ap_delta"]) for year in sorted(by_year)
    ]
    clean_deltas = [float(by_year[year]["clean_ap_delta"]) for year in sorted(by_year)]
    positive_year_count = sum(delta > 0.0 for delta in primary_deltas)
    mean_primary_delta = sum(primary_deltas) / 3
    quantitatively_supported = (
        positive_year_count >= 2 and mean_primary_delta >= SCREEN_PRIMARY_DELTA
    )
    reliable = (
        quantitatively_supported
        and positive_year_count == 3
        and min(clean_deltas) >= SCREEN_CLEAN_FLOOR
    )
    return {
        "schema_version": 1,
        "primary_scenarios": list(primary_scenarios),
        "years": {str(year): by_year[year] for year in sorted(by_year)},
        "positive_year_count": positive_year_count,
        "mean_primary_ap_delta": mean_primary_delta,
        "quantitatively_supported": quantitatively_supported,
        "reliable_contribution_candidate": reliable,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--primary", nargs="+", required=True)
    args = parser.parse_args(argv)
    baseline = json.loads(args.baseline.resolve(strict=True).read_text())
    candidate = json.loads(args.candidate.resolve(strict=True).read_text())
    if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
        raise ValueError("summaries must be JSON objects")
    result = compare_2021(
        baseline, candidate, primary_scenarios=tuple(args.primary)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
