"""Predeclared AP comparisons for rapid quantitative screening."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path


SCREEN_PRIMARY_DELTA = 0.005
SCREEN_CLEAN_FLOOR = -0.010
SCREEN_SCENARIO_FLOOR = -0.005


def _scenario_ap(summary: Mapping[str, object], scenario_id: str) -> float:
    if (
        summary.get("schema_version") != 1
        or summary.get("status") != "pass"
        or summary.get("year") != 2021
    ):
        raise ValueError("comparison requires a passing 2021 summary")
    results = summary.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("comparison requires a passing 2021 summary")
    scenario = results.get(scenario_id)
    if not isinstance(scenario, Mapping):
        raise ValueError(f"2021 summary lacks scenario {scenario_id}")
    metrics = scenario.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"2021 summary lacks scenario {scenario_id} metrics")
    value = metrics.get("avg_precision")
    if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"2021 summary has invalid {scenario_id} AP")
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
    deltas = {
        scenario: _scenario_ap(candidate, scenario)
        - _scenario_ap(baseline, scenario)
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
