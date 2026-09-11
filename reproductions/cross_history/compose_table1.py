"""Compose matched test-only Table 1 statistics from summary JSON files."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


SCENARIOS = ("M00", "M01", "M06", "M07")
TEST_YEARS = (2022, 2023)
SEEDS = (0, 1, 2)


def compose_table(paths):
    groups = defaultdict(list)
    for path in paths:
        summary = json.loads(Path(path).read_text())
        if summary["year"] not in TEST_YEARS:
            continue
        key = (summary["architecture"], summary["history"], summary["method"])
        groups[key].append(summary)

    required_cells = {(seed, year) for seed in SEEDS for year in TEST_YEARS}
    controls = {}
    rows = []
    for key in sorted(groups, key=lambda item: (item[0], item[1], item[2] != "control", item[2])):
        architecture, history, method = key
        summaries = groups[key]
        cells = {(item["seed"], item["year"]) for item in summaries}
        if cells != required_cells or len(summaries) != len(required_cells):
            raise ValueError(
                f"{key} requires matched seeds 0/1/2 in both 2022 and 2023")
        scenario_ap = {
            scenario: mean(item["results"][scenario]["avg_precision"]
                           for item in summaries)
            for scenario in SCENARIOS
        }
        primary = mean(scenario_ap[scenario] for scenario in SCENARIOS[1:])
        seed_primary = [mean(
            item["results"][scenario]["avg_precision"]
            for item in summaries if item["seed"] == seed
            for scenario in SCENARIOS[1:]
        ) for seed in SEEDS]
        row = dict(architecture=architecture, history=history, method=method,
                   **scenario_ap, primary=primary,
                   seed_std=pstdev(seed_primary), delta=0.0)
        if method == "control":
            controls[(architecture, history)] = primary
        else:
            control_key = (architecture, history)
            if control_key not in controls:
                raise ValueError(f"missing matched control for {key}")
            row["delta"] = primary - controls[control_key]
        rows.append(row)
    return {"years": list(TEST_YEARS), "seeds": list(SEEDS), "rows": rows}


def render_markdown(table):
    lines = [
        "| Architecture | T | Method | M00 | M01 | M06 | M07 | Primary | Delta |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in table["rows"]:
        primary = f'{row["primary"]:.6f} ± {row["seed_std"]:.6f}'
        lines.append(
            f'| {row["architecture"]} | {row["history"]} | {row["method"]} | '
            f'{row["M00"]:.6f} | {row["M01"]:.6f} | {row["M06"]:.6f} | '
            f'{row["M07"]:.6f} | {primary} | {row["delta"]:+.6f} |')
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    args = parser.parse_args()
    table = compose_table(args.summaries)
    args.output.write_text(json.dumps(table, indent=2) + "\n")
    args.markdown_output.write_text(render_markdown(table))


if __name__ == "__main__":
    main()
