"""Markdown rendering for the reproducible Phase 0 data gate."""

from collections import defaultdict
from collections.abc import Sequence

import pandas as pd

from wildfire_phase0.contract import ContractDecision
from wildfire_phase0.schema import EventInventory


_FROZEN_SPLIT = "2016–2020 train / 2021 validation / 2022–2023 test"
_TARGET = "next-calendar-day active-fire proxy"


def _markdown_target_counts(
    title: str,
    rows: Sequence[tuple[object, int, int, int, int]],
) -> list[str]:
    lines = [
        f"### Target counts by {title}",
        "",
        f"| {title} | events | target days | zero-target days | positive target pixels |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    if rows:
        lines.extend(
            f"| {value} | {events} | {target_days} | {zero_days} | {positive_pixels} |"
            for value, events, target_days, zero_days, positive_pixels in rows
        )
    else:
        lines.append("| none | 0 | 0 | 0 | 0 |")
    return lines


def _target_counts(
    inventory: Sequence[EventInventory],
    groups: Sequence[object],
) -> list[tuple[object, int, int, int, int]]:
    totals: dict[object, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for item, group in zip(inventory, groups):
        row = totals[group]
        row[0] += 1
        row[1] += item.target_days
        row[2] += item.zero_target_days
        row[3] += item.positive_target_pixels
    return [
        (group, *totals[group])
        for group in sorted(totals, key=str)
    ]


def render_phase0_report(
    inventory: Sequence[EventInventory],
    split_manifest: pd.DataFrame,
    decision: ContractDecision,
    invalid_file_error: str | None,
    commands: Sequence[str],
) -> str:
    """Render a deterministic human-readable summary of a Phase 0 gate run."""
    year_counts = _target_counts(inventory, [item.year for item in inventory])
    inventory_by_id = {
        f"{item.year}:{item.fire_name}": item for item in inventory
    }
    split_inventory = [
        inventory_by_id[str(event_id)]
        for event_id in split_manifest["event_id"]
        if str(event_id) in inventory_by_id
    ]
    split_counts = _target_counts(
        split_inventory,
        [
            str(row["split"])
            for row in split_manifest.to_dict("records")
            if str(row["event_id"]) in inventory_by_id
        ],
    )
    pixel_counts = [item.n_days * item.n_channels * item.height * item.width for item in inventory]
    total_pixels = sum(pixel_counts)
    dataset_nan_fraction = (
        sum(item.nan_fraction * count for item, count in zip(inventory, pixel_counts))
        / total_pixels
        if total_pixels
        else 0.0
    )
    maximum_event_nan_fraction = max((item.nan_fraction for item in inventory), default=0.0)
    error_text = invalid_file_error if invalid_file_error is not None else "none"

    if decision.status == "continue_controlled":
        implication = (
            "continue_controlled preserves the approved project route but removes "
            "natural-missingness and operational claims."
        )
    elif decision.status == "continue_natural":
        implication = "Proceed with natural-missingness analysis under the audited contract."
    else:
        implication = "Pivot: do not proceed until the listed inventory or contract blocker is resolved."

    lines = [
        "# Phase 0 Data Gate",
        "",
        "## Target definition",
        "",
        _TARGET,
        "",
        "## Frozen split",
        "",
        _FROZEN_SPLIT,
        "",
        *_markdown_target_counts("year", year_counts),
        "",
        *_markdown_target_counts("split", split_counts),
        "",
        "## Inventory validation",
        "",
        f"Invalid-file errors: {error_text}",
        "",
        "## NaN summary",
        "",
        f"Events scanned: {len(inventory)}",
        f"Pixel-weighted dataset NaN fraction: {dataset_nan_fraction:.12g}",
        f"Maximum event NaN fraction: {maximum_event_nan_fraction:.12g}",
        "",
        "## Contract audit",
        "",
        "Missing contract fields: " + (", ".join(decision.missing_required) or "none"),
        "",
        "Operational blockers: " + (", ".join(decision.operational_blockers) or "none"),
        "",
        "Notes: " + (" ".join(decision.notes) or "none"),
        "",
        "## Gate decision",
        "",
        f"Status: {decision.status}",
        "",
        f"Implication: {implication}",
        "",
        "## Reproducible commands used",
        "",
        "```powershell",
        *commands,
        "```",
        "",
    ]
    return "\n".join(lines)
