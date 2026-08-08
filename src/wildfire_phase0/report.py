"""Markdown rendering for the reproducible Phase 0 data gate."""

from collections import Counter
from collections.abc import Sequence

import pandas as pd

from wildfire_phase0.contract import ContractDecision
from wildfire_phase0.schema import EventInventory


_FROZEN_SPLIT = "2016–2020 train / 2021 validation / 2022–2023 test"
_TARGET = "next-calendar-day active-fire proxy"


def _markdown_counts(title: str, rows: Sequence[tuple[object, int]]) -> list[str]:
    lines = [f"### Counts by {title}", "", f"| {title} | events |", "| --- | ---: |"]
    if rows:
        lines.extend(f"| {value} | {count} |" for value, count in rows)
    else:
        lines.append("| none | 0 |")
    return lines


def render_phase0_report(
    inventory: Sequence[EventInventory],
    split_manifest: pd.DataFrame,
    decision: ContractDecision,
    invalid_file_error: str | None,
    commands: Sequence[str],
) -> str:
    """Render a deterministic human-readable summary of a Phase 0 gate run."""
    year_counts = sorted(Counter(item.year for item in inventory).items())
    split_counts = sorted(
        ((str(split), int(count)) for split, count in split_manifest["split"].value_counts().items()),
        key=lambda item: item[0],
    )
    total_nan_fraction = sum(item.nan_fraction for item in inventory)
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
        *_markdown_counts("year", year_counts),
        "",
        *_markdown_counts("split", split_counts),
        "",
        "## Inventory validation",
        "",
        f"Invalid-file errors: {error_text}",
        "",
        "## NaN summary",
        "",
        f"Events scanned: {len(inventory)}",
        f"Sum of event NaN fractions: {total_nan_fraction:.12g}",
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
