#!/usr/bin/env python3
# Code lifecycle: active_support. Paper evidence and figure preparation.
# Scope and settings: paper/materials/README.md; docs/CODE_LIFECYCLE.md.
"""Render the publication figures from the paper package's auditable CSVs.

Lifecycle: active_support. Settings: docs/CODE_LIFECYCLE.md.
The renderer reads derived tables, never models or evaluation data. Run it in a
Slurm CPU allocation; even Matplotlib is imported only after the allocation
check. All labels and captions are intended for the English paper materials.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


CELLS = tuple((h, y) for h in (1, 5) for y in (2021, 2022, 2023))
SCENARIOS = ("clean", "fire", "mild", "severe")
METHOD_LABELS = {
    "ERM": "ERM",
    "X22": "X22",
    "X14_ROUTE": "X14 + ERM",
    "X17_ROUTE": "X17 + ERM",
    "X22_X14_ERM_CLEAN": "X22 + X14",
    "X22_X17_ERM_CLEAN": "X22 + X17 (ERM clean)",
    "X22_X17_HISTORICAL": "X22 + X17 (X22 clean)",
}
PALETTE = {
    "ERM": "#666666",
    "X22": "#D55E00",
    "X14_ROUTE": "#56B4E9",
    "X17_ROUTE": "#009E73",
    "X22_X14_ERM_CLEAN": "#0072B2",
    "X22_X17_ERM_CLEAN": "#006747",
    "X22_X17_HISTORICAL": "#CC79A7",
}
MARKERS = {
    "ERM": "o", "X22": "^", "X14_ROUTE": "v", "X17_ROUTE": "P",
    "X22_X14_ERM_CLEAN": "s", "X22_X17_ERM_CLEAN": "D",
    "X22_X17_HISTORICAL": "X",
}


def _csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class _Renderer:
    def __init__(self, root: Path):
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("Figure generation requires a Slurm CPU allocation")
        import math
        import statistics
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
        from matplotlib.lines import Line2D
        from matplotlib.patches import FancyBboxPatch

        self.plt = plt
        self.mpl = matplotlib
        self.Line2D = Line2D
        self.FancyBboxPatch = FancyBboxPatch
        self.TwoSlopeNorm = TwoSlopeNorm
        self.mean = statistics.mean
        self.stdev = statistics.stdev
        self.root = root.resolve()
        self.output = self.root / "figures"
        self.output.mkdir(parents=True, exist_ok=True)
        self.manifest = []
        # The package also retains historical and screen-only experiments,
        # which deliberately have no false matched-ERM delta columns. Their
        # separate protocols must not enter any of the current-mainline plots.
        self.rows = [row for row in _csv(self.root / "data/metrics_by_seed.csv")
                     if row["method"] in METHOD_LABELS]
        self.effects = _csv(self.root / "tables/paired_effects.csv")
        self.attribution = _csv(self.root / "tables/attribution.csv")
        self.costs = _csv(self.root / "tables/costs.csv")
        numeric = ("primary", "block", "clean", "fire", "mild", "severe",
                   "primary_delta", "block_delta", "clean_delta")
        self.lookup = {}
        for row in self.rows:
            for key in ("history", "seed", "year"):
                row[key] = int(row[key])
            for key in numeric:
                row[key] = float(row[key])
                if not math.isfinite(row[key]):
                    raise ValueError(f"Non-finite figure input: {row['method']} {key}")
            key = (row["method"], row["history"], row["seed"], row["year"])
            if key in self.lookup:
                raise ValueError(f"Duplicate figure input cell: {key}")
            self.lookup[key] = row
        # A missing panel is an incomplete result, not a reason to silently draw
        # a smaller experiment. The publication figures require all 18 cells.
        for method in METHOD_LABELS:
            for history, year in CELLS:
                self.group(method, history, year)
        self.effect_lookup = {}
        for row in self.effects:
            key = (row["method"], int(row["history"]), int(row["year"]), row["metric"])
            if key in self.effect_lookup:
                raise ValueError(f"Duplicate paired-effects cell: {key}")
            self.effect_lookup[key] = row
        self.attr_lookup = {}
        for row in self.attribution:
            key = (int(row["history"]), int(row["seed"]), int(row["year"]))
            if key in self.attr_lookup:
                raise ValueError(f"Duplicate attribution cell: {key}")
            self.attr_lookup[key] = row

    def group(self, method, history, year):
        result = []
        for seed in (0, 1, 2):
            key = (method, history, seed, year)
            if key not in self.lookup:
                raise ValueError(f"Missing seed for publication figure: {key}")
            result.append(self.lookup[key])
        return result

    def effect(self, method, history, year, metric):
        key = (method, history, year, metric)
        row = self.effect_lookup.get(key)
        if row is None:
            row = self.effect_lookup.get((method, history, year, f"{metric}_delta"))
        if row is None:
            raise ValueError(f"Missing paired effect: {key}")
        values = [r[f"{metric}_delta"] for r in self.group(method, history, year)]
        if int(row["n"]) != 3:
            raise ValueError(f"Publication figure requires n=3: {key}")
        mean, sd = float(row["mean_delta"]), float(row["sd_delta"])
        if abs(mean - self.mean(values)) > 1e-8 or abs(sd - self.stdev(values)) > 1e-8:
            raise ValueError(f"CSV mean/sample SD does not match paired seed values: {key}")
        return mean * 100, sd * 100

    @staticmethod
    def axes_style(ax, direction="y"):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis=direction, color="#dddddd", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=0.65)

    def save(self, fig, stem, caption, sources, kind="experimental-results", **extra):
        # Avoid bbox_inches='tight': it changes the physical publication width
        # and silently invalidates the promised final font size after scaling.
        paths = {}
        for extension in ("pdf", "svg", "png"):
            path = self.output / f"{stem}.{extension}"
            fig.savefig(path, format=extension, dpi=300, facecolor="white")
            paths[extension] = str(path.relative_to(self.root))
        record = {
            "stem": stem, "type": kind, "caption": caption,
            "files": paths, "width_inches": float(fig.get_figwidth()),
            "height_inches": float(fig.get_figheight()),
            "intended_width": "double-column", "minimum_font_pt": 8,
            "font_family": "DejaVu Serif", "png_dpi": 300,
            "data_sources": sources,
            "uncertainty": "Sample SD across three paired seeds (ddof=1), not a confidence interval",
            "units": "AP percentage points: 100 times the raw AP difference",
            "palette": "Okabe-Ito categorical; blue-white-red continuous, with numeric labels",
            **extra,
        }
        self.manifest.append(record)
        self.plt.close(fig)

    def protocol(self):
        fig, ax = self.plt.subplots(figsize=(7, 4.25))
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.015, top=0.98)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")

        def box(x, y, w, h, text, face="#f4f4f4", edge="#777777", fontsize=9):
            patch = self.FancyBboxPatch((x, y), w, h,
                boxstyle="round,pad=0.035,rounding_size=0.07", linewidth=0.8,
                facecolor=face, edgecolor=edge)
            ax.add_patch(patch)
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

        ax.text(0.2, 5.78, "(a) Matched forecasting and continuation protocol", fontsize=10, weight="bold")
        box(0.2, 4.55, 2.65, 0.9, "2016–2020\nfoundation / training data")
        box(3.55, 4.55, 2.5, 0.9, "2021 selection\nthree paired seeds")
        box(6.8, 4.55, 2.95, 0.9, "2022 / 2023\nhistorical evaluation")
        for start, end in ((2.87, 3.51), (6.07, 6.76)):
            ax.annotate("", xy=(end, 5), xytext=(start, 5),
                        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#555555"})
        ax.text(5, 4.2, "T1: Res18 U-Net, 40 channels   |   T5: Res18 U-TAE, 33 channels", ha="center", fontsize=9)
        ax.text(5, 3.88, "Shared initialization per setting · 3,000 optimizer steps per model · batch 64", ha="center", fontsize=8.5)
        ax.text(0.2, 3.45, "(b) Current mainline composition by known evaluation condition", fontsize=10, weight="bold")
        columns = [
            ("M00", "Complete input", "Fresh ERM", "ERM"),
            ("M01", "Fire channel missing", "X22: cosine ERM", "X22"),
            ("M06", "25% spatial block", "X17: 25% expert", "X17_ROUTE"),
            ("M07", "50% spatial block", "X17: 50% expert", "X17_ROUTE"),
        ]
        for index, (scenario, condition, model, method) in enumerate(columns):
            x = 0.2 + index * 2.45
            box(x, 2.22, 2.2, 0.85, f"{scenario}\n{condition}", fontsize=8.5)
            ax.annotate("", xy=(x + 1.1, 1.77), xytext=(x + 1.1, 2.19),
                        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": PALETTE[method]})
            box(x, 0.98, 2.2, 0.73, model, face="white", edge=PALETTE[method], fontsize=8.5)
        ax.text(5, 0.56, "Existing evidence: composition of scenario-level evaluation summaries", ha="center", fontsize=9, weight="bold")
        ax.text(5, 0.18, "No learned or deployed router is evaluated; historical test years have already been inspected.", ha="center", fontsize=8)
        self.save(fig, "fig01_protocol_and_composition",
            "The current system assigns a fixed model to each controlled observation condition. "
            "T1 and T5 are distinct forecasting configurations, differing in architecture and input channels; "
            "their difference does not isolate history length. Each component continues from the corresponding shared "
            "initialization for 3,000 optimizer steps at batch size 64, with seeds 0, 1 and 2. "
            "M00 is complete input, M01 removes the fire channel, and M06/M07 apply 25%/50% spatial block missingness; "
            "the evidence composes scenario-level summaries and does not evaluate a learned or deployed router.",
            ["data/metrics_by_seed.csv", "tables/costs.csv"], kind="solution-overview",
            uncertainty="Not applicable: conceptual protocol schematic", units="Not applicable",
            limitations=["2021 is selection data; 2022/2023 are already-inspected historical test years",
                         "ERM-clean mainline differs from the historical X22-clean composition"])

    def primary_gain(self):
        methods = ("X22", "X22_X14_ERM_CLEAN", "X22_X17_ERM_CLEAN")
        fig, axes = self.plt.subplots(1, 2, figsize=(7, 3.5), sharey=True)
        fig.subplots_adjust(left=0.09, right=0.985, bottom=0.225, top=0.77, wspace=0.16)
        extrema = [0.0]
        for panel, (history, ax) in enumerate(zip((1, 5), axes)):
            for mi, method in enumerate(methods):
                offset = (mi - 1) * 0.24
                for yi, year in enumerate((2021, 2022, 2023)):
                    values = [r["primary_delta"] * 100 for r in self.group(method, history, year)]
                    center, sd = self.effect(method, history, year, "primary")
                    extrema.extend(values + [center - sd, center + sd])
                    ax.scatter([yi + offset + j for j in (-0.048, 0, 0.048)], values,
                               s=16, marker=MARKERS[method], facecolors="white",
                               edgecolors=PALETTE[method], linewidths=0.85, zorder=3)
                    ax.errorbar(yi + offset, center, yerr=sd, fmt=MARKERS[method],
                                color=PALETTE[method], markersize=5.1, capsize=3,
                                elinewidth=1.2, markeredgecolor="white", markeredgewidth=0.4,
                                zorder=4)
            ax.axhline(0, color="#444444", linestyle="--", linewidth=0.85)
            ax.set_xlim(-0.5, 2.5)
            ax.set_xticks([0, 1, 2], ["2021\nselection", "2022\nhistorical", "2023\nhistorical"])
            ax.set_title(f"({'ab'[panel]}) T{history}: " + ("Res18 U-Net" if history == 1 else "Res18 U-TAE"), fontsize=10, pad=8)
            self.axes_style(ax)
        span = max(extrema) - min(extrema)
        axes[0].set_ylim(min(extrema) - max(0.12, span * 0.08), max(extrema) + max(0.12, span * 0.08))
        axes[0].set_ylabel("Missing-observation AP gain (pp)")
        handles = [self.Line2D([], [], color=PALETTE[m], marker=MARKERS[m],
                    linestyle="none", markersize=5.5, label=METHOD_LABELS[m]) for m in methods]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.975),
                   ncol=3, frameon=False, fontsize=8.5, columnspacing=1.3, handletextpad=0.45)
        fig.text(0.54, 0.88, "Hollow markers: individual seeds   |   Filled markers and bars: mean ± sample SD", ha="center", fontsize=8)
        fig.text(0.54, 0.045, "Paired differences from the same-history, same-year, same-seed ERM; n = 3 per condition.", ha="center", fontsize=8)
        self.save(fig, "fig02_primary_gain",
            "Matched controls distinguish the gains of cosine continuation from those of the composed systems. "
            "Each point summarizes the mean AP over M01, M06 and M07, expressed as a paired difference from fresh ERM "
            "in AP percentage points (pp). Hollow markers are individual seeds; filled markers and error bars are "
            "the mean and one sample standard deviation across three seeds, not confidence intervals. "
            "X22+X14 and X22+X17 use ERM on complete inputs; 2021 is selection data and 2022/2023 are historical test data.",
            ["data/metrics_by_seed.csv", "tables/paired_effects.csv"])

    def severity_attribution(self):
        fig, ax = self.plt.subplots(figsize=(7, 3.65))
        fig.subplots_adjust(left=0.225, right=0.975, bottom=0.175, top=0.82)
        positions = (6, 5, 4, 2.8, 1.8, 0.8)
        all_values = [0.0]
        for (history, year), y in zip(CELLS, positions):
            values = []
            for seed in (0, 1, 2):
                key = (history, seed, year)
                if key not in self.attr_lookup:
                    raise ValueError(f"Missing X17-X14 attribution cell: {key}")
                values.append(float(self.attr_lookup[key]["block_delta"]) * 100)
                expected = (self.lookup[("X17_ROUTE", *key)]["block"]
                            - self.lookup[("X14_ROUTE", *key)]["block"]) * 100
                if abs(values[-1] - expected) > 1e-6:
                    raise ValueError(f"Attribution table and per-seed metrics disagree: {key}")
            center, sd = self.mean(values), self.stdev(values)
            all_values.extend(values + [center - sd, center + sd])
            color = "#0072B2" if center >= 0 else "#D55E00"
            marker = "o" if history == 1 else "s"
            ax.scatter(values, [y + j for j in (-0.13, 0, 0.13)], marker=marker,
                       s=18, facecolors="white", edgecolors=color, linewidths=0.9, zorder=3)
            ax.errorbar(center, y, xerr=sd, fmt=marker, color=color, capsize=3,
                        markersize=5.8, elinewidth=1.25, markeredgecolor="white", markeredgewidth=0.4, zorder=4)
        ax.axvline(0, color="#444444", linestyle="--", linewidth=0.9)
        ax.axhline(3.4, color="#cccccc", linewidth=0.65)
        ax.set_yticks(positions, [f"T{h} | {y}" + (" (selection)" if y == 2021 else " (historical)") for h, y in CELLS])
        limit = max(abs(v) for v in all_values) * 1.15
        ax.set_xlim(-max(limit, 0.05), max(limit, 0.05))
        ax.set_ylim(0.25, 6.55)
        ax.set_xlabel("Block-missing AP: X17 − mixed-severity X14 (pp)")
        self.axes_style(ax, "x")
        ax.tick_params(axis="y", length=0)
        fig.text(0.22, 0.93, "Negative: mixed expert better", color="#D55E00", fontsize=9)
        fig.text(0.65, 0.93, "Positive: fixed experts better", color="#0072B2", fontsize=9)
        fig.text(0.60, 0.865, "Individual seeds + mean ± sample SD; circles T1, squares T5", ha="center", fontsize=8)
        self.save(fig, "fig03_severity_attribution",
            "Severity-specific experts do not uniformly improve on a mixed-severity expert. "
            "The horizontal axis is the paired AP difference between X17 and X14, averaged over M06 and M07, "
            "in percentage points. Each row shows three seed differences and their mean ± one sample SD; "
            "negative values favor X14, and the zero-centered axis preserves unsuccessful cells. "
            "Circles denote T1 and squares T5; 2021 is selection data and 2022/2023 are already-inspected historical tests.",
            ["tables/attribution.csv", "data/metrics_by_seed.csv"])

    def scenario_effects(self):
        methods = ("X22", "X14_ROUTE", "X17_ROUTE", "X22_X17_ERM_CLEAN")
        matrices = {}
        for method in methods:
            matrix = []
            for history, year in CELLS:
                matrix.append([self.mean([
                    (self.lookup[(method, history, seed, year)][metric]
                     - self.lookup[("ERM", history, seed, year)][metric]) * 100
                    for seed in (0, 1, 2)]) for metric in SCENARIOS])
            matrices[method] = matrix
        bound = max(abs(v) for matrix in matrices.values() for row in matrix for v in row)
        bound = max(bound, 0.01)
        norm = self.TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
        fig = self.plt.figure(figsize=(7, 5.55))
        grid = fig.add_gridspec(2, 3, width_ratios=(1, 1, 0.055),
                               left=0.105, right=0.91, bottom=0.145, top=0.935,
                               hspace=0.48, wspace=0.50)
        cmap = self.mpl.colormaps["RdBu"]
        for panel, method in enumerate(methods):
            ax = fig.add_subplot(grid[panel // 2, panel % 2])
            matrix = matrices[method]
            # pcolormesh remains vector in PDF/SVG; imshow would embed a bitmap.
            mesh = ax.pcolormesh(matrix, cmap=cmap, norm=norm, edgecolors="white", linewidth=0.6)
            ax.invert_yaxis()
            ax.set_xticks([i + 0.5 for i in range(4)], ["M00\nclean", "M01\nfire", "M06\n25%", "M07\n50%"])
            ax.set_yticks([i + 0.5 for i in range(6)], [f"T{h} | {y}" for h, y in CELLS])
            ax.tick_params(length=0, pad=4, labelsize=8)
            ax.set_title(f"({'abcd'[panel]}) {METHOD_LABELS[method]}", fontsize=9, pad=9)
            ax.axhline(3, color="#333333", linewidth=0.8)
            for y, row in enumerate(matrix):
                for x, value in enumerate(row):
                    color = "white" if abs(value) > bound * 0.55 else "#1a1a1a"
                    label = "0.00" if abs(value) < 0.005 else f"{value:+.2f}"
                    ax.text(x + 0.5, y + 0.5, label, ha="center", va="center", fontsize=8, color=color)
            for spine in ax.spines.values():
                spine.set_visible(False)
        cax = fig.add_subplot(grid[:, 2])
        colorbar = fig.colorbar(mesh, cax=cax)
        colorbar.ax.tick_params(labelsize=8, length=3)
        colorbar.set_label("AP difference from matched ERM (pp)", fontsize=8, labelpad=7)
        colorbar.solids.set_rasterized(False)
        fig.text(0.49, 0.052, "Cell values are three-seed means; 2021 selection, 2022/2023 historical tests.", ha="center", fontsize=8)
        fig.text(0.49, 0.019, "Zero clean-input change in routed systems is inherited from ERM by construction.", ha="center", fontsize=8)
        self.save(fig, "fig04_scenario_effects",
            "Scenario-level differences show which components supply each system gain. "
            "Cells report mean paired AP differences from same-configuration, same-year, same-seed ERM over three seeds, "
            "in percentage points; a common zero-centered color scale and numeric annotations show both gains and losses. "
            "M00 is complete input, M01 removes the fire channel, and M06/M07 remove 25%/50% spatial blocks. "
            "Routed systems copy ERM on M00, so a zero clean-input difference is a composition identity rather than "
            "evidence that a newly trained model preserves clean-input accuracy.",
            ["data/metrics_by_seed.csv"], uncertainty="Three-seed means; per-seed values and sample SD are supplied in companion CSV tables")

    def comparator_and_budget(self):
        fig, axes = self.plt.subplots(1, 2, figsize=(7, 4.1))
        fig.subplots_adjust(left=0.09, right=0.985, bottom=0.265, top=0.765, wspace=0.41)
        ax = axes[0]
        offsets = {
            "ERM": (5, -13), "X22": (7, -1), "X14_ROUTE": (6, -13),
            "X17_ROUTE": (7, -13), "X22_X14_ERM_CLEAN": (-9, 11),
            "X22_X17_ERM_CLEAN": (-5, 12), "X22_X17_HISTORICAL": (-8, -16),
        }
        # Concise integer labels avoid crossing labels on nearby effects.
        order = tuple(METHOD_LABELS)
        available = {}
        for row in self.costs:
            if row["method"] not in order:
                continue
            method = row["method"]
            if method in available:
                raise ValueError(f"Duplicate cost row: {method}")
            available[method] = row
        maxima = [0.0]
        for index, method in enumerate(order, 1):
            if method not in available:
                raise ValueError(f"Missing component budget: {method}")
            row = available[method]
            models, steps = int(row["models"]), int(row["continuation_steps"])
            if models < 1 or steps != models * 3000:
                raise ValueError(f"Symbolic budget must be retained model count × 3000: {row}")
            effect = float(row["primary_delta_mean"]) * 100
            expected = self.mean(r["primary_delta"] for r in self.rows if r["method"] == method) * 100
            if abs(effect - expected) > 1e-6:
                raise ValueError(f"Cost effect and seed metrics disagree: {method}")
            maxima.append(effect)
            ax.scatter(steps / 1000, effect, marker=MARKERS[method], s=43,
                       color=PALETTE[method], edgecolors="white", linewidths=0.45, zorder=4)
            dx, dy = offsets[method]
            ax.annotate(str(index), (steps / 1000, effect), xytext=(dx, dy),
                        textcoords="offset points", fontsize=8.5, ha="right" if dx < 0 else "left")
        ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
        ax.set_xticks((3, 6, 9, 12), ("3k\n1 model", "6k\n2 models", "9k\n3 models", "12k\n4 models"))
        ax.set_xlim(2, 13)
        span = max(maxima) - min(maxima)
        ax.set_ylim(min(maxima) - max(0.2, span * 0.2), max(maxima) + max(0.25, span * 0.25))
        ax.set_ylabel("Mean primary AP gain (pp)")
        ax.set_xlabel("Total continuation optimizer steps", labelpad=7)
        ax.set_title("(a) Symbolic component budget", fontsize=9.5, pad=9)
        self.axes_style(ax)
        ax = axes[1]
        variants = ("X22_X17_ERM_CLEAN", "X22_X17_HISTORICAL")
        metrics = ("primary", "block", "clean")
        for mi, method in enumerate(variants):
            for index, metric in enumerate(metrics):
                values = [r[f"{metric}_delta"] * 100 for r in self.rows if r["method"] == method]
                center = self.mean(values)
                # Configuration/year differences are heterogeneous, not 18
                # independent seeds. Do not draw an unjustified pooled CI/SD.
                ax.scatter(index + (mi - 0.5) * 0.18, center, color=PALETTE[method],
                           marker=MARKERS[method], s=45, edgecolors="white", linewidths=0.4, zorder=3)
        ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
        ax.set_xlim(-0.5, 2.5)
        ax.set_xticks((0, 1, 2), ("Primary", "Block", "Clean"))
        ax.set_ylabel("Mean AP gain vs. ERM (pp)")
        ax.set_title("(b) Clean-input role changes", fontsize=9.5, pad=9)
        self.axes_style(ax)
        fig.legend(handles=[self.Line2D([], [], color=PALETTE[m], marker=MARKERS[m], linestyle="none", markersize=5,
                                       label=f"{i}. {METHOD_LABELS[m]}") for i, m in enumerate(order, 1)],
                   loc="upper center", bbox_to_anchor=(0.51, 1.0), frameon=False, ncol=3,
                   fontsize=8, columnspacing=1.0, handletextpad=0.35)
        fig.text(0.52, 0.084, "Budget counts retained components × 3,000 steps; it is not measured wall time or FLOPs.", ha="center", fontsize=8)
        fig.text(0.52, 0.041, "Shared foundation training, exploration, evaluation and deployment overhead are excluded.", ha="center", fontsize=8)
        self.save(fig, "fig05_comparator_and_budget",
            "The composed systems incur multiple continuation budgets, and their clean-input role must be stated explicitly. "
            "Panel (a) plots mean primary AP gain over the six configuration/year cells against the sum of 3,000-step "
            "continuations for retained component models; this accounting excludes shared foundation training, "
            "exploration, evaluation and deployment overhead and does not establish measured compute efficiency. "
            "Panel (b) contrasts the current four-model ERM-clean composition with the historical three-model X22-clean "
            "composition: primary and block roles are identical, while complete-input predictions come from different models. "
            "Values average three paired seeds within each of T1/T5 × 2021/2022/2023; heterogeneous cells are not treated "
            "as independent replicates for a pooled confidence interval.",
            ["tables/costs.csv", "data/metrics_by_seed.csv"],
            uncertainty="Descriptive means over six equally weighted configuration/year cells; no pooled uncertainty interval",
            limitations=["Symbolic continuation steps only; no measured wall-time, FLOP or inference-cost claim",
                         "Historical composition uses X22 on clean inputs; current composition uses ERM"])

    def historical_attribution(self):
        source = self.root / "tables/historical_chain_effects.csv"
        if not source.exists():
            return
        rows = _csv(source)
        lookup = {}
        for row in rows:
            key = (row["comparison"], int(row["year"]), row["metric"])
            if key in lookup:
                raise ValueError(f"Duplicate historical attribution: {key}")
            lookup[key] = float(row["delta"]) * 100
        comparisons = ("D1_KL_vs_ERM", "D12_vs_D2_STD", "D12_vs_ERM")
        labels = ("D1 KL − ERM", "D12 − D2 standard", "D12 − ERM")
        colors = {"primary": "#333333", "block": "#0072B2", "clean": "#D55E00"}
        markers = {"primary": "o", "block": "s", "clean": "^"}
        fig, axes = self.plt.subplots(1, 3, figsize=(7, 3.6), sharey=True)
        fig.subplots_adjust(left=0.09, right=0.985, bottom=0.25, top=0.77, wspace=0.15)
        extrema = [0.0]
        for pi, (comparison, label, ax) in enumerate(zip(comparisons, labels, axes)):
            for mi, metric in enumerate(("primary", "block", "clean")):
                values = []
                for year in (2021, 2022, 2023):
                    key = (comparison, year, metric)
                    if key not in lookup:
                        raise ValueError(f"Missing historical attribution: {key}")
                    values.append(lookup[key])
                extrema.extend(values)
                ax.scatter([y + (mi - 1) * 0.2 for y in range(3)], values,
                           marker=markers[metric], color=colors[metric], s=27,
                           edgecolors="white", linewidths=0.4, zorder=3)
            ax.axhline(0, color="#555555", linestyle="--", linewidth=0.8)
            ax.set_xticks((0, 1, 2), ("2021", "2022", "2023"))
            ax.set_xlim(-0.5, 2.5)
            ax.set_title(f"({'abc'[pi]}) {label}", fontsize=9, pad=9)
            ax.set_xlabel("Evaluation year")
            self.axes_style(ax)
        span = max(extrema) - min(extrema)
        axes[0].set_ylim(min(extrema) - max(0.1, span * 0.08), max(extrema) + max(0.1, span * 0.08))
        axes[0].set_ylabel("AP difference from named control (pp)")
        fig.legend(handles=[self.Line2D([], [], color=colors[m], marker=markers[m],
                    linestyle="none", markersize=5, label=m.capitalize()) for m in ("primary", "block", "clean")],
                   loc="upper center", bbox_to_anchor=(0.52, 0.995), ncol=3, frameon=False)
        fig.text(0.52, 0.885, "Archived T1 experiment chain · one run per method · no uncertainty estimate", ha="center", fontsize=8.5)
        fig.text(0.52, 0.104, "Historical protocol: these effects are not pooled with the current three-seed mainline.", ha="center", fontsize=8)
        fig.text(0.52, 0.050, "2021 is selection data; 2022/2023 have already been inspected. Negative values favor the control.", ha="center", fontsize=8)
        self.save(fig, "fig06_historical_attribution",
            "The archived T1 chain separates the effects of a consistency term and the D12 combination against named controls. "
            "Primary, block and clean AP differences are shown for D1 KL versus ERM, D12 versus D2 standard, and D12 versus ERM, "
            "in percentage points. Each point comes from one historical run, so no seed variance or confidence interval is implied. "
            "These historical-protocol comparisons are retained as supporting evidence and are not pooled with the current "
            "three-seed X22+X17 mainline; negative values and clean-input costs remain visible.",
            ["tables/historical_chain_effects.csv"],
            uncertainty="Single historical run per method; no estimate of run-to-run uncertainty",
            limitations=["Archived T1 protocol, not a matched comparison with current mainline",
                         "Single-run evidence does not confirm an independent contribution"])

    def render(self):
        style = {
            "font.family": "DejaVu Serif", "font.size": 9,
            "axes.labelsize": 9, "axes.titlesize": 10,
            "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5, "axes.linewidth": 0.7,
            "lines.linewidth": 1.2, "savefig.dpi": 300,
            "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
            "mathtext.fontset": "dejavuserif", "figure.facecolor": "white",
            "savefig.facecolor": "white", "axes.unicode_minus": True,
        }
        with self.plt.rc_context(style):
            self.protocol()
            self.primary_gain()
            self.severity_attribution()
            self.scenario_effects()
            self.comparator_and_budget()
            self.historical_attribution()
        (self.output / "figure_manifest.json").write_text(json.dumps(self.manifest, indent=2) + "\n", encoding="utf-8")
        return self.manifest


def render_all(package_root: Path) -> list[dict]:
    """Render five core figures plus optional history; return their manifests."""
    return _Renderer(Path(package_root)).render()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_root", type=Path, help="Package containing data/ and tables/")
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Figure generation requires a Slurm CPU allocation")
    figures = render_all(args.package_root)
    print(f"Rendered {len(figures)} publication figures in {args.package_root / 'figures'}")


if __name__ == "__main__":
    main()
