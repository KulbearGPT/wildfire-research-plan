# Wildfire forecasting: X22 + X17

The active research route is **X22 + X17 and its necessary controls**, selected
on 2026-09-22. We study next-day active-fire prediction under controlled missing
satellite observations. Other implementations remain available as explicitly
inactive experiments, diagnostics, or teaching references.

- **Code and scope:** [lifecycle map](docs/CODE_LIFECYCLE.md), [cross-history implementation](reproductions/cross_history/README.md).
- **Run the mainline:** [environment/data setup](docs/research/reproduce.md), [B3/B5 foundations](docs/research/baselines.md), [method recipes](docs/research/method-recipes.md).
- **Start learning:** [student onboarding](docs/START_HERE.md), [English course website](index.html), [independent official baseline tutorial](docs/tutorials/res18-baseline-slurm.md).
- **Develop with AI:** [project agreement](AGENTS.md), [development guide](docs/DEVELOPMENT.md).
- **Write the paper:** [materials, figures and build record](paper/README.md).
- **Audit and organize:** [reproducibility audit](docs/repository-audit-plan-20260929.md), [current commit-history review plan](docs/commit-history-audit-plan-20260929.md).

## Retained comparison

| Component | Role |
|---|---|
| Corrected B3 / B5 | Shared initialization and separate frozen references for canonical T1 / T5 |
| Fresh `control` continuation | Matched ERM comparator and M00 route component |
| X22 `cosine_erm` | M01 route component; cosine-schedule training change |
| X14 mixed `block_specialist` | Closest attribution control for fixed-severity specialization |
| X17 fixed 25% / 50% `block_specialist` | M06 / M07 route components |

The final route has three-seed evidence across 2021/2022/2023 in both existing
T1/T5 settings. Its mean primary AP gain is +0.014642 against fresh ERM and
+0.034527 against frozen B3/B5. Keep these comparators separate. X17's
closest-control attribution did not pass every required held-out comparison;
retaining the system does not establish an additional independent mechanism.
See the [original evidence](docs/experiments/t1_t5_innovations.md) and
[claim boundaries](docs/CODE_LIFECYCLE.md).

## Archive and provenance

[method-inventory.json](docs/research/method-inventory.json) distinguishes
historical evidence `status` from current `lifecycle`. The [positive-signal
archive](docs/research/positive-signals.md) includes isolated gains; the
[negative/invalid/unrun archive](docs/research/negative-results.md) preserves
failures without conflating their causes. X8, D12, RF/TD, belief-state methods,
restoration variants and alternative architectures are outside the active route.

Use the [mainline recipe](docs/research/method-recipes.md) and the narrow
`reproductions.cross_history.mainline` command for new work. The complete
[archive index](docs/archive/README.md) preserves the other directions and their
original recipes without mixing them into the active execution sequence.

Shared modules stay at their original import paths. Source comments and the
inventory's file map identify active support, mixed modules, references, and
archived implementations. Two one-off confirmation bundles with historical job
IDs are retired and refuse execution. Earlier README/mainline descriptions remain
in Git history; they are not the current scope.

Use `scripts/research/submit.sh` with a personal site configuration for the portable
research route. It snapshots committed HEAD. Computation and model execution run
through Slurm; see [development instructions](docs/DEVELOPMENT.md). This cleanup
preserves experiment settings and results and does not launch new experiments.
