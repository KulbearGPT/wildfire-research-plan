> **学生实操：** [只从官方仓库开始的 Res18-U-Net 教程](docs/tutorials/res18-baseline-slurm.md) · [项目 corrected B0 教程](docs/tutorials/project-b0-slurm.md)。

> **教学接手入口（2026-09-11）：** [完整实验 Roadmap](docs/research-roadmap.md) · [Git 提交核对](docs/project-handoff-git-audit.md)。最新 T=1/T=5 与跨架构实现位于 `research/t1-t5-innovations` 分支；下方 README 正文保留早期 T=1 阶段说明，当前状态以 Roadmap 为准。

# Reliable T=1 Wildfire Spread Forecasting

This repository studies next-calendar-day wildfire active-fire forecasting
when satellite observations are incomplete. The active research scope is the
Res18-U-Net `T=1` setting on WSTS+. It retains the baseline chain and methods
that have quantitative support across 2021, 2022, and 2023.

## Current evidence

- The audited WSTS+ tree contains 999 event HDF5 files. The frozen split is
  2016--2020 train, 2021 validation, and 2022--2023 test.
- The official Res18-U-Net `T=1` training path and all twelve released weights
  have executable reproduction records.
- Corrected pooled-year indexing is used by every active learned experiment.
- FireDrop routing improves M01 AP by `+0.150109` on average across three
  years; incremental BlockDrop improves mean M06/M07 AP by `+0.027821`.
- D1 predictive consistency improves mean M01/M06/M07 AP over matched ERM by
  `+0.008970` across three years.
- D12 severity-adaptive reliability prompting improves the same joint metric
  over D1-ERM by `+0.020699`. Its prompt module improves mean M06/M07 AP over
  D2-STD by `+0.005764`; both comparisons are positive in every year.

The full AP table, matched comparisons, jobs, and claim boundary are in
[`docs/experiments/quantitative_reliability_ledger.md`](docs/experiments/quantitative_reliability_ledger.md).
Experiments removed from the active line are summarized once in
[`docs/experiments/rejected_experiments.md`](docs/experiments/rejected_experiments.md).

## Retained experiment chain

| ID | Role | Retained reason |
| --- | --- | --- |
| B0 | corrected clean C00 baseline | common T=1 reference |
| B2 | FireDrop specialist | reliable M01 improvement with observable routing |
| B3 | FireDrop + BlockDrop | reliable incremental spatial robustness |
| D1-ERM | paired supervised continuation | matched control for D1-KL and D12 total effect |
| D1-KL | predictive consistency | positive matched delta in all three years |
| D2-STD | standard single-corrupt-view continuation | matched module control for D12 |
| D12-SARP | severity-adaptive prompts | retained method contribution |

The runnable code is in
[`reproductions/wsts_fast_track/`](reproductions/wsts_fast_track/). Large
datasets, environments, checkpoints, and logs remain outside Git.

## Repository map

- [`src/wildfire_phase0/`](src/wildfire_phase0/) — data audit, repair, split,
  deterministic baselines, and reports.
- [`reproductions/wsts_res18_unet_t1/`](reproductions/wsts_res18_unet_t1/) —
  pinned official `T=1` reproduction controls.
- [`reproductions/wsts_fast_track/`](reproductions/wsts_fast_track/) — corrected
  B0/B2/B3, D1, D2-STD, and D12 code and Nibi runners.
- [`docs/experiments/`](docs/experiments/) — foundation reports, current
  quantitative results, rejected-results summary, and artifact manifest.
- [`docs/research-roadmap.md`](docs/research-roadmap.md) — next experiment and
  claim boundary.
- [`docs/cluster-migration.md`](docs/cluster-migration.md) — Alliance Nibi
  setup and lightweight Slurm workflow.
- [`related-work/`](related-work/) and [`baseline-reproduction/`](baseline-reproduction/)
  — research context and baseline walkthrough.

## Nibi execution

All training and model evaluation runs through Slurm. The login node is used
for source edits, small metadata reads, Git operations, queue inspection, and
submission. Before requesting a GPU, compare Nibi resource slices; if the
expected wait exceeds ten minutes, choose the faster slice when its allocation
is at most twice the minimum required resource.

The active runners are:

```text
run_corrected_baseline_on_nibi.sh                 B0/B2/B3
run_predictive_consistency_on_nibi.sh             D1-ERM/D1-KL
run_standard_reliability_control_on_nibi.sh       D2-STD
run_severity_adaptive_reliability_prompting_on_nibi.sh  D12-SARP
run_reliability_evaluation_on_nibi.sh             fixed 2022/2023 evaluation
run_d12_heldout_on_nibi.sh                        D12/D2 paired evaluation
```

## Recovery

The complete tree before research-line cleanup is preserved on branch
`archive/pre-t1-cleanup-2026-09-04` at commit `4b843ad`. Removed Nibi artifacts
are preserved at
`/project/6085198/kulbear/wildfire/archive/pre-t1-cleanup-2026-09-04/runs`.
The archive manifest is
[`docs/experiments/artifact-archive-manifest.tsv`](docs/experiments/artifact-archive-manifest.tsv).
