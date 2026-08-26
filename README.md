# Reliable Next-Day Wildfire Spread Forecasting

This repository is the research record and prototype workspace for reliable
next-day wildfire spread forecasting under asynchronous or missing
observations. It hosts the public research-plan website, audited data-contract
code, completed baseline-reproduction controls, and the small amount of cluster
infrastructure needed to run the next experiments.

## Current status

- The eight-year WSTS+ audit covers 999 valid event-level HDF5 files and freezes
  the temporal split to 2016–2020 train, 2021 validation, and 2022–2023 test.
- The fixed no-fire and latest-mask persistence rules have been evaluated under
  that split. See [`docs/experiments/phase0.md`](docs/experiments/phase0.md).
- The official Res18-U-Net, `T=1`, Fold-2 training path has been executed, and
  all twelve official released weights have been evaluated independently. See
  [`docs/experiments/res18_unet_t1_reproduction.md`](docs/experiments/res18_unet_t1_reproduction.md).
- The active-fixed 999-event WSTS+ tree is prepared on Nibi. All six C00/C02
  seed-0/1/2 10,000-step clean runs and the 96-task M00--M07 formal
  missingness matrix have completed. Active-fire-history loss (M01) is the
  dominant diagnosed failure. See
  [`reproductions/wsts_fast_track/`](reproductions/wsts_fast_track/).
- P00 (`FireDrop-C00`) remains the frozen legacy comparison checkpoint. P01's
  explicit active-fire-validity channel did not improve P00, while P02's
  structured BlockDrop exposed a useful spatial-missingness expert but harmed
  other regimes. P03 now routes frozen P00/P02 logits only inside known missing
  blocks: it preserved P00 and improved 2021 M06/M07, but the final held-out
  check was negative in 2022 and positive in 2023. P03 is therefore not
  promoted. P09 corrected an upstream multi-year indexing defect and applied
  year-corruption GroupDRO; it improved routed M06/M07 AP over P00 in 2021,
  2022, and 2023. P10 then held every training choice fixed and replaced
  GroupDRO with ordinary ERM. P09 and P10 were effectively tied across all
  three years, so the gain is attributed to corrected, year-balanced
  multi-year training rather than GroupDRO. P11 tried reusing P10 for complete
  active-fire-history loss, but regressed 2021 M01 and was rejected. P12 then
  replaced the accidentally disabled focal class weighting with the normalized
  positive-class weight while keeping P10's 3,000-step recipe fixed. It caused
  severe overprediction and regressed M01/M06/M07 on 2021, so it was
  rejected without opening 2022--2023. See
  [`docs/experiments/p00_p06_rapid_reliability.md`](docs/experiments/p00_p06_rapid_reliability.md).

## Scientific boundary

The target is a next-calendar-day active-fire proxy, not a complete fire
perimeter. The current public data contract supports prespecified controlled
missingness. It does not establish natural-missingness or
operational-deployment performance because acquisition time, availability time,
QA, coverage, and target-validity provenance are not all recoverable.

The twelve-fold publication supports released-weight executable
reproducibility. Its agreement with the paper reference does not prove
paper-table provenance identity. The local Windows/RTX 3090 evidence remains a
frozen lineage; future cluster runs begin a linked but separate lineage.

## Repository map

- [`index.html`](index.html) — public research plan and status.
- [`related-work/`](related-work/) — Lecture 02, research foundations and
  related work.
- [`baseline-reproduction/`](baseline-reproduction/) — Lecture 03, complete
  baseline-reproduction walkthrough.
- [`src/wildfire_phase0/`](src/wildfire_phase0/) — data audit, repair, split,
  rule-baseline, and reporting code.
- [`reproductions/wsts_res18_unet_t1/`](reproductions/wsts_res18_unet_t1/) —
  pinned official-code reproduction controls and manifests.
- [`reproductions/wsts_fast_track/`](reproductions/wsts_fast_track/) — frozen
  WSTS+ split and C00/C02 seed-0 screening controls.
- [`docs/experiments/`](docs/experiments/) — reviewed result summaries and
  claim boundaries.
- [`docs/research-roadmap.md`](docs/research-roadmap.md) — authoritative order
  of future experiments.
- [`docs/cluster-migration.md`](docs/cluster-migration.md) — Slurm migration and
  first-run guide.

## Fast local verification

Use a Python 3.13 environment that satisfies `pyproject.toml`:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Large datasets, weights, environments, caches, checkpoints, and run artifacts
are deliberately excluded from Git.

## Slurm cluster quick start

The current deployment target is **Alliance Nibi**. The cluster layer is a thin
research boundary, not a deployment system. Nibi supplies one ignored profile
containing the user's account, partition, module, and path values. The generic
scripts validate the profile and data manifest, print the exact `sbatch`
command in dry-run mode, and execute one scientific command once. Each job
performs a formal preflight before the scientific command and seals the
production-data, runtime, command, upstream-code, and applicable
official-weight identities into its run record. Follow
[`docs/cluster-migration.md`](docs/cluster-migration.md) after cloning the
reviewed migration commit.

## Rapid prototype outcome

P03 spatial expert routing completed in job `20458324`. It is a training-free
2021 engineering prototype: P00 is used everywhere except inside the known
M06/M07 missing block, where P02 is used. AP is 0.585322/0.299465/0.350878/
0.166982 on M00/M01/M06/M07. Relative to P00 this is exactly neutral on
M00/M01 and +0.0339/+0.0378 on M06/M07, making P03 the 2021-selected routed
candidate for the final held-out comparison reported below.

P04 tested whether P03 could be compressed to one P00 forward pass plus a
17-parameter `1x1` residual head trained for 1,000 steps. Job `20462018`
preserved M00/M01 exactly and improved P00 AP by 0.0151/0.0107 on M06/M07,
but remained 0.0188/0.0271 below P03. P04 is retained as the minimum-compute
matched control and is not promoted.

P05 expanded that residual head to a 145-parameter `3x3` convolution. Job
`20464221` preserved M00/M01 but reached only 0.327723/0.138767 AP on M06/M07,
slightly below P04. Kernel size is therefore not the limiting factor; P05 is
not promoted.

P06 adapted the final decoder block and head while sharing the rest of P00. Job
`20464396` reached 0.330240/0.141035 on 2021 M06/M07 and also failed to beat
P03. The spatial-router branch then closed. Final P03 jobs `20465333/20465334`
showed M06/M07 deltas of -0.0231/-0.0261 in 2022 and +0.0135/+0.0096 in 2023.
The P03 result is not cross-year robust. P09 subsequently produced positive
M06/M07 deltas in both 2022 and 2023, but it also exposed and corrected an
upstream multi-year sample resolver that had mapped pooled training indices to
the final year. The matched P10 ERM control retained the gain and was
effectively tied with P09 across 2021--2023. P10 is therefore the parsimonious
leading candidate, and the evidence does not support a GroupDRO contribution.
P11 preserved that block result but reduced M01 AP when it routed complete
active-fire loss to P10, so no P11 held-out evaluation was run. Held-out years
must not be used for further tuning. P12 properly enabled focal positive-class
weighting (`alpha=0.998688`) in an otherwise matched P10 run, but reduced M01
AP to 0.252888 and mean M06/M07 AP to 0.066132 through extreme overprediction.
The prevalence-derived focal weighting is rejected; no P12 held-out evaluation
was run.
