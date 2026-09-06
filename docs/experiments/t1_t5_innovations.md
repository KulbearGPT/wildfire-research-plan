# Cross-history innovation campaign — 2026-09-06

Origin: main `8c9c072`. User authorizes autonomous design, implementation,
cluster submission, repair, and iteration until three supported directions
are found. This document is the design, execution plan, and live evidence log.

## Frozen problem and evidence rule

WSTS+ next-calendar-day active-fire proxy under controlled missingness.
Train 2016–2020, select on 2021, report fixed 2022/2023 tests. T=1 uses
Res18-U-Net (40 channels); T=5 uses the previously reproduced Res18-UTAE
(33 selected channels per day). Both evaluate identical event/target dates
using history adjustment six and M00/M01/M06/M07. T changes architecture and
feature selection too: this demonstrates transfer across existing settings,
not a pure history-length causal effect.

Fresh matched controls and candidates start from corrected B3 (T1) or B5
(T5), run 3000 AdamW steps, lr=0.001, effective batch 64, identical seed,
augmentation, normalization, sampling, and loss. Final-step checkpoint only.
First screen seed=0; confirmation seeds=1,2 for candidates passing both T.
Primary metric: mean AP over M01/M06/M07. Also report M06/M07 block mean,
M00, individual corrupted AP, runtime and parameter count. A candidate counts
only when primary improvement is positive in both T settings across the
confirmation seed mean and each fixed test year, with no M00 regression
greater than 0.01. Seed-0 screening signal: primary >=0.005 in both T and
M00 guardrail. These are practical thresholds, not significance claims.
Seek >=0.02 average improvement for at least one direction, preserving the
user's earlier magnitude requirement. Do not count three hyperparameters of
one method as three directions. Keep all negative results. No selection using
2022/2023; if test evidence fails, mark failure and disclose adaptive reuse
before any later campaign rather than pretending a fresh unseen test.

## First candidates and falsifiable mechanisms

- **X1 context transport:** at encoder skip levels, replace missing-region
  features by learned residuals from normalized valid-region context pooled
  at multiple spatial resolutions. Zero-initialized residuals preserve the
  initial predictor. Spatial mask comes only from the controlled intervention.
  Distinct from rejected D2-RNC: retain standard convolutions; transport
  multi-scale feature content across holes, not input convolution rescaling.
- **X2 fire-weighted feature distillation:** frozen initial model sees clean
  training inputs; student matches normalized deep features primarily inside
  missing regions and near next-day positive targets. Auxiliary weight 0.05.
  Distinct from D1 output KL and failed ranking losses: intermediate spatial
  representation targets, with no extra inference parameters.
- **X3 latent transition head:** decoder predicts previous-day occupancy,
  conditional continuation, and new activity. Forecast marginalizes uncertain
  occupancy in missing regions, clamps occupancy to observed latest fire where
  valid, and receives latest-fire auxiliary supervision from clean train
  inputs. This is joint forecasting, not a separate frozen reconstruction
  pipeline. Conditional heads initialize to the original forecast.

Prior art: [partial convolutions](https://arxiv.org/abs/1804.07723),
[privileged multimodal segmentation](https://pubmed.ncbi.nlm.nih.gov/34633927/),
[DIS2 remote-sensing distillation](https://openaccess.thecvf.com/content/WACV2026W/CV4EO/papers/Kieu_DIS2_Disentanglement_Meets_Distillation_with_Classwise_Attention_for_Robust_Remote_WACVW_2026_paper.pdf),
[wildfire reconstruction then prediction](https://arxiv.org/abs/2603.09042).
These establish related mechanisms, not novelty or quantitative support for
our implementations. A publication claim needs precise comparison to them.

## Minimal implementation plan

- [ ] Add a separate `reproductions/cross_history/` package for the campaign.
  Reuse corrected resolver, normalization, raw evaluation corruptions, exact
  AP code and archived B5 checkpoint (explicit relocated path).
- [ ] Implement shared T1/T5 encoder-decoder access and the three methods.
  Keep clean training targets only in the training loss, absent at inference.
- [ ] Add one Slurm runner with train/eval modes, job-specific output path,
  committed source snapshot, fixed seed metadata, checkpoint and results.
- [ ] Run a GPU one-batch check inside Slurm for each method/history, including
  original-vs-control output agreement and backward pass. No global tests.
- [ ] Submit eight seed-0 experiments (control + three methods, both T).
  Inspect resource availability; prefer 10/20GB slices, at most 2x overflow
  unless a blocking prerequisite requires more. Revisit jobs pending >10min.
- [ ] Compare against fresh controls, diagnose failures, register any changed
  hypothesis before further experiments; confirm and test only passing recipes.
- [ ] Commit all progress and report three only when evidence satisfies above.

## Status

No new quantitative result yet. Historical B3/B5 and D2/D13 controls are
starting evidence, not proof of any new direction.

Implementation: `357e475`, numerical mixture fix `8c654e2`. Shared raw
evaluation permits T5 only by an explicit opt-in; mainline T1 default stays.
Physical batch 16, accumulation 4 applies to every new control/candidate.
Existing full-batch D2/D13 numbers are context, not the matched comparison.

GPU smoke jobs: T1 `21210705`, T5 `21210706`, each checks control/context/
distill/transition in sequence on a 20GB H100 slice, 4 CPUs, 32GB host memory.
Earlier 10GB requests `21210683/21210684` were cancelled while pending because
no start estimate was available; 20GB had an approximately 11-minute estimate.
No model or dataset computation has been run on the login node.

Next action: inspect those exact job IDs and their `slurm-X-smoke-T*-JOB.out`
logs under the persistent runs root. Fix failures before seed-0 training.
After successful smoke, submit eight 3000-step jobs (four methods x two T)
with the same batch/worker settings; allow longer T5 walltime based on smoke
timing and choose available GPU slices under the resource policy.
