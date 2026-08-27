# Observation-Consistent Fire Belief-State Prototype Design

**Date:** 2026-08-27  
**Status:** Approved design, implementation not started  
**Scope:** Six fast prototypes: three state-inference directions crossed with
`T=1` and `T=5`

## 1. Decision

The P00--P13 engineering branch is closed. Dropout, block corruption, expert
routing, GroupDRO, focal-alpha correction, and dedicated FireDrop fine-tuning
produced useful diagnostics but not a stable method contribution. The next
stage tests one research hypothesis:

> Infer an explicit, task-relevant current-fire belief state from the available
> context and history, preserve every valid active-fire observation exactly,
> and pass the corrected state to one shared frozen spread predictor.

Three state-proposal mechanisms will be developed in parallel:

- **A — recurrent filter:** a shared convolutional state updater unrolled once
  for `T=1` and five times for `T=5`;
- **B — temporal attention:** a lightweight pixelwise temporal cross-attention
  estimator with a prior token, so `T=1` remains a meaningful attention case;
- **C — reconstruction-first:** a TinyMaskUNet trained only to recover the
  current active-fire state before frozen forecasting.

A and B are candidate task-oriented methods. C is the direct matched
reconstruction baseline. Parallel development does not imply that all three
will be claimed as paper methods: only an A/B candidate that beats C under the
frozen screen can become the main method.

## 2. Research and novelty boundary

This is a **Technique Paper** direction, not a new-problem paper. Wildfire
forecasting under partial observability has already been formulated using a
two-stage probabilistic reconstruction-to-prediction pipeline in *Robust
Wildfire Forecasting under Partial Observability: From Reconstruction to
Prediction* (Yang et al., 2026,
<https://arxiv.org/abs/2603.09042>). FireEx already studies fixed
modality-specialized experts for next-day wildfire forecasting
(Andrianarivony and Akhloufi, 2026,
<https://doi.org/10.3390/rs18091416>). Timestamp-aware asynchronous fusion and
incomplete-modality reliability weighting also exist outside this task, for
example AnytimeFormer (<https://doi.org/10.1016/j.rse.2025.115120>) and SGMA
(<https://arxiv.org/abs/2603.02505>).

The project must therefore not claim novelty for missing-data reconstruction,
temporal attention, modality experts, timestamps, or partial observability
alone. The candidate contribution is the combination of:

1. a task-oriented active-fire belief state rather than full-input recovery;
2. hard observation consistency that changes only unavailable active-fire
   pixels;
3. one frozen downstream spread predictor shared across mechanisms and history
   lengths, isolating state inference from backbone and routing gains;
4. a matched comparison between recurrent, attention, and reconstruction-first
   state proposals under the same data and budget.

This contribution remains provisional until A or B beats C and the frozen
engineering controls. If neither does, the mechanism is rejected rather than
tuned against held-out years.

## 3. Goals and non-goals

### Goals

- Test whether explicit belief-state inference improves M01, M06, and M07.
- Test `T=1` and `T=5` simultaneously under one mathematical state contract.
- Preserve M00 exactly by construction, not through a learned penalty.
- Separate task-oriented state learning from reconstruction fidelity.
- Obtain decisive 2021 results with six single-seed, 3,000-step jobs.

### Non-goals

- Natural-missingness, cloud, QA, acquisition-time, availability-time, or
  operational-deployment claims.
- Reconstruction of every remote-sensing modality.
- Detection-hour distribution modeling in the first prototype.
- Probabilistic latent-variable training, KL objectives, uncertainty sampling,
  contrastive learning, physics losses, or calibration redesign.
- Hyperparameter sweeps, multi-seed confirmation, new backbones, global test
  suites, infrastructure refactors, or production engineering.
- Any tuning based on 2022--2023.

## 4. Common data and state contract

### 4.1 Population and temporal alignment

The existing WSTS+ active-fixed tree and event-disjoint split remain frozen:

- train: 2016--2020;
- selection: 2021;
- fixed reporting only: 2022--2023.

Both history lengths predict the same sixth-day target population by retaining
the existing five-day test adjustment:

- `T=1` observes day 5 and predicts day 6;
- `T=5` observes days 1--5 and predicts day 6.

The corrected multi-year index resolver and inverse-year sampler from P10/P13
are mandatory.

### 4.2 Processed tensors

All six prototypes use the full processed feature layout instead of switching
`T=5` to the 33-channel C02 feature subset:

- features: `X` with shape `[B,T,40,H,W]`;
- clean binary active fire: `A*` with shape `[B,T,1,H,W]`;
- validity: `R` with shape `[B,T,1,H,W]`, where one means observed and zero
  means unavailable;
- next-day target: `Y` with shape `[B,H,W]`.

Processed channel 38 is active-fire detection hour and channel 39 is binary
active fire. State-proposal context uses channels `0:33`; target-period
forecast-weather channels `33:38` are excluded from current-state inference.
The frozen spread predictor still receives all latest-day channels.

The prototype estimates only binary current fire. At the latest time step,
channel 39 is replaced by the corrected belief state. Channel 38 remains at
the corruption-defined value. This is compatible with P00's FireDrop training
and prevents detection-hour reconstruction from becoming a second research
question.

### 4.3 Observation correction

Every direction implements the same correction:

\[
S_T = R_T \odot A_T + (1-R_T) \odot \hat A_T,
\]

where `A_T` is the available binary observation and `hat A_T` is the inferred
probability. On M00, `R_T=1`, so the tensor given to the downstream model is
bit-identical to the original P00 input and its logits must match P00 exactly.

All directions use the same frozen P00 Res18-U-Net as the spread predictor.
The existing C02 Res18-UTAE remains an external temporal reference, not the
`T=5` implementation, because changing the backbone and feature set would
confound the history comparison.

## 5. Direction A: recurrent belief-state filter

The recurrent operator implements

\[
P_t = U_\theta(C_t,S_{t-1},R_t), \qquad
S_t = R_t \odot A_t + (1-R_t) \odot \sigma(P_t),
\]

where `C_t=X_t[:,0:33]`. `S_0` is zero. The same `U_theta` parameters are
unrolled once or five times.

`U_theta` is a three-level tiny U-Net with GroupNorm and SiLU, widths
`16/32/64/96`, bilinear upsampling, and one output logit. Its expected size is
approximately 0.3--0.5 million trainable parameters. No ConvLSTM, second
temporal backbone, stochastic state, or auxiliary decoder is included.

The latest corrected state replaces channel 39 in the latest-day full feature
tensor. The frozen P00 forward pass produces the next-day logits. Gradients
flow through P00 to the state updater, but no P00 parameter is updated.

## 6. Direction B: temporal-attention belief state

Direction B uses a lightweight pixelwise temporal cross-attention estimator:

- context projection: current non-fire context plus a fixed relative lag to a
  16-channel token;
- fire-evidence projection: detection hour, binary fire, and validity to a
  16-channel token, multiplied by validity before fusion;
- prior token: a projection of latest valid non-fire context that is present
  for both `T=1` and `T=5`;
- temporal attention: latest prior queries the `T` evidence tokens plus the
  prior token at each pixel;
- one shared depthwise `3x3` spatial layer and a `3x3` one-channel state head.

There are no learned Q/K/V projections or transformer blocks. The estimated
size is approximately 1.5--2 thousand trainable parameters and is independent
of `T`. The extra prior token means `T=1` attends over two tokens rather than
degenerating into a one-element softmax. Active-fire values with `R=0` must not
affect the output, and attention weights must be finite and sum to one.

The common observation correction and frozen P00 interface are then applied.

## 7. Direction C: reconstruction-first baseline

Direction C is deliberately not trained with the forecast target. A
three-level TinyMaskUNet consumes the temporal stack of context, corrupted
active-fire evidence, and validity, and reconstructs the latest binary
active-fire mask. Widths are `16/32/64/96`; expected size is approximately
0.5 million parameters for either history length.

The imputer is trained only against the clean current active-fire state. Its
output passes through the common observation correction and frozen P00. This
keeps C a true reconstruction-first control: if A/B improve beyond C, the
difference can be attributed to task-oriented forecast training rather than
generic inpainting capacity.

P07/P08 are not substitutes for C. They learned decoder-feature or logit
residuals and did not reconstruct an explicit active-fire input state.

## 8. Training protocol

Each method/history pair is one independent Nibi job:

- seed: 0;
- steps: 3,000;
- effective batch size: 64; if a `T=5` job runs out of memory, resubmit only
  that configuration with physical batch size 32 and two-step gradient
  accumulation, recording the change as a runtime constraint;
- optimizer: AdamW;
- learning rate: `1e-3` for the newly initialized small modules;
- frozen parameters: all P00 parameters;
- sampling: corrected resolver plus inverse-year balanced sampler;
- corruption: uniform choice among M01, M06, and M07;
- evaluation after training: 2021 M00/M01/M06/M07 in the same job.

M01 removes active-fire evidence for the whole input history. M06/M07 use the
existing deterministic 25%/50% spatial masks consistently across the input
history. Corruption is generated after a single crop/augmentation so clean
state targets and corrupted inputs remain aligned.

For A and B:

\[
L = L_{forecast} + 0.1L_{state}.
\]

`L_forecast` is the current alpha-disabled sigmoid focal loss with `gamma=2`.
`L_state` is the same focal form evaluated only on unavailable latest-state
pixels against the corruption-free binary state. For C, `L=L_state`; the
next-day label is not used to train the imputer. The coefficient `0.1` is
frozen and will not be searched.

## 9. Screening and stopping rules

### 9.1 Implementation validity

- M00 input correction and logits must be exactly equal to P00.
- Checkpoint metadata must identify method, history, base P00, split, seed,
  steps, optimizer, corruption policy, and Git commit.
- Non-finite loss, attention, state, or checkpoint values invalidate the run.

### 9.2 Signal screen

For each prototype, using 2021 only:

1. mean AP across M01/M06/M07 must exceed the P00 mean;
2. at least two of the three scenarios must improve over P00;
3. no individual scenario may decrease by more than `0.01 AP`.

A result failing this screen is recorded and stopped without a tuning retry.

### 9.3 Method-candidate screen

An A/B prototype becomes a method candidate only if it:

1. passes the signal screen;
2. exceeds reconstruction-first C at the same history length in mean
   M01/M06/M07 AP;
3. is compared explicitly with the frozen P13 expert on M01 and P10 expert on
   M06/M07.

Only a method candidate and its matched C control may receive one frozen
2022/2023 reporting run. Those years cannot change architecture, loss,
corruption, threshold, or selection.

If neither A nor B beats C and the frozen expert controls, the task-oriented
belief-state premise is rejected. No loss, width, step, or alpha sweep follows.

## 10. Minimal implementation boundary

Implementation should add only:

- `latent_state_common.py`: tensor contract, corruption wrapper, hard
  observation correction, masked state loss, and frozen P00 adapter;
- `state_filter.py`: direction A;
- `state_attention.py`: direction B;
- `reconstruction_baseline.py`: direction C;
- `train_belief_state.py`: one trainer selected by method and history;
- `evaluate_belief_state.py`: matched P00/method evaluation;
- `run_belief_state_on_nibi.sh`: one runner parameterized by method/history;
- `test_wsts_fast_track_belief_state.py`: focused checks only.

The existing P00--P13 modules and pinned upstream repository are not
refactored. One common contract is implemented first; the three model files can
then be developed independently in parallel. The shared trainer/evaluator is
wired after those interfaces are fixed.

The focused checks are limited to:

1. `T=1`/`T=5` forward shapes and finite backward pass for all directions;
2. exact observation preservation and exact M00 P00 logits;
3. missing active-fire values cannot leak through B and attention sums to one;
4. masked state loss and A/B total-loss arithmetic.

Final local verification is only the focused test file, `py_compile`,
`bash -n`, and `git diff --check`. Do not run the global test suite, large
dataset smoke tests, multiple seeds, or login-node training/evaluation.

## 11. Parallel execution and commits

The common contract is a small prerequisite commit. After it lands, A, B, and
C are implemented against separate model files in parallel and committed in
small reviewable commits. The shared trainer/evaluator and Nibi runner are a
separate integration commit. The six jobs are then submitted independently so
one failure does not block the other five.

No GPU/data workload runs on the login node. Expected requests are one H100
MIG GPU per job, with `T=1` and `T=5` jobs running concurrently subject to Nibi
allocation limits.

## 12. Paper logic and claims

### Positioning

Working title: **Observation-Consistent Fire Belief States for Next-Day
Wildfire Forecasting under Partial Observability**.

The paper's logical chain is:

1. existing next-day predictors are brittle when the current fire observation
   is absent or spatially incomplete;
2. full-input reconstruction is not necessarily forecast-sufficient, ordinary
   temporal attention does not enforce observation consistency, and expert
   routing changes the downstream predictor;
3. infer an explicit fire belief state, preserve reliable observations, and
   keep the spread predictor fixed;
4. solve state proposal, observation preservation, and attribution with one
   proposal operator, hard correction, and a shared downstream interface;
5. compare recurrent and attention proposals against reconstruction-first
   under `T=1` and `T=5`.

### Provisional contributions

1. An observation-consistent, task-oriented fire belief-state interface for
   next-day wildfire forecasting.
2. Lightweight recurrent and temporal-attention state-proposal instances under
   a shared frozen spread predictor.
3. A matched analysis separating state inference, input reconstruction, and
   history-length effects under event-disjoint controlled missingness.

These are not final empirical claims. Contribution 1--2 are removed if A/B do
not beat C; cross-year robustness is claimed only if the frozen reporting years
support it.

## 13. Known failure modes

- When the whole active-fire history is missing, fire location may be
  statistically unidentifiable from the remaining public features.
- The recovered state is an active-fire proxy, not a complete perimeter or
  physical combustion state.
- Synthetic validity is oracle-known and cannot establish performance under
  natural clouds, smoke, QA failures, or acquisition delays.
- Fixing P00 gives clean attribution but may cap absolute performance.
- Reconstructing binary fire without detection hour may limit state fidelity.
- A `T=5` gain demonstrates useful history under the shared contract, not
  continuous-time or timestamp-aware inference.

These limitations are reported even if the 2021 screen succeeds.
