# Quantitative Reliability Baselines Design

## Decision

The next research phase is a baseline-first, single-variable funnel. The
project will not call a method a reliable direction until it has quantitative
evidence against a matched corrected-index baseline. The previously proposed
consistency, mask-normalized encoder, and reliability-conditioned temporal
fusion ideas remain hypotheses until that evidence exists.

## Scientific correction

The pinned upstream `find_image_index_from_dataset_index` continues iterating
over later years after it finds a sample. Consequently, the original C00/C02
and P00--P08 training records do not establish the intended 2016--2020
exposure. P09/P10/P13 install the corrected resolver, but they fine-tune legacy
checkpoints and therefore do not replace corrected-index training from
scratch.

The first experiment wave must create four seed-0, 3,000-step baselines from
scratch:

| ID | Backbone | Training policy | Purpose |
| --- | --- | --- | --- |
| B0 | C00 Res18-U-Net, T=1 | clean ERM | corrected clean reference |
| B1 | C02 Res18-UTAE, T=5 | clean ERM | corrected temporal reference |
| B2 | C00 Res18-U-Net, T=1 | 30% FireDrop | corrected M01 baseline |
| B3 | C00 Res18-U-Net, T=1 | 30% FireDrop + 30% BlockDrop | corrected joint reliability baseline |
| B4 | C00 Res18-U-Net, T=1 | B3 + equal total sampler mass per year | matched year-balance training control |

All five runs use the same train-only statistics, 2016--2020 training years,
2021 validation population, optimizer settings inherited from the pinned
upstream baseline, batch size 64, and seed 0. Their engineering evaluation is
limited to M00/M01/M06/M07 on 2021.

## Candidate directions

### D1: clean-corrupt predictive consistency

Use paired clean and synthetically corrupted views of the same crop. Optimize
the same supervised objective as the matched B3 continuation plus a
stop-gradient Bernoulli KL term from the clean prediction to the corrupt
prediction. A matched continuation without KL is mandatory, so any gain is not
attributed to additional optimizer steps. Its primary metric is frozen before
execution as mean M01/M06/M07 AP because the paired corrupt view contains both
FireDrop and BlockDrop; M00 AP remains the clean-performance guardrail.

### D2: reliability-normalized encoder

Replace only the first dynamic-feature convolution with a masked and locally
renormalized operator. It must equal an ordinary convolution when every input
is valid, ignore placeholder values under the mask, and preserve the remaining
encoder/decoder. Compare it with the same training policy and budget as B3.

### D3: reliability-conditioned temporal fusion

This direction is gated under the current controlled scenarios. M01 removes
active fire at every history step, while M06/M07 apply the same spatial block
at every history step. A temporal softmax therefore has no valid alternative
observation to select inside the intended missing region. The masked-softmax
primitive is retained, but no GPU run is justified unless a predeclared
time-varying missingness scenario makes the mechanism identifiable. B1 remains
a corrected temporal baseline, not permission to invent such a scenario after
seeing test results.

### T1: corruption-mixture tuning control

At most one tuning contribution may be retained. Test a three-arm corruption
mixture/curriculum only as a control for D1--D3. It cannot be counted as a
module or architecture contribution.

B4 is the first T1 arm because P10 supplied prior quantitative motivation for
equal-year sampling. It differs from B3 only in sampler mass. Do not introduce
another tuning contribution if B4 is retained.

## Quantitative evidence levels

Every table must name the matched baseline and show absolute AP plus delta.

1. **Candidate:** code or mechanism only; no performance claim.
2. **Screen-positive:** on the frozen 2021 population, primary AP improves by
   at least 0.005, M00 AP drops by no more than 0.010, and the intended
   corruption family does not contain an offsetting regression larger than
   0.005.
3. **Quantitatively supported direction:** after freezing the 2021-selected
   configuration, the primary AP delta is positive in at least two of
   2021/2022/2023 and the three-year mean delta is at least 0.005. Results for
   2022 and 2023 remain separate.
4. **Reliable contribution candidate:** positive primary AP delta in all three
   years, three-year mean delta at least 0.005, and no year's M00 AP drops by
   more than 0.010. Multiple seeds are then justified for the final candidate,
   not for every screen.

Primary metrics are M01 AP for FireDrop/consistency hypotheses, mean M06/M07
AP for spatial reliability hypotheses, and mean M01/M06/M07 AP only for a
method explicitly designed for both families. F1, IoU, Brier, precision,
recall, and loss are reported as diagnostics, not substituted for the declared
primary AP after results are known.

A modality-specialist baseline may use a deterministic router only when the
trigger is directly observable from the input, such as complete active-fire
history absence. Its routing rule must be frozen before opening 2022/2023,
reuse the matched baseline outside the intended failure regime, and report
both standalone checkpoints as well as the composed system. This does not turn
routing into a method contribution; it prevents a deliberately specialized
training baseline from hiding its clean-performance trade-off.

The project will continue experiments until three directions have at least
level 3 evidence, or until the candidate set is exhausted and a new design
decision is required. Unrun hypotheses will never be reported as reliable.

## Scope boundaries

- Do not reopen GroupDRO, prevalence-derived focal alpha, stochastic residual,
  frozen posterior, output-residual capacity, or generic reconstruction unless
  new evidence invalidates their existing negative controls.
- Do not build a 2016--2020 natural target-QA cohort: added-year data lacks the
  required geolocation. The 2021 QA result remains a case study.
- Do not tune on 2022--2023. A configuration is frozen after its 2021 screen,
  then evaluated once per held-out year.
- Do not run training, full inference, HDF5 scans, or other heavy computation on
  the login node.
- Prefer one small GPU and minimal CPU/RAM. Inspect `squeue`, `sinfo`, and
  `sprio` immediately before each submission wave and choose the resource with
  the shortest plausible start time.
- Use focused behavioral tests only. No full-suite pytest or large validation
  campaign is required for a rapid prototype.

## Artifacts and reporting

Each run archives the committed project, records the project/upstream commits,
stores the exact command and environment, writes an immutable completion
record, and evaluates M00/M01/M06/M07 into JSON. The experiment ledger records
the Slurm job IDs, resource request, terminal state, metrics, matched baseline,
delta, and evidence level. Documentation and implementation are committed at
each independently usable milestone.
