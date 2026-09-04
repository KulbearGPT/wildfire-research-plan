# Quantitative Reliability Experiment Ledger

This ledger separates completed quantitative evidence from untested
hypotheses. A candidate is not called a reliable direction until it satisfies
the evidence levels in
`docs/superpowers/specs/2026-09-03-quantitative-reliability-baselines-design.md`.

## Historical experiment audit

| Experiment | Quantitative evidence | Resolver/data validity | Scientific classification | Current decision |
| --- | --- | --- | --- | --- |
| Phase 0 inventory/repair | 999 events; 653/156/190 train/validation/test events | valid | data foundation | retain |
| no-fire and persistence rules | test event-macro AP 0.000531 and 0.070538 | valid | deterministic baselines | retain |
| official Res18-U-Net reproduction | trained fold-2 AP 0.554664; 12 released folds AP 0.452764 ± 0.088217 | valid for original WSTS | reproduction evidence | retain; not a new method |
| legacy C00/C02 3K | 2021 AP 0.547372 / 0.584540 | pooled-year resolver invalid | runtime evidence only | replace with B0/B1 |
| legacy C00/C02 six 10K runs | completed and used for 96 missingness evaluations | pooled-year resolver invalid | legacy diagnostics | do not use as corrected baselines |
| M00--M07 diagnosis | M01 dominant; blocks M06/M07 next; weather M03--M05 much smaller | evaluation is single-year and valid | failure-regime evidence | retain |
| P00 FireDrop | 2021 M01 AP 0.045920 → 0.299465 | matched legacy resolver | quantitatively supported training direction, attribution limited | rerun as B2 |
| P01 concatenated validity | M00/M01/M02 decreased; M07 +0.0002 | legacy resolver | negative module result | reject |
| P02 Fire+BlockDrop | versus P00: M06 +0.0183, M07 +0.0303; M00 -0.0200, M01 -0.0322 | legacy resolver | robustness/clean trade-off | rerun as B3 |
| P03 hard spatial router | block gain in 2021/2023 but regression in 2022 | evaluation valid, experts legacy | unstable routing result | reject |
| P04/P05/P06 residual/capacity controls | M06/M07 all below P03 | legacy experts | capacity negative controls | reject |
| P07 stochastic residual | predictive variance approximately 1e-8 | legacy expert | identifiability failure | reject |
| P08 teacher posterior | M06/M07 AP 0.306096/0.123222, below P00 | legacy expert | posterior/reconstruction negative result | reject |
| P09 GroupDRO | 2021 M06/M07 0.365531/0.185136 | corrected resolver, legacy P02 initialization | confounded ablation | superseded by P10 |
| P10 matched ERM | versus P00 mean block AP +0.052345/+0.023389/+0.020189 in 2021/2022/2023 | corrected resolver, legacy P02 initialization | quantitatively supported exposure/sampling direction | retain as motivation; rebuild from scratch |
| P11 missingness routing | M01 0.291202 versus P00 0.299465 | corrected block expert | negative routing result | reject |
| P12 prevalence focal alpha | M06/M07 0.070432/0.061832 with severe overprediction | corrected resolver | loss diagnostic | reject this alpha policy |
| P13 FireDrop expert | M01 delta +0.004771/-0.002058/+0.019928 in 2021/2022/2023 | corrected resolver, legacy P00 initialization | temporally mixed | do not promote |
| filter/attention/reconstruction × T=1/T=5 | every row below P00/P10 on its primary corruptions | corrected resolver, frozen legacy predictor | six negative belief-state probes | stop this branch |
| natural VIIRS reliability screen | attention AP +0.000287; F1/Brier/loss worse | 24 fixed 2021 event-days only | neutral/negative natural-input probe | do not promote |
| target-QA censoring | P00 AP 0.382293 → 0.398789; 12.54% zero-label pixels lacked reliable observation | valid for the 24-event 2021 cohort | target-observation case study | retain; not a forecasting method gain |
| 2016--2020 QA cohort attempt | preparation failed because added-year TIFFs lack CRS/ROI | invalid for cross-year geolocation | data feasibility failure | stop; dependent jobs cancelled |

Two historical mechanisms have quantitative signal: FireDrop for M01 and
corrected/year-balanced ERM exposure for M06/M07. They do not satisfy the new
request for three newly verified directions because the former uses the legacy
resolver and the latter starts from the legacy P02 checkpoint. B2/B3 and the
new matched candidates are designed to close that attribution gap.

## Corrected-index baseline wave

| ID | Training policy | Matched baseline | Slurm job | Resource | State | 2021 M00 AP | M01 AP | M06 AP | M07 AP | Classification |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| B0 | clean C00 | -- | `21093263` | H100 MIG 20GB, 8 CPU, 32GB, 1h | completed | 0.580988 | 0.039387 | 0.323805 | 0.133672 | corrected baseline |
| B1 | clean C02 | B0 | `21102675` | H100 MIG 20GB, 8 CPU, 64GB, 2h | completed | 0.595265 | 0.093697 | 0.329582 | 0.140190 | corrected temporal baseline |
| B2 | FireDrop C00 | B0 | `21093265` | H100 MIG 20GB, 8 CPU, 32GB, 1h | completed | 0.559282 | 0.262091 | 0.308248 | 0.125715 | specialist baseline; standalone clean gate fails |
| B3 | FireDrop + BlockDrop C00 | B2 | `21093266` | H100 MIG 20GB, 8 CPU, 32GB, 1h | completed | 0.558318 | 0.276890 | 0.343010 | 0.172086 | screen-positive joint training baseline |
| B4 | B3 + equal-year sampling | B3 | `21094665` (`afterok:21093263`) | H100 MIG 20GB, 8 CPU, 32GB, 1h | completed | 0.534606 | 0.252345 | 0.329461 | 0.166855 | rejected tuning candidate |

B4 was submitted at 2026-09-03 16:23 EDT. The live queue still showed five
pending 20GB MIG requests versus 354 pending 40GB MIG, 34 pending A100, and
867 pending full-H100 requests. Its dependency prevents execution unless the
common B0 runner succeeds.

Submission snapshot at 2026-09-03 15:59 EDT showed five pending 20GB H100
MIG requests, compared with 353 pending 40GB MIG, 34 pending A100, and 868
pending full-H100 requests. The baseline wave therefore selected 20GB MIG.

B0 completed successfully at 2026-09-03 16:26 EDT in 26m51s. Its archived
record confirms corrected indexing, from-scratch initialization, seed 0, and
3,000 maximum steps; the four reported scenarios each contain 3,181 samples.

At the 2026-09-03 16:57 EDT monitor, B1 job `21093264` had used 31m02s but was
only 1,363/3,000 steps through the slower T=5 model. It had produced no result
and was cancelled before its one-hour limit; replacement `21096854` keeps the
scientific configuration and 20GB MIG request fixed but uses a two-hour limit.
That replacement reached 2,444/3,000 steps but was killed at 41m14s after using
31.97/32GB host memory. Retry `21102675` changes only host memory to 64GB.
It completed successfully in 44m18s. B1 exceeds B0 on all four 2021 scenarios;
job `21104532` performs the frozen 2022/2023 baseline evaluation. B1 remains an
existing temporal architecture baseline, not a new contribution.
That job completed: B1 minus B0 M00/M01/M06/M07 AP deltas are
+0.014277/+0.054310/+0.005778/+0.006518 in 2021,
+0.039479/+0.041762/+0.016716/+0.013413 in 2022, and
+0.000400/+0.010055/-0.002538/-0.000651 in 2023. Thus C02 is the stronger
clean/fire-missing architecture baseline, but its block advantage is not
uniform and it does not replace the corruption-trained C00 comparisons.

B2 completed at 2026-09-03 17:19 EDT in 22m13s. Against B0, M01 AP improves
by 0.222704 but M00 AP drops by 0.021705, so the standalone checkpoint fails
the clean guardrail. Because complete active-fire-history absence is directly
observable and the project already froze this specialist-routing convention
for P03/P10, the held-out candidate is the explicit B0/B2 route: B0 for M00
and B2 only for M01. That composed baseline has zero M00 delta and passes the
2021 screen. Jobs `21098505` and `21098506` evaluate both standalone
checkpoints on 2022 and 2023; routing is applied only during aggregation.

B3 completed at 2026-09-03 17:37 EDT in 18m02s. Relative to B2 it changes
M00/M06/M07 AP by -0.000964/+0.034762/+0.046371, for a +0.040566 block mean;
it passes every 2021 screen condition. The frozen B3 checkpoint is queued for
both held-out years in job `21099907`. B4 completed at 17:56 EDT but regressed
M00 by 0.023712 and mean M06/M07 by 0.009390 versus B3. Equal-year sampling is
therefore rejected without held-out evaluation.

## Follow-on candidate register

| ID | Hypothesis | Matched control | Primary metric | Evidence level | Quantitative conclusion |
| --- | --- | --- | --- | --- | --- |
| D1-ERM | paired clean-corrupt supervised continuation | same B3 checkpoint | mean M01/M06/M07 AP | matched control | M00/M01/M06/M07 0.583021/0.251367/0.361236/0.176152 |
| D1-KL | clean-corrupt predictive consistency | D1-ERM | mean M01/M06/M07 AP | reliable (level 4) | three-year mean delta +0.008970; all years positive |
| D2-STD | standard first convolution continuation | same B3 checkpoint | mean M06/M07 AP | matched control | M00/M06/M07 0.583073/0.368775/0.189082 |
| D2-RNC | reliability-normalized first convolution | D2-STD | mean M06/M07 AP | rejected | primary delta -0.003218; M07 delta -0.007200 |
| D3 | reliability-conditioned temporal fusion | corrected C02 temporal fusion | not identifiable on current scenarios | gated | M01 has no valid fire step; M06/M07 use one block across all steps; no GPU run |
| D4-TOKEN | 64-parameter invalid-region token after standard first convolution | D2-STD | mean M06/M07 AP | borderline reject | primary delta +0.004960, below frozen +0.005 gate |
| D5-CIWC | counterfactual impact-weighted consistency | D1-ERM; D1-KL closest ablation | mean M01/M06/M07 AP | rejected | primary delta +0.015468, below frozen +0.020 target; no held-out evaluation |
| D6-CIRC | CIWC plus counterfactual spatial-risk rank consistency | D1-ERM; D1-KL and D5-CIWC closest ablations | mean M01/M06/M07 AP | rejected | primary delta +0.008733; M00 -0.012626 and M06 -0.006122 fail guardrails; no held-out evaluation |
| D7-CRA | joint two-regime counterfactual reliability adapter | D1-ERM; D5-CIWC closest ablation | mean M01/M06/M07 AP | rejected | primary delta +0.015015, below frozen +0.020 target; no held-out evaluation |
| D8-FFCA | failure-factorized BlockDrop-only counterfactual adapter | D1-ERM; D5-CIWC and D7-CRA closest ablations | mean M01/M06/M07 AP | rejected | primary delta +0.014379, below frozen +0.020 target; no held-out evaluation |
| T1/B4 | equal-year corruption sampling | corrected B3 policy | mean M06/M07 AP | rejected | primary delta -0.009390; no held-out evaluation |

The older P00/P10 and target-QA measurements motivate these candidates but do
not quantitatively validate D1--D3.

The D1/D2 jobs were submitted at 2026-09-03 16:28 EDT with the same 20GB
H100 MIG, 8 CPU, 32GB, and one-hour request as the corrected baseline wave.
All four are held on the B3 success dependency, so no continuation can run
against a missing or failed base checkpoint.

Initial D1 jobs `21094927/21094928` failed before optimization because applying
FireDrop before the upstream fire-aware crop could select a different crop for
the corrupt view despite identical RNG state. Commit `3c70a99` now performs
the single upstream crop first and corrupts that exact processed tensor. The
matched retries were submitted after the 2026-09-03 19:04 EDT queue check.

## Cross-year quantitative directions

| Direction | Category | Year | Matched baseline primary AP | Candidate primary AP | Primary delta | M00 delta | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| R1: observable FireDrop specialist (B0/B2 route) | training/augmentation | 2021 | 0.039387 | 0.262091 | +0.222704 | 0.000000 | frozen screen |
| R1: observable FireDrop specialist (B0/B2 route) | training/augmentation | 2022 | 0.006669 | 0.130812 | +0.124143 | 0.000000 | held-out |
| R1: observable FireDrop specialist (B0/B2 route) | training/augmentation | 2023 | 0.010101 | 0.113583 | +0.103482 | 0.000000 | held-out |
| R2: incremental BlockDrop (B2 to B3) | training/augmentation | 2021 | 0.216982 | 0.257548 | +0.040566 | -0.000964 | frozen screen |
| R2: incremental BlockDrop (B2 to B3) | training/augmentation | 2022 | 0.099024 | 0.118356 | +0.019331 | +0.009361 | held-out |
| R2: incremental BlockDrop (B2 to B3) | training/augmentation | 2023 | 0.132137 | 0.155703 | +0.023566 | +0.012106 | held-out |
| R3: predictive consistency (D1-ERM to D1-KL) | method | 2021 | 0.262918 | 0.272121 | +0.009203 | +0.001355 | frozen screen |
| R3: predictive consistency (D1-ERM to D1-KL) | method | 2022 | 0.120516 | 0.133662 | +0.013146 | +0.005051 | held-out |
| R3: predictive consistency (D1-ERM to D1-KL) | method | 2023 | 0.140323 | 0.144884 | +0.004561 | +0.001804 | held-out |

R1 has positive M01 AP deltas in all three years and a three-year mean delta
of +0.150109. It therefore reaches level 4, **reliable contribution
candidate**. This consumes the single training/tuning contribution slot. The
standalone B2 checkpoint remains fully disclosed: its M00 deltas are
-0.021705/-0.002329/-0.024170 in 2021/2022/2023 and do not meet the reliable
clean guardrail. Held-out jobs were `21098505/21098506`.

R2 has positive mean M06/M07 AP deltas in all three years and a three-year
mean delta of +0.027821, with no clean regression below the -0.010 guardrail.
It therefore also reaches level 4. R1 and R2 are separate quantitatively
supported directions/ablations, but contribution accounting combines them as
one dual-regime corruption-training contribution rather than claiming two
method contributions. B3 held-out job was `21099907`.

D2 jobs `21094929/21094930` completed their strict matched comparison. RNC
changed M06/M07 by +0.000764/-0.007200 relative to the standard continuation,
for a -0.003218 primary delta; M00 also changed by -0.009428. D2 fails the
2021 screen and is rejected without held-out evaluation.

D4 is the single smallest follow-up to the D2 failure: it retains the standard
convolution and adds one learned scalar per output channel, activated in
proportion to local invalid coverage. It is exactly D2-STD when inputs are all
valid, adds 64 parameters, and reuses the completed D2-STD control. Fifteen
focused D2/D4 and runner tests passed before job `21103691` was submitted on
the least-contended 20GB MIG class at 2026-09-03 19:39 EDT.

D1 retries `21102676/21102677` completed successfully. KL improves the frozen
mean M01/M06/M07 AP by +0.009203 versus paired ERM, with scenario deltas
+0.016018/+0.003003/+0.008588 and M00 delta +0.001355. It passes the 2021
screen; job `21105328` evaluated both frozen checkpoints on 2022 and 2023 in
one allocation.
That job completed in 5m13s. D1-KL improves the joint primary AP in all three
years by +0.009203/+0.013146/+0.004561, for a three-year mean of +0.008970;
M00 deltas are +0.001355/+0.005051/+0.001804. D1 therefore reaches level 4,
**reliable contribution candidate**.

D4 completed in job `21103691`. Its M06/M07 deltas are
+0.003115/+0.006804, primary mean +0.00495975, and M00 delta -0.005243. The
primary result misses the predeclared +0.005 threshold by 0.00004025, so D4 is
recorded as a borderline negative and is not evaluated on held-out years.

## D5 counterfactual impact-weighted consistency

D5 tests whether D1's global consistency signal was diluted by forecast pixels
that did not respond to the synthetic observation failure. CIWC weights each
pixel's Bernoulli KL by the stop-gradiented clean/corrupt probability change,
normalizes the weights per sample, and otherwise keeps D1-KL's B3
initialization, paired sample stream, optimizer, 3,000 steps, and
`lambda=0.1` fixed. The novelty claim is limited to this same-model
counterfactual forecast-impact support; masking, consistency, and disagreement
weighting alone are not claimed as new.

The gate was frozen before training: relative to D1-ERM, 2021 mean
M01/M06/M07 AP must improve by at least `+0.020`, every primary scenario must
remain above `-0.005`, and M00 must remain above `-0.010`; CIWC must also beat
global D1-KL. Only a passing model may access 2022--2023, and final retention
requires positive primary deltas in every year plus a three-year mean of at
least `+0.020`.

Focused implementation checks passed (`17 passed`) before job `21108872` was
submitted at 2026-09-03 22:29 EDT with one 20GB H100 MIG. Its scheduler
estimate slipped to 01:07 EDT without starting. At the planned 23:00 queue
check, `sbatch --test-only` predicted immediate starts for both 40GB MIG and a
full H100. The untouched pending job was cancelled and replaced by job
`21111043` on one 40GB H100 MIG, 8 CPU, 32GB, and 30 minutes; it started on
`g30` in four seconds. All training and evaluation occur inside Slurm.

Job `21111043` completed in 10m03s with exit `0:0`. CIWC produced
M00/M01/M06/M07 AP `0.584364/0.294861/0.361160/0.179137`. Relative to
D1-ERM, the scenario deltas are `+0.001344/+0.043494/-0.000076/+0.002985`;
the primary mean is `+0.015468`. It also improves the primary mean over global
D1-KL by `+0.006265`, but the predeclared `+0.020` requirement is not met.
D5 is therefore rejected without accessing 2022--2023. The result supports
counterfactual focusing for complete FireDrop but shows that pixelwise impact
weighting does not recover BlockDrop ranking.

## D6 counterfactual impact-and-rank consistency

D6 keeps D5-CIWC and adds a fixed `0.01` normalized Jensen-Shannon divergence
between the clean and corrupt forecasts' spatial softmax distributions. This
is a listwise constraint on future-risk ordering, aimed directly at the M06
and M07 weakness exposed by D5. The novelty boundary, exact objective, matched
contract, and unchanged `+0.020` decision gate were frozen before code or
training in
`docs/superpowers/specs/2026-09-03-counterfactual-rank-consistency-design.md`.

The new focused tests and its D5/D1 dependencies passed (`19 passed`), along
with Python compilation and shell syntax. At 2026-09-03 23:45 EDT,
`sbatch --test-only` estimated a 20GB H100 MIG start at 00:51, while a 40GB
MIG and full H100 could start immediately. Job `21112470` therefore requested
the smaller immediately available option: one 40GB H100 MIG, 8 CPU, 32GB,
and 30 minutes. It started on `g36` within seconds. Training and 2021
evaluation run entirely inside Slurm.

Job `21112470` completed in 8m24s with exit `0:0`. D6 produced
M00/M01/M06/M07 AP `0.570395/0.280630/0.355114/0.179208`. Relative to
D1-ERM, the deltas are `-0.012626/+0.029264/-0.006122/+0.003056`, giving a
primary mean of only `+0.008733`. D6 is also `-0.006735` below D5-CIWC on the
primary mean. It therefore misses the target and violates both the M00 and
per-scenario guardrails. D6 is rejected without 2022--2023 evaluation or a
rank-weight sweep. The listwise term weakened the useful D5 signal instead of
repairing BlockDrop ranking.

## D7 counterfactual reliability adapter

D7 is the final predeclared failure-ladder candidate. It keeps D5's paired
supervision and CIWC objective, but jointly trains the base predictor with a
2,769-parameter forecast-logit adapter conditioned on separate FireDrop and
BlockDrop maps plus block extent. Fully observed samples bypass the adapter
exactly. The method and bounded novelty claim were frozen in
`docs/superpowers/specs/2026-09-04-counterfactual-reliability-adapter-design.md`;
adapter, distillation, and reliability gating alone are explicitly treated as
prior art.

All D7 and related focused tests passed (`28 passed`), along with Python
compilation and shell syntax. At 2026-09-04 00:23 EDT, `sbatch --test-only`
estimated a 20GB H100 MIG start at 00:36, while 40GB MIG and full H100 were
immediately available. Job `21114921` therefore requested one 40GB H100 MIG,
8 CPU, 32GB, and 30 minutes. It started on `g30` within seconds; all training
and 2021 evaluation run inside Slurm.

Job `21114921` completed in 8m33s with exit `0:0`. D7 produced
M00/M01/M06/M07 AP `0.579199/0.281434/0.366924/0.185442`. Relative to
D1-ERM, the deltas are `-0.003822/+0.030068/+0.005688/+0.009290`, for a
primary mean of `+0.015015`. The clean and per-scenario guardrails pass, but
the frozen materiality target does not. Relative to D5, D7 gains
`+0.005763/+0.006305` on M06/M07 while losing `-0.013426` on M01. D7 is
therefore rejected without held-out evaluation. This complementary pattern
quantitatively motivates separating global FireDrop consistency from a
BlockDrop-only correction instead of sharing one adapter across both regimes.

## D8 failure-factorized counterfactual adapter

D8 implements the D5/D7 complementary finding as one shared-backbone model:
CIWC handles FireDrop-only samples with an exact adapter bypass, while a
2,625-parameter residual activates only when BlockDrop is present. The fixed
architecture, unchanged loss, novelty boundary, and decision rule are recorded
in
`docs/superpowers/specs/2026-09-04-failure-factorized-counterfactual-adapter-design.md`.

The D8-focused and related checks passed (`26 passed`), including an explicit
nonzero-weight FireDrop bypass test, Python compilation, and shell syntax. At
2026-09-04 00:59 EDT, `sbatch --test-only` predicted an immediate 40GB MIG
start and a 20GB MIG start about one minute later. Job `21116301` requested
one 40GB H100 MIG, 8 CPU, 32GB, and 30 minutes, and started on `g30` within
seconds. Training and 2021 evaluation run only inside Slurm.

Job `21116301` completed in 8m26s with exit `0:0`. D8 produced
M00/M01/M06/M07 AP `0.578203/0.289113/0.361630/0.181149`. Relative to
D1-ERM, the deltas are `-0.004817/+0.037747/+0.000394/+0.004997`, giving a
primary mean of `+0.014379`. It passes scenario guardrails but misses the
materiality target and is `-0.001089` below D5 on the primary mean. D8 is
rejected without held-out evaluation. Restricting the last-layer adapter
recovers much of D5's FireDrop behavior but removes D7's block gain, so this
output-adapter family is closed rather than widened or swept.

## Contribution accounting and stop decision

| Direction | Matched change | Trainable parameter delta | Matched budget | Three-year mean primary delta | Final level |
| --- | --- | ---: | --- | ---: | --- |
| R1 FireDrop specialist | B0 clean to B2 FireDrop; observable route selects one checkpoint | 0 | 3K from scratch each | +0.150109 | reliable, level 4 |
| R2 incremental BlockDrop | B2 FireDrop to B3 FireDrop + BlockDrop | 0 | 3K from scratch each | +0.027821 | reliable, level 4 |
| R3 predictive consistency | D1-ERM to D1-KL, lambda 0.1 | 0 | 3K continuation each from B3 | +0.008970 | reliable, level 4 |

The required three quantitatively reliable directions are now present:

1. R1 FireDrop specialist for complete active-fire-history absence;
2. R2 incremental BlockDrop for structured spatial missingness;
3. R3 clean-corrupt predictive consistency beyond its matched continuation.

R1 and R2 are two supported ablation directions but form one publishable
dual-regime corruption-training contribution. R3 is the independent method
contribution. B4, D2, and D4 are quantitative negative controls; D3 is gated
because the frozen scenarios cannot identify temporal selection. No result is
promoted merely from code or motivation. The next compute, if pursued, is
multi-seed/longer-budget confirmation of the combined B3 + D1-KL recipe, not
another test-year-guided architecture search.
