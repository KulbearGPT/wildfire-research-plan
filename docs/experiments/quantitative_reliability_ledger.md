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
| B1 | clean C02 | B0 | `21096854` | H100 MIG 20GB, 8 CPU, 32GB, 2h | resubmitted | -- | -- | -- | -- | temporal baseline repair |
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
| D1-ERM | paired clean-corrupt supervised continuation | same B3 checkpoint | mean M01/M06/M07 AP | candidate/control | job `21094927`, `afterok:21093266`; no result yet |
| D1-KL | clean-corrupt predictive consistency | D1-ERM | mean M01/M06/M07 AP | candidate | job `21094928`, `afterok:21093266`; metric frozen before execution |
| D2-STD | standard first convolution continuation | same B3 checkpoint | mean M06/M07 AP | candidate/control | job `21094929`, `afterok:21093266`; no result yet |
| D2-RNC | reliability-normalized first convolution | D2-STD | mean M06/M07 AP | candidate | job `21094930`, `afterok:21093266`; no result yet |
| D3 | reliability-conditioned temporal fusion | corrected C02 temporal fusion | not identifiable on current scenarios | gated | M01 has no valid fire step; M06/M07 use one block across all steps; no GPU run |
| T1 | corruption mixture/curriculum | corrected B3 policy | declared joint mean AP | candidate | no new result yet; at most one tuning contribution |

The older P00/P10 and target-QA measurements motivate these candidates but do
not quantitatively validate D1--D3.

The D1/D2 jobs were submitted at 2026-09-03 16:28 EDT with the same 20GB
H100 MIG, 8 CPU, 32GB, and one-hour request as the corrected baseline wave.
All four are held on the B3 success dependency, so no continuation can run
against a missing or failed base checkpoint.
