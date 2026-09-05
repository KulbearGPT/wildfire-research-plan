# Rejected and Out-of-Scope Experiment Record

This is the single active-tree record for experiments removed from the current
`T=1` mainline. It keeps the quantitative conclusion needed to avoid repeating
failed work. Full code and narrative are available on Git branch
`archive/pre-t1-cleanup-2026-09-04` at commit `4b843ad`. Checkpoints, run
records, and Slurm logs are archived under
`/project/6085198/kulbear/wildfire/archive/pre-t1-cleanup-2026-09-04/runs`;
[`artifact-archive-manifest.tsv`](artifact-archive-manifest.tsv) records every
active or moved top-level artifact.

## Invalid or superseded foundations

| Experiment | Quantitative observation | Decision |
| --- | --- | --- |
| Legacy C00/C02 3K | 2021 AP 0.547372/0.584540 | pooled-year resolver was invalid; runtime evidence only |
| Legacy C00/C02 six 10K runs | completed and used for 96 M00--M07 evaluations | training exposure invalid; retain only the failure-regime diagnosis |
| P00 FireDrop | M01 0.045920 → 0.299465 | useful signal rebuilt correctly as B2 |
| P02 FireDrop+BlockDrop | versus P00: M06 +0.0183, M07 +0.0303, M00 -0.0200, M01 -0.0322 | useful signal rebuilt correctly as B3 |
| P09 GroupDRO | 2021 M06/M07 0.365531/0.185136 | tied matched P10 ERM; no GroupDRO attribution |
| P10 matched ERM | mean M06/M07 delta versus P00 +0.052345/+0.023389/+0.020189 in 2021/2022/2023 | motivation only; rebuilt from scratch in corrected baselines |

The valid diagnosis retained from the old matrix is that M01 complete
active-fire-history loss is the dominant failure, followed by M06/M07
structured spatial loss. Weather-only M03--M05 failures were smaller.

## P-series probes

| ID | Result | Rejection reason |
| --- | --- | --- |
| P01 validity channel | M00/M01/M02 decreased; M07 +0.0002 | no useful gain |
| P03 hard router | M06/M07 improved in 2021 and 2023 but changed -0.0231/-0.0261 in 2022 | temporally unstable |
| P04 1×1 residual | M06/M07 0.332083/0.139913, below P03 0.350878/0.166982 | insufficient correction |
| P05 3×3 residual | M06/M07 0.327723/0.138767 | extra output capacity did not help |
| P06 decoder adaptation | M06/M07 0.330240/0.141035 | below P03 |
| P07 stochastic residual | predictive variance approximately 1e-8 | latent state not identifiable |
| P08 teacher posterior | M06/M07 0.306096/0.123222 | below P00 |
| P11 missingness router | M01 0.291202 versus P00 0.299465 | routed complete FireDrop regressed |
| P12 focal alpha | M06/M07 0.070432/0.061832 with severe overprediction | prevalence-derived alpha rejected |
| P13 FireDrop expert | M01 delta +0.004771/-0.002058/+0.019928 in 2021/2022/2023 | mixed temporal result |

## Belief-state and natural-observation probes

| Probe | Key result | Decision |
| --- | --- | --- |
| belief filter T=1 | M01/M06/M07 0.260952/0.316738/0.131167; mean delta -0.012264 versus P00 | reject |
| belief attention T=1 | M01/M06/M07 0.274733/0.316787/0.130474; mean delta -0.007884 | reject |
| belief reconstruction T=1 | M01/M06/M07 0.206976/0.315143/0.129389; mean delta -0.031380 | reject |
| corresponding T=5 probes | every candidate had negative corrupted mean delta | outside current scope and no rescue of the branch |
| natural VIIRS screen | attention AP +0.000287, with worse F1/Brier/loss | no forecasting-method gain |
| target-QA censoring | P00 AP 0.382293 → 0.398789; 12.54% of zero-label pixels lacked reliable target-day observation | useful 24-event case study, outside the method mainline |
| 2016--2020 QA cohort | preparation failed because added-year TIFFs lack CRS/geotransform | cross-year alignment infeasible with current files |

## Corrected T=1 candidates that did not pass

All AP values below use the frozen 2021 screen. Candidates that failed were
not evaluated on 2022--2023.

| ID | M00/M01/M06/M07 AP or primary result | Matched conclusion | Job |
| --- | --- | --- | ---: |
| B4 equal-year sampling | 0.534606/0.252345/0.329461/0.166855 | mean M06/M07 -0.009390 and M00 -0.023712 versus B3 | 21094665 |
| D2-RNC | M06 +0.000764, M07 -0.007200 versus D2-STD | primary -0.003218; M00 -0.009428 | 21094930 |
| D3 temporal fusion | no GPU run | frozen corruptions cannot identify temporal selection | — |
| D4 input token | primary +0.004960; M00 -0.005243 versus D2-STD | missed +0.005 gate by 0.000040 | 21103691 |
| D5 CIWC | 0.584364/0.294861/0.361160/0.179137 | primary +0.015468 versus D1-ERM, below +0.020 | 21111043 |
| D6 CIWC+rank | 0.570395/0.280630/0.355114/0.179208 | primary +0.008733; M00 and M06 guardrails failed | 21112470 |
| D7 reliability adapter | 0.579199/0.281434/0.366924/0.185442 | primary +0.015015, below +0.020 | 21114921 |
| D8 factorized adapter | 0.578203/0.289113/0.361630/0.181149 | primary +0.014379, below +0.020 | 21116301 |
| D9 error-pair rank | 0.528919/0.260790/0.325537/0.153634 | primary -0.016265; M00 -0.054101 | 21117395 |
| D10 prompt pyramid | 0.583725/0.302593/0.374300/0.191693 | total +0.026610, but module +0.002915 and below D4 | 21120041 |
| D11 complete prompts | 0.583391/0.300108/0.375842/0.191872 | total +0.026356; module +0.004928, 0.000072 below gate | 21121610 |

D4, D10, and D11 supplied the ablations that led to D12. Their necessary
mechanisms are consolidated into the retained D12 implementation; their
independent training/evaluation interfaces and checkpoints remain only in the
archives.

## T=5 boundary

T=5 is outside the active research scope. The final matched transfer used the
corrected T=5 B5 base and identical 3,000-step continuations. On 2021,
D13-STD had M00/M01/M06/M07 AP 0.600267/0.357375/0.387873/0.196784 and
D13-SARP had 0.602925/0.355388/0.389202/0.200609. The SARP module changed mean
M06/M07 by +0.002577, below the frozen +0.005 gate, so 2022--2023 were not
opened and no T=5 transfer claim is retained.

## Reuse rule

Do not reopen one of these implementations because its single scenario looks
promising. A future proposal may reuse the documented observation only when it
states a new mechanism, uses the corrected T=1 baseline chain, freezes its 2021
gate before training, and leaves the 2022--2023 tests untouched until the gate
passes.
