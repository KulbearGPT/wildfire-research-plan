# P00--P11 Rapid Reliability Prototypes

## Outcome

The rapid spatial-reliability branch is complete. P03 was the strongest 2021
prototype, but its fixed temporal test comparison was inconsistent across
years. It is therefore retained as a diagnostic result rather than promoted as
the main method. A later stochastic belief-residual probe (P07) improved P00
under block missingness but collapsed to an effectively deterministic
correction and did not beat P04. P08 used a training-only clean posterior and
produced non-collapsed uncertainty, but its block-missingness AP fell below
P00. P09 then passed the 2021 gate and improved M06/M07 over P00 in both fixed
test years. P10 supplied the matched corrected-index ERM control and retained
the gain, showing that GroupDRO adds no stable benefit. P11 then routed complete
active-fire-history loss to P10, but its M01 AP regressed and the route was
rejected without opening the held-out years.

## 2021 selection results

| Prototype | Added trainable parameters | M00 AP | M01 AP | M06 AP | M07 AP | Decision |
|---|---:|---:|---:|---:|---:|---|
| P00 FireDrop | full C00 training | 0.585322 | 0.299465 | 0.317005 | 0.129176 | Mainline baseline |
| P03 P00/P02 spatial router | 0 | 0.585322 | 0.299465 | 0.350878 | 0.166982 | Selected for final check |
| P04 `1x1` residual | 17 | 0.585322 | 0.299465 | 0.332083 | 0.139913 | Compute control |
| P05 `3x3` residual | 145 | 0.585322 | 0.299465 | 0.327723 | 0.138767 | Rejected |
| P06 final-block router | adapted tail only | 0.585322 | 0.299465 | 0.330240 | 0.141035 | Rejected |
| P07 stochastic belief residual | 4,945 | 0.585322 | 0.299465 | 0.331009 | 0.140771 | Rejected: variance collapse |
| P08 teacher-posterior belief | 14,465 | 0.585322 | 0.299465 | 0.306096 | 0.123222 | Rejected: AP regression |
| P09 year-corruption GroupDRO router | full P02 fine-tune | 0.585322 | 0.299465 | 0.365531 | 0.185136 | Ablation; no stable ERM gain |
| P10 corrected-index balanced ERM router | full P02 fine-tune | 0.585322 | 0.299465 | 0.365203 | 0.185669 | Parsimonious candidate |
| P11 P00/P10 missingness router | 0 | 0.585322 | 0.291202 | 0.365203 | 0.185669 | Rejected: M01 regression |

P03 improved P00 by 0.033872 AP on M06 and 0.037806 on M07 while preserving
M00/M01 exactly. Increasing the capacity of the single-forward correction in
P04--P06 did not close the gap.

## P07 stochastic belief probe

P07 kept P00 frozen, drew four reparameterized residual samples inside known
missing blocks, and trained its 4,945-parameter belief/output heads for 3,000
steps on 2016--2020. It exactly preserved P00 on M00/M01. Relative to P00, its
2021 AP improved by approximately 0.01400 on M06 and 0.01160 on M07.

The stochastic mechanism did not earn promotion. Relative to the deterministic
P04 control, P07 changed AP by -0.001074 on M06 and +0.000858 on M07, leaving
the two-scenario mean slightly lower. Its mean predictive variance was only
1.204e-8 on M06 and 1.503e-8 on M07, with zero variance on M00/M01 by
construction. This is effectively posterior collapse: the learned predictor
uses its mean correction but not a meaningful belief distribution. The P07
checkpoint is therefore not advanced to the fixed 2022--2023 test set.

## P08 teacher-posterior belief probe

P08 keeps P00 frozen and trains 14,465 parameters in posterior, prior,
feature-reconstruction, and output heads. Training uses the clean aligned input
only as a posterior teacher; inference uses only the corrupted input and mask.
The run is fixed at one seed, four samples, Adam `1e-3`, and 3,000 steps on
2016--2020, followed by 2021 M00/M01/M06/M07 selection. It does not access the
2022--2023 test set unless it passes the written promotion rule.

P08 exactly preserved M00/M01 within the `1e-6` tolerance. Its mean predictive
variance was `7.605e-5` on M06 and `1.123e-4` on M07, so it passed the
non-collapse threshold by more than an order of magnitude. The teacher signal
therefore fixed P07's identifiability failure.

It nevertheless failed the mandatory AP gate. M06 AP was `0.306096`, down
`0.010909` from P00, and M07 AP was `0.123222`, down `0.005954`. Mean M06/M07
AP was `0.214659`, below both P00 (`0.223091`) and P04 (`0.235998`). P08 is
rejected without a P04 Brier follow-up or 2022--2023 evaluation. This result
separates uncertainty generation from useful forecast ranking: the posterior
made samples diverse, but the learned inference prior degraded the forecast.
Per the pre-registered decision, the next method direction is
temporal/environment robust optimization rather than another belief residual.

## P09 year-corruption GroupDRO

P09 initialized from P02 and fine-tuned all expert parameters for 3,000 steps
with 15 environments: five training years crossed with clean, 25%, and 50%
block states. Inverse-year sampling balanced expected year mass, while
exponentiated GroupDRO weights emphasized high-loss environments. The final
weight of group 14 (2020 at 50% BlockDrop) was `0.222623`, versus the uniform
initial value `0.066667`.

During the first attempt, runtime weights changed only for groups 12--14. A
minimal reproduction showed that the pinned upstream dataset resolver breaks
only its inner fire loop and then continues through later years, so every
multi-year index is overwritten by the last year. P09 locally replaces that
resolver with a first-match implementation. The corrected run showed distinct
updates in all 15 groups from step 1 onward.

On 2021, P09 preserved M00/M01 exactly and reached `0.365531/0.185136` AP on
M06/M07. Mean block AP was `0.275333`, improving P00 by `0.052243` and P03 by
`0.016404`; it therefore passed all three pre-registered selection conditions.

P09 also passed the frozen temporal gate. Its mean M06/M07 AP improvement over
P00 was `0.018699` in 2022 and `0.022378` in 2023. However, P09 differs from
P03 in two coupled ways: correct five-year sample resolution and GroupDRO.
These results established a strong corrected-data candidate but did not by
themselves identify a GroupDRO effect. P10 below resolves that attribution.

## P10 matched corrected-index ERM

P10 held fixed the P02 initialization, corrected five-year index resolver,
inverse-year sampler, corruption draws, seed 0, batch size 64, AdamW optimizer,
`1e-4` learning rate, and 3,000-step budget. Its only scientific change from
P09 was replacing the GroupDRO-weighted objective with the ordinary mean of
the per-sample focal losses.

On 2021, P10 reached `0.365203/0.185669` M06/M07 AP. Its mean block AP was
`0.275436`, only `+0.000102` above P09. The fixed-year differences were also
small and changed sign: P10 exceeded P09 mean block AP by `+0.004690` in 2022
and trailed it by `-0.002189` in 2023. P10 remained above P00 in every tested
year, by `+0.052345`, `+0.023389`, and `+0.020189` mean block AP in
2021--2023 respectively, while routing preserved M00/M01 exactly.

The matched control therefore does not support a meaningful or temporally
stable GroupDRO contribution. Corrected five-year exposure with balanced-year
ERM explains essentially all of P09's gain. P10 is retained as the simpler
leading candidate; P09 remains an ablation showing that adaptive environment
weighting is unnecessary under this prototype budget.

## P11 active-fire routing probe

P11 was a training-free routing test. It retained P00 for clean inputs, used
P10 only inside M06/M07 missing blocks, and used P10 over the full image when
active-fire history was entirely absent in M01. This exactly preserved P00 on
M00 and exactly reproduced P10 on M06/M07.

The proposed M01 route failed its mandatory gate. M01 AP decreased from
P00's `0.299465` to `0.291202` (`-0.008264`), while F1 decreased from
`0.208634` to `0.134932`. The slightly lower focal loss did not compensate for
the degraded ranking and thresholded forecast. P11 was therefore rejected and
was not evaluated on 2022--2023.

This negative result shows that P10's corrected five-year BlockDrop expert is
not automatically interchangeable with a fire-history-loss expert under the
legacy focal weighting. P12 first isolates the suspected focal-alpha inversion
while holding the complete P10 training recipe fixed; a dedicated FireDrop-only
expert remains conditional on that one-variable result.

## Fixed temporal test comparison

| Year | Scenario | P00 AP | P03 AP | P09 AP | P10 AP | P10 minus P00 | P10 minus P09 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2022 | M00 | 0.281701 | 0.281701 | 0.281701 | 0.281701 | 0.000000 | 0.000000 |
| 2022 | M01 | 0.163668 | 0.163668 | 0.163668 | 0.163668 | 0.000000 | 0.000000 |
| 2022 | M06 | 0.128030 | 0.104961 | 0.145384 | 0.149283 | +0.021253 | +0.003899 |
| 2022 | M07 | 0.062677 | 0.036584 | 0.082721 | 0.088202 | +0.025525 | +0.005482 |
| 2023 | M00 | 0.406200 | 0.406200 | 0.406200 | 0.406200 | 0.000000 | 0.000000 |
| 2023 | M01 | 0.136357 | 0.136357 | 0.136357 | 0.136357 | 0.000000 | 0.000000 |
| 2023 | M06 | 0.195445 | 0.208950 | 0.220648 | 0.218045 | +0.022600 | -0.002603 |
| 2023 | M07 | 0.076796 | 0.086391 | 0.096349 | 0.094573 | +0.017778 | -0.001776 |

P03 helps in 2021 and 2023 but harms both block-missingness conditions in 2022.
P09 and P10 are positive in both test years. These results are final reporting
evidence and cannot be used to tune the corrected resolver, ERM, or GroupDRO.

## Execution evidence

- P03 selection: job `20458324`.
- P04: job `20462018`.
- P05: job `20464221`.
- P06: job `20464396`; failed setup job `20464346` produced no result.
- P07: job `20558998` completed on Nibi node `g30` in 10:58 with exit `0:0`.
- P08: job `20561688` completed on Nibi node `g2` in 12:18 with exit `0:0`.
  Checkpoint SHA-256: `d9de805f8a7abd96f435f33deaf6ec254759990f131d8cfed6278eee87c1b142`;
  summary SHA-256: `ecbc80e624a769a02d7c22886cd07aa4265cd7fff21bafe78c38a807d398b30c`.
- P08 scheduling attempts `20561593`, `20561616`, and `20561672` were cancelled
  before allocation and consumed no GPU time.
- P09 selection: job `20563978` completed on Nibi node `g30` in 24:26 with
  exit `0:0`. Checkpoint SHA-256:
  `0b8787f53b024cd176c5e5a977f231d004bf0a4c575f53b37b546db5b01de680`;
  2021 summary SHA-256:
  `b70b8704858fb307685d12b974007c474c6866c57fefc27f5f0d68a516a8a6eb`.
- P09 fixed tests: jobs `20564992` (2022, 9:30) and `20564993` (2023,
  5:12) completed on `g30` with exit `0:0`. Summary SHA-256 values are
  `a78cae9485342f40195357570606551d4d27e7271a73fe559bef0aee5d6440b1`
  and `10c7439632f0dc945c9f58eb8cd303fea66eddf28cbcbe6065f4bbfb5ab9928c`.
- P09 job `20563415` was cancelled after its diagnostics exposed the upstream
  year-index bug. Jobs `20563639` and `20563807` were cancelled before a
  scientific result because of allocation and wall-time constraints.
- P10 selection: job `20566466` completed on Nibi node `g35` in 30:56 with
  exit `0:0`. Checkpoint SHA-256:
  `1f91ed533db7baf36b34a5709665f1ca5831f8ed277d465b2504f0a4e60c0b8b`;
  2021 summary SHA-256:
  `8b9949edbc0198666799c4158f167d3689764e1b397f5c046e4234c2a093aa44`.
- P10 fixed tests: jobs `20582647` (2022, 13:03, `g32`) and `20582648`
  (2023, 7:47, `g33`) completed with exit `0:0`. Summary SHA-256 values are
  `9225bc337cc3ac7690b77fa0bb280ae74b7206d8c5be82c51f3aa96c950d1e58`
  and `0c5ae952ec20db82a8a67586bf3d1df2aafa0a2e327c371d6a2ed5fd8989fafc`.
- P11 selection: job `20588449` completed on Nibi node `g34` in 8:04 with
  exit `0:0`. Summary SHA-256:
  `d8ee79786f08db5026a45688ade33c52c0c78e87d42213656ba3f0ad2f496594`.
  P11 failed the 2021 M01 gate, so no fixed-test jobs were submitted.
- P03 fixed-test jobs: `20465333` (2022) and `20465334` (2023).
- Failed fixed-test setup jobs `20464512`/`20464513` produced no result.

All training and full-dataset evaluation ran through Slurm compute nodes. The
fixed-test results are final reporting evidence, not feedback for another
prototype iteration.
