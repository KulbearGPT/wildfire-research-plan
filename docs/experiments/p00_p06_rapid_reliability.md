# P00--P07 Rapid Reliability Prototypes

## Outcome

The rapid spatial-reliability branch is complete. P03 was the strongest 2021
prototype, but its fixed temporal test comparison was inconsistent across
years. It is therefore retained as a diagnostic result rather than promoted as
the main method. A later stochastic belief-residual probe (P07) improved P00
under block missingness but collapsed to an effectively deterministic
correction and did not beat P04. P00 remains the accepted mainline checkpoint.

## 2021 selection results

| Prototype | Added trainable parameters | M00 AP | M01 AP | M06 AP | M07 AP | Decision |
|---|---:|---:|---:|---:|---:|---|
| P00 FireDrop | full C00 training | 0.585322 | 0.299465 | 0.317005 | 0.129176 | Mainline baseline |
| P03 P00/P02 spatial router | 0 | 0.585322 | 0.299465 | 0.350878 | 0.166982 | Selected for final check |
| P04 `1x1` residual | 17 | 0.585322 | 0.299465 | 0.332083 | 0.139913 | Compute control |
| P05 `3x3` residual | 145 | 0.585322 | 0.299465 | 0.327723 | 0.138767 | Rejected |
| P06 final-block router | adapted tail only | 0.585322 | 0.299465 | 0.330240 | 0.141035 | Rejected |
| P07 stochastic belief residual | 4,945 | 0.585322 | 0.299465 | 0.331009 | 0.140771 | Rejected: variance collapse |

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

## Fixed temporal test comparison

| Year | Scenario | P00 AP | P03 AP | P03 minus P00 |
|---:|---|---:|---:|---:|
| 2022 | M00 | 0.281701 | 0.281701 | 0.000000 |
| 2022 | M01 | 0.163668 | 0.163668 | 0.000000 |
| 2022 | M06 | 0.128030 | 0.104961 | -0.023069 |
| 2022 | M07 | 0.062677 | 0.036584 | -0.026093 |
| 2023 | M00 | 0.406200 | 0.406200 | 0.000000 |
| 2023 | M01 | 0.136357 | 0.136357 | 0.000000 |
| 2023 | M06 | 0.195445 | 0.208950 | +0.013506 |
| 2023 | M07 | 0.076796 | 0.086391 | +0.009595 |

The spatial expert helps in 2021 and 2023 but harms both block-missingness
conditions in 2022. This violates the cross-year robustness requirement. No
P03 promotion or further tuning against 2022--2023 is authorized.

## Execution evidence

- P03 selection: job `20458324`.
- P04: job `20462018`.
- P05: job `20464221`.
- P06: job `20464396`; failed setup job `20464346` produced no result.
- P07: job `20558998` completed on Nibi node `g30` in 10:58 with exit `0:0`.
- P03 fixed-test jobs: `20465333` (2022) and `20465334` (2023).
- Failed fixed-test setup jobs `20464512`/`20464513` produced no result.

All training and full-dataset evaluation ran through Slurm compute nodes. The
fixed-test results are final reporting evidence, not feedback for another
prototype iteration.
