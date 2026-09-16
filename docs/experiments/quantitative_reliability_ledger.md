# T=1 Quantitative Reliability Results

This is the source of truth for the active research line. All learned methods
use Res18-U-Net with `T=1`, the corrected pooled-year resolver, 2016--2020
training, 2021 validation, and fixed 2022--2023 tests. AP is computed on the
same controlled populations for M00 clean input, M01 missing active-fire
history, and M06/M07 structured missing blocks over 25%/50% of dynamic input.

Rejected and out-of-scope experiments are recorded once in
[`rejected_experiments.md`](rejected_experiments.md). Their code is recoverable
from Git branch `archive/pre-t1-cleanup-2026-09-04`; their Nibi artifacts are
stored under
`/project/6085198/kulbear/wildfire/archive/pre-t1-cleanup-2026-09-04/runs`.

## Foundation baselines

| Baseline | Result | Role |
| --- | ---: | --- |
| No-fire rule | test event-macro AP 0.000531 | deterministic lower bound |
| Latest-mask persistence | test event-macro AP 0.070538 | deterministic temporal baseline |
| Official Res18-U-Net T=1, trained Fold 2 | AP 0.554664 | executable training reproduction |
| Twelve official released folds | AP 0.452764 ± 0.088217 | released-weight reproduction |

The WSTS+ audit contains 999 valid event HDF5 files: 653 train events,
156 validation events, and 190 test events. The official reproduction is
documented in [`res18_unet_t1_reproduction.md`](res18_unet_t1_reproduction.md);
the data and rule baselines are documented in [`phase0.md`](phase0.md).

## Retained learned results

| Model | Year | M00 AP | M01 AP | M06 AP | M07 AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 2021 | 0.580988 | 0.039387 | 0.323805 | 0.133672 |
| B0 | 2022 | 0.278852 | 0.006669 | 0.135278 | 0.071779 |
| B0 | 2023 | 0.412674 | 0.010101 | 0.205411 | 0.078691 |
| B2 | 2021 | 0.559282 | 0.262091 | 0.308248 | 0.125715 |
| B2 | 2022 | 0.276522 | 0.130812 | 0.131567 | 0.066482 |
| B2 | 2023 | 0.388503 | 0.113583 | 0.191418 | 0.072856 |
| B3 | 2021 | 0.558318 | 0.276890 | 0.343010 | 0.172086 |
| B3 | 2022 | 0.285884 | 0.153926 | 0.150064 | 0.086647 |
| B3 | 2023 | 0.400610 | 0.113681 | 0.214931 | 0.096476 |
| D1-ERM | 2021 | 0.583021 | 0.251367 | 0.361236 | 0.176152 |
| D1-ERM | 2022 | 0.296328 | 0.127506 | 0.150888 | 0.083153 |
| D1-ERM | 2023 | 0.416895 | 0.093385 | 0.225807 | 0.101777 |
| D1-KL | 2021 | 0.584376 | 0.267385 | 0.364239 | 0.184740 |
| D1-KL | 2022 | 0.301379 | 0.140425 | 0.161299 | 0.099263 |
| D1-KL | 2023 | 0.418699 | 0.100499 | 0.230400 | 0.103755 |
| D2-STD | 2021 | 0.583073 | 0.301982 | 0.368775 | 0.189082 |
| D2-STD | 2022 | 0.281631 | 0.150352 | 0.148539 | 0.087816 |
| D2-STD | 2023 | 0.415177 | 0.138737 | 0.222552 | 0.099945 |
| D12-SARP | 2021 | 0.585151 | 0.302867 | 0.376175 | 0.195433 |
| D12-SARP | 2022 | 0.284437 | 0.165423 | 0.155997 | 0.097884 |
| D12-SARP | 2023 | 0.413585 | 0.137976 | 0.225100 | 0.100707 |

Each row contains 3,181 samples in 2021, 2,856 in 2022, and 2,102 in 2023.
The former 2,312 figure was a documentation error: all 21 original result
summaries agree with these counts, and all 84 AP cells above are unchanged.
See the [source verification](../research/evaluation-population.md).
The retained training jobs are B0 `21093263`, B2 `21093265`, B3 `21093266`,
D1-ERM/KL `21102676/21102677`, D2-STD `21094929`, and D12 `21122172`.
Held-out jobs are `21098505/21098506`, `21099907`, `21105328`, and `21122938`.

## Quantified directions

| Direction | Matched change | 2021 delta | 2022 delta | 2023 delta | Three-year mean | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| R1 FireDrop specialist | B0 M01 → observable B0/B2 route | +0.222704 | +0.124143 | +0.103482 | +0.150109 | retain |
| R2 BlockDrop | B2 → B3 mean M06/M07 | +0.040566 | +0.019331 | +0.023566 | +0.027821 | retain |
| R3 predictive consistency | D1-ERM → D1-KL mean M01/M06/M07 | +0.009203 | +0.013146 | +0.004561 | +0.008970 | retain |
| R4 SARP total | D1-ERM → D12 mean M01/M06/M07 | +0.028574 | +0.019252 | +0.014271 | +0.020699 | retain at T=1 |
| R4 SARP module | D2-STD → D12 mean M06/M07 | +0.006875 | +0.008763 | +0.001654 | +0.005764 | positive every year |

R1 and R2 jointly form the corruption-training contribution. R3 supplies a
matched consistency contribution without extra inference parameters. R4 adds
1,088 reliability-prompt parameters and routes 25% missing blocks to deep
encoder prompts and 50% blocks to an input token using the fixed observed
severity threshold `0.375`.

## Claim boundary

B2 is a specialist: its M00 AP changes by -0.021705, -0.002329, and -0.024170
relative to B0, so the reported FireDrop contribution uses the observable
B0/B2 route. B3 adds BlockDrop and stays within the frozen clean guardrail in
its matched comparison with B2.

D12 is an incremental module on the D2-STD recipe. Its M00/M01 deltas relative
to D2-STD are +0.002079/+0.000886 in 2021, +0.002806/+0.015071 in 2022, and
-0.001593/-0.000762 in 2023. Its full-stack 2022 M00 delta relative to D1-ERM
is -0.011891, so the evidence does not support universal clean dominance.

These results support controlled missingness on the published WSTS+ data
contract. They do not establish natural missingness, operational deployment,
or physical fire-perimeter forecasting. The 2022--2023 results are final tests
and must not be used to select another architecture or threshold.

## Next experiment

The next justified compute is confirmation of the unchanged T=1 recipe with
additional seeds or a longer fixed budget. The priority is D12-SARP plus its
D2-STD control; B0/B2/B3 and D1-ERM/KL remain the comparison chain. Any new
idea starts from this retained chain and is screened on 2021 before one fixed
2022--2023 evaluation.
