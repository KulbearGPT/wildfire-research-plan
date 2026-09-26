# Experimental evaluation

## Task and protocol

We evaluate next-day active-fire forecasting under controlled missing
observations. The target is a VIIRS-derived active-fire proxy rather than a
complete fire perimeter. Training and normalization use 2016–2020. We separate
2021 validation and method selection from reporting on the historical 2022 and
2023 test years. These test-year results have been inspected during the research
campaign and are not treated as untouched confirmation for newly selected claims.

The two evaluation settings share target dates and contain 3181, 2856, and 2102
samples in 2021, 2022, and 2023, respectively. Each target has spatial size
128 by 128 pixels. T1 uses Res18-U-Net with 40 features and one observation day;
T5 uses Res18-UTAE with 33 features per day and five observation days. Because
the architecture and features also change, the T1/T5 comparison does not isolate
the effect of additional history.

We report four observation conditions: M00 uses complete inputs; M01 removes
the active-fire history; M06 and M07 impose the recorded 25% and 50% spatial
block missingness on dynamic inputs, including fire observations, while retaining
static features. Average precision is computed across all evaluated
pixels within a scenario and year. The primary score is the arithmetic mean of
M01, M06, and M07 AP, and the block score is the mean of M06 and M07 AP. M00 is
reported separately. Differences are absolute AP units. Scenario-averaged AP
does not equal AP obtained by pooling all scenarios into one prediction list.

## Compared continuations and compositions

All mainline continuations start from the shared corrected B3 checkpoint in T1
or B5 checkpoint in T5. The recorded continuation protocol uses 3000 optimizer
steps, physical and effective batch size 64, seeds 0, 1, and 2, and final-step
checkpoints. Fresh ERM is the constant-learning-rate continuation under the same
initialization and budget. X22 changes its schedule to cosine decay. X14
specializes training to block missingness while mixing the two severities. X17
replaces that mixed training distribution with two separate fixed-severity
continuations, one for 25% and one for 50% blocks.

The current complete system selects the fresh ERM score for M00, the X22 score
for M01, and the corresponding X17 expert for M06 and M07. This system therefore
retains four component models per setting and seed. We construct the system
evaluation from the recorded scenario scores; no new router inference or
per-sample logit mixing is implied. A companion X22+X14 composition uses the
mixed-severity specialist for both block conditions, providing a system-level
control for the extra severity split.

The preserved historical complete-route artifact used X22 for M00. We report
that artifact separately from the newly derived ERM-clean composition. Both
have identical missingness primary and block scores. The zero M00 difference
of the current composition follows from reusing fresh ERM, and should not be
interpreted as measured clean robustness of the experts.

## System performance

The recorded X22+X17 missingness composition improves mean primary AP over
fresh ERM in each setting/year cell. Three-seed gains are 0.010386, 0.006153,
and 0.010109 in T1 and 0.028162, 0.020169, and 0.012873 in T5 for 2021, 2022,
and 2023, respectively. The unweighted mean across the 18 setting/year/seed
cells is 0.014642 AP. This is a positive system effect in the evaluated
conditions, but it does not reach the campaign's historical 0.020 magnitude
target against fresh ERM.

The same missingness composition improves mean primary AP by 0.034527 over
the frozen B3/B5 reference. The larger difference includes the effect of
continuation relative to those source checkpoints. We therefore use fresh ERM
as the principal method comparator and report the frozen-reference result as
a separate total-effect comparison.

For every setting/year cell, the accompanying figures show all three paired
seed differences and the tables report the mean and sample standard deviation.
These summarize continuation variability from shared initial checkpoints. We
do not interpret three seeds as independent wildfire populations or infer
event-level statistical significance from these summaries.

## Schedule and severity attribution

X22 alone improves mean primary AP by 0.012009 over fresh ERM across the same
18 cells. Its three-seed primary effects remain positive in all six
setting/year cells. This supports the value of the schedule change within the
matched continuation protocol. It does not establish a novel optimization
algorithm or uniform improvement for every scenario and seed.

X17 also has a positive total effect over fresh ERM: its overall primary and
block gains are 0.008250 and 0.012376 AP. The closer comparison with X14 gives
a more limited conclusion. The three-seed block effect of severity
factorization is negative for T1/2023 (-0.000649), T5/2022 (-0.000898), and
T5/2023 (-0.001614). Consequently, gains over ERM do not establish that two
fixed-severity experts are consistently better than a mixed-severity expert.
We retain these failed attribution cells in the main analysis. Composing the
existing X22 and X14 scores gives a mean primary gain of 0.014287 AP over ERM.
X22+X17 exceeds this closer system control by only 0.000355 AP across all years.
The difference is 0.001516 AP on 2021 selection data and -0.000226 AP across
the historical 2022/2023 years. X22+X17 also exceeds X22 alone by 0.002633 AP
on the all-year average. Thus the current evidence supports a useful composed
system, but not a robust advantage of the extra fixed-severity expert over the
mixed specialist. These comparisons use recorded evaluations and introduce no
new training or inference.

## Resource accounting and scope

A 3000-step per-model budget does not make the complete system equal in total
training cost to a single ERM continuation. The current route retains four
continuations per setting/seed, while X22+X14 uses three. These nominal counts
exclude shared foundation training and do not estimate actual accelerator
hours or inference latency. The material package reports component counts and
optimizer-step budgets as accounting quantities, without presenting them as
measured throughput or an efficiency claim.

The results concern controlled missing observations in two existing
architecture/feature configurations. They do not establish natural operational
robustness, a causal history-length effect, or superiority over all related
methods. Relevant archived experiments remain in a separate evidence catalogue,
with their original comparator, setting, outcome, and maintenance status.

## Source note for authors

The experimental values above come from the completed X17, X22, full-route,
and frozen-reference sections of `docs/experiments/t1_t5_innovations.md` and
their retained JSON artifacts. The population counts follow
`docs/research/evaluation-population.md`; the active protocol follows
`docs/CODE_LIFECYCLE.md`. Pixel-level AP semantics follow
`reproductions/wsts_fast_track/evaluate_missingness.py`. The newly derived
compositions and raw-versus-documented M00 discrepancy are identified separately
in this package. This draft is experimental prose rather than a complete paper;
literature comparison and formal dataset/method citations require the final
manuscript's verified reference list.
