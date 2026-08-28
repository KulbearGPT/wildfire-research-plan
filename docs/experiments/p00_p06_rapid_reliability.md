# P00--P13 Rapid Reliability Prototypes

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
rejected without opening the held-out years. P12 isolated the previously
disabled focal class weighting; enabling the normalized positive weight caused
severe overprediction and was also rejected on 2021. P13's dedicated
corrected-index FireDrop expert produced a small 2021 M01 gain, but the frozen
test comparison changed sign across years. The engineering branch is closed.

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
| P12 corrected-alpha ERM router | full P02 fine-tune | 0.585322 | 0.252888 | 0.070432 | 0.061832 | Rejected: severe overprediction |
| P13 dedicated FireDrop/P10 router | full P00 fine-tune | 0.585322 | 0.304237 | 0.365203 | 0.185669 | 2021 gate passed; fixed years mixed |

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
legacy unweighted focal objective.

## P12 corrected focal-alpha diagnostic

P12 held fixed P10's P02 initialization, corrected resolver, inverse-year
sampler, corruption and sample RNG, seed, batch size, AdamW optimizer, learning
rate, and 3,000-step budget. The only training change was focal `alpha`. The
upstream subclass overwrites the base class's normalized positive weight with
the raw ratio (`761.078570`); the legacy expression then supplies a negative
`alpha`, which makes torchvision skip alpha weighting. P12 normalizes that raw
ratio to `0.998687799344` and passes it as the positive-class weight.

This replacement failed decisively. M01 AP fell from P00's `0.299465` to
`0.252888`, while recall rose to `0.939741` and precision collapsed to
`0.024290`. M06/M07 AP fell from P10's `0.365203/0.185669` to
`0.070432/0.061832`; their mean dropped by `0.209304`. The prevalence-derived
weight makes positive pixels too dominant for this crop and sampling regime.
P12 failed its 2021 gate and was not evaluated on 2022--2023. This rejects the
specific near-one alpha, not every possible moderate class weight.

## P13 dedicated FireDrop expert

P13 started from P00 and retained the corrected five-year resolver,
inverse-year sampler, seed 0, batch size 64, AdamW `1e-4`, 3,000 steps, and the
legacy unweighted focal objective. Its only training corruption was the same
30% FireDrop used by P00. The frozen diagnostic router used P00 on M00, the
P13 expert on complete active-fire loss M01, and P10 inside M06/M07 blocks.

P13 passed the frozen 2021 gate. M01 AP increased from `0.299465` to `0.304237`
(`+0.004771`), while M00 exactly matched P00 and M06/M07 exactly matched P10.
The fixed test result was not temporally stable: M01 AP changed from P00 by
`-0.002058` in 2022 and `+0.019928` in 2023. The expert specialization is a
useful diagnostic but not a robust standalone contribution. No P13 setting is
changed from these reporting-only results, and the engineering branch closes
in favor of explicit latent fire-state inference.

| Year | P00 M01 AP | P13 M01 AP | P13 minus P00 |
|---:|---:|---:|---:|
| 2021 | 0.299465 | 0.304237 | +0.004771 |
| 2022 | 0.163668 | 0.161609 | -0.002058 |
| 2023 | 0.136357 | 0.156285 | +0.019928 |

## Fire belief-state 2021 screen — complete, no promotion

This is a separate, frozen follow-up to the completed P00--P13 engineering
branch. The approved design is `2e9bf3c` and the execution plan is `b560075`.
The immutable implementation lineage is `eaad31f` (common data/state
contract), `7965c44` (temporal attention), `33a1bc6` (reconstruction-first),
`9102655` (recurrent filter), `71cc582` (matched trainer/evaluator/runner
integration), and `baaa62c` (crop-leakage fix). All six submissions use the
clean immutable worktree
`/scratch/kulbear/wildfire-research-plan/.worktrees/belief-submit` at
`baaa62c`.

The frozen P00 input is
`/project/6085198/kulbear/wildfire/runs/prototype-P00-FireDrop-C00-20398173/completed.json`.
The protocol crosses recurrent `filter`, temporal `attention`, and
`reconstruction`-first state proposals with history `T=1` and `T=5`. It keeps
the corrected resolver, inverse-year sampler, seed 0, 3,000 steps, effective
batch 64, AdamW `1e-3`, the frozen P00 Res18-U-Net spread predictor, and the
2021 M00/M01/M06/M07 evaluation fixed. The active-fire observation correction
preserves valid pixels exactly; `T=1` and `T=5` predict the same sixth-day
target. Training corruption is the uniform M01/M06/M07 choice after one shared
crop/augmentation. Filter and attention use forecast plus `0.1` state loss;
reconstruction uses state loss only.

The final implementation fix is material to this protocol: its
target-independent augmentation chooses crop geometry without ranking candidate
crops by the target, carries the target through the selected geometric
transform, and uses `drop_last=True` so every optimizer microbatch has the
configured physical size. The focused CPU verification jobs recorded for this
implementation are `20651743`, `20652580`, and `20653344`.

The submissions were made independently on account `def-vislearn_gpu`, QoS
`interac`, with one `nvidia_h100_80gb_hbm3_3g.40gb` slice, eight CPUs, 32 GB,
and a one-hour limit. The intended run roots were absent before submission
because their scheduler IDs were newly allocated.

The first submission (`20653449`--`20653454`) never reached training. All six
jobs terminated at runner startup because `SLURM_SUBMIT_DIR` inherited
`/scratch/kulbear`, which is not a Git repository; consequently no model,
dataset, checkpoint, or evaluation was loaded and no scientific result was
produced. This was a submission-context error rather than a code or compute
failure. The jobs were resubmitted from the immutable worktree itself, leaving
the runner and scientific protocol unchanged.

| Method | History | Replacement job | Intended run root | Immediate acceptance state |
|---|---:|---:|---|---|
| filter | 1 | `20655223` | `prototype-fire-belief-filter-t1-20655223` | `R` on `g32` |
| filter | 5 | `20655224` | `prototype-fire-belief-filter-t5-20655224` | `R` on `g32` |
| attention | 1 | `20655225` | `prototype-fire-belief-attention-t1-20655225` | `R` on `g35` |
| attention | 5 | `20655226` | `prototype-fire-belief-attention-t5-20655226` | `PD` `(QOSMaxJobsPerUserLimit)` |
| reconstruction | 1 | `20655227` | `prototype-fire-belief-reconstruction-t1-20655227` | `PD` `(QOSMaxJobsPerUserLimit)` |
| reconstruction | 5 | `20655228` | `prototype-fire-belief-reconstruction-t5-20655228` | `PD` `(QOSMaxJobsPerUserLimit)` |

At the first coarse monitor, filter T=1 (`20655223`) and attention T=1
(`20655225`) completed, while their T=5 counterparts `20655224` and `20655226`
were killed for GPU memory before evaluation. The pre-registered bounded
fallback was implemented in `c8ae91f`: only T=5 may change from physical batch
64/accumulation 1 to physical batch 32/accumulation 2, preserving effective
batch 64 and every scientific setting. A one-core CPU Slurm job, `20657191`,
validated the runner syntax and diff before the commit. Filter and attention
were resubmitted as soon as their terminal failures were observed;
reconstruction T=5 was left untouched while running and received the identical
fallback only after its own OOM terminal state:

| Method | History | Replacement job | Physical batch / accumulation | Immediate state |
|---|---:|---:|---:|---|
| filter | 5 | `20657718` | `32 / 2` | `R` on `g31` |
| attention | 5 | `20657719` | `32 / 2` | `R` on `g32` |
| reconstruction | 5 | `20659132` | `32 / 2` | `R` on `g31` |

The first two fallback jobs reached only steps 1500--1600 after about 32
minutes, so the one-hour limit could not cover 3,000 steps plus evaluation.
Nibi denied an in-place time-limit extension. Jobs `20657718`, `20657719`, and
the newly started `20659132` were therefore cancelled before their predictable
timeouts and replaced immediately with the same committed code and exact
scientific configuration, changing only the resource wall time to two hours:

| Method | History | Final replacement job | Physical batch / accumulation | Immediate state |
|---|---:|---:|---:|---|
| filter | 5 | `20659221` | `32 / 2` | `R` on `g33` |
| attention | 5 | `20659234` | `32 / 2` | `R` on `g30` |
| reconstruction | 5 | `20659235` | `32 / 2` | `R` on `g33` |

Attention T=5 completed all 3,000 training steps in `20659234` and wrote its
checkpoint, but the default evaluation batch of 64 caused a separate OOM
before any scenario completed. The checkpoint was preserved and no training
was repeated. Eval-only job `20662156` runs the archived project and checkpoint
with batch 8 into the new non-overwriting `results-2021-b8` directory; all
metric definitions and samples remain unchanged.

The first eval-only attempt also terminated before a scenario: `MaxRSS` reached
the 32 GB allocation and the traceback identified a killed DataLoader worker,
confirming host-memory rather than GPU-memory pressure. Its replacement,
`20663826`, keeps batch 8, reduces workers from 8 to 2, requests 64 GB host
memory, and writes to `results-2021-b8-w2`. No checkpoint or metric setting is
changed.

That two-worker attempt reached `MaxRSS` of about 64 GB and was likewise killed
during the repeated P00 traversal. By then filter `20659221` and reconstruction
`20659235` had also completed 3,000 training steps and preserved checkpoints
before their default evaluations OOMed. Final eval-only jobs use batch 8,
`num_workers=0`, and 128 GB host memory, avoiding worker copies without changing
examples or metrics: filter `20666788`, attention `20666789`, and reconstruction
`20666790`. Each writes to its own `results-2021-b8-w0` directory; all three
were initially pending for priority.

All three worker-free jobs still exhausted 128 GB. Static diagnosis found the
actual cause: controlled evaluation materialized each full-resolution T-day
HDF5 slice before applying the 128-pixel center crop. Commit `b75380c` moves
that identical center slice into the HDF5 read and retains the old padding path
for images smaller than the crop. Focused CPU Slurm RED job `20669325` failed
before the helper existed; GREEN job `20670008` passed four targeted crop,
spatial-mask, and T=1/T=5 population cases. The three existing checkpoints are
now evaluated without retraining by filter `20670986`, attention `20670987`,
and reconstruction `20670988`, using batch 32, four workers, 64 GB host memory,
and separate `results-2021-directcrop-b32-w4` directories. They were initially
pending for priority.

Review found the first direct-crop test used spatially constant planes. The
test-only follow-up `6e28efa` encodes row/column position and compares raw,
target, and stale reads against the old materialized oracle for larger, equal,
and single-axis-smaller images. CPU Slurm job `20671057` passed all nine related
cases, and the focused re-review approved the fix.

Direct-crop attention job `20670987` completed with M00 exact and APs
`0.585323/0.265400/0.316956/0.130732` for M00/M01/M06/M07. Its corrupted AP
deltas were `-0.034065/-0.000050/+0.001556`; mean delta `-0.010853` and only
one improved scenario make the frozen screen fail. Filter `20670986` completed
M00/M01 before host OOM in M06, while reconstruction `20670988` OOMed during
the M00 P00 traversal. Their checkpoints remain valid. Only these two failed
evaluations were resubmitted, using direct-crop, batch 8, no workers, and 256 GB
host memory: filter `20671736` and reconstruction `20671737`. Both were
immediately running on `g31`; no training or metric definition changed.

Those two processes each completed exact M00 but exhausted 256 GB while moving
to the next scenario, confirming cross-scenario retained memory rather than a
single-scenario requirement. Commit `86a8abf` adds only a bounded `--scenario`
evaluation mode; CPU Slurm RED/GREEN jobs `20675333` and `20675624` verify its
selection contract. M00 is reused, and the six remaining isolated evaluations
are filter M01/M06/M07 `20675921/20675922/20675923` and reconstruction
M01/M06/M07 `20675924/20675925/20675926`. Each uses the same checkpoint,
direct-crop evaluator, batch 8, no workers, and 256 GB; all were initially
pending for priority.

All six isolated jobs completed. CPU metadata job `20676758` read only the six
small checkpoint records and confirmed the trainable parameter counts below.
Mean delta is the mean M01/M06/M07 AP change against the P00 result evaluated
in the same route. A/B minus C compares corrupted-scenario mean AP with the
matched reconstruction baseline at the same history. P00 is the direct delta
baseline; P10 and P13 are historical promotion references rather than members
of the matched T=1/T=5 factorial, so their parameter and loss entries are not
directly comparable.

| Method | T | Trainable params | Final loss | M00 AP / exact | M01 AP | M06 AP | M07 AP | Mean delta | A/B minus C | Screen |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| P00 FireDrop baseline | -- | full C00 | -- | 0.585322 / reference | 0.299465 | 0.317005 | 0.129176 | 0.000000 | -- | reference |
| P10 block baseline | -- | full P02 fine-tune | -- | 0.585322 / yes | 0.299465 | 0.365203 | 0.185669 | +0.034897 | -- | pass (reference) |
| P13 routed baseline | -- | full P00 fine-tune | -- | 0.585322 / yes | 0.304237 | 0.365203 | 0.185669 | +0.036488 | -- | pass (reference) |
| filter | 1 | 391,745 | 0.003879 | 0.585322 / yes | 0.260952 | 0.316738 | 0.131167 | -0.012264 | +0.019116 | fail |
| filter | 5 | 391,745 | 0.012004 | 0.585321 / yes | 0.255844 | 0.317777 | 0.133643 | -0.012794 | +0.013886 | fail |
| attention | 1 | 1,521 | 0.006301 | 0.585322 / yes | 0.274733 | 0.316787 | 0.130474 | -0.007884 | +0.023496 | fail |
| attention | 5 | 1,521 | 0.010850 | 0.585323 / yes | 0.265400 | 0.316956 | 0.130732 | -0.010853 | +0.015828 | fail |
| reconstruction | 1 | 391,889 | 0.004719 | 0.585322 / yes | 0.206976 | 0.315143 | 0.129389 | -0.031380 | -- | fail |
| reconstruction | 5 | 412,625 | 0.010589 | 0.585321 / yes | 0.220937 | 0.315243 | 0.129423 | -0.026680 | -- | fail |

Filter and attention exceed their matched reconstruction controls, so the
learned task-oriented state heads contain more useful signal than explicit
reconstruction alone. That relative result is insufficient for promotion:
every method/history loses substantially on full FireDrop M01, no configuration
has positive corrupted mean delta, and all six fail the frozen signal screen.
The strongest new M01 result, attention T=1 at `0.274733`, trails P00 by
`0.024732` and P13 by `0.029504`. The strongest new block results are filter
T=5 at `0.317777/0.133643`; these improve P00 by only
`0.000772/0.004467` and trail P10 by `0.047426/0.052026` on M06/M07. In
contrast, the P10/P13 reference rows have positive mean deltas and pass the
same signal rule. T=5 therefore does not rescue filter or attention or
establish useful historical state inference under this interface.

The final decision is **stop this three-direction screen without tuning and do
not open 2022--2023**. The direct-HDF5 crop and isolated-scenario evaluation
remain as bounded operational fixes, but none of the six models is a research
candidate. All large data, training, model, and metric computation ran through
Slurm compute jobs. The login node was limited to source inspection and edits,
small JSON/log reads, Git operations, submissions, and coarse scheduler checks;
it did not load a dataset or model, run pytest, or execute CPU/GPU scientific
work.

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
- P12 selection: job `20595907` completed on Nibi node `g30` in 27:43 with
  exit `0:0`. Checkpoint SHA-256:
  `3747f1ac13ae4e22bf8479e4e7067fb2ee0b11181863a5a0ecd9bead8e1e5b85`;
  2021 summary SHA-256:
  `a838507e007f30e92a00a2f34724b90fa3573ee15f0bec78421bf4a4892afbd7`.
  Setup job `20591988` failed before training after exposing the raw-ratio
  checkpoint metadata; it produced no scientific result. P12 failed the 2021
  gate, so no fixed-test jobs were submitted.
- P13 selection: job `20611586` completed on Nibi node `g36` in 25:36 with
  exit `0:0`. Checkpoint SHA-256:
  `d69cf3f22eba5105c47d1057eed98601f207b0f42a804781d26c9c31e61f4241`;
  2021 summary SHA-256:
  `b213631a96a8ec8c759e7a6b7c842412de0471f8c472615215992544617ef47b`.
- P13 fixed tests: jobs `20612762` (2022, 15:35, `g32`) and `20612763`
  (2023, 8:29, `g32`) completed with exit `0:0`. Summary SHA-256 values are
  `9a3b9febac42131fbdb1ff2a99ef82974d28021871d2f9396d39b06af5a40174`
  and `42d736f9a7b19df6930635bdbd7ddcf5c8c71f14bf3d17ed8f03c573a5eff790`.
- P03 fixed-test jobs: `20465333` (2022) and `20465334` (2023).
- Failed fixed-test setup jobs `20464512`/`20464513` produced no result.

All training and full-dataset evaluation ran through Slurm compute nodes. The
fixed-test results are final reporting evidence, not feedback for another
prototype iteration.
