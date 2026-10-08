# After Fold 2: designing your first fair comparison

This lesson follows the [official Fold 2 tutorial](res18-baseline-slurm.md).
You should have a completed training run, a selected checkpoint and its test
results. The sections below explain how to plan your next comparison; the
linked setup and method guides provide the execution commands.

In Fold 2, you prepared data, trained a model and evaluated its predictions.
Now you will use that experience to design a comparison.

Suppose you change a trained model's learning-rate schedule, give it another
3,000 updates and obtain a better score. The schedule may have helped, but so
may the extra training. To find out, you need a model that receives the same
additional training with the original schedule.

That model is the **matched ERM continuation control** you will prepare in
this lesson. You will use it later to evaluate X22 and X14.

## 1. Why we need a continuation control

The first experiment asks:

> Under the project's fixed data and evaluation protocol, what changes when
> we give the shared T1 foundation another 3,000 updates of ordinary supervised
> training?

The following lesson adds X22 and asks:

> With the same starting weights and continuation budget, does changing only
> the learning-rate schedule improve prediction under missing observations?

We therefore need three versions of the model:

| Role | What happens to the model? | What does it tell us? |
|---|---|---|
| A: frozen B3 reference | Evaluate the starting checkpoint without additional training | Performance before the continuation recipe |
| B: matched ERM control | Start from B3 and perform 3,000 constant-learning-rate supervised updates | Performance after ordinary continuation |
| C: X22 candidate, in the next lesson | Independently start from the same B3 and perform 3,000 cosine-schedule updates | Performance after the schedule intervention |

For the same score definition, write:

```text
Continuation effect       = score(B) - score(A)
Schedule effect           = score(C) - score(B)
Total change from B3      = score(C) - score(A)
```

The total change is the sum of the first two differences. A positive total
change alone does not establish a positive schedule effect. For example, if
ordinary continuation improves the score more than X22 does, the new schedule
has not helped relative to its matched control.

Here, the continuation effect means the effect of the **specified continuation
recipe**, including its fresh optimizer and training setup. It does not isolate
the number of updates from every difference between foundation training and
continuation.

**Discussion:** If you evaluated only A and C, could you tell whether the
schedule helped? Explain which comparison is missing.

## 2. Moving from Fold 2 to the project experiments

You have completed an official reproduction. The project's method experiments
use a different protocol. Preserve your Fold 2 results as a separate experiment;
do not put its AP in the control column of the new comparison.

| Choice | Your completed official Fold 2 | This project's T1 continuation exercise |
|---|---|---|
| Training years | 2018 and 2020 | 2016–2020 |
| Validation year | 2019 | 2021 |
| Test/reporting years | 2021 | Historical 2022/2023; outside this first exercise |
| Model setting | Res18-U-Net, one observation day, all features | Res18-U-Net, one observation day, 40 processed features |
| Starting point | ImageNet encoder initialization for official training | Corrected project B3 wildfire checkpoint |
| Training budget | 10,000 updates | 3,000 additional updates per continuation |
| Checkpoint rule | Best validation AP checkpoint | Final continuation checkpoint |
| Data processing | Pinned official tutorial contract | Project-corrected indexing and training-only normalization |

The same architecture name does not make two scores directly comparable.
Different years, preprocessing, initialization and checkpoint rules can each
change a result.

You will also need the project code. The official tutorial was self-contained;
this exercise uses the project's corrected data processing and training code.
The [new-cluster guide](../research/reproduce.md) explains how to obtain it
through a local Git bundle. Keep this setup separate from your official run so
that each experiment retains its own code and configuration.

The first exercise uses **T1, seed 0 and 2021 validation only**. B3 is the shared
T1 foundation trained with FireDrop and BlockDrop; it is not the clean B0 model
and is not your official Fold 2 checkpoint. You may regenerate B3 using the
[foundation guide](../research/baselines.md), or use an instructor-provided B3
with its provenance and checksum. Every compared continuation must use the
same actual starting checkpoint bytes.

Retraining B3 is useful practice, but it is not necessary for every student to
repeat foundation training to learn this comparison. Record which route you
used. Newly trained B3 weights need not match the historical project's weights.

## 3. What must match?

ERM means empirical risk minimization: fit predictions to the training labels
using the chosen supervised loss. Here it identifies the ordinary supervised
continuation, without adding a new specialist objective or module.

**ERM does not mean clean inputs only.** The project's `control` already uses
the shared stochastic missingness augmentation. Its FireDrop and BlockDrop
decisions each have probability 0.3 and can overlap; sampled block severity is
25% or 50%. X22 retains that distribution. Removing these augmentations from
the control would change the question and make the comparison unfair.

In the project records, “fresh ERM” refers to a new continuation run: load B3's
weights and create a new optimizer. Resuming an interrupted job would instead
restore its optimizer moments and scheduler state as well. Record which operation
you performed; they are different starting conditions.

Before submitting a run, record the following settings in your experiment note:

| Keep matched between ERM and X22 | What you must record |
|---|---|
| Initialization | B3 file identity, checksum and source/completion record |
| Inputs and targets | Corrected dataset version, T1 features, target dates and training years |
| Normalization | The same statistics computed from training years only |
| Supervised training | Loss settings, optimizer settings and starting learning rate |
| Missingness augmentation | Same FireDrop/BlockDrop rules and severity sampling |
| Budget | 3,000 optimizer updates; physical and effective batch size 64 |
| Randomness | Seed 0 for both runs; same sampling/worker setup |
| Checkpoint rule | Final step for both continuations |
| Evaluation | Same 2021 samples, scenario masks, evaluator and metric definitions |
| Intended difference | Constant schedule for ERM; cosine schedule for X22 |

Use the recorded project configuration rather than guessing unspecified values
from a previous tutorial. Matching the seed reduces uncontrolled variation; it
does not guarantee identical behavior across hardware or make one run conclusive.

**Discussion:** A student selects X22's best validation checkpoint but uses
ERM's final checkpoint. What else, besides the schedule, could explain the
result?

## 4. Running the comparison

### Step 1: Save the official experiment

Save your official Fold 2 configuration, selected checkpoint, evaluation,
source revision and Slurm job IDs. Write a short account of which years were
used for training, selection and testing. Create a separate run group for the
project exercise so official and project artifacts cannot be confused.

### Step 2: Prepare the project data and B3 weights

Follow the [research setup guide](../research/reproduce.md) and
[data preparation guide](../research/data-preparation.md). Confirm that the
corrected data and training statistics are available. Identify the B3 source
and its provenance before submitting a continuation. A path that happens to
contain `b3` is not sufficient evidence of its identity.

Review the settings and resource budget with your instructor. Plan one B3
reference evaluation and one ERM continuation, evaluated on 2021. Include a
small smoke check if you have not used this execution path before. Additional
methods, seeds and years can wait until this comparison is complete.

### Step 3: Evaluate B3 before further training

Evaluate frozen B3 using the same project evaluation population and four
missingness scenarios intended for the continuation. Retain that result as A.
Do not substitute its training-time validation score, a different clean-only
evaluation, or the official Fold 2 score.

Here “frozen” means no further training. Evaluation should use inference mode;
it must not adapt the model or update its normalization statistics.

### Step 4: Check the setup, then train the ERM control

Use the T1 `control` role in the
[active method recipe](../research/method-recipes.md). A smoke check establishes
that loading, an update and saving/reloading work. Its checkpoint is not the
starting point for the formal experiment and its score is not a research result.

Start the formal run independently from B3. Use the fixed 3,000-update budget
and preserve the final checkpoint. Do not extend the run because a curve looks
promising or stop early because the result looks disappointing. Those decisions
would alter the comparison plan.

Perform installation, data processing, model checks, training and evaluation
through Slurm. A returned job ID is a submission receipt. Check completion,
exit status, actual completed updates and the saved artifacts before treating
the experiment as finished. Keep failed attempts and their logs distinguishable
from valid completed results.

### Step 5: Evaluate the continuation

Use the same 2021 evaluation contract for B as for A. Confirm sample identities
and preprocessing, not merely equal sample counts. If a training output already
contains a valid matching evaluation, retain it rather than needlessly creating
another run. Use a unique destination for any separate evaluation.

Do not copy missingness masks from a different protocol or average incomparable
summary files. Record the original metric files so another student can locate
each number you report.

### Step 6: Read the results

Check the recorded settings for A and B before interpreting their scores.
Describe which conditions improved and which regressed. An unexpectedly strong
ERM result is worth keeping: it tells you how much a new method must improve
on. Choosing a weaker control seed afterwards would bias the comparison.

## 5. What to measure

Report all four conditions, not only the one that improves:

| Condition / summary | Meaning |
|---|---|
| M00 | Complete inputs |
| M01 | Active-fire history removed |
| M06 | 25% controlled spatial block missingness on dynamic inputs |
| M07 | 50% controlled spatial block missingness on dynamic inputs |
| Primary | Arithmetic mean of M01, M06 and M07 AP |
| Block | Arithmetic mean of M06 and M07 AP |

Static inputs are retained in the block scenarios. AP is computed using the
project evaluator within each scenario; the Primary average is not AP computed
by pooling predictions from the three scenarios. Keep the AP scale explicit:
an absolute difference of 0.01 is one AP point on a 0–100 presentation scale.

Use Primary as the main missingness summary and report clean AP separately.
Do not choose the main metric after inspecting which one improved most.

Copy this table into your experiment note and fill it with your results:

| Run | M00 AP | M01 AP | M06 AP | M07 AP | Primary | Block |
|---|---|---|---|---|---|---|
| A: frozen B3 | | | | | | |
| B: matched ERM | | | | | | |
| B minus A | | | | | | |

All cells must share the same evaluation contract. Historical project averages
across settings, seeds and years are not target values for this single T1/2021
exercise.

## 6. What the results tell you

| Observation | Appropriate interpretation | What it does not establish |
|---|---|---|
| ERM improves Primary over B3 | This continuation recipe helped in this run | A new algorithm is effective |
| ERM improves Primary but reduces clean AP | Continuation introduces a trade-off under the chosen metrics | Uniform robustness improvement |
| ERM does not improve | The completed continuation did not help under this contract | The run is automatically broken, or a different method cannot help |
| Later X22 beats B3 but loses to ERM | Additional training explains the positive total result better than the schedule intervention | Positive X22 attribution |
| Inputs, starting weights or evaluation differ | Repair or relabel the comparison before attributing a gain | A valid positive or negative method result |

A single seed is a pilot. After fixing the comparison, repeat the matched pair
with additional seeds to examine variability. Do not present a single run as
statistically significant or a result under synthetic masks as proof of natural
satellite-outage robustness.

Use 2021 for this exercise's development. The project's 2022/2023 results have
already been examined historically; later evaluation there is historical
reporting, not a newly untouched confirmation set. The official tutorial also
used 2021 as a test year, whereas it is explicitly a validation year here.

## 7. Your experiment report

Submit a short experiment note containing:

1. Your question, predicted outcome and an outcome that would contradict it,
   written before training.
2. The completed comparison plan, including B3 identity and checkpoint rule.
3. Source revision, configurations, job IDs, completion evidence and paths to
   checkpoints and original evaluations.
4. The A/B result table, available training logs/curves and a paragraph explaining
   both improvements and regressions. A lower training loss alone is not evidence
   of improved validation AP.
5. One conclusion supported by this comparison, one claim it cannot support,
   and a proposed next comparison.

A well-executed experiment with no improvement is a successful exercise.
Your report will be assessed on the comparison, the supporting records and your
interpretation. Be ready to explain why frozen B3, clean B0, official Fold 2 and
fresh ERM cannot be substituted for one another.

## 8. Next lessons: X22 first, then X14

For **X22**, start again from the same B3, use the same continuation budget and
change only the schedule. Compare C against your matched ERM B. Do not initialize
X22 from B's final checkpoint: that would give it another stage of training.
Reuse B only if its full contract matches the candidate.

For **X14**, start independently from B3 and change the training missingness
distribution to mixed-severity BlockDrop specialization. Keep the remaining
comparison settings matched and first inspect the specialist in all four
conditions. This tests the value and cost of specialization, including possible
regressions outside its intended condition.

Only then study a condition-specific composition such as X22+X14. Separate
standalone model performance from the system that chooses different models for
different conditions. Selecting ERM for clean inputs makes the system's clean
score equal to ERM by construction; it does not prove the specialist preserves
clean performance. System-level comparisons must also account for the extra
models and continuation budgets.

Once you understand X14's results, X17 provides the next comparison: do two
fixed-severity experts perform better than one mixed-severity expert? X14 then
becomes the closest control, and the additional training and model storage must
be included in the comparison.
