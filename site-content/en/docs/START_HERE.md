# Start here: a guide for new students

Course website: [Home](../index.html) · [Research foundations](../related-work/index.html) · [Baseline reproduction](../baseline-reproduction/index.html).

Welcome to the wildfire forecasting project. This page takes you from understanding the task to your first experiment and then to the existing research. **For your first practical exercise, complete one official fold. You do not need to rerun every experiment.**

## 1. Understand the research question

We study how to predict next-day active fire when satellite inputs have predefined missing observations. The model outputs a fire probability for each pixel. The target is an active-fire proxy, not a complete fire perimeter. The existing evidence primarily concerns controlled missingness.

Start with the problem definition, research process and experimental contract in the [research roadmap](research-roadmap.md). Understand why data audits, index corrections and baselines precede method comparisons. The roadmap preserves the earlier teaching narrative; **use the [research handoff guide](research/reproduce.md) for current methods, evidence and execution status**.

After reading, explain in your own words: what goes into the model, which day it predicts, what the label represents, and how we decide whether a change helps.

## 2. Choose your first experiment

| Your goal | Starting point | First milestone |
| --- | --- | --- |
| Learn the full download, training and evaluation workflow using only official code | **[Official Res18-U-Net practical tutorial](tutorials/res18-baseline-slurm.md)** (recommended starting point) | Start in an empty personal directory, prepare the environment and data, pass the smoke check, then train and evaluate Fold 2 |
| You know the workflow and want the project's corrected B0 | [Project B0 tutorial](tutorials/project-b0-slurm.md) | Follow the B0 route using the fixed chronological split and seed 0; this does not involve selecting another fold |
| You know the baseline and want to migrate the environment or reproduce our methods | [New-cluster reproduction guide](research/reproduce.md) | Configure personal paths and Slurm resources, prepare data and required weights, then choose one method and its matched control |

**Official Fold 2 and project B0 are different experiments.** The former uses the official split of the original four-year dataset and 10,000 optimizer steps. The latter uses a fixed chronological split of eight years, corrected indexing and training statistics, with 3,000 steps. Their score difference is not a method gain.

The first route does not require cloning this project, using your supervisor's environment or accessing their wildfire checkpoints. All twelve folds are an optional extension; do not launch them all for your first exercise.

## 3. Prepare before running commands

- Your own cluster login, CPU/GPU accounts, and the GPU types and partitions available at your site.
- A personal experiment directory that compute nodes can read and write, with sufficient storage quota.
- The Python/module environment required by the tutorial and a supported download route from compute nodes.
- An experiment notebook for configurations, commit IDs, job IDs and result paths.

The tutorials give the exact directory, storage and command instructions. Use your site's accounts, module names and GPU names; do not copy a supervisor's personal paths or Slurm account.

Use login nodes only for editing, Git, small log reads and job submission/status checks. **Run installation, downloads, data conversion, training and model evaluation through Slurm.** A job ID from `sbatch` confirms submission only. Check for `COMPLETED`, exit code `0:0`, and the expected artifacts. Do not submit duplicate experiments while a job is queued.

## 4. What counts as completing onboarding?

For the official tutorial, bring the following to your supervisor:

- An explanation of the dataset, year split, T=1 input and next-day target.
- Successful environment and data preparation jobs and a passing smoke check.
- A completed Fold 2 training run with inspectable step count, logs and selected checkpoint.
- The tutorial's prescribed checkpoint evaluation, including AP and the corresponding split.
- A clear distinction between your training result, an author's released-checkpoint result and the paper's multi-fold aggregate.

A smoke check establishes that the execution path works; it does not reproduce the paper's performance. If scores differ, check the split, preprocessing, budget, checkpoint selection and metric definition first. Do not repeatedly adjust choices on the test set to chase a target number.

Keep at least the following in your experiment notebook:

```text
Experiment purpose and tutorial used:
Code commit and environment configuration:
Data version/preparation records and year split:
Model, seed, batch size and training steps:
Slurm job ID, final state and exit code:
Log, checkpoint and evaluation-result paths:
Metrics, comparison source, differences and unresolved questions:
```

For the project B0 route, replace Fold 2 above with that tutorial's B0 training and evaluation protocol. Keep the two acceptance criteria separate.

## 5. Move from reproduction to research

Read the [positive-signal catalog](research/positive-signals.md), followed by the [negative and unrun archive](research/negative-results.md). The positive catalog includes improvements on a single seed, metric or scenario. **Inclusion is not confirmation of a contribution.** An unrun experiment or invalid comparison is not evidence that a method fails.

After selecting a direction with your supervisor, find its nearest control in the [method recipes](research/method-recipes.md). Reproduce the existing comparison before discussing changes. For base weights, see [baseline training and export](research/baselines.md); for teacher models or distillation, see [teacher generation and use](research/teachers.md).

Before every experiment, write down the hypothesis, comparator, fixed conditions, and observations that would support or contradict the hypothesis. The project's historical test years have already been evaluated; agree on a confirmation plan before designing new methods.

## 6. Where to find help

| Question | Documentation |
| --- | --- |
| Installing, configuring and submitting jobs on a new server | [New-cluster reproduction guide](research/reproduce.md) |
| Downloading, converting, repairing data and computing statistics | [Data preparation](research/data-preparation.md) |
| Obtaining, transferring or regenerating weights | [External artifacts](research/artifacts.md), [baselines](research/baselines.md), [teachers](research/teachers.md) |
| Which execution checks actually ran, and what they establish | [Qualification record](research/qualification.md) |
| Why historical evaluation sample counts had inconsistent wording | [Evaluation-population audit](research/evaluation-population.md) |

When reporting a problem, include the submitted command, code commit, job ID, exit state and first error in the log. Preserve failed-run artifacts instead of overwriting them. Do not rerun a model on a login node to debug it.
