---
name: wildfire-experiment
description: Prepare or execute a wildfire research experiment with a matched control and bounded Slurm budget. Use for experiment design, reproduction, or method trials, not routine edits or existing-job diagnosis.
---

# Wildfire experiment

Complete the requested experimental stage with a credible comparison and
recoverable evidence. A preparation-only request ends at a reviewable plan,
configuration, and command; it does not authorize job submission.

Paths below are relative to this skill directory. Shell commands run from the
repository root. Reuse the current task's authorization and experiment note.

## Establish only the missing experimental decisions

Find the method and closest control in the
[inventory](../../../docs/research/method-inventory.json) and
[recipes](../../../docs/research/method-recipes.md).
For an existing direction, inspect its linked result and limitations before
designing another trial. For a new direction, state one falsifiable prediction
and the cheapest valid comparison that could change the next decision.

Record the fixed protocol, intended difference, success/stop criteria, authorized
stage, total budget, per-run limit, and retry bound in the existing experiment
note. Use the short format in [development](../../../docs/DEVELOPMENT.md) if needed.
Do not create a second project plan when these facts already exist.

Preserve the relevant data population, normalization, physical batch, optimizer
steps, seed, initialization and checkpoint selection. Continuation experiments
must use the required shared foundation: T1/B3 or T5/B5 where the recipe specifies
them. Do not substitute a continuation wrapper for a foundation checkpoint.
Disclose any unmatched factor before attributing a gain to the new method.

## Prepare and run within scope

Use [reproduce.md](../../../docs/research/reproduce.md) for setup and submission.
Use the [official tutorial](../../../docs/tutorials/res18-baseline-slurm.md) if
the requested task is the independent official fold rather than project B0.
Read environment, data, or teacher instructions only when required by this run.

Check code/configuration and required inputs; reuse valid checks already done.
For a changed execution path, choose a short allocated check that addresses its
risk. A smoke run is not an effect estimate and need not be repeated gratuitously.
If compute authorization or budget is missing, finish preparation and obtain
that decision before submitting; do not infer a budget from a historical run.

The research submitter archives committed HEAD. Include the intended changes
in a task-owned commit before submission and verify the archived commit ID.
Use a new output directory and record job IDs. Follow current site configuration;
do not copy a personal path/account or silently change batch to fit a GPU.
All numerical/model execution, installation, downloads, and data processing
belong in Slurm allocations. Scheduler queries and small log reads are control work.

Continue monitoring and diagnosing within the authorized stage and retry bound.
Do not duplicate pending jobs. If the task is interrupted, record job IDs,
run paths, current state, remaining budget, and the exact next action in its note.

## Interpret and deliver

Reconcile scheduler state, exit code, expected completion markers/checkpoints,
and the actual evaluation artifacts. Submission, execution success, and scientific
support are different facts. Never fill absent measurements with expected numbers.

Report the paired comparison and its population, the narrowest supported claim,
and whether the next decision is to continue, revise, stop this hypothesis, or
remain inconclusive. Separate implementation faults and invalid proxies from
scientific negatives. Single-seed improvements remain signals; historical test
years cannot become untouched confirmation data by renaming them.

Link the command/config, commit, inputs, job/accounting record, logs and outputs.
Update the relevant evidence entry when the task includes recording new results;
preserve earlier evidence and distinguish measured results from planned follow-up.
End at the agreed completion condition or resource boundary.
