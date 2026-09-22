---
name: wildfire-run-diagnosis
description: Diagnose an existing wildfire Slurm job from its ID, run directory, logs, and outputs. Use for pending, failed, incomplete, or inconsistent runs; this does not authorize a new experiment.
---

# Diagnose a wildfire run

Identify the current execution state and the first supported cause of trouble.
Deliver evidence and a bounded recovery action. Reuse supplied job IDs and paths;
ask only for missing identifiers that cannot be found in the task's records.

Paths in links are relative to this skill directory. Run scheduler commands on
the correct cluster; a missing scheduler on another host does not mean job failure.
Do not import a model or scan a dataset on a login node to reproduce an error.

## Read the smallest useful evidence

Start with scheduler state and the specific run's records. For the research
submitter these include `job-id.txt`, `source-commit.txt`,
`submission-status.txt`, `command.txt`, `site-at-submission.env`,
`slurm-JOB_ID.out`, `slurm-JOB_ID.err`, and `allocation-JOB_ID/`.
Other runners may use different paths; locate them from the submission command.
Avoid dumping an entire environment or unrelated logs.

Use current observations where available, for example:

```bash
# Replace 123456 with the supplied job ID; these commands are read-only.
squeue -j 123456 -o '%.18i %.12T %.30R'
scontrol show job 123456
sacct -j 123456 --format=JobID,State,ExitCode,Elapsed,ReqTRES,AllocTRES,MaxRSS
```

Completed jobs may be absent from `squeue`/`scontrol`; accounting can lag or expire.
Inspect job steps as well as the parent row and corroborate with saved logs.
Read bounded log excerpts around the first error, plus the end of the log.
Do not infer failure from one empty or stale scheduler response.

## Distinguish causes before changing anything

- **Pending:** interpret Resources, Priority, Dependency, account/QOS, or node
  constraints. Waiting alone is not an implementation failure or a reason to
  cancel/resubmit. Check dependencies before suggesting another job.
- **Environment/input:** check the archived source, actual environment, selected
  site file, data and checkpoint identities. Use
  [setup recovery](../../../docs/research/setup-recovery.md) for incomplete setup;
  do not delete an environment and reinstall blindly.
- **Resource failure:** distinguish host RAM from GPU memory. `MaxRSS` is host
  memory, not VRAM. Do not change scientific batch size to conceal an OOM.
- **Execution/artifact mismatch:** inspect step exit codes and expected outputs.
  `COMPLETED` with missing required results is not a successful experiment.
  A checkpoint alone does not prove the requested training budget completed.
- **Unexpected metric:** check population, split, initialization, budget,
  checkpoint selection and metric definition before judging the method.
  Use the [population contract](../../../docs/research/evaluation-population.md)
  when denominators disagree. A smoke score is not a reproduction result.

## Recover only as authorized

For a diagnosis-only request, provide the supported cause and proposed repair
without cancelling, submitting, or editing the experiment. If repair and retries
are already authorized, complete them within the existing bounds without asking
again. A materially different comparison or larger budget needs a new decision.

Before replacing a run, recheck its live state, preserve its job ID, reason and
logs, and ensure no duplicate remains. Commit necessary code repairs before the
research submitter snapshots HEAD. Use a new run directory and record how it
relates to the old run. Preserve data, checkpoints, and failed-run evidence.

Report observed state, first supported cause, evidence paths, actions actually
taken, and what remains unknown. Keep scheduler/implementation failures separate
from scientific negative results; do not start a fresh research phase as recovery.
