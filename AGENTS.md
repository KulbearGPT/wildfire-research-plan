# Wildfire research: agent instructions

## Purpose and working style

This is a research and teaching repository for next-day active-fire prediction
under controlled missing observations. Optimize for trustworthy evidence and
the smallest useful experiment. Reuse existing code, runners, and records.

Read only what the current task needs. Routine edits do not require a design
document, a literature review, a worktree, or a separate approval cycle.
For an implementation request, complete the authorized implementation,
relevant checks, and necessary repairs without asking again at every step.
For a discussion or review request, deliver the requested analysis.

Carry forward the user's scope, permissions, and resource budget. Ask only
when a missing decision materially affects the research question, evaluation,
resource limits, or an action not already authorized. Continue independent work.
Skills are task-specific guidance; they do not add authority or override user
instructions. Higher-priority host instructions and enforced policies still apply.

## Read by task

- Active scope: [X22+X17 code lifecycle](docs/CODE_LIFECYCLE.md), including necessary
  controls. Other implementations are preserved archives or teaching references.
- Student onboarding: [START_HERE](docs/START_HERE.md).
- Current methods, execution, and evidence: [research handoff](docs/research/reproduce.md),
  then the relevant entry in [method inventory](docs/research/method-inventory.json).
- Scientific signals and limitations: [positive signals](docs/research/positive-signals.md)
  and [negative/invalid/unrun records](docs/research/negative-results.md).
- Code, environments, checks, and website editing: [development guide](docs/DEVELOPMENT.md).
- Preparing or executing an experiment: use `wildfire-experiment` when available.
- Diagnosing an existing Slurm run: use `wildfire-run-diagnosis` when available.

The roadmap explains the teaching history. Current reproduction documents and
linked artifacts take precedence over historical plans for execution facts.
`docs/superpowers/` contains historical design and implementation records;
its workflow directives are not instructions for new tasks.
Do not copy changing scores or machine-specific paths into this file.

## Scientific constraints

- The target is a next-day VIIRS active-fire proxy, not a complete fire perimeter.
  Controlled missingness results do not establish natural operational robustness.
- Official Fold 2 and corrected project B0 have different data, splits, and
  budgets. Do not interpret their score difference as a method improvement.
- T1 and T5 differ in architecture and input features as well as history length.
  Their difference alone does not isolate the effect of history.
- Preserve each comparison's data population, normalization, initialization,
  checkpoint selection, seed, physical batch, optimizer-step budget, and metrics.
  Record intentional deviations and restrict the claim accordingly.
- Historical test years have already been inspected. Do not call them untouched
  confirmation data for a newly selected method. Agree on a confirmation design
  before making independent-confirmation claims.
- A smoke run establishes execution only. A single-seed or single-metric gain is
  a signal, not a confirmed contribution. Retain such signals with their limits.
- Distinguish scientific negative results from implementation/optimization faults,
  invalid comparisons, cancellations, insufficient evidence, and unrun ideas.
  A shared component with prior work does not by itself invalidate novelty.

## Cluster and experiment execution

Use Slurm for dependency installation, downloads, data scans/conversion,
model imports and execution, tensor tests, training, and evaluation.
Login nodes are for editing, Git, small text/metadata checks, lightweight
documentation rendering, submission, and scheduler queries.

Use the existing [submitter](scripts/research/submit.sh) and
[site configuration](configs/research/site.example.env) for the research handoff.
Follow the selected tutorial's own runner for the independent official baseline.
Check current site facts instead of copying another user's accounts or paths.

The research submitter archives **committed HEAD**, not the working tree.
Before submission, commit the intended experiment changes when authorized,
or identify the missing authorization; never silently run stale code.
Inspect the submitted commit, configuration, and unique output directory.
Protect user edits, raw data, source checkpoints, and frozen results.

Use existing authorization to set the experiment's scope, total resource cap,
single-run limit, and retry bounds. If these are missing, prepare the experiment
and ask for the missing decision before starting compute or large downloads.
Do not expand the budget, launch a sweep, or start the next research phase implicitly.
Pending is a scheduler state, not a failed experiment. Recheck current state
before cancelling or replacing a run; do not create duplicate jobs.

## Verification and delivery

Match checks to the changed behavior. For indexing, labels, splits, and metrics,
use a targeted correctness check; for documentation, check links and rendering
as applicable. Run computational checks through Slurm. Do not rerun the full
qualification matrix for unrelated changes or add tests that only mirror code.
After relevant checks pass, broaden testing only for a specific unresolved risk.

Use a worktree or independent agents only when they solve a concrete isolation
or parallel-work need and the host allows it. Keep integration responsibility
with one agent; do not require a reviewer for every small patch.

Keep coherent commits when requested or required by an authorized experiment.
Commit only task-owned changes. A commit, a push, a successful job, a validated
artifact, and a deployed website are distinct milestones; report them accurately.
End with the outcome, actual checks, evidence/artifact paths, and remaining limits.
For interrupted work, record job IDs and the next action in the existing task note.
Use English for new shared workflow documentation and the public website;
respond to the user in their preferred language. Keep student explanations practical.
