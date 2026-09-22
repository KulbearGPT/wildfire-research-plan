# Developing with AI assistance

Use [AGENTS.md](../AGENTS.md) for the project-wide agreement. This guide contains
operational details to read when relevant. The [student homepage](START_HERE.md)
remains the starting point for learning the scientific workflow.

## Start from the appropriate source

The active research scope is [X22+X17 and necessary controls](CODE_LIFECYCLE.md).
The inclusive method inventory also contains inactive historical explorations;
check its `lifecycle` field before treating an entry as a development target.

| Task | Source or entry point |
|---|---|
| Reproduce or extend a current method | [Research handoff](research/reproduce.md), [method recipes](research/method-recipes.md), [inventory](research/method-inventory.json) |
| Understand a promising or unsuccessful direction | [Positive signals](research/positive-signals.md), [negative results](research/negative-results.md) and their linked evidence |
| Change data preparation or evaluation populations | [Data preparation](research/data-preparation.md), [population contract](research/evaluation-population.md), relevant `src/` code and tests |
| Change a model or training method | The inventory's implementation path under `reproductions/`, its config and closest control |
| Set up an environment or diagnose installation | [Reproduction setup](research/reproduce.md), [setup recovery](research/setup-recovery.md) |
| Edit the course | `site-content/en/docs/` for generated guides; `index.html`, `related-work/index.html`, or `baseline-reproduction/index.html` for standalone pages |

Do not read every linked document for a small change. Historical plans under
`docs/superpowers/` preserve past decisions, not a required development sequence.
Existing artifacts should resolve factual questions before a new run is proposed.

## Small changes and research changes

For a routine patch, inspect the relevant implementation, make the smallest
coherent change, and run the checks that could detect its failure. No mandatory
design document, test-first ceremony, branch isolation, or multi-agent review.
Retain focused regression tests for substantive correctness bugs.

For a new experiment, reuse an existing experiment note or write a short note
under `docs/experiments/`. Record only what affects the decision:

```text
Question and falsifiable prediction:
Method and closest matched control:
Fixed protocol; intentional differences:
Evidence that would support, weaken, or leave the hypothesis unresolved:
Authorized stage, total budget, per-run limit, and retry bound:
Completion condition and intended command/config:

After execution: commit, config, data/checkpoint identity, job IDs, status,
artifact/log paths, measured comparison, limitations, next decision.
```

Link configuration and machine-generated records rather than transcribing them.
An unknown value is not authorization or a measured result. Record changes to
selection rules as exploratory decisions rather than rewriting the original plan.
Use the two [project skills](../.agents/skills/) only for their stated tasks.

## Environments and checks

The root `pyproject.toml` belongs to the Python 3.13 data-audit environment.
Research model execution uses the separate Python 3.10 training environment.
Do not install the root package into the training environment.
Dependencies are in [research-training.txt](../environments/research-training.txt)
and [research-audit.txt](../environments/research-audit.txt).

From the repository root, with an already prepared and trusted site file:

```bash
export WILDFIRE_SITE_ENV="$HOME/wildfire-config/site.env"
git status --short
git diff --check
```

For an authorized numerical code check, choose the affected test file or node:

```bash
# Replace the test path below with a real affected test; this submits CPU compute.
bash scripts/research/submit.sh cpu python -m pytest tests/REPLACE_WITH_AFFECTED_TEST.py -q
# For tests requiring the audit dependencies, select the audit environment:
bash scripts/research/submit.sh cpu audit python -m pytest tests/REPLACE_WITH_AFFECTED_TEST.py -q
```

These are command patterns, not evidence that a test ran. Inspect the test's
dependencies before choosing its environment or CPU/GPU resource class.
The intended changes must already be committed: the submitter snapshots HEAD.
Commit those changes, run the necessary allocated checks, and commit any repairs;
do not describe the first commit as tested before the job finishes.

`job.sh` rejects execution without a Slurm allocation. Do not bypass it by
running model imports, numerical tests, or dataset processing on the login node.
Full migration qualification is documented in
[qualification.md](research/qualification.md); it is not the default patch check.

## English course pages

The maintained website Markdown is under `site-content/en/docs/`. The parallel
`docs/` pages retain the Chinese teaching material. Keep scientific commands and
claims consistent when either version changes. Generated HTML under `guides/`
must be rebuilt from its English source rather than edited by hand.

The renderer resolves Markdown links relative to the canonical `docs/` path,
even though it reads English source. Use that convention when adding links.
With Pandoc available, a lightweight documentation build is:

```bash
python3 scripts/build-course-guides.py
git diff --check
git diff --stat
```

The renderer checks English source availability and rejects untranslated CJK
text. Inspect the affected HTML and local links as well. No model tests or Slurm
training jobs are needed for prose-only changes. Commit both source and generated
pages. If Pandoc is absent, prepare it through the site's allowed compute setup.

Adding a document under `docs/research/` also adds it to the renderer's discovery
set and requires an English counterpart. Developer-only notes belong elsewhere.
Git publication and website deployment require checking the actual remote and
hosting configuration; a local build or a push alone does not prove deployment.

## Codex setup and task prompts

Open Codex with this repository as the working directory, not its parent, so the
repository instructions and `.agents/skills/` can be discovered. The project
configuration and its limits are explained in the
[workflow migration record](ai-workflow-migration.md).

Describe the outcome and constraints rather than prescribing every tool call:

```text
Implement [change] against [matched control]. Preserve [protocol].
This round permits [stage and resource budget]. Finish when [observable outcome].
Complete implementation, relevant checks, and necessary repairs within that scope.
Report evidence and limitations; do not start the next experimental phase.
```

For a preparation-only request, say so; commands and configs can be reviewed
without submitting a job. For diagnosis, provide the job ID and run directory.
Existing permissions carry forward; do not restate an entire project protocol
in every prompt. Humans retain responsibility for research questions and claims.
