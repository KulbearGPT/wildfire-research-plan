# WSTS+ Post-Control Replication Implementation Plan

**Goal:** Extend the immutable manifest gate from seed-0 promotion to the four
declared seed 1/2 10K replications while preserving the test lock and prepare a
reviewed boundary for later M00--M07 implementation.

**Design:** `docs/superpowers/specs/2026-08-23-wsts-post-control-design.md`

## Task 1: Generalize the prerequisite manifest gate

**Files:**

- Modify: `reproductions/wsts_fast_track/promotion.py`
- Modify: `tests/test_wsts_fast_track_promotion.py`

1. Add failing tests proving each seed 1/2 target requires passing C00/C02
   seed-0 10K records and renders its declared seed.
2. Prove 3K records, nonzero prerequisite seeds, incomplete pairs, and active
   screening targets are rejected.
3. Allow promotion and replication stages through the same strict prerequisite
   validator; keep screening targets non-renderable.
4. Preserve manifest schema, deterministic prerequisite ordering,
   non-overwriting output, and the absence of Slurm submission.
5. Run the focused promotion tests.

## Task 2: Publish actual screening evidence and next gates

**Files:**

- Modify: `reproductions/wsts_fast_track/README.md`
- Modify: `README.md`
- Modify: `docs/research-roadmap.md`

1. Record C00/C02 3K job IDs, wall time, checkpoint step, AP, F1, loss, and CUDA
   peak from the sealed completion records.
2. Mark both seed-0 10K jobs as submitted fresh runs and keep 2022--2023
   withheld.
3. State that seed 1/2 manifests are code-ready but evidence-gated.
4. Link the post-control design and retain the screening/non-claim wording.

## Task 3: Verify and commit

1. Run `tests/test_wsts_fast_track.py`,
   `tests/test_wsts_fast_track_promotion.py`, and
   `tests/test_project_research_organization.py`.
2. Run `git diff --check` and inspect the branch status.
3. Commit implementation and documentation separately.
4. Do not submit seed 1/2 jobs until both real seed-0 10K completion records
   pass the generalized gate.
