# My matched ERM comparison

Use this template for your own experiment. Do not copy synthetic teaching values.
Keep AP on a 0–1 scale. Missing results stay blank; a queued job is not completed.

## Before running

- Question:
- Predicted outcome:
- An outcome that would contradict my prediction:
- Agreed resource budget:

## Fixed comparison contract

- Setting: T1, seed 0; train 2016–2020, evaluate 2021 validation.
- A: frozen B3, no further training or state updates.
- B: independent B3 initialization, fresh optimizer, constant schedule,
  3,000 updates, physical/effective batch 64, final checkpoint.
- B3 path, checksum and provenance:
- Corrected data identity and training-only normalization statistics:
- Code commit and environment:
- Loss, optimizer and starting learning rate:
- FireDrop/BlockDrop settings and sampling/worker configuration:
- Evaluation samples, masks and evaluator identity shared by A and B:
- Intentional deviations (if any), and the claim they permit:

## Completion and provenance

| Record | A: frozen B3 | B: ERM |
|---|---|---|
| Configuration / actual command | | |
| Slurm job ID | | |
| Terminal state / exit code | | |
| Completed optimizer updates | 0 additional | |
| Checkpoint path | | |
| Original evaluation path | | |
| Logs / completion evidence | | |

## Results

| Run | M00 AP | M01 AP | M06 AP | M07 AP | Primary | Block |
|---|---|---|---|---|---|---|
| A: frozen B3 | | | | | | |
| B: matched ERM | | | | | | |
| B minus A | | | | | | |

Primary = (M01 + M06 + M07) / 3. Block = (M06 + M07) / 2.
Compute AP separately in each scenario, then average the AP values.
Report clean M00 separately. Equal sample counts alone do not prove matching samples.

## Interpretation

- Which conditions improved, and which regressed?
- One supported conclusion:
- One claim this comparison cannot support:
- Relevant training curves (training loss alone does not establish validation gains):
- My proposed next matched comparison:

A single-seed result is a pilot. Controlled masks do not establish natural
operational robustness. Historical 2022/2023 years are not untouched confirmation
sets. For X22, start independently from the same B3, not B's final checkpoint.
