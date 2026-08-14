# Baseline Reproduction Lecture Note Design

## Purpose

Create an English-only courseware page that explains the complete Res18-U-Net
`T=1` baseline reproduction to a beginner. The page must match the visual and
interaction language of `02 · Research foundations and related work`, while
turning the existing scientific evidence into a teachable sequence of
questions, decisions, checks, results, and limits.

The page is educational documentation, not a new experiment. Every numerical
claim must come from the committed twelve-fold publication, the Fold-2
calibration/full-run evidence, or the independently verified reproduction
documentation already in this repository.

## Audience and learning outcome

The intended reader understands basic machine learning but may not yet know
what a fold, checkpoint, released weight, AP, provenance seal, or independent
verifier is. After the lecture, the reader should be able to:

1. explain why a paper baseline is reproduced before a new method is proposed;
2. distinguish training reproduction from released-weight test reproduction;
3. describe the frozen Res18-U-Net `T=1` scientific contract;
4. follow the data, environment, calibration, one-fold, and twelve-fold path;
5. explain why engineering compatibility fixes are not scientific model edits;
6. read the twelve-fold AP distribution and its uncertainty;
7. state what the result proves and what it does not prove; and
8. reuse a practical checklist for the next baseline or ablation.

## Chosen narrative

Use a hybrid concept-first and chronological narrative. Each major stage first
answers “why is this necessary?” and then shows “what did we do?”. This avoids
the two common failures of a pure lab diary (commands without understanding)
and a pure conceptual lecture (principles without a reproducible path).

The page title is:

- Eyebrow: `03 · Baseline reproduction`
- Heading: `From a Paper Claim to a Verified Baseline`
- Subtitle: `A beginner-friendly walkthrough of reproducing Res18-U-Net with T=1`

## Page structure

The page will live at `baseline-reproduction/index.html` and contain sixteen
sidebar-addressable sections:

1. Why baselines matter
2. Reproduction vocabulary
3. The exact target
4. Freeze the contract
5. Reuse official code
6. Audit data and folds
7. Build the environment
8. Calibrate before committing
9. Complete one fold
10. Evaluate released weights
11. Verify independently
12. Read the results
13. Learn from failures
14. Evidence boundaries
15. Reusable checklist
16. Summary and next step

The first half teaches the workflow. The second half teaches how to interpret
and audit it. The twelve-fold table will show AP for every fold, with a compact
aggregate table for AP, F1, IoU, Precision, Recall, and Loss. Detailed raw
artifact hashes remain linked through the experiment evidence page instead of
dominating the lecture.

## Visual system

Reuse the existing courseware shell and behavior from
`related-work/index.html`:

- sticky top toolbar, progress line, theme toggle, font controls, Print/PDF;
- dark green grid hero, four summary pills, and a truth banner;
- sticky searchable table of contents and responsive mobile drawer;
- white section panels, numbered cards, callouts, timelines, tables, badges,
  and a dark closing section;
- identical responsive breakpoints and print treatment.

New content components may be composed from existing primitives. No framework,
external font, chart library, or runtime dependency is introduced. A small
inline SVG/process diagram is allowed because it remains editable, accessible,
and dependency-free.

## Evidence and claim boundaries

The lecture must preserve these distinctions:

- The twelve-fold result is a **test-only evaluation of official released
  weights**, not twelve new training runs.
- The measured AP is `0.45276400446891785 ± 0.08821731990844857`
  (population standard deviation), compared with the paper reference
  `0.460 ± 0.084`.
- Close agreement supports **released-weight executable reproducibility**. It
  does not prove that the released checkpoints are identical to the original
  training runs behind the paper table.
- Fold 0 recovery and Fold 2 parser qualification were offline observer and
  provenance repairs. They did not relaunch scientific evaluation children.
- The seven-line runtime import patch removed unused broken eager imports. It
  did not modify the Res18-U-Net model, loss, optimizer, data, or metrics.
- GPU values are sampled WDDM observations and may miss transient peaks.
- The focal-alpha/config-code discrepancy is a future training ablation, not a
  correction silently applied to this baseline.

## Navigation

Add a visible `03 · Baseline reproduction` lecture card/link to the main
research-plan page and a `Next lecture` link from the related-work courseware.
The new page links back to the research plan and to lecture 02. Existing public
URLs and section anchors remain unchanged.

## Verification

Add a standard-library publication test that parses the final HTML and proves:

- the complete course shell, sixteen unique section IDs, and matching TOC links;
- the exact twelve AP values and aggregate/reference numbers;
- beginner definitions and all evidence-boundary statements;
- no accidental claim of twelve training runs;
- navigation from the main page and lecture 02;
- scripts parse as JavaScript and all local links/assets resolve;
- interactive controls and search/progress hooks remain present.

Render the page locally at desktop and mobile widths, in light and dark themes,
and inspect screenshots before delivery. The final page must remain readable
without JavaScript; JavaScript only enhances navigation, theme, search, and
progress.

