# Wildfire course: curriculum and ERM pilot design

Status: proposed curriculum and pilot specification for review. No website
implementation or deployment is included in this document.

## Purpose and audience

Help a student with basic Python and CNN knowledge move from understanding a
forecast sample to completing and interpreting a matched experiment. The site
must always show where the student is, what to do now, what completion looks
like, and what comes next. Use English for the public course.

Follow the teaching pattern of https://kulbeargpt.github.io/MedCourse/#lesson2:
short explanations, purposeful interactions, worked examples, expandable
details, practical deliverables, and explicit lesson navigation.

## Central directory

The primary navigation contains Course, Practical handbook, and Reference
library. All lessons appear in one persistent course directory. Desktop uses a
sidebar; mobile uses an accessible collapsible directory. Each lesson has a
short local step list and previous/next navigation. Students may revisit any
lesson without passing an artificial lock.

The course home shows prerequisites, the ordered route, deliverables, and a
start/continue action. Classroom time and cluster execution time are separate;
do not promise a fixed training turnaround.

| Stage | Lesson | Ordered sections | Deliverable |
|---|---|---|---|
| Understand the task | 1. What are we predicting? | One fire event; input dates and target day; active fire versus burned area; legal forecast inputs | Annotated input/target timeline |
| Understand the task | 2. From satellite channels to a fire map | Dynamic/static channels; tensor axes; spatial prediction; logits, probabilities and masks | Annotated sample and small prediction exercise |
| Understand the task | 3. How do we judge a prediction? | Sparse positives; false alarms and misses; thresholds; AP; training/validation/test roles | Short metric interpretation worksheet |
| Complete a reproduction | 4. Your first official Fold 2 | Personal environment; official data and split; smoke; training; evaluation; completion evidence | Fold 2 experiment record |
| Enter the project | 5. What changes when observations are missing? | Missing versus zero; controlled masks; M00/M01/M06/M07; official-to-project protocol bridge; corrected data and B3 | Scenario sheet and project readiness checklist |
| Design a comparison | 6. Why do we need an ERM control? | Predict; trace the branches; understand ERM; repair a comparison; plan A/B; interpret results; submit | Matched A/B plan and result report |
| Attribute an improvement | 7. Does the schedule help? | Independent X22 branch; matched ERM; B-to-C gain; variability and limits | A/B/C comparison and supported conclusion |
| Advanced research | 8. When does specialization help? | X14 mixed severity; X17 fixed severity; nearest controls; condition-specific composition; cost and claim limits | Proposed controlled follow-up experiment |

## Practical handbook and reference library

The handbook groups setup, data preparation, official Fold 2 commands, project
foundations, continuation recipes, job diagnosis, and report templates. Each
lesson links to the exact relevant entry with an explanation of when to use it
and what result to bring back. Core explanations and task instructions remain
inside the lesson, rather than sending students through a chain of documents.

The reference library holds related literature, full twelve-fold reproduction
records, qualification evidence, positive signals, negative/invalid/unrun
records, and archived methods. These are searchable references, not parallel
mandatory learning routes. Preserve old URLs and original evidence links.

## ERM pilot: lesson 6

Title: **The score improved. Did the method help?**

Prerequisites: distinguish a checkpoint from a run; know AP and the four
project scenarios. The conceptual part can be completed without a cluster.
The experiment requires corrected project data, training-only statistics, an
identified B3 checkpoint and the existing Slurm setup.

Suggested classroom session: 45–60 minutes, excluding setup and computation.

### Step 1 — Make a prediction

Show an explicitly synthetic teaching example: frozen B3 has AP 0.40 and a
candidate has AP 0.44 after 3,000 additional updates. Ask whether the schedule
has helped. Reveal the missing matched ERM score, 0.46, after the student answers.
Explain that the candidate improved over B3 but lost to ERM. These values are
not project results or browser-generated model predictions.

### Step 2 — Follow the shared starting point

Use a native SVG or HTML branch diagram:

- A: evaluate frozen B3; no additional updates.
- B: start independently from B3; fresh optimizer; constant schedule; 3,000 updates.
- C: start independently from the same B3; fresh optimizer; cosine schedule;
  3,000 updates. C belongs to the next practical lesson.

Let the student switch clearly labelled synthetic score presets, including a
candidate worse than ERM and a continuation that does not help. Show B−A,
C−B, and C−A with short interpretations and text alternatives to the diagram.
Describe B−A as the effect of the specified continuation recipe, not a pure
causal estimate of update count alone.

### Step 3 — Understand what ERM means here

Explain supervised fitting in plain language, with an optional empirical-loss
formula. Follow one batch through input, prediction, loss and optimizer update.
Explain that fresh continuation loads model weights and creates a new optimizer;
resuming also restores optimizer/scheduler state.

The project control already includes FireDrop and BlockDrop, each with
probability 0.3; they can overlap, with block severity sampled at 25% or 50%.
ERM does not mean clean-only training. Keep these details near the comparison
exercise rather than front-loading the lesson with all configuration fields.

### Step 4 — Repair an unfair comparison

Use select controls and immediate explanatory feedback for three cases:

1. ERM and X22 have different update budgets.
2. X22 starts from ERM's final checkpoint instead of the shared B3.
3. X22 uses its best validation checkpoint while ERM uses its final checkpoint.

Students identify the extra changed factor and choose a repair. Avoid
drag-only interactions. Answers must remain available with JavaScript disabled.

### Step 5 — Prepare and execute A/B

Present one protocol card: T1, seed 0, train 2016–2020, evaluate 2021 validation,
same B3 bytes and training statistics, 3,000 continuation updates, physical and
effective batch 64, final continuation checkpoint, and matching evaluation
samples/masks. Official Fold 2 scores cannot populate this table.

Present ordered action cards with action, expected artifact, and completion:

1. Identify data/statistics and B3 provenance/checksum.
2. Evaluate frozen B3 under the four scenarios, without state updates.
3. Smoke the continuation path if needed; start formal ERM afresh from B3.
4. Run the existing control recipe through Slurm and retain the final checkpoint.
5. Evaluate ERM under the same contract, or reuse an already matching evaluation.
6. Record original metric paths, code revision, job IDs, exit status and updates.

Use existing English setup and recipe pages as execution references. Review
their exact anchors during implementation. Do not invent CLI flags, run
experiments, prescribe another user's account, or duplicate long mutable runners.
Job submission and successful training are distinct states.

### Step 6 — Read the complete result

Provide a blank downloadable worksheet with A, B and B−A rows and M00, M01,
M06, M07, Primary and Block columns. Primary is the arithmetic mean of the
three missingness AP scores; Block averages M06 and M07. Neither is AP from
pooled scenario predictions. M00 is reported separately.

Use a separate synthetic worked table to teach both gains and regressions.
Never prefill the student's real report with demonstration values. Include
the case where Primary improves but clean AP falls. A valid negative result
still completes the exercise; one seed does not establish a confirmed gain.

### Step 7 — Submit and continue

The report includes a pre-run question and falsifying outcome, matched settings,
artifact identities and completion evidence, the full A/B table, one supported
conclusion and one limitation. Exit question: why can X22 beat frozen B3 yet
fail to improve on its matched control?

Offer a download of the report template and an explicit next-lesson entry.
Do not label unimplemented future lessons as completed interactive lessons.

## Presentation and implementation boundaries

Use dark blue headings, a light neutral background, white reading cards and
teal interactive panels. Keep body width readable. Each step begins with one
action and ends with a completion criterion. Full formulas, detailed settings,
answers and reference records use labelled disclosures.

The first implementation slice is the central directory and ERM pilot, not a
claim that all eight lessons have been rewritten. Existing material remains
accessible and is explicitly labelled as current reading where applicable.
Use lightweight static HTML/CSS/JavaScript compatible with the existing site.
Maintain English guide sources under site-content/en/docs; never hand-edit
generated guides. No new backend, authentication or training service is needed.

Any progress state is local, optional and resettable; unavailable storage must
not break navigation. Semantic headings, keyboard controls, visible focus,
textual feedback, mobile layout, no-JavaScript reading and print output are
required. Do not treat a clicked completion checkbox as evidence of mastery.

## Acceptance and validation

- Every lesson is findable from a single directory and has an honest readiness label.
- The ERM lesson makes its next action and completion evidence explicit.
- Students can explain the three branches and calculate all three differences.
- The pilot distinguishes teaching values from real evidence and official Fold 2
  from the project protocol; it introduces no new research claim.
- Verify links/anchors, keyboard navigation, mobile overflow, worked-example
  arithmetic, feedback, template downloads, no-JavaScript fallback and print.
- Check source diffs and preserve old entry URLs. No model tests or cluster
  computation are needed to validate this teaching-only slice.

Source of the scientific contract:
site-content/en/docs/tutorials/matched-erm-experiment-design.md.
