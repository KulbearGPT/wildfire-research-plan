# Course directory and ERM pilot implementation plan

Spec: [approved design](course-redesign.md). Execution authorized by the user on
2026-10-13; implement inline in the existing workspace.

Goal: a single course directory and a complete, interactive ERM pilot.
Architecture: standalone static HTML with shared, scoped CSS and progressive
JavaScript enhancement. Preserve the existing guides and their build pipeline.

## Constraints and decisions

English public content; no fabricated research results; no model execution or
new dependencies. Existing guides remain reachable. Other lessons must identify
existing reading versus material still awaiting an interactive rewrite.
Use the existing workspace for this documentation/UI change; no automatic
commit, push, deployment or experiment submission.

## Tasks

- [x] Replace index.html with the eight-lesson course directory, practical
  handbook and reference library; add assets/learning.css.
- [x] Add lessons/erm/index.html with seven short stages, accessible worked
  examples, comparison repair exercises, protocol/actions and exit task.
- [x] Add assets/learning.js for staged navigation, optional local progress,
  worked examples and feedback; add static CSV and Markdown report downloads.
- [x] Connect the pilot from the maintained ERM guide and legacy course page;
  rebuild generated guides and document the new file ownership.
- [x] Verify internal links/anchors, HTML structure, JavaScript where a runtime
  is available, worked arithmetic, downloads, no-JS and print behavior. Inspect
  responsive CSS; report browser limitations rather than claiming screenshots.

## Review focus

1. Deep links, reload and browser back must select the correct ERM stage.
2. Blocked or malformed storage must not stop navigation; reset remains scoped.
3. Teaching scores must never become prefilled student results.
4. No-JS and print must expose every stage and answers.
5. Reference links must specify the relevant role, not encourage an unintended
   five-model sweep or historical-test evaluation during the first exercise.

## Execution record

Implemented on 2026-10-13. Existing guides and standalone lecture URLs are
preserved. The first slice consists of the directory and lesson 6; other lessons
retain explicit reading/practical readiness labels.

Verification:
- Rebuilt all 19 generated guides; only the two intentionally updated guide
  outputs changed (START_HERE and matched ERM).
- Checked 165 local links/assets, including destination anchors: no errors.
- Checked new-page HTML nesting, unique IDs, labels, 15 select options, seven
  no-JavaScript-readable stages and the blank CSV layout/content.
- Ran the actual learning.js in V8 with a minimal DOM/storage harness: 27
  assertions passed for deep links, hash navigation/focus, all three score
  presets, feedback, completion/reset, blocked/malformed storage, home resume
  and print expansion/restoration. This is not a rendered-browser test.
- Checked synthetic worked-table arithmetic and git diff whitespace.
- Structural verification caught a malformed option tag in the first repair
  exercise; corrected the tag and reran the complete structural/link check.

Limits: this host has no browser or browser automation runtime. Responsive and
print styles have been inspected, but rendered desktop/mobile screenshots,
actual keyboard traversal and print/PDF appearance remain to be checked in a
browser. No new dependencies, model jobs, commits, pushes or deployment were
performed. The implementation uses no new training commands beyond a verified
existing text-only control preview; actual execution stays in the linked guides.
