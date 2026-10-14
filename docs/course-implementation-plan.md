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

## Phase 2 — Full course and descriptive experiment names

Authorized after review of the ERM pilot. Keep the accepted visual language.

- [x] Add complete lessons 1–5, 7–8 with meaningful interactions, short stages,
  exit questions and downloadable worksheets; retain lesson 6.
- [x] Share navigation/progress/printing across all lessons; preserve old ERM
  progress and exact deep links, including nested reference headings.
- [x] Use one static shell for the two historical lecture pages and all 19
  generated guides. Preserve original anchors, commands, images and evidence;
  make each reference chapter independently readable.
- [x] Add descriptive experiment names and a searchable name/ID catalog covering
  the inventory and foundations; keep machine identifiers unchanged.
- [x] Link the home directory directly to all eight completed lessons and make
  handbook/library resources navigable in the same shell.
- [x] Rebuild, verify all local links, original command preservation, interaction
  logic and markup. Record browser limitations explicitly if still unavailable.

Implementation units: scripts/course_shell.py (shared static layout),
scripts/build-course-lessons.py (lesson generation),
site-content/en/lessons/*.html and course-lessons.json (lesson sources),
assets/learning.js (navigation), assets/lesson-demos.js (teaching interactions),
assets/learning.css (presentation), and the existing guide renderer.
Experiment naming is an independent content task delegated with exclusive
ownership of site-content/en/experiment-names.json and docs/experiment-names.md.


### Phase 2 completion and retirement

All eight lessons, 19 handbook pages, two reference readings, and 107 named
experiment/scenario walkthroughs now use the shared shell. Every walkthrough
includes high-level steps and a blank worksheet. The catalog prioritizes the
mainline and expands historical entries on search. Machine identifiers remain
unchanged. The user explicitly superseded the phase-1 URL-preservation rule:
all 21 old HTML pages are removed, with no redirects or retirement notices.

Validation: 138 pages and 7,011 local links/assets pass structural and destination
checks. All 96 original reference/handbook code blocks and content anchors were
preserved. V8 DOM-harness checks cover staged navigation, interactions, storage,
print state, and resume behavior; these are not browser rendering tests.
No browser runtime is installed, so desktop/mobile rendering, keyboard traversal,
and print appearance remain unverified. No model jobs or dependencies were run.
This phase has not been committed, pushed, or deployed.

## Student-path review follow-up

The user approved closing the gaps identified by walking the course as a novice.
Added five staged support pages: practical readiness, a synthetic running sample,
official-to-project handover, output-to-report example and troubleshooting.
Connected these at the relevant lesson steps and in global navigation. Added a
continuous blank course record and explicitly labelled synthetic downloads.
The report guide traces actual, different AP field paths in foundation and
continuation summaries. The B3-only classroom recipe precedes optional paired
B3/B5 work. Mainline experiment guides specify required branches, artifacts and
acceptance conditions. Historical numerical shorthand links to original evidence
instead of appearing as unexplained classroom targets.

Validation: all 143 generated pages and 8,323 local links/assets pass the
structural/link checker; whitespace checks pass. Synthetic confusion counts and
A/B/C summary arithmetic were checked against the downloadable values. Actual
foundation and continuation output schemas were inspected without model imports.
51 V8 DOM-harness assertions pass across all five new pages for deep links,
section isolation, blocked storage, mobile menu state and print restoration.
No real browser rendering, training, commit, push or deployment was performed.

## Real-data visualization follow-up

Authorized budget: one CPU job, 4 cores, 8GB, at most 15 minutes; no neural
inference or training. Export source was committed as 8ca0932 for the existing
HEAD-snapshot submitter. Sandbox submission failed before obtaining a job ID;
the successful submission was job 23941309. It ran for 23 seconds, completed
two five-figure sets, then failed when the third fixed record had no qualifying
window. No retry or expanded data scan was launched.

All ten produced PNGs were individually inspected. They show real source
observations for fire_24935867 and fire_24935874, input 2021-02-16 and target
2021-02-17, plus the preceding day. Similar spatial structures mean these must
not be presented as independent fire cases. The third record is explicitly
recorded as unavailable under the bounded selection rule.

The original exporter saved its manifest only at the end. Recovery therefore
records PNG checksums, visible dates, source commit/job and the crop rule; it
leaves array hashes, source crop offsets and exact confusion counts unavailable.
The recovery script is narrowly scoped to these reviewed outputs. Future exports
now save manifests incrementally and skip nonqualifying records. That numerical
export change has not been rerun.

Ten real figures are integrated into lessons 1/2/3/5 and a shared gallery, with
native magnification disclosures. The main course points to real observations;
Cedar remains explicitly optional synthetic arithmetic. Persistence error maps
are labelled as copying yesterday's mask, not as ERM/cosine predictions.

Validation: 144 pages and 8,438 local links/assets pass the structural/link
checker; tracked whitespace checks pass. All ten imported PNG checksums match
the individually inspected exports, and maintained Python files parse. Images
were inspected directly; rendered desktop/mobile browser layouts remain untested.
The source preparation commit is local; final page/asset changes are not pushed.

## Linear navigation and interactive concepts

Requested: reduce competing links, keep a reliable return to the originating
lesson step, and add more visual interaction without new compute. Implemented
per-step reading disclosures, collapsed lesson/directory navigation, text-only
experiment-name definitions and an inherited, validated lesson-return URL.
Supporting pages carry the route even when storage is blocked or a new tab is
opened. Raw documents and external sources open separately.

Added four in-place real-image explorers (two records × five views, guided
questions, revealable explanations and 1–4× magnification), a live comparison
contract builder in the schedule lesson, and specialist selection/cost visuals.
Existing quizzes and demos remain. No new images, data processing or jobs.

229 V8 DOM-harness assertions passed for lesson/hash state, exact and nested
return paths, invalid-route rejection, blocked storage, all image/view choices,
zoom, all eight contract combinations, specialist routes/costs and print restore.
The URL API was represented by a minimal test stand-in; this is not a rendered
browser test. Final structural checks are recorded after rebuilding.

Final rebuild: 144 pages and 8,095 local links/assets pass; whitespace checks pass.
Exposed HTML links within lesson steps, excluding the main next buttons, fell
from 63 to 9; supporting targets remain available in disclosures.
Real image legends and scaling caveats remain explicit. Browser rendering and
actual keyboard traversal were not available on this host. No commit or push
was performed for this follow-up.

## Task-led learning flow (2026-10-14)

User-approved scope: one clear task and next action at every point; a single
start/continue entry; contextual reading; separate ERM study/practical paths;
reliable step and interaction restoration; guided real observations before free
exploration. Implement in the existing static shell without dependencies.

- [x] Simplify the homepage and give every lesson step an explicit task and
  named next action (`build-course-lessons.py`, maintained lesson fragments).
- [x] Split ERM preparation into sequential practical tasks, preserving existing
  lesson anchors; browser study proceeds from planning to interpretation.
- [x] Replace detached reading menus with contextual disclosures; make supporting
  chapter footers return to the originating task, with further reading secondary.
- [x] Restore valid saved steps before saving; persist validated controls and
  disclosures; explicit restart clears only the current lesson. Handle blocked
  storage, malformed state, old anchors, direct links and browser history.
- [x] Start real-image activities with a fixed observation task and move optional
  controls behind an exploration disclosure, with a return-to-task-view action.
- [x] Rebuild all pages; check links, task coverage, default/explicit resume,
  practical routes, reference return, interaction restoration and print behavior.
  Use actual scripts in a DOM harness; do not claim browser rendering evidence.


Validation: 236 assertions passed against generated page fixtures and the actual
JavaScript in the dependency-free DOM/URL harness. Covered default and explicit
resume, all lesson task labels, ERM study/practical sequences, deep-link priority,
per-option feedback, persisted selections and computed displays, image/view/zoom
restoration, explicit restart, nested reference returns, invalid/corrupt state,
blocked storage, pageshow restoration and print restore. The final site checker
reports 144 pages and 8,707 local links/assets with zero errors. Whitespace checks
pass. No browser is installed here; desktop/mobile rendering, native history and
keyboard traversal remain unverified. No new compute or data export was run.
The preceding question-analysis update was committed and pushed as 2947f06.


## Continuous reading instead of slide navigation

Implemented the user's long-page request across lessons and supporting chapters.
All sections remain visible. The lesson contents is a simple in-page anchor list;
intermediate lesson pagers are removed, leaving only navigation to the next
lesson at the end. ERM's five optional practical sections are labelled and always
readable, with a shortcut to the worked result for browser study. The sidebar's
five unrelated resource links are removed; contextual resources remain.

Reading progress follows natural scrolling, with replaceState rather than an
extra history entry for every section. Saved entry and direct anchors scroll to
the correct section. Reference visits preserve explicit origin; visits without
an origin can continue the last saved lesson under an honest fallback label.

Validation: 332 assertions passed on actual generated HTML and scripts in the
simulated DOM/URL harness, including all-visible navigation, scroll save/resume,
quiz and figure state, print, reference returns and blocked storage. The site
checker reports 144 pages and 7,907 local links/assets with zero errors. Browser
rendering and native scrolling/history remain untested on this host.


## Grouped course navigation

User approved separate Learning path, Practical handbook and Reference library
sidebar groups. The shell now renders all three on every page, including home.
Only the relevant group starts open, and the actual page is highlighted rather
than a related lesson. The handbook follows environment, data/weights, execution,
verification and recovery stages. The library groups concepts/examples, experiment
lookup and historical research records. Individual experiment pages add their
current title beneath the experiment index. Lookup groups are explicitly optional.
Existing originating-lesson returns remain active for sidebar and inline links.

Static checks passed for the three groups, exact current-page destination and
active-group expansion across all 144 pages. The site link checker passes 12,046
local links/assets. Continuous-reading behavior was rechecked with the DOM harness;
real browser rendering remains unverified in this environment.
