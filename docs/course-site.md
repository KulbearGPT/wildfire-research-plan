# Maintaining the teaching website

The root `index.html` is the central directory for eight progressive lessons,
practical handbook chapters, reference readings, and experiment walkthroughs.
Public content is English. All pages are static and share the same course shell.

## Maintained sources

| Content | Edit here | Generated output |
|---|---|---|
| Student readiness, running case, handover, report and help | `site-content/en/student-support/` | `handbook/student-*/index.html` and teaching JSON downloads |
| Curriculum and lesson sections | `site-content/en/course-lessons.json`, `site-content/en/lessons/*.html` | `index.html`, `lessons/*/index.html` |
| Homepage resources | `site-content/en/course-resources.html` | `index.html` |
| Experiment names and high-level instructions | `site-content/en/experiment-names.json`, `site-content/en/experiment-walkthroughs.json` | `experiments/index.html`, `experiments/*/index.html` and worksheets |
| Practical and research handbook | `site-content/en/docs/` | `handbook/*/index.html` |
| Historical reference readings | `site-content/en/reference/` | `library/foundations/index.html`, `library/reproduction/index.html` |
| Shared layout and naming/link transforms | `scripts/course_shell.py` | All generated HTML |
| Presentation and interactions | `assets/learning.css`, `assets/learning.js`, `assets/lesson-demos.js` | Loaded directly |

The lesson builder also runs `scripts/build-student-support.py`. Maintain the
continuous blank course record in `student-record.md`. Cedar arrays and report
excerpts are explicitly synthetic and must never be presented as measured data.
The report guide maps the distinct foundation and continuation output schemas;
check its field paths against the evaluators when those schemas change.

Edit lesson worksheets under `lessons/` directly. Experiment worksheets are generated.
Keep the ERM report template and results CSV blank; teaching values are synthetic.
Experiment names explain the method, while existing IDs remain stable in code,
configurations, records, and URLs. See [naming policy](experiment-names.md).

Handbook Markdown links are written relative to the corresponding canonical
`docs/` file, even though the renderer reads the English counterpart. Update both
versions when scientific commands or protocols change. Adding research documents
to the discovery set requires their English counterparts. Lesson fragment links
are relative to their generated output pages.

## Build and check

From the repository root with Pandoc available:

```bash
python scripts/build-course-lessons.py
python scripts/build-course-references.py
python scripts/build-course-guides.py
python scripts/check-course-site.py
git diff --check
```

These are lightweight documentation operations with no model imports, network,
or dataset processing. Commit maintained sources and generated pages together.
The checker verifies HTML nesting, IDs, labels, local links and destination
anchors, and absence of retired pages. It is not a rendered-browser test.
Check desktop/mobile layouts, keyboard navigation, and print output in a browser
when available. No model jobs are needed for teaching-only changes.

## Retirement and publication

The old `guides/**/*.html`, `related-work/index.html`, and
`baseline-reproduction/index.html` are retired. Do not recreate them, redirect
stubs, or migration notices. Old URL aliases in source are translated only at
build time into current handbook/library links. Preserve `related-work/assets/`
for figures still used by the new reference pages. Canonical research Markdown,
source code, and evidence records remain available as supporting material.

Deploy the repository root together with assets, downloads, and linked evidence.
A local build, commit, push, and successful deployment are separate milestones;
report only those actually verified.

## Teaching and scientific checks

Each lesson or experiment should explain its question, prerequisites, action,
comparison, and deliverable. Keep high-level execution steps on walkthrough pages;
link detailed commands and original evidence from the appropriate handbook chapter.
Mainline experiments appear first; historical entries remain searchable and are
explicitly scoped as historical, diagnostic, invalid, or unrun where applicable.

Preserve the distinction between official Fold 2 and the project protocol,
controlled and natural missingness, execution qualification and effectiveness,
and historical signals and independent confirmation. Preserve original commands,
anchors, figures, and evidence dates when reorganizing reference material.

Lessons and reference chapters remain visible as continuous articles. JavaScript
tracks the reading section while scrolling and supports per-page resume. The existing ERM storage key remains compatible.
Without JavaScript all content is readable; lesson/reference printing expands
sections and disclosures and restores their prior state. Deep links open the
containing section and disclosures. Completion checkboxes record review, not
verified experimental completion. Storage failure must not prevent reading.

## Real-data examples

The real-data exporter is `scripts/course/export-real-examples.py`. It must run
in a bounded Slurm CPU allocation, using read-only prepared HDF5 inputs and a new
external output directory. The exporter records selected date/crop identities, array digests, corruption
fractions, source commit and job ID incrementally in `manifest.json`. An event
without a qualifying window is recorded as skipped within the fixed scan bound. These are teaching cases selected for visibility, not a random
performance sample. No neural-model inference is performed: the error panels use
persistence (copy the previous observed mask), labelled explicitly.

After reviewing exported figures, import only those figures and their manifest:

```bash
python scripts/course/build-real-examples.py --export /absolute/reviewed/export
python scripts/build-course-lessons.py
python scripts/build-course-guides.py
python scripts/build-course-references.py
python scripts/check-course-site.py
```

The import writes maintained lesson fragments, the real-data support page, and
`assets/real-examples/`. Normal documentation rebuilds do not access raw data.
Colors, date labels and captions must distinguish observed background, controlled
hidden inputs, next-day targets and prediction errors. Raw channel displays are
not natural-color photographs; independently clipped color scales are not shared
physical units. Synthetic Cedar and score examples remain labelled separately.

The first published asset set uses a reviewed partial export from job 23941309.
Its manifest explicitly records the job failure at the third event and unavailable
array-level metadata. `recover-reviewed-export.py` preserves this exceptional
provenance; do not use it to invent metadata for a new run. See the implementation
record for the bounds and actual checks.

## Task-led lessons and interactive examples

Each maintained lesson section declares `data-task` and `data-title`. The lesson
builder generates `assets/course-map.js` from these declarations, so navigation,
resume validation and reference return labels share the same task IDs. Preserve
old ERM `step1`–`step7` anchors; optional practical tasks use `step5-*` IDs.
The five optional practical sections remain visible and labelled. Browser study
can use an anchor to jump past them to the worked result. Review reminders never certify execution.

`course_flow.py` keeps optional explanations and full-reference links beside the
paragraph that needs them. Its short reading notes are high-level teaching copy,
not replacements for the authoritative execution records. Supplementary chapter
footers prioritize returning to the originating lesson task, inherited through
validated `via` query parameters. A direct reference visit without an origin can
offer “Continue your last lesson” from valid local progress; that fallback is
never labelled as the exact origin. Continued reading is secondary.

`learning.js` resolves explicit anchors or saved positions and scrolls to them
without hiding sections. Natural scrolling saves the current section and updates
the hash with replaceState, so it does not add a history entry per section. Ordinary directory links resume; the separate restart action clears
only that page’s activities and review reminder. `course-state.js` validates and
restores choice, slider, prediction and disclosure state, then replays normal UI
handlers to reconstruct feedback. Storage failure leaves navigation functional.
Printing must not save the temporary all-expanded state. Required preparation,
practical instructions, figure legends and task contents use `open
data-default-open`: they begin expanded on every visit, overriding old saved
collapsed states. Students can still collapse them while reading. Answers and
optional exploration remain closed initially; directory navigation also starts
expanded on mobile. The sidebar contains only the course sequence; support
resources are offered at their relevant lesson locations.

`course_visuals.py` supplies fixed observation tasks on real exported panels.
Additional records, views and zoom are optional exploration with a task-view
reset. `visual-lab.js` handles rendering and the comparison/routing exercises.
No probability values or exact pixel counts are inferred from PNG colors.

For behavioral checks, export actual generated markup and scripts with
`python scripts/course-test-fixtures.py`. Load the resulting JSON and call
`runCourseChecks(fixtures)` from `scripts/check-course-interactions.js` in a
JavaScript engine. With Node available, run `python scripts/course-test-fixtures.py | node scripts/check-course-interactions.js`. The dependency-free harness checks navigation, persisted
activities, ERM routes, reference returns, invalid state and printing. Its DOM
and URL objects are stand-ins: this does not replace rendered desktop/mobile,
keyboard or browser-history testing.
