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

JavaScript progressively selects one section, with optional per-page local
progress and course resume. The existing ERM storage key remains compatible.
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

## Linear lessons and interactive examples

`course_flow.py` groups exposed lesson reading links into a per-step disclosure;
links already inside disclosures remain there. It preserves instructional prose
and original targets. Lesson and course contents are collapsed by default during
a lesson, and experiment names are text rather than automatic navigation links.

`assets/course-route.js` adds a validated `via=lesson.stepN` query to supporting
HTML links. Every supporting page provides an exact return route and propagates
it into deeper reading. It uses no storage and preserves destination fragments.
Opening another main lesson intentionally establishes that lesson as the new
origin. With JavaScript disabled, use browser Back; all content remains readable.
Images, raw documents and external sources open separately so the course remains.

`course_visuals.py` emits real-export explorers, a schedule comparison design
exercise and a specialist route/cost display. `assets/visual-lab.js` enhances
them in place; static initial views and explanations remain readable without JS.
The explorers reuse existing PNGs without claiming new inference or recoverable
pixel values. Design feedback checks only the named choices, not full provenance.
Costs count continuation updates, not GPU-hours or a full confirmation campaign.
