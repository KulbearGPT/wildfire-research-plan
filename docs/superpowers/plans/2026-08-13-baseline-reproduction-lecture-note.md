# Baseline Reproduction Lecture Note Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish and verify an English-only, beginner-friendly baseline reproduction lecture that matches the existing related-work courseware.

**Architecture:** Create one self-contained static HTML courseware page by reusing the established lecture shell and interactions. Add only navigation links to existing pages and a standard-library HTML contract test; consume the already committed reproduction evidence without executing scientific code or regenerating artifacts.

**Tech Stack:** HTML5, inline CSS, vanilla JavaScript, Python standard-library `html.parser`, pytest.

## Global Constraints

- Use only verified values from the committed reproduction documentation and atomic twelve-fold publication.
- State that the twelve-fold experiment is released-weight test-only, not twelve new training runs.
- Do not change, rerun, or regenerate scientific artifacts.
- Preserve all existing public URLs, translations, content, and interactive behavior.
- Match `related-work/index.html` in visual shell, spacing, controls, responsive behavior, and print treatment.
- Keep the new lecture English-only.

---

### Task 1: Freeze the HTML publication contract

**Files:**
- Create: `tests/test_baseline_reproduction_lecture.py`
- Read: `related-work/index.html`
- Read: `docs/experiments/res18_unet_t1_reproduction.md`

**Interfaces:**
- Consumes: verified twelve-fold values and the existing courseware DOM pattern.
- Produces: a failing executable contract for the new page and navigation.

- [ ] **Step 1: Write the failing structural and evidence tests**

Use `html.parser.HTMLParser` to collect element IDs, headings, links, tables,
controls, and scripts. Assert the new file exists; its sixteen IDs are unique;
the TOC exactly covers them; the twelve AP values, aggregate values, paper
reference, released-weight boundary, no-relaunch boundary, focal-alpha future
ablation, and sampled-GPU caveat are present.

- [ ] **Step 2: Add failing navigation and local-link tests**

Assert `index.html` links to `baseline-reproduction/`,
`related-work/index.html` links to `../baseline-reproduction/`, and every local
`href`/`src` in the new page resolves inside the repository.

- [ ] **Step 3: Run RED**

Run: `python -m pytest tests/test_baseline_reproduction_lecture.py -q`

Expected: FAIL because the lecture page and navigation do not yet exist.

---

### Task 2: Build the beginner-friendly lecture page

**Files:**
- Create: `baseline-reproduction/index.html`
- Modify: `index.html`
- Modify: `related-work/index.html`
- Test: `tests/test_baseline_reproduction_lecture.py`

**Interfaces:**
- Consumes: Task 1 contract, existing courseware CSS/JS shell, verified evidence.
- Produces: the final public lecture and discoverable course navigation.

- [ ] **Step 1: Create the courseware shell**

Copy the structural pattern of lecture 02: semantic topbar, hero, truth banner,
layout, sidebar, main sections, footer, and progressive-enhancement script.
Change only lecture-specific branding, metadata, navigation, and content.

- [ ] **Step 2: Write the sixteen-section teaching narrative**

Introduce vocabulary before procedure. For each stage, explain why it exists,
what was done, what evidence proves completion, and how it helps later research.
Use cards for terminology, a flow diagram for evidence movement, timelines for
execution, tables for folds/aggregates, and callouts for scientific decisions,
engineering fixes, and evidence boundaries.

- [ ] **Step 3: Add course navigation**

Add a lecture 03 link/card to the main site and a `Next lecture` link to lecture
02. Add `Previous lecture` and `Research plan` links to lecture 03.

- [ ] **Step 4: Run GREEN and inspect the diff**

Run: `python -m pytest tests/test_baseline_reproduction_lecture.py -q`

Expected: PASS. Then run `git diff --check` and confirm no existing scientific
number or claim outside the intended navigation lines changed.

---

### Task 3: Render, audit, and deliver

**Files:**
- Modify if required: `baseline-reproduction/index.html`
- Modify if required: `tests/test_baseline_reproduction_lecture.py`

**Interfaces:**
- Consumes: the completed HTML and tests.
- Produces: verified desktop/mobile screenshots, a clean commit, and a final clickable page.

- [ ] **Step 1: Run focused and full verification**

Run:

```powershell
python -m pytest tests/test_baseline_reproduction_lecture.py -q
python -m pytest -q
git diff --check
```

Expected: all tests pass and the diff check exits zero.

- [ ] **Step 2: Render the page locally**

Serve the repository with a read-only local HTTP server. Inspect desktop
1440×1000 and mobile 390×844 views in light and dark themes. Exercise sidebar,
search, progress, font controls, and Print/PDF styling. Confirm no overflow,
clipped text, broken local links, missing controls, or unreadable tables.

- [ ] **Step 3: Fix only verified rendering defects and re-run checks**

For each defect, first add or strengthen a focused contract where practical,
apply the smallest HTML/CSS/JS fix, and re-run the focused suite plus the
relevant render inspection.

- [ ] **Step 4: Commit the final lecture**

```powershell
git add docs/superpowers/specs/2026-08-13-baseline-reproduction-lecture-note-design.md docs/superpowers/plans/2026-08-13-baseline-reproduction-lecture-note.md baseline-reproduction/index.html index.html related-work/index.html tests/test_baseline_reproduction_lecture.py
git commit -m "docs: add baseline reproduction lecture"
```

- [ ] **Step 5: Request final review and show the page**

Review exact values, claim boundaries, navigation, DOM semantics, scripts,
responsive screenshots, full test output, diff/status, and commit contents.
After approval, open the final local HTML page for the user and provide its
clickable repository path.

