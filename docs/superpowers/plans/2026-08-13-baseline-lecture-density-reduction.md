# Baseline Lecture Density Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the lecture's prose and card density while preserving its complete verified scientific story.

**Architecture:** Keep the existing single-file courseware architecture. Strengthen the publication contract test with density and evidence-preservation assertions, then consolidate repeated card grids directly in the lecture HTML and verify the rendered result.

**Tech Stack:** Static HTML/CSS/JavaScript, Python pytest contract tests, Node syntax checking, in-app browser visual QA.

## Global Constraints

- Preserve all 16 section IDs, exact twelve-fold AP values, aggregate results, and evidence-boundary claims.
- Keep English-only content and Lecture 02's existing visual system.
- Target 15–20% fewer visible English words and 16–18 card-like blocks.
- Do not alter experiment artifacts, model code, scientific processes, or published result files.

---

### Task 1: Freeze the compactness contract

**Files:**
- Modify: `tests/test_baseline_reproduction_lecture.py`
- Test: `tests/test_baseline_reproduction_lecture.py`

**Interfaces:**
- Consumes: `baseline-reproduction/index.html` as UTF-8 text.
- Produces: regression assertions for word count, card count, section IDs, and exact evidence markers.

- [ ] **Step 1: Add a failing compactness test**

Add a test that strips `<style>` and `<script>` blocks, counts visible English tokens, counts the page's card-class uses, and requires 2,250–2,400 words plus 16–18 card-like blocks. Reuse the existing exact result and boundary-marker assertions.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest tests/test_baseline_reproduction_lecture.py -q`

Expected: the new density assertion fails against the current 2,819-word, roughly 27-card page.

### Task 2: Consolidate prose and cards

**Files:**
- Modify: `baseline-reproduction/index.html`
- Test: `tests/test_baseline_reproduction_lecture.py`

**Interfaces:**
- Consumes: the existing 16-section lecture and Task 1 density contract.
- Produces: a shorter lecture with the same IDs, exact results, interactions, and evidence boundaries.

- [ ] **Step 1: Replace repeated card grids with compact structures**

Convert conceptual grids in framing, contract, verification, failure analysis, and evidence-boundary sections into numbered lists, short tables, or paired callouts. Retain cards for the hero metrics, vocabulary definitions, and genuine side-by-side comparisons.

- [ ] **Step 2: Tighten repeated explanations**

Remove repeated statements about official code, scientific reruns, and provenance where the same fact appears in adjacent sections. Keep one complete authoritative statement for every scientific boundary.

- [ ] **Step 3: Run the focused test and verify GREEN**

Run: `python -m pytest tests/test_baseline_reproduction_lecture.py -q`

Expected: all lecture contract tests pass.

### Task 3: Render, verify, and commit

**Files:**
- Verify: `baseline-reproduction/index.html`
- Verify: `tests/test_baseline_reproduction_lecture.py`

**Interfaces:**
- Consumes: the compact lecture from Task 2.
- Produces: browser-verified desktop/mobile/dark-mode output and a clean local commit.

- [ ] **Step 1: Inspect desktop and mobile rendering**

Render at 1440×1000 and 390×844. Confirm no page-level horizontal overflow, readable tables, correct sidebar behavior, and a visibly calmer card rhythm.

- [ ] **Step 2: Verify interactions and repository tests**

Run the focused lecture test, `python -m pytest -q`, `git diff --check`, and inspect browser console warnings/errors.

- [ ] **Step 3: Commit the implementation**

Stage only the lecture, its test, this plan, and its design spec. Commit with `docs: streamline baseline reproduction lecture`.

