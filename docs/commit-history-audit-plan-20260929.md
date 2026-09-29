# Commit-history audit plan after local integration

Date: 2026-09-29. Status: reviewed proposal; no history rewrite or push performed.
This is the current history plan. It supersedes the pre-merge grouping proposal
in section 6 of the [repository audit](repository-audit-plan-20260929.md).
That document still owns the separate reproducibility work.

The objective is a history in which a reviewer can understand and, where useful,
revert one behavior together with its tests and documentation. Minimizing the
number of commits is not the objective. Experimental sources and the sequence
of corrections, results and decisions must remain recoverable.

## 1. Frozen audit input and review coverage

These counts precede the documentation commit containing this plan:

| Item | Verified state |
|---|---|
| Local main | `d2228d1da011254240adff6998b62ccc13f595d8` |
| Remote main | `5b457c3ea6f2f6063ee260edae7ff8404709b505`, checked with `git ls-remote --heads origin` |
| Reachable history | 552 commits; 356 on the first-parent line; 6 merge commits |
| Ahead of remote | 28 reachable commits, but only 10 first-parent commits |
| Composition of those 28 | 4 ordinary commits before integration + 18 original side-branch commits + 5 integration merges + 1 integration record |
| Branch integration | All local branch tips are ancestors of main; the nine original worktrees remain |
| Existing pre-integration backup | `archive/pre-local-merge-20260929` at `38736a2`; this does **not** cover the later merge result |

The four ordinary commits are `5256e2d`, `82c6b4d`, `04a1945`, and `38736a2`.
The final integration record is `d2228d1`. The
[integration report](local-merge-20260929.md) records all five merge decisions.
Being reachable from main does not make an old worktree's ignored assets safe
to delete, and being unpublished does not make an experiment SHA disposable.

Coverage of this audit: current refs/parents and first-parent subjects; file
boundaries for the named candidates below; targeted source and diff inspection
of the paper pipeline, mainline checks, retirement guards, workflow update and
integration. It is **not** a hunk-by-hunk certification of all 552 commits.
Older unlisted commits default to **retain pending review**, not squash.

## 2. Scope and decision rules

Two rounds are proposed:

1. **Local pilot:** reorganize the unpublished first-parent work after
   `5b457c3` on a review branch, preserving the original side-branch parents.
   Leave the published prefix and existing main untouched during review.
2. **Published-history review:** inspect the September handoff, cleanup and
   website groups below. Build a separate candidate only where the benefit is
   clear. Do not rewrite the entire research history simply to normalize titles.

Use `split` for independently meaningful behaviors with separate failure modes;
`squash` for a follow-up that completes the same change without hiding a result
or correction; `keep` for an atomic change, scientific milestone, or merge
that preserves source ancestry. `Defer` means the boundary needs more evidence.

Each implementation unit carries its relevant tests, config, lifecycle entries
and usable documentation. A test that imports a module lands with or after that
module. English website source and generated HTML stay together. Repeated
lifecycle headers may belong to one large mechanical commit.

History editing must preserve the final tracked tree. Bug fixes, path migration,
test additions and scientific reanalysis are separate additive work. In
particular, the collector's unconditional `git rev-parse` in an archive without
`.git` is a known reproduction issue, not a fix to hide inside this rewrite.
Inherited limitations must be recorded; intermediate commits are not promised
to provide capabilities that the original final tree does not provide.

## 3. Round 1: exact local candidate series

Start at the unchanged published `5b457c3`. Replace the pair `5256e2d` /
`82c6b4d` with the following **three** units. The earlier four-way suggestion is
refined here: the build logic and its scientific reporting are closely coupled,
so splitting author interpretation into an artificial fourth unit adds little.

| Unit / proposed subject | Source and ownership | Dependency and acceptance |
|---|---|---|
| P1 — `feat(paper): collect normalized evidence with original source provenance` | `5256e2d`: `scripts/paper/collect_evidence.py`, collector-specific `method-inventory.json` registration, `paper/materials/source-notes.md`, and the materials README title/evidence-boundary/reporting-convention hunks | No builder/renderer imports. Preserve original source identities, archived RF/TD separation, gaps and explicit historical/current route distinction. Do not label collection as a complete paper build. |
| P2 — `feat(paper): render the six figures from exported evidence tables` | `5256e2d`: `scripts/paper/render_figures.py`, `FIGURE_SPECIFICATION.md`, `FIGURE_CAPTIONS.md`, renderer lifecycle registration; include the caption correction from `82c6b4d` | Documents the table schema it consumes; no claim that tables have been produced by this commit. The caption must retain the increased training-budget caveat. |
| P3 — `feat(paper): validate and package evidence with closest-control attribution` | `5256e2d`: `build_materials.py`, `test_paper_materials.py`, remaining author prose/README, `.gitignore`, builder lifecycle registration, route-audit notes in `CODE_LIFECYCLE.md` and `t1_t5_innovations.md`; plus remaining `82c6b4d` hunks | Requires P1/P2. Keep AP-source validation, its report field, closest-control preview text and matching claims together. All five aggregation tests must import the builder present in this commit. All package inputs and maintained document links must resolve. |

All three lifecycle entries use `paper/materials/README.md` as `settings_ref`,
so its applicable foundation sections must exist in P1. Leave its full-package
description, contents, commands and links to later author files for P3.
For all three units, move any other forward-referencing documentation hunk to P3
instead of introducing a broken link or duplicating instructions. Shared JSON
and Markdown files require hunk ownership, not whole-file cherry-picks. Avoid
refactoring functions just to create these boundaries. Require P3's final tree
to equal original `82c6b4d` before replaying any subsequent commit.

The `82c6b4d` split is specifically:

- P2: the `FIGURE_CAPTIONS.md` correction from an isolated-effect claim to a
  comparison that also increases specialist training budget.
- P3: the `build()` original-summary AP loop and `original_ap_matches` report
  field/check; the `build_preview()` closest-control calculation and sentence;
  corresponding claims in `CLAIMS_AND_LIMITATIONS.md`, `EXPERIMENTS_DRAFT.md`
  and the materials README.

The implementation already composes X22+mixed-X14 in `5256e2d`; `82c6b4d`
adds source verification and explicit interpretation. Do not describe it as
new training or the first implementation of that control.

Then preserve these boundaries in order:

| Original | Proposed treatment | Required preservation |
|---|---|---|
| `04a1945` | Keep the delivery record separate after P3 | `paper/README.md` reports the real v2 build and original source SHAs; it must not claim the newly edited series was run by the old job |
| `38736a2` | Keep the reproducibility/audit plan separate | Its counts are an explicitly dated pre-integration snapshot |
| `2caeabc`, `2458b61`, `eea6640`, `08add57` | Retain four ancestry merges in the candidate topology | Zero tree change relative to each new first parent; exact original second parents below |
| `a83d0eb` | Retain the TD merge with its resolution | Three archived helpers, distinct archived TD plan, lifecycle map and documentation additions only |
| `d2228d1` | Keep the integration record separate | It describes the original integration and original SHAs, not a newly executed experiment |
| Later audit-only commits, including this plan | Replay as a final documentation group | Re-freeze the actual input HEAD before starting; do not silently omit commits added after this snapshot |

The proposed core is 11 first-parent commits instead of the current 10 after
`5b457c3`, before later audit-only commits. The extra boundary makes the paper
pipeline reviewable; this is not a commit-count reduction exercise.

### Preserve the merge topology

| Merge | Original second parent to retain unchanged | Candidate-tree condition |
|---|---|---|
| `2caeabc` | `f451934` — belief attention | Identical to its new first parent |
| `2458b61` | `fac140f` — belief filter | Identical to its new first parent |
| `eea6640` | `12cdc2d` — belief reconstruction | Identical to its new first parent |
| `08add57` | `f91d7eb` — tutorial branch | Identical to its new first parent |
| `a83d0eb` | `3761e8b` — TD campaign | Same resolved tracked tree as the original merge, before any later audit-only changes |

The original cross-history merge `d5183c0` is below the pilot base and remains
untouched. The 18 newly reachable side commits are retained as original commits;
do not replay them as 18 new patches or delete them as apparent duplicates.
Keep the intentional legacy-test deletion and current portable/BN fixes.
An ancestry-only merge is useful here even though its file diff is empty.

Do not run a plain linear interactive rebase over `origin/main..HEAD`. That
range includes the side history and merge resolutions. Use explicit series
construction with the recorded parents, or carefully inspect a merge-preserving
rebase plan. `--rebase-merges` alone does not verify the conflict resolutions.

## 4. Round 2: published candidates and concrete boundaries

These are recommendations for a separate review, not authorization to replace
published main. The priorities below express review value, not execution order.
Actual reconstruction follows original ancestry and the dependencies in section 5.

### Highest value: separate mixed behavior changes

| Original commit(s) | Proposed units | Boundary to inspect before accepting |
|---|---|---|
| `5b457c3` | Four units: route validation; evaluation identity; archived BN entrypoint guards; archive/navigation documentation | Route unit owns `route_validation.py`, both affected composers and `test_route_composition.py`. Identity unit owns `checkpoint_contract.py`, the evaluation-only hunks of `run.py`, and `test_evaluation_identity.py`. BN unit owns both audit scripts and `test_archived_bn_entrypoints.py`. Partition inventory/README hunks accordingly; English/generated changes follow their source. |
| `6756188` | Two units: narrow mainline CLI with tests; active/archive recipe separation | `mainline.py` + `test_mainline_cli.py` + CLI registration first; recipe/archive files and corresponding guides second. Keep links and command names valid at each boundary. |
| `f0b95af` | A mechanical lifecycle-classification unit; a retirement-guard unit; a navigation unit only if independently useful | Preserve the consistent inventory/header classification together. The two `run_*confirmation_bundle.sh` scripts add `exit 2`, which is a behavior change, not just a comment. Keep those guards with their retirement documentation and relevant verification. |
| `454f192` | Three units: portable data preparation; verified teacher bundle transfer; pip isolation after module loading | Own `prepare-data.sh` + data-preparation test/recipe; `artifact_bundle.py` + its test/artifact docs; `job.sh` isolation hunk + launcher regression tests. Partition `qualify-cpu.sh` registrations and shared reproduction prose. |
| `58cf86a` | Baseline completion/export contract; scientific evidence catalogue | `complete_baseline.py`, trainer receipt/export changes and its test belong together. Positive/negative result pages and inventory classification are a separate evidence organization change. |
| `e2b893f` | Restore routed TD campaign; repair temporal bootstrap export | Keep TD methods, runner, source contracts, report tests and restored evidence identities together. Put the `runtime.py` fix with `test_wsts_fast_track_runtime.py`. Do not relabel the TD campaign as RF. |
| `573206d` | Portable path contract; dependency-complete method-family restorations | `paths.py` and `test_portable_paths.py` must precede consumers. Review shared `contract.py`, `runtime.py`, cross-history data/models and dependencies before splitting model families. A module's archived status does not mean active callers can lose it. Exact family count remains deferred. |
| `7416004` + `19132b0` | Diagnostic recovery/portability; teacher manifest and checkpoint provenance; legacy P00 recovery; qualification setup/results | Requires hunk review of shared diagnostic helpers, `run.py`, `checkpoint_contract.py`, `job.sh`, qualification scripts, inventory and recipes. Place later portability fixes into the unit they repair; preserve actual execution records separately. Do not squash the entire pair. |
| `c617100` | Upstream patch/setup recovery; architecture asset and qualification support | Keep `res18_import_scope.patch`, `prepare-upstream.py`, corresponding `job.sh` hunks and regression tests together. Place `fetch-architectures.sh` and architecture-specific qualification after their prerequisites. |
| `eb702c2` + `53ddec4` | Scientific population correction; converter/replay qualification code; completed converter qualification record | Retain the 2023 population correction from 2,312 to 2,102 without changing AP values. Keep `53ddec4` as the actual completed-run record, not part of an implementation allegedly tested earlier. |

### Bounded squash candidates

| Original commit(s) | Recommendation | Why / limiting condition |
|---|---|---|
| `89da1d4` + `b74e500` | Squash into one workflow update | Agreement, two skills, project defaults, development guide and onboarding links implement the same update. Preserve the limited/static validation wording and synthetic exercise status. Include English source and generated onboarding page. |
| `5088fc8` + `619787e` | Regroup into renderer/style infrastructure, then complete English course publication; allow one combined commit if the renderer/content interface cannot be separated cleanly | Renderer must have the English sources it requires when used. Keep each generated page paired with its source, and standalone HTML separate from generated output ownership. Do not introduce an intermediate published language mismatch. |
| `c20b87d` and `533b325` | Consider folding navigation/closure into their matching handoff documentation units | First check for new outcome, qualification or source information. Do not fold the closure into initial `410c237` as if the original plan already had execution evidence. |
| Submission/pending/status-only updates | Consider consolidation within the same campaign | Only after checking for changed configs, seeds, selection rules, failures, cancellations or interpretation. Preserve job-to-source mappings and decision chronology in a dated record. No title-only automatic squash. |

`08ebabd` is a useful onboarding milestone and can stay. `98d12f0` is a
coherent site-configured Slurm foundation and should remain unless diff review
finds an independent behavior mixed into it. Avoid moving runtime qualification
records earlier merely to make the history look more compact.

### Scientific and provenance anchors to retain

| Area / examples | Treatment |
|---|---|
| Original data/label contracts and corrected baseline development, including `5128537` and `4b6717a` | Preserve implementation, corrected evaluation and outcome boundaries. Earlier July/August history needs further diff review before any squash proposal. |
| X22 `e3e309a` and X17 `9ba378a` | Keep separate interventions: cosine schedule versus fixed-severity specialization |
| `a133338`, `3d01992`, `28874e5` | Keep X17 attribution outcome, X22 adoption and complete-route evidence as distinct decisions |
| BN development/corrections: `fce9583`, `535b234`, `eb01d0f`, `0cdddc2`, `7a58b38` | Preserve the distinction between implementation, efficiency fix, corrected normalization behavior and valid/invalid outcomes; do not hide the reason earlier screens changed interpretation |
| `a7e8e0a`, `bf6bff6`, `2fa7834`, `fb565b5` and qualification JSONs | Preserve original run receipts and tested source identities. These are limited execution qualifications, not full retraining claims |
| `875a4b2` | Review a possible split: `qualify-legacy.py` entrypoint extension versus completed first-GPU-batch report; do not treat all changes as documentation |
| `aa46860` | Keep the bounded attention qualification script as a distinct test capability |
| `d5183c0` and the five local integration merges | Preserve source ancestry and resolution semantics |

These names are a starting set, not the complete list of evidence-linked SHAs.
Before moving any other commit, inspect its references in ledgers, qualification
records, source manifests, Slurm receipts and the paper package. A `docs:` title
can introduce a result or change a scientific conclusion.

## 5. Dependency order and audit record

For the published round, maintain this dependency order where the touched hunks
interact: portable paths and site setup → data/teacher/baseline contracts →
restored consumers and checkpoint provenance → qualification capabilities →
actual qualification records → complete teaching publication → workflow update
→ lifecycle/retirement → narrow CLI/recipes → comparison validation → paper
pipeline. Independent commits need not be reordered to match this outline.
Research interventions and the original cross-history merge retain their
existing earlier ancestry. A corrected outcome stays after the correction.

Create one review row for **every commit in each selected rewrite range**, not
just the large commits named here. A compact Markdown table or CSV is enough:

```text
old_sha, parents, published, role, inspected_paths_or_hunks,
action, target_units, prerequisites, source_or_job_references,
protected_original_ref, review_state, new_shas, validation
```

Use full SHAs in the executable mapping. A split has one old SHA and multiple
new SHAs; a squash has multiple old SHAs and one new SHA. A regrouping can be
many-to-many. Record merge parent mappings separately. Keep `new_shas` empty
until the candidate exists; metadata inspection is not hunk-review completion.
Keep this record outside the candidate tree until exact-tree verification, then
add it as an explicit final documentation commit.

## 6. Execution stages and acceptance gates

| Stage | Work | Exit condition |
|---|---|---|
| A. Freeze and recoverability | Recheck main/remote/worktrees; protect the actual current main, remote tip and original branch tips; create and verify an all-refs Git bundle, including a trial restore in a separate repository | Every protected SHA resolves after restoration; baseline tree and ref list recorded. The bundle is a Git-source backup, not a backup of ignored checkpoints or deliverables. No worktree deletion or GC |
| B. Finalize hunk ownership | Fill the review rows for the selected round; inspect referenced evidence; confirm dependency boundaries and inherited failures | Every selected commit has an explicit decision; all uncertain groups stay outside the rewrite or remain unchanged |
| C. Construct local pilot | New review branch/worktree from `5b457c3`; P1–P3; replay records; reconstruct the five merges with exact old side parents; replay later audit-only commits | Existing main and original branch refs unchanged; no missing source parents or resurrected archived behavior |
| D. Verify candidate | Inspect each boundary and final tree; perform relevant static checks; use targeted Slurm checks for computational paths if needed | Exact final tree equality against the frozen input before adding the mapping/report; source reachability and merge conditions pass; intermediate dependency checks documented |
| E. Review published groups | Apply A–D to a separately scoped candidate, using the inventory above; keep untouched research history intact | All selected published changes have a reviewed map, preserved evidence anchors and final-tree verification; no unreviewed range silently rewritten |
| F. Deliver review result | Add mapping and validation report after the tree-equality checkpoint; show the candidate graph and remaining caveats | User can inspect the concrete series before choosing a local promotion or publication strategy |

The protected original source history must remain retrievable even if a curated
branch is later published. Prefer initially publishing the candidate as a
separate branch. Replacing published main is a distinct publication decision;
this plan neither performs nor requests a force-push.

A future promotion is a local ref change after review, not a normal merge of
the old and curated series: merging both would put both commit organizations
back into the main history. Keep the original on its protected archive ref.
If the remote advances, refreeze and review the additional work first.

### Verification details

During construction, use Git/source checks on the login node: whitespace,
changed-path review, Python AST/shell syntax where relevant, import/config path
presence without importing models, and local documentation links. Inspect
source/generated page pairing at every affected website boundary. Rebuild
generated guides with the repository renderer only in a disposable check area;
do not silently change frozen output to compensate for tool-version drift.

For the final candidate, the following are command patterns, run from the
repository with `history_original_ref`, `history_candidate_ref` and
`history_base_ref` assigned to the frozen input, candidate and rewrite base:

```bash
git diff --exit-code "$history_original_ref" "$history_candidate_ref" --
git rev-parse "${history_original_ref}^{tree}" "${history_candidate_ref}^{tree}"
git log --graph --oneline --decorate "$history_candidate_ref"
git range-diff "${history_base_ref}..${history_original_ref}" "${history_base_ref}..${history_candidate_ref}"
```

Matching tree IDs are the final tracked-content gate. `range-diff` supports
review of ordinary patch correspondence; it is not a merge/provenance proof.
Separately check the recorded parent lists, each ancestry-only merge's empty
first-parent diff, the TD resolution tree and each preserved branch tip with
`git merge-base --is-ancestor`. Unchanged original refs must still resolve to
their recorded SHAs. No stage may rely on unreachable reflog objects as the
only copy of an experimental source.

If numerical tests are needed, P3's existing `test_paper_materials.py` is the
local pilot's focused CPU target. Published splits use their own route,
identity, launcher or restoration tests, in the correct environment and Slurm
allocation. Verify the candidate is committed before using the submitter,
which archives HEAD. Do not rerun all qualifications or training solely because
commit IDs changed. Exact source preservation does not newly certify runtime
reproducibility; a new execution record names its actual new source SHA.

The paper build cites evidence snapshot `5b457c3`, builder `82c6b4d` and CPU
job `22717106`. Preserve those identities in `paper/README.md`, original JSON
and the existing ZIP. Do not regenerate historical packages merely to replace
their SHAs. Record old-to-new correspondence alongside them, without claiming
the new SHA produced the historical result. Git-source protection is required
before rewriting; full data/checkpoint replay is a separate reproduction task,
not a prerequisite for reviewing same-content commit boundaries.

If a boundary fails, repair the grouping or leave that original commit intact.
Abort the candidate before touching main if provenance or final-tree equality
cannot be established. Rollback means returning to the protected original ref;
never resetting a dirty user worktree or deleting ignored assets.

## 7. Deliverables and current status

The implementation round should deliver a review branch, full-SHA many-to-many
mapping, protected refs plus a verified source backup, boundary-check results,
tree/ancestry verification, and an explicit list of deferred groups. No promise
is made now about the final total commit count or full experimental replay.

For this plan, only Git metadata, selected diffs/source and documentation were
inspected. No model imports, scientific aggregation, Slurm submission, artifact
regeneration, backup creation, branch rewrite or push was performed. The present
change is the plan and its navigation links. The recommended next implementation
step is stages A–D for the **local pilot**, before any published-history rewrite.

Documentation checks: 30 local links across the three changed Markdown files,
balanced code fences, all 54 cited commit identifiers, frozen history counts,
five recorded merge parents, four empty first-parent merge diffs, and original
branch-tip ancestry were checked. These checks validate the plan's references
and topology, not a rewritten series or experimental results.
