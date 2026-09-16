# Reproducible two-part research handoff

> Use superpowers:subagent-driven-development for bounded evidence inventory and runtime portability work, with independent review. Controller owns integration, environment qualification, Slurm execution and final handoff.

Goal: implement the user's two-part handoff of the complete existing research: positive signals (including single-seed/setting evidence) retain runnable code, results and reproduction instructions; unsuccessful variants retain implementation ideas/details and measured results. Commit both, and verify a clean checkout on a fresh configured runtime without project-code portability failures.

Scope: current main, reliability-fusion, three-directions and historical research/archives. Existing branch dates/ancestry determine whether a branch is old historical evidence, not new unmerged work. No new efficacy experiments or hypothesis tuning.

## Global constraints

- Model/tensor execution, dependency qualification and bulk data work only in Slurm allocations; login only source, small metadata, submission and Git.
- Preserve scientific contracts and raw negative results. Positive signal is not equivalent to confirmed contribution; record seed/history/year/control and limitations explicitly.
- Never call unrun, cancelled or invalid comparisons ineffective; retain their true status in the archive.
- Fresh environment must configure site paths/account/GPU/modules without source editing; support upstream pinned checkout, dependencies, corrected dataset/stats and required checkpoints with checksums or regeneration commands.
- No dependency on another local worktree, original user's absolute path, ignored helper or untracked patch. Large external data/weights stay outside Git with acquisition/regeneration instructions.
- Use existing lightweight scripts/modules, no workflow platform or unrelated refactor.

## Tasks and acceptance

- [x] 1. Evidence inventory: inspect committed experiment ledgers and available compact result JSONs across branches; write docs/research/positive-signals.md, docs/research/negative-results.md and an explicit machine-readable method/recipe/source inventory. Document each positive's nearest control, observed gain, evidence level, code availability and required artifacts; negatives include mechanism/settings/results and invalid/unrun distinctions. Audit historical branches rather than blanket merging them.
- [x] 2. Runtime portability: centralize environment-selected paths for cross_history and reliability-fusion code; preserve legacy defaults only for historical entrypoints. New documented runner requires explicit configuration and accepts relocatable checkpoint manifests. Cover missing config, path relocation and no original-worktree dependencies with targeted tests; tensor checks run in CPU Slurm. Fix actionable fresh-checkout defects without changing method math.
- [x] 3. Complete retained code/artifacts: integrate separately namespaced positive recipes or recover archived code as required by inventory; provide executable regeneration chain from public data/upstream to corrected B3/B5/stats/teachers. Preserve seed-matched initialization, optimizer and evaluation population. For external non-public artifacts provide verified transfer manifests AND regeneration, not inaccessible links as the sole path.
- [x] 4. Practical onboarding: docs/research/reproduce.md and portable setup/Slurm scripts, pinned environment inputs and example site config. Commands cover acquisition, preprocessing, baseline/bootstrap, continuation, evaluation and result comparison. README points to the two collections; negatives need not be new runtime-supported methods.
- [x] 5. Qualification: clean git export, new venv, relocated data/checkpoints, no original project import paths. Run installation/import plus real-data forward/backward/save/reload/evaluation smokes for retained distinct execution families on Slurm, and verify commands/metadata on CPU jobs. Log exact commits, jobs, runtime/data prerequisites and remaining external constraints honestly.
- [x] 6. Independent whole-delivery review: compare catalog coverage to retained code, results, recipe commands and executed qualification. Fix load-bearing findings, commit positive implementation and negative archive, and make the deliverable discoverable from main without overwriting unrelated work. Only mark goal complete once all required evidence is verified.

User approval already authorizes implementing this practical handoff. A clean isolated branch is used because several historical worktrees coexist; no additional consent round is needed for reversible preparation and commits.

## Completion evidence — 2026-09-16

The inventory contains 100 entries, including 51 inclusive positive signals. Positive code/results/recipes and negative/invalid/unrun documentation are committed. Fresh public-PyPI training and audit environments passed 143 and 117 CPU checks, respectively; four committed GPU report batches contain 91 unique passing short-budget cases. The relocated 999-event audit, synthetic actual-converter checks, two 24-sample historical diagnostic replays, and both-history route composition CLIs completed on Slurm. See [qualification evidence](../../research/qualification.md) for exact jobs, source versions, and limits.

Independent source/coverage review found no remaining load-bearing omission; a separate final report audit verified all 91 case names and previously pending families. Documentation local links resolve. Main was clean at 864f675 and fast-forwarded to fb565b5 after qualification; no unrelated files were overwritten. This closing plan record is committed afterward. No remote push performed.

Qualification used the existing physical cluster, newly created software environments and relocated existing data/weights; it does not claim a full redownload/rebuild or full-budget reproduction of every historical score. All model/data execution was on Slurm compute nodes.
