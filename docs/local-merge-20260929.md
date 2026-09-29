# Local branch integration: 2026-09-29

The user requested local integration before the wider reproducibility and
commit-history cleanup. The starting main commit was `38736a2`; it remains
protected by `archive/pre-local-merge-20260929`. No original commit, branch tip,
worktree or experiment artifact was rewritten or deleted.

Integration was prepared on `integration/local-branches-20260929`. The five
previously unmerged branch tips are now ancestors of its reviewed result,
which is fast-forwarded into local main. This preserves their 18 original
commits instead of replacing them with new scientific source identities.

| Branch | Original tip | Merge commit | Resolution |
|---|---|---|---|
| `codex/belief-attention` | `f451934` | `2caeabc` | Current model differs only by archive comments; retain the intentional legacy-test deletion |
| `codex/belief-filter` | `fac140f` | `2458b61` | Same; branch test functions already exist in pre-cleanup main history |
| `codex/belief-reconstruction` | `12cdc2d` | `eea6640` | Same; do not reactivate the archived experiment |
| `research/t1-t5-innovations` | `f91d7eb` | `08add57` | Both tutorial commits have patch-equivalent main counterparts; retain the current homepage and roadmap |
| `research/three-directions` | `3761e8b` | `a83d0eb` | Keep current portable/BN fixes and existing evidence; recover three omitted historical helpers and the distinct TD plan |

The remaining original local branches were already ancestors of main. The
first four merge commits make no file-tree changes. Belief test functions were
checked against `0f8ba17^:tests/test_wsts_fast_track_belief_state.py`: all branch
functions/classes are already present there with identical parsed definitions.
Commit `0f8ba17` deliberately removed that old test file, so its removal remains.

The TD additions are `audit_checkpoints.py`, `audit_statistics.py` and
`run_slurm.sh` under `reproductions/three_directions/`. Only lifecycle comments
were added to their original code. They retain historical paths/assumptions and
must be invoked inside Slurm, including the Python modules' import/help paths.
They have not been newly qualified as portable tools. Their scope is explained
in the [TD code guide](../reproductions/three_directions/README.md).

Both campaigns used `docs/superpowers/plans/2026-09-12-three-directions.md` for
different plans. The current RF file stays intact; the complete original TD
body is preserved in [its own archive file](archive/three-directions-td-plan-20260912.md),
with a source-commit and historical-instructions banner.

## Verification

- All 222 pre-existing Python/shell/PowerShell source and test files under
  `reproductions/`, `src/`, `scripts/` and `tests/` are byte-identical to the
  protected starting main.
- All 100 scientific inventory records, including lifecycle selection, are
  unchanged. The source-file map increases from 143 to 146 solely for the
  recovered TD helpers.
- Existing experiment records, tutorial files, English website sources,
  generated guides and paper sources are byte-identical to starting main.
- Restored Python syntax trees match the original branch exactly. The original
  shell body is unchanged and passes `bash -n`; changed-document local links
  and `git diff --check` pass.
- No unresolved conflicts remain; original branch-tip ancestry is checked
  before promotion to main. Original branch worktrees are left intact.

These are Git/source-preservation checks. No tensor tests, model imports,
scientific recomputation, training or Slurm submissions were performed, and
there is no new reproduction claim. No remote push was made. The wider
[reproducibility and commit-grouping plan](repository-audit-plan-20260929.md)
remains separate; local integration does not complete those later stages.
