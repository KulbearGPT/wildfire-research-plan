# Archived TD exploration

This package is inactive under the selected X22+X17 mainline. It preserves the
separate four-bank normalization (TD-N), weight merge (TD-W), and routed-expert
distillation (TD-D) campaign. It is not the RF implementation in `cross_history/`.

Keep original code, settings, source identities and result limitations for
reproduction. See [three-directions-results.md](../../docs/experiments/three-directions-results.md)
and the TD entries in [method-inventory.json](../../docs/research/method-inventory.json).
Runtime qualification does not establish a successful research contribution.
Current active code and comparisons are listed in [CODE_LIFECYCLE.md](../../docs/CODE_LIFECYCLE.md).

## Original diagnostic and execution helpers

The local branch integration on 2026-09-29 recovered these files from
`3761e8b40896e63c1f1a63f6c88f316ebd1edc99`, with lifecycle comments only:

- `audit_checkpoints.py`: the original completed student-checkpoint checks,
  including strict loading and source/metric identity assertions.
- `audit_statistics.py`: the original four-bank normalization diagnostic,
  including its fixed historical jobs, source manifest and sample indices.
- `run_slurm.sh`: the original Nibi runner and six-mode smoke suite, with its
  original paths, modules and environment.

These are historical helpers, not portable mainline entrypoints. The Python
diagnostics import Torch at module import time: invoke them only inside Slurm,
including `--help`. The statistics helper and shell runner retain site-specific
paths; this merge does not certify their present artifact availability or rerun
their numerical behavior. Use the configured research submitter for current work.
The distinct [TD implementation plan](../../docs/archive/three-directions-td-plan-20260912.md)
is preserved separately from the similarly named RF plan.
