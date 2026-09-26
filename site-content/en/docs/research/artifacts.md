# Artifacts for the X22 + X17 route

The active handoff needs corrected B3/B5 initialization, the X22+X17 components,
and their matched controls. Keep weights, data and large logs outside Git. This
page describes what to retain and transfer; it does not establish that the
historical files are currently accessible. Use the [active recipe](method-recipes.md)
and [code lifecycle](../CODE_LIFECYCLE.md) for execution and claim limits.

## Required files and roles

| Asset | What to preserve | Role in the comparison |
|---|---|---|
| Corrected data and statistics | Dataset identity, repair/audit records, checksums and training-only normalization statistics | Same event/target-date population for every paired result; see [data preparation](data-preparation.md) |
| B3 at T1 and B5 at T5 | Original Lightning checkpoint, its SHA256 and available completion/training records | Shared initialization within each setting and separate frozen-reference evaluation |
| Fresh ERM | `control` continuation checkpoint and results | Same-budget comparator; supplies M00 to the complete route |
| X22 | `cosine_erm` continuation checkpoint and results | Cosine-schedule component; supplies M01 |
| Mixed expert / X14 | `block_specialist` without a fixed block fraction, checkpoint and results | Nearest attribution control for severity decomposition |
| Mild expert / X17 | `block_specialist --block-fraction 0.25`, checkpoint and results | Supplies M06 |
| Severe expert / X17 | `block_specialist --block-fraction 0.5`, checkpoint and results | Supplies M07 |

Retain a matched set of all five continuation roles for each required history
and seed. Full recorded evidence covers histories 1/5, seeds 0/1/2 and years
2021/2022/2023. Each trained role has a `checkpoint.pt`; each evaluated year has
its own `summary.json` and `M00.json`, `M01.json`, `M06.json`, `M07.json`. The
2022/2023 evaluations reuse the frozen trained checkpoint. Preserve `started.json`
and the associated job records as well. Smoke outputs qualify execution only.

The canonical settings are T1 Res18-U-Net with 40 features and T5 Res18-UTAE
with 33 features/day. Continuations share the relevant B3/B5 checkpoint, seed,
3000 optimizer updates, physical/effective batch 64, data, normalization and
final-step checkpoint selection. B3/B5 bootstrap selection follows its own
[baseline contract](baselines.md); a baseline's selected checkpoint need not be
its final training step. Do not replace a fresh ERM control with frozen B3/B5.

## Preserve source and comparison provenance

A result summary contains scenario metrics and identifiers; it is not the
complete experiment record. Keep a transfer inventory recording each file's
role, history/seed/year where applicable, original location, destination, byte
count and SHA256. Record missing assets explicitly. This inventory is a handoff
record, not an input format accepted by the current runner.

For every matched set, preserve:

- The exact source commit or source archive, executed command, configuration,
  upstream revision/patch and environment record. The submitter saves
  `source-commit.txt`, `command.txt`, `site-at-submission.env`, `job-id.txt` and
  `submission-status.txt` beside the archived source under `$WILDFIRE_ROOT/jobs/`.
  Retain the relevant allocation logs and setup `pip freeze` records.
- B3/B5 file identity and the relationship between that source checkpoint and
  each continuation. Verify initialization, physical batch, optimizer budget,
  schedule, corruption policy and checkpoint selection from the source/config
  and checkpoint training metadata, rather than inferring them from a directory
  name. `started.json` records the invocation; an evaluate-only invocation does
  not certify the original training settings.
- Dataset/normalization identity and matched evaluation populations. The
  recorded 2021/2022/2023 counts are 3181/2856/2102 in both settings; see the
  [population audit](evaluation-population.md). Historical reporting years are
  already exposed and are not untouched confirmation data for a new method.
- Per-role results and the generated comparison reports. Keep X22 versus fresh
  ERM, X17 versus mixed X14, and the complete route versus fresh ERM distinct.
  Preserve frozen B3/B5 evaluation results for the separate reference comparison.

Use `compare.py`, `compose_severity_routes.py` and `compose_complete_routes.py`
under `reproductions/cross_history/` as described in the [active recipe](method-recipes.md).
Their checks do not certify the complete artifact inventory, shared initialization,
batch, data identity or source provenance. Check exactly one matched row per
history/seed/year and align repeated arguments. For the active complete route,
verify `component_methods.control` is `control` and `component_methods.fire` is
`cosine_erm`. The composer assembles scenario scores; it does not export a fused
model. Preserve failed X17 attribution cells and the distinction between gains
over fresh ERM and frozen B3/B5.

## Transfer or regenerate

1. Locate the source files and make an inventory before copying. A historical
   absolute path or committed manifest is provenance, not proof of availability.
2. Copy the required checkpoints, records, result directories and provenance into
   a new external artifact directory. Compute and verify SHA256 on both sides;
   perform weight hashing, bulk copying and model validation in Slurm allocations.
3. Set `WILDFIRE_B3_CHECKPOINT` and `WILDFIRE_B5_CHECKPOINT` to the transferred
   Lightning baseline files in the site's configuration. Use the relocated
   continuation checkpoint path with `--evaluate-only` when evaluation is needed.
   Follow [baseline record relocation](baselines.md) for a completion record's
   runtime `checkpoint` field; preserve the scientific provenance fields and
   original initialization identities in historical checkpoints.
4. Confirm file integrity, matching roles and settings, result completeness and
   the exact submitted source before any authorized reproduction. The submitter
   archives committed HEAD, so working-tree changes alone do not enter a job.
   If originals are unavailable, use the [baseline](baselines.md) and
   [active continuation](method-recipes.md) recipes within the agreed budget.
   Regenerated artifacts have new checksums and provenance; they do not acquire
   the historical identities or establish the old measured results automatically.

## Current tooling limits and archive

`reproductions.artifact_bundle` supports only the `reliability` and `routed`
teacher-manifest formats from the archived RF and TD campaigns. It does not
provide a complete X22+X17 bundle, export B3/B5 initialization, collect the five
matched roles with all yearly evaluations, or verify the whole comparison
contract. No active-route manifest schema or automatic bundler is provided here.

The original verified teacher-copy commands and format-specific relocation rules
are preserved in [archived teacher artifact transfer](../archive/teacher-artifacts.md).
They are useful for deliberate recovery of those campaigns and are not part of
the active route's prerequisites.
