# Res18-U-Net T=1 Official Weights 12-Fold Verification Design

## Objective

Recompute the test metrics of all twelve authors-released Res18-U-Net `T=1`
All-feature weights under the official leave-one-year-out folds, then independently
aggregate the twelve measured AP values. This closes the immediate evidence gap
between the already verified Fold 2 result and the paper's reported
`0.460 +/- 0.084` twelve-fold result without retraining or changing the
scientific protocol.

## Scope and claim boundary

This campaign is a released-weight, test-only verification. It does not train,
fine-tune, validate, predict, select checkpoints, or modify weights. It does not
attempt to fix the official focal-loss alpha behavior discovered during source
review. Fixed-weight test inference does not execute the training loss, so that
issue is scientifically separate and will be handled later as an explicit
training ablation rather than silently folded into this baseline.

Three aggregates remain distinct:

1. The paper reports Res18-U-Net `T=1` All as `0.460 +/- 0.084` across twelve
   leave-one-year-out folds.
2. The AP numbers encoded in the twelve released filenames imply
   `0.45291666666666663 +/- 0.08827179460179917` using population standard
   deviation. Filenames are release metadata, not proof of paper-table
   provenance.
3. This campaign will report the mean and population standard deviation rebuilt
   from twelve raw Lightning test tables. It is the only value described as the
   locally recomputed released-weight aggregate.

No agreement or disagreement claim may be made by rounding away fold-level
differences. Per-fold absolute differences against the filename AP and aggregate
absolute differences against both reference aggregates must be reported.

## Frozen scientific protocol

- Official code: `slahrichi/WildfireSpreadTS` commit
  `ed221d491fe2142a4b2e93462c2c0b7a1c7c31ad`.
- Official weights: `saadlahrichi/WSTSPlus` revision
  `acf70a37394849f4ec8d108a51d6f4325a554d0a`.
- Model/config: Res18-U-Net, `T=1`, all 40 features, batch 64, crop 128,
  FP32, seed 0, and the official data/config/model YAML files already frozen by
  the Fold 2 reproduction.
- Folds: exactly `0..11`, using the twelve tuples hard-coded by the official
  `FireSpreadDataModule.split_fires` implementation and the unchanged 607-file
  source-data inventory totaling 24,242,259,023 bytes.
- Weight set: exactly one file for each fold under
  `trained_model_weights/Res18Unet_T1/All/`:

  ```text
  fold0_testAP0.528.pth
  fold1_testAP0.426.pth
  fold2_testAP0.571.pth
  fold3_testAP0.307.pth
  fold4_testAP0.483.pth
  fold5_testAP0.322.pth
  fold6_testAP0.577.pth
  fold7_testAP0.474.pth
  fold8_testAP0.478.pth
  fold9_testAP0.471.pth
  fold10_testAP0.324.pth
  fold11_testAP0.474.pth
  ```

Before any scientific launch, a pinned manifest must record every file's Hub
path, byte size, LFS SHA-256, fold ID, and filename AP. The manifest must reject
missing, duplicate, additional, renamed, size-mismatched, or hash-mismatched
weights. Weight downloads are content-address checked before use.

## Runtime compatibility boundary

The original official checkout must remain clean. Scientific execution uses the
already audited derived checkout at the same commit, with only the authorized
seven unused eager exports removed from `src/models/__init__.py`. Windows data
loading remains at `num_workers=8`, the setting proven by the one-batch
train/validation smoke and by the completed Fold 2 runs. No model body, forward
path, metric, transform, feature, split, batch, precision, or state-dict content
may change.

Each released weight is a raw state dictionary. The official model is
instantiated through the same three YAML files and fold-specific datamodule;
the state dictionary must load with `strict=True`, and every expected tensor
must be accounted for. The only scientific action is `Trainer.test` on that
fold's official test years. Training, sanity validation, standalone validation,
prediction, checkpoint selection, optimizer creation, and resume are forbidden.

## Campaign architecture and data flow

The implementation has four narrow responsibilities:

1. **Pinned manifest builder/verifier.** Resolves the frozen Hub revision into
   the exact twelve-weight inventory and verifies local content by size and
   SHA-256. It performs no model execution.
2. **Single-fold test-only controller.** Accepts one fold from `0..11`, freezes
   the effective official command, creates atomic campaign and fold launch
   evidence, invokes the narrow official entrypoint once, and captures raw
   stdout, stderr, timestamped stream events, GPU samples, effective config,
   source inventories, exit code, and official PR-curve output.
3. **Independent fold verifier.** Uses an implementation independent of the
   controller to reconstruct strict-loading evidence, command and provenance,
   the official test batch completion, six finite metrics, runtime, GPU sampling
   boundaries, and unchanged inputs from raw evidence.
4. **Independent campaign aggregator.** Consumes only twelve independently
   passing fold verification records, reconstructs each raw test table again,
   checks fold uniqueness and full `0..11` coverage, and writes the final table
   and aggregate. Controller-authored summary values are not trusted inputs.

Execution is sequential on the single RTX 3090. The already completed Fold 2
run may be reused rather than scientifically rerun, but only if the new campaign
verifier proves that its frozen revision, weight hash, fold mapping, command,
strict load, data inventory, raw seal, and metrics meet the same campaign
contract. Folds without qualifying evidence launch exactly once.

## Failure and retry policy

The campaign has one atomic global lock plus one immutable launch lock per fold.
Before each fold, gates require a clean tracked worktree, no conflicting Python
or GPU process, sufficient disk/GPU memory, exact environment, clean original
checkout, exact derived patch, exact source inventory, verified weight, and all
previous fold verifiers passing.

There is no automatic retry. A failed scientific process or failed independent
verification preserves its full directory and stops the campaign before the next
fold. Partial results may be inspected diagnostically but must not be published
as a twelve-fold aggregate. Any manual recovery requires a separately recorded,
reason-specific authorization and a fresh preflight; it may not overwrite or
erase the first attempt.

Parser or finalization defects discovered after an exit-zero scientific process
must be repaired by test-driven, offline `finalize-existing` processing of the
unchanged raw artifacts. They do not authorize another scientific test launch.

## Per-fold and aggregate outputs

Each fold result must contain:

- fold ID, official train/validation/test year mapping, weight path/size/SHA;
- strict-load status and loaded tensor count;
- exact official test progress total and exit code;
- `test_AP`, `test_f1`, `test_iou`, `test_precision`, `test_recall`, and
  `test_loss` reconstructed from the raw Lightning table;
- wall time, GPU sampling cadence, median utilization, sampled peak memory, and
  PyTorch peak allocated memory, with sampled peaks explicitly bounded as
  observations rather than continuous maxima;
- filename AP and exact absolute difference;
- raw-evidence manifest and independent-verification status.

The final campaign artifact must contain all twelve rows in fold order, AP mean,
population standard deviation, min/max and their fold IDs, total/median per-fold
runtime, and comparisons against both reference aggregates. CSV and JSON are
written atomically. Documentation reports observations without inferring model
quality causes from aggregate values alone.

## Testing and verification

Implementation follows test-driven development. Unit tests cover manifest
cardinality and hash fail-closed behavior, fold/filename mapping, exact command
allowlists, prohibited actions, atomic locks, reuse qualification, stop-on-first
failure, raw-table parsing including Windows carriage returns, finite/range
metric checks, fold completeness, population-standard-deviation math, and
tamper detection.

Integration tests use synthetic files and mocked processes only. No test suite
may accidentally launch training or a real scientific evaluation. Before the
campaign, focused and full repository tests, `git diff --check`, secret/path
scans, upstream cleanliness, source inventory, and the independent Fold 2
verifier must pass. After every scientific fold, its independent verifier must
pass before continuing. After all twelve folds, the campaign verifier must
recompute the result from sealed raw evidence and pass on a clean tracked tree.

## Runtime expectation and completion criteria

The measured Fold 2 released-weight test took 560.1275238990784 seconds. Eleven
remaining folds therefore imply about 102.7 minutes of raw test execution at the
same rate. Allowing weight download, preflight, finalization, and verification,
the expected remaining wall time is approximately 1.8--2.0 hours. This is an
estimate, not a stop condition.

The milestone is complete only when:

1. all twelve official weights and fold mappings are independently verified;
2. exactly one qualifying test result exists for every fold, with no unapproved
   scientific retry;
3. all twelve raw test tables and six metrics pass independent verification;
4. the measured AP mean and population standard deviation are independently
   reconstructed and compared with both reference aggregates;
5. tracked code, tests, experiment documentation, evidence boundaries, and the
   project website are updated and verified; and
6. no claim implies that released-weight verification resolves the separate
   focal-alpha training behavior or reproduces twelve new training runs.
