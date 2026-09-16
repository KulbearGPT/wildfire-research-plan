# Restored-family GPU qualification

`qualify-legacy.py` exercises actual data, archived training entrypoints, checkpoint save/strict reload and two-sample M00/M06 evaluation. Run one case per Slurm GPU allocation from the clean exported checkout and configured fresh training environment. This is an execution check, not an efficacy result.

Required environment: `WILDFIRE_ROOT`, `WILDFIRE_UPSTREAM`, `WILDFIRE_DATA`, `WILDFIRE_STATS`; the upstream checkout must have the documented import patch. Continuation cases additionally require valid original or regenerated completion records at `$WILDFIRE_ROOT/checkpoints/B3-completed.json` and/or `B5-completed.json`, with checkpoint paths resolving to the transferred files. B1/B5 bootstrap cases require cached or downloadable ImageNet encoder weights but do not need existing wildfire checkpoints.

Inside an allocated GPU job after site modules and training environment are loaded:

```bash
export PYTHONPATH="$PWD/src:$PWD:$WILDFIRE_UPSTREAM/src"
export WANDB_MODE=disabled WANDB_SILENT=true HDF5_USE_FILE_LOCKING=FALSE
export TORCH_HOME="$WILDFIRE_ROOT/cache/torch"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python scripts/research/qualify-legacy.py --case token \
  --output "$WILDFIRE_ROOT/qualification/token-$SLURM_JOB_ID"
```

For a site configured shell, an explicit allocation example is:

```bash
srun --account="$SLURM_ACCOUNT" --partition="$GPU_PARTITION" \
  --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:30:00 \
  "$WILDFIRE_ROOT/envs/train/bin/python" scripts/research/qualify-legacy.py \
  --case B5 --output "$WILDFIRE_ROOT/qualification/B5-bootstrap-check"
```

Choose a new output path every time. Available cases are `rnc`, `token`, `ciwc`, `rank`, `cra`, `ffca`, `pyramid`, `complete-pyramid`, `sarp-t1`, `standard-t5`, `sarp-t5`, `B1`, `B5`, `legacy-P00`. Use separate jobs for bounded failures and resource accounting. `rnc` verifies the combined D2/D4 implementation dependency; it does not designate RNC as a positive result.

Continuation cases temporarily set only that process's `TRAINING_STEPS=1`, use batch2/workers0 and leave formal defaults untouched. Saved checkpoints carry `status: qualification`, `scientific_claim: false` and actual one-step metadata. The driver asserts that the original formal validator rejects them. For strict loader qualification, it calls the original validator on an ephemeral metadata copy with **only** `status=pass` and `steps=3000` substituted. All architecture, variant, loss and recipe checks are unchanged; tensor weights and saved checkpoint metadata are never substituted. This limited validator adapter is local to the qualification process and is reported explicitly in `qualification.json`.

B1/B5 cases temporarily replace the baseline spec's `max_steps` with two and set batch2/workers0, validation interval one and one validation batch. The successful two-step receipt is relabeled `bootstrap-qualification.json` and removed from the normal completion-record location. Its max_steps remains two, so it cannot satisfy the formal 3,000-step completion contract. The enclosing `QUALIFICATION_ONLY.json` is written before any work, including for failed jobs. All these artifacts must remain in the qualification tree.

A passing `qualification.json` records finite real-data training loss (or bootstrap selected validation AP), exact equality of checkpoint and reloaded state tensors, and evaluation metrics for two actual 2021 examples per scenario. It does not establish numerical reproduction, full evaluation-population agreement or a scientific positive signal. Archive the job log and source/environment identity with the output.

`legacy-P00` exercises the separately documented historical 10,000-step bootstrap with a two-step qualification override. It preserves the original pooled-index behavior solely for replaying the archived diagnostics; it is not a corrected baseline. Its short receipt is likewise rejected by the formal completion contract.
