# Res18-U-Net T=1 Fold-2 Full Reproduction Design

## Objective

Run one complete authors-protocol Fold 2 experiment for Res18-U-Net, `T=1`,
all 40 features, then independently evaluate the authors' released Fold 2 raw
state dictionary. Report both AP values, runtime, provenance, and discrepancies
without generalizing a single fold to the paper's twelve-fold mean.

## Frozen scientific protocol

- Code: `slahrichi/WildfireSpreadTS` commit
  `ed221d491fe2142a4b2e93462c2c0b7a1c7c31ad`.
- Released weights: `saadlahrichi/WSTSPlus` revision
  `acf70a37394849f4ec8d108a51d6f4325a554d0a`.
- Fold 2: train 2018/2020, validation 2019, test 2021.
- Res18-U-Net, `T=1`, All features, batch 64, crop 128, FP32, seed 0,
  focal loss, AdamW `lr=0.001`, ImageNet encoder initialization.
- Official sweep termination: exactly 10,000 optimizer steps.
- Test the validation-AP-best checkpoint after training, exactly as `train.py`
  selects `ckpt_path="best"`.
- Preserve the official dynamic positive-class-weight behavior. For Fold 2 the
  effective value must be `608.4653828020165`; record the source-YAML value
  `236` as an unresolved official config/code discrepancy.

## Runtime-only compatibility controls

The pinned original checkout remains clean. Execution uses the already-audited
derived checkout at the same commit with only the seven unused eager exports
removed from `src/models/__init__.py`. Native Windows uses `num_workers=8`
because the preserved workers-64 attempt failed before optimization with CPU
allocator exhaustion. No model, loss, optimizer, batch, data, precision, or
metric code changes are allowed.

The 1.5-hour allocation is a monitoring checkpoint, not a scientific stop
condition: the 500-step calibration predicts 4.68--5.69 hours for 10,000
steps. The process must continue to step 10,000 unless it fails. There is one
atomic full-run launch lock and no automatic retry.

## Full-run evidence

Before launch, require the existing workers-8 train/validation smoke, live 607
file inventory, pinned environment, clean original checkout, exact derived
patch, free GPU/disk, and clean tracked worktree. Atomically preserve command,
environment allowlist, source inventory, configs, patch/diff, PID, stdout,
stderr, timestamped stream events, GPU samples, exit code, effective config,
checkpoint inventory, and pre/post source metadata.

Success requires exit code zero, exact `max_steps=10000 reached` evidence,
one peak-allocation sentinel, a unique best checkpoint, a completed test table,
finite metrics in `[0,1]`, no predict/validate action, and unchanged sources.
An independent standard-library verifier reconstructs these claims from the
raw artifacts and live provenance.

## Released-weight evaluation

The pinned Hub tree must contain exactly twelve files under
`trained_model_weights/Res18Unet_T1/All/`. Fold 2 is
`fold2_testAP0.571.pth`, 57,889,221 bytes, LFS SHA-256
`e17cd58e29ee7b91f6a8ba85ddcb5783ec69b9541e2de93298ba3241e785a9ec`.
Download that exact revision and verify size/SHA before loading.

The authors publish raw `state_dict`, not a Lightning checkpoint. A narrow
entrypoint instantiates the same official CLI/config/datamodule, loads the
state dict with `strict=True`, then calls `Trainer.test` without training,
validation, prediction, or checkpoint selection. Success requires exact load,
exit zero, finite test metrics, unchanged data/code, and an independently
verified result. Compare recomputed AP with filename AP `0.571` using an
explicit absolute difference; do not silently round it into agreement.

## Claim boundary

The trained Fold 2 AP and released Fold 2 AP are single-split results. The
paper's `0.460 +/- 0.084` is a twelve-fold aggregate and is not reproduced by
one fold. Separately report that the twelve released All/T=1 filenames imply
their own aggregate; do not assume directory labels or filenames prove the
paper table provenance.

