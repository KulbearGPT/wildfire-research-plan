# Res18-U-Net T=1 Fold-2 500-Step Calibration Design

## Purpose

Measure the end-to-end and steady-state training time of the authors' released
Res18-U-Net, `T=1`, `All`-features configuration on the local RTX 3090. This is
an execution calibration for the later paper-native 12-fold reproduction, not a
scientific result and not a run under this project's frozen 2016--2023 split.

The calibration stops after 500 optimizer steps on official WSTS fold 2. It
must not evaluate, tune on, or otherwise read samples from the fold-2 test
loader.

## Frozen Scientific Target

- Paper: *Improved Wildfire Spread Prediction with Time-Series Data and the
  WSTS+ Benchmark*, WACV 2026, arXiv v3.
- Paper result to reproduce later: Res18-U-Net, `T=1`, `All`, twelve-fold WSTS
  leave-one-year-out test AP `0.460 +/- 0.084`.
- This paper table uses the original WSTS 2018--2021 benchmark with 607 events;
  it does not use the project's 999-event 2016--2023 split.
- Fold 2 is fixed by the authors' datamodule as train years `2018, 2020`,
  validation year `2019`, and test year `2021`.
- Official model configuration: ResNet-18 encoder, ImageNet encoder weights,
  40 input channels, focal loss, positive-class weight 236, AdamW with learning
  rate `0.001`, batch size 64, crop size 128, FP32, and seed 0.
- The later full run uses 10,000 optimizer steps. This calibration changes only
  that budget to 500 steps.

## Upstream Provenance

- Code URL: `https://github.com/slahrichi/WildfireSpreadTS.git`
- Code commit: `ed221d491fe2142a4b2e93462c2c0b7a1c7c31ad`
- Weight URL: `https://huggingface.co/saadlahrichi/WSTSPlus`
- Weight revision: `acf70a37394849f4ec8d108a51d6f4325a554d0a`
- The upstream checkout remains unmodified. Any necessary change is represented
  as an explicit patch or wrapper owned by this repository.

The authors' released checkpoint filenames do not trivially aggregate to the
paper's published `0.460 +/- 0.084`. That provenance discrepancy belongs to the
later checkpoint-verification gate and must not be resolved by changing this
timing calibration.

## Isolation and Repository Layout

The calibration will run from a new feature worktree based on the completed
`rule-baseline-evaluation` branch. It will not add learned-model dependencies to
the existing Python 3.13 `wildfire_phase0` package.

Tracked reproduction control files will live under:

```text
reproductions/wsts_res18_unet_t1/
  upstream.lock.json
  README.md
  patches/
  scripts/
docs/experiments/res18_unet_t1_reproduction.md
```

The pinned upstream clone, Conda environment, downloaded weights, checkpoints,
logs, and timing artifacts are local derivatives and remain ignored. They must
not be committed or served by the research-plan website.

## Environment Strategy

Use a dedicated Conda environment based on Python 3.10. Start from the authors'
pinned versions: PyTorch 2.0.0, torchvision 0.15.1, PyTorch Lightning 2.0.1,
segmentation-models-pytorch 0.3.2, and the remaining official requirements.
Record the fully resolved environment after installation.

The current Python 3.13 / PyTorch 2.11 environment must not be modified. W&B is
disabled for calibration; logs are local. ImageNet encoder weights are fetched
and cached before timing so network download time is not counted as training.

Platform-only changes are allowed when they do not alter samples, tensors,
optimization, checkpoint selection, or metrics. On native Windows, reducing
`num_workers` from the upstream value of 64 is allowed after a loader smoke test
because it changes throughput only. The selected value and reason must be
recorded and then reused for the eventual full reproduction timing estimate.

## Data Gate

The data root is `D:\WildFire Project\data\hdf5` and is read-only. Before model
construction, require:

- exactly 607 HDF5 files across 2018--2021: 176, 74, 201, and 156 respectively;
- no use of 2016, 2017, 2022, or 2023;
- the official fold-2 train/validation/test year mapping;
- one deterministic train batch whose loader-output and model-input shapes are
  recorded separately and agree with the authors' preprocessing plus the
  configured `T=1`, 40-channel, crop-128 model boundary;
- finite model inputs, binary targets, and a next-day target distinct from the
  input-day active-fire mask;
- no writes anywhere under the source data root.

Failure of any data condition stops the calibration. It is not repaired inside
the timing task.

## Calibration Procedure

1. Materialize the pinned upstream checkout in an ignored local directory and
   verify its commit.
2. Create the isolated environment and record package, CUDA, driver, GPU, CPU,
   RAM, and storage information.
3. Apply no source patch unless the unmodified code fails a captured smoke
   test. Every patch requires a minimal regression check and a written
   classification: environment compatibility, runtime safety, or scientific
   behavior. Scientific-behavior patches are outside this calibration and stop
   the run.
4. Instantiate fold 2, verify the data gate, construct the model, and cache the
   ImageNet weights without starting the timer.
5. Run exactly 500 optimizer steps with seed 0, official model/data/training
   settings, `do_test=false`, and no hyperparameter selection.
6. Record process wall time, training-loop time, median steady-state step time,
   samples per second, peak allocated GPU memory, GPU utilization samples, and
   the final finite training/validation metrics available before exit.
7. Extrapolate one-fold time as `median steady-state step time * 10,000`, then
   add separately measured startup and validation overhead. Report a range, not
   a single over-precise number.
8. Stop. Do not continue to 10,000 steps or another fold without a new explicit
   instruction.

## Allowed Configuration Differences

Only these differences from the official full experiment are allowed:

- `trainer.max_steps: 500` instead of 10,000;
- `do_test: false`;
- local absolute `data_dir`, output, and cache paths;
- W&B disabled/local logging;
- a recorded Windows-safe `num_workers` value if 64 fails the loader smoke;
- progress/logging frequency needed to measure timing.

Batch size, crop size, seed, precision, model, input feature set, fold, loss,
positive-class weight, optimizer, learning rate, augmentation, normalization,
and checkpoint monitor must not change.

## Outputs

The local calibration artifact directory must contain:

- immutable upstream and environment provenance;
- the fully resolved effective configuration and its allowlisted diff from the
  official configuration;
- stdout/stderr and process exit code;
- a machine-readable timing JSON;
- a one-row calibration CSV;
- a Markdown note explaining the observed timing and extrapolation limits.

No AP claim from 500-step training is published on the website or compared to
the rule baselines.

## Acceptance and Stop Conditions

The calibration passes only if:

- the provenance and data gates pass;
- the effective configuration has no non-allowlisted difference;
- training reaches exactly optimizer step 500 with finite loss and no OOM;
- the test loader is never invoked;
- timing and hardware evidence are complete;
- source HDF5 files are unchanged and the tracked worktree contains only the
  intended reproduction-control documentation/code.

Stop without retrying the scientific run when a data mismatch, non-finite loss,
model/data semantic discrepancy, or required scientific-code change appears.
Environment/setup failures may be repaired through tested, documented patches;
after three unsuccessful repair rounds, return to design rather than continuing
an ad hoc patch loop.

## Non-Goals

- No 12-fold training or testing.
- No evaluation of the released checkpoints.
- No comparison with the project's rule baselines.
- No use of the 2016--2017 or 2022--2023 WSTS+ extension years.
- No adaptation to the project's frozen cross-year split.
- No UTAE, controlled corruption, missingness model, gate, threshold tuning, or
  website result publication.
