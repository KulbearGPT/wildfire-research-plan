# WSTS Res18-U-Net T=1 fold-2 timing calibration

## Full Fold-2 reproduction

`run_full_fold.py` is the separate, single-use controller for the authorized
scientific run. It keeps the same pinned code/data/model configuration and the
audited Windows workers-8 compatibility setting, changes the calibration stop
to the official sweep value of 10,000 optimizer steps, and restores
`do_test=true`. The unchanged official `train.py` tests the validation-AP-best
checkpoint after fitting. The controller has a distinct global atomic lock and
never retries.

Before launch, run:

```powershell
python reproductions/wsts_res18_unet_t1/scripts/run_full_fold.py --preflight-only
```

The launch action is:

```powershell
python reproductions/wsts_res18_unet_t1/scripts/run_full_fold.py --launch
```

The 1.5-hour allocation is an observation checkpoint, not a stop condition.
The preserved 500-step calibration predicts roughly 4.68--5.69 hours for the
complete run on this Windows host. A partial checkpoint must not be reported as
a complete reproduction.

The sole completed full launch reached exact optimizer step 10,000 in
18,759.618905067 s (5.211005251 h), then tested the epoch-79 / validation-AP
filename-label-0.33 best checkpoint. The label is rounded: checkpoint metadata
records global step 9,680 and the raw progress display reports `0.326` at three
decimal places. Fold-2 test AP was `0.5546640157699585`; checkpoint
SHA-256 was
`7c2fa769e3fa66c8833b8d94b339e6d205abad96f25249cadf335bf3474e076c`.
An observer-only Windows table-parser failure occurred after the scientific
child had exited 0. Its evidence remains in the run, and a TDD-fixed,
no-launch `--finalize-existing` path recovered the result by reading the
preserved raw artifact set. Independent verification passed; there was no
second full launch.
It reconstructs exact step 10,000 from raw events and wall/GPU statistics from
raw observer files. A later no-launch seal found identical before/after hashes
for the fixed 14-entry raw set; this current-state check is not presented as
retrospective proof about the first parser recovery.

## Released Fold-2 weight

The pinned Hub revision contains the raw state dictionary
`trained_model_weights/Res18Unet_T1/All/fold2_testAP0.571.pth`. Fetching is
separate from evaluation and verifies 57,889,221 bytes plus SHA-256
`e17cd58e29ee7b91f6a8ba85ddcb5783ec69b9541e2de93298ba3241e785a9ec`:

```powershell
python reproductions/wsts_res18_unet_t1/scripts/evaluate_released_weight.py --fetch-only
```

The preflight instantiates the official model and applies the raw state dict
with `strict=True`, but does not call any loader:

```powershell
python reproductions/wsts_res18_unet_t1/scripts/evaluate_released_weight.py --preflight-only
```

The launch action is gated on a completed full Fold-2 run. It invokes only
`Trainer.test`; train, validation, prediction, and Lightning checkpoint-resume
paths are forbidden:

```powershell
python reproductions/wsts_res18_unet_t1/scripts/evaluate_released_weight.py --launch
```

The sole completed release evaluation strict-loaded 182 tensors and finished
all 3,337 official test batches without train, validation, predict, resume, or
retry. Recomputed Fold-2 AP was `0.5709022879600525` (absolute difference
`0.0000977120399474618` from filename `0.571`) in 560.127523899 s. Its
independent verifier passed after a TDD-only fix for the Windows borderless
Lightning result table; no scientific child was relaunched.

Across the pinned official folds 0--11, the 12 filename AP labels have mean
`0.45291666666666663` and population standard deviation
`0.08827179460179917`. This is derived only from the filenames in the pinned
`Res18Unet_T1/All` manifest; it is not paper-table provenance and is not a
recomputation of 12 fold metrics. Fetch, future preflight/result summaries, and
the independent verifier carry the exact filename manifest and this boundary.
For the preserved historical launch, a separate offline augmentation records
the manifest without rewriting launch-time `preflight.json`.

The weight verifier independently recomputes wall time and every GPU summary
from raw markers/CSV, requires six finite legal metrics, and confirms the fixed
14-entry raw manifest was unchanged during the verifier process. Its first
Windows-table parser failure remains documented separately; the scientific
child had already exited 0 and was never relaunched.

This directory controls a provenance-checked timing calibration of the authors'
released Res18-U-Net, `T=1`, All-features configuration on official WSTS fold 2.
The 500-step timing run is **not a scientific reproduction result**: it does not
invoke the test loader and cannot establish the paper target `0.460 +/- 0.084`.

The official baseline command, run only from the pinned upstream checkout and
its isolated environment, is:

```powershell
python src/train.py
```

Before any run, validate the data inventory and exact non-scientific calibration
overrides with `scripts/control.py`. Keep the upstream checkout unmodified and
store all local clones, runs, logs, and learned artifacts under ignored paths.

## Isolated bootstrap and real fold-2 smoke gate

The bootstrap script always targets the dedicated prefix
`D:\WildFire Project\.conda-envs\wsts-res18-t1`; it never activates an
environment or invokes the current Python. The prefix is not configurable.
Before the first pip command it rejects the target if it is active or is the
Conda base prefix, then requires the target `python.exe` to report that same
resolved `sys.prefix` and exactly Python 3.10.4. It checks out the commit from
`upstream.lock.json` in detached mode, rejects local upstream changes and a
non-official `origin` URL, installs
the authors' pinned requirements with the CUDA 11.8 PyTorch 2.0.0 wheels, pins
`setuptools==80.9.0` because Lightning Fabric 2.0.1 still imports the removed
`pkg_resources` compatibility module, and applies the bounded Windows-only
`numpy==1.23.5` compatibility override. The latter is required because the
PyTorch 2.0.0+cu118 Windows wheel uses NumPy C API `0x10`, while the authors'
`numpy==1.22.3` pin exposes `0x0f`; bootstrap verifies both `torch.from_numpy`
and `tensor.numpy()` before continuing. This override does not change official
source, data, model, or optimization settings, and the smoke does not produce
an AP result. Bootstrap also caches the ResNet-18 ImageNet encoder weights.
Supply an ignored run directory
so the complete install transcript and environment/hardware inventories stay
with the gate evidence:

```powershell
$run = 'artifacts/reproductions/wsts-res18-t1/fold2-smoke-001'
powershell -ExecutionPolicy Bypass -File reproductions/wsts_res18_unet_t1/scripts/bootstrap.ps1 -RunDirectory $run
```

The isolation and checkout guards can be exercised without installing,
fetching, or changing either location:

```powershell
powershell -ExecutionPolicy Bypass -File reproductions/wsts_res18_unet_t1/scripts/bootstrap.ps1 -ValidateEnvironmentOnly
powershell -ExecutionPolicy Bypass -File reproductions/wsts_res18_unet_t1/scripts/bootstrap.ps1 -ValidateCheckoutOnly
powershell -ExecutionPolicy Bypass -File reproductions/wsts_res18_unet_t1/scripts/bootstrap.ps1 -ValidateRuntimeOnly
```

Then run exactly one seeded training-loader batch in the isolated prefix:

```powershell
conda run --no-capture-output --prefix 'D:\WildFire Project\.conda-envs\wsts-res18-t1' python reproductions/wsts_res18_unet_t1/scripts/smoke_fold2.py `
  --upstream-root reproductions/wsts_res18_unet_t1/.local/WildfireSpreadTS `
  --data-root 'D:\WildFire Project\data\hdf5' `
  --output-dir $run `
  --num-workers 64
```

On native Windows, reduce workers only after the immediately preceding attempt
fails specifically in worker creation or IPC, using `64 -> 8 -> 4 -> 0` and
stopping at the first passing value. Preserve every attempt's complete log.
The smoke calls only `setup('fit')` and `train_dataloader()`. Upstream creates
2021 test-dataset metadata during fit setup, but this gate never calls the test
loader and never loads a 2021 sample. A passing attempt atomically writes
`smoke.json`; it does not start training and does not support a 500-step AP
claim.

## Windows train-and-validation smoke and calibration runner

The original workers-64 smoke remains the baseline evidence. The full trainer
later showed that native Windows could exhaust CPU allocator memory when 64
validation workers were spawned during Lightning sanity checking. Before the
explicit worker recovery, run a second no-training smoke that loads exactly one
train batch and one validation batch at workers 8 and records RAM samples:

```powershell
& 'D:\WildFire Project\.conda-envs\wsts-res18-t1\python.exe' `
  reproductions/wsts_res18_unet_t1/scripts/smoke_fold2.py `
  --upstream-root reproductions/wsts_res18_unet_t1/.local/WildfireSpreadTS `
  --data-root 'D:\WildFire Project\data\hdf5' `
  --output-dir artifacts/reproductions/wsts-res18-t1/fold2-train-val-smoke-workers8-20260809 `
  --num-workers 8 --include-validation
```

`run_calibration.py` is an external observer. It validates environment, GPU,
data, original checkout, derived runtime patch, configs, import-only help, and
smoke evidence before it can create an atomic launch lock. It passes
`do_test=false`, never passes `do_predict` or `do_validate`, timestamps raw
stdout/stderr, and samples `nvidia-smi`; it does not change the model,
datamodule, loss, optimizer, callbacks, Trainer, or metrics.

The pinned upstream package initializer eagerly imports seven architectures
that the Res18 training path does not use, including mixed-package and missing
imports. The tracked `patches/res18_import_scope.patch` only removes those seven
exports from an ignored derived checkout. The original pinned checkout remains
clean, the four exports required by `train.py` remain unchanged, and no class or
module body is patched.

The 2026-08-09 launch lineage is deliberately immutable:

- launch 1 stopped on the native import defect before Trainer or optimization;
- launch 2 used the exact import-scope patch but workers 64 exhausted Windows
  CPU allocator memory during validation sanity checking, before optimization;
- launch 3 changed only `data.num_workers=64` to `8` (besides its unique output
  directory), reached exactly optimizer step 500, and exited 0.

There is no automatic retry path and launch 3 is final. An observer parser
initially treated Lightning's completed-epoch tqdm teardown reset as optimizer
regression after the successful child exited. `--finalize-existing` performs
only offline validation and summary generation for that preserved run; it
cannot launch a child. The run therefore retains `failure.json` as evidence of
the observer-only postprocessing failure, while `completed.json` is the
governing recovered status and explicitly classifies the distinction.

`calibration.csv` contains one summary row. Per-step details are in
`step-timing.csv`. `timing.json` distinguishes the optimistic empirical
compute-only extrapolation/reference, an epoch-aware estimate that includes recurring loader and
validation cycles, and a conservative wall-linear estimate. On Windows WDDM,
per-child memory attribution may be unavailable; in that case it is recorded as
JSON `null`, not zero, while total GPU memory and PyTorch allocated memory remain
reported.

The source Res18 YAML contains `pos_class_weight: 236`, but this is not the
effective runtime value. The official, unchanged
`train.py.before_instantiate_classes` recomputes `1 / fire_rate` for the selected
fold and overwrites the YAML value before model construction. For fold 2 the
saved effective `config.yaml` records `608.4653828020165`. The finalizer and
independent verifier both require that exact runtime value. This official
config/code discrepancy does not invalidate the timing observation, but the
intended positive-class setting must be resolved before claiming a full paper
reproduction.

Future GPU observation uses absolute monotonic one-second deadlines, rather
than waiting one second after each `nvidia-smi` query. Every result reports its
actual cadence from `observer_seconds`. The existing successful run contains
877 samples over 1,023.770009 s: mean interval 1.168687225 s, median interval
1.164594750 s, and effective rate 0.855660932 Hz. Its 6,309 MiB total-GPU peak
is the maximum at that observed cadence and may miss a between-sample transient;
it is not described as an exact target-cadence peak.

Independent verification constructs the entire ordered attempt-3 command and
requires list equality with no extra arguments. It cross-checks the command
hash across `started.json`, global/run locks, worker authorization,
`effective-command.json`, and `timing.json`. Each marker must contain its
schema-specific command and/or hash field with the expected JSON type; missing
fields fail verification. It also derives the expected
runtime initializer by deleting exactly seven authorized exports from pristine
content, requires every other derived file clean, matches the tracked/copied
patch and copied git diff, and derives the positive-weight override from the
pinned `train.py` AST rather than hard-coding the conclusion.

Source-data verification requires the exact data root, 607 unique direct
`year/file.hdf5` entries, per-year counts `176/74/201/156`, internally
consistent entry/byte totals, and integer size/mtime metadata. The independent
verifier also stats the live selected-year tree and exact-compares its paths,
sizes, and nanosecond mtimes with the post-run snapshot; equal pre/post JSON
alone is insufficient.
