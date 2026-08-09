# WSTS Res18-U-Net T=1 fold-2 timing calibration

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
