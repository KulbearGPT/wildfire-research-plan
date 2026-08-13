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

## Released official weights (folds 0 through 11)

The tracked manifest freezes all twelve raw state dictionaries at Hub revision
`acf70a37394849f4ec8d108a51d6f4325a554d0a`. Select a fold with `--fold-id`
(`0` through `11`); omitting it retains the historical Fold 2 behavior. Fetching
is separate from evaluation and verifies the selected manifest entry's exact
filename, byte count, and SHA-256:

```powershell
python reproductions/wsts_res18_unet_t1/scripts/evaluate_released_weight.py --fold-id 0 --fetch-only
```

The preflight instantiates the official model and applies the raw state dict
with `strict=True`, but does not call any loader:

```powershell
python reproductions/wsts_res18_unet_t1/scripts/evaluate_released_weight.py --fold-id 0 --preflight-only
```

The launch action is gated on the completed full Fold-2 run. It invokes only
`Trainer.test`; train, validation, prediction, and Lightning checkpoint-resume
paths are forbidden:

```powershell
python reproductions/wsts_res18_unet_t1/scripts/evaluate_released_weight.py --fold-id 0 --launch
```

Each fold has a separate cache filename, run-directory prefix, and global/run
lock. A run must be finalized with the same `--fold-id` used at launch. The
independent verifier also requires that fold ID and reconstructs the command,
official split years, filename AP, size, and SHA exclusively from the tracked
manifest. The sealed Fold 2 artifact remains independently verifiable with the
default fold or explicit `--fold-id 2`.

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

The twelve-fold campaign controller fixes the order to folds `0` through `11`.
It adopts only the sealed Fold 2 run
`fold2-weight-20260810T133336Z-2e071197`, after a fresh pass by the generic
independent fold verifier, and launches each of the other eleven folds once and
sequentially. Every fetch, launch, finalization, and independent verification
uses the existing single-fold scripts through their formal CLIs under the fixed
environment Python. The unique immutable root lock is the transaction authority
and identifies its campaign directory; every later immutable fold lock belongs
to that authority. Any qualification, download, lock, child, or verifier failure
writes campaign-owned failure provenance and stops before the next fold. Locks
are preserved fail-closed; there is no automatic retry or partial aggregation.
A future authorized campaign is launched with a new directory directly under
the fixed campaign artifact root:

```powershell
python reproductions/wsts_res18_unet_t1/scripts/run_weight_campaign.py --launch --campaign-directory artifacts/reproductions/wsts-res18-t1-official-weight-12fold/official-weight-12fold-<timestamp>
```

Preserved child output can be finalized without starting or restarting a
scientific evaluation child by pairing the original run with its exact fold.
The finalizer and independent verifier may still run provenance-checking helper
subprocesses such as Git:

```powershell
python reproductions/wsts_res18_unet_t1/scripts/run_weight_campaign.py --fold-id 5 --finalize-existing artifacts/reproductions/wsts-res18-t1-official-weight/fold5-weight-<timestamp>
```

After all twelve fold states pass, the separate campaign verifier ignores any
controller-authored metric fields. It invokes the generic verifier for each
fold, reparses the sealed raw Lightning stdout/stderr for all six metrics,
compares the two reconstructions exactly, and seals all fold-verifier files and
raw manifests across the complete verification window. It writes the ordered
per-fold CSV, aggregate JSON (population standard deviations), and independent
verification JSON inside a new immutable generation. Only the final atomic
`publication.json` marker commits that generation and its exact file hashes;
uncommitted generation files are never a successful aggregate:

```powershell
python reproductions/wsts_res18_unet_t1/scripts/verify_weight_campaign.py --campaign-directory artifacts/reproductions/wsts-res18-t1-official-weight-12fold/official-weight-12fold-<timestamp>
```

The software gate itself does not execute a campaign. A separately authorized,
committed **test-only evaluation of released weights** is published below; it is
not twelve new training runs.

## Committed twelve-fold released-weight publication

Campaign `official-weight-12fold-20260812T052555Z` committed generation
`generations/3c076108f46e4b519e65f8603b9d97a5`. The atomic publication marker
SHA-256 is `31547d503f4217d7a2654aebbb7d46f14798c9cd515beb1544e8ad8bb279a2c1`.
The CSV, summary, and independent artifact SHA-256 values are
`55f90d2d6fe4ff175d888873188d7db7e86f0f9f7026e52a782816973b78605b`,
`769412aa56b3d972422a58771fa3fdcd82640959557b8d7c5bd5a15ed7557ff5`, and
`a24a444783440b1d65ee7620e26b1476ca56d654fe70c2c12b871b6a07841542`.

| Fold | AP | F1 | IoU | Precision | Recall | Loss | AP minus filename AP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.5276636481285095 | 0.3984317481517792 | 0.24877601861953735 | 0.7460055351257324 | 0.27179765701293945 | 0.006696981843560934 | -0.0003363518714905034 |
| 1 | 0.4256492853164673 | 0.3436006009578705 | 0.20743824541568756 | 0.6813897490501404 | 0.2297201305627823 | 0.008592699654400349 | -0.00035071468353270463 |
| 2 | 0.5709022879600525 | 0.433401882648468 | 0.27665162086486816 | 0.7646416425704956 | 0.3024023771286011 | 0.005918989889323711 | -9.77120399474618e-05 |
| 3 | 0.3066331744194031 | 0.3074829876422882 | 0.18167202174663544 | 0.5418749451637268 | 0.21463923156261444 | 0.002553164027631283 | -0.00036682558059691894 |
| 4 | 0.483346164226532 | 0.4384116232395172 | 0.2807472348213196 | 0.6878276467323303 | 0.32174304127693176 | 0.008048299700021744 | 0.0003461642265319975 |
| 5 | 0.3223552405834198 | 0.31128206849098206 | 0.18433041870594025 | 0.5684785842895508 | 0.2143182009458542 | 0.002759539522230625 | 0.00035524058341979137 |
| 6 | 0.5765069723129272 | 0.49881884455680847 | 0.33228424191474915 | 0.7197784781455994 | 0.3816567361354828 | 0.005978269036859274 | -0.0004930276870727113 |
| 7 | 0.4736124575138092 | 0.3464559018611908 | 0.20952323079109192 | 0.7017239928245544 | 0.23000779747962952 | 0.005102857947349548 | -0.0003875424861907728 |
| 8 | 0.4777773916721344 | 0.4322243332862854 | 0.27569273114204407 | 0.6997684240341187 | 0.3126775920391083 | 0.00949658639729023 | -0.00022260832786558105 |
| 9 | 0.4709045886993408 | 0.3679426908493042 | 0.2254471629858017 | 0.7005905508995056 | 0.24948470294475555 | 0.005510237999260426 | -9.541130065915393e-05 |
| 10 | 0.3237844705581665 | 0.2595313787460327 | 0.14911580085754395 | 0.608970582485199 | 0.16490542888641357 | 0.0026944933924824 | -0.0002155294418335063 |
| 11 | 0.47403237223625183 | 0.4335690140724182 | 0.27678781747817993 | 0.6478978395462036 | 0.32579419016838074 | 0.005694412160664797 | 3.237223625185415e-05 |

| Metric | Mean | Population std | Min (fold) | Max (fold) |
|---|---:|---:|---:|---:|
| AP | 0.45276400446891785 | 0.08821731990844857 | 0.3066331744194031 (Fold 3) | 0.5765069723129272 (Fold 6) |
| F1 | 0.3809294228752454 | 0.06670647249234603 | 0.2595313787460327 (Fold 10) | 0.49881884455680847 (Fold 6) |
| IoU | 0.23737221211194992 | 0.0509113682276183 | 0.14911580085754395 (Fold 10) | 0.33228424191474915 (Fold 6) |
| Precision | 0.6724123309055964 | 0.06541761579107148 | 0.5418749451637268 (Fold 3) | 0.7646416425704956 (Fold 2) |
| Recall | 0.26826225717862445 | 0.059124666341402274 | 0.16490542888641357 (Fold 10) | 0.3816567361354828 (Fold 6) |
| Loss | 0.005753877630922943 | 0.0021862333356973012 | 0.002553164027631283 (Fold 3) | 0.00949658639729023 (Fold 8) |
| Runtime (s) | total 5749.570263385773 | median per fold 549.7098723649979 | — | — |

The paper reference is `0.460 +/- 0.084`, with provenance
`upstream.lock.json paper.target`. The filename reference is
`0.45291666666666663 +/- 0.08827179460179917`, with provenance official weight manifest filename labels, explicitly not paper-table provenance. GPU sampling
recorded 5,756 samples and sampled peaks of 15,525--19,976 MiB; WDDM child
attribution was unavailable and observed sample maxima may miss transients.

The preserved Fold 0 failure was resolved by an approved offline Fold 0 recovery of the existing output; no scientific evaluation child was relaunched.
The Fold 2 legacy/offline parser qualification is a provenance boundary rather
than a scientific retry. The preserved failures and one-time reviewed
continuation remain in campaign evidence.

The agreement supports released-weight executable reproducibility but does not
prove paper-table provenance identity. The official focal-alpha behavior is a
separate future training ablation and is not part of this baseline claim.

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
