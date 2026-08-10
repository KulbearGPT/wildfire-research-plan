# Res18-U-Net T=1 fold-2 reproduction evidence

Status: **PASS for the 500-step timing calibration, the single full Fold-2
official-code-path run, and the released Fold-2 weight test.** The calibration
did not invoke test or predict. The later full run reached exactly 10,000
optimizer steps and the separate released-weight run invoked only official
`Trainer.test`. These are single-fold observations, not a reproduction of the
paper's 12-fold aggregate `0.460 +/- 0.084`; the official 236-versus-608.465
positive-weight discrepancy below also remains unresolved as to paper intent.

## Completed full Fold-2 run

The only full training launch used controller PID 24216 and scientific PID
22944. It ran for 18,759.618905067 s (5.211005251 h), stopped at exactly 10,000
optimizer steps, exited 0, selected
`best-epoch=79-val_avg_precision=0.33.ckpt`, and then completed the unchanged
official 3,337-batch test loader. The checkpoint is 173,524,843 bytes with
SHA-256
`7c2fa769e3fa66c8833b8d94b339e6d205abad96f25249cadf335bf3474e076c`.
The test metrics were:

| Metric | Full trained Fold 2 | Released Fold-2 weight |
|---|---:|---:|
| AP | 0.5546640158 | 0.5709022880 |
| F1 | 0.4988675117 | 0.4334018826 |
| IoU | 0.3323274553 | 0.2766516209 |
| Precision | 0.7076455951 | 0.7646416426 |
| Recall | 0.3852163851 | 0.3024023771 |
| Loss | 0.0064315894 | 0.0059189899 |

The full run's AP is 0.0162382722 below the independently recomputed released
weight AP. This difference is descriptive for two Fold-2 states only. Neither
number should be compared as though it were the paper's mean over 12 folds.

The full run recorded 3,560 GPU samples. Their mean/median interval was
5.268894689/3.007272100 s (effective 0.189793127 Hz), median utilization was 6%,
and the sampled total-GPU peak was 15,765 MiB. That peak is only the maximum at
the observed cadence and may miss a transient. PyTorch reported peak allocated
memory of 12,167,387,136 bytes; WDDM child attribution was unavailable rather
than zero.

At the scheduled 1.5-hour checkpoint the same two PIDs were healthy at
approximately step 2,904, epoch 24 batch 0, with a then-best epoch-22 AP-0.29
checkpoint. The atomic checkpoint explicitly recorded continue-not-truncate;
the process was not restarted or replaced.

The scientific child produced complete raw evidence and exited 0, but the
controller initially failed after exit because its table parser did not accept
Lightning's Windows borderless test-result rendering. The run retains that
`failure.json`. A RED/GREEN parser fix and no-launch `--finalize-existing`
reconstructed the result from the same raw files, wrote an observer-recovery
lineage, and made `completed.json` governing. No second full child was started.
The independent verifier rebuilt exact steps, metrics, command, checkpoint,
data inventory, effective positive weight, and runtime-patch provenance and
returned PASS.

## Completed official released-weight test

After the full run and its independent verification passed, exactly one
released-weight controller created run
`fold2-weight-20260810T133336Z-2e071197`. Scientific PID 37668 loaded all 182
state-dict tensors with `strict=True`, invoked only `Trainer.test`, completed
3,337/3,337 batches, and exited 0. Train, validation, predict, resume, and retry
were not invoked. The measured AP was `0.5709022879600525`, an absolute
difference of `0.0000977120399474618` from filename value `0.571`.

This test took 560.127523899 s. Its 561 GPU samples had mean/median intervals
0.999990586/0.999306600 s (effective 1.000009414 Hz), median utilization 50%,
and sampled total-GPU peak 15,525 MiB. PyTorch peak allocated memory was
12,050,196,992 bytes; WDDM child attribution was unavailable. Official test
code emitted `test_pr_curve_data.npz` in the derived runtime working directory;
the 1,976-byte output was preserved unchanged in the run directory with
SHA-256
`08c47d22aa87102c88104e42b5c7cc37c08e6481b09d573a5726d0b73f8dc785`
and an explicit provenance marker, restoring the derived checkout to exactly
the authorized import-scope patch.

The independent released-weight verifier initially exposed the same offline
Windows table-parser gap. A focused RED/GREEN fix added the strict borderless
table form without importing the controller parser. Re-running only the
verifier returned PASS and independently confirmed the exact command hash,
strict-load sentinel, six metrics, pinned weight hash, full-run dependency,
607-file live source inventory, original checkout, derived patch, and copied
provenance. No evaluation child was relaunched.

## Frozen experiment

The child used official WSTS code commit
`ed221d491fe2142a4b2e93462c2c0b7a1c7c31ad`, released-weight revision
`acf70a37394849f4ec8d108a51d6f4325a554d0a`, fold 2 (train 2018/2020,
validation 2019, test metadata 2021), Res18-U-Net, `T=1`, all 40 features,
batch 64, crop 128, FP32, seed 0, focal loss, and AdamW at `0.001`. The source
YAML contains/comment-documents `pos_class_weight: 236`, but unchanged official
`train.py.before_instantiate_classes` recomputed `1 / fire_rate` for fold 2 and
overwrote it before model construction. The successful saved `config.yaml`
therefore records the actual effective dynamic value
`608.4653828020165`. It stopped at exactly 500 optimizer steps.
`do_test=false`; neither `do_predict` nor `do_validate` was passed.

The runtime was Python 3.10.4, setuptools 80.9.0, NumPy 1.23.5, PyTorch
2.0.0+cu118, torchvision 0.15.1+cu118, and PyTorch Lightning 2.0.1 in
`D:\WildFire Project\.conda-envs\wsts-res18-t1`. Hardware was one NVIDIA
GeForce RTX 3090 (24,576 MiB), driver 610.74. The final Windows-compatible
loader setting was `num_workers=8`; no batch, model, loss, optimizer, data, or
precision setting changed.

The original upstream checkout remained clean. A derived checkout at the same
commit carried one runtime-safety import-scope patch, SHA-256
`e7b0211e762cb7a888b0ce699e2d22537b372a49cfc0baf52e078513fbdc0c83`.
It deletes only seven unused eager architecture exports from
`src/models/__init__.py`, retains the four imports required by `train.py`, and
touches no scientific class/module body.

The 236-versus-608.4653828020165 difference is an official config/code
discrepancy, not an observer modification. It does not invalidate this timing
measurement because the child followed the official runtime path. Before a
full paper reproduction or AP comparison, however, the intended paper setting
must be explicitly resolved; no 500-step validation metric here is treated as
a paper result.

## Transparent launch lineage

The runner never retried automatically. Every launch has a distinct atomic
global and run lock.

1. Launch 1, PID 44068, exited 1 before model, datamodule, Trainer, optimizer,
   or progress because the native initializer raised `ModuleNotFoundError: No
   module named 'src'`. Raw stdout was empty.
2. Launch 2, PID 8872, exited 1 at `Sanity Checking: 0it`. Windows workers 64
   raised `Caught RuntimeError in DataLoader worker process 0` and
   `DefaultCPUAllocator: not enough memory`. It had zero optimizer steps, no
   max-step stop, and no peak-allocation sentinel.
3. Launch 3, PID 14672, used workers 8, reached exact global step 500, emitted
   the peak-allocation sentinel, showed the exact Lightning
   `max_steps=500 reached` stop, and exited 0. No fourth launch occurred.

Before launch 3, a no-training workers-8 smoke loaded one exact train batch and
one exact validation batch. Both boundaries were
`[64,1,40,128,128] -> [64,128,128]`, finite, binary, and next-day distinct.
It never called the test loader and never started training. RAM sampling found
68,564,467,712 total bytes, 49,022,062,592 available before,
44,821,446,656 minimum available, 48,365,293,568 available after, and
23,743,021,056 peak used bytes across 261 samples.

The successful child was followed by an observer-only postprocessing failure:
the original parser interpreted Lightning's same-epoch `0/121` tqdm teardown
row after a completed epoch as optimizer regression. The raw child evidence was
complete and exit code was already 0. A TDD fix ignores only that exact
completed-epoch teardown case; ordinary progress regressions remain errors.
Offline `--finalize-existing` then verified and summarized the same artifacts
without launching a child. The run retains `failure.json` as observer-failure
evidence; `completed.json` is the governing status after this authorized
offline recovery and explicitly says the failure was not a child/training
failure.

## Measurements

The 500-step child wall time was 1,024.878947258 s (17.081316 min), measured
from the atomic child-start UTC marker to the exit-code marker mtime. Progress
ran from 191.376329200 s through 1,015.341621200 s, an observed interval of
823.965292000 s. Startup to the first progress row was 191.376329200 s.
Complete, non-sanity validation progress intervals totaled 45.907097400 s.

After excluding steps 0--49, 451 step intervals had median
0.091461900 s, p25 0.087502700 s, and p75 0.124109150 s. This instantaneous
median corresponds to 699.744921 samples/s, but it omits recurring data-loader
stalls and validation. The end-to-end observed throughput was only
31.223200 samples/s (`500 * 64 / wall_seconds`). Median GPU utilization was 6%
(min 1%, max 100%), consistent with a pipeline-bound run rather than continuous
GPU compute.

PyTorch reported peak allocated memory of 1,826,138,624 bytes
(1,741.541504 MiB). Across 877 `nvidia-smi` samples spanning 1,023.770009 s,
the actual cadence was mean interval 1.168687225 s, median interval
1.164594750 s, and effective rate 0.855660932 Hz. Peak total GPU used memory
observed at that cadence was 6,309 MiB; this sampled maximum may miss a
between-sample transient. Windows WDDM did not expose reliable
per-child memory attribution, so `peak_child_process_mib` is `null` with
`child_process_memory_available=false`; it must not be interpreted as zero GPU
memory.

## 10,000-step interpretation

The instantaneous compute-only product is
`10,000 * 0.091461900 = 914.618999996 s` (15.243650 min). It is an optimistic
empirical compute-only extrapolation/reference, not a realistic total or a
mathematical lower bound.

The raw first-completed-step boundaries were global steps 1, 122, 243, 364,
and 485. Their four full 121-step cycle durations were 202.148682600,
204.319298800, 204.324568600, and 203.342022400 s. These cycles include the
recurring loader/validation/next-epoch gaps that the instantaneous median
misses. With median cycle 203.830660600 s (p25 203.043687450, p75
204.320616250), the epoch-aware projection is:

`startup + 82 * full_epoch_cycle + 78 * instantaneous_step`

This gives 16,912.624526600 s (4.697951 h), with a percentile-combined range
of 16,847.783910700--16,955.347375400 s (4.679940--4.709819 h). A deliberately
conservative empirical projection that linearly scales the complete 500-step
wall time is 20,497.578945160 s (5.693772 h). For 12 fold/runs, those figures
imply about 56.375 h epoch-aware (56.159--56.518 h) or 68.325 h wall-linear,
assuming similar host and loader behavior.

The paper table's approximate `0.4 h` is therefore not directly reproduced by
this Windows end-to-end calibration: the epoch-aware estimate is about 11.74x
and wall-linear estimate about 14.23x larger. The paper does not document the
hardware or an equivalent timing boundary, so this comparison is contextual,
not evidence that either result is wrong.

## Evidence and independent verification

The successful ignored run is
`artifacts/reproductions/wsts-res18-t1/fold2-calibration-20260809T221432Z-9329670d`.
It contains raw stdout/stderr, timestamped events, observed-cadence GPU samples, atomic
locks and PID, exit code, source inventories, configs/provenance, one-row
`calibration.csv`, 501-row `step-timing.csv`, `timing.json`, and the recovered
completion markers.

A separate standard-library process (`verify_calibration.py`) imports none of
the runner. From raw files it independently reconstructed all 501 progress
points (steps 0--500), epoch boundaries/cycles, wall/startup/validation times,
instantaneous and end-to-end throughput, the three projections, GPU statistics,
and peak sentinel. It exactly matched `timing.json`, confirmed exit 0 and no
test/predict/OOM, and validated the two source snapshots internally: exact
data root, 607 unique direct entries with year counts 176/74/201/156, matching
entry and byte totals, and valid integer size/mtime metadata. It independently
statted the live selected-year tree and exactly matched all post-run paths,
sizes, and nanosecond mtimes (24,242,259,023 bytes total). It also confirmed the
original checkout clean at the pinned commit and the derived patch status and
hash. Its result is
`independent-verification.json` in the successful run directory.
It also independently read `config.yaml`, required effective
`pos_class_weight=608.4653828020165`, and matched the recorded provenance; the
effective config SHA-256 is
`743e3d02b784f22a390a208905d3c60067db20fa2aadf5ae346c2774f97458fc`.

The verifier constructs the full ordered child command independently: exact
environment Python and entrypoint, exact derived root and three absolute config
paths, exact data root/fold/All/T=1/deduplication/workers-8/max-500/local-root
arguments, and `do_test=false`, with no extra argument. It exact-compares the
list and cross-checks SHA-256
`1302459851e26669ca61a6c9ea551797f8421db4bf95c35059ecee3ee2b7b75a`
across the start marker, global/run locks, worker authorization, effective
command, and timing summary. Every marker must contain its schema-specific
command and/or hash field with the correct JSON type; a missing field fails
closed.

For code provenance, it independently reads both initializers, derives the
expected runtime content by deleting the seven authorized unused exports, and
requires byte/content equality with no other derived file changed. It matches
the tracked patch, copied patch, actual git diff, copied diff, patch SHA-256,
and diff SHA-256. It parses the pinned `train.py` AST to establish that
`before_instantiate_classes` computes `float(1 / fire_rate)` and assigns it to
the model config, then combines that observation with source YAML 236 and saved
effective config 608.4653828020165; the dynamic-override conclusion is derived,
not asserted as a constant.
