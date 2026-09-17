# Delivery qualification record

The verification covers a fresh virtual environment, an independent checkout of the official source, per-job Git archives, and relocated data and weight paths. It ran on compute nodes of the existing cluster; **there is no claim of execution on another physical cluster**. A new cluster selects its account, GPU, Python/module, and storage paths through the site configuration.

Runtime qualification of the new environment is complete: 143 training-environment checks and 117 audit-environment checks passed, 91 GPU short-budget cases passed, and the 999-event audit and historical diagnostic replays completed. This conclusion verifies the execution pipeline; it is not a full-budget reproduction of every historical score. All tensor/model operations, dependency installation, downloads, and large-file verification ran under Slurm. The login node handled only source code, small metadata, Git, and submission checks.

## Verified environment and CPU evidence

| Job | Source commit | Node | Result and scope |
|---|---|---|---|
| 22101442 | 98d12f0 | c142 | 66 tests pass; old environment, early regression checks only |
| 22101813 | 58cf86a | c56 | 106 tests pass; imports of the actual official T5 class and drivers pass; old environment |
| 22102010 | 7416004 | c376 | 133 tests pass; imports of the actual official T5 class and drivers pass; old environment |
| 22101774 | 454f192 | c14 | Data moved to a separate directory; B3/B5 and both sets of seed 0 teacher weights relocated; teacher digests checked file by file |
| 22102009 | 7416004 | c350 | P00, attention, and 48 QA maps relocated, with digests recorded |
| 22101441 | 98d12f0 | c83 | Installation failed: a missing final newline in an official file made the old patch inapplicable; module loading was also found to reset the pip wheelhouse |
| 22101770 | 454f192 | c56 | Fresh training environment installed from public PyPI; stopped at the same old patch |
| 22102098 | c617100 | c3 | Continued from the public-source environment above, precisely corrected the official-source patch; training/audit `pip check` both pass; setup completed |
| 22102101 | c617100 | c38 | Three Swin and MiT-B2 assets downloaded from public sources; all match pinned SHA256 values |
| 22102100 | c617100 | c3 | New environment: 134 tests pass, 2 fail; failures were old assertions of original-server paths, changed to explicit temporary directories and passed on rerun in 22102394 |
| 22102394 | 19132b0 | c333 | New training environment: 143 tests pass; imports of the actual official T5 class and retained drivers pass |
| 22102400 | 19132b0 | c332 | New audit environment: 117 tests pass, covering repair/validation, schema, inventory, target gate, and split |
| 22102613 | eb702c2 | c422 | In the new training environment, both the official original-year converter CLI and the added-year converter pass synthetic 23-band GeoTIFF → HDF5 checks, covering values, dates, coordinates, and active-fire conversion semantics |

Logs from both installation failures are retained; failed jobs are not reported as successful. The public-source environment came from the empty directory created by `22101770`, without reusing the old training venv; `22102098` reused this newly installed public-source environment. Pip variables were isolated again after module configuration, and installation records do not depend on this site's wheelhouse. Observed versions are recorded in the [training freeze](environment/train-pip-freeze.txt) and [audit freeze](environment/audit-pip-freeze.txt); official-source changes appear in [upstream.diff](environment/upstream.diff). These are observations; the installation inputs remain `environments/research-*.txt`.

The complete data came from the existing audited 999-event dataset, relocated with hard links on the same filesystem. This therefore verifies new paths and a new software environment, without repeating the complete public download/conversion. Weights were actually copied and verified. Complete public-data reconstruction commands, stage-by-stage checks, and original checksums are in [data preparation](data-preparation.md).

## GPU verification boundaries

Cross-history and both September smoke campaigns use real data, one short training or calibration run, checkpoint saving/reloading, and limited-sample evaluation. Historical D-series and base models use the [qualification driver](../../scripts/research/README-qualification.md); short-budget outputs are explicitly marked as qualification and cannot serve as 3,000/10,000-step results, teachers, or completed full-baseline records.

The formal D-series evaluator requires a fixed training budget. Qualification replaces its budget/status fields only in an in-memory copy to call the original model constructor; all other method-metadata checks and strict state_dict loading are retained. The formal validator still rejects the short-budget weights on disk. This demonstrates the loading/computation pipeline, not method efficacy or numerical agreement with full training.

GPU job `22102452` (`e13b9fb5db55`, `g36`) completed two actual 24-sample diagnostics. The [replay results](environment/diagnostics-replay.json) agree with historical records in comparison direction, with a maximum absolute numerical difference of approximately `4.3e-6`. This verifies inference after weight relocation, not retraining.

The first 6 training/calibration GPU jobs exited successfully. Reports for their 19 short-budget cases are in the [raw records](environment/gpu-qualification-first-batch.json):

| Job | Node | Passed scope |
|---|---|---|
| 22102102 / 22102104 | g32 / g35 | T1 / T5: control, cosine_erm, context, transition; each actually ran 1 step, strict reload, prediction reload difference 0 |
| 22102396 | g36 | B5: actual Lightning 2 steps, best-weight selection, strict reload, 2-sample evaluation each for M00/M06 |
| 22102397 | g32 | RNC: actual 1 step, formal validator rejects short-budget weights, strict reload, 2-sample evaluation each for M00/M06 |
| 22102787 | g36 | RF T1: control, mixed, typed, distill, BN batch_stats; reload difference 0 |
| 22102798 | g35 | Routed T1: student_control, student_distill, merge, bn_shared; evaluation after strict equality of reloaded state |

These short-budget samples are not used to assess efficacy; AP of 0 on a small sample cannot be interpreted as method failure. Later batches complete the remaining execution families, as detailed below.

The conversion job's [raw compact report](environment/conversion-qualification.json) retains year-by-year checks. It uses eight synthetic events and does not replace the full public download, 999-event conversion, or repair-stage validation.

## Subsequent actual verification

All 9 GPU jobs in the second batch (`22102808/09/10/82/84/88/93/94/95`, source `a7e8e0a`) exited normally, adding 50 passing cases; see the [per-case report](environment/gpu-qualification-second-batch.json). These include short training and reloading of the remaining positive-list cross-history methods at T1/T5, the RF and routed T5 branches, and B1, token, and CRA. Architectures, the remaining D-series, and attention training were verified in the final batch.

CPU job `22102402` (source `19132b0`, `c333`) completed the full 999-event audit. The [report](environment/data-audit-phase0_report.md) shows no invalid-file errors, with gate decision `continue_controlled`. The original data lack observation/availability times, QA, coverage, target validity, and related fields, so they support the specified controlled-missingness experiments but not natural-missingness or operational claims. The report's target days are daily counts from original events and use a different counting unit from model evaluation samples constructed with history windows.

The next 10 GPU cases for historical entrypoints also passed; see the [per-case report](environment/gpu-qualification-legacy-batch.json): D1-KL / paired0, original D12-SARP, B2, B3, legacy-P00, CIWC, rank, FFCA, and prompt-pyramid. B2/B3/P00 use actual Lightning 2 steps; the others use 1 step. All reports retain the qualification marker. CPU job `22103003` (`c86`) also ran the T1/T5 complete-route and severity-route CLIs on relocated historical summaries; see the [output](environment/composition-qualification.json). This verifies the aggregation commands, not a new model evaluation.

All 12 cases across the final 10 GPU jobs passed; see the [per-case report](environment/gpu-qualification-final-batch.json): complete-pyramid, T5 standard/SARP, Swin/SegFormer T1/T5 short training from public initialization, T1/T5 training with fixed 25%/50% missingness, and actual 1-step attention training and reloading. Swin/SegFormer use microbatch 2 for pipeline verification; formal reproduction still uses the historical batch in the architecture tutorial. Their actual completed_steps is 1, not the target budget of 10,000 in metadata. All jobs have exit code 0.

An independent coverage review checked the 51 positive signals, code entrypoints, controls, historical batches, route compositions, and short-budget validation boundaries, finding no new critical gaps. Retained negative results, invalid comparisons, and unrun states are archived according to the original evidence; no additional efficacy experiments were performed for them.
