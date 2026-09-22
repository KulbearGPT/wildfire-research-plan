# Code lifecycle: X22 + X17 mainline

Scope selected by the project owner on 2026-09-22: **only X22 + X17 and its
necessary controls are active**. This is a maintenance decision, separate from
whether a historical experiment measured a positive or negative result.

## Active implementation and comparison

| Role | Implementation / setting | Why retained |
|---|---|---|
| Final system | `cross_history/compose_complete_routes.py`, X22 + X17 | M00 from fresh ERM, M01 from X22, M06/M07 from fixed-severity specialists |
| X22 component | `cross_history/run.py --method cosine_erm` | Confirmed cosine-schedule continuation; an optimizer/training contribution |
| Fresh ERM control | `cross_history/run.py --method control` | Same initialization and continuation budget; do not substitute frozen B3/B5 |
| X17 components | `--method block_specialist --block-fraction 0.25` and `0.5` | The fixed 25% and 50% experts used in the final system |
| X14 attribution control | `--method block_specialist` without `--block-fraction` | Mixed-severity specialist; needed to distinguish factorization from specialization |
| X17 attribution report | `cross_history/compose_severity_routes.py` | Preserve comparison with X14, including failed attribution cells |
| Initialization / frozen reference | corrected B3 at T1 and B5 at T5 | Shared source checkpoints and the separate frozen-reference comparison |
| Shared support | `src/wildfire_phase0/`, corrected data/evaluation utilities, `scripts/research/` | Data contract, reproducible execution, and artifact verification |

Paths above are relative to `reproductions/` unless otherwise stated. See the
[cross-history code guide](../reproductions/cross_history/README.md) and
[foundation guide](../reproductions/wsts_fast_track/README.md).

The retained system has three-seed, 2021/2022/2023 evidence in both existing T1
and T5 settings. Its mean primary gain is +0.014642 against fresh ERM and
+0.034527 against frozen B3/B5. These are different comparisons. The former
does not meet the historical +0.020 magnitude target. X17 failed some held-out
attribution comparisons against X14; system retention does not turn it into an
independently confirmed mechanism. Evidence and exact cells remain in
[the campaign ledger](experiments/t1_t5_innovations.md).

## Fixed settings and claim limits

The active paired protocol uses training years 2016–2020, validation year 2021,
and historical reporting years 2022/2023. Normalization uses training data only.
T1 uses Res18-U-Net with 40 features; T5 uses Res18-UTAE with 33 features/day.
Both use the same target dates and populations: 3181/2856/2102 for 2021/22/23.
See the [population audit](research/evaluation-population.md) for the corrected
historical 2312 typo; do not reconstruct a second population from that typo.

The later mainline uses 3000 continuation optimizer steps, physical batch 64,
effective batch 64, seeds 0/1/2 for the recorded confirmation, final-step
checkpoints, and matched initialization. X22 changes the schedule; fixed-severity
experts change the training corruption mixture. The primary metric is mean AP
over M01/M06/M07, with M00 and block AP reported separately. CLI defaults in a
shared historical runner are not a substitute for these explicit settings.

The complete-route composer combines scenario summaries from separately trained
experts. It is a system evaluation, not a new single-model architecture or a
general deployment router. This cleanup does not add an inference service.
Historical test years have been inspected, and controlled missingness is not
evidence of operational robustness or a pure causal effect of history length.

## Preserved but inactive directions

Every existing method retains its evidence `status`, result, comparator, setting
scope, source document/commit, and recipe in
[method-inventory.json](research/method-inventory.json). The additional `lifecycle`
field determines current maintenance scope. `positive_signal` never implies active.

| Preserved code | Exploration / original setting | Current status |
|---|---|---|
| Other branches in `cross_history/run.py` and `models.py` | X1–X31 repairs, consistency, restoration, adapters, auxiliary losses; individual T1/T5 settings and budgets in the ledger | Archived exploration; includes X8 and the X8+X17 composition |
| `cross_history/run_three_directions.py`, `three_directions.py`, BN audits | Reliability-fusion campaign: BN, shallow branches, same-input KD | Archived exploration/diagnostics; RF naming in inventory |
| `three_directions/` | Separate normalization, weight-merge, and KD campaign | Archived exploration; TD naming, not interchangeable with RF |
| `wsts_fast_track/` D1/D2/D12 and other optional trainers | Earlier corrected T1 continuation and module attribution | Archived research, including previously retained positive D12 results |
| Belief-state, legacy P00, natural-observation and target-QA modules | Earlier initialization regimes or diagnostic populations | Archived experiments/diagnostics; retain invalid-foundation and population caveats |
| Noncanonical branches in `cross_history/architectures.py` | SegFormer, SwinUnet and ConvLSTM transfer screens | Archived architecture exploration; runtime qualification is not efficacy confirmation |
| B0/B1/B2 and `wsts_res18_unet_t1/` | Clean/corruption references and independent official-fold teaching | Reference/teaching support, outside the selected contribution route |
| Site-specific `run_*on_nibi.sh`, `run_slurm.sh`, `submit_heldout.py`, `scripts/cluster/` | Original Nibi environment, paths, resources, and campaign helpers | Legacy execution/reference support; use `scripts/research/submit.sh` for the portable mainline |
| `run_confirmation_bundle.sh`, `run_t5_confirmation_bundle.sh` | One-off September queue replacements with hardcoded job IDs and `scancel` | Retired; refuse execution before any scheduler operation |

Shared modules remain at their original import paths because active and archived
methods use them. File-level comments identify mixed modules; per-method lifecycle
in the inventory resolves the distinction. Archived implementation and tests are
retained for traceability, not included in a new full-reproduction requirement.
No result artifact, checkpoint format, loss, parameter, or data split is deleted
or changed by this cleanup. Do not reactivate a direction merely because its
CLI switch still exists; reopening it requires a new research decision.

## Working on the retained route

Start at [the reproduction guide](research/reproduce.md) for environment/data and
[baseline regeneration](research/baselines.md) for B3/B5. The
[method recipes](research/method-recipes.md) retain historical commands; select
only `control`, `cosine_erm`, and `block_specialist` for this mainline.
Use distinct output paths for mixed, 25%, and 50% specialists, and preserve the
recorded physical batch explicitly. Prepare one matched comparison before any
larger reproduction; this scope selection does not authorize a new compute budget.

When modifying shared code, inspect both the active dependency and affected
archive callers. Verify the actual changed behavior rather than retraining every
preserved method. Historical scripts with retired job IDs must remain inert.

## Cleanup verification (2026-09-22)

The lifecycle map covers all 138 tracked Python, shell and PowerShell source
files under `reproductions/`, `src/` and `scripts/`. Every entry links a settings
record. All 100 pre-existing method records retain their original fields and
values; lifecycle metadata was added separately. The active method IDs are X14
(control), X17, X22, X22+X17, B3, B5 and FOUNDATION-RULES (data support).

All 115 Python files have identical parsed syntax trees to the pre-cleanup
version; changes there are comments only. All 22 shell scripts pass `bash -n`.
The two retired bundles were invoked without an allocation and returned their
retirement message with exit code 2, before any scheduler action. PowerShell
received comments only; it was not executed on this Linux host.

All 18 English course guides rebuilt successfully. The course homepage and
18 generated guides passed English-language and local-link-target checks; new
navigation documents passed local-link and code-fence checks. No model imports,
numerical tests, training, evaluation or Slurm jobs ran for this cleanup.
These checks establish source/metadata preservation and retirement behavior,
not a fresh scientific reproduction or a new website deployment.
