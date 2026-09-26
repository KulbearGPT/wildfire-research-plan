# Archived architecture follow-up

These are the historical follow-up rules from the teaching roadmap compiled on
2026-09-11 from records through `6111d9e` (2026-09-08). The project owner selected
[only X22+X17 and necessary controls](../CODE_LIFECYCLE.md) on 2026-09-22.
SwinUnet, SegFormer-B2 and ConvLSTM are inactive. The instructions below preserve
that snapshot; they do not schedule inspections, additional seeds or new runs.
No live queue query or remote metric recomputation was performed for this archive.

For experimental settings, commands and original outcomes, see the
[architecture recipe](../research/architectures.md) and
[cross-history ledger](../experiments/t1_t5_innovations.md). The architecture-transfer
system used X22 for M00/M01, unlike the active system's fresh ERM M00 route.
Alternative architectures used their own public initializations and recipes;
their results cannot be folded into the canonical B3/B5 comparison.

## Historical snapshot and proposed follow-up


Architecture status here is the snapshot at `6111d9e`, not the current queue or an instruction to launch new experiments. Consult the current handoff before taking action.

| Setting | Evidence in the snapshot | Follow-up and stopping rule |
| --- | --- | --- |
| Res18-U-Net T1 / Res18-UTAE T5 | X14/X22 three-seed and fixed 2022/2023 evaluation completed | Assemble auditable main tables, seed variation and costs; do not rescreen established results |
| SwinUnet T1/T5 | Both bootstraps completed; T5 seed-0 mixed BlockDrop passed the screen; T1 results missing from that record | Inspect original jobs and artifacts first; only directions passing both settings proceed to additional seeds and fixed tests |
| SegFormer-B2 T1/T5 | Paired seed-0 results complete; no transfer evidence passing both settings | Screen closed; no additional seeds/tests or required student rerun |
| ConvLSTM T5 | Bootstrap checkpoint/evaluation saved; five continuation jobs recorded | Inspect complete paired 2021 results and apply the existing protocol; a T5-only result cannot support a cross-T claim |

The follow-up sequence specified in that snapshot was:

1. **Close the evidence record:** inspect existing Swin T1 and ConvLSTM jobs/artifacts and update the ledger. Determine what already exists before allocating computation.
2. **Confirm the gates:** for architecture/method pairs passing the original protocol, complete seeds 1/2 and evaluate 2022/2023 only after freezing the recipe. Do not change routing or select extra seeds because one scenario looks favorable.
3. **Build the main table:** use `reproductions/cross_history/compose_table1.py` and its tests to check matched dates, seed completeness, controls and routes. Report 2022/2023 per-scenario AP, primary metric, gain over matched ERM and seed standard deviation; keep the 2021 screen in the appendix.
4. **Produce teaching deliverables:** data audit, baseline reproduction report, single-variable comparison, three-seed table and claim-boundary statement. Preserve code/config versions, data/weight manifests, Slurm job IDs and artifact paths.


Return to the [current teaching roadmap](../research-roadmap.md) for evidence
closure of the active route. Reopening this archive requires a separately defined
research scope and compute budget; historical positive signals remain evidence
within their original setting.
