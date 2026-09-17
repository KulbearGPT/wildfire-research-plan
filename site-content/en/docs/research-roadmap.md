# Wildfire forecasting: a research-methods teaching roadmap

This page preserves the teaching narrative as of 2026-09-11. For the two September campaigns and the full retained-method collection, see the [current research handoff](research/reproduce.md) and [positive-signal catalog](research/positive-signals.md).

Compiled on 2026-09-11 from committed records through research commit `6111d9e` (2026-09-08). The historical snapshot below is not a live Slurm query or a recomputation of remote metrics.

## 1. The question students should understand first

We study reliable prediction of next-UTC-calendar-day VIIRS active-fire proxy labels when satellite inputs have predefined missingness. The label is not a complete fire perimeter. Existing evidence does not support general performance claims for natural missingness or operational deployment.

The teaching sequence moves from trustworthy data and baseline reproduction to fair comparisons and method validation. It retains the experiment chain without requiring students to rerun failed explorations, abandoned modules or historical scheduling repairs.

## 2. The retained research process

| Stage | Work completed | Research-methods lesson | Evidence |
| --- | --- | --- | --- |
| A: Problem and literature | Research plan, related-work courseware and a next-day active-fire prediction task | Define inputs, targets and scope before proposing novelty | `index.html`, `related-work/` |
| B: Data contract and quality gate | Audit of 999 event HDF5 files, necessary label repair, independent verification and a frozen chronological split | Data repair is part of a trustworthy experiment; preserve provenance, checksums and audit evidence | `docs/experiments/phase0.md`, `src/wildfire_phase0/` |
| C: Simple references | Fixed no-fire and latest-mask persistence baselines | Establish whether a learned model adds information beyond simple rules | Rule-baseline results in the phase0 report |
| D: Official reproduction | Full Res18-U-Net T=1 Fold-2 training and executable records for twelve released checkpoints | Separate own training, released-weight reevaluation and paper aggregates | `docs/experiments/res18_unet_t1_reproduction.md`, `baseline-reproduction/` |
| E: Controlled-reliability baselines | Corrected pooled-year sample indexing; B0 → B2 → B3, missingness interventions and AP evaluation | Match data and budget; distinguish total recipe gain from module gain | `docs/experiments/quantitative_reliability_ledger.md` |
| F: T=1 methods | D1-KL consistency and D12-SARP prompts with their matched controls | Each module needs its nearest attribution control; single-setting evidence has limits | Same ledger; `reproductions/wsts_fast_track/` |
| G: Cross-setting validation | Matched T=1/T=5 target-date protocol; three-seed X14/X22 and fixed-year evaluation | Move beyond a single score to stability across seeds, years and settings | `docs/experiments/t1_t5_innovations.md` |
| H: System audit and architecture extension | Frozen-baseline audit of retained combinations; paired SwinUnet, SegFormer-B2 and ConvLSTM experiment workflows | Separate system gains, attribution and architecture transfer; include only protocol-compliant results in the main table | `reproductions/cross_history/` and architecture records |

Necessary data and evaluation corrections remain in the teaching chain because they determine whether later scores are trustworthy. They are not failed ideas that students need to rescreen.

## 3. The experimental contract

- Data: 999 events; 2016–2020 training (653), 2021 validation (156), and 2022–2023 test (190).
- Scenarios: M00 complete input; M01 missing active-fire history; M06/M07 structured blocks covering 25%/50% of dynamic inputs. Controlled masks may be used by prescribed routing, but this does not establish complete knowledge of real-world missingness.
- Cross-setting primary metric: `(AP_M01 + AP_M06 + AP_M07) / 3`. Also report block mean, M00, individual scenario AP, runtime and parameter count. AP gains are absolute differences, not relative percentages.
- T=1 uses Res18-U-Net with 40 channels; T=5 uses Res18-UTAE with 33 selected channels per day. History, architecture and features all change, so this is cross-setting validation rather than an isolated history-length effect.
- Matched event/target dates use `target_index = in_fire_index + 6`. Sample counts for 2021/2022/2023 are 3181/2856/2102 in both settings. The earlier 2023 count of 2312 was a ledger typo; the verified original results use 2102. See the [population audit](research/evaluation-population.md).
- Original cross-setting continuation uses the same B3/B5 initialization and seed, 3000 AdamW updates, initial LR 0.001, effective batch 64, and the final-step checkpoint. Match other settings between candidate and control. New architectures use their own public initializations and recipes, not an assumed universal learning rate.
- Decision order: 2021 seed-0 screen → prespecified seeds 1/2 confirmation → recipe freeze → fixed 2022/2023 evaluation. Screening requires primary gain of at least 0.005 in both settings and no M00 loss greater than 0.010. Confirmation uses prespecified seed means, not the best seed. Added modules must also pass their nearest attribution comparison.
- Historical test years have already been evaluated. Students may reproduce fixed results. New method development must disclose this exposure and define an independent confirmation plan before starting; those years cannot be relabelled as unseen tests.

## 4. Retained experiment chain and conclusions

### 4.1 Baselines and T=1 teaching modules

| Experiment | Comparator and role | Evidence and limitations |
| --- | --- | --- |
| B0 | Corrected clean baseline | Shared foundation; verify indexing and label semantics first |
| B2 FireDrop | Versus B0; routing by observable missingness type | Historical three-year mean M01 AP gain +0.150109 |
| B3 FireDrop + BlockDrop | Versus B2 | Historical three-year block-mean gain +0.027821 |
| D1-KL | Matched paired-supervision continuation D1-ERM | Historical three-year primary missingness gain +0.008970; retained as T=1 evidence |
| D12-SARP | D2-STD for module attribution; D1-ERM for total comparison | Module block mean +0.005764; total missingness primary +0.020699; retained T=1 evidence, with a 2022 M00 limitation in the total comparison |

These early conclusions come from the historical ledger, not cross-setting three-seed confirmation. D1 and D12 are optional advanced teaching modules; rescreening them is not a prerequisite for the main route.

### 4.2 Two supported cross-setting improvements in this snapshot

Each cell is a three-seed mean primary-metric AP gain for a matched year and setting, compared with fresh ERM continuation at the same budget.

| Direction | T1 2021 | T1 2022 | T1 2023 | T5 2021 | T5 2022 | T5 2023 | Overall mean over 18 paired rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| X14 BlockDrop specialist continuation | +0.005467 | +0.001416 | +0.006792 | +0.016685 | +0.011528 | +0.005485 | +0.007895 |
| X22 cosine ERM | +0.006035 | +0.008133 | +0.003442 | +0.021346 | +0.019504 | +0.013596 | +0.012009 |

X14 uses its specialist only for block-missing scenarios and matched ERM for M00/M01; its clean preservation follows from routing. X22 improves the training recipe through standard cosine learning-rate decay; it is not a new architecture. Both have three-seed and fixed-test-year support. This snapshot does not establish three independent innovations.

### 4.3 Retained system result

The X22 + fixed-severity block-specialist combination is retained as a system result: fresh ERM for M00, X22 for M01, and 25%/50% specialists for M06/M07. Its mean primary gain over 18 paired rows is **+0.014642** versus same-budget ERM. Against the original frozen B3/B5 reproduction checkpoints, its mean gain over six setting/year cells is **+0.034527**.

These answer different questions: gain beyond ordinary continuation versus gain of the complete system over the original baseline. Severity decomposition is not an independent contribution, and the combination does not increase the method count. The architecture-transfer design separately uses X22 for M00/M01; label that distinct system definition explicitly.

## 5. Historical stopping points and follow-up rules

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

## 6. Student route and acceptance criteria

For one fold without this repository, use the [official-codebase tutorial](tutorials/res18-baseline-slurm.md). For B0 with project indexing and splits, use the [project B0 tutorial](tutorials/project-b0-slurm.md). Do not mix their result definitions.

| Step | Student task | Acceptance criterion |
| --- | --- | --- |
| 1: Understand the problem | Read this page, related-work courseware and phase0 report | Explain proxy labels, controlled missingness and chronological splits; distinguish fire detections from perimeters |
| 2: Understand the data | Reproduce the audit and rule baselines at configured paths | Match event counts, splits and label integrity; explain why sampling protocols produce different counts |
| 3: Reproduce a model | Verify a released checkpoint, then learn corrected B0/B3 | Distinguish released-weight scores, own training and paper means; save traceable configuration |
| 4: Make a fair comparison | Reproduce fixed ERM/X22 seed-0 pairing, then learn X14 routing | Same initialization, budget, data and evaluation; change only the intended factor and report all four scenarios |
| 5: Learn reliable conclusions | Summarize existing three-seed/year results and reproduce confirmation as needed | Do not cherry-pick seeds; separate validation selection, fixed tests and module attribution |
| 6: Write the report | Audit the main table and evidence chain; explain architecture-transfer coverage | Trace every claim to configuration, commit, run and result; do not call pending experiments successful |

Run training and model evaluation through Nibi Slurm. Use login nodes only for source work, small metadata, result aggregation and submission. Environment entry points include `environments/README.md` and `docs/cluster-migration.md`. Keep data, environments, weights, checkpoints and large logs outside Git, with manifests and artifact paths for recovery. Check available environments and artifacts before starting training to avoid unnecessary full reruns.

## 7. Code and evidence entry points

- `main` now includes the integrated research handoff, cross-setting and cross-architecture code, and both September campaigns. Follow the [current reproduction guide](research/reproduce.md).
- `research/t1-t5-innovations` and `.worktrees/t1-t5-innovations/` identify historical branch/worktree provenance; students do not need that personal worktree to reproduce the handoff.
- The earlier T=1 roadmap is retained at `docs/experiments/t1-stage-roadmap.md` as historical context.
- The historical project-wide Git audit is `docs/project-handoff-git-audit.md`. Exploration remains in Git and original ledgers; students are not required to rerun it.

Suggested reading: this page → `docs/experiments/phase0.md` → `docs/experiments/res18_unet_t1_reproduction.md` → `docs/experiments/quantitative_reliability_ledger.md` → retained directions and architecture results in `docs/experiments/t1_t5_innovations.md`, with the current handoff for execution.
