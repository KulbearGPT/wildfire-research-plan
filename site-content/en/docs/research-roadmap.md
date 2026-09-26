# Wildfire forecasting: a research-methods teaching roadmap

> Current scope: [X22+X17 and its necessary controls](CODE_LIFECYCLE.md). This roadmap preserves the teaching history; other experimental directions are inactive archives or references.

This page preserves the teaching narrative as of 2026-09-11. The scope selected on 2026-09-22 is X22+X17 and necessary controls only. The September campaigns and other directions remain historical evidence; see the [current research handoff](research/reproduce.md), [positive-signal catalog](research/positive-signals.md) and [archive index](archive/README.md).

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
- Current mainline continuation uses the same B3/B5 initialization and seed, 3000 AdamW updates, initial LR 0.001, physical/effective batch 64, and the final-step checkpoint. Match other settings between candidate and control. Earlier batch settings and alternative-architecture recipes remain in their original records and do not belong in this paired comparison.
- Historical decision order: 2021 seed-0 screen → prespecified seeds 1/2 confirmation → recipe freeze → fixed 2022/2023 evaluation. Screening requires primary gain of at least 0.005 in both settings and no M00 loss greater than 0.010. Confirmation uses prespecified seed means, not the best seed. Added modules must also pass their nearest attribution comparison.
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

These early conclusions come from the historical ledger, not cross-setting three-seed confirmation. D1 and D12 are now archived teaching references; their historical signals do not make them active or require rescreening.

### 4.2 Historical cross-setting evidence: X22 and the necessary X14 control

Each cell is a three-seed mean primary-metric AP gain for a matched year and setting, compared with fresh ERM continuation at the same budget.

| Direction | T1 2021 | T1 2022 | T1 2023 | T5 2021 | T5 2022 | T5 2023 | Overall mean over 18 paired rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| X14 BlockDrop specialist continuation | +0.005467 | +0.001416 | +0.006792 | +0.016685 | +0.011528 | +0.005485 | +0.007895 |
| X22 cosine ERM | +0.006035 | +0.008133 | +0.003442 | +0.021346 | +0.019504 | +0.013596 | +0.012009 |

X14 is currently retained only as the mixed-severity attribution control for X17. The historical X14 route uses its specialist only for block-missing scenarios and matched ERM for M00/M01; its clean preservation follows from routing. X22 improves the training recipe through standard cosine learning-rate decay; it is not a new architecture. Both have three-seed and fixed-test-year support. This snapshot does not establish three independent innovations.

### 4.3 Retained system result

The X22 + X17 fixed-severity block-specialist combination is the active retained system: fresh ERM for M00, X22 for M01, and 25%/50% specialists for M06/M07. Its mean primary gain over 18 paired rows is **+0.014642** versus same-budget ERM. Against the original frozen B3/B5 reproduction checkpoints, its mean gain over six setting/year cells is **+0.034527**.

These answer different questions: gain beyond ordinary continuation versus gain of the complete system over the original baseline. The gain over fresh ERM does not meet the historical +0.020 magnitude target, and X17 failed some held-out attribution comparisons against mixed X14. Severity decomposition is not an independently confirmed contribution, and the combination does not increase the method count. The [archived architecture-transfer design](archive/architecture-followup.md) separately uses X22 for M00/M01 and is a distinct system definition.

## 5. Current task: close the X22+X17 evidence and comparisons

Current work reviews X22+X17 and necessary controls. Historical SwinUnet,
SegFormer-B2 and ConvLSTM status and follow-up rules are preserved in the
[architecture archive](archive/architecture-followup.md); they are not the next
execution steps.

| Object | Evidence to review | Completion criterion |
|---|---|---|
| B3/B5 and shared data | Source checkpoints, checksums, data and training-only normalization identity, frozen-reference evaluations | Establish shared initialization within each setting; keep frozen references separate from fresh ERM |
| Fresh ERM and X22 | Matched T1/T5, seeds 0/1/2, 2021/2022/2023 results and training provenance | Match data, initialization, physical batch, updates and checkpoint selection; identify the cosine schedule difference |
| X14 mixed and X17 mild/severe | Matched mixed, fixed-25% and fixed-50% specialist results | Retain every attribution cell, including failed held-out comparisons |
| Complete X22+X17 system | M00=ERM, M01=X22, M06=mild, M07=severe composition and source results | Explicit route, no duplicate or missing pairs; separate gains over fresh ERM and frozen B3/B5 |

Follow this sequence:

1. **Inventory existing assets:** check weights, yearly results, configurations,
   source commits and job records against the [active artifact checklist](research/artifacts.md).
   Record unavailable files as gaps; a historical path does not prove availability.
2. **Review the paired contract:** align history, seed, year, target dates,
   sample counts and training provenance. `summary.json` alone does not establish
   initialization, batch or training budget; review checkpoint metadata, source
   and configuration.
3. **Build traceable comparisons:** follow the [active recipe](research/method-recipes.md)
   using `compare.py`, `compose_severity_routes.py` and `compose_complete_routes.py`.
   Retain scenario AP, primary missingness AP, block mean, seed variation and cost
   records. Separate 2021 selection from historical 2022/2023 reporting. A single
   seed's composition cannot establish full confirmation.
4. **Produce teaching deliverables:** data audit, baseline reproduction, paired
   X22 comparison, X17 attribution against X14, complete-system comparison, and
   evidence limits and gaps. Fill a real gap only within authorized compute scope
   using the active recipe; do not automatically add seeds, architectures or screens.

## 6. Student route and acceptance criteria

For one fold without this repository, use the [official-codebase tutorial](tutorials/res18-baseline-slurm.md). For B0 with project indexing and splits, use the [project B0 tutorial](tutorials/project-b0-slurm.md). Do not mix their result definitions.

| Step | Student task | Acceptance criterion |
| --- | --- | --- |
| 1: Understand the problem | Read this page, related-work courseware and phase0 report | Explain proxy labels, controlled missingness and chronological splits; distinguish fire detections from perimeters |
| 2: Understand the data | Reproduce the audit and rule baselines at configured paths | Match event counts, splits and label integrity; explain why sampling protocols produce different counts |
| 3: Reproduce a model | Verify a released checkpoint, then learn corrected B0/B3 | Distinguish released-weight scores, own training and paper means; save traceable configuration |
| 4: Make a fair comparison | Review fixed ERM/X22 pairs and X17 attribution against mixed X14; reproduce required items within scope | Same initialization, budget, physical batch, data and evaluation; report all four scenarios |
| 5: Learn reliable conclusions | Summarize existing three-seed/year results; record gaps and reproduction scope | Do not cherry-pick seeds; separate validation selection, fixed tests and module attribution |
| 6: Write the report | Audit the X22+X17 table, attribution and evidence chain; distinguish active work from archives | Trace every claim to configuration, commit, run and result; do not call pending experiments successful |

Run training and model evaluation through Slurm on the selected cluster. Use CPU Slurm jobs for numerical checks and result aggregation. Login nodes are for source editing, Git, small metadata checks, documentation and submission. Environment entry points include the [current handoff](research/reproduce.md), `environments/README.md` and `docs/cluster-migration.md`. Keep data, environments, weights, checkpoints and large logs outside Git, with manifests and artifact paths for recovery. Check available environments and artifacts before starting training to avoid unnecessary full reruns.

## 7. Code and evidence entry points

- `main` now includes the integrated research handoff, cross-setting and cross-architecture code, and both September campaigns. Follow the [current reproduction guide](research/reproduce.md).
- `research/t1-t5-innovations` and `.worktrees/t1-t5-innovations/` identify historical branch/worktree provenance; students do not need that personal worktree to reproduce the handoff.
- The earlier T=1 roadmap is retained at `docs/experiments/t1-stage-roadmap.md` as historical context.
- The historical project-wide Git audit is `docs/project-handoff-git-audit.md`. Exploration remains in Git and original ledgers; students are not required to rerun it.

Suggested reading: this page → [current handoff](research/reproduce.md) → [active recipe](research/method-recipes.md) and [artifact checklist](research/artifacts.md) → X22/X17 and necessary-control evidence in `docs/experiments/t1_t5_innovations.md`. Consult phase0, official reproduction and the earlier quantitative ledger for foundations and teaching history as needed.
