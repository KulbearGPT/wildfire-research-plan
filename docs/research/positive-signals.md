# Positive signals, including limited and historical findings

This collection uses the user’s inclusive criterion: retain a measured positive change even when it is confined to one seed, history, year, scenario, or weak comparator. **A positive signal is not a confirmed contribution or a passed adoption gate.** Mixed results also appear in the negative archive. No new efficacy experiment was performed for this inventory.

AP differences are absolute, not percentages. Primary is mean AP over M01/M06/M07; block is mean AP over M06/M07. M00 is clean. T1/T5 differ in architecture and features as well as history. The corrected retained T1 and paired campaigns use 3,181/2,856/2,102 samples in 2021/2022/2023. The old ledger's 2,312 was a documentation typo, corrected against all original result summaries; see [population audit](evaluation-population.md). Unless stated otherwise, a screen is seed 0 on 2021; later confirmation/heldout detail is in the source ledger, not implied by a positive sign.

The [machine-readable inventory](method-inventory.json) records source commits, entrypoints, scope, unavailable dependencies, and unrun states. The [negative collection](negative-results.md) preserves failed attribution, mixed settings, invalid foundations and cancelled work separately. Runtime commands and artifact provisioning belong to [the reproduction guide](reproduce.md); the entrypoints below identify real existing source interfaces and are not complete configured job commands.

## Corrected T1 and cross-history observations

| ID / implementation | Observation and nearest comparator | Source |
|---|---|---|
| X1 — X1 context transport | fresh continuation; observable spatial route; yes; routed +.005954; routed +.000874; cross-T reject | docs/experiments/t1_t5_innovations.md |
| X1-adapter — X1 frozen adapter | frozen initial B3/B5 plus fresh continuation; alternative X1 implementation; +.012151 vs frozen, -.013537 vs fresh; reject; cancelled after T1 adoption failure | docs/experiments/t1_t5_innovations.md |
| X3 — X3 latent transition | fresh continuation; yes; original +.011831; decoupled -.045065; original -.009413; repair cancelled | docs/experiments/t1_t5_innovations.md |
| X4 — X4 spatial risk weighting | fresh continuation; one training-objective contribution; routed +.003463; routed -.001659; reject | docs/experiments/t1_t5_innovations.md |
| X6 — X6 balanced corruption coverage | fresh continuation; sole corruption-rate tuning contribution; +.005203; +.002426 with M00 -.017601; reject | docs/experiments/t1_t5_innovations.md |
| X8 — X8 counterfactual-impact consistency | unchanged global D1 consistency; fresh ERM adoption; incremental FireDrop specialist; incrementally reliable; ERM adoption fails 2022; incrementally reliable and ERM-positive | docs/experiments/t1_t5_innovations.md |
| X10 — X10 forecast-aware dynamic inpainting | fresh continuation with spatial route; yes; typed input restoration optimized by forecast loss; 2022/23 -.001356/-.000347; reject cross-history; 2022/23 +.004783/+.002139; T5-only positive | docs/experiments/t1_t5_innovations.md |
| X12 — X12 counterfactual FireDrop specialist | fresh continuation; plain specialist ablation if screen passes; one specialist-training contribution; routed +.001589; reject; cancelled after T1 failure | docs/experiments/t1_t5_innovations.md |
| X14 — X14 BlockDrop specialist continuation | fresh continuation with spatial route; one block-specialization training contribution; reliable: +.005467/+.001416/+.006792 in 2021/22/23; reliable: +.016685/+.011528/+.005485 in 2021/22/23 | docs/experiments/t1_t5_innovations.md |
| X18 — X18 block-specialized context transport | X14 specialist plus fresh ERM; yes only if positive vs X14; +.013205 vs ERM; +.004405 vs X14; +.006524 vs ERM but -.000922 vs X14; reject | docs/experiments/t1_t5_innovations.md |
| X19 — X19 severity-conditioned latent adapters | X14 specialist; X17 two-checkpoint upper bound; fresh ERM; no; T5 closest-control confirmation fails; primary +.007772 vs ERM; block +.003458 vs X14; primary +.015795 vs ERM but block -.001335 vs X14; reject | docs/experiments/t1_t5_innovations.md |
| X22 — X22 cosine-decayed ERM | fresh constant-LR ERM; sole optimizer/tuning contribution; reliable: +.006035/+.008133/+.003442 in 2021/22/23; reliable: +.021346/+.019504/+.013596 in 2021/22/23 | docs/experiments/t1_t5_innovations.md |
| X23 — X23 impact-consistent BlockDrop specialist | X14 BlockDrop specialist plus fresh ERM; no; closest-control gate fails; +.011199 vs ERM but only +.002400 vs X14; reject; cancelled after T1 attribution failure | docs/experiments/t1_t5_innovations.md |
| X25 — X25 block-specialized severity reliability prompts | X14 specialist plus fresh ERM; inherited D12 mechanism disclosed; no; closest-control gate fails; +.010081 vs ERM but only +.001282 vs X14; reject; smoke host-OOM; formal cancelled after T1 failure | docs/experiments/t1_t5_innovations.md |
| X26 — X26 cosine-anchored impact consistency | cosine FireDrop global-consistency control; X22 cosine ERM adoption; no; frozen closest-control screen fails; +.001555 vs global, +.004877 vs X22; reject; +.003225 vs global, +.002167 vs X22; reject | docs/experiments/t1_t5_innovations.md |
| X27 — X27 forecast-guided hard BlockDrop | cosine random-BlockDrop specialist; X22 cosine ERM adoption; no; closest-control screen fails; +.000465 vs random, +.007683 vs X22; reject; cancelled before start after T1 attribution failure | docs/experiments/t1_t5_innovations.md |
| X29 — X29 valid-context memory attention | cosine rectangular BlockDrop specialist; X22 cosine ERM adoption; no; T5 attention interference; +.000661 vs specialist, +.007878 vs X22; reject; -.007990 vs specialist, -.000656 vs X22; reject | docs/experiments/t1_t5_innovations.md |
| X1+X3 — X1+X3 composition | fresh continuation / component ablations; no; interaction only; +.008200, below X3; reject; cancelled | docs/experiments/t1_t5_innovations.md |
| X2-block-signal — Feature KD, original variant | T1 block AP +0.003103 against fresh continuation although primary -0.005219; repaired distill_block primary -0.013556/block -0.001936. | docs/experiments/t1_t5_innovations.md |
| X8+X17 — X8+X17 complete observable route | Three seeds; all six history/year cells positive; final primary mean +0.013853 vs fresh ERM. | docs/experiments/t1_t5_innovations.md |
| X22+X17 — X22+X17 complete observable route | Three seeds 2021/22/23; primary +0.014642 vs fresh ERM, +0.034527 vs frozen B3/B5; T1 +.010386/+.006153/+.010109, T5 +.028162/+.020169/+.012873 vs ERM. | docs/experiments/t1_t5_innovations.md |
| X17 — severity-factorized block specialists | Three seeds, all six history/year cells positive vs ERM; final primary +.008250. Closest-control heldout attribution fails, so retain as limited total-effect signal. | docs/experiments/t1_t5_innovations.md |
| B2 — FireDrop observable B0/B2 route | M01 gains +.222704/+.124143/+.103482 vs B0. | docs/experiments/quantitative_reliability_ledger.md |
| B3 — Additional BlockDrop | Block mean gains +.040566/+.019331/+.023566 vs B2. | docs/experiments/quantitative_reliability_ledger.md |
| D1-KL — Predictive consistency | Primary gains +.009203/+.013146/+.004561 vs D1-ERM. | docs/experiments/quantitative_reliability_ledger.md |
| D12 — Severity-adaptive reliability prompts | Block mean +.006875/+.008763/+.001654 vs D2-STD; primary +.028574/+.019252/+.014271 vs D1-ERM. | docs/experiments/quantitative_reliability_ledger.md |


## Corrected archived D-series: strict rejection does not erase gains

D4–D8, D10–D11 and D13 are retained as limited positive signals. Their standalone historical interfaces were removed during cleanup; current D12/X implementations are not exact substitutes for every old recipe. Source recovery is from `4b843ad`, preserving the old dependency chain in a namespace.

| ID / implementation | Observation and nearest comparator | Source |
|---|---|---|
| natural-viirs-screen — natural VIIRS screen | attention AP +0.000287, with worse F1/Brier/loss; no forecasting-method gain | docs/experiments/rejected_experiments.md |
| target-qa-censoring — target-QA censoring | P00 AP 0.382293 → 0.398789; 12.54% of all pixels (12.60% of zero-label pixels) lacked reliable target-day observation; useful 24-event case study, outside the method mainline | docs/experiments/rejected_experiments.md |
| D2-RNC — D2-RNC | M06 +0.000764, M07 -0.007200 versus D2-STD; primary -0.003218; M00 -0.009428; 21094930 | docs/experiments/rejected_experiments.md |
| D4 — D4 input token | primary +0.004960; M00 -0.005243 versus D2-STD; missed +0.005 gate by 0.000040; 21103691 | docs/experiments/rejected_experiments.md |
| D5 — D5 CIWC | 0.584364/0.294861/0.361160/0.179137; primary +0.015468 versus D1-ERM, below +0.020; 21111043 | docs/experiments/rejected_experiments.md |
| D6 — D6 CIWC+rank | 0.570395/0.280630/0.355114/0.179208; primary +0.008733; M00 and M06 guardrails failed; 21112470 | docs/experiments/rejected_experiments.md |
| D7 — D7 reliability adapter | 0.579199/0.281434/0.366924/0.185442; primary +0.015015, below +0.020; 21114921 | docs/experiments/rejected_experiments.md |
| D8 — D8 factorized adapter | 0.578203/0.289113/0.361630/0.181149; primary +0.014379, below +0.020; 21116301 | docs/experiments/rejected_experiments.md |
| D10 — D10 prompt pyramid | 0.583725/0.302593/0.374300/0.191693; total +0.026610, but module +0.002915 and below D4; 21120041 | docs/experiments/rejected_experiments.md |
| D11 — D11 complete prompts | 0.583391/0.300108/0.375842/0.191872; total +0.026356; module +0.004928, 0.000072 below gate; 21121610 | docs/experiments/rejected_experiments.md |
| D13 — T5 severity-adaptive prompting | Against D13-STD: block mean +0.002577; standard AP .600267/.357375/.387873/.196784, SARP .602925/.355388/.389202/.200609. Below +.005 gate; no heldout. | docs/experiments/rejected_experiments.md |

## Two distinct September three-direction campaigns

These are different implementations and must not be merged into one result. The underscore-named reliability-fusion report studies two-bank batch-stat normalization, typed shallow branches, and KD; the hyphenated `research/three-directions` report studies four-bank normalization, fixed weight averaging, and a separate KD student. Both have positive KD evidence despite failing their overall compression gates.

| ID / implementation | Observation and nearest comparator | Source |
|---|---|---|
| RF-BN — Missingness-conditional BN | Conditional minus common primary -.008551/-.013704 in T1/T5. T1 M00 +.000452 and M01 +.000973 versus common are isolated metric positives, not robustness gain. | docs/experiments/three_directions_results.md |
| RF-MIXED — Equal-capacity mixed shallow branch | Primary +.003981/+.001866 T1/T5 vs matched control (derived from rounded table). | docs/experiments/three_directions_results.md |
| RF-TYPED — Static/dynamic shallow branches | T1 -.008309 vs mixed/-.004328 vs control; T5 +.005372 vs mixed/+.007238 vs control. | docs/experiments/three_directions_results.md |
| RF-KD — Same-input routed-expert distillation | T1/T5 primary +.003262/+.005630 vs no-KD; -.003851/-.007891 vs teacher; M00 improves both. | docs/experiments/three_directions_results.md |
| TD-W — Fixed 0.5 weight merge | Primary -.033409/-.060303 vs recalibrated X22; T1 block +.003249 but M01 -.106724. | docs/experiments/three-directions-results.md |
| TD-D — Routed-expert KD student | Primary +.001570/+.005707 vs no-KD; vs route +.000710/-.005418; T1 replacement gate passes, T5 M07 -.013862 fails. | docs/experiments/three-directions-results.md |

## Architecture transfer screens

| ID / implementation | Observation and nearest comparator | Source |
|---|---|---|
| ARCH-SegFormer-B2 — SegFormer-B2 | T1 cosine +.007702; T5 +.003001 vs matched ERM. Routed mixed/25%/50% BlockDrop T1 +.001955/-.001070/-.001525, T5 +.004720/+.005466/-.000420. All cross-T gates fail. | docs/experiments/t1_t5_innovations.md |
| ARCH-SwinUnet — SwinUnet | T5 cosine +.003831; mixed/25%/50% BlockDrop +.006723/+.002831/+.003828 vs matched ERM. T1 not resolved in committed ledger; do not infer a negative. | docs/experiments/t1_t5_innovations.md |
| ARCH-ConvLSTM — ConvLSTM | Bootstrap saved, continuation replacement jobs 21362024–21362028 submitted; committed ledger has no final screen values. | docs/experiments/t1_t5_innovations.md |

## Historical signals with invalid or confounded baseline ancestry

These observations are archived so weak or diagnostic positives are not lost. P00/P02 inherited invalid pooled-year exposure; P09 onward repaired continuation indexing but retained old initialization and comparators. Their gains cannot establish a corrected-data causal claim. Rebuilt B2/B3 are the corrected replacements.

| ID / implementation | Observation and nearest comparator | Source |
|---|---|---|
| P00 — P00 FireDrop | M01 0.045920 → 0.299465; useful signal rebuilt correctly as B2 | docs/experiments/rejected_experiments.md |
| P02 — P02 FireDrop+BlockDrop | versus P00: M06 +0.0183, M07 +0.0303, M00 -0.0200, M01 -0.0322; useful signal rebuilt correctly as B3 | docs/experiments/rejected_experiments.md |
| P10 — P10 matched ERM | mean M06/M07 delta versus P00 +0.052345/+0.023389/+0.020189 in 2021/2022/2023; motivation only; rebuilt from scratch in corrected baselines | docs/experiments/rejected_experiments.md |
| P01 — P01 validity channel | M00/M01/M02 decreased; M07 +0.0002; no useful gain | docs/experiments/rejected_experiments.md |
| P03 — P03 hard router | M06/M07 improved in 2021 and 2023 but changed -0.0231/-0.0261 in 2022; temporally unstable | docs/experiments/rejected_experiments.md |
| P13 — P13 FireDrop expert | M01 delta +0.004771/-0.002058/+0.019928 in 2021/2022/2023; mixed temporal result | docs/experiments/rejected_experiments.md |
| P04-P07-block — Residual corrections over legacy P00 | P04/P05/P06/P07 2021 M06/M07 AP .332083/.139913, .327723/.138767, .330240/.141035, .331009/.140771 vs P00 .317005/.129176. All improve this weak reference; P03 remains stronger and P07 variance ~1e-8. | archive/pre-t1-cleanup-2026-09-04:docs/experiments/p00_p06_rapid_reliability.md |

## Code and artifacts

The X-series entrypoint is `python -m reproductions.cross_history.run`, with `--history`, `--method`, `--seed`, `--output`, and optional `--block-fraction`/`--evaluate-only`. Exact method strings are in the inventory. X17 requires distinct `.25`/`.5` specialists and observable routing; it is not one newly learned architecture. Cross-history routes are composed by `compose_routes.py`, `compose_severity_routes.py`, and `compose_complete_routes.py`.

Reliability-fusion uses `python -m reproductions.cross_history.run_three_directions --arm {bn,control,mixed,typed,distill}`. The distinct legacy campaign uses `python -m reproductions.three_directions.run --mode {bn_shared,bn_conditional,merge,x14_shared,student_control,student_distill}` in source `3761e8b`. Required teacher hashes are in `docs/experiments/three_directions_manifest.json` and legacy `reproductions/three_directions/sources.json`, respectively. These manifests initially contain site-specific provenance paths and require relocation.

Corrected B3/B5, statistics, dataset and retained expert checkpoints are external artifacts. Original archive location `/project/6085198/kulbear/wildfire/archive/pre-t1-cleanup-2026-09-04/runs` and `docs/experiments/artifact-archive-manifest.tsv` are recovery evidence, not sufficient portable reproduction instructions. Source inventory does not claim these artifacts have been transferred or every historical recipe has been freshly qualified.

## Additional positive controls and compositions

The frozen register’s column headings say seed 0 but some rows were updated with later confirmation results. X8, X10, X14, X17 and X22 have three-seed 2021/2022/2023 evidence; X19 has three-seed 2021 evidence. Other register entries should not inherit that strength.

| ID | Scope and observation |
|---|---|
| X8 | unchanged global D1 consistency; fresh ERM adoption; incremental FireDrop specialist; incrementally reliable; ERM adoption fails 2022; incrementally reliable and ERM-positive Final three-seed primary deltas vs global D1: T1 +.003979/+.004533/+.004699; T5 +.005834/+.003820/+.005231. Against fresh ERM: T1 +.003841/-.002114/+.002570; T5 +.016981/+.015578/+.010831 (2021/22/23). |
| X10 | fresh continuation with spatial route; yes; typed input restoration optimized by forecast loss; 2022/23 -.001356/-.000347; reject cross-history; 2022/23 +.004783/+.002139; T5-only positive 2021 three-seed routed primary +.001186 T1/+.009652 T5; T1 seed 2 -.004452. Seed-0 +.007381/+.005883. |
| X17 | X14 mixed-severity specialist plus fresh ERM; no; secondary only after heldout attribution failure; ERM-positive all years; X14 2023 block -.000649; ERM-positive all years; X14 2022/23 block -.000898/-.001614 Final three-seed primary vs ERM: T1 +.007392/+.002622/+.006359; T5 +.017792/+.010929/+.004409 for 2021/22/23. |
| GLOBAL-D1 | T5 three-seed primary +.011147/+.011758/+.005600 versus ERM in 2021/22/23; T1 -.000138/-.006647/-.002129; overall +.003265. |
| X8+X10 | Overall primary +.008279 versus fresh ERM; T1 2022 -.003212; mixed secondary system evidence. |
| X11-vs-ERM | T1 seed-0 routed primary +.002519 versus fresh ERM, derived from X10 +.007381 and X11 minus X10 -.004862. Incremental reconstruction itself fails. |

Source: `docs/experiments/t1_t5_innovations.md`, including its global-consistency reuse audit and completed confirmation sections. The X11 total effect is arithmetic on rounded reported deltas and does not reverse its failed X10 attribution.

## Baseline and attribution audit additions

| ID | Observation |
|---|---|
| X16 | X14 specialist plus fresh-control total check; no; restoration loses to X14; -.000452 vs X14; reject; cancelled after T1 attribution failure T1 routed primary +.008348 vs fresh ERM; M06 +.001764 versus X14 even though block mean -.000678. |
| B1 | B1 minus B0 M00/M01/M06/M07: 2021 +.014277/+.054310/+.005778/+.006518; 2022 +.039479/+.041762/+.016716/+.013413; 2023 +.000400/+.010055/-.002538/-.000651. Architecture/history confounded; not a new method. |
| B5 | 2021 AP .557715/.288461/.336086/.158021; block .247054. Compared with B1 .595265/.093697/.329582/.140190, M01/block improve and M00 regresses; not clean dominance. |
| P09-total | Block gain vs P00 +.052243/+.018699/+.022378 in 2021/22/23; matched P10 is +.000102/+.004690/-.002189 relative to P09, so no stable GroupDRO attribution. |
| FOUNDATION-RULES | Test event-macro AP .000531 and .070538; deterministic reference, not learned contribution. |
| FOUNDATION-OFFICIAL | Fold-2 trained AP .554664; twelve released folds .452764 ± .088217; original WSTS contract. |

B1/B5 and P09 source: archived quantitative ledger at `4b843ad`; X16 source: current cross-history ledger. B5’s comparator values here are explicit APs, not a claim of matched history against B0.

## Complete retained-positive index

All 51 inclusive positive entries are listed here; the narrative tables above group selected observations. `positive_signal` means an observed limited gain, not a passed contribution gate.

| ID | Name | Evidence scope |
|---|---|---|
| X1 | X1 context transport | 2021; seed 0 unless explicitly stated |
| X1-adapter | X1 frozen adapter | 2021; seed 0 unless explicitly stated |
| X3 | X3 latent transition | 2021; seed 0 unless explicitly stated |
| X4 | X4 spatial risk weighting | 2021; seed 0 unless explicitly stated |
| X6 | X6 balanced corruption coverage | 2021; seed 0 unless explicitly stated |
| X8 | X8 counterfactual-impact consistency | seeds 0/1/2; T1/T5; 2021/2022/2023; register mixes initial and final findings; final values explicitly identified |
| X10 | X10 forecast-aware dynamic inpainting | seeds 0/1/2; T1/T5; 2021/2022/2023; register mixes initial and final findings; final values explicitly identified |
| X12 | X12 counterfactual FireDrop specialist | 2021; seed 0 unless explicitly stated |
| X14 | X14 BlockDrop specialist continuation | seeds 0/1/2; T1/T5; 2021/2022/2023; register mixes initial and final findings; final values explicitly identified |
| X16 | X16 block-specialized dynamic restoration | 2021; seed 0 unless explicitly stated |
| X18 | X18 block-specialized context transport | 2021; seed 0 unless explicitly stated |
| X19 | X19 severity-conditioned latent adapters | seeds 0/1/2; T1/T5; 2021 confirmation; no heldout success established |
| X22 | X22 cosine-decayed ERM | seeds 0/1/2; T1/T5; 2021/2022/2023; register mixes initial and final findings; final values explicitly identified |
| X23 | X23 impact-consistent BlockDrop specialist | 2021; seed 0 unless explicitly stated |
| X25 | X25 block-specialized severity reliability prompts | 2021; seed 0 unless explicitly stated |
| X26 | X26 cosine-anchored impact consistency | 2021; seed 0 unless explicitly stated |
| X27 | X27 forecast-guided hard BlockDrop | 2021; seed 0 unless explicitly stated |
| X29 | X29 valid-context memory attention | 2021; seed 0 unless explicitly stated |
| X1+X3 | X1+X3 composition | 2021; seed 0 unless explicitly stated |
| X2-block-signal | Feature KD, original variant | 2021; seed 0 unless explicitly stated |
| X8+X17 | X8+X17 complete observable route | seeds 0/1/2; T1/T5; 2021/2022/2023 |
| X22+X17 | X22+X17 complete observable route | seeds 0/1/2; T1/T5; 2021/2022/2023 |
| natural-viirs-screen | natural VIIRS screen | T1 unless explicitly T5; 2021 screen unless years stated; inherited legacy P initialization is not a corrected baseline |
| target-qa-censoring | target-QA censoring | T1 unless explicitly T5; 2021 screen unless years stated; inherited legacy P initialization is not a corrected baseline |
| D2-RNC | D2-RNC | T1 unless explicitly T5; 2021 screen unless years stated; inherited legacy P initialization is not a corrected baseline |
| D4 | D4 input token | T1 unless explicitly T5; 2021 screen unless years stated; inherited legacy P initialization is not a corrected baseline |
| D5 | D5 CIWC | T1 unless explicitly T5; 2021 screen unless years stated; inherited legacy P initialization is not a corrected baseline |
| D6 | D6 CIWC+rank | T1 unless explicitly T5; 2021 screen unless years stated; inherited legacy P initialization is not a corrected baseline |
| D7 | D7 reliability adapter | T1 unless explicitly T5; 2021 screen unless years stated; inherited legacy P initialization is not a corrected baseline |
| D8 | D8 factorized adapter | T1 unless explicitly T5; 2021 screen unless years stated; inherited legacy P initialization is not a corrected baseline |
| D10 | D10 prompt pyramid | T1 unless explicitly T5; 2021 screen unless years stated; inherited legacy P initialization is not a corrected baseline |
| D11 | D11 complete prompts | T1 unless explicitly T5; 2021 screen unless years stated; inherited legacy P initialization is not a corrected baseline |
| D13 | T5 severity-adaptive prompting | T5; 2021; one screen |
| B2 | FireDrop observable B0/B2 route | T1; 2021/2022/2023; 3181/2856/2102 samples; historical one-run chain |
| B3 | Additional BlockDrop | T1; 2021/2022/2023; 3181/2856/2102 samples; historical one-run chain |
| D1-KL | Predictive consistency | T1; 2021/2022/2023; 3181/2856/2102 samples; historical one-run chain |
| D12 | Severity-adaptive reliability prompts | T1; 2021/2022/2023; 3181/2856/2102 samples; historical one-run chain |
| RF-BN | Missingness-conditional BN | 2021; seed 0; 3181 samples/scenario; source e7db913 learning or 7a58b38 revised BN |
| RF-MIXED | Equal-capacity mixed shallow branch | 2021; seed 0; 3181 samples/scenario; source e7db913 learning or 7a58b38 revised BN |
| RF-TYPED | Static/dynamic shallow branches | 2021; seed 0; 3181 samples/scenario; source e7db913 learning or 7a58b38 revised BN |
| RF-KD | Same-input routed-expert distillation | 2021; seed 0; 3181 samples/scenario; source e7db913 learning or 7a58b38 revised BN |
| TD-W | Fixed 0.5 weight merge | 2021; seed 0; 3181 samples/scenario; formal source e5593d3 |
| TD-D | Routed-expert KD student | 2021; seed 0; 3181 samples/scenario; formal source e5593d3 |
| ARCH-SegFormer-B2 | SegFormer-B2 | 2021; seed 0; no seed confirmation or heldout transfer established |
| ARCH-SwinUnet | SwinUnet | 2021; seed 0; no seed confirmation or heldout transfer established |
| GLOBAL-D1 | Global predictive consistency | seeds 0/1/2; T1/T5; 2021/2022/2023; register mixes initial and final findings; final values explicitly identified |
| X8+X10 | Impact + restoration route | seeds 0/1/2; T1/T5; 2021/2022/2023; register mixes initial and final findings; final values explicitly identified |
| X11-vs-ERM | Reconstruction-supervised restoration total effect | T1 2021 seed 0; arithmetic derived from reported rounded deltas |
| B1 | Corrected clean T5 temporal baseline | As explicitly stated; historical source contract |
| B5 | Corrected T5 FireDrop+BlockDrop foundation | As explicitly stated; historical source contract |
| X17 | X17 severity-factorized block specialists | seeds 0/1/2; T1/T5; 2021/2022/2023; register mixes initial and final findings; final values explicitly identified |
