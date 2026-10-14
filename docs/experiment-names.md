# Experiment names

Use a descriptive mechanism name first and retain the original experiment ID in parentheses, for example **Cosine-schedule continuation (X22)**. IDs remain the stable keys for configurations, artifacts, and evidence. These labels change presentation only; they do not rename code, change a recipe, or revise any scientific status.

The machine-readable catalog is [experiment-names.json](../site-content/en/experiment-names.json). It covers every method-inventory record plus explicit controls, the official Fold 2 reference, and the four core evaluation scenarios. English names are the public labels; Chinese translations are available in the JSON for explanation and lookup.

Names describe an intervention, comparator, or recorded investigation rather than promise effectiveness. Prefixes such as “legacy,” “T1,” and “cross-history” distinguish similar mechanisms from separate campaigns. Some historical inventory rows group probes, data preparation, or total-effect comparisons; their explanations preserve that scope rather than invent a single algorithm. The catalog's broad `method` kind includes those investigation records. `scenario` is reserved for input conditions, `control` identifies comparators, `baseline` identifies reference recipes, and `system` identifies compositions or routing systems.

“Fresh ERM” means a newly executed matched continuation from B3 or B5, not a randomly initialized model. T1 and T5 differ in architecture and inputs as well as history. Official Fold 2 and project B0 use different experimental contracts. The condition-specific specialist route has both a preserved historical clean-input choice and a current ERM-clean variant; its name does not collapse those evidence differences. Consult the source record for the precise variant and its status.

For archive-only source references that are not files in this checkout, the source link points to the existing method inventory, which preserves the original branch-qualified provenance. A source link is evidence provenance, not a claim that the experiment ran successfully or is currently maintained.

| ID | Descriptive name | Kind | Meaning and source |
| --- | --- | --- | --- |
| `X1` | Valid-context feature transport | method | Moves pooled valid-region feature information into missing regions through learned residuals. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X1-adapter` | Frozen-backbone context adapter | method | Trains the context adapter while retaining the initial foundation backbone. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X2` | Fire-weighted feature distillation | method | Matches student features to a frozen clean-input teacher with spatial and fire-target weighting. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X3` | Latent fire-transition head | method | Combines occupancy, continuation, and new-activity predictions into a next-day forecast. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X4` | Spatial-risk loss weighting | method | Reweights the forecasting objective spatially under controlled missingness. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X5` | Missing-region consistency | method | Localizes predictive consistency instead of applying the global consistency objective. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X6` | Balanced corruption sampling | method | Changes training corruption coverage through balanced corruption sampling. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X7` | Missingness-conditioned residual experts | method | Proposes residual experts conditioned on the observed missingness pattern. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X8` | Impact-weighted predictive consistency | method | Weights clean-to-corrupted prediction consistency by counterfactual corruption impact. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X9` | Spatial-impact feature modulation | method | Uses spatial context to modulate decoder features with a FiLM mechanism. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X10` | Forecast-trained dynamic inpainting | method | Restores missing dynamic inputs using the forecasting objective. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X11` | Reconstruction-supervised dynamic inpainting | method | Adds reconstruction supervision to the forecast-trained restoration mechanism. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X12` | Counterfactual FireDrop specialist | method | Specializes continuation training for complete fire-history removal using counterfactual consistency. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X13` | Normalized diffusion inpainting | method | Propagates observed spatial values into missing regions without learned restoration parameters. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X14` | Mixed-severity block specialist | method | Trains one block-only specialist with both 25% and 50% missing-block severities. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X15` | Distance-to-observation prompts | method | Injects encoder prompts scaled by depth inside a missing spatial region. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X16` | Block-trained dynamic inpainting | method | Trains the dynamic restoration module under the block-only specialist schedule. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X17` | Fixed-severity block specialists | method | Trains separate 25% and 50% block-only specialists and selects them by observed missing fraction. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X18` | Block-trained context transport | method | Adds valid-context transport to the block-specialist training recipe. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X19` | Severity-conditioned latent adapters | method | Conditions latent feature adapters on block-missingness severity. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X20` | ERM-anchored impact consistency | method | Anchors the impact-consistency recipe to the matched ERM continuation objective. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X21` | First-layer reliability calibration | method | Proposes reliability-dependent calibration at the first model layer. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X22` | Cosine-schedule continuation | method | Continues ERM training with cosine learning-rate decay instead of the matched constant-rate schedule. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X23` | Impact-consistent block specialist | method | Adds impact-weighted consistency to the block-specialist recipe. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X24` | Frozen-clean FireDrop distillation | method | Proposes a frozen clean-input teacher for FireDrop-specialist distillation. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X25` | Block-specialist reliability prompts | method | Adds severity-adaptive reliability prompts to block-specialist training. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X26` | Cosine-scheduled impact consistency | method | Combines cosine-scheduled continuation with impact-weighted FireDrop consistency. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X27` | Forecast-guided block selection | method | Selects hard missing blocks using forecast information during specialist training. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X28` | VIIRS-footprint missingness augmentation | method | Uses VIIRS reliability-footprint geometry for training corruption instead of rectangular blocks. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X29` | Valid-context memory attention | method | Retrieves valid-context information through attention during block-specialist training. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X30` | Training-only fire-change heads | method | Adds auxiliary fire-change prediction heads during training. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X31` | Multiscale forecast supervision | method | Adds forecasting supervision at multiple feature scales. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X1+X3` | Context-and-transition composition | system | Combines context transport with the latent fire-transition head. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X2-block-signal` | Original feature-distillation block analysis | method | Records the original feature-distillation variant's block results; it is not a separate new mechanism. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X3-decoupled` | Decoupled fire-transition head | method | Changes the coupling in the latent-transition prediction mechanism. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X4-strong` | Fivefold spatial-risk weighting | method | Increases the spatial-risk weighting strength in a repair variant. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X2-block-repair` | Block-local feature distillation | method | Restricts feature distillation to the block-local repair variant. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X8+X17` | Impact-and-block specialist route | system | Routes fire-history loss to impact consistency and spatial loss to fixed-severity block specialists. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X22+X17` | Condition-specific specialist route | system | Combines cosine continuation for FireDrop with fixed-severity block specialists; the historical and current clean-input routes differ. [Source](../docs/experiments/t1_t5_innovations.md) |
| `legacy-c00-c02-3k` | Legacy 3000-step foundation comparison | baseline | Records the legacy C00/C02 foundation runs with an invalid pooled-year resolver. [Source](../docs/experiments/rejected_experiments.md) |
| `legacy-c00-c02-six-10k-runs` | Legacy six-run 10000-step foundation comparison | baseline | Records the six legacy C00/C02 runs with invalid training exposure. [Source](../docs/experiments/rejected_experiments.md) |
| `P00` | Legacy FireDrop training | method | Trains with complete fire-history removal in the legacy experiment family. [Source](../docs/experiments/rejected_experiments.md) |
| `P02` | Legacy FireDrop-and-BlockDrop training | method | Adds spatial block corruption to the legacy FireDrop recipe. [Source](../docs/experiments/rejected_experiments.md) |
| `P09` | Legacy group-robust optimization | method | Uses GroupDRO in the legacy continuation comparison. [Source](../docs/experiments/rejected_experiments.md) |
| `P10` | Legacy matched ERM continuation | control | Provides the matched ERM continuation control for legacy GroupDRO. [Source](../docs/experiments/rejected_experiments.md) |
| `P01` | Legacy validity-channel input | method | Adds an input validity channel in the legacy model family. [Source](../docs/experiments/rejected_experiments.md) |
| `P03` | Legacy hard expert router | system | Uses hard routing in the legacy specialist comparison. [Source](../docs/experiments/rejected_experiments.md) |
| `P04` | Legacy pointwise residual correction | method | Adds a 1-by-1 residual correction to the legacy predictor. [Source](../docs/experiments/rejected_experiments.md) |
| `P05` | Legacy spatial residual correction | method | Adds a 3-by-3 residual correction to the legacy predictor. [Source](../docs/experiments/rejected_experiments.md) |
| `P06` | Legacy decoder adaptation | method | Adapts the decoder in the legacy correction family. [Source](../docs/experiments/rejected_experiments.md) |
| `P07` | Legacy stochastic residual correction | method | Uses a stochastic residual correction in the legacy predictor. [Source](../docs/experiments/rejected_experiments.md) |
| `P08` | Legacy teacher-posterior correction | method | Uses teacher-posterior information in the legacy correction family. [Source](../docs/experiments/rejected_experiments.md) |
| `P11` | Legacy missingness-pattern router | system | Routes predictions according to missingness in the legacy FireDrop comparison. [Source](../docs/experiments/rejected_experiments.md) |
| `P12` | Prevalence-weighted focal loss | method | Sets focal-loss alpha from prevalence in the legacy probe. [Source](../docs/experiments/rejected_experiments.md) |
| `P13` | Legacy FireDrop specialist | method | Evaluates a dedicated FireDrop expert in the legacy family. [Source](../docs/experiments/rejected_experiments.md) |
| `belief-filter-t-1` | T1 belief-state filter | method | Uses belief-state filtering in the T1 legacy probe. [Source](../docs/experiments/rejected_experiments.md) |
| `belief-attention-t-1` | T1 belief-state attention | method | Uses attention over belief-state information in the T1 legacy probe. [Source](../docs/experiments/rejected_experiments.md) |
| `belief-reconstruction-t-1` | T1 belief-state reconstruction | method | Uses belief-state reconstruction in the T1 legacy probe. [Source](../docs/experiments/rejected_experiments.md) |
| `corresponding-t-5-probes` | T5 belief-state probe suite | method | Groups the corresponding T5 belief-state probes; it is a suite rather than one mechanism. [Source](../docs/experiments/rejected_experiments.md) |
| `natural-viirs-screen` | Natural VIIRS observation screen | method | Screens prediction behavior on naturally varying VIIRS observations. [Source](../docs/experiments/rejected_experiments.md) |
| `target-qa-censoring` | Target-observation quality censoring | method | Examines evaluation after censoring targets without reliable target-day observations. [Source](../docs/experiments/rejected_experiments.md) |
| `2016-2020-qa-cohort` | Multiyear observation-quality cohort preparation | method | Attempts to prepare the 2016–2020 observation-quality cohort; this is a data-preparation record. [Source](../docs/experiments/rejected_experiments.md) |
| `B4` | Equal-year training sampler | baseline | Samples training years equally in a corrected-baseline comparison. [Source](../docs/experiments/rejected_experiments.md) |
| `D2-RNC` | Reliability-normalized convolution | method | Changes convolution normalization according to reliability information. [Source](../docs/experiments/rejected_experiments.md) |
| `D3` | Temporal reliability fusion | method | Proposes temporal fusion using reliability information. [Source](../docs/experiments/rejected_experiments.md) |
| `D4` | Input reliability token | method | Adds a reliability token at the model input. [Source](../docs/experiments/rejected_experiments.md) |
| `D5` | T1 counterfactual-impact consistency | method | Applies counterfactual-impact-weighted consistency in the corrected T1 campaign. [Source](../docs/experiments/rejected_experiments.md) |
| `D6` | T1 impact-and-rank consistency | method | Adds rank consistency to the corrected T1 impact-consistency recipe. [Source](../docs/experiments/rejected_experiments.md) |
| `D7` | Counterfactual reliability adapter | method | Adds a reliability adapter in the corrected T1 campaign. [Source](../docs/experiments/rejected_experiments.md) |
| `D8` | Factorized reliability adapter | method | Uses a factorized variant of the reliability adapter. [Source](../docs/experiments/rejected_experiments.md) |
| `D9` | Error-pair ranking objective | method | Adds an error-pair ranking objective to the corrected T1 comparison. [Source](../docs/experiments/rejected_experiments.md) |
| `D10` | Reliability prompt pyramid | method | Injects reliability prompts across multiple model scales. [Source](../docs/experiments/rejected_experiments.md) |
| `D11` | Complete-variant reliability prompt pyramid | method | Uses the archived complete-prompt-pyramid variant, distinct from the D10 recipe. [Source](../docs/experiments/rejected_experiments.md) |
| `D13` | T5 severity-adaptive reliability prompts | method | Transfers severity-adaptive reliability prompting to the corrected T5 foundation. [Source](../docs/experiments/rejected_experiments.md) |
| `B0` | Corrected T1 clean-input baseline | baseline | Trains the corrected project T1 baseline on clean inputs under the project contract. [Source](../docs/experiments/quantitative_reliability_ledger.md) |
| `B2` | Corrected T1 FireDrop-trained baseline | baseline | Uses stochastic FireDrop during corrected T1 baseline training; its reported specialist contribution selects B2 for FireDrop and B0 elsewhere. [Source](../docs/experiments/quantitative_reliability_ledger.md) |
| `B3` | T1 missingness-trained foundation | baseline | Adds BlockDrop to corrected T1 FireDrop training and supplies the T1 continuation foundation. [Source](../docs/experiments/quantitative_reliability_ledger.md) |
| `D1-KL` | T1 global predictive consistency | method | Adds global prediction consistency in the corrected T1 campaign against D1-ERM. [Source](../docs/experiments/quantitative_reliability_ledger.md) |
| `D2-STD` | Standard reliability control | control | Provides the matched standard-reliability recipe for module attribution. [Source](../docs/experiments/quantitative_reliability_ledger.md) |
| `D12` | T1 severity-adaptive reliability prompts | method | Adds severity-adaptive reliability prompts to the corrected T1 standard-reliability recipe. [Source](../docs/experiments/quantitative_reliability_ledger.md) |
| `RF-BN` | Batch-stat missingness normalization | method | Calibrates missingness-conditional batch normalization using the corrected batch-stat diagnostic. [Source](../docs/experiments/three_directions_results.md) |
| `RF-BN-fixed` | Fixed-forward missingness normalization | method | Records the original fixed-forward conditional-normalization diagnostic, later superseded. [Source](../docs/experiments/three_directions_results.md) |
| `RF-MIXED` | Mixed-input shallow residual branch | control | Processes static and dynamic inputs jointly in an equal-capacity shallow residual branch. [Source](../docs/experiments/three_directions_results.md) |
| `RF-TYPED` | Separate static-and-dynamic residual branches | method | Separates static and dynamic shallow processing against the equal-capacity mixed-input control. [Source](../docs/experiments/three_directions_results.md) |
| `RF-KD` | Reliability-fusion same-input expert distillation | method | Distills a routed teacher using the same corrupted input for teacher and student in the reliability-fusion campaign. [Source](../docs/experiments/three_directions_results.md) |
| `TD-N` | Legacy four-bank normalization | method | Separates batch-normalization statistics into four condition banks in the earlier three-directions campaign. [Source](../docs/experiments/three-directions-results.md) |
| `TD-W` | Half-and-half expert weight merge | method | Averages generalist and block-specialist model weights with a fixed 0.5 coefficient. [Source](../docs/experiments/three-directions-results.md) |
| `TD-D` | Legacy routed-expert distillation | method | Distills the expert route into one student in the earlier three-directions campaign. [Source](../docs/experiments/three-directions-results.md) |
| `ARCH-SegFormer-B2` | SegFormer-B2 recipe transfer | method | Tests continuation and specialist recipes with the SegFormer-B2 architecture. [Source](../docs/experiments/t1_t5_innovations.md) |
| `ARCH-SwinUnet` | Swin-Unet recipe transfer | method | Tests continuation and specialist recipes with the Swin-Unet architecture. [Source](../docs/experiments/t1_t5_innovations.md) |
| `ARCH-ConvLSTM` | ConvLSTM recipe transfer | method | Records the architecture-transfer campaign for ConvLSTM without implying completed screen results. [Source](../docs/experiments/t1_t5_innovations.md) |
| `P04-P07-block` | Legacy residual-correction block comparison | method | Groups legacy residual variants compared with P00 on block-missingness scenarios. [Source](../docs/research/method-inventory.json) |
| `GLOBAL-D1` | Cross-history global predictive consistency | control | Provides global predictive consistency as the cross-history control for localized or impact-weighted variants. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X8+X10` | Impact-and-inpainting route | system | Combines impact consistency for FireDrop with dynamic inpainting for spatial missingness. [Source](../docs/experiments/t1_t5_innovations.md) |
| `X11-vs-ERM` | Reconstruction-inpainting total-effect comparison | method | Compares reconstruction-supervised restoration with fresh ERM to measure the full recipe rather than isolate reconstruction. [Source](../docs/experiments/t1_t5_innovations.md) |
| `B1` | Corrected T5 clean-input baseline | baseline | Provides the corrected clean T5 reference; architecture and features also differ from T1. [Source](../docs/research/method-inventory.json) |
| `B5` | T5 missingness-trained foundation | baseline | Uses corrected T5 FireDrop-and-BlockDrop training as the T5 continuation foundation. [Source](../docs/research/method-inventory.json) |
| `P09-total` | Legacy group-robust total-effect comparison | method | Compares GroupDRO plus continuation with P00 without isolating the GroupDRO contribution. [Source](../docs/research/method-inventory.json) |
| `FOUNDATION-RULES` | No-fire and persistence references | baseline | Groups deterministic no-fire and persistence references rather than a learned method. [Source](../docs/research/method-inventory.json) |
| `FOUNDATION-OFFICIAL` | Official WSTS reproduction reference | baseline | Records the official Res18 U-Net reproduction and released-weight reference under the original WSTS contract. [Source](../docs/research/method-inventory.json) |
| `control` | Matched ERM continuation | control | Continues from the matching B3 or B5 foundation with the fixed constant-learning-rate ERM recipe. Fresh means a newly run continuation, not training from random initialization. [Source](../docs/experiments/t1_t5_innovations.md) |
| `D1-ERM` | Corrected T1 consistency control | control | Provides the corrected T1 ERM comparator for D1-KL; this historical control is distinct from the later cross-history fresh continuation. [Source](../docs/experiments/quantitative_reliability_ledger.md) |
| `official-fold-2` | Official WSTS Fold 2 reproduction | baseline | Uses the official Fold 2 data, split, and training contract; its score cannot isolate a method gain over corrected project B0. [Source](../reproductions/wsts_res18_unet_t1/README.md) |
| `M00` | Complete-input scenario | scenario | Evaluates inputs without the controlled missingness intervention. [Source](../docs/experiments/quantitative_reliability_ledger.md) |
| `M01` | Fire-history-missing scenario | scenario | Removes active-fire history from the input; this is an evaluation condition, not an algorithm. [Source](../docs/experiments/quantitative_reliability_ledger.md) |
| `M06` | Quarter-block-missing scenario | scenario | Applies structured missing blocks over 25% of the dynamic input. [Source](../docs/experiments/quantitative_reliability_ledger.md) |
| `M07` | Half-block-missing scenario | scenario | Applies structured missing blocks over 50% of the dynamic input. [Source](../docs/experiments/quantitative_reliability_ledger.md) |
