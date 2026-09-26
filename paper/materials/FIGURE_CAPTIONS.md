# Figure captions

Use these captions with the generated figures bearing the corresponding file
names. All numeric effects are absolute AP differences unless an axis explicitly
states otherwise. T1 and T5 are distinct architecture/feature/history settings.
Points represent continuation seeds, not independent wildfire events.

## Figure 1 — Protocol and component roles

**Controlled missingness protocol and system composition.** Training and
normalization use 2016–2020; 2021 supports validation and selection, while
2022/2023 provide historical reporting. T1 and T5 share target dates and
evaluation populations but use different architectures and feature sets.
The current ERM-clean composition uses fresh ERM for complete inputs (M00),
cosine ERM/X22 for FireDrop (M01), and fixed-severity X17 experts for the two
block conditions (M06/M07). The mixed-severity X14 expert supplies the closest
specialization control. Arrows represent component selection in the experiment
definition; the reported complete-system scores are composed from recorded
scenario evaluations and do not establish a newly executed inference router.

File: `figures/fig01_protocol_and_composition.pdf`.
Placement: problem setup or method overview.

## Figure 2 — Paired primary effects

**Primary effects relative to matched fresh ERM.** Primary AP averages the
FireDrop and two BlockDrop conditions. Each setting/year cell shows all three
paired continuation seeds and their mean; error bars, where displayed, are
sample standard deviations (`ddof=1`). The 2021 selection results are separated
from historical 2022/2023 reporting. X22+X17 has a positive mean effect in all
six cells, with an overall mean of 0.014642 AP. This average gives equal weight
to the two settings, three years, and three seeds. It remains below the
historical 0.020 magnitude target against fresh ERM. Seed variation is not an
event-level confidence interval. Historical and ERM-clean X22+X17 compositions
have the same primary values because M00 is excluded.

File: `figures/fig02_primary_gain.pdf`.
Placement: main quantitative results, beside the complete scenario table.

## Figure 3 — The value and limits of severity factorization

**Fixed-severity X17 versus mixed-severity X14.** Paired block-AP differences
isolate the extra severity split within the specialist training recipe; block
AP averages M06 and M07. Points show the three matched seeds in each
setting/year cell and the summary marks show their means. Positive total
effects over ERM do not imply positive attribution against X14. The mean
factorization effect is negative in T1/2023, T5/2022, and T5/2023. These results
limit the claim that two fixed-severity experts are preferable to a single
mixed-severity specialist. Historical years have already been inspected and
are not new independent confirmation.

File: `figures/fig03_severity_attribution.pdf`.
Placement: main ablation or attribution analysis; retain negative cells.

## Figure 4 — Scenario-specific effects

**Effects by observation condition.** Entries are mean AP differences from
fresh ERM across the three matched seeds, shown separately for complete inputs
(M00), FireDrop (M01), and the two BlockDrop conditions (M06/M07). A common
zero-centered color scale preserves the sign and comparability of effects.
The displayed current X22+X17 composition uses fresh ERM for M00, so its clean
effect is exactly zero by construction. The preserved historical composition
instead uses X22 for M00 and is a separate record. Primary scores omit M00;
the heatmap exposes scenario tradeoffs hidden by an aggregate score. T1/T5
differences cannot be attributed to history alone.

File: `figures/fig04_scenario_effects.pdf`.
Placement: main results or supplementary breakdown if space is constrained.

## Figure 5 — Clean-input roles and resource accounting

**Composition definition and nominal continuation cost.** Panel (a) reports
mean primary AP gains over fresh ERM against the declared continuation steps
and number of retained component models per setting/seed. Each continuation
uses 3000 steps. The current ERM-clean system retains four component models,
whereas the historical X22-clean system and the X22+X14 comparison each retain
three. Panel (b) shows that the historical and current X22+X17 variants have
identical primary and block effects but different clean-input effects. These
means give equal weight to the six setting/year cells and their three seeds;
heterogeneous cells are not treated as independent replicates for a pooled
uncertainty interval. Cost counts exclude shared foundation training,
exploration, evaluation, and deployment overhead. They are not measurements of
GPU-hours, latency, FLOPs, or memory, and do not establish equal total compute
with a single-model comparator.

File: `figures/fig05_comparator_and_budget.pdf`.
Placement: discussion of composition definition and system cost. The separate
frozen-reference effects are provided in `tables/frozen_reference.csv`.

## Figure 6 — Archived attribution in the corrected T1 chain

**Historical effects with the original controls.** The archived corrected T1
chain compares predictive consistency (D1-KL) with its ERM control, and
severity-adaptive prompts (D12) with both the standard prompt control and the
historical ERM reference. Each comparison retains its named control; a larger
total effect over ERM does not establish the same magnitude of incremental
prompt attribution. Values come from one historical continuation per method,
reported separately for 2021, 2022, and 2023. No seed uncertainty is estimated.
These findings provide supporting research context and remain outside the
selected X22+X17 mainline; they are not additional three-seed confirmation.

File: `figures/fig06_historical_attribution.pdf`.
Placement: supplementary historical evidence, with the source chain identified.

## Supplementary evidence catalogue

**Retained research evidence and its scope.** The catalogue preserves
contribution-relevant controls and useful archived observations together with
their comparator, settings, seed/year coverage, and current status. Positive
signals remain distinguishable from confirmed comparisons, invalid ancestry,
implementation faults, cancelled runs, and unrun ideas. Inclusion does not
reactivate an archived direction or imply a new efficacy reproduction.

Source: `tables/supplementary_catalogue.csv`.
Placement: supplementary evidence table; use the machine-readable catalogue
when the full list is too long for a page layout.
