# Wildfire forecasting: paper materials

This author working package organizes the retained X22+X17 experiment evidence,
its necessary controls, and relevant archived findings for a computer-vision
paper. It includes editable figures, machine-readable results, LaTeX tables,
source records, and draft experimental prose. It is not an anonymized submission
or a claim that the method has already met a venue's scientific acceptance bar.

## Start with the evidence boundary

The main question is whether a simple combination of training schedule and
missingness-specific specialists improves next-day active-fire prediction beyond
a matched ERM continuation. X22 is cosine-scheduled ERM; X17 uses two separately
trained BlockDrop experts for the fixed 25% and 50% missingness conditions. The
closest specialization control is mixed-severity X14.

The source ledger reports a mean primary gain of **0.014642 AP** for X22+X17
against fresh ERM, and **0.034527 AP** against frozen B3/B5. These are different
comparisons. The former remains below the campaign's historical 0.020 magnitude
target. All differences are absolute AP units; 0.01 AP is one percentage point.

The newly composed X22+mixed-X14 control gains **0.014287 AP** over fresh ERM.
X22+X17 exceeds it by only **0.000355 AP** overall and falls behind by
**0.000226 AP** on the historical 2022/2023 average, despite retaining one more
model. The contribution discussion must preserve this closest-control result.

**The material audit found a clean-scenario discrepancy.** The original complete
route JSON uses X22 for M00, whereas the current documented mainline uses fresh
ERM for M00. The package preserves the original as
`X22_X17_HISTORICAL` and separately derives `X22_X17_ERM_CLEAN` from the recorded
component summaries. Their M01/M06/M07 scores, primary score, and block score are
identical; their clean score is not. A zero M00 change in the ERM-clean variant is
an identity from reusing ERM, not an empirical clean-preservation result for the
experts. Neither variant represents a newly executed end-to-end router.

Read [Claims and limitations](CLAIMS_AND_LIMITATIONS.md) before copying any
number or sentence into a manuscript.

## Package contents

| Location | Use |
|---|---|
| `figures/fig01_protocol_and_composition.*` | Protocol, component roles, and the boundary between summary composition and inference |
| `figures/fig02_primary_gain.*` | Paired primary effects, with all three seeds shown in each setting/year |
| `figures/fig03_severity_attribution.*` | Fixed-severity X17 versus mixed-severity X14 |
| `figures/fig04_scenario_effects.*` | Clean, FireDrop, and BlockDrop effects shown separately |
| `figures/fig05_comparator_and_budget.*` | Historical/current clean-input roles and nominal continuation/model counts |
| `figures/fig06_historical_attribution.*` | Archived D-series effects with their own matched comparators |
| `tables/main_results.csv` and `.tex` | Scenario and aggregate scores for the principal comparisons |
| `tables/paired_effects.csv` | Matched method-minus-control effects |
| `tables/attribution.csv` | Severity-factorization attribution |
| `tables/system_attribution.csv` | Current X22+X17 minus X22+mixed-X14, by setting/year/metric |
| `tables/overall_effects.csv` | Equal-weight overall effects, with selection and historical years also separated |
| `tables/frozen_reference.csv` | Separate B3/B5-reference effects; the same frozen checkpoint is reused across continuation seeds |
| `tables/costs.csv` | Declared continuation budget and number of retained component models |
| `tables/historical_chain_effects.csv` | D1 and D12 effects against their original historical controls |
| `tables/all_normalized_results.csv` | Mainline and recovered archival scores, retaining their distinct evidence scopes |
| `tables/archive_comparison_rows.csv` | Original archived comparisons and raw rows; comparator details remain in each source payload |
| `tables/supplementary_catalogue.csv` | Archived evidence, scope, comparator, and current status |
| `data/scenario_rows.csv` and `data/evidence.json` | Auditable numeric records underlying the figures and tables |
| `sources/` | Copied original summaries and supporting repository records |
| `report.pdf` | Offline inspection copy of the assembled materials |
| [EXPERIMENTS_DRAFT.md](EXPERIMENTS_DRAFT.md) | Grounded English experiment prose |
| [FIGURE_CAPTIONS.md](FIGURE_CAPTIONS.md) | Figure captions and placement notes |

PDF is the preferred figure format for LaTeX; SVG is editable and PNG supports
quick review. Figures use the same method names, units, and comparator definitions
as their underlying tables. Consult the generated validation record and source
manifest for what was checked during this build. Presence in a source catalogue
does not establish that an archived experiment was reproduced anew.

## Reporting conventions

AP is computed over all evaluated pixels within one scenario and year. Primary
AP is the arithmetic mean of M01, M06, and M07 AP; block AP is the mean of M06 and
M07. The clean score M00 is reported separately. These scenario means are not AP
computed by pooling the predictions of several scenarios.

Results are paired by setting, year, and continuation seed. Per-cell summaries
use seeds 0, 1, and 2; variability is sample standard deviation with `ddof=1`.
Seed points describe continuation variability. They are not independent events,
independent datasets, or confidence intervals for operational performance.
An overall mean gives equal weight to the two settings, three years, and three
seeds. The figure panels distinguish 2021 selection/validation from historical
2022/2023 reporting; the overall average should not conceal that distinction.
These three-seed conventions apply to the mainline. Archived single-run results
retain their smaller scope and do not receive invented error bars. The three
comparisons to a frozen checkpoint reuse one fixed reference per setting;
they do not create three independently trained frozen baselines.

T1 is Res18-U-Net with 40 input features. T5 is Res18-UTAE with 33 features per
day. Comparing them does not isolate the causal effect of history length.
The prediction target is a next-day VIIRS active-fire proxy, not a fire perimeter.

## Suggested manuscript use

Use the protocol and complete comparison table to define the problem before
presenting gains. Lead with the comparison against fresh ERM and keep the frozen
reference result secondary. Show the X17-versus-X14 attribution even where it is
negative. The newly derived X22+X14 system control belongs beside X22+X17; it uses
existing scores and adds no training. Its interpretation must follow its actual
values rather than an assumption that fixed-severity experts are superior.

Keep archived positives in a supplementary evidence catalogue unless a specific
comparison supports the paper's selected question. The retained archive includes
limited positives, failed closest-control tests, invalid comparisons, cancelled
runs, and unrun ideas; those states must remain distinct.

The bundle contains local provenance paths and project identifiers. Before a
blind submission, make a separate review copy, remove identifying metadata, and
recheck venue instructions. Preserve this author copy as the audit trail.
