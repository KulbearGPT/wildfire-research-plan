# Claims, evidence, and limits

The proposed paper studies next-day active-fire forecasting under controlled
missing observations. Its strongest current story combines a reproducible
comparison protocol, a simple condition-specific system, and evidence on when
specialization helps. The current results do not establish a novel backbone,
general operational robustness, or superiority to all relevant prior methods.

## Proposed contribution list

1. **A controlled evaluation and comparison protocol.** The study aligns target
   dates and evaluation populations across two existing forecasting settings,
   separates missingness conditions, and compares continuations against matched
   controls. This is a reproducibility and evaluation contribution. Data repairs
   and bookkeeping should not be presented as separate algorithmic inventions.

2. **A simple condition-specific forecasting system.** X22 combines ordinary
   ERM continuation with cosine decay, while X17 provides two fixed-severity
   BlockDrop experts. The current system uses ERM for complete inputs, X22 for
   FireDrop, and the corresponding X17 expert for each fixed block condition.
   Existing scores support a positive system effect relative to fresh ERM.
   The evidence currently comes from scenario-summary composition, not a fresh
   end-to-end inference test of a router.

3. **An empirical separation of training, system, and attribution effects.**
   Comparisons across seeds and historical years distinguish improvements over
   frozen initialization from improvements over continued ERM. X17-versus-X14
   comparisons expose limits of severity factorization despite positive total
   effects. These findings support a bounded empirical contribution; they do not
   convert a failed attribution test into a successful mechanism claim.

## Quantitative statements supported by the campaign ledger

The values below are the ledger's six-decimal reporting, not newly invented
precision. Machine-readable source-derived tables accompany the package.

| Statement | Existing evidence | Permitted interpretation |
|---|---|---|
| X22 improves mean primary AP by 0.012009 over fresh ERM | Two settings, three years, three matched seeds | Cosine-schedule continuation has a positive average effect in these settings |
| X22+X17 improves mean primary AP by 0.014642 over fresh ERM | Same 18 setting/year/seed cells | Positive system effect; below the historical 0.020 target |
| The same missingness composition gains 0.034527 over frozen B3/B5 | Separate frozen-reference audit | Total improvement over initialization; not isolated method attribution |
| X17 gains 0.008250 primary AP and 0.012376 block AP over fresh ERM | Same 18 matched cells | Positive total effect of the routed specialization system |
| X17 loses block AP to X14 in T1/2023, T5/2022, and T5/2023 | Three-seed means of -0.000649, -0.000898, and -0.001614 | Fixed-severity factorization is not uniformly better than mixed-severity specialization |

The source for these values is `docs/experiments/t1_t5_innovations.md`, especially
the completed X17 and X22 sections and the complete-route/frozen-reference audit.
`docs/CODE_LIFECYCLE.md` defines the selected maintenance scope.
`docs/research/evaluation-population.md` records the population correction.
Original result files and per-row provenance in this package take precedence over
rounded narrative values when constructing plots.

## A discrepancy that must remain visible

The original `x22-x17-complete-route-final.json` contains X22 clean-scenario
scores. The later documented mainline and current composer use fresh ERM for
M00. For example, the original T1/2021/seed-0 route has a nonzero positive clean
delta; an ERM-clean route must have zero clean delta by definition.

The package retains both definitions under different identifiers:

| Identifier | M00 | M01 | M06 | M07 | Evidence type |
|---|---|---|---|---|---|
| `X22_X17_HISTORICAL` | X22 | X22 | X17 25% | X17 50% | Preserved original composition |
| `X22_X17_ERM_CLEAN` | Fresh ERM | X22 | X17 25% | X17 50% | Newly derived composition of existing component scores |

The primary and block metrics exclude M00, so their values are unchanged by
this correction. This does not make the historical clean score interchangeable
with the newly derived clean score. No old source artifact is overwritten.

## Scope and evidence limits

- **Selection and reporting years:** training uses 2016–2020 and validation uses
  2021. The 2022/2023 results have already been inspected in the research
  campaign. They are historical test-year evidence, not untouched confirmation
  for a newly selected story. Older JSON gate names containing `heldout` retain
  their historical meaning; they do not establish new independent confirmation.
- **Architecture and feature confounding:** T1 and T5 differ in history,
  architecture, and feature set. Their comparison cannot establish a pure
  temporal-history effect.
- **Synthetic observation failure:** M01/M06/M07 are controlled perturbations.
  A gain under these perturbations does not establish robustness to the full
  distribution of naturally missing satellite observations.
- **Statistical unit:** three seeds quantify variability of continuations from
  shared source checkpoints. They do not represent three independent wildfire
  populations. No event-bootstrap confidence interval or significance claim
  follows from these aggregate summaries.
- **Cost:** the current complete route retains four component models per
  setting/seed, each with a 3000-step continuation. The common foundation cost
  is separate. Step totals and checkpoint counts do not measure GPU-hours,
  FLOPs, latency, peak memory, or an equal-total-compute advantage.
- **Clean behavior:** zero clean change for the ERM-clean composition follows
  from selecting ERM. It does not prove that X22 or either X17 expert preserves
  complete-input performance.
- **Summary composition:** each scenario score selects the recorded score from
  its designated model. This is not a new ensemble AP obtained by mixing logits
  within a sample, nor evidence that a deployment router was executed.
- **Provenance:** result summaries alone do not prove physical batch,
  initialization, normalization identity, completed training, or checkpoint
  selection. Source/configuration records and checkpoint metadata are needed
  for those assertions. File hashes establish identity of copied evidence,
  not correctness of the underlying experiment.
- **Novelty and comparisons:** cosine decay is an existing optimization tool.
  X17 is a factorization of the X14 specialization recipe. A new backbone,
  a universally superior mechanism, and state-of-the-art performance are not
  established by these results. No completed matched comparison to all closest
  reconstruction-then-forecast methods is asserted.
- **Archived evidence:** a positive number does not restore an inactive method
  to the active contribution list. Single-seed screens, invalid ancestry,
  cancellations, and unresolved outcomes remain separately labelled.

## Work needed before stronger claims

The X22+X14 composition provides a necessary system-level ablation from existing
evidence and should be reported regardless of its sign. To claim deployed routing,
run an explicit inference-path test with observed missingness and record actual
costs. To claim independent generalization, define a prospective confirmation
design on data not used for selection. To claim superiority to nearby methods,
run an agreed matched comparison under the same data and evaluation contract.
None of these future claims is a prerequisite for preserving and presenting the
current bounded findings accurately.
