# Fixed-severity block specialists (X17)

Question: Does separating 25% and 50% training help beyond one mixed specialist?

## Preparation

- [ ] Prepare the same corrected data, training-only statistics, and verified B3 or B5 starting checkpoint for the candidate and its controls; keep each history setting separate.
- [ ] Freeze seed, physical batch, optimizer settings, per-run update budget, final-checkpoint rule, and evaluation samples from the source recipe; obtain authorization for the planned compute.

## Intervention

- [ ] Initialize separate 25% and 50% block specialists from the same foundation as the mixed-severity specialist and ERM; do not continue one expert from another.
- [ ] After authorization, perform the bounded smoke check specified by the execution plan; verify input, loss, update, and checkpoint paths before the formal run.
- [ ] Train each planned branch for its declared budget in a distinct output location, retain the final continuation checkpoint, and record actual completion rather than relying on submission status.

## Comparison

- [ ] Locate the source comparison: X14 mixed-severity specialist plus fresh ERM.
- [ ] Report two models and twice the per-specialist updates; compare each severity with the mixed specialist before claiming specialization benefit.

## Report

- [ ] Evaluate the retained checkpoints on the same complete-input, FireDrop, 25% block, and 50% block samples and masks; keep individual-model and routed scores separately identified.
- [ ] Compare each fixed expert with the mixed specialist at its matching severity; report two training runs, total cost, and the composed route separately from component attribution.
- [ ] For a new run, label new measurements separately from the historical record and report any deviations or incomplete work.

## My record

- Data / statistics / initialization identities:
- Source version and intended differences:
- Budget and selection/evaluation contract:
- Job IDs, terminal states and actual artifact paths (if run):
- Result table or preparation-only status:
- Supported conclusion and limitation:
- Next comparison:

Do not fill missing measurements with teaching values.
