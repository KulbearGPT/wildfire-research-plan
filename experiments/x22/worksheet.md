# Cosine-schedule continuation (X22)

Question: Does cosine learning-rate decay help under a matched continuation budget?

## Preparation

- [ ] Prepare the same corrected data, training-only statistics, and verified B3 or B5 starting checkpoint for the candidate and its controls; keep each history setting separate.
- [ ] Freeze seed, physical batch, optimizer settings, per-run update budget, final-checkpoint rule, and evaluation samples from the source recipe; obtain authorization for the planned compute.

## Intervention

- [ ] Initialize independent ERM and cosine branches from the same foundation; change only the declared learning-rate schedule.
- [ ] After authorization, perform the bounded smoke check specified by the execution plan; verify input, loss, update, and checkpoint paths before the formal run.
- [ ] Train each planned branch for its declared budget in a distinct output location, retain the final continuation checkpoint, and record actual completion rather than relying on submission status.

## Comparison

- [ ] Locate the source comparison: fresh constant-LR ERM.
- [ ] Start both continuations independently from the same foundation; continuing the ERM result would add training exposure.

## Report

- [ ] Evaluate the retained checkpoints on the same complete-input, FireDrop, 25% block, and 50% block samples and masks; keep individual-model and routed scores separately identified.
- [ ] Report cosine-minus-ERM separately from cosine-minus-foundation, with artifact identities, per-condition scores, the missingness mean, and the single-seed or multi-seed scope.
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
