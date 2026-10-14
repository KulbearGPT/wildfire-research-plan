# Condition-specific specialist route (X22+X17)

Question: What does the condition-specific route add over a matched single continuation?

## Preparation

- [ ] Identify the matched ERM, cosine, fixed-25%, and fixed-50% checkpoints for one history and seed; verify their common foundation and comparison contract.
- [ ] Declare whether this is the current ERM-clean route or the preserved historical X22-clean variant, and record the exact checkpoint used for every condition.
- [ ] If an arm is missing, prepare its independent training plan and obtain authorization before running it; do not silently mix histories, seeds, populations, or selection rules.

## Intervention

- [ ] For any authorized missing arm, run the bounded smoke check and its fixed training plan from the matching foundation, then verify the retained checkpoint.
- [ ] Evaluate all required arms on the same scenario samples and masks; retain the full individual-model results before composing the route.
- [ ] Compose the declared route: cosine for FireDrop, the matching fixed-severity expert for each block condition, and the explicitly chosen clean-input model.

## Comparison

- [ ] Locate the source comparison: fresh matched ERM and explicitly separate frozen B3/B5 audit.
- [ ] Distinguish the historical X22-clean artifact from the current ERM-clean composition; missingness means do not reveal this clean-route difference.

## Report

- [ ] Report route-minus-matched-ERM and route-minus-frozen-foundation separately, with the selected arm for each condition and full checkpoint provenance.
- [ ] Count independent training runs, total update budget, and retained checkpoints; distinguish these costs from the single selected model used per routed prediction.
- [ ] Keep the historical and current clean-route results separately labeled and preserve the limitation that system composition does not isolate a new mechanism.

## My record

- Data / statistics / initialization identities:
- Source version and intended differences:
- Budget and selection/evaluation contract:
- Job IDs, terminal states and actual artifact paths (if run):
- Result table or preparation-only status:
- Supported conclusion and limitation:
- Next comparison:

Do not fill missing measurements with teaching values.
