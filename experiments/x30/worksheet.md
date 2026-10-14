# Training-only fire-change heads (X30)

Question: Do auxiliary fire-change tasks help the primary forecast during training?

## Preparation

- [ ] Identify the B3 or B5 starting checkpoint and the exact cross-history variant; T1 and T5 change architecture and features as well as history.
- [ ] Read the source and current lifecycle before planning work; the historical record is not authorization for a new run.

## Intervention

- [ ] Draw auxiliary heads connected to training supervision and mark their absence from the inference task.
- [ ] Recover which parameters, inputs, losses, or sampling rules changed in this variant, and keep all other recorded choices visible.

## Comparison

- [ ] Locate the source comparison: X22 cosine ERM.
- [ ] Read primary-task losses and gains against cosine ERM; auxiliary prediction is not the final forecast metric.

## Report

- [ ] Produce a method diagram and a comparison table using only recorded outcomes; label raw, routed, and total-effect results wherever applicable.
- [ ] State what this comparison can attribute and what it cannot; retain the source’s negative, invalid, incomplete, or single-setting limitations.

## My record

- Data / statistics / initialization identities:
- Source version and intended differences:
- Budget and selection/evaluation contract:
- Job IDs, terminal states and actual artifact paths (if run):
- Result table or preparation-only status:
- Supported conclusion and limitation:
- Next comparison:

Do not fill missing measurements with teaching values.
