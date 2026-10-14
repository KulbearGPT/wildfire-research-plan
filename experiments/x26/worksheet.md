# Cosine-scheduled impact consistency (X26)

Question: Does impact weighting still help when the consistency recipe uses cosine decay?

## Preparation

- [ ] Identify the B3 or B5 starting checkpoint and the exact cross-history variant; T1 and T5 change architecture and features as well as history.
- [ ] Read the source and current lifecycle before planning work; the historical record is not authorization for a new run.

## Intervention

- [ ] Separate the learning-rate schedule, FireDrop specialization, and consistency weights in the diagram.
- [ ] Recover which parameters, inputs, losses, or sampling rules changed in this variant, and keep all other recorded choices visible.

## Comparison

- [ ] Locate the source comparison: cosine FireDrop global-consistency control; X22 cosine ERM adoption.
- [ ] Use cosine global-consistency for weighting attribution and cosine ERM for adoption; they answer different questions.

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
