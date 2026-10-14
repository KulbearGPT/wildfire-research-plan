# Severity-conditioned latent adapters (X19)

Question: Can severity-conditioned adapters share a specialist across severities?

## Preparation

- [ ] Identify the B3 or B5 starting checkpoint and the exact cross-history variant; T1 and T5 change architecture and features as well as history.
- [ ] Read the source and current lifecycle before planning work; the historical record is not authorization for a new run.

## Intervention

- [ ] Show where observed severity conditions the latent adapter and where the base features enter.
- [ ] Recover which parameters, inputs, losses, or sampling rules changed in this variant, and keep all other recorded choices visible.

## Comparison

- [ ] Locate the source comparison: X14 specialist; X17 two-checkpoint upper bound; fresh ERM.
- [ ] Use the mixed specialist for attribution and the two-checkpoint specialist route as a separate capacity reference.

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
