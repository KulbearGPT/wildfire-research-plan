# T1 missingness-trained foundation (B3)

Question: What does adding block augmentation contribute to the T1 foundation?

## Preparation

- [ ] Prepare the corrected T1 input pipeline, training-year normalization statistics, declared data split, and the foundation recipe’s encoder initialization; do not start from a later continuation checkpoint.
- [ ] Identify the foundation-specific training budget, validation-AP checkpoint-selection rule, and export/provenance requirements from the source; declare compute limits before execution.

## Intervention

- [ ] Keep the clean reference and FireDrop/BlockDrop foundation recipes distinct, including their declared stochastic augmentation probabilities.
- [ ] Once authorized, run the bounded setup and loader/model smoke checks, then train the identified foundation for its fixed budget in a new output location.
- [ ] Verify that training completed and export the checkpoint selected by the foundation validation rule, which can precede the final training update; preserve its checksum and selection metadata.

## Comparison

- [ ] Locate the source comparison: corrected T1 FireDrop-trained baseline (B2).
- [ ] Use B2 for the block-augmentation comparison; later continuations answer a different question.

## Report

- [ ] Evaluate the exported foundation under the same project scenario and sample contract used for the intended baseline comparison.
- [ ] Deliver data/statistics identities, initialization provenance, completion evidence, selected-checkpoint identity, and per-condition scores; keep this stage separate from final-step continuation selection.
- [ ] Record any missing comparison arm or protocol deviation rather than treating unmatched scores as a method gain.

## My record

- Data / statistics / initialization identities:
- Source version and intended differences:
- Budget and selection/evaluation contract:
- Job IDs, terminal states and actual artifact paths (if run):
- Result table or preparation-only status:
- Supported conclusion and limitation:
- Next comparison:

Do not fill missing measurements with teaching values.
