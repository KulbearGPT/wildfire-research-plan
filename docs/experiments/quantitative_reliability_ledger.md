# Quantitative Reliability Experiment Ledger

This ledger separates completed quantitative evidence from untested
hypotheses. A candidate is not called a reliable direction until it satisfies
the evidence levels in
`docs/superpowers/specs/2026-09-03-quantitative-reliability-baselines-design.md`.

## Corrected-index baseline wave

| ID | Training policy | Matched baseline | Slurm job | Resource | State | 2021 M00 AP | M01 AP | M06 AP | M07 AP | Classification |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| B0 | clean C00 | -- | not submitted | -- | candidate | -- | -- | -- | -- | baseline repair |
| B1 | clean C02 | B0 | not submitted | -- | candidate | -- | -- | -- | -- | temporal baseline repair |
| B2 | FireDrop C00 | B0 | not submitted | -- | candidate | -- | -- | -- | -- | training baseline |
| B3 | FireDrop + BlockDrop C00 | B0/B2 | not submitted | -- | candidate | -- | -- | -- | -- | joint training baseline |

## Follow-on candidate register

| ID | Hypothesis | Matched control | Primary metric | Evidence level | Quantitative conclusion |
| --- | --- | --- | --- | --- | --- |
| D1 | clean-corrupt predictive consistency | matched ERM continuation | M01 or declared joint mean AP | candidate | no new result yet |
| D2 | reliability-normalized first convolution | standard first convolution | mean M06/M07 AP | candidate | no new result yet |
| D3 | reliability-conditioned temporal fusion | corrected C02 temporal fusion | declared temporal reliability AP | candidate | no new result yet |
| T1 | corruption mixture/curriculum | corrected B3 policy | declared joint mean AP | candidate | no new result yet; at most one tuning contribution |

The older P00/P10 and target-QA measurements motivate these candidates but do
not quantitatively validate D1--D3.
