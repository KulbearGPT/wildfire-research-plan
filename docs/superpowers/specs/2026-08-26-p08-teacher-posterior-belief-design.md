# P08 Teacher-Posterior Belief: Fast-Prototype Design

## Decision

Run one final belief-state prototype that makes latent uncertainty identifiable
from the synthetic-corruption training contract. P08 uses the clean input as a
training-only teacher posterior and learns a corrupted-input prior for
inference. If P08 collapses or fails its 2021 selection rule, stop the
belief-state branch rather than adding P09.

## Motivation from P07

P07 optimized only the mean of four sampled forecasts. Its objective therefore
did not require the samples to encode different plausible observation states.
The resulting predictive variance was approximately `1e-8`, and P07 behaved as
another deterministic residual. More steps or an ungrounded entropy reward
would not correct that identifiability problem.

## Minimal architecture

Training retains both tensors already available before controlled corruption:
the clean 40-channel P00 input and the 41-channel corrupted input plus mask.

1. Run frozen P00 once on the clean input and once on the corrupted input to
   obtain clean and corrupted 16-channel decoder features.
2. A posterior head maps the clean feature and detached corrupted feature to
   16-channel diagonal-Gaussian posterior parameters
   `q(z | clean, corrupt)`.
3. A prior head maps the corrupted feature and missing mask to
   16-channel diagonal-Gaussian prior parameters `p(z | corrupt, mask)`.
4. During training, draw four posterior samples. A shared output head maps each
   sample to a missing-region residual forecast.
5. During evaluation, discard the clean path and draw four samples from the
   corrupted-input prior. Average probabilities and report sample variance.
6. Outside the missing mask, return P00 logits exactly.

P08 remains `T=1`, freezes P00, and does not add a new backbone, temporal model,
CVAE image decoder, diffusion model, expert ensemble, or generic uncertainty
framework.

## Loss

Use only three terms:

- the existing next-day forecasting loss on the marginalized prediction;
- diagonal-Gaussian `1e-3 * KL(q || p)` so the inference prior follows the clean-state
  posterior;
- `1e-2` times masked feature-reconstruction MSE from each sampled latent toward
  the clean minus corrupted decoder-feature residual, grounding the latent in
  the hidden clean state.

Do not sweep the weights. Clamp both posterior and prior log-scales to
`[-2, 2]`; the lower bound prevents numerical zero-scale collapse but does not
guarantee that the forecast head uses sampled variation. Log the three loss
components every 100 steps so that failure remains diagnosable from one run.

## Training and evaluation

- Train only the posterior, prior, feature-reconstruction, and output heads on
  2016--2020.
- Use one seed, four samples, existing 25%/50% structured BlockDrop, Adam at
  `1e-3`, and exactly 3,000 steps.
- Select only on 2021 M00, M01, M06, and M07.
- Compare against existing P00 and P04 results. Obtain P04 Brier only if AP does
  not decide the promotion result; do not launch a broad baseline campaign.
- Only a selected and frozen P08 proceeds to the fixed 2022--2023 test set.

## Promotion rule

P08 advances only if all conditions hold:

1. The absolute M00 and M01 AP differences from P00 are each at most `1e-6`.
2. Mean M06/M07 AP is strictly greater than mean P00 M06/M07 AP.
3. Mean M06/M07 predictive variance is at least `1e-6` and at least ten times
   M00 predictive variance. The absolute threshold is roughly two orders of
   magnitude above P07's collapsed variance.
4. Mean M06/M07 AP is strictly greater than P04, or mean M06/M07 Brier is
   strictly lower than P04 while mean AP is no more than `0.002` below P04.

Failure ends the belief-state prototype branch. The next research direction is
temporal/environment robust optimization, not another stochastic residual.

## Implementation boundary

Extend the existing residual model, trainer, evaluator, and Nibi launcher with
one `teacher-belief` variant. During training only, append or otherwise retain
the clean processed tensor before applying BlockDrop; evaluation continues to
consume the current 41-channel corrupted-input contract.

Verification is limited to one CPU test covering clean-teacher/prior shapes,
finite loss components, gradients, and exact unmasked P00 preservation, plus
Python compilation and shell syntax. All training and dataset evaluation run
through one Slurm job on a Nibi compute node, never on the login node.
