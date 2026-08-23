# WSTS+ Post-Control Replication and Missingness Design

## Purpose

Continue the Fast Experiment Track after the passing C00/C02 seed-0 screening
runs. The immediate execution stage is the already-submitted fresh seed-0 10K
pair. While those jobs run, prepare the next reproducibility gate and freeze the
controlled-missingness evaluation boundary without accessing 2022--2023.

## Execution order

1. Complete `C00-S0-10K` and `C02-S0-10K` as fresh runs selected on 2021.
2. Render seed 1/2 manifests only after both seed-0 10K completion records pass.
3. Run the four matched replications without selecting a model between seeds.
4. Review clean mean/dispersion and accept clean checkpoints before any
   controlled-missingness evaluation.
5. Implement M00--M07 against synthetic fixtures, then execute the frozen
   matrix once on the held-out evaluation years.

## Replication gate

The existing typed matrix remains authoritative. A replication manifest is
renderable only when exactly one passing `C00-S0-10K` record and one passing
`C02-S0-10K` record are supplied. Records must have seed 0, 10,000 steps, the
frozen fit split, `test_enabled=false`, finite metrics, a positive CUDA peak,
and a valid checkpoint step. The target seed is resolved only from its run ID.

Rendering remains read-only and non-overwriting. It emits a scientific command
but never calls Slurm. The four replication manifests are therefore prepared
by code now and remain operationally locked until the real seed-0 evidence
exists.

## Controlled-missingness boundary

M00--M07 are evaluation conditions, not retraining conditions in the first
diagnostic pass. Accepted clean checkpoints are evaluated under every scenario
with the same event/date population.

- M00 preserves the input exactly.
- M01 removes all available active-fire history.
- M02 replaces active-fire history with the immediately preceding available
  day and is invalid when that day cannot be recovered without changing the
  sample population.
- M03 removes observed-weather features 5--11.
- M04 removes forecast-weather features 17--21.
- M05 applies M03 and M04 together.
- M06/M07 apply deterministic structured blocks to the declared dynamic
  inputs at 25%/50% area.

Every stochastic-looking mask is derived from a stable hash of schema version,
scenario ID, matrix seed, event-relative path, and target date. Target pixels,
predictions, and model metrics are forbidden inputs to mask generation.

The upstream validation dataset currently enables training augmentation.
Formal M00--M07 results must therefore use a separate `is_train=false`
evaluation dataset. Implementation tests use synthetic arrays only and may not
open the real 2022--2023 files. Missing-value encoding and the M02 sample
contract must be frozen in a later reviewed implementation change after the
clean replication gate passes.

## Claim boundary

The seed-0 3K and 10K runs provide screening and clean-validation evidence,
respectively. They do not provide held-out test performance. M00--M07 diagnose
controlled failures only and do not establish natural missingness or
operational availability behavior.
