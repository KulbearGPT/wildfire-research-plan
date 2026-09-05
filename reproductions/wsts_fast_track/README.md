# Corrected WSTS+ T=1 Mainline

This package contains the runnable Res18-U-Net `T=1` reliability experiments
that remain supported after the quantitative screen. Every active run uses the
corrected pooled-year sample resolver, 2016--2020 training, seed 0, 3,000
optimizer steps, 2021 validation, and fixed 2022--2023 tests.

## Execution order

1. Train B0, B2, and B3 independently from scratch with
   `run_corrected_baseline_on_nibi.sh B0|B2|B3`.
2. Continue B3 through paired ERM/KL with
   `run_predictive_consistency_on_nibi.sh B3_COMPLETED_RECORD erm|kl`.
3. Train the matched single-corrupt-view control with
   `run_standard_reliability_control_on_nibi.sh B3_COMPLETED_RECORD`.
4. Train D12 with
   `run_severity_adaptive_reliability_prompting_on_nibi.sh B3_COMPLETED_RECORD`.
5. Use `run_reliability_evaluation_on_nibi.sh` only for the fixed 2022/2023
   evaluation of `baseline`, `d1`, `d2`, or `sarp` checkpoints.

Each runner archives the Git tree used by the job, environment metadata,
inputs, Slurm allocation, training log, checkpoint, and result JSON under its
Nibi run directory. The scripts contain no nested `sbatch` call.

## Retained modules

| Module | Responsibility |
| --- | --- |
| `contract.py`, `runtime.py` | frozen C00 data, split, normalization, and upstream CLI contract |
| `corruption_training.py` | B2 FireDrop and B3 FireDrop+BlockDrop augmentation |
| `corrected_baselines.py` | corrected sample resolver and B0/B2/B3 registry |
| `missingness.py`, `matrix.py`, `evaluation.py` | deterministic M00--M07 corruption and T=1 datasets |
| `predictive_consistency.py` | D1 paired samples and Bernoulli KL |
| `processed_reliability.py` | shared single-corrupt-view data for D2/D12 |
| `severity_adaptive_reliability_prompting.py` | D12 input/deep prompts and severity routing |
| `evaluate_missingness.py` | common checkpoint loading and metrics |

## Method contracts

B2 drops complete active-fire history with probability 0.3. B3 independently
adds a 0.3 probability of a deterministic 25% or 50% dynamic-input block.
Their contribution is reported through observable specialist routing and the
incremental B2-to-B3 comparison.

D1-ERM and D1-KL share every training choice; D1-KL adds Bernoulli predictive
consistency with weight 0.1. D2-STD is the standard-convolution control trained
on one processed corrupt view with an appended invalidity map removed before
the base-model forward pass.

D12 consumes the same 41-channel processed tensor. Missingness coverage at or
below 0.375 activates five deep encoder prompt vectors; greater coverage
activates the input token. The fully valid path is exactly the underlying
standard convolution and encoder. D12 adds 1,088 trainable parameters.

## Results and archives

See
[`../../docs/experiments/quantitative_reliability_ledger.md`](../../docs/experiments/quantitative_reliability_ledger.md)
for retained numbers and
[`../../docs/experiments/rejected_experiments.md`](../../docs/experiments/rejected_experiments.md)
for prior attempts. The complete pre-cleanup implementation remains on branch
`archive/pre-t1-cleanup-2026-09-04`.
