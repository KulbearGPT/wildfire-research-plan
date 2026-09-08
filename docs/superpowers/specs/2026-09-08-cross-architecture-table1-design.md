# Cross-architecture Table 1 validation design

## Goal

Test whether the retained robustness system transfers beyond Res18-U-Net and
Res18-UTAE, using only forecasting architectures already evaluated by WSTS or
WSTS+. Produce matched results suitable for the paper's main comparison table.

## Scientific scope

The required new backbone is SwinUnet at both T=1 and T=5. WSTS+ already uses
this architecture in both settings, its implementation is present in the
pinned upstream tree, and it supplies the cleanest spatial-CNN/temporal-UTAE to
Transformer transfer test. ConvLSTM at T=5 is the first conditional extension
because original WSTS uses it as a recurrent temporal baseline. SegFormer-B2
at T=1 is the second conditional extension if Swin transfers and another
Transformer family materially strengthens the main table. Res50-U-Net is not
in the first campaign because changing encoder capacity inside the same U-Net
family provides less independent evidence.

The experiment does not reopen model, loss, corruption-rate, or route search.
It transfers only mechanisms that require no backbone-internal feature API:

- X22: cosine learning-rate decay against matched constant-rate ERM;
- X14: BlockDrop-only specialist training against matched ERM;
- the full route: X22 for M00/M01 and fixed-severity BlockDrop specialists for
  M06/M07, following the existing X22+X17 deployment rule.

X8 is excluded from the first cross-backbone main-table campaign because it
failed the existing fresh-ERM adoption check in T1/2022. It remains an ablation
result rather than being used to define `Ours`.

## Architecture boundary

Add an explicit architecture identifier to the cross-history runner and saved
checkpoint schema. Existing checkpoints without this field remain compatible:
history 1 implies `res18_unet`, and history 5 implies `res18_utae`.

The existing `Forecaster` remains untouched for architecture-specific modules.
A small direct forecaster wraps any upstream `BaseModel` for `control`,
`cosine_erm`, and `block_specialist`; these methods differ only in schedule or
training distribution and need no encoder/decoder access. Unsupported
architecture/method pairs fail before allocating a dataset.

New backbones use an explicit bootstrap checkpoint. For each seed, train the
published backbone recipe for 10,000 steps, freeze that checkpoint, then branch
matched 3,000-step constant ERM, cosine ERM, and BlockDrop specialists from it.
Every branch for a backbone/seed shares the exact initialization, data order,
sample count, optimizer family, and initial learning rate. Swin/SegFormer use
the pretrained initialization specified by WSTS+ for both control and method;
ConvLSTM uses its published random initialization and architecture-specific
learning rate. No test year selects a checkpoint or recipe.

## Evidence protocol

Wave 0 consists only of one-step real-data Slurm smokes. Wave 1 trains seed 0
and evaluates 2021. A mechanism advances only if its primary AP delta is at
least +0.005 against its matched control in both Swin histories and its M00
delta is at least -0.010. Failure in either history stops that mechanism on new
backbones.

Wave 2 adds seeds 1 and 2 plus frozen 2022/2023 evaluation only for advancing
methods. Main-table claims require positive three-seed mean primary delta in
both Swin histories and both test years. ConvLSTM and SegFormer seed-0 screens
may run concurrently with Swin to reduce wall-clock latency; each architecture
still gates its own seeds 1/2 and held-out work and does not trigger another
search.

All model work runs through Slurm. Login-node work is restricted to source,
tiny metadata inspection, queue inspection, JSON composition, and Git. Pending
jobs older than ten minutes trigger slice/partition probes; an alternative is
used only if it starts earlier and requests no more than twice the resources,
except for a task that blocks the remaining dependency chain.

## Main table contract

Table 1 reports the corrected fixed 2022/2023 test population only. It groups
rows by backbone and history and includes M00, M01, M06, M07, corrupted-primary
mean, and delta against matched ERM. Values are means over seeds 0/1/2; the
paper reports the seed standard deviation. The 2021 selection results belong
in the ablation or appendix, not the test mean.

Published AP values are not copied into the table because their folds and
missingness protocol differ. Published work establishes that a backbone is a
legitimate wildfire baseline; every numerical comparison is rerun under the
same corrected split, target dates, feature set, corruption masks, optimizer
budget, and AP implementation used by this repository.

## Minimal verification

Use one focused CPU unit test for architecture compatibility and checkpoint
metadata, followed by one-step real-data GPU smokes in Slurm. Do not run the
repository-wide suite, synthetic architecture sweeps, or full held-out jobs
before the registered seed-0 gate passes.
