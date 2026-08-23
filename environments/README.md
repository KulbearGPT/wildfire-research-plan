# Research environments

This repository keeps two environment boundaries because data auditing and GPU
model execution have different risks and dependencies.

## Audit environment

The audit environment installs the root `wildfire-phase0` package and its test
dependencies. It runs deterministic inventory, repair, split, rule-baseline,
manifest, and reporting code. It does not need CUDA, Lightning, the authors'
model repository, or released weights.

Record Python, package lock or freeze, Git commit, and the data-manifest hash.
This environment can run on a CPU login or scheduled CPU node, subject to the
site's policy.

## Training environment

The training environment runs the pinned official code and future learned
models. Record at minimum:

- Python and compiler versions;
- PyTorch, torchvision, CUDA runtime, and NVIDIA driver;
- PyTorch Lightning, NumPy, h5py, and segmentation-models-pytorch;
- GPU model and count visible to the task;
- module list, requirements input, Git commit, upstream commit, and weight/data
  manifest hashes.

Use one H100 per task. Folds and seeds are independent Slurm jobs; do not add
DDP until a measured single-run bottleneck justifies it.

**Freeze exact training versions only after the target-cluster smoke** and the
Fold-2 test-only checkpoint gate pass. The authors' requirements are the
candidate starting point, but the first reviewed Linux/H100 lock must reflect
the versions actually qualified with the site's driver, CUDA stack, compiler,
and wheel source. Do not silently reuse the Windows environment lock.

## Qualification boundary

A passing import check is necessary but insufficient. Qualification proceeds
through:

1. module, Python, compiler, CUDA, driver, and H100 inventory;
2. strict profile and 999-file data-manifest verification;
3. pinned upstream commit and import-scope patch verification;
4. one training batch plus one validation batch with no optimization;
5. strict loading and test-only execution of the official Fold-2 weight;
6. a 500-step no-test timing calibration only after steps 1–5 pass.

Agreement with the sealed Fold-2 checkpoint establishes numerical executable
equivalence, not byte-identical hardware equivalence. Report the tolerance,
all six test metrics, precision mode, and any deterministic-algorithm warnings.
It does not recreate paper-table provenance and does not turn the calibration
into a scientific result.

After qualification, freeze the environment input, record `python -m pip
freeze`, and keep the resulting small text evidence with the reviewed run. Do
not commit the virtual environment, wheel cache, CUDA cache, or checkpoints.
