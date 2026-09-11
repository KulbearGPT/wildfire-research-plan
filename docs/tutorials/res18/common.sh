#!/usr/bin/env bash
# Shared job setup. This file is sourced by the tutorial jobs.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script through sbatch; do not run it on a login node}"
: "${WF_TUTORIAL_ENV:?Export WF_TUTORIAL_ENV as described in the tutorial}"
source "$WF_TUTORIAL_ENV"
: "${WF_ROOT:?}" "${WF_REPO:?}"
WF_SCRIPTS="$WF_REPO/docs/tutorials/res18"
WF_UPSTREAM="$WF_ROOT/upstream"
export TORCH_HOME="$WF_ROOT/cache/torch"
export WANDB_MODE=disabled WANDB_SILENT=true PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE PYTHONDONTWRITEBYTECODE=1
# Lightning 2.0 expects the legacy loader for our own/pinned trusted checkpoints.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="${SLURM_TMPDIR:?}"
train_env() {
  module purge
  module load StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.2
  source "$WF_ROOT/envs/train/bin/activate"
  export PYTHONPATH="$WF_REPO:$WF_UPSTREAM/src"
}
audit_env() {
  module purge
  module load StdEnv/2023 gcc/12.3 python/3.13.2
  source "$WF_ROOT/envs/audit/bin/activate"
  export PYTHONPATH="$WF_REPO/src:$WF_REPO"
}
new_run() {
  WF_RUN="$WF_ROOT/runs/$1-${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+-$SLURM_ARRAY_TASK_ID}"
  mkdir "$WF_RUN"
  mkdir "$WF_RUN/work"
  cp "$WF_TUTORIAL_ENV" "$WF_RUN/tutorial.env"
  cp -r "$WF_SCRIPTS" "$WF_RUN/tutorial-scripts"
  mkdir "$WF_RUN/project"
  git -C "$WF_REPO" archive HEAD | tar -xf - -C "$WF_RUN/project"
  git -C "$WF_REPO" rev-parse HEAD > "$WF_RUN/project-commit.txt"
  git -C "$WF_REPO" status --porcelain > "$WF_RUN/project-status.txt"
  git -C "$WF_UPSTREAM" rev-parse HEAD > "$WF_RUN/upstream-commit.txt"
  git -C "$WF_UPSTREAM" diff > "$WF_RUN/upstream-runtime.patch"
  python -m pip freeze > "$WF_RUN/pip-freeze.txt"
  module list > "$WF_RUN/modules.txt" 2>&1
  scontrol show job "$SLURM_JOB_ID" > "$WF_RUN/slurm-job.txt"
  if [[ "${SLURM_JOB_GPUS:-}" != "" ]]; then
    nvidia-smi -q > "$WF_RUN/nvidia-smi.txt"
  fi
  WF_REPO="$WF_RUN/project"
  WF_SCRIPTS="$WF_RUN/tutorial-scripts"
  export PYTHONPATH="$WF_REPO/src:$WF_REPO:$WF_UPSTREAM/src"
  cd "$WF_RUN/work"
}
