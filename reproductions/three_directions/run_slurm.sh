#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Slurm required}"
: "${WILDFIRE_SOURCE_COMMIT:?Pinned commit required}"
repo=$(git -C "${SLURM_SUBMIT_DIR}" rev-parse --show-toplevel)
root=/project/6085198/kulbear/wildfire/runs/three-directions-${SLURM_JOB_ID}
mkdir -p "${root}/project"
git -C "${repo}" archive "${WILDFIRE_SOURCE_COMMIT}" | tar -xf - -C "${root}/project"
git -C "${repo}" rev-parse "${WILDFIRE_SOURCE_COMMIT}^{commit}" > "${root}/commit.txt"
scontrol show job "${SLURM_JOB_ID}" > "${root}/allocation.txt"
module load StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.2
source /project/6085198/kulbear/wildfire/envs/wsts-res18-t1-nibi-smoke/bin/activate
export WANDB_MODE=disabled WANDB_SILENT=true HDF5_USE_FILE_LOCKING=FALSE
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
cd "${root}/project"
if [[ "${1:-}" == smoke-suite ]]; then
  shift
  for mode in bn_shared bn_conditional merge x14_shared student_control student_distill; do
    python -m reproductions.three_directions.run --mode "${mode}" --smoke \
      --output "${root}/smoke-${mode}" "$@" 2>&1 | tee "${root}/${mode}.log"
  done
else
  python -m reproductions.three_directions.run --output "${root}/result" "$@" 2>&1 | tee "${root}/run.log"
fi
