#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Slurm required}"
history=$1
method=$2
shift 2
repo=$(git -C "${SLURM_SUBMIT_DIR}" rev-parse --show-toplevel)
source_commit=${WILDFIRE_SOURCE_COMMIT:-HEAD}
root=/project/6085198/kulbear/wildfire/runs/cross-history-${history}-${method}-${SLURM_JOB_ID}
module purge
module load StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.2
source /project/6085198/kulbear/wildfire/envs/wsts-res18-t1-nibi-smoke/bin/activate
mkdir -p "${root}/project"
git -C "${repo}" archive "${source_commit}" | tar -xf - -C "${root}/project"
git -C "${repo}" rev-parse "${source_commit}^{commit}" > "${root}/commit.txt"
scontrol show job "${SLURM_JOB_ID}" > "${root}/allocation.txt"
export WANDB_MODE=disabled WANDB_SILENT=true HDF5_USE_FILE_LOCKING=FALSE
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
cd "${root}/project"
python -m reproductions.cross_history.run --history "${history}" --method "${method}" \
  --output "${root}/result" "$@" 2>&1 | tee "${root}/run.log"
