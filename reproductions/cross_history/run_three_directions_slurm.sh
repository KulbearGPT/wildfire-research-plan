#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Slurm required}"
: "${WILDFIRE_SOURCE_COMMIT:?Pin a committed source revision}"
history=$1
arm=$2
shift 2
repo=$(git -C "${SLURM_SUBMIT_DIR}" rev-parse --show-toplevel)
run_root=/project/6085198/kulbear/wildfire/runs/three-directions-${history}-${arm}-${SLURM_JOB_ID}
module purge
module load StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.2
source /project/6085198/kulbear/wildfire/envs/wsts-res18-t1-nibi-smoke/bin/activate
mkdir -p "${run_root}/project"
git -C "${repo}" archive "${WILDFIRE_SOURCE_COMMIT}" | tar -xf - -C "${run_root}/project"
git -C "${repo}" rev-parse "${WILDFIRE_SOURCE_COMMIT}^{commit}" > "${run_root}/commit.txt"
scontrol show job "${SLURM_JOB_ID}" > "${run_root}/allocation.txt"
export WANDB_MODE=disabled WANDB_SILENT=true HDF5_USE_FILE_LOCKING=FALSE
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export PYTHONPATH="/project/6085198/kulbear/wildfire/deps/pytest-8.4.2${PYTHONPATH:+:${PYTHONPATH}}"
cd "${run_root}/project"
python -m reproductions.cross_history.run_three_directions \
  --history "${history}" --arm "${arm}" --output "${run_root}/result" "$@" \
  2>&1 | tee "${run_root}/run.log"
