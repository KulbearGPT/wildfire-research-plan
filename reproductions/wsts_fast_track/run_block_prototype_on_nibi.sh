#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "usage: $0" >&2
  exit 2
fi

started_at_epoch=$(date +%s)
runner_path=$(realpath "${BASH_SOURCE[0]}")
repo=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
base=/project/6085198/kulbear/wildfire
upstream=${base}/cache/WildfireSpreadTS-res18-runtime
data=${base}/hdf5/wstsplus-active-fixed
data_evidence=${base}/runs/nibi-wstsplus-data-20260823
stats=${data_evidence}/train-2016-2020-stats.npz
run_root=${base}/runs/prototype-P02-FireBlockDrop-C00-${SLURM_JOB_ID}

test -f "${data_evidence}/completed.json"
test -f "${data_evidence}/data-summary.json"
test -f "${stats}"
if [[ -e "${run_root}" ]]; then
  echo "refusing existing run root: ${run_root}" >&2
  exit 2
fi

module purge
module load StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.2
source "${base}/envs/wsts-res18-t1-nibi-smoke/bin/activate"

mkdir -p "${run_root}/project"
cp "${runner_path}" "${run_root}/run_block_prototype_on_nibi.sh"
git -C "${repo}" rev-parse HEAD > "${run_root}/project-commit.txt"
git -C "${repo}" status --porcelain=v1 > "${run_root}/project-status.txt"
git -C "${repo}" archive --format=tar HEAD | tar -xf - -C "${run_root}/project"
git -C "${upstream}" rev-parse HEAD > "${run_root}/upstream-commit.txt"
git -C "${upstream}" status --porcelain=v1 > "${run_root}/upstream-status.txt"
git -C "${upstream}" diff --binary > "${run_root}/derived-runtime.diff"
python -m pip freeze > "${run_root}/pip-freeze.txt"
module list > "${run_root}/modules.txt" 2>&1
nvidia-smi -q > "${run_root}/nvidia-smi.txt"
sha256sum "${stats}" > "${run_root}/training-stats.sha256"
cp "${data_evidence}/data-summary.json" "${run_root}/data-summary.json"

cd "${run_root}/project"
export WANDB_MODE=disabled
export WANDB_SILENT=true
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

python -m reproductions.wsts_fast_track.block_prototype_entrypoint \
  --upstream-root "${upstream}" \
  --data-root "${data}" \
  --run-root "${run_root}" \
  --stats-path "${stats}" 2>&1 | tee "${run_root}/training.log"

python -m reproductions.wsts_fast_track.block_prototype_completion \
  --run-root "${run_root}" \
  --started-at-epoch "${started_at_epoch}" \
  --slurm-job-id "${SLURM_JOB_ID}"
