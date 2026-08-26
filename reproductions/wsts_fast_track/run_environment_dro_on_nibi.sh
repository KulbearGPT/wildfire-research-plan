#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 P00_COMPLETED_RECORD P02_COMPLETED_RECORD" >&2
  exit 2
fi

p00_record=$(realpath "$1")
p02_record=$(realpath "$2")
runner_path=$(realpath "${BASH_SOURCE[0]}")
repo=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
base=/project/6085198/kulbear/wildfire
upstream=${base}/cache/WildfireSpreadTS-res18-runtime
data=${base}/hdf5/wstsplus-active-fixed
stats=${base}/runs/nibi-wstsplus-data-20260823/train-2016-2020-stats.npz
run_root=${base}/runs/prototype-P09-YearCorruptionGroupDRO-${SLURM_JOB_ID}
dro_checkpoint=${run_root}/groupdro.pt
output_root=${run_root}/results-2021

test -f "${p00_record}"
test -f "${p02_record}"
test -f "${stats}"
if [[ -e "${run_root}" ]]; then
  echo "refusing existing run root: ${run_root}" >&2
  exit 2
fi

module purge
module load StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.2
source "${base}/envs/wsts-res18-t1-nibi-smoke/bin/activate"

mkdir -p "${run_root}/project"
cp "${p00_record}" "${run_root}/p00-completed.json"
cp "${p02_record}" "${run_root}/p02-completed.json"
cp "${runner_path}" "${run_root}/run_environment_dro_on_nibi.sh"
git -C "${repo}" rev-parse HEAD > "${run_root}/project-commit.txt"
git -C "${repo}" status --porcelain=v1 > "${run_root}/project-status.txt"
git -C "${repo}" archive --format=tar HEAD | tar -xf - -C "${run_root}/project"
sha256sum "${p00_record}" "${p02_record}" "${stats}" > "${run_root}/inputs.sha256"

cd "${run_root}/project"
export WANDB_MODE=disabled
export WANDB_SILENT=true
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

python -m reproductions.wsts_fast_track.train_environment_dro \
  --p02-record "${p02_record}" \
  --upstream-root "${upstream}" \
  --data-root "${data}" \
  --stats-path "${stats}" \
  --output-path "${dro_checkpoint}" \
  --device cuda 2>&1 | tee "${run_root}/training.log"

python -m reproductions.wsts_fast_track.evaluate_environment_dro \
  --p00-record "${p00_record}" \
  --p02-record "${p02_record}" \
  --dro-checkpoint "${dro_checkpoint}" \
  --output-root "${output_root}" \
  --upstream-root "${upstream}" \
  --data-root "${data}" \
  --stats-path "${stats}" \
  --year 2021 \
  --device cuda 2>&1 | tee "${run_root}/evaluation-2021.log"

test -f "${output_root}/summary.json"
