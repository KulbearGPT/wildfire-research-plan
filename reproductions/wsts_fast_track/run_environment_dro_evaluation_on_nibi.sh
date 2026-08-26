#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 P00_COMPLETED_RECORD P02_COMPLETED_RECORD P09_CHECKPOINT YEAR" >&2
  exit 2
fi

p00_record=$(realpath "$1")
p02_record=$(realpath "$2")
dro_checkpoint=$(realpath "$3")
year=$4
case "${year}" in
  2022|2023) ;;
  *) echo "fixed P09 evaluation year must be 2022 or 2023" >&2; exit 2 ;;
esac
runner_path=$(realpath "${BASH_SOURCE[0]}")
repo=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
base=/project/6085198/kulbear/wildfire
upstream=${base}/cache/WildfireSpreadTS-res18-runtime
data=${base}/hdf5/wstsplus-active-fixed
stats=${base}/runs/nibi-wstsplus-data-20260823/train-2016-2020-stats.npz
run_root=${base}/runs/prototype-P09-GroupDRO-eval-${year}-${SLURM_JOB_ID}

test -f "${p00_record}"
test -f "${p02_record}"
test -f "${dro_checkpoint}"
test -f "${stats}"
if [[ -e "${run_root}" ]]; then
  echo "refusing existing run root: ${run_root}" >&2
  exit 2
fi

module purge
module load StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.2
source "${base}/envs/wsts-res18-t1-nibi-smoke/bin/activate"

mkdir -p "${run_root}/project"
cp "${runner_path}" "${run_root}/run_environment_dro_evaluation_on_nibi.sh"
git -C "${repo}" rev-parse HEAD > "${run_root}/project-commit.txt"
git -C "${repo}" status --porcelain=v1 > "${run_root}/project-status.txt"
git -C "${repo}" archive --format=tar HEAD | tar -xf - -C "${run_root}/project"
sha256sum "${p00_record}" "${p02_record}" "${dro_checkpoint}" "${stats}" > "${run_root}/inputs.sha256"

cd "${run_root}/project"
export WANDB_MODE=disabled
export WANDB_SILENT=true
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

python -m reproductions.wsts_fast_track.evaluate_environment_dro \
  --p00-record "${p00_record}" \
  --p02-record "${p02_record}" \
  --dro-checkpoint "${dro_checkpoint}" \
  --output-root "${run_root}/results-${year}" \
  --upstream-root "${upstream}" \
  --data-root "${data}" \
  --stats-path "${stats}" \
  --year "${year}" \
  --heldout-authorized \
  --device cuda 2>&1 | tee "${run_root}/evaluation-${year}.log"

test -f "${run_root}/results-${year}/summary.json"
