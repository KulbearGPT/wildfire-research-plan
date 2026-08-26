#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 P00_RECORD P02_RECORD P09_CHECKPOINT P10_CHECKPOINT P13_CHECKPOINT YEAR" >&2
  exit 2
fi

p00_record=$(realpath "$1")
p02_record=$(realpath "$2")
p09_checkpoint=$(realpath "$3")
p10_checkpoint=$(realpath "$4")
p13_checkpoint=$(realpath "$5")
year=$6
case "${year}" in
  2022|2023) ;;
  *) echo "fixed P13 evaluation year must be 2022 or 2023" >&2; exit 2 ;;
esac
runner_path=$(realpath "${BASH_SOURCE[0]}")
repo=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
base=/project/6085198/kulbear/wildfire
upstream=${base}/cache/WildfireSpreadTS-res18-runtime
data=${base}/hdf5/wstsplus-active-fixed
stats=${base}/runs/nibi-wstsplus-data-20260823/train-2016-2020-stats.npz
run_root=${base}/runs/prototype-P13-eval-${year}-${SLURM_JOB_ID}

test -f "${p00_record}"
test -f "${p02_record}"
test -f "${p09_checkpoint}"
test -f "${p10_checkpoint}"
test -f "${p13_checkpoint}"
test -f "${stats}"
if [[ -e "${run_root}" ]]; then
  echo "refusing existing run root: ${run_root}" >&2
  exit 2
fi

module purge
module load StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.2
source "${base}/envs/wsts-res18-t1-nibi-smoke/bin/activate"

mkdir -p "${run_root}/project"
cp "${runner_path}" "${run_root}/run_fire_only_evaluation_on_nibi.sh"
git -C "${repo}" rev-parse HEAD > "${run_root}/project-commit.txt"
git -C "${repo}" status --porcelain=v1 > "${run_root}/project-status.txt"
git -C "${repo}" archive --format=tar HEAD | tar -xf - -C "${run_root}/project"
sha256sum "${p00_record}" "${p02_record}" "${p09_checkpoint}" "${p10_checkpoint}" "${p13_checkpoint}" "${stats}" > "${run_root}/inputs.sha256"

cd "${run_root}/project"
export WANDB_MODE=disabled
export WANDB_SILENT=true
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

python -m reproductions.wsts_fast_track.evaluate_environment_dro \
  --p00-record "${p00_record}" \
  --p02-record "${p02_record}" \
  --dro-checkpoint "${p09_checkpoint}" \
  --erm-checkpoint "${p10_checkpoint}" \
  --fire-only-checkpoint "${p13_checkpoint}" \
  --output-root "${run_root}/results-${year}" \
  --upstream-root "${upstream}" \
  --data-root "${data}" \
  --stats-path "${stats}" \
  --year "${year}" \
  --heldout-authorized \
  --device cuda 2>&1 | tee "${run_root}/evaluation-${year}.log"

test -f "${run_root}/results-${year}/summary.json"
