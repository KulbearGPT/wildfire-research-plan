#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 COMPLETED_RECORD RUN_LABEL" >&2
  exit 2
fi

record_path=$(realpath "$1")
run_label=$2
case "${run_label}" in
  P00-M06|P02) ;;
  *) echo "run label must be P00-M06 or P02" >&2; exit 2 ;;
esac
runner_path=$(realpath "${BASH_SOURCE[0]}")
repo=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
base=/project/6085198/kulbear/wildfire
upstream=${base}/cache/WildfireSpreadTS-res18-runtime
data=${base}/hdf5/wstsplus-active-fixed
stats=${base}/runs/nibi-wstsplus-data-20260823/train-2016-2020-stats.npz
run_root=${base}/runs/prototype-${run_label}-eval-2021-${SLURM_JOB_ID}
output_root=${run_root}/results

test -f "${record_path}"
test -f "${stats}"
if [[ -e "${run_root}" ]]; then
  echo "refusing existing run root: ${run_root}" >&2
  exit 2
fi

module purge
module load StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.2
source "${base}/envs/wsts-res18-t1-nibi-smoke/bin/activate"

mkdir -p "${run_root}/project"
cp "${record_path}" "${run_root}/training-completed.json"
cp "${runner_path}" "${run_root}/run_block_prototype_evaluation_on_nibi.sh"
git -C "${repo}" rev-parse HEAD > "${run_root}/project-commit.txt"
git -C "${repo}" status --porcelain=v1 > "${run_root}/project-status.txt"
git -C "${repo}" archive --format=tar HEAD | tar -xf - -C "${run_root}/project"
sha256sum "${record_path}" > "${run_root}/training-completed.sha256"
sha256sum "${stats}" > "${run_root}/training-stats.sha256"

cd "${run_root}/project"
export WANDB_MODE=disabled
export WANDB_SILENT=true
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

python -m reproductions.wsts_fast_track.evaluate_prototype \
  --record "${record_path}" \
  --output-root "${output_root}" \
  --upstream-root "${upstream}" \
  --data-root "${data}" \
  --stats-path "${stats}" \
  --year 2021 \
  --scenarios M00 M01 M06 M07 \
  --device cuda 2>&1 | tee "${run_root}/evaluation.log"

test -f "${output_root}/summary.json"
