#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 INPUT_RELIABILITY_ROOT TARGET_RELIABILITY_ROOT" >&2
  exit 2
fi

input_root=$(realpath "$1")
target_root=$(realpath "$2")
runner_path=$(realpath "${BASH_SOURCE[0]}")
repo=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
base=/project/6085198/kulbear/wildfire
upstream=${base}/cache/WildfireSpreadTS-res18-runtime
data=${base}/hdf5/wstsplus-active-fixed
stats=${base}/runs/nibi-wstsplus-data-20260823/train-2016-2020-stats.npz
p00_record=${base}/runs/prototype-P00-FireDrop-C00-20398173/completed.json
attention_checkpoint=${base}/runs/prototype-fire-belief-attention-t1-20655225/belief-state.pt
run_root=${base}/runs/viirs-target-quality-${SLURM_JOB_ID}
output=${run_root}/result.json

test -f "${input_root}/manifest.json"
test -f "${target_root}/manifest.json"
test -f "${p00_record}"
test -f "${attention_checkpoint}"
test -f "${stats}"
if [[ -e "${run_root}" ]]; then
  echo "refusing existing run root: ${run_root}" >&2
  exit 2
fi

module purge
module load StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.2
source "${base}/envs/wsts-res18-t1-nibi-smoke/bin/activate"

mkdir -p "${run_root}/project"
cp "${runner_path}" "${run_root}/run_viirs_target_quality_on_nibi.sh"
git -C "${repo}" rev-parse HEAD > "${run_root}/project-commit.txt"
git -C "${repo}" status --porcelain=v1 > "${run_root}/project-status.txt"
git -C "${repo}" archive --format=tar HEAD | tar -xf - -C "${run_root}/project"
python -m pip freeze > "${run_root}/pip-freeze.txt"
module list > "${run_root}/modules.txt" 2>&1

cd "${run_root}/project"
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

printf '%q ' python -m reproductions.wsts_fast_track.evaluate_viirs_target_quality \
  --p00-record "${p00_record}" \
  --attention-checkpoint "${attention_checkpoint}" \
  --input-reliability-root "${input_root}" \
  --target-reliability-root "${target_root}" \
  --output "${output}" \
  --upstream-root "${upstream}" \
  --data-root "${data}" \
  --stats-path "${stats}" \
  --batch-size 8 \
  --device cuda > "${run_root}/command.txt"
printf '\n' >> "${run_root}/command.txt"

python -m reproductions.wsts_fast_track.evaluate_viirs_target_quality \
  --p00-record "${p00_record}" \
  --attention-checkpoint "${attention_checkpoint}" \
  --input-reliability-root "${input_root}" \
  --target-reliability-root "${target_root}" \
  --output "${output}" \
  --upstream-root "${upstream}" \
  --data-root "${data}" \
  --stats-path "${stats}" \
  --batch-size 8 \
  --device cuda 2>&1 | tee "${run_root}/evaluation.log"

test -f "${output}"
