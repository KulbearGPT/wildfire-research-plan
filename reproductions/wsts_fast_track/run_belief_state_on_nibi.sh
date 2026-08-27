#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 P00_COMPLETED_RECORD {filter|attention|reconstruction} {1|5}" >&2
  exit 2
fi

: "${SLURM_JOB_ID:?run_belief_state_on_nibi.sh must run inside a Slurm allocation}"

p00_record=$(realpath "$1")
method=$2
history=$3
case "${method}" in
  filter|attention|reconstruction) ;;
  *) echo "method must be filter, attention, or reconstruction" >&2; exit 2 ;;
esac
case "${history}" in
  1|5) ;;
  *) echo "history must be 1 or 5" >&2; exit 2 ;;
esac

runner_path=$(realpath "${BASH_SOURCE[0]}")
repo=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
commit=$(git -C "${repo}" rev-parse HEAD)
project_status=$(git -C "${repo}" status --porcelain=v1)
if [[ -n "${project_status}" ]]; then
  echo "refusing to archive a dirty worktree" >&2
  exit 2
fi

base=/project/6085198/kulbear/wildfire
upstream=${base}/cache/WildfireSpreadTS-res18-runtime
data=${base}/hdf5/wstsplus-active-fixed
stats=${base}/runs/nibi-wstsplus-data-20260823/train-2016-2020-stats.npz
run_root=${base}/runs/prototype-fire-belief-${method}-t${history}-${SLURM_JOB_ID}
belief_checkpoint=${run_root}/belief-state.pt
output_root=${run_root}/results-2021

test -f "${p00_record}"
test -f "${stats}"
if [[ -e "${run_root}" ]]; then
  echo "refusing existing run root: ${run_root}" >&2
  exit 2
fi

module purge
module load StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.2
source "${base}/envs/wsts-res18-t1-nibi-smoke/bin/activate"

mkdir -p "${run_root}/project"
cp "${runner_path}" "${run_root}/run_belief_state_on_nibi.sh"
printf '%s\n' "${commit}" > "${run_root}/project-commit.txt"
printf '%s\n' "${project_status}" > "${run_root}/project-status.txt"
git -C "${repo}" archive --format=tar "${commit}" | tar -xf - -C "${run_root}/project"
sha256sum "${p00_record}" "${stats}" "${runner_path}" > "${run_root}/inputs.sha256"

cd "${run_root}/project"
export WANDB_MODE=disabled
export WANDB_SILENT=true
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

python -m reproductions.wsts_fast_track.train_belief_state \
  --method "${method}" \
  --history "${history}" \
  --p00-record "${p00_record}" \
  --upstream-root "${upstream}" \
  --data-root "${data}" \
  --stats-path "${stats}" \
  --output-path "${belief_checkpoint}" \
  --git-commit "${commit}" \
  --device cuda 2>&1 | tee "${run_root}/training.log"

python -m reproductions.wsts_fast_track.evaluate_belief_state \
  --p00-record "${p00_record}" \
  --belief-checkpoint "${belief_checkpoint}" \
  --output-root "${output_root}" \
  --upstream-root "${upstream}" \
  --data-root "${data}" \
  --stats-path "${stats}" \
  --device cuda 2>&1 | tee "${run_root}/evaluation-2021.log"

test -f "${output_root}/summary.json"
sha256sum "${belief_checkpoint}" "${output_root}/summary.json" > "${run_root}/outputs.sha256"
