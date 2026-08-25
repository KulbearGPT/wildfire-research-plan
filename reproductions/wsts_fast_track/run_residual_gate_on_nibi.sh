#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 P00_COMPLETED_RECORD" >&2
  exit 2
fi

p00_record=$(realpath "$1")
runner_path=$(realpath "${BASH_SOURCE[0]}")
repo=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
base=/project/6085198/kulbear/wildfire
upstream=${base}/cache/WildfireSpreadTS-res18-runtime
data=${base}/hdf5/wstsplus-active-fixed
stats=${base}/runs/nibi-wstsplus-data-20260823/train-2016-2020-stats.npz
run_root=${base}/runs/prototype-P04-residual-gate-${SLURM_JOB_ID}
gate_checkpoint=${run_root}/gate.pt
output_root=${run_root}/results

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
cp "${p00_record}" "${run_root}/p00-completed.json"
cp "${runner_path}" "${run_root}/run_residual_gate_on_nibi.sh"
git -C "${repo}" rev-parse HEAD > "${run_root}/project-commit.txt"
git -C "${repo}" status --porcelain=v1 > "${run_root}/project-status.txt"
git -C "${repo}" archive --format=tar HEAD | tar -xf - -C "${run_root}/project"
sha256sum "${p00_record}" "${stats}" > "${run_root}/inputs.sha256"

cd "${run_root}/project"
export WANDB_MODE=disabled
export WANDB_SILENT=true
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

python -m reproductions.wsts_fast_track.train_residual_gate \
  --p00-record "${p00_record}" \
  --upstream-root "${upstream}" \
  --data-root "${data}" \
  --stats-path "${stats}" \
  --output-path "${gate_checkpoint}" \
  --device cuda 2>&1 | tee "${run_root}/training.log"

python -m reproductions.wsts_fast_track.evaluate_residual_gate \
  --p00-record "${p00_record}" \
  --gate-checkpoint "${gate_checkpoint}" \
  --output-root "${output_root}" \
  --upstream-root "${upstream}" \
  --data-root "${data}" \
  --stats-path "${stats}" \
  --device cuda 2>&1 | tee "${run_root}/evaluation.log"

test -f "${output_root}/summary.json"
