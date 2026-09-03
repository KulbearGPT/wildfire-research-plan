#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 BASELINE_ID" >&2
  exit 2
fi
baseline_id=$1
case "${baseline_id}" in
  B0|B1|B2|B3|B4) ;;
  *)
    echo "baseline ID must be B0, B1, B2, B3, or B4" >&2
    exit 2
    ;;
esac

started_at_epoch=$(date +%s)
runner_path=$(realpath "${BASH_SOURCE[0]}")
repo=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
base=/project/6085198/kulbear/wildfire
upstream=${base}/cache/WildfireSpreadTS-res18-runtime
data=${base}/hdf5/wstsplus-active-fixed
data_evidence=${base}/runs/nibi-wstsplus-data-20260823
stats=${data_evidence}/train-2016-2020-stats.npz
run_root=${base}/runs/corrected-${baseline_id}-S0-3K-${SLURM_JOB_ID}
record=${run_root}/completed.json
output_root=${run_root}/results-2021

test -f "${data_evidence}/completed.json"
test -f "${stats}"
if [[ -e "${run_root}" ]]; then
  echo "refusing existing run root: ${run_root}" >&2
  exit 2
fi

module purge
module load StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.2
source "${base}/envs/wsts-res18-t1-nibi-smoke/bin/activate"

mkdir -p "${run_root}/project"
cp "${runner_path}" "${run_root}/run_corrected_baseline_on_nibi.sh"
git -C "${repo}" rev-parse HEAD > "${run_root}/project-commit.txt"
git -C "${repo}" status --porcelain=v1 > "${run_root}/project-status.txt"
git -C "${repo}" archive --format=tar HEAD | tar -xf - -C "${run_root}/project"
git -C "${upstream}" rev-parse HEAD > "${run_root}/upstream-commit.txt"
git -C "${upstream}" status --porcelain=v1 > "${run_root}/upstream-status.txt"
git -C "${upstream}" diff --binary > "${run_root}/derived-runtime.diff"
python -m pip freeze > "${run_root}/pip-freeze.txt"
module list > "${run_root}/modules.txt" 2>&1
nvidia-smi -q > "${run_root}/nvidia-smi.txt"
scontrol show job -dd "${SLURM_JOB_ID}" > "${run_root}/slurm-job.txt"
sha256sum "${stats}" > "${run_root}/training-stats.sha256"
cp "${data_evidence}/data-summary.json" "${run_root}/data-summary.json"

cd "${run_root}/project"
export WANDB_MODE=disabled
export WANDB_SILENT=true
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

printf '%q ' python -m reproductions.wsts_fast_track.train_corrected_baseline \
  --baseline-id "${baseline_id}" \
  --upstream-root "${upstream}" \
  --data-root "${data}" \
  --run-root "${run_root}" \
  --stats-path "${stats}" > "${run_root}/command.txt"
printf '\n' >> "${run_root}/command.txt"

python -m reproductions.wsts_fast_track.train_corrected_baseline \
  --baseline-id "${baseline_id}" \
  --upstream-root "${upstream}" \
  --data-root "${data}" \
  --run-root "${run_root}" \
  --stats-path "${stats}" 2>&1 | tee "${run_root}/training.log"

python -m reproductions.wsts_fast_track.complete_corrected_baseline \
  --baseline-id "${baseline_id}" \
  --run-root "${run_root}" \
  --started-at-epoch "${started_at_epoch}" \
  --slurm-job-id "${SLURM_JOB_ID}"

python -m reproductions.wsts_fast_track.evaluate_corrected_baseline \
  --record "${record}" \
  --output-root "${output_root}" \
  --upstream-root "${upstream}" \
  --data-root "${data}" \
  --stats-path "${stats}" \
  --device cuda 2>&1 | tee "${run_root}/evaluation-2021.log"

test -f "${output_root}/summary.json"
