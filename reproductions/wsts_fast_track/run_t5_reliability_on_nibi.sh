#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 VARIANT B5_COMPLETED_RECORD" >&2
  exit 2
fi
variant=$1
case "${variant}" in
  standard) candidate=D13-STD-T5 ;;
  sarp) candidate=D13-SARP-T5 ;;
  *)
    echo "variant must be standard or sarp" >&2
    exit 2
    ;;
esac
b5_record=$(realpath "$2")

runner_path=$(realpath "${BASH_SOURCE[0]}")
repo=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
base=/project/6085198/kulbear/wildfire
upstream=${base}/cache/WildfireSpreadTS-res18-runtime
data=${base}/hdf5/wstsplus-active-fixed
stats=${base}/runs/nibi-wstsplus-data-20260823/train-2016-2020-stats.npz
run_root=${base}/runs/${candidate}-S0-3K-${SLURM_JOB_ID}
checkpoint=${run_root}/${candidate}.pt
output_root=${run_root}/results-2021

test -f "${b5_record}"
test -f "${stats}"
if [[ -e "${run_root}" ]]; then
  echo "refusing existing run root: ${run_root}" >&2
  exit 2
fi

module purge
module load StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.2
source "${base}/envs/wsts-res18-t1-nibi-smoke/bin/activate"

mkdir -p "${run_root}/project"
cp "${b5_record}" "${run_root}/b5-completed.json"
cp "${runner_path}" "${run_root}/run_t5_reliability_on_nibi.sh"
git -C "${repo}" rev-parse HEAD > "${run_root}/project-commit.txt"
git -C "${repo}" status --porcelain=v1 > "${run_root}/project-status.txt"
git -C "${repo}" archive --format=tar HEAD | tar -xf - -C "${run_root}/project"
git -C "${upstream}" rev-parse HEAD > "${run_root}/upstream-commit.txt"
git -C "${upstream}" status --porcelain=v1 > "${run_root}/upstream-status.txt"
python -m pip freeze > "${run_root}/pip-freeze.txt"
module list > "${run_root}/modules.txt" 2>&1
nvidia-smi -q > "${run_root}/nvidia-smi.txt"
scontrol show job -dd "${SLURM_JOB_ID}" > "${run_root}/slurm-job.txt"
sha256sum "${b5_record}" "${stats}" > "${run_root}/inputs.sha256"

cd "${run_root}/project"
export WANDB_MODE=disabled
export WANDB_SILENT=true
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

python -m reproductions.wsts_fast_track.train_temporal_reliability_prompting \
  --b5-record "${b5_record}" \
  --upstream-root "${upstream}" \
  --data-root "${data}" \
  --stats-path "${stats}" \
  --output-path "${checkpoint}" \
  --variant "${variant}" \
  --device cuda 2>&1 | tee "${run_root}/training.log"

python -m reproductions.wsts_fast_track.evaluate_temporal_reliability_prompting \
  --checkpoint "${checkpoint}" \
  --output-root "${output_root}" \
  --upstream-root "${upstream}" \
  --data-root "${data}" \
  --stats-path "${stats}" \
  --device cuda 2>&1 | tee "${run_root}/evaluation-2021.log"

test -f "${output_root}/summary.json"
