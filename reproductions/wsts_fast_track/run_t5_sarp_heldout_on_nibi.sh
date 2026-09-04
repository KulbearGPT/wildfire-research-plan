#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 STD_CHECKPOINT SARP_CHECKPOINT" >&2
  exit 2
fi
standard=$(realpath "$1")
sarp=$(realpath "$2")

runner_path=$(realpath "${BASH_SOURCE[0]}")
repo=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
base=/project/6085198/kulbear/wildfire
upstream=${base}/cache/WildfireSpreadTS-res18-runtime
data=${base}/hdf5/wstsplus-active-fixed
stats=${base}/runs/nibi-wstsplus-data-20260823/train-2016-2020-stats.npz
run_root=${base}/runs/heldout-D13-T5-${SLURM_JOB_ID}

test -f "${standard}"
test -f "${sarp}"
test -f "${stats}"
if [[ -e "${run_root}" ]]; then
  echo "refusing existing run root: ${run_root}" >&2
  exit 2
fi

module purge
module load StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.2
source "${base}/envs/wsts-res18-t1-nibi-smoke/bin/activate"

mkdir -p "${run_root}/project"
cp "${runner_path}" "${run_root}/run_t5_sarp_heldout_on_nibi.sh"
git -C "${repo}" rev-parse HEAD > "${run_root}/project-commit.txt"
git -C "${repo}" status --porcelain=v1 > "${run_root}/project-status.txt"
git -C "${repo}" archive --format=tar HEAD | tar -xf - -C "${run_root}/project"
git -C "${upstream}" rev-parse HEAD > "${run_root}/upstream-commit.txt"
git -C "${upstream}" status --porcelain=v1 > "${run_root}/upstream-status.txt"
python -m pip freeze > "${run_root}/pip-freeze.txt"
module list > "${run_root}/modules.txt" 2>&1
nvidia-smi -q > "${run_root}/nvidia-smi.txt"
scontrol show job -dd "${SLURM_JOB_ID}" > "${run_root}/slurm-job.txt"
sha256sum "${standard}" "${sarp}" "${stats}" > "${run_root}/inputs.sha256"

cd "${run_root}/project"
export WANDB_MODE=disabled
export WANDB_SILENT=true
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

for year in 2022 2023; do
  python -m reproductions.wsts_fast_track.evaluate_temporal_reliability_prompting \
    --checkpoint "${standard}" \
    --output-root "${run_root}/results-D13-STD-T5-${year}" \
    --upstream-root "${upstream}" \
    --data-root "${data}" \
    --stats-path "${stats}" \
    --year "${year}" \
    --heldout-authorized \
    --device cuda 2>&1 | tee "${run_root}/evaluation-D13-STD-T5-${year}.log"
  python -m reproductions.wsts_fast_track.evaluate_temporal_reliability_prompting \
    --checkpoint "${sarp}" \
    --output-root "${run_root}/results-D13-SARP-T5-${year}" \
    --upstream-root "${upstream}" \
    --data-root "${data}" \
    --stats-path "${stats}" \
    --year "${year}" \
    --heldout-authorized \
    --device cuda 2>&1 | tee "${run_root}/evaluation-D13-SARP-T5-${year}.log"
done

test -f "${run_root}/results-D13-STD-T5-2022/summary.json"
test -f "${run_root}/results-D13-SARP-T5-2022/summary.json"
test -f "${run_root}/results-D13-STD-T5-2023/summary.json"
test -f "${run_root}/results-D13-SARP-T5-2023/summary.json"
