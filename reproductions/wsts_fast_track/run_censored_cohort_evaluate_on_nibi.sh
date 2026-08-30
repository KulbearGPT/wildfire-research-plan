#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 LEGACY_CHECKPOINT CENSORED_CHECKPOINT INPUT_RELIABILITY_ROOT TARGET_RELIABILITY_ROOT" >&2
  exit 2
fi
: "${SLURM_JOB_ID:?run inside a Slurm compute allocation}"
legacy=$(realpath "$1")
censored=$(realpath "$2")
input_root=$(realpath "$3")
target_root=$(realpath "$4")
test -f "${legacy}"
test -f "${censored}"
test -f "${input_root}/manifest.json"
test -f "${target_root}/manifest.json"

runner_path=$(realpath "${BASH_SOURCE[0]}")
repo=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
commit=$(git -C "${repo}" rev-parse HEAD)
if [[ -n $(git -C "${repo}" status --porcelain=v1) ]]; then
  echo "refusing to archive a dirty worktree" >&2
  exit 2
fi
base=/project/6085198/kulbear/wildfire
upstream=${base}/cache/WildfireSpreadTS-res18-runtime
data=${base}/hdf5/wstsplus-active-fixed
stats=${base}/runs/nibi-wstsplus-data-20260823/train-2016-2020-stats.npz
p00_record=${base}/runs/prototype-P00-FireDrop-C00-20398173/completed.json
run_root=${base}/runs/target-qa-cohort-eval-${SLURM_JOB_ID}
output=${run_root}/result.json
test ! -e "${run_root}"

module purge
module load StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.2
source "${base}/envs/wsts-res18-t1-nibi-smoke/bin/activate"
mkdir -p "${run_root}/project"
cp "${runner_path}" "${run_root}/"
printf '%s\n' "${commit}" > "${run_root}/project-commit.txt"
git -C "${repo}" archive --format=tar "${commit}" | tar -xf - -C "${run_root}/project"
cd "${run_root}/project"
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

python -m reproductions.wsts_fast_track.evaluate_censored_cohort \
  --p00-record "${p00_record}" \
  --legacy-checkpoint "${legacy}" \
  --censored-checkpoint "${censored}" \
  --input-reliability-root "${input_root}" \
  --target-reliability-root "${target_root}" \
  --output "${output}" \
  --upstream-root "${upstream}" \
  --data-root "${data}" \
  --stats-path "${stats}" \
  --batch-size 8 \
  --device cuda 2>&1 | tee "${run_root}/evaluation.log"

test -f "${output}"
sha256sum "${output}" > "${run_root}/outputs.sha256"
