#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 {legacy|censored} PREPARE_RUN_ROOT" >&2
  exit 2
fi
: "${SLURM_JOB_ID:?run inside a Slurm compute allocation}"
objective=$1
case "${objective}" in legacy|censored) ;; *) exit 2 ;; esac
prepare_root=$(realpath "$2")
cohort=${prepare_root}/cohort.json
reliability=${prepare_root}/target-reliability
test -f "${cohort}"
test -f "${reliability}/manifest.json"

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
run_root=${base}/runs/target-qa-cohort-${objective}-${SLURM_JOB_ID}
checkpoint=${run_root}/cohort-model.pt
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
export WANDB_MODE=disabled

python -m reproductions.wsts_fast_track.train_censored_cohort \
  --objective "${objective}" \
  --p00-record "${p00_record}" \
  --cohort-manifest "${cohort}" \
  --target-reliability-root "${reliability}" \
  --upstream-root "${upstream}" \
  --data-root "${data}" \
  --stats-path "${stats}" \
  --output-path "${checkpoint}" \
  --batch-size 64 \
  --device cuda 2>&1 | tee "${run_root}/training.log"

test -f "${checkpoint}"
sha256sum "${checkpoint}" > "${run_root}/outputs.sha256"
