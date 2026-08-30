#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?run inside a Slurm compute allocation}"
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
archive=${base}/downloads/WildfireSpreadTS.zip
cache_root=${base}/cache/viirs-c2
token_file=${HOME}/.config/wildfire/earthdata_token
run_root=${base}/runs/target-qa-cohort-prepare-${SLURM_JOB_ID}
cohort=${run_root}/cohort.json
reliability=${run_root}/target-reliability

test -f "${archive}"
test -f "${stats}"
test -f "${token_file}"
if [[ $(stat -c '%a' "${token_file}") != 600 ]]; then
  echo "Earthdata token file must have mode 600" >&2
  exit 2
fi
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

python -m reproductions.wsts_fast_track.prepare_censored_cohort \
  --upstream-root "${upstream}" \
  --data-root "${data}" \
  --stats-path "${stats}" \
  --output "${cohort}" 2>&1 | tee "${run_root}/selection.log"

python -m reproductions.wsts_fast_track.viirs_reliability \
  --archive "${archive}" \
  --output-root "${reliability}" \
  --cache-root "${cache_root}" \
  --token-file "${token_file}" \
  --gate training-cohort-target \
  --cohort-manifest "${cohort}" 2>&1 | tee "${run_root}/extraction.log"

test -f "${cohort}"
test -f "${reliability}/manifest.json"
sha256sum "${cohort}" "${reliability}/manifest.json" > "${run_root}/outputs.sha256"
