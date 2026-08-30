#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [provenance|model]" >&2
  exit 2
fi
gate=${1:-provenance}
if [[ "${gate}" != provenance && "${gate}" != model ]]; then
  echo "gate must be provenance or model" >&2
  exit 2
fi

runner_path=$(realpath "${BASH_SOURCE[0]}")
repo=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
base=/project/6085198/kulbear/wildfire
archive=${base}/downloads/WildfireSpreadTS.zip
cache_root=${base}/cache/viirs-c2
token_file=${HOME}/.config/wildfire/earthdata_token
run_root=${base}/runs/viirs-reliability-24-${gate}-${SLURM_JOB_ID}
output_root=${run_root}/output

test -f "${archive}"
test -f "${token_file}"
if [[ $(stat -c '%a' "${token_file}") != 600 ]]; then
  echo "Earthdata token file must have mode 600" >&2
  exit 2
fi
if [[ -e "${run_root}" ]]; then
  echo "refusing existing run root: ${run_root}" >&2
  exit 2
fi

module purge
module load StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.2
source "${base}/envs/wsts-res18-t1-nibi-smoke/bin/activate"

mkdir -p "${run_root}/project"
cp "${runner_path}" "${run_root}/run_viirs_reliability_on_nibi.sh"
git -C "${repo}" rev-parse HEAD > "${run_root}/project-commit.txt"
git -C "${repo}" status --porcelain=v1 > "${run_root}/project-status.txt"
git -C "${repo}" archive --format=tar HEAD | tar -xf - -C "${run_root}/project"
python -m pip freeze > "${run_root}/pip-freeze.txt"
module list > "${run_root}/modules.txt" 2>&1

cd "${run_root}/project"
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

printf '%q ' python -m reproductions.wsts_fast_track.viirs_reliability \
  --archive "${archive}" \
  --output-root "${output_root}" \
  --cache-root "${cache_root}" \
  --token-file '<mode-600-token-file>' \
  --gate "${gate}" > "${run_root}/command.txt"
printf '\n' >> "${run_root}/command.txt"

python -m reproductions.wsts_fast_track.viirs_reliability \
  --archive "${archive}" \
  --output-root "${output_root}" \
  --cache-root "${cache_root}" \
  --token-file "${token_file}" \
  --gate "${gate}" 2>&1 | tee "${run_root}/extraction.log"

test -f "${output_root}/manifest.json"
