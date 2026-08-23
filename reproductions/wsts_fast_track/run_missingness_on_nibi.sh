#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 MISSINGNESS_MANIFEST EVALUATION_ID" >&2
  exit 2
fi
manifest_path=$(realpath "$1")
evaluation_id=$2
if [[ ! "${evaluation_id}" =~ ^[A-Za-z0-9-]+$ ]]; then
  echo "evaluation ID contains unsupported characters" >&2
  exit 2
fi

script_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "${script_root}" rev-parse --show-toplevel)
base=/project/6085198/kulbear/wildfire
upstream=${base}/cache/WildfireSpreadTS-res18-runtime
data=${base}/hdf5/wstsplus-active-fixed
data_evidence=${base}/runs/nibi-wstsplus-data-20260823
stats=${data_evidence}/train-2016-2020-stats.npz
run_root=${base}/runs/missingness-job-${evaluation_id}-${SLURM_JOB_ID}

test -f "${manifest_path}"
test -f "${data_evidence}/completed.json"
test -f "${stats}"
if [[ -e "${run_root}" ]]; then
  echo "refusing existing run root: ${run_root}" >&2
  exit 2
fi

module purge
module load StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.2
source "${base}/envs/wsts-res18-t1-nibi-smoke/bin/activate"

result_path=$(python - "${manifest_path}" "${evaluation_id}" <<'PY'
import json
import sys
from pathlib import Path
from reproductions.wsts_fast_track.evaluate_missingness import select_task

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
selected = select_task(manifest, sys.argv[2])
print(selected.task["output"])
PY
)
if [[ -e "${result_path}" ]]; then
  echo "refusing existing result: ${result_path}" >&2
  exit 2
fi

mkdir -p "${run_root}/project"
cp "${manifest_path}" "${run_root}/missingness-manifest.json"
cp "${script_root}/run_missingness_on_nibi.sh" "${run_root}/"
printf '%s\n' "${result_path}" > "${run_root}/result-path.txt"
git -C "${repo}" rev-parse HEAD > "${run_root}/project-commit.txt"
git -C "${repo}" status --porcelain=v1 > "${run_root}/project-status.txt"
git -C "${repo}" archive --format=tar HEAD | tar -xf - -C "${run_root}/project"
git -C "${upstream}" rev-parse HEAD > "${run_root}/upstream-commit.txt"
git -C "${upstream}" status --porcelain=v1 > "${run_root}/upstream-status.txt"
git -C "${upstream}" diff --binary > "${run_root}/derived-runtime.diff"
python -m pip freeze > "${run_root}/pip-freeze.txt"
module list > "${run_root}/modules.txt" 2>&1
nvidia-smi -q > "${run_root}/nvidia-smi.txt"
sha256sum "${manifest_path}" > "${run_root}/missingness-manifest.sha256"
sha256sum "${stats}" > "${run_root}/training-stats.sha256"

cd "${run_root}/project"
export WANDB_MODE=disabled
export WANDB_SILENT=true
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

printf '%q ' python -m reproductions.wsts_fast_track.evaluate_missingness \
  --manifest "${manifest_path}" \
  --evaluation-id "${evaluation_id}" \
  --upstream-root "${upstream}" \
  --data-root "${data}" \
  --stats-path "${stats}" \
  --device cuda > "${run_root}/command.txt"
printf '\n' >> "${run_root}/command.txt"

python -m reproductions.wsts_fast_track.evaluate_missingness \
  --manifest "${manifest_path}" \
  --evaluation-id "${evaluation_id}" \
  --upstream-root "${upstream}" \
  --data-root "${data}" \
  --stats-path "${stats}" \
  --device cuda 2>&1 | tee "${run_root}/evaluation.log"

test -f "${result_path}"
