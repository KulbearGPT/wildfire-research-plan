#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_ID REPLICATION_MANIFEST" >&2
  exit 2
fi
run_id=$1
manifest_path=$(realpath "$2")
case "${run_id}" in
  C00-S1-10K|C00-S2-10K|C02-S1-10K|C02-S2-10K) ;;
  *)
    echo "run must be C00-S1-10K, C00-S2-10K, C02-S1-10K, or C02-S2-10K" >&2
    exit 2
    ;;
esac

started_at_epoch=$(date +%s)
script_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "${script_root}" rev-parse --show-toplevel)
base=/project/6085198/kulbear/wildfire
upstream=${base}/cache/WildfireSpreadTS-res18-runtime
data=${base}/hdf5/wstsplus-active-fixed
data_evidence=${base}/runs/nibi-wstsplus-data-20260823
stats=${data_evidence}/train-2016-2020-stats.npz
run_root=${base}/runs/fast-track-${run_id}-${SLURM_JOB_ID}

test -f "${manifest_path}"
test -f "${data_evidence}/completed.json"
test -f "${data_evidence}/data-summary.json"
test -f "${stats}"
if [[ -e "${run_root}" ]]; then
  echo "refusing existing run root: ${run_root}" >&2
  exit 2
fi

module purge
module load StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.2
source "${base}/envs/wsts-res18-t1-nibi-smoke/bin/activate"

python - "${manifest_path}" "${run_id}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
run_id = sys.argv[2]
payload = json.loads(path.read_text(encoding="utf-8"))
expected_seeds = {
    "C00-S1-10K": ("C00", 1),
    "C00-S2-10K": ("C00", 2),
    "C02-S1-10K": ("C02", 1),
    "C02-S2-10K": ("C02", 2),
}
experiment, seed = expected_seeds[run_id]
run = payload.get("run")
if not isinstance(run, dict):
    raise SystemExit("manifest run is missing")
required = {
    "run_id": run_id,
    "experiment_id": experiment,
    "stage": "replication",
    "seed": seed,
    "max_steps": 10000,
    "prerequisites": ["C00-S0-10K", "C02-S0-10K"],
    "launch_state": "gated",
}
if run != required:
    raise SystemExit("manifest run contract does not match the requested replication")
expected_boundary = json.loads(
    '{"train_years":[2016,2017,2018,2019,2020],"validation_years":[2021],'
    '"test_enabled": false,"withheld_years":[2022,2023]}'
)
if payload.get("boundary") != expected_boundary:
    raise SystemExit("manifest split or test boundary is invalid")
prerequisites = payload.get("prerequisites")
if not isinstance(prerequisites, list) or [
    item.get("run_id") for item in prerequisites if isinstance(item, dict)
] != ["C00-S0-10K", "C02-S0-10K"]:
    raise SystemExit("manifest does not contain the reviewed seed-0 prerequisite pair")
command = payload.get("command")
if not isinstance(command, list) or "--run-id" not in command:
    raise SystemExit("manifest command is missing the run selector")
selector = command.index("--run-id")
if selector + 1 >= len(command) or command[selector + 1] != run_id:
    raise SystemExit("manifest command run identity does not match")
PY

mkdir -p "${run_root}/project"
cp "${manifest_path}" "${run_root}/replication-manifest.json"
cp "${script_root}/run_replication_on_nibi.sh" "${run_root}/"
git -C "${repo}" rev-parse HEAD > "${run_root}/project-commit.txt"
git -C "${repo}" status --porcelain=v1 > "${run_root}/project-status.txt"
git -C "${repo}" archive --format=tar HEAD | tar -xf - -C "${run_root}/project"
git -C "${upstream}" rev-parse HEAD > "${run_root}/upstream-commit.txt"
git -C "${upstream}" status --porcelain=v1 > "${run_root}/upstream-status.txt"
git -C "${upstream}" diff --binary > "${run_root}/derived-runtime.diff"
python -m pip freeze > "${run_root}/pip-freeze.txt"
module list > "${run_root}/modules.txt" 2>&1
nvidia-smi -q > "${run_root}/nvidia-smi.txt"
sha256sum "${manifest_path}" > "${run_root}/replication-manifest.sha256"
sha256sum "${stats}" > "${run_root}/training-stats.sha256"
cp "${data_evidence}/data-summary.json" "${run_root}/data-summary.json"

cd "${run_root}/project"
export WANDB_MODE=disabled
export WANDB_SILENT=true
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

printf '%q ' python -m reproductions.wsts_fast_track.entrypoint \
  --upstream-root "${upstream}" \
  --run-id "${run_id}" \
  --data-root "${data}" \
  --run-root "${run_root}" \
  --stats-path "${stats}" > "${run_root}/command.txt"
printf '\n' >> "${run_root}/command.txt"

python -m reproductions.wsts_fast_track.entrypoint \
  --upstream-root "${upstream}" \
  --run-id "${run_id}" \
  --data-root "${data}" \
  --run-root "${run_root}" \
  --stats-path "${stats}" 2>&1 | tee "${run_root}/training.log"

python -m reproductions.wsts_fast_track.completion \
  --run-id "${run_id}" \
  --run-root "${run_root}" \
  --started-at-epoch "${started_at_epoch}" \
  --slurm-job-id "${SLURM_JOB_ID}"
