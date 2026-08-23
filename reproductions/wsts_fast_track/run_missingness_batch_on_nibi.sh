#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 MISSINGNESS_MANIFEST RUN_ID" >&2
  exit 2
fi
manifest_path=$(realpath "$1")
run_id=$2
case "${run_id}" in
  C00-S0-10K|C00-S1-10K|C00-S2-10K|C02-S0-10K|C02-S1-10K|C02-S2-10K) ;;
  *)
    echo "run ID is not a declared clean 10K run" >&2
    exit 2
    ;;
esac

repo=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
single_runner=${repo}/reproductions/wsts_fast_track/run_missingness_on_nibi.sh
test -x "${single_runner}"
test -f "${manifest_path}"

module purge
module load StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.2
source /project/6085198/kulbear/wildfire/envs/wsts-res18-t1-nibi-smoke/bin/activate
cd "${repo}"

mapfile -t evaluation_ids < <(python - "${manifest_path}" "${run_id}" <<'PY'
import json
import sys
from pathlib import Path
from reproductions.wsts_fast_track.evaluate_missingness import select_task

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
run_id = sys.argv[2]
tasks = manifest.get("tasks")
if not isinstance(tasks, list):
    raise SystemExit("manifest tasks must be a list")
matches = [
    task for task in tasks
    if isinstance(task, dict) and task["run_id"] == run_id
]
expected = 8 if manifest.get("mode") == "engineering" else 16
if len(matches) != expected:
    raise SystemExit(f"run requires exactly {expected} evaluation tasks")
for task in matches:
    evaluation_id = str(task["evaluation_id"])
    select_task(manifest, evaluation_id)
    print(evaluation_id)
PY
)

for evaluation_id in "${evaluation_ids[@]}"; do
  "${single_runner}" "${manifest_path}" "${evaluation_id}"
done
