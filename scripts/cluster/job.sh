#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: job.sh PROFILE -- COMMAND..." >&2
  exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
profile="$1"
shift
if [[ "$1" != "--" ]]; then
  echo "job.sh requires -- before the command" >&2
  exit 2
fi
shift

modules_csv="$(awk -F= '$1 == "MODULES" { print substr($0, index($0, "=") + 1); count++ } END { if (count != 1) exit 2 }' "$profile")"
IFS=',' read -r -a modules <<< "$modules_csv"
module load "${modules[@]}"
python3 "$repo_root/scripts/cluster/clusterctl.py" profile validate "$profile"
env_activate="$(python3 "$repo_root/scripts/cluster/clusterctl.py" profile get "$profile" ENV_ACTIVATE)"
runs_root="$(python3 "$repo_root/scripts/cluster/clusterctl.py" profile get "$profile" RUNS_ROOT)"
scratch_name="$(python3 "$repo_root/scripts/cluster/clusterctl.py" profile get "$profile" SCRATCH_ENV)"
source "$env_activate"

scratch_value="${!scratch_name:-}"
if [[ -n "$scratch_value" ]]; then
  export TMPDIR="$scratch_value"
fi
run_id="${SLURM_JOB_ID:-manual}-${SLURM_ARRAY_TASK_ID:-0}"
run_dir="$runs_root/$run_id"

python "$repo_root/scripts/cluster/clusterctl.py" run start "$profile" "$run_dir" -- "$@"
cd "$run_dir"
set +e
"$@"
exit_code=$?
set -e
python "$repo_root/scripts/cluster/clusterctl.py" run finish "$run_dir" "$exit_code"
exit "$exit_code"
