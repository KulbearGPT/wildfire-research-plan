#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: bootstrap.sh PROFILE ENV_DIR REQUIREMENTS_FILE" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
profile="$1"
env_dir="$2"
requirements="$3"

modules_csv="$(awk -F= '$1 == "MODULES" { print substr($0, index($0, "=") + 1); count++ } END { if (count != 1) exit 2 }' "$profile")"
IFS=',' read -r -a modules <<< "$modules_csv"
module load "${modules[@]}"
python3 "$repo_root/scripts/cluster/clusterctl.py" profile validate "$profile"

if [[ -e "$env_dir" ]] && [[ -n "$(find "$env_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "environment directory already exists and is nonempty: $env_dir" >&2
  exit 2
fi
if [[ ! -f "$requirements" ]]; then
  echo "requirements file does not exist: $requirements" >&2
  exit 2
fi

virtualenv --no-download "$env_dir"
source "$env_dir/bin/activate"
python -m pip install --no-index -r "$requirements"
