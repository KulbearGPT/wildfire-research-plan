#!/usr/bin/env bash
set -euo pipefail
source "${WF_TUTORIAL_ENV:?}"
source "$WF_REPO/docs/tutorials/res18/common.sh"
module purge
module load StdEnv/2023 gcc/12.3 python/3.10.13
python "$WF_SCRIPTS/weights.py" fetch "$WF_REPO" "$WF_ROOT/weights" "${1:-2}"
