#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 D12_CHECKPOINT D2_CHECKPOINT" >&2
  exit 2
fi

repo=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
runner=${repo}/reproductions/wsts_fast_track/run_reliability_evaluation_on_nibi.sh

"${runner}" sarp "$1" 2022 D12-SARP
"${runner}" sarp "$1" 2023 D12-SARP
"${runner}" d2 "$2" 2022 D2-STD
"${runner}" d2 "$2" 2023 D2-STD
