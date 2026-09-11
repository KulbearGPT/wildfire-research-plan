#!/usr/bin/env bash
set -euo pipefail
source "${WF_TUTORIAL_ENV:?}"
source "$WF_REPO/docs/tutorials/res18/common.sh"
record=${1:?Pass the B0 scientific/completed.json absolute path}
year=${2:?Pass 2022 or 2023}
case "$year" in 2022|2023) ;; *) exit 2;; esac
train_env
new_run "B0-test-$year"
cd "$WF_REPO"
python -m reproductions.wsts_fast_track.evaluate_corrected_baseline \
  --record "$record" --output-root "$WF_RUN/results-$year" \
  --upstream-root "$WF_UPSTREAM" --data-root "$WF_ROOT/hdf5/combined" \
  --stats-path "$WF_ROOT/hdf5/train-stats.npz" --device cuda \
  --year "$year" --heldout-authorized 2>&1 | tee "$WF_RUN/evaluation.log"
