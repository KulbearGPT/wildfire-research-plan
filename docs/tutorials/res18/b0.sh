#!/usr/bin/env bash
set -euo pipefail
source "${WF_TUTORIAL_ENV:?}"
source "$WF_REPO/docs/tutorials/res18/common.sh"
train_env
test -f "$WF_ROOT/hdf5/combined/READY"
new_run B0
export WANDB_DIR="$WF_RUN/work"
cd "$WF_REPO"
started=$(date +%s)
command=(python -m reproductions.wsts_fast_track.train_corrected_baseline \
  --baseline-id B0 --upstream-root "$WF_UPSTREAM" \
  --data-root "$WF_ROOT/hdf5/combined" --run-root "$WF_RUN/scientific" \
  --stats-path "$WF_ROOT/hdf5/train-stats.npz")
printf '%q ' "${command[@]}" > "$WF_RUN/command.txt"
printf '\n' >> "$WF_RUN/command.txt"
"${command[@]}" 2>&1 | tee "$WF_RUN/training.log"
# The existing trainer creates its own new work directory below scientific/.
cp "$WF_RUN/training.log" "$WF_RUN/scientific/training.log"
python -m reproductions.wsts_fast_track.complete_corrected_baseline \
  --baseline-id B0 --run-root "$WF_RUN/scientific" \
  --started-at-epoch "$started" --slurm-job-id "$SLURM_JOB_ID"
python -m reproductions.wsts_fast_track.evaluate_corrected_baseline \
  --record "$WF_RUN/scientific/completed.json" --output-root "$WF_RUN/results-2021" \
  --upstream-root "$WF_UPSTREAM" --data-root "$WF_ROOT/hdf5/combined" \
  --stats-path "$WF_ROOT/hdf5/train-stats.npz" --device cuda \
  2>&1 | tee "$WF_RUN/evaluation-2021.log"
