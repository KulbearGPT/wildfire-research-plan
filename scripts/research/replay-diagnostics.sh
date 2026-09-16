#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?diagnostic replay requires Slurm}"
: "${WILDFIRE_ROOT:?}" "${WILDFIRE_UPSTREAM:?}" "${WILDFIRE_DATA:?}" "${WILDFIRE_STATS:?}"
diagnostics=${1:-"$WILDFIRE_ROOT/diagnostics"}
output=${2:-"$WILDFIRE_ROOT/qualification/diagnostics-$SLURM_JOB_ID"}
mkdir -p "$(dirname "$output")"
mkdir "$output"
common=(--p00-record "$diagnostics/p00-completed.json"
        --attention-checkpoint "$diagnostics/attention.pt"
        --artifact-map "$diagnostics/artifact-map.json"
        --upstream-root "$WILDFIRE_UPSTREAM" --data-root "$WILDFIRE_DATA"
        --stats-path "$WILDFIRE_STATS" --batch-size 8 --device cuda)
python -m reproductions.wsts_fast_track.evaluate_viirs_reliability "${common[@]}" \
  --reliability-root "$diagnostics/input" --output "$output/natural.json"
python -m reproductions.wsts_fast_track.evaluate_viirs_target_quality "${common[@]}" \
  --input-reliability-root "$diagnostics/input" --target-reliability-root "$diagnostics/target" \
  --output "$output/target-qa.json"
