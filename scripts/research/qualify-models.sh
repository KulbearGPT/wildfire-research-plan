#!/usr/bin/env bash
# Bounded real-data smoke cases; outputs are never formal experiment results.
set -euo pipefail
: "${SLURM_JOB_ID:?GPU qualification requires Slurm}"
: "${WILDFIRE_ROOT:?}"
(( $# >= 3 )) || { echo 'usage: qualify-models.sh cross|rf|routed HISTORY METHOD [METHOD...]' >&2; exit 2; }
family=$1
history=$2
shift 2
root="$WILDFIRE_ROOT/qualification/${SLURM_JOB_ID}"
mkdir -p "$root"
for method in "$@"; do
  output="$root/$family-t$history-$method"
  case "$family" in
    cross)
      python -m reproductions.cross_history.run --history "$history" --method "$method" \
        --seed 0 --batch-size 8 --workers 2 --smoke --output "$output"
      ;;
    rf)
      extra=()
      if [[ $method == bn ]]; then extra=(--bn-forward-mode batch_stats); fi
      python -m reproductions.cross_history.run_three_directions --history "$history" \
        --arm "$method" --seed 0 --batch-size 8 --workers 2 --smoke --output "$output" "${extra[@]}"
      ;;
    routed)
      : "${WILDFIRE_ROUTED_SOURCES:?}"
      python -m reproductions.three_directions.run --history "$history" --mode "$method" \
        --sources "$WILDFIRE_ROUTED_SOURCES" --seed 0 --workers 2 --smoke --output "$output"
      ;;
    *) echo "unknown qualification family: $family" >&2; exit 2 ;;
  esac
done
