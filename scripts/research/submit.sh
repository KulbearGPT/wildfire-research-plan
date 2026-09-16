#!/usr/bin/env bash
set -euo pipefail
if (( $# < 2 )) || [[ $1 != cpu && $1 != gpu ]]; then
  echo 'usage: WILDFIRE_SITE_ENV=/absolute/site.env submit.sh cpu|gpu ACTION [ARGS...]' >&2
  exit 2
fi
resource=$1
shift
: "${WILDFIRE_SITE_ENV:?set WILDFIRE_SITE_ENV to a trusted Bash site configuration}"
site=$(realpath "$WILDFIRE_SITE_ENV")
source "$site"
: "${WILDFIRE_REPO:?set WILDFIRE_REPO}" "${WILDFIRE_ROOT:?set WILDFIRE_ROOT}"
repo=$(realpath "$WILDFIRE_REPO")
commit=$(git -C "$repo" rev-parse HEAD)
mkdir -p "$WILDFIRE_ROOT/jobs"
run=$(mktemp -d "$WILDFIRE_ROOT/jobs/$(date -u +%Y%m%dT%H%M%SZ)-${commit:0:12}-XXXXXX")
mkdir "$run/source"
git -C "$repo" archive "$commit" | tar -x -C "$run/source"
printf '%s\n' "$commit" > "$run/source-commit.txt"
git -C "$repo" status --porcelain=v1 > "$run/submission-status.txt"
cp "$site" "$run/site-at-submission.env"
if [[ ! -f "$run/source/scripts/research/job.sh" ]]; then
  echo 'Committed snapshot lacks scripts/research/job.sh; commit the handoff before submitting.' >&2
  exit 2
fi
if [[ $resource == cpu ]]; then
  declare -p WILDFIRE_CPU_SBATCH >/dev/null 2>&1 || { echo 'configure WILDFIRE_CPU_SBATCH array' >&2; exit 2; }
  flags=("${WILDFIRE_CPU_SBATCH[@]}")
else
  declare -p WILDFIRE_GPU_SBATCH >/dev/null 2>&1 || { echo 'configure WILDFIRE_GPU_SBATCH array' >&2; exit 2; }
  flags=("${WILDFIRE_GPU_SBATCH[@]}")
fi
printf '%q ' "$@" > "$run/command.txt"
printf '\n' >> "$run/command.txt"
job=$(sbatch --parsable "${flags[@]}" --export=ALL --chdir="$run/source" \
  --output="$run/slurm-%j.out" --error="$run/slurm-%j.err" \
  "$run/source/scripts/research/job.sh" "$run/source" "$run/site-at-submission.env" "$@")
printf '%s\n' "$job" > "$run/job-id.txt"
printf 'Submitted %s\nSource commit: %s\nRun directory: %s\n' "$job" "$commit" "$run"
