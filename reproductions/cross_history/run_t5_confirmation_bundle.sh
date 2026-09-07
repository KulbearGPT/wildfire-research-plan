#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Slurm required}"

# Four remaining T5 confirmations, one H100 per experiment.
run_one() {
  local old_job=$1
  local tag=$2
  local method=$3
  local seed=$4
  local state

  state=$(sacct -X -j "${old_job}" -n -P -o State | head -n 1 | cut -d'|' -f1)
  case "${state}" in
    COMPLETED*)
      echo "SKIP ${tag}: ${old_job} already completed"
      return 0
      ;;
    RUNNING*)
      echo "SKIP ${tag}: ${old_job} already running"
      return 0
      ;;
    PENDING*)
      scancel --state=PENDING "${old_job}"
      echo "REPLACE ${tag}: cancelled pending ${old_job}"
      ;;
  esac

  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=8 \
    --mem=128G --gres=gpu:h100:1 \
    env WILDFIRE_RUN_TAG="${tag}" WILDFIRE_SOURCE_COMMIT="${WILDFIRE_SOURCE_COMMIT:-HEAD}" \
    bash reproductions/cross_history/run_slurm.sh \
      5 "${method}" --seed "${seed}" --batch-size 64 --workers 7
}

run_one 21244236 x19-t5-s1 block_specialist_severity_adapter 1 &
run_one 21244242 x19-t5-s2 block_specialist_severity_adapter 2 &
run_one 21244239 x22-t5-s1 cosine_erm 1 &
run_one 21244244 x22-t5-s2 cosine_erm 2 &

fail=0
for pid in $(jobs -pr); do
  wait "${pid}" || fail=1
done
exit "${fail}"
