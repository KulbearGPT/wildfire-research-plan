#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Slurm required}"

# One-off fast path for the blocking X19/X22 confirmations and X23 screen.
# Each lane uses one H100. A second T1 job follows only on lanes 5 and 6.
# Existing jobs are authoritative: completed/running work is never duplicated,
# while a still-pending duplicate is cancelled immediately before its bundle
# replacement starts.
run_one() {
  local old_job=$1
  local tag=$2
  local history=$3
  local method=$4
  local seed=$5
  local memory=$6
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
      scancel "${old_job}"
      echo "REPLACE ${tag}: cancelled pending ${old_job}"
      ;;
  esac

  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=8 \
    --mem="${memory}" --gres=gpu:h100:1 \
    env WILDFIRE_RUN_TAG="${tag}" WILDFIRE_SOURCE_COMMIT="${WILDFIRE_SOURCE_COMMIT:-HEAD}" \
    bash reproductions/cross_history/run_slurm.sh \
      "${history}" "${method}" --seed "${seed}" --batch-size 64 --workers 7
}

lane_0() { run_one 21244236 x19-t5-s1 5 block_specialist_severity_adapter 1 128G; }
lane_1() { run_one 21244242 x19-t5-s2 5 block_specialist_severity_adapter 2 128G; }
lane_2() { run_one 21244239 x22-t5-s1 5 cosine_erm 1 128G; }
lane_3() { run_one 21244244 x22-t5-s2 5 cosine_erm 2 128G; }
lane_4() { run_one 21244246 x23-t5-s0 5 block_specialist_impact 0 128G; }
lane_5() {
  run_one 21244235 x19-t1-s1 1 block_specialist_severity_adapter 1 64G
  run_one 21244245 x23-t1-s0 1 block_specialist_impact 0 64G
}
lane_6() {
  run_one 21244241 x19-t1-s2 1 block_specialist_severity_adapter 2 64G
  run_one 21244243 x22-t1-s2 1 cosine_erm 2 64G
}
lane_7() { run_one 21244237 x22-t1-s1 1 cosine_erm 1 64G; }

fail=0
for lane in {0..7}; do
  "lane_${lane}" >"/project/6085198/kulbear/wildfire/slurm/X19-X23-bundle-${SLURM_JOB_ID}-lane${lane}.out" 2>&1 &
done
for pid in $(jobs -pr); do
  wait "${pid}" || fail=1
done
exit "${fail}"
