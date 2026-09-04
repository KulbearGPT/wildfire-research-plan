#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 KIND INPUT YEAR LABEL" >&2
  exit 2
fi
kind=$1
input=$(realpath "$2")
year=$3
label=$4
case "${kind}" in baseline|d1|d2|ciwc|circ|cra|ffca|cepr|rpp|crpp|sarp) ;; *) echo "kind must be baseline, d1, d2, ciwc, circ, cra, ffca, cepr, rpp, crpp, or sarp" >&2; exit 2 ;; esac
case "${year}" in 2022|2023) ;; *) echo "year must be 2022 or 2023" >&2; exit 2 ;; esac
[[ "${label}" =~ ^[A-Za-z0-9-]+$ ]] || { echo "invalid label" >&2; exit 2; }

runner_path=$(realpath "${BASH_SOURCE[0]}")
repo=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
base=/project/6085198/kulbear/wildfire
upstream=${base}/cache/WildfireSpreadTS-res18-runtime
data=${base}/hdf5/wstsplus-active-fixed
stats=${base}/runs/nibi-wstsplus-data-20260823/train-2016-2020-stats.npz
run_root=${base}/runs/heldout-${label}-${year}-${SLURM_JOB_ID}
output_root=${run_root}/results-${year}

test -f "${input}"
test -f "${stats}"
if [[ -e "${run_root}" ]]; then
  echo "refusing existing run root: ${run_root}" >&2
  exit 2
fi

module purge
module load StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.2
source "${base}/envs/wsts-res18-t1-nibi-smoke/bin/activate"

mkdir -p "${run_root}/project"
cp "${runner_path}" "${run_root}/run_reliability_evaluation_on_nibi.sh"
git -C "${repo}" rev-parse HEAD > "${run_root}/project-commit.txt"
git -C "${repo}" status --porcelain=v1 > "${run_root}/project-status.txt"
git -C "${repo}" archive --format=tar HEAD | tar -xf - -C "${run_root}/project"
git -C "${upstream}" rev-parse HEAD > "${run_root}/upstream-commit.txt"
sha256sum "${input}" "${stats}" > "${run_root}/inputs.sha256"
scontrol show job -dd "${SLURM_JOB_ID}" > "${run_root}/slurm-job.txt"

cd "${run_root}/project"
export WANDB_MODE=disabled
export WANDB_SILENT=true
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

common=(--output-root "${output_root}" --upstream-root "${upstream}" --data-root "${data}" --stats-path "${stats}" --year "${year}" --heldout-authorized --device cuda)
case "${kind}" in
  baseline) python -m reproductions.wsts_fast_track.evaluate_corrected_baseline --record "${input}" "${common[@]}" ;;
  d1) python -m reproductions.wsts_fast_track.evaluate_predictive_consistency --checkpoint "${input}" "${common[@]}" ;;
  d2) python -m reproductions.wsts_fast_track.evaluate_reliability_normalized --checkpoint "${input}" "${common[@]}" ;;
  ciwc) python -m reproductions.wsts_fast_track.evaluate_counterfactual_impact_consistency --checkpoint "${input}" "${common[@]}" ;;
  circ) python -m reproductions.wsts_fast_track.evaluate_counterfactual_rank_consistency --checkpoint "${input}" "${common[@]}" ;;
  cra) python -m reproductions.wsts_fast_track.evaluate_counterfactual_reliability_adapter --checkpoint "${input}" "${common[@]}" ;;
  ffca) python -m reproductions.wsts_fast_track.evaluate_counterfactual_reliability_adapter --checkpoint "${input}" "${common[@]}" ;;
  cepr) python -m reproductions.wsts_fast_track.evaluate_counterfactual_error_pair_ranking --checkpoint "${input}" "${common[@]}" ;;
  rpp) python -m reproductions.wsts_fast_track.evaluate_reliability_prompt_pyramid --checkpoint "${input}" "${common[@]}" ;;
  crpp) python -m reproductions.wsts_fast_track.evaluate_reliability_prompt_pyramid --checkpoint "${input}" "${common[@]}" ;;
  sarp) python -m reproductions.wsts_fast_track.evaluate_reliability_prompt_pyramid --checkpoint "${input}" "${common[@]}" ;;
esac 2>&1 | tee "${run_root}/evaluation-${year}.log"

test -f "${output_root}/summary.json"
