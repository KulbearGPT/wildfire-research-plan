#!/usr/bin/env bash
set -euo pipefail
source "${WF_TUTORIAL_ENV:?}"
source "$WF_REPO/docs/tutorials/res18/common.sh"
mode=${1:?Use smoke, full, or weight}
fold=${2:-${SLURM_ARRAY_TASK_ID:-2}}
case "$mode" in smoke|full|weight) ;; *) echo 'Invalid mode' >&2; exit 2;; esac
[[ "$fold" =~ ^([0-9]|1[01])$ ]] || { echo 'Fold must be 0..11' >&2; exit 2; }
train_env
test -f "$WF_ROOT/hdf5/original/READY"
new_run "official-$mode-fold$fold"
export WANDB_DIR="$WF_RUN/work"
steps=10000
batch=64
test_enabled=true
extra=()
if [[ "$mode" == smoke ]]; then
  steps=1
  batch=4
  test_enabled=false
  extra+=(--trainer.limit_val_batches=1 --trainer.num_sanity_val_steps=1)
fi
args=(
  "--config=$WF_UPSTREAM/cfgs/unet/res18_monotemporal.yaml"
  "--trainer=$WF_UPSTREAM/cfgs/trainer_single_gpu.yaml"
  "--data=$WF_UPSTREAM/cfgs/data_monotemporal_full_features.yaml"
  --seed_everything=0
  "--data.data_dir=$WF_ROOT/hdf5/original"
  --data.additional_data=false "--data.data_fold_id=$fold"
  --data.features_to_keep=null --data.n_leading_observations=1
  --data.n_leading_observations_test_adjustment=5
  --data.remove_duplicate_features=true
  "--data.batch_size=$batch" --data.num_workers=8
  "--trainer.max_steps=$steps" "--trainer.default_root_dir=$WF_RUN/work"
  --trainer.logger.init_args.log_model=false
  --do_train=true --do_validate=false "--do_test=$test_enabled" --do_predict=false
  "${extra[@]}"
)
entry="$WF_REPO/reproductions/wsts_res18_unet_t1/scripts"
if [[ "$mode" == weight ]]; then
  weight=$(python - "$WF_REPO" "$fold" "$WF_ROOT" <<'PY'
import json, sys
from pathlib import Path
m = json.loads((Path(sys.argv[1]) / 'reproductions/wsts_res18_unet_t1/official_weights_manifest.json').read_text())
w = next(x for x in m['weights'] if x['fold_id'] == int(sys.argv[2]))
p = Path(sys.argv[3]) / 'weights' / w['filename']
assert p.is_file(), p
print(p)
PY
)
  # Exact hash is checked again before strict loading.
  python "$WF_SCRIPTS/weights.py" verify "$WF_REPO" "$WF_ROOT/weights" "$fold"
  command=(python "$entry/official_weight_entrypoint.py" --upstream-root "$WF_UPSTREAM"
    --weights-path "$weight" "${args[@]}" --do_train=false)
else
  command=(python "$entry/official_entrypoint.py" --upstream-root "$WF_UPSTREAM" "${args[@]}")
fi
printf '%q ' "${command[@]}" > "$WF_RUN/command.txt"
printf '\n' >> "$WF_RUN/command.txt"
"${command[@]}" 2>&1 | tee "$WF_RUN/training.log"
if [[ "$mode" == smoke ]]; then
  grep -F '`Trainer.fit` stopped: `max_steps=1` reached.' "$WF_RUN/training.log"
  grep -Eq 'WSTS_OBSERVER_PEAK_ALLOCATED_BYTES=[1-9][0-9]*' "$WF_RUN/training.log"
  touch "$WF_RUN/SMOKE_PASSED"
else
  python "$WF_SCRIPTS/results.py" record "$WF_RUN" "$mode" "$fold"
fi
