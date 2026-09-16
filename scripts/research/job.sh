#!/usr/bin/env bash
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo 'Slurm allocation required before setup or runtime commands' >&2
  exit 2
fi
if (( $# < 3 )); then
  echo 'usage: job.sh SNAPSHOT SITE_ENV setup|audit|COMMAND [ARGS...]' >&2
  exit 2
fi
readonly research_snapshot=$(realpath "$1")
readonly research_site=$(realpath "$2")
shift 2
source "$research_site"
: "${WILDFIRE_REPO:?set WILDFIRE_REPO}" "${WILDFIRE_ROOT:?set WILDFIRE_ROOT}"
: "${WILDFIRE_UPSTREAM:?set WILDFIRE_UPSTREAM}" "${WILDFIRE_DATA:?set WILDFIRE_DATA}" "${WILDFIRE_STATS:?set WILDFIRE_STATS}"
configure_runtime_environment() {
  # Reapply after modules and activation: sites may restore vendor pip/Python paths.
  export WILDFIRE_REPO="$research_snapshot" WILDFIRE_SITE_ENV="$research_site"
  export WILDFIRE_ROOT WILDFIRE_UPSTREAM WILDFIRE_DATA WILDFIRE_STATS
  export PYTHONNOUSERSITE=1
  export PYTHONPATH="$research_snapshot/src:$research_snapshot:$WILDFIRE_UPSTREAM/src"
  export TORCH_HOME="$WILDFIRE_ROOT/cache/torch" HF_HOME="$WILDFIRE_ROOT/cache/huggingface"
  export WANDB_MODE=disabled WANDB_SILENT=true HDF5_USE_FILE_LOCKING=FALSE
  export PIP_NO_INDEX=0 PIP_CONFIG_FILE=/dev/null
  export PIP_INDEX_URL="${WILDFIRE_PIP_INDEX_URL:-https://pypi.org/simple}"
  unset PIP_EXTRA_INDEX_URL PIP_FIND_LINKS PYTHONHOME
  # Lightning 2.0 resumes trusted legacy checkpoints using torch.load without
  # weights_only; PyTorch 2.6 changed that default. Match the historical runner.
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
}
configure_runtime_environment
readonly research_logs="$(dirname "$research_snapshot")/allocation-${SLURM_JOB_ID}"
mkdir -p "$research_logs"
cd "$research_snapshot"
scontrol show job -dd "$SLURM_JOB_ID" > "$research_logs/slurm.txt"
printf '%q ' "$@" > "$research_logs/command.txt"
printf '\n' >> "$research_logs/command.txt"

load_modules() {
  local requested=$1
  if [[ -n $requested ]]; then
    local modules
    read -r -a modules <<< "$requested"
    module purge
    module load "${modules[@]}"
    module list 2> "$research_logs/modules-${2}.txt"
  fi
  configure_runtime_environment
}
record_upstream() {
  git -C "$WILDFIRE_UPSTREAM" rev-parse HEAD > "$research_logs/upstream-commit.txt"
  git -C "$WILDFIRE_UPSTREAM" diff --binary > "$research_logs/upstream.diff"
}
setup_runtime() {
  local training_env="$WILDFIRE_ROOT/envs/train" audit_env="$WILDFIRE_ROOT/envs/audit"
  for target in "$training_env" "$audit_env" "$WILDFIRE_UPSTREAM"; do
    if [[ -e $target ]] && [[ ! -d $target || -n $(ls -A "$target") ]]; then
      echo "Refusing nonempty setup target: $target" >&2
      exit 2
    fi
  done
  mkdir -p "$WILDFIRE_ROOT/envs" "$(dirname "$WILDFIRE_UPSTREAM")" "$TORCH_HOME" "$HF_HOME"
  load_modules "${WILDFIRE_TRAIN_MODULES:-}" train
  "${WILDFIRE_TRAIN_PYTHON:-python3.10}" -m venv "$training_env"
  "$training_env/bin/python" -m pip install --index-url "$PIP_INDEX_URL" -r "$research_snapshot/environments/research-training.txt"
  "$training_env/bin/python" -m pip freeze > "$research_logs/train-pip-freeze.txt"
  git clone https://github.com/slahrichi/WildfireSpreadTS.git "$WILDFIRE_UPSTREAM"
  git -C "$WILDFIRE_UPSTREAM" checkout --detach ed221d491fe2142a4b2e93462c2c0b7a1c7c31ad
  git -C "$WILDFIRE_UPSTREAM" apply "$research_snapshot/reproductions/wsts_res18_unet_t1/patches/res18_import_scope.patch"
  "$training_env/bin/python" - "$WILDFIRE_UPSTREAM" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1]) / 'src/dataloader/FireSpreadDataset.py'
line = 'from torch.utils.data.dataset import T_co\n'
text = path.read_text()
if text.count(line) != 1:
    raise ValueError('expected exactly one obsolete T_co import in pinned upstream')
path.write_text(text.replace(line, ''))
PY
  record_upstream
  "$training_env/bin/python" - <<'PY'
import segmentation_models_pytorch as smp
smp.encoders.get_encoder('resnet18', in_channels=40, weights='imagenet')
PY
  load_modules "${WILDFIRE_AUDIT_MODULES:-}" audit
  "${WILDFIRE_AUDIT_PYTHON:-python3.13}" -m venv "$audit_env"
  "$audit_env/bin/python" -m pip install --index-url "$PIP_INDEX_URL" -r "$research_snapshot/environments/research-audit.txt"
  "$audit_env/bin/python" -m pip install --index-url "$PIP_INDEX_URL" --no-deps "$research_snapshot"
  "$audit_env/bin/python" -m pip freeze > "$research_logs/audit-pip-freeze.txt"
  printf 'Setup completed from archived source.\n' > "$research_logs/setup-completed.txt"
}

if [[ $1 == setup ]]; then
  (( $# == 1 )) || { echo 'setup accepts no additional arguments' >&2; exit 2; }
  setup_runtime
  exit 0
fi
if [[ $1 == audit ]]; then
  shift
  (( $# )) || { echo 'audit requires a command' >&2; exit 2; }
  load_modules "${WILDFIRE_AUDIT_MODULES:-}" audit
  source "$WILDFIRE_ROOT/envs/audit/bin/activate"
else
  load_modules "${WILDFIRE_TRAIN_MODULES:-}" train
  source "$WILDFIRE_ROOT/envs/train/bin/activate"
fi
configure_runtime_environment
python -c 'from reproductions.paths import load_paths; load_paths(require_explicit=True)'
python -m pip freeze > "$research_logs/runtime-pip-freeze.txt"
record_upstream
exec "$@"
