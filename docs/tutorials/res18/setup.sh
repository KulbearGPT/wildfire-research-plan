#!/usr/bin/env bash
set -euo pipefail
source "${WF_TUTORIAL_ENV:?}"
source "$WF_REPO/docs/tutorials/res18/common.sh"
test ! -e "$WF_ROOT/envs/train"
test ! -e "$WF_ROOT/envs/audit"
test ! -e "$WF_UPSTREAM"
module purge
module load StdEnv/2023 gcc/12.3 python/3.10.13 cuda/12.2
virtualenv --no-download "$WF_ROOT/envs/train"
source "$WF_ROOT/envs/train/bin/activate"
python -m pip install --no-index -r "$WF_SCRIPTS/requirements-training.txt"
git clone https://github.com/slahrichi/WildfireSpreadTS.git "$WF_UPSTREAM"
git -C "$WF_UPSTREAM" checkout --detach ed221d491fe2142a4b2e93462c2c0b7a1c7c31ad
git -C "$WF_UPSTREAM" apply "$WF_REPO/reproductions/wsts_res18_unet_t1/patches/res18_import_scope.patch"
python - "$WF_UPSTREAM" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1]) / 'src/dataloader/FireSpreadDataset.py'
s = p.read_text()
line = 'from torch.utils.data.dataset import T_co\n'
assert s.count(line) == 1
p.write_text(s.replace(line, ''))
PY
# Fetch exactly the encoder initialization used by SMP 0.3.3 (CPU only).
python - <<'PY'
import segmentation_models_pytorch as smp
encoder = smp.encoders.get_encoder('resnet18', in_channels=40, weights='imagenet')
print('ResNet18 ImageNet initialization cached')
PY
python -m pip freeze > "$WF_ROOT/envs/training-freeze.txt"
deactivate
module purge
module load StdEnv/2023 gcc/12.3 python/3.13.2
virtualenv --no-download "$WF_ROOT/envs/audit"
source "$WF_ROOT/envs/audit/bin/activate"
python -m pip install --no-index -e "$WF_REPO[dev]"
python -m pip freeze > "$WF_ROOT/envs/audit-freeze.txt"
touch "$WF_ROOT/envs/READY"
