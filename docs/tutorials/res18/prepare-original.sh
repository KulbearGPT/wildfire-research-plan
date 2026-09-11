#!/usr/bin/env bash
set -euo pipefail
source "${WF_TUTORIAL_ENV:?}"
source "$WF_REPO/docs/tutorials/res18/common.sh"
train_env
test -f "$WF_ROOT/envs/READY"
test ! -e "$WF_ROOT/hdf5/original"
mkdir "$WF_ROOT/hdf5/original"
stage="$SLURM_TMPDIR/original"
mkdir "$stage"
unzip -q "$WF_ROOT/downloads/WildfireSpreadTS.zip" -d "$stage"
mapfile -t roots < <(find "$stage" -type d -name 2018 -print)
[[ ${#roots[@]} -eq 1 ]]
raw="$(dirname "${roots[0]}")"
python "$WF_UPSTREAM/src/preprocess/CreateHDF5Dataset.py" \
  --data_dir "$raw" --target_dir "$WF_ROOT/hdf5/original"
python - "$WF_ROOT/hdf5/original" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
expected = {2018: 176, 2019: 74, 2020: 201, 2021: 156}
counts = {y: len(list((root / str(y)).glob('*.hdf5'))) for y in expected}
assert counts == expected, counts
print(json.dumps({'status': 'pass', 'events': sum(counts.values()), 'years': counts}))
(root / 'READY').touch()
PY
