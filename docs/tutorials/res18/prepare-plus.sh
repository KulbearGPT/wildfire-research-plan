#!/usr/bin/env bash
set -euo pipefail
source "${WF_TUTORIAL_ENV:?}"
source "$WF_REPO/docs/tutorials/res18/common.sh"
test -f "$WF_ROOT/hdf5/original/READY"
train_env
new_run prepare-plus
stage="$SLURM_TMPDIR/plus"
mkdir "$stage"
unzip -q "$WF_ROOT/downloads/WSTSPlus.zip" -d "$stage"
mapfile -t roots < <(find "$stage" -type d -name 2016 -print)
[[ ${#roots[@]} -eq 1 ]]
raw="$(dirname "${roots[0]}")"
python "$WF_SCRIPTS/convert-wstsplus-added.py" \
  --source-root "$raw" --target-root "$WF_ROOT/hdf5/added-source" \
  --summary "$WF_RUN/conversion.json"
deactivate
audit_env
python -m wildfire_phase0.cli repair-active-fire \
  --source-tiff-root "$raw" --hdf5-root "$WF_ROOT/hdf5/added-source" \
  --staging-root "$WF_ROOT/hdf5/added-verified" --years 2016 2017 2022 2023
python -m wildfire_phase0.verify_repair \
  --hdf5-root "$WF_ROOT/hdf5/added-verified" --source-tiff-root "$raw" \
  --expect 2016:92:2102:886:303648 --expect 2017:110:2490:1481:177972 \
  --expect 2022:122:3424:2158:105377 --expect 2023:68:2442:1297:167952 \
  > "$WF_RUN/repair-verification.json"
python - "$WF_ROOT/hdf5/added-verified" <<'PY'
from pathlib import Path
import sys
from wildfire_phase0.repair import verify_repair_evidence
result = verify_repair_evidence(Path(sys.argv[1]))
assert result.status == 'ready', result
expected = (
    '2022/fire_CA4186812327820220730', '2022/fire_ID4570411652620220904',
    '2022/fire_OR4513211711020220825', '2022/fire_WA4687912083320220803',
    '2022/fire_WA4796412068520220909',
)
assert result.excluded_empty_source_directories == expected, result
PY
python "$WF_SCRIPTS/assemble-wstsplus.py" \
  --original-root "$WF_ROOT/hdf5/original" \
  --repaired-root "$WF_ROOT/hdf5/added-verified" \
  --target-root "$WF_ROOT/hdf5/combined" --summary "$WF_RUN/data-summary.json"
python -m wildfire_phase0.cli audit \
  --data-root "$WF_ROOT/hdf5/combined" --output-root "$WF_RUN/audit"
python - "$WF_RUN/audit/contract_decision.json" <<'PY'
import json, sys
print(json.dumps(json.load(open(sys.argv[1])), indent=2))
PY
deactivate
train_env
cd "$WF_REPO"
python -m reproductions.wsts_fast_track.compute_stats \
  --data-root "$WF_ROOT/hdf5/combined" --output "$WF_ROOT/hdf5/train-stats.npz"
python - "$WF_ROOT/hdf5/combined" "$WF_ROOT/hdf5/train-stats.npz" <<'PY'
from pathlib import Path
import sys
from reproductions.wsts_fast_track.contract import validate_inventory
from reproductions.wsts_fast_track.runtime import load_training_stats
print(validate_inventory(Path(sys.argv[1])))
load_training_stats(Path(sys.argv[2]))
PY
touch "$WF_ROOT/hdf5/combined/READY"
