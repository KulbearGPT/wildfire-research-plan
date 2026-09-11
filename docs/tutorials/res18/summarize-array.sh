#!/usr/bin/env bash
set -euo pipefail
source "${WF_TUTORIAL_ENV:?}"
source "$WF_REPO/docs/tutorials/res18/common.sh"
mode=${1:?Use full or weight}
array=${2:?Pass array job ID}
case "$mode" in full|weight) ;; *) exit 2;; esac
[[ "$array" =~ ^[0-9]+$ ]] || exit 2
module purge
module load StdEnv/2023 gcc/12.3 python/3.10.13
python - "$WF_ROOT/runs" "$mode" "$array" "$WF_SCRIPTS" <<'PY'
import json, re, sys
from pathlib import Path
sys.path.insert(0, sys.argv[4])
from results import aggregate
root, mode, array = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
rows = []
for directory in root.glob(f'official-{mode}-fold*'):
    metadata = directory / 'slurm-job.txt'
    if not metadata.is_file():
        continue
    if not re.search(r'\bArrayJobId=' + re.escape(array) + r'\b', metadata.read_text()):
        continue
    rows.append(json.loads((directory / 'result.json').read_text()))
result = aggregate(rows, mode)
path = root / f'summary-{mode}-{array}.json'
with path.open('x') as handle:
    json.dump(result, handle, indent=2)
print(json.dumps(result, indent=2))
PY
