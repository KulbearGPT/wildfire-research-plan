#!/usr/bin/env bash
# Run through submit.sh; conversion uses train, repair/assembly use audit.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo 'Slurm allocation required before download or data preparation' >&2
  exit 2
fi
usage() { echo 'usage: prepare-data.sh download original|plus | original | added | repair | assemble | stats' >&2; exit 2; }
case "${1:-}" in
  download) (( $# == 2 )) && [[ $2 == original || $2 == plus ]] || usage ;;
  original|added|repair|assemble|stats) (( $# == 1 )) || usage ;;
  *) usage ;;
esac
stage_name=$1
: "${WILDFIRE_ROOT:?}" "${WILDFIRE_REPO:?}" "${WILDFIRE_UPSTREAM:?}" "${WILDFIRE_DATA:?}" "${WILDFIRE_STATS:?}"
case "$stage_name" in repair|assemble) expected_env=audit ;; *) expected_env=train ;; esac
if [[ ${VIRTUAL_ENV:-} != "$WILDFIRE_ROOT/envs/$expected_env" ]]; then
  echo "Stage $stage_name requires the $expected_env environment; use submit.sh cpu $([[ $expected_env == audit ]] && printf 'audit ')bash scripts/research/prepare-data.sh $*" >&2
  exit 2
fi
readonly preparation="$WILDFIRE_ROOT/preparation"
readonly original="$WILDFIRE_ROOT/hdf5/original"
readonly added="$WILDFIRE_ROOT/hdf5/added-source"
readonly repaired="$WILDFIRE_ROOT/hdf5/added-verified"
mkdir -p "$preparation" "$WILDFIRE_ROOT/downloads" "$WILDFIRE_ROOT/hdf5" "$WILDFIRE_ROOT/raw"
readonly evidence=$(mktemp -d "$preparation/${stage_name}-${SLURM_JOB_ID}-XXXXXX")
printf '%q ' "$@" > "$evidence/command.txt"
printf '\n' >> "$evidence/command.txt"
archive_metadata() {
  if [[ $1 == original ]]; then
    filename=WildfireSpreadTS.zip; record=8006177; checksum=dc1a04e63ccc70037b277d585b8fe761
  else
    filename=WSTSPlus.zip; record=17584629; checksum=42da7598cc33a170064e78d8027148c9
  fi
}
verify_archive() {
  archive_metadata "$1"
  (cd "$WILDFIRE_ROOT/downloads"; printf '%s  %s\n' "$checksum" "$filename" | md5sum -c -)
}
extract_archive() {
  local dataset=$1 year=$2 destination="$WILDFIRE_ROOT/raw/$1"
  verify_archive "$dataset"
  if [[ -e $destination ]]; then
    echo "Refusing existing extraction destination: $destination" >&2; exit 2
  fi
  mkdir "$destination"
  unzip -q "$WILDFIRE_ROOT/downloads/$filename" -d "$destination"
  local roots
  mapfile -d '' -t roots < <(find "$destination" -type d -name "$year" -print0)
  if (( ${#roots[@]} != 1 )); then
    echo "Expected one $year directory, found ${#roots[@]}" >&2; exit 2
  fi
  raw=$(dirname "${roots[0]}")
  printf '%s\n' "$raw" > "$preparation/$dataset-raw-root.txt"
}
case "$stage_name" in
  download)
    archive_metadata "$2"
    (
      cd "$WILDFIRE_ROOT/downloads"
      if [[ ! -f $filename ]]; then
        curl -fL --retry 8 --retry-delay 5 --continue-at - \
          "https://zenodo.org/api/records/$record/files/$filename/content" -o "$filename.partial"
        printf '%s  %s\n' "$checksum" "$filename.partial" | md5sum -c -
        mv "$filename.partial" "$filename"
      fi
      printf '%s  %s\n' "$checksum" "$filename" | md5sum -c -
      stat -c '%n %s bytes' "$filename"
    ) | tee "$evidence/archive-verification.txt"
    ;;
  original)
    [[ ! -e $original ]] || { echo "Refusing existing target: $original" >&2; exit 2; }
    extract_archive original 2018
    mkdir "$original"
    python "$WILDFIRE_UPSTREAM/src/preprocess/CreateHDF5Dataset.py" --data_dir "$raw" --target_dir "$original"
    python - "$original" <<'PY' | tee "$evidence/original-counts.json"
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
expected = {2018: 176, 2019: 74, 2020: 201, 2021: 156}
counts = {year: len(list((root / str(year)).glob('*.hdf5'))) for year in expected}
if counts != expected:
    raise ValueError(f'original-year inventory mismatch: {counts}')
(root / 'READY').touch()
print(json.dumps(dict(status='pass', events=sum(counts.values()), years=counts)))
PY
    ;;
  added)
    [[ ! -e $added ]] || { echo "Refusing existing target: $added" >&2; exit 2; }
    extract_archive plus 2016
    python "$WILDFIRE_REPO/docs/tutorials/res18/convert-wstsplus-added.py" \
      --source-root "$raw" --target-root "$added" --summary "$evidence/conversion.json"
    touch "$added/READY"
    ;;
  repair)
    test -f "$added/READY"
    [[ ! -e $repaired ]] || { echo "Refusing existing target: $repaired" >&2; exit 2; }
    raw=$(cat "$preparation/plus-raw-root.txt")
    python -m wildfire_phase0.cli repair-active-fire \
      --source-tiff-root "$raw" --hdf5-root "$added" \
      --staging-root "$repaired" --years 2016 2017 2022 2023
    python -m wildfire_phase0.verify_repair \
      --hdf5-root "$repaired" --source-tiff-root "$raw" \
      --expect 2016:92:2102:886:303648 --expect 2017:110:2490:1481:177972 \
      --expect 2022:122:3424:2158:105377 --expect 2023:68:2442:1297:167952 \
      > "$evidence/repair-verification.json"
    python - "$repaired" <<'PY'
from pathlib import Path
import sys
from wildfire_phase0.repair import verify_repair_evidence
result = verify_repair_evidence(Path(sys.argv[1]))
if result.status != 'ready':
    raise ValueError(result)
expected = (
    '2022/fire_CA4186812327820220730', '2022/fire_ID4570411652620220904',
    '2022/fire_OR4513211711020220825', '2022/fire_WA4687912083320220803',
    '2022/fire_WA4796412068520220909',
)
if result.excluded_empty_source_directories != expected:
    raise ValueError(result)
PY
    touch "$repaired/READY"
    ;;
  assemble)
    test -f "$original/READY"
    test -f "$repaired/READY"
    python "$WILDFIRE_REPO/docs/tutorials/res18/assemble-wstsplus.py" \
      --original-root "$original" --repaired-root "$repaired" \
      --target-root "$WILDFIRE_DATA" --summary "$evidence/data-summary.json"
    python -m wildfire_phase0.cli audit --data-root "$WILDFIRE_DATA" --output-root "$evidence/audit"
    python - "$WILDFIRE_DATA" <<'PY'
from pathlib import Path
import sys
from reproductions.wsts_fast_track.contract import validate_inventory
print(validate_inventory(Path(sys.argv[1])))
PY
    touch "$WILDFIRE_DATA/ASSEMBLED"
    ;;
  stats)
    test -f "$WILDFIRE_DATA/ASSEMBLED"
    python -m reproductions.wsts_fast_track.compute_stats \
      --data-root "$WILDFIRE_DATA" --output "$WILDFIRE_STATS" | tee "$evidence/training-stats.json"
    python - "$WILDFIRE_DATA" "$WILDFIRE_STATS" <<'PY'
from pathlib import Path
import sys
from reproductions.wsts_fast_track.contract import validate_inventory
from reproductions.wsts_fast_track.runtime import load_training_stats
print(validate_inventory(Path(sys.argv[1])))
load_training_stats(Path(sys.argv[2]))
PY
    sha256sum "$WILDFIRE_STATS" > "$evidence/training-stats.sha256"
    touch "$WILDFIRE_DATA/READY"
    ;;
esac
printf 'Stage %s passed; evidence: %s\n' "$stage_name" "$evidence"
touch "$evidence/COMPLETED"
