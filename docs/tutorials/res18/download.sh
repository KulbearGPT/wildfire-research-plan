#!/usr/bin/env bash
set -euo pipefail
source "${WF_TUTORIAL_ENV:?}"
source "$WF_REPO/docs/tutorials/res18/common.sh"
case "${1:-original}" in original|plus) dataset=${1:-original};; *) exit 2;; esac
if [[ "$dataset" == original ]]; then
  filename=WildfireSpreadTS.zip
  record=8006177
  checksum=dc1a04e63ccc70037b277d585b8fe761
else
  filename=WSTSPlus.zip
  record=17584629
  checksum=42da7598cc33a170064e78d8027148c9
fi
cd "$WF_ROOT/downloads"
if [[ ! -f "$filename" ]]; then
  curl -fL --retry 8 --retry-delay 5 --continue-at - \
    "https://zenodo.org/api/records/$record/files/$filename/content" -o "$filename.partial"
  printf '%s  %s\n' "$checksum" "$filename.partial" | md5sum -c -
  mv "$filename.partial" "$filename"
fi
printf '%s  %s\n' "$checksum" "$filename" | md5sum -c -
stat -c '%n %s bytes' "$filename"
