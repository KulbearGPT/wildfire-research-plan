#!/usr/bin/env bash
# Code lifecycle: reference_only. Official-baseline teaching or legacy cluster tooling; not the contribution route.
# Scope and settings: docs/CODE_LIFECYCLE.md; docs/research/method-inventory.json.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python3 "$repo_root/scripts/cluster/clusterctl.py" preflight "$@"
