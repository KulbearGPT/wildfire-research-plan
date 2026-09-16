#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?qualification requires Slurm}"
: "${WILDFIRE_UPSTREAM:?}"
export OMP_NUM_THREADS=1
python -m pytest -q \
  tests/test_cra_portability.py tests/test_portable_paths.py tests/test_routed_source_paths.py \
  tests/test_teacher_manifest_builder.py tests/test_cross_history_checkpoint_contract.py \
  tests/test_diagnostic_artifacts.py tests/test_wsts_fast_track_viirs_reliability.py \
  tests/test_wsts_fast_track_viirs_screen.py \
  tests/test_artifact_bundle.py tests/test_research_launchers.py \
  tests/test_research_data_preparation.py tests/test_upstream_import_patch.py \
  tests/test_wsts_fast_track_runtime.py \
  tests/test_wsts_fast_track_complete_baseline.py tests/test_wsts_fast_track_legacy_p00.py \
  tests/test_wsts_fast_track_complete_reliability_prompt_pyramid.py \
  tests/test_wsts_fast_track_counterfactual_impact_consistency.py \
  tests/test_wsts_fast_track_counterfactual_rank_consistency.py \
  tests/test_wsts_fast_track_counterfactual_reliability_adapter.py \
  tests/test_wsts_fast_track_reliability_normalized_conv.py \
  tests/test_wsts_fast_track_reliability_prompt_pyramid.py \
  tests/test_wsts_fast_track_t5_sarp.py \
  tests/test_wsts_fast_track_corrected_baselines.py \
  tests/test_wsts_fast_track_evaluation.py \
  tests/test_wsts_fast_track_corrected_training.py \
  tests/test_cross_history_architectures.py \
  tests/test_three_directions.py tests/test_three_directions_report.py
python - <<'PY'
import importlib
import os
from pathlib import Path
import numpy as np
from reproductions.wsts_fast_track.runtime import _install_runtime_contract

# Resolve the actual patched official package, not mocked module fixtures.
_install_runtime_contract(Path(os.environ['WILDFIRE_UPSTREAM']), 'C02',
                          (np.zeros(23), np.ones(23), np.zeros(23)))
models = importlib.import_module('models')
temporal = importlib.import_module('models.SMPTempModel')
assert models.SMPTempModel is temporal.SMPTempModel
assert isinstance(models.SMPTempModel, type)
for name in (
    'reproductions.cross_history.run',
    'reproductions.cross_history.run_three_directions',
    'reproductions.three_directions.run',
    'reproductions.wsts_fast_track.train_temporal_reliability_prompting',
    'reproductions.wsts_fast_track.train_counterfactual_rank_consistency',
):
    importlib.import_module(name)
print('actual upstream temporal class and retained driver imports: PASS')
PY
