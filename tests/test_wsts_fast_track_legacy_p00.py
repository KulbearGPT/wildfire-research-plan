import ast
from pathlib import Path

import pytest

from reproductions.wsts_fast_track.legacy_p00_completion import validate_receipt


def receipt():
    return dict(status='pass', prototype_id='P00-FireDrop-C00', experiment='C00',
                seed=0, max_steps=10000, fit_global_step=10000,
                scientific_claim=False, legacy_index_semantics=True,
                training_policy={'active_fire_dropout_probability': .3, 'training_only': True},
                validation_years=[2021], test_enabled=False,
                selection_monitor='val_avg_precision', selection_mode='max', selection_score=.4)


def test_archived_full_fit_contract_is_required():
    validate_receipt(receipt())
    for changes in ({'fit_global_step': 9999}, {'max_steps': 2, 'fit_global_step': 2},
                    {'status': 'qualification'}, {'legacy_index_semantics': False},
                    {'training_policy': {'active_fire_dropout_probability': .3, 'training_only': False}},
                    {'selection_score': float('nan')}):
        with pytest.raises(ValueError):
            validate_receipt({**receipt(), **changes})


def test_legacy_entrypoint_never_installs_corrected_index():
    path = Path(__file__).resolve().parents[1] / 'reproductions/wsts_fast_track/legacy_p00.py'
    tree = ast.parse(path.read_text())
    calls = [node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert 'install_training_fire_dropout' in calls
    assert 'install_corrected_baseline' not in calls
    assert 'resolve_dataset_index' not in path.read_text()
