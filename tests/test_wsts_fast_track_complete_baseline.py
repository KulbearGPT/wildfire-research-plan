import json
from pathlib import Path

import pytest
import torch

from reproductions.wsts_fast_track.complete_baseline import complete, sha256
from reproductions.wsts_fast_track.evaluate_corrected_baseline import validate_evaluation_record


def fixture_run(tmp_path, baseline='B3'):
    root = tmp_path / 'run'
    root.mkdir()
    checkpoint = root / 'best.ckpt'
    torch.save({'global_step': 2400, 'state_dict': {}, 'hyper_parameters': {},
                'callbacks': {'ModelCheckpoint': {'best_model_path': str(checkpoint),
                                                  'best_model_score': 0.5}}}, checkpoint)
    receipt = dict(schema_version=1, status='pass', baseline_id=baseline,
                   experiment='C00' if baseline == 'B3' else 'C02',
                   training_policy='fire-block', seed=0, max_steps=3000,
                   corrected_index=True, initialization='from_scratch',
                   validation_years=[2021], test_enabled=False,
                   fit_global_step=3000, slurm_job_id='test',
                   selection=dict(checkpoint='best.ckpt', monitor='val_avg_precision',
                                  mode='max', score=0.5, sha256=sha256(checkpoint)))
    (root / 'training-completion.json').write_text(json.dumps(receipt))
    return root, receipt


@pytest.mark.parametrize('baseline', ['B3', 'B5'])
def test_best_checkpoint_before_final_step_can_be_sealed(monkeypatch, tmp_path, baseline):
    monkeypatch.setenv('SLURM_JOB_ID', 'test')
    root, _ = fixture_run(tmp_path, baseline)
    record = complete(baseline, root, tmp_path / 'sealed.ckpt', tmp_path / 'completed.json')
    assert record['checkpoint_global_step'] == 2400
    assert record['fit_global_step'] == 3000
    assert validate_evaluation_record(record).baseline_id == baseline
    assert sha256(root / 'best.ckpt') == sha256(tmp_path / 'sealed.ckpt')
    with pytest.raises(FileExistsError):
        complete(baseline, root, tmp_path / 'sealed.ckpt', tmp_path / 'another.json')


@pytest.mark.parametrize('change', ['incomplete', 'config', 'score', 'digest', 'receipt_absent'])
def test_rejects_unsubstantiated_training_success(monkeypatch, tmp_path, change):
    monkeypatch.setenv('SLURM_JOB_ID', 'test')
    root, receipt = fixture_run(tmp_path)
    if change == 'incomplete':
        receipt['fit_global_step'] = 2400
    elif change == 'config':
        receipt['seed'] = 9
    elif change == 'score':
        receipt['selection']['score'] = 0.6
    elif change == 'digest':
        receipt['selection']['sha256'] = '0' * 64
    path = root / 'training-completion.json'
    path.write_text(json.dumps(receipt))
    if change == 'receipt_absent':
        path.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        complete('B3', root, tmp_path / 'sealed.ckpt', tmp_path / 'completed.json')
    assert not (tmp_path / 'sealed.ckpt').exists()


def test_completion_refuses_login_before_checkpoint_read(monkeypatch, tmp_path):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(RuntimeError, match='Slurm'):
        complete('B3', tmp_path / 'absent', tmp_path / 'weight', tmp_path / 'record')
