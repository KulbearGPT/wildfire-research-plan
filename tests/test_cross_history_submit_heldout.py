import json
import shlex
import sys

import pytest

from reproductions.cross_history import submit_heldout


def _run(tmp_path, name, architecture=None, history=1):
    run = tmp_path / name
    run.mkdir()
    metadata = dict(history=history, method='control', seed=0)
    if architecture is not None:
        metadata['architecture'] = architecture
    (run / 'started.json').write_text(json.dumps(metadata))
    (run / 'checkpoint.pt').touch()
    return run


def _commands(tmp_path, monkeypatch, capsys, runs):
    comparison = tmp_path / 'comparison.json'
    comparison.write_text(json.dumps({'confirmation_pass': True}))
    argv = ['submit_heldout', '--comparison', str(comparison), '--source-commit', 'HEAD']
    for run in runs:
        argv.extend(['--run', str(run)])
    monkeypatch.setattr(sys, 'argv', argv)
    submit_heldout.main()
    return [shlex.split(line) for line in capsys.readouterr().out.splitlines()]


@pytest.mark.parametrize('architecture,history', [('swin_unet', 1), ('segformer_b2', 5), ('convlstm', 5)])
def test_forwards_architecture_to_each_test_year(tmp_path, monkeypatch, capsys, architecture, history):
    run = _run(tmp_path, 'model', architecture, history)
    commands = _commands(tmp_path, monkeypatch, capsys, [run])
    assert len(commands) == 2
    assert {c[c.index('--year') + 1] for c in commands} == {'2022', '2023'}
    for command in commands:
        assert command[command.index('--architecture') + 1] == architecture


def test_same_seed_on_different_backbones_is_not_a_duplicate(tmp_path, monkeypatch, capsys):
    runs = [_run(tmp_path, 'res18', 'res18_unet'), _run(tmp_path, 'swin', 'swin_unet')]
    assert len(_commands(tmp_path, monkeypatch, capsys, runs)) == 4


@pytest.mark.parametrize('history,architecture', [(1, 'res18_unet'), (5, 'res18_utae')])
def test_legacy_metadata_uses_canonical_architecture(tmp_path, monkeypatch, capsys, history, architecture):
    run = _run(tmp_path, 'legacy', history=history)
    commands = _commands(tmp_path, monkeypatch, capsys, [run])
    assert commands[0][commands[0].index('--architecture') + 1] == architecture


def test_duplicate_backbone_seed_is_still_rejected(tmp_path, monkeypatch, capsys):
    run = _run(tmp_path, 'model', 'swin_unet')
    with pytest.raises(ValueError, match='duplicate training run'):
        _commands(tmp_path, monkeypatch, capsys, [run, run])
