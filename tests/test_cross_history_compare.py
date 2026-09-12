import json
import sys

import pytest

from reproductions.cross_history import compare


def _compare(tmp_path, monkeypatch, pairs):
    argv = ['compare']
    for i, (control, candidate) in enumerate(pairs):
        for name, metadata, ap in [('control', control, .2), ('candidate', candidate, .22)]:
            path = tmp_path / f'{name}-{i}.json'
            path.write_text(json.dumps(dict(
                method=name, **metadata,
                results={s: {'avg_precision': ap} for s in compare.SCENARIOS},
            )))
            argv.extend([f'--{name}', str(path)])
    output = tmp_path / 'comparison.json'
    argv.extend(['--output', str(output)])
    monkeypatch.setattr(sys, 'argv', argv)
    compare.main()
    return json.loads(output.read_text())


def _meta(history=1, seed=0, year=2021, architecture=None):
    metadata = dict(history=history, seed=seed, year=year)
    if architecture is not None:
        metadata['architecture'] = architecture
    return metadata


def test_rejects_mismatched_control_and_candidate_architectures(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='architecture'):
        _compare(tmp_path, monkeypatch, [(
            _meta(architecture='res18_unet'), _meta(architecture='swin_unet'),
        )])


@pytest.mark.parametrize('seed,year', [(1, 2021), (0, 2022)])
def test_rejects_mixed_architectures_within_history(tmp_path, monkeypatch, seed, year):
    canonical = _meta(architecture='res18_unet')
    swin = _meta(seed=seed, year=year, architecture='swin_unet')
    with pytest.raises(ValueError, match='architecture'):
        _compare(tmp_path, monkeypatch, [(canonical, canonical), (swin, swin)])


def test_legacy_metadata_matches_explicit_canonical_architectures(tmp_path, monkeypatch):
    pairs = [(_meta(history=h, seed=s), _meta(history=h, seed=s, architecture=a))
             for h, a in [(1, 'res18_unet'), (5, 'res18_utae')] for s in (0, 1, 2)]
    result = _compare(tmp_path, monkeypatch, pairs)
    assert result['confirmation_pass']
    assert {r['architecture'] for r in result['rows']} == {'res18_unet', 'res18_utae'}
    assert [g['architecture'] for g in result['grouped']] == ['res18_unet', 'res18_utae']


def test_matching_noncanonical_architecture_keeps_both_histories(tmp_path, monkeypatch):
    metadata = [_meta(history=h, architecture='swin_unet') for h in (1, 5)]
    result = _compare(tmp_path, monkeypatch, [(m, m) for m in metadata])
    assert result['screen_pass']
    assert {g['architecture'] for g in result['grouped']} == {'swin_unet'}


def test_aggregate_accepts_legacy_route_rows():
    row = dict(history=5, year=2021, seed=0, primary_delta=.02,
               block_delta=.03, delta={'M00': 0.})
    grouped = compare.aggregate([row], 5, 2021)
    assert grouped['seeds'] == [0]
    assert grouped['primary_delta'] == .02
    assert grouped['clean_delta'] == 0.
