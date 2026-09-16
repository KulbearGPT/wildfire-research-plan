import hashlib
import json
from pathlib import Path

import pytest

from reproductions import artifact_bundle as artifacts


def source_manifest(tmp_path, format, summary=True):
    source = tmp_path / 'source'
    source.mkdir()
    weights = source / 'weights.pt'
    weights.write_bytes(b'tiny test checkpoint, not a tensor')
    (source / 'summary.json').write_text('{"year": 2021}')
    (source / 'M00.json').write_text('{"avg_precision": 0.5}')
    record = {
        'checkpoint': 'source/weights.pt',
        'summary': 'source/summary.json' if summary else None,
        'sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
        'metadata': {'initial_checkpoint': '/original/identity.ckpt'},
    }
    if format == 'reliability':
        record['bytes'] = weights.stat().st_size
        payload = {'teacher_route': 'preserve me', 'sources': {'1': {'0': {'clean': record}}}}
    else:
        payload = {'t1-s0': {'erm': record}}
    manifest = tmp_path / 'sources.json'
    manifest.write_text(json.dumps(payload))
    return manifest, record


def test_refuses_login_before_manifest_io(monkeypatch, tmp_path):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(RuntimeError, match='Slurm'):
        artifacts.bundle(tmp_path / 'absent.json', 'routed', tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('format,summary', [('reliability', True), ('routed', True), ('routed', False)])
def test_bundle_is_relocatable_and_preserves_provenance(monkeypatch, tmp_path, format, summary):
    monkeypatch.setenv('SLURM_JOB_ID', 'unit-test')
    manifest, original = source_manifest(tmp_path, format, summary)
    before = manifest.read_bytes()
    out = tmp_path / 'bundle'
    result = artifacts.bundle(manifest, format, out, histories=[1], seeds=[0])
    payload = json.loads(result.read_text())
    record = payload['sources']['1']['0']['clean'] if format == 'reliability' else payload['t1-s0']['erm']
    assert not Path(record['checkpoint']).is_absolute()
    assert (out / record['checkpoint']).read_bytes() == (tmp_path / 'source/weights.pt').read_bytes()
    assert record['metadata'] == original['metadata']
    assert record['summary'] is None if not summary else (out / record['summary']).is_file()
    if not summary:
        assert (out / record['checkpoint']).with_name('M00.json').is_file()
    inventory = json.loads((out / 'checksums.json').read_text())
    for path, metadata in inventory.items():
        assert hashlib.sha256((out / path).read_bytes()).hexdigest() == metadata['sha256']
    assert manifest.read_bytes() == before
    with pytest.raises(FileExistsError):
        artifacts.bundle(manifest, format, out)
    assert result.exists()


@pytest.mark.parametrize('failure', ['digest', 'bytes', 'copy'])
def test_integrity_failure_removes_partial_bundle(monkeypatch, tmp_path, failure):
    monkeypatch.setenv('SLURM_JOB_ID', 'unit-test')
    manifest, _ = source_manifest(tmp_path, 'reliability')
    payload = json.loads(manifest.read_text())
    if failure == 'digest':
        payload['sources']['1']['0']['clean']['sha256'] = '0' * 64
    elif failure == 'bytes':
        payload['sources']['1']['0']['clean']['bytes'] = 1
    else:
        monkeypatch.setattr(artifacts.shutil, 'copyfile', lambda source, destination: destination.write_bytes(b'corrupt'))
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='mismatch'):
        artifacts.bundle(manifest, 'reliability', tmp_path / 'bundle')
    assert not (tmp_path / 'bundle').exists()
    assert (tmp_path / 'source/weights.pt').is_file()


def test_missing_selection_refuses_output(monkeypatch, tmp_path):
    monkeypatch.setenv('SLURM_JOB_ID', 'unit-test')
    manifest, _ = source_manifest(tmp_path, 'routed')
    with pytest.raises(ValueError, match='requested history'):
        artifacts.bundle(manifest, 'routed', tmp_path / 'bundle', histories=[1, 5])
    assert not (tmp_path / 'bundle').exists()
