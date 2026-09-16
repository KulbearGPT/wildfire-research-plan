"""Explicit byte-preserving relocation for historical diagnostic provenance."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def resolve_artifact(recorded: str, *, artifact_map: Path | None = None,
                     relative_to: Path | None = None) -> Path:
    if not isinstance(recorded, str) or not recorded:
        raise ValueError('artifact provenance path must be a nonempty string')
    if artifact_map is None:
        path = Path(recorded)
        if not path.is_absolute() and relative_to is not None:
            path = relative_to / path
        return path.resolve(strict=True)
    manifest = Path(artifact_map).resolve(strict=True)
    mapping = json.loads(manifest.read_text())
    if mapping.get('schema_version') != 1:
        raise ValueError('diagnostic artifact map requires schema_version 1')
    entry = mapping['artifacts'].get(recorded)
    if not isinstance(entry, dict):
        raise ValueError(f'artifact map lacks recorded provenance: {recorded}')
    path = Path(entry['path'])
    if not path.is_absolute():
        path = manifest.parent / path
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    if path.stat().st_size != entry['bytes'] or digest.hexdigest() != entry['sha256']:
        raise ValueError(f'relocated diagnostic artifact changed: {path}')
    return path


def provenance_matches(recorded: object, actual: Path, *, artifact_map: Path | None = None) -> bool:
    if not isinstance(recorded, str):
        return False
    if artifact_map is None:
        return recorded == str(actual)
    return resolve_artifact(recorded, artifact_map=artifact_map) == actual.resolve(strict=True)


def verified_reliability_path(root: Path, record: dict) -> Path:
    """Verify the manifest digest before an input/target NPZ is decoded."""
    import re
    relative = record.get('output')
    expected = record.get('sha256')
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError('reliability manifest output must be a relative path')
    if not isinstance(expected, str) or re.fullmatch(r'[0-9a-f]{64}', expected) is None:
        raise ValueError(f'reliability manifest lacks a valid SHA256: {relative}')
    root = Path(root).resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'reliability artifact must be a file within its root: {relative}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f'reliability artifact SHA256 mismatch: {path}')
    return path


def main():
    """Create a relocation map from transferred, byte-preserved P00 artifacts."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--p00-record', type=Path, required=True)
    parser.add_argument('--p00-checkpoint', type=Path, required=True)
    parser.add_argument('--recorded-p00-record', required=True,
                        help='original provenance string stored in the attention checkpoint')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    record_path = args.p00_record.resolve(strict=True)
    source = json.loads(record_path.read_text())
    entries = {}
    for original, path in ((args.recorded_p00_record, record_path),
                           (source['checkpoint'], args.p00_checkpoint.resolve(strict=True))):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(chunk)
        entry = dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())
        entries[original] = entry
        entries[str(path)] = dict(entry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(dict(schema_version=1, artifacts=entries), stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
