"""Copy verified teacher artifacts into a relocatable manifest inside Slurm."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _source(value: str, manifest: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else manifest.parent / path).resolve(strict=True)


def _copy_verified(source: Path, destination: Path, expected: str | None = None,
                   expected_bytes: int | None = None) -> dict:
    if not source.is_file():
        raise ValueError(f'artifact is not a file: {source}')
    if expected_bytes is not None and source.stat().st_size != expected_bytes:
        raise ValueError(f'byte count mismatch: {source}')
    digest = _digest(source)
    if expected is not None and digest != expected:
        raise ValueError(f'SHA256 mismatch: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if _digest(destination) != digest:
        raise ValueError(f'copied SHA256 mismatch: {destination}')
    return {'sha256': digest, 'bytes': destination.stat().st_size}


def bundle(manifest: Path, format: str, output: Path, *,
           histories: list[int] | None = None, seeds: list[int] | None = None) -> Path:
    # Check before even opening the manifest: this operation can copy many GB.
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('artifact bundling requires a Slurm allocation (SLURM_JOB_ID)')
    if format not in {'reliability', 'routed'}:
        raise ValueError('format must be reliability or routed')
    manifest = Path(manifest).expanduser().resolve(strict=True)
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    original = json.loads(manifest.read_text())
    result = copy.deepcopy(original)
    groups = []
    if format == 'reliability':
        result['sources'] = {}
        for history, by_seed in original['sources'].items():
            for seed, records in by_seed.items():
                groups.append((int(history), int(seed), records))
    else:
        result = {}
        for key, records in original.items():
            match = re.fullmatch(r't(\d+)-s(\d+)', key)
            if not match:
                raise ValueError(f'invalid routed source key: {key}')
            groups.append((int(match[1]), int(match[2]), records))
    groups = [(h, s, records) for h, s, records in groups
              if (histories is None or h in histories) and (seeds is None or s in seeds)]
    if not groups:
        raise ValueError('no sources match selected histories/seeds')
    if histories is not None and set(histories) - {h for h, _, _ in groups}:
        raise ValueError('requested history is absent from selected sources')
    if seeds is not None and set(seeds) - {s for _, s, _ in groups}:
        raise ValueError('requested seed is absent from selected sources')
    output.mkdir(parents=True, exist_ok=False)
    inventory = {}
    try:
        for history, seed, records in groups:
            relocated = copy.deepcopy(records)
            for role, record in relocated.items():
                if not re.fullmatch(r'[A-Za-z0-9_-]+', role):
                    raise ValueError(f'invalid role: {role}')
                directory = Path(f't{history}-s{seed}') / role
                source = _source(record['checkpoint'], manifest)
                target = directory / 'checkpoint.pt'
                expected = record['sha256']
                if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
                    raise ValueError(f'invalid checkpoint SHA256 for {role}')
                inventory[str(target)] = _copy_verified(
                    source, output / target, expected,
                    record['bytes'] if format == 'reliability' else record.get('bytes'))
                record['checkpoint'] = str(target)
                if record.get('summary'):
                    summary = _source(record['summary'], manifest)
                    target = directory / 'summary.json'
                    inventory[str(target)] = _copy_verified(summary, output / target)
                    record['summary'] = str(target)
                elif format == 'routed':
                    # teacher_reference uses the checkpoint's sibling M00.json
                    # for the archived T5 seed-zero ERM with no full summary.
                    target = directory / 'M00.json'
                    inventory[str(target)] = _copy_verified(source.parent / 'M00.json', output / target)
                else:
                    raise ValueError(f'reliability summary is missing for {role}')
                # Keep metadata.initial_checkpoint byte-for-byte: provenance
                # identity, not a runtime file dependency.
            if format == 'reliability':
                result['sources'].setdefault(str(history), {})[str(seed)] = relocated
            else:
                result[f't{history}-s{seed}'] = relocated
        target_manifest = output / 'manifest.json'
        target_manifest.write_text(json.dumps(result, indent=2) + '\n')
        (output / 'checksums.json').write_text(json.dumps(inventory, indent=2) + '\n')
        return target_manifest
    except BaseException:
        shutil.rmtree(output)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--format', choices=('reliability', 'routed'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--histories', type=int, nargs='+')
    parser.add_argument('--seeds', type=int, nargs='+')
    args = parser.parse_args(argv)
    print(bundle(args.manifest, args.format, args.output,
                 histories=args.histories, seeds=args.seeds))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
