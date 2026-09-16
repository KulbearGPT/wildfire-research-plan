"""Build fresh relative teacher manifests from completed matched source runs.

Only main loads torch, after the Slurm guard, and every checkpoint is loaded on
CPU. Metadata validation and hashing remain standard-library functions.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

# directory suffix, method, block fraction, reliability role, routed role
SOURCES = (
    ('control', 'control', None, 'clean', 'erm'),
    ('cosine_erm', 'cosine_erm', None, 'fire', 'x22'),
    ('block_specialist', 'block_specialist', None, 'block_mixed', 'x14'),
    ('block025', 'block_specialist', .25, 'mild', 'mild'),
    ('block050', 'block_specialist', .5, 'severe', 'severe'),
)


def validate_source(payload, summary, *, history, seed, method, block_fraction):
    architecture = {1: 'res18_unet', 5: 'res18_utae'}[history]
    common = dict(history=history, seed=seed, method=method,
                  architecture=architecture, block_fraction=block_fraction)
    for name, values, expected in (
        ('checkpoint', payload, dict(common, steps=3000, physical_batch=64, effective_batch=64)),
        ('summary', summary, dict(common, year=2021)),
    ):
        for key, value in expected.items():
            if key not in values or values[key] != value:
                raise ValueError(f'{name} {key} must equal {value!r}; got {values.get(key)!r}')
    if payload.get('smoke'):
        raise ValueError('smoke checkpoint cannot become a formal teacher')
    if not isinstance(payload.get('initial_checkpoint'), str) or not payload['initial_checkpoint']:
        raise ValueError('checkpoint requires a nonempty initial_checkpoint identity')
    for scenario in ('M00', 'M01', 'M06', 'M07'):
        metrics = summary.get('results', {}).get(scenario, {})
        if metrics.get('sample_count') != 3181:
            raise ValueError(f'summary {scenario} requires the complete 3181-sample 2021 evaluation')


def source_record(payload, summary, checkpoint, output, **expected):
    validate_source(payload, summary, **expected)
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    digest = hashlib.sha256()
    with checkpoint.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    metadata_keys = ('history', 'seed', 'method', 'architecture', 'steps', 'physical_batch',
                     'effective_batch', 'block_fraction', 'initial_checkpoint', 'learning_rate',
                     'learning_rate_schedule', 'fire_dropout_probability',
                     'block_dropout_probability', 'job')
    metadata = {key: payload[key] for key in metadata_keys if key in payload}
    return dict(checkpoint=os.path.relpath(checkpoint, output.parent),
                summary=os.path.relpath(checkpoint.with_name('summary.json'), output.parent),
                history=payload['history'], seed=payload['seed'], method=payload['method'],
                block_fraction=payload['block_fraction'], architecture=payload['architecture'],
                bytes=checkpoint.stat().st_size, sha256=digest.hexdigest(), metadata=metadata)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--format', choices=('reliability', 'routed'), required=True)
    parser.add_argument('--histories', type=int, choices=(1, 5), nargs='+', default=[1, 5])
    parser.add_argument('--seeds', type=int, choices=(0, 1, 2), nargs='+', default=[0])
    args = parser.parse_args(argv)
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm allocation required before loading teacher checkpoints')
    if args.output.exists():
        raise ValueError(f'refusing existing manifest: {args.output}')
    import torch

    sources = {}
    for history in dict.fromkeys(args.histories):
        initial_identity = None
        for seed in dict.fromkeys(args.seeds):
            records = {}
            for suffix, method, fraction, reliability_role, routed_role in SOURCES:
                run = args.runs_root / f't{history}-s{seed}-{suffix}'
                if (run / 'smoke.json').exists():
                    raise ValueError(f'refusing smoke source run: {run}')
                checkpoint = run / 'checkpoint.pt'
                payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
                summary = json.loads((run / 'summary.json').read_text())
                record = source_record(payload, summary, checkpoint, args.output,
                                       history=history, seed=seed, method=method,
                                       block_fraction=fraction)
                identity = payload['initial_checkpoint']
                if initial_identity is not None and identity != initial_identity:
                    raise ValueError(f'T={history} sources have inconsistent initial_checkpoint identities')
                initial_identity = identity
                role = reliability_role if args.format == 'reliability' else routed_role
                records[role] = record
                del payload
            if args.format == 'reliability':
                sources.setdefault(str(history), {})[str(seed)] = records
            else:
                sources[f't{history}-s{seed}'] = records
    result = dict(sources=sources) if args.format == 'reliability' else sources
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(f'Validated fresh teacher manifest: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
