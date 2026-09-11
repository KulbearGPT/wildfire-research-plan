#!/usr/bin/env python3
"""Read a completed tutorial run or aggregate exactly twelve official folds."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def parse_metrics(text):
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text).replace('\r', '\n')
    result = {}
    for name in ('AP', 'f1', 'iou', 'precision', 'recall', 'loss'):
        matches = re.findall(r'\btest_' + name + r'\b[ \t│┃|:=]*([^\s│┃|]+)', text)
        if not matches:
            raise ValueError(f'Missing finite test_{name}; inspect the full log')
        value = float(matches[-1])
        if not math.isfinite(value) or value < 0 or (name != 'loss' and value > 1):
            raise ValueError(f'Invalid test_{name}: {value}')
        result[name] = value
    return result


def aggregate(rows, mode):
    if mode not in ('full', 'weight') or len(rows) != 12:
        raise ValueError('Exactly 12 full-training OR 12 released-weight results are required')
    if any(r['mode'] != mode for r in rows) or sorted(r['fold'] for r in rows) != list(range(12)):
        raise ValueError('Missing/duplicate folds or mixed experiment types')
    values = [r['metrics']['AP'] for r in rows]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('AP must be finite and in [0, 1]')
    return {'mode': mode, 'folds': 12, 'mean_AP': statistics.mean(values),
            'population_std_AP': statistics.pstdev(values), 'sample_std_AP': statistics.stdev(values),
            'per_fold': sorted(rows, key=lambda r: r['fold'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    record = sub.add_parser('record')
    record.add_argument('run', type=Path)
    record.add_argument('mode', choices=('full', 'weight'))
    record.add_argument('fold', type=int, choices=range(12))
    summary = sub.add_parser('aggregate')
    summary.add_argument('mode', choices=('full', 'weight'))
    summary.add_argument('records', type=Path, nargs='+')
    args = parser.parse_args()
    if args.action == 'aggregate':
        print(json.dumps(aggregate([json.loads(p.read_text()) for p in args.records], args.mode), indent=2))
        return
    text = (args.run / 'training.log').read_text(errors='replace')
    expected = ('`Trainer.fit` stopped: `max_steps=10000` reached.' if args.mode == 'full'
                else 'WSTS_OFFICIAL_WEIGHT_STRICT_LOAD=1')
    if expected not in text or 'WSTS_OBSERVER_PEAK_ALLOCATED_BYTES=' not in text:
        raise ValueError('Missing completion/strict-load or entrypoint exit evidence')
    payload = {'mode': args.mode, 'fold': args.fold, 'metrics': parse_metrics(text),
               'run': str(args.run.resolve()), 'additional_data': False,
               'log_sha256': hashlib.sha256((args.run / 'training.log').read_bytes()).hexdigest()}
    if args.mode == 'full':
        import torch
        checkpoints = list((args.run / 'work').rglob('*.ckpt'))
        if len(checkpoints) != 1:
            raise ValueError(f'Expected one validation-best checkpoint, found {len(checkpoints)}')
        step = torch.load(checkpoints[0], map_location='cpu', weights_only=False)['global_step']
        if not 0 < step <= 10000:
            raise ValueError(f'Invalid best-checkpoint step: {step}')
        payload.update(checkpoint=str(checkpoints[0]), checkpoint_global_step=step)
    with (args.run / 'result.json').open('x') as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))

if __name__ == '__main__':
    main()
