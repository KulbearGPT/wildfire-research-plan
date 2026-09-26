# Code lifecycle: archived. Preserved historical exploration; not an active contribution direction.
# Scope and settings: docs/CODE_LIFECYCLE.md; docs/research/method-inventory.json.
"""Archived RF BatchNorm statistics diagnostic; execution requires Slurm."""
import argparse
import json
import os
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False,
        epilog='Preserves the historical pooled-bank statistics calculation; '
               'missing checkpoints are skipped as in the original diagnostic.')
    parser.add_argument('--run-root', type=Path, required=True,
                        help='Directory containing the intended archived RF run directories.')
    parser.add_argument('--t1-run', type=Path, default=Path('three-directions-1-bn-21787290'),
                        help='T1 run directory, absolute or relative to --run-root '
                             '(historical default: %(default)s).')
    parser.add_argument('--t5-run', type=Path, default=Path('three-directions-5-bn-21787291'),
                        help='T5 run directory, absolute or relative to --run-root '
                             '(historical default: %(default)s).')
    parser.add_argument('--output', type=Path,
                        help='Output JSON path (default: RUN_ROOT/three-directions-analysis/'
                             'bn-statistics-audit.json).')
    args = parser.parse_args(argv)
    args.checkpoints = [
        (1, args.run_root / args.t1_run / 'result/checkpoint.pt'),
        (5, args.run_root / args.t5_run / 'result/checkpoint.pt'),
    ]
    if args.output is None:
        args.output = args.run_root / 'three-directions-analysis/bn-statistics-audit.json'
    return args


def run(args):
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Slurm allocation required for the archived BN statistics audit.')

    import torch

    reports = {}
    for h, checkpoint in args.checkpoints:
        if not checkpoint.exists():
            continue
        s = torch.load(checkpoint, map_location='cpu', weights_only=False)
        rows = []
        for name, c in s['common'].items():
            a, b = s['banks'][0][name], s['banks'][1][name]
            n = a['count'] + b['count']
            assert n == c['count']
            m = (a['mean'].double()*a['count'] + b['mean'].double()*b['count'])/n
            v = ((a['count']-1)*a['var'].double() + (b['count']-1)*b['var'].double()
                 + a['count']*(a['mean'].double()-m)**2
                 + b['count']*(b['mean'].double()-m)**2)/(n-1)
            orig = s['state_dict'][name+'.running_var']
            row = dict(layer=name, common_mean_merge_max=float((m-c['mean']).abs().max()),
                       common_var_merge_max=float((v-c['var']).abs().max()),
                       common_var_max=float(c['var'].max()), original_min_var=float(orig.min()))
            for tag, bank in [('absent', a), ('present', b)]:
                row[tag] = dict(
                    min_var=float(bank['var'].min()),
                    zero_variance=int((bank['var'] == 0).sum()), channels=bank['var'].numel(),
                    max_gain_ratio=float(((orig+1e-5)/(bank['var']+1e-5)).sqrt().max()))
            rows.append(row)
        reports[str(h)] = dict(samples=s['bank_samples'], layers=rows)
        print(h, 'layers', len(rows),
              'maxmeanmergeerror', max(r['common_mean_merge_max'] for r in rows),
              'maxvarmergeerror', max(r['common_var_merge_max'] for r in rows))
        print(json.dumps(sorted(rows, key=lambda r: r['present']['max_gain_ratio'],
                                reverse=True)[:5], indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2) + '\n')


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':
    main()
