# Code lifecycle: active_support. Narrow CLI for the recorded X22+X17 protocol.
# Scope and settings: docs/CODE_LIFECYCLE.md; docs/research/method-inventory.json.
"""Prepare the mainline command without importing numerical/model dependencies."""
import argparse
import os
import runpy
import shlex
import sys


def command(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--history', type=int, choices=(1, 5), required=True)
    parser.add_argument('--method', choices=('control', 'cosine_erm', 'block_specialist'),
                        required=True)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--block-fraction', type=float, choices=(0.25, 0.5))
    parser.add_argument('--output', required=True)
    parser.add_argument('--year', type=int, choices=(2021, 2022, 2023), default=2021)
    parser.add_argument('--evaluate-only', metavar='CHECKPOINT')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--print-command', action='store_true',
                        help='Print the planned command without imports, writes or execution.')
    args = parser.parse_args(argv)
    if args.workers < 0:
        parser.error('--workers must be nonnegative')
    if args.block_fraction is not None and args.method != 'block_specialist':
        parser.error('--block-fraction requires --method block_specialist')
    if args.year != 2021 and not args.evaluate_only:
        parser.error('2022/2023 require --evaluate-only; training uses 2021 validation')
    if args.smoke and args.evaluate_only:
        parser.error('smoke and formal checkpoint evaluation are separate operations')
    forwarded = ['--history', str(args.history), '--method', args.method,
                 '--seed', str(args.seed), '--steps', '3000', '--batch-size', '64',
                 '--workers', str(args.workers), '--year', str(args.year),
                 '--output', args.output]
    if args.block_fraction is not None:
        forwarded += ['--block-fraction', str(args.block_fraction)]
    if args.evaluate_only:
        forwarded += ['--evaluate-only', args.evaluate_only]
    if args.smoke:
        forwarded += ['--smoke']
    return args.print_command, forwarded


def main(argv=None):
    print_only, forwarded = command(argv)
    module = 'reproductions.cross_history.run'
    if print_only:
        print(shlex.join([sys.executable, '-m', module, *forwarded]))
        return
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Slurm allocation required; use --print-command for preparation.')
    previous = sys.argv
    try:
        sys.argv = [module, *forwarded]
        runpy.run_module(module, run_name='__main__')
    finally:
        sys.argv = previous


if __name__ == '__main__':
    main()
