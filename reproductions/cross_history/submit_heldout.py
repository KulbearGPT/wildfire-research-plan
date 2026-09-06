"""Submit fixed-year evaluation only after a comparison passes confirmation."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--run', type=Path, action='append', required=True,
                        help='training result directory containing started.json/checkpoint.pt')
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()

    comparison = json.loads(args.comparison.read_text())
    if not comparison.get('confirmation_pass'):
        raise RuntimeError('heldout submission requires confirmation_pass=true')

    seen = set()
    jobs = []
    for run in args.run:
        metadata = json.loads((run / 'started.json').read_text())
        history = int(metadata['history'])
        method = metadata['method']
        seed = int(metadata['seed'])
        block_fraction = metadata.get('block_fraction')
        key = history, method, seed, block_fraction
        if key in seen:
            raise ValueError(f'duplicate training run: {key}')
        seen.add(key)
        checkpoint = (run / 'checkpoint.pt').resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        for year in (2022, 2023):
            variant = '' if block_fraction is None else f'-b{int(100*block_fraction)}'
            extra = [] if block_fraction is None else [
                '--block-fraction', str(block_fraction),
            ]
            jobs.append([
                'sbatch', '--parsable',
                f'--job-name=H{history}-{method[:8]}{variant}-s{seed}-y{year}',
                '--partition=gpubase_bygpu_b1',
                '--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1',
                '--cpus-per-task=8', '--mem=64G', '--time=01:30:00',
                f'--export=ALL,WILDFIRE_SOURCE_COMMIT={args.source_commit}',
                'reproductions/cross_history/run_slurm.sh', str(history), method,
                '--seed', str(seed), '--batch-size', '16', '--workers', '7',
                '--evaluate-only', str(checkpoint), '--year', str(year),
                *extra,
            ])

    for command in jobs:
        if args.submit:
            job = subprocess.check_output(command, text=True).strip()
            print(job, shlex.join(command[2:]))
        else:
            print(shlex.join(command))


if __name__ == '__main__':
    main()
