#!/usr/bin/env python3
"""Apply only the two approved compatibility changes to the pinned checkout."""
import argparse
from pathlib import Path
import subprocess

PINNED_COMMIT = 'ed221d491fe2142a4b2e93462c2c0b7a1c7c31ad'
MODEL_PATH = 'src/models/__init__.py'
DATASET_PATH = 'src/dataloader/FireSpreadDataset.py'
T_CO = b'from torch.utils.data.dataset import T_co\n'


def expected_changes(original_models, original_dataset):
    if original_dataset.count(T_CO) != 1:
        raise ValueError('pinned dataset must contain exactly one obsolete T_co import')
    lines = original_models.splitlines(keepends=True)
    if len(lines) != 11 or not lines[3].startswith(b'from .SMPModel import '):
        raise ValueError('unexpected pinned model import layout')
    return b''.join(lines[:4]), original_dataset.replace(T_CO, b'')


def apply_known_changes(root, patch, original_models, original_dataset):
    root = Path(root)
    models, dataset = root / MODEL_PATH, root / DATASET_PATH
    wanted_models, wanted_dataset = expected_changes(original_models, original_dataset)
    # Validate both complete files before mutating either; do not tolerate drift.
    if models.read_bytes() not in (original_models, wanted_models):
        raise ValueError('unrecognized model import changes; refusing repair')
    if dataset.read_bytes() not in (original_dataset, wanted_dataset):
        raise ValueError('unrecognized dataset changes; refusing repair')
    reverse = models.read_bytes() == wanted_models
    command = ['git', '-C', str(root), 'apply', '--check']
    if reverse:
        command.append('--reverse')
    subprocess.run([*command, str(patch)], check=True)
    if not reverse:
        subprocess.run(['git', '-C', str(root), 'apply', str(patch)], check=True)
    if models.read_bytes() != wanted_models:
        raise ValueError('tracked patch produced unexpected model import bytes')
    if dataset.read_bytes() != wanted_dataset:
        dataset.write_bytes(wanted_dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--patch', type=Path, required=True)
    args = parser.parse_args()
    def git(*arguments):
        return subprocess.check_output(['git', '-C', str(args.upstream), *arguments])
    if git('rev-parse', 'HEAD').decode().strip() != PINNED_COMMIT:
        raise ValueError('upstream HEAD differs from the pinned commit')
    changed = set(git('diff', '--name-only', 'HEAD').decode().splitlines())
    untracked = git('ls-files', '--others', '--exclude-standard').decode().splitlines()
    if changed - {MODEL_PATH, DATASET_PATH} or untracked:
        raise ValueError('upstream contains unrelated changes or untracked files')
    apply_known_changes(args.upstream, args.patch.resolve(),
                        git('show', f'HEAD:{MODEL_PATH}'), git('show', f'HEAD:{DATASET_PATH}'))


if __name__ == '__main__':
    main()
