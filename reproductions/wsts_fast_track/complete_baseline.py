"""Seal a portable baseline from a successful trainer completion receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def complete(baseline_id, run_root, checkpoint_output, record_output):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('baseline completion requires a Slurm allocation')
    import torch
    from .corrected_baselines import corrected_baseline_spec
    from .evaluate_corrected_baseline import validate_evaluation_record

    baseline = corrected_baseline_spec(baseline_id)
    root = Path(run_root).resolve(strict=True)
    receipt = json.loads((root / 'training-completion.json').read_text())
    expected = dict(baseline_id=baseline_id, experiment=baseline.experiment_id,
                    training_policy=baseline.training_policy, seed=baseline.seed,
                    max_steps=baseline.max_steps, corrected_index=True,
                    initialization='from_scratch', validation_years=[2021],
                    test_enabled=False, schema_version=1, status='pass')
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError('completion receipt differs from baseline configuration')
    if receipt.get('fit_global_step') != baseline.max_steps:
        raise ValueError('trainer did not complete the required fit steps')
    selection = receipt['selection']
    if selection.get('monitor') != 'val_avg_precision' or selection.get('mode') != 'max':
        raise ValueError('checkpoint selection must maximize validation AP')
    score = float(selection['score'])
    if not math.isfinite(score):
        raise ValueError('checkpoint selection score must be finite')
    source = (root / selection['checkpoint']).resolve(strict=True)
    if not source.is_relative_to(root):
        raise ValueError('selected checkpoint must belong to the training run')
    digest = sha256(source)
    if digest != selection['sha256']:
        raise ValueError('selected checkpoint SHA256 mismatch')
    payload = torch.load(source, map_location='cpu', weights_only=False)
    step = payload.get('global_step')
    if type(step) is not int or not 0 < step <= baseline.max_steps:
        raise ValueError('selected checkpoint has invalid global_step')
    if not isinstance(payload.get('state_dict'), dict) or not isinstance(payload.get('hyper_parameters'), dict):
        raise ValueError('selected checkpoint lacks model state/hyperparameters')
    callbacks = payload.get('callbacks', {})
    matches = [state for key, state in callbacks.items()
               if 'ModelCheckpoint' in str(key) and isinstance(state, dict)
               and Path(state.get('best_model_path', '')).name == source.name]
    if len(matches) != 1 or float(matches[0]['best_model_score']) != score:
        raise ValueError('checkpoint callback does not corroborate best validation selection')
    destination = Path(checkpoint_output).expanduser().absolute()
    record_path = Path(record_output).expanduser().absolute()
    if destination == record_path:
        raise ValueError('checkpoint and record destinations must differ')
    for path in (destination, record_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    record = {**expected, 'train_years': [2016, 2017, 2018, 2019, 2020],
              'withheld_years': [2022, 2023], 'checkpoint': str(destination),
              'checkpoint_global_step': step, 'fit_global_step': receipt['fit_global_step'],
              'checkpoint_sha256': digest, 'selection': selection,
              'metrics': {'val_avg_precision': score},
              'initialization_detail': 'ImageNet encoder; no wildfire task checkpoint',
              'slurm_job_id': receipt['slurm_job_id']}
    validate_evaluation_record(record)
    created_checkpoint = False
    try:
        with destination.open('xb') as output, source.open('rb') as input_file:
            created_checkpoint = True
            shutil.copyfileobj(input_file, output)
        if sha256(destination) != digest:
            raise ValueError('copied checkpoint SHA256 mismatch')
        with record_path.open('x') as handle:
            json.dump(record, handle, indent=2)
            handle.write('\n')
    except BaseException:
        if created_checkpoint:
            destination.unlink(missing_ok=True)
        raise
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-id', required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--checkpoint-output', type=Path, required=True)
    parser.add_argument('--record-output', type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(complete(args.baseline_id, args.run_root,
                             args.checkpoint_output, args.record_output), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
