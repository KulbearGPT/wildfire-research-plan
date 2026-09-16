"""Seal a fully completed legacy P00 replay, without correcting its science."""
import argparse
import json
import math
import os
from pathlib import Path

from .complete_baseline import sha256


def validate_receipt(receipt):
    expected = dict(status='pass', prototype_id='P00-FireDrop-C00', experiment='C00',
                    seed=0, max_steps=10000, fit_global_step=10000,
                    scientific_claim=False, legacy_index_semantics=True,
                    training_policy={'active_fire_dropout_probability': .3, 'training_only': True},
                    validation_years=[2021], test_enabled=False,
                    selection_monitor='val_avg_precision', selection_mode='max')
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError('legacy P00 receipt does not prove full archived training')
    if not math.isfinite(float(receipt['selection_score'])):
        raise ValueError('legacy P00 selection score is nonfinite')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('legacy P00 completion requires Slurm')
    root = args.run_root.resolve(strict=True)
    output = root / 'completed.json'
    if output.exists():
        raise FileExistsError(output)
    receipt = json.loads((root / 'legacy-training-completion.json').read_text())
    validate_receipt(receipt)
    checkpoint = (root / receipt['checkpoint']).resolve(strict=True)
    if not checkpoint.is_relative_to(root):
        raise ValueError('checkpoint must belong to legacy training run')
    if sha256(checkpoint) != receipt['checkpoint_sha256']:
        raise ValueError('legacy P00 checkpoint SHA256 mismatch')
    import torch
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    step = payload.get('global_step')
    if type(step) is not int or not 0 < step <= 10000:
        raise ValueError('legacy P00 best checkpoint has invalid step')
    if not isinstance(payload.get('state_dict'), dict) or not isinstance(payload.get('hyper_parameters'), dict):
        raise ValueError('legacy P00 checkpoint lacks state/hyperparameters')
    matches = [state for key, state in payload.get('callbacks', {}).items()
               if 'ModelCheckpoint' in str(key) and isinstance(state, dict)
               and Path(state.get('best_model_path', '')).name == checkpoint.name]
    if len(matches) != 1 or float(matches[0]['best_model_score']) != receipt['selection_score']:
        raise ValueError('legacy P00 best selection is not corroborated by checkpoint')
    record = {**receipt, 'checkpoint': str(checkpoint), 'checkpoint_global_step': step,
              'purpose': 'historical P00 diagnostic replay; known invalid pooled-index foundation',
              'train_years': [2016, 2017, 2018, 2019, 2020], 'withheld_years': [2022, 2023],
              'metrics': {'val_avg_precision': receipt['selection_score']}}
    with output.open('x') as handle:
        json.dump(record, handle, indent=2)
        handle.write('\n')
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
