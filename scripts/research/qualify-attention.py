#!/usr/bin/env python3
"""One actual attention optimizer update; artifacts are qualification only."""
import argparse
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import re
import sys


@contextmanager
def override(obj, name, value):
    previous = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, previous)


def validation_copy(payload):
    if (payload.get('status') != 'qualification' or
            payload.get('scientific_claim') is not False or
            payload.get('qualification_only') is not True or payload.get('steps') != 1):
        raise ValueError('not a one-step attention qualification')
    # The production validator has an exact-key schema. Strip only our markers;
    # all scientific recipe, provenance and tensor checks remain unchanged.
    return {**{key: value for key, value in payload.items()
               if key not in {'scientific_claim', 'qualification_only'}},
            'status': 'pass', 'steps': 3000}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('qualification requires a Slurm GPU allocation')
    commit = os.environ.get('WILDFIRE_SOURCE_COMMIT', '')
    if re.fullmatch(r'[0-9a-f]{40}', commit) is None:
        raise RuntimeError('qualification requires WILDFIRE_SOURCE_COMMIT from the source archive')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('qualification requires CUDA')
    checkout = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(checkout))
    from reproductions.paths import load_paths
    from reproductions.wsts_fast_track import train_belief_state as trainer
    # Import before the budget override: evaluator keeps its formal 3000 steps.
    from reproductions.wsts_fast_track import evaluate_belief_state as evaluator
    from reproductions.wsts_fast_track.latent_state_common import (
        load_trainable_state_dict, trainable_state_dict)
    paths = load_paths(require_explicit=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'QUALIFICATION_ONLY.json').write_text(json.dumps({
        'scientific_claim': False, 'slurm_job_id': os.environ['SLURM_JOB_ID']}) + '\n')
    record = paths.root / 'diagnostics/p00-completed.json'
    artifact_map = paths.root / 'diagnostics/artifact-map.json'
    checkpoint = output / 'attention-qualification.ckpt'
    original_save = torch.save

    def save_qualification(payload, destination, *save_args, **save_kwargs):
        # The trainer saves through an open file handle.
        if Path(getattr(destination, 'name', destination)) == checkpoint:
            payload = {**payload, 'status': 'qualification',
                       'scientific_claim': False, 'qualification_only': True}
        return original_save(payload, destination, *save_args, **save_kwargs)

    with override(trainer, 'TRAINING_STEPS', 1), override(torch, 'save', save_qualification):
        trainer.main(['--method', 'attention', '--history', '1',
            '--p00-record', str(record), '--artifact-map', str(artifact_map),
            '--upstream-root', str(paths.upstream), '--data-root', str(paths.data),
            '--stats-path', str(paths.stats), '--output-path', str(checkpoint),
            '--git-commit', commit, '--batch-size', '2', '--accumulation-steps', '32',
            '--num-workers', '0', '--device', 'cuda'])
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    p00_checkpoint = trainer._checkpoint_from_record(record,
        prototype_id=trainer.P00_ID, artifact_map=artifact_map,
        training_policy={'active_fire_dropout_probability': trainer.FIRE_DROPOUT_PROBABILITY,
                         'training_only': True})
    validation_args = dict(p00_record=record, p00_checkpoint=p00_checkpoint, artifact_map=artifact_map)
    try:
        evaluator._load_validated_checkpoint(checkpoint, **validation_args)
    except ValueError:
        pass
    else:
        raise ValueError('formal evaluator accepted a qualification checkpoint')
    ephemeral = validation_copy(payload)
    original_load = torch.load

    def load_validation_copy(source, *load_args, **load_kwargs):
        if Path(source).resolve() == checkpoint:
            return ephemeral
        return original_load(source, *load_args, **load_kwargs)

    with override(torch, 'load', load_validation_copy):
        evaluator._load_validated_checkpoint(checkpoint, **validation_args)
    device = torch.device('cuda')
    default = evaluator.load_checkpoint_model(p00_checkpoint, experiment_id='C00',
                                               upstream_root=paths.upstream, device=device)
    model = trainer.build_model('attention', default, 1).to(device)
    load_trainable_state_dict(model, payload['model_state'])
    actual = trainable_state_dict(model)
    if payload['model_state'].keys() != actual.keys() or any(
            not torch.equal(value.cpu(), actual[key].detach().cpu())
            for key, value in payload['model_state'].items()):
        raise ValueError('attention reload tensor state differs')
    if sum(p.numel() for p in model.trainable_parameters()) != payload['trainable_parameters']:
        raise ValueError('attention parameter count differs')
    model.eval()
    p00 = evaluator.LatestDayP00(default).to(device).eval()
    dataset_class = importlib.import_module('dataloader.FireSpreadDataset').FireSpreadDataset
    metrics = {}
    for scenario in ('M00', 'M06'):
        dataset = evaluator._build_evaluation_dataset(dataset_class, data_root=paths.data,
                                                       history=1, scenario_id=scenario)
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, [0, 1]),
                                             batch_size=2, num_workers=0)
        metrics[scenario] = {
            'model': evaluator.evaluate_batches(model, loader, device=device),
            'p00': evaluator.evaluate_batches(p00, loader, device=device)}
        if scenario == 'M00':
            evaluator._require_exact_m00(model, p00, loader, device=device)
    report = {'status': 'qualification-pass', 'scientific_claim': False,
        'method': 'attention', 'history': 1, 'actual_optimizer_steps': payload['steps'],
        'physical_batch_size': 2, 'accumulation_steps': 32, 'effective_batch_size': 64,
        'final_training_loss': payload['final_loss'],
        'final_loss_components': payload['final_loss_components'],
        'formal_validator_rejected_smoke': True,
        'qualification_validator_substitutions': {'status': 'pass', 'steps': 3000},
        'qualification_validator_removed_markers': ['scientific_claim', 'qualification_only'],
        'strict_reload_tensor_equality': True, 'm00_exact': True,
        'evaluation_year': 2021, 'evaluation_samples_per_scenario': 2,
        'natural_viirs_evaluation': False, 'metrics': metrics}
    (output / 'qualification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
