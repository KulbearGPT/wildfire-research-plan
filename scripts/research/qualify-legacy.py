#!/usr/bin/env python3
"""One real-data GPU qualification case; never a scientific experiment."""
import argparse
from contextlib import contextmanager
from dataclasses import replace
import importlib
import json
import math
import os
from pathlib import Path
import sys

CASES = {
    'rnc': ('reliability_normalized', ['--variant', 'rnc']),
    'token': ('reliability_normalized', ['--variant', 'token']),
    'ciwc': ('counterfactual_impact_consistency', []),
    'rank': ('counterfactual_rank_consistency', []),
    'cra': ('counterfactual_reliability_adapter', ['--adapter-scope', 'all']),
    'ffca': ('counterfactual_reliability_adapter', ['--adapter-scope', 'block']),
    'pyramid': ('reliability_prompt_pyramid', ['--variant', 'prompt-pyramid']),
    'complete-pyramid': ('reliability_prompt_pyramid', ['--variant', 'complete-prompt-pyramid']),
    'sarp-t1': ('reliability_prompt_pyramid', ['--variant', 'severity-adaptive-prompts']),
    'standard-t5': ('temporal_reliability_prompting', ['--variant', 'standard']),
    'sarp-t5': ('temporal_reliability_prompting', ['--variant', 'sarp']),
}
PREFIX = 'reproductions.wsts_fast_track.'


@contextmanager
def override(obj, name, value):
    previous = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, previous)


def qualification_validator(original):
    def validate(payload):
        if (payload.get('status') != 'qualification' or
                payload.get('scientific_claim') is not False or
                payload.get('steps') != 1 or payload.get('global_step') != 1):
            raise ValueError('not a one-step qualification checkpoint')
        # Only budget/status fields are substituted in an ephemeral copy.
        # All architecture, variant and scientific recipe metadata checks run
        # unchanged; weights and the on-disk payload are never substituted.
        return original({**payload, 'status': 'pass', 'steps': 3000})
    return validate


def verify_state(torch, expected, actual):
    if expected.keys() != actual.keys():
        raise ValueError('reload state keys differ')
    for key in expected:
        if not torch.equal(expected[key].cpu(), actual[key].detach().cpu()):
            raise ValueError(f'reload tensor differs: {key}')


def baseline_case(case, paths, output, torch):
    module = importlib.import_module(PREFIX + 'train_corrected_baseline')
    original_spec = module.corrected_baseline_spec
    original_arguments = module.upstream_arguments

    def smoke_arguments(*args, **kwargs):
        arguments = original_arguments(*args, **kwargs)
        prefixes = ('--data.batch_size=', '--data.num_workers=')
        return [arg for arg in arguments if not arg.startswith(prefixes)] + [
            '--data.batch_size=2', '--data.num_workers=0',
            '--trainer.val_check_interval=1', '--trainer.limit_val_batches=1',
            '--trainer.enable_progress_bar=false',
        ]

    with override(module, 'corrected_baseline_spec',
                  lambda name: replace(original_spec(name), max_steps=2)), \
            override(module, 'upstream_arguments', smoke_arguments):
        module.main(['--baseline-id', case, '--upstream-root', str(paths.upstream),
                     '--data-root', str(paths.data), '--stats-path', str(paths.stats),
                     '--run-root', str(output / 'run')])
    receipt_path = output / 'run/training-completion.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt['fit_global_step'] != 2 or receipt['max_steps'] != 2:
        raise ValueError('bootstrap qualification budget differs')
    receipt.update(status='qualification', scientific_claim=False,
                   qualification_only=True)
    (output / 'bootstrap-qualification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    receipt_path.unlink()
    checkpoint = output / 'run' / receipt['selection']['checkpoint']
    from reproductions.wsts_fast_track.evaluate_missingness import load_checkpoint_model
    model = load_checkpoint_model(checkpoint, experiment_id='C02',
                                  upstream_root=paths.upstream, device=torch.device('cuda'))
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    verify_state(torch, payload['state_dict'], model.state_dict())
    return model, False, False, {'fit_global_step': 2, 'selected_global_step': payload['global_step'],
                                 'selection_score': receipt['selection']['score']}


def legacy_p00_case(paths, output, torch):
    module = importlib.import_module(PREFIX + 'legacy_p00')
    original_arguments = module.upstream_arguments

    def smoke_arguments(*args, **kwargs):
        arguments = original_arguments(*args, **kwargs)
        prefixes = ('--data.batch_size=', '--data.num_workers=')
        return [arg for arg in arguments if not arg.startswith(prefixes)] + [
            '--data.batch_size=2', '--data.num_workers=0',
            '--trainer.val_check_interval=1', '--trainer.limit_val_batches=1',
            '--trainer.enable_progress_bar=false',
        ]

    with override(module, 'MAX_STEPS', 2), override(module, 'upstream_arguments', smoke_arguments):
        module.main(['--upstream-root', str(paths.upstream), '--data-root', str(paths.data),
                     '--stats-path', str(paths.stats), '--run-root', str(output / 'run')])
    receipt_path = output / 'run/legacy-training-completion.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt['status'] != 'qualification' or receipt['fit_global_step'] != 2:
        raise ValueError('legacy bootstrap qualification receipt differs')
    from reproductions.wsts_fast_track.legacy_p00_completion import validate_receipt
    try:
        validate_receipt(receipt)
    except ValueError:
        pass
    else:
        raise ValueError('legacy formal completion accepted a two-step qualification')
    (output / 'legacy-p00-qualification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    receipt_path.unlink()
    checkpoint = output / 'run' / receipt['checkpoint']
    from reproductions.wsts_fast_track.evaluate_missingness import load_checkpoint_model
    model = load_checkpoint_model(checkpoint, experiment_id='C00',
                                  upstream_root=paths.upstream, device=torch.device('cuda'))
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    verify_state(torch, payload['state_dict'], model.state_dict())
    return model, False, False, {'fit_global_step': 2, 'selected_global_step': payload['global_step'],
                                 'legacy_index_semantics': True,
                                 'formal_completion_rejected_smoke': True,
                                 'selection_score': receipt['selection_score']}


def continuation_case(case, paths, output, torch):
    family, extra = CASES[case]
    module = importlib.import_module(PREFIX + 'train_' + family)
    checkpoint = output / 'qualification.ckpt'
    temporal = case.endswith('-t5')
    baseline = 'B5' if temporal else 'B3'
    record = paths.root / 'checkpoints' / (baseline + '-completed.json')
    original_save = torch.save

    def save_qualification(payload, path, *args, **kwargs):
        if Path(path) == checkpoint:
            payload = {**payload, 'status': 'qualification', 'scientific_claim': False,
                       'qualification_only': True}
        return original_save(payload, path, *args, **kwargs)

    with override(module, 'TRAINING_STEPS', 1), override(torch, 'save', save_qualification):
        module.main(['--' + baseline.lower() + '-record', str(record),
                     '--upstream-root', str(paths.upstream), '--data-root', str(paths.data),
                     '--stats-path', str(paths.stats), '--output-path', str(checkpoint),
                     '--batch-size', '2', '--num-workers', '0', '--device', 'cuda', *extra])
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload['steps'] != 1 or not math.isfinite(payload['final_loss']):
        raise ValueError('qualification did not produce a finite one-step loss')
    evaluator = importlib.import_module(PREFIX + 'evaluate_' + family)
    device = torch.device('cuda')
    adapter = case in {'cra', 'ffca'}
    if family == 'reliability_normalized':
        validator_name, loader_name = 'validate_d2_checkpoint', 'load_d2_model'
    elif family == 'temporal_reliability_prompting':
        validator_name, loader_name = 'validate_t5_reliability_checkpoint', 'load_t5_reliability_model'
    elif family == 'reliability_prompt_pyramid':
        validator_name, loader_name = 'validate_prompt_checkpoint', 'load_rpp_model'
    elif adapter:
        validator_name, loader_name = 'validate_adapter_checkpoint', 'load_cra_model'
    else:
        validator_name = 'validate_ciwc_checkpoint' if case == 'ciwc' else 'validate_circ_checkpoint'
        loader_name = None
    original_validator = getattr(evaluator, validator_name)
    try:
        original_validator(payload)
    except ValueError:
        pass
    else:
        raise ValueError('formal validator accepted a qualification checkpoint')
    # Explicit call covers loaders that leave metadata validation to their CLI.
    qualification_validator(original_validator)(payload)
    if loader_name:
        with override(evaluator, validator_name, qualification_validator(original_validator)):
            model = getattr(evaluator, loader_name)(payload, upstream_root=paths.upstream, device=device)
    else:
        model = evaluator.load_checkpoint_model(checkpoint, experiment_id='C00',
                                               upstream_root=paths.upstream, device=device)
    if adapter:
        verify_state(torch, payload['base_state_dict'], model.base_model.state_dict())
        verify_state(torch, payload['adapter_state_dict'], model.adapter.state_dict())
    else:
        core = model.model if family in {'reliability_normalized', 'temporal_reliability_prompting'} else model
        verify_state(torch, payload['state_dict'], core.state_dict())
    return model, family not in {'counterfactual_impact_consistency', 'counterfactual_rank_consistency'}, adapter, {
        'actual_training_steps': 1, 'final_training_loss': payload['final_loss'],
        'formal_validator_rejected_smoke': True,
        'qualification_validator_substitutions': {'status': 'pass', 'steps': 3000},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=(*CASES, 'B1', 'B5', 'legacy-P00'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('qualification requires a Slurm GPU allocation')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('qualification requires CUDA')
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from reproductions.paths import load_paths
    paths = load_paths(require_explicit=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'QUALIFICATION_ONLY.json').write_text(json.dumps({
        'scientific_claim': False, 'case': args.case, 'slurm_job_id': os.environ['SLURM_JOB_ID']}) + '\n')
    if args.case == 'legacy-P00':
        model, routing, adapter, details = legacy_p00_case(paths, output, torch)
        history = 1
    elif args.case in {'B1', 'B5'}:
        model, routing, adapter, details = baseline_case(args.case, paths, output, torch)
        history = 5
    else:
        model, routing, adapter, details = continuation_case(args.case, paths, output, torch)
        history = 5 if args.case.endswith('-t5') else 1
    from reproductions.wsts_fast_track.evaluation import build_controlled_dataset
    from reproductions.wsts_fast_track.evaluate_missingness import evaluate_batches
    from reproductions.wsts_fast_track.counterfactual_reliability_adapter import TwoRegimeReliabilityDataset
    model.eval()
    metrics = {}
    for scenario in ('M00', 'M06'):
        dataset = build_controlled_dataset(upstream_root=paths.upstream, data_root=paths.data,
            stats_path=paths.stats, experiment_id='C02' if history == 5 else 'C00',
            scenario_id=scenario, evaluation_year=2021, heldout_authorized=False,
            routing_mask_channel=routing)
        if adapter:
            dataset = TwoRegimeReliabilityDataset(dataset, scenario)
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, [0, 1]),
                                             batch_size=2, num_workers=0)
        metrics[scenario] = evaluate_batches(model, loader, device=torch.device('cuda'))
    report = {'status': 'qualification-pass', 'scientific_claim': False, 'case': args.case,
              'history': history, 'strict_reload_tensor_equality': True,
              'evaluation_samples_per_scenario': 2, 'evaluation_year': 2021,
              'metrics': metrics, **details}
    (output / 'qualification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
