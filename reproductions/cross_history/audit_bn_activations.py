# Code lifecycle: archived. Preserved historical exploration; not an active contribution direction.
# Scope and settings: docs/CODE_LIFECYCLE.md; docs/research/method-inventory.json.
"""Archived RF BatchNorm activation diagnostic; execution requires Slurm and CUDA."""
import argparse
import json
import os
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False,
        epilog='Preserves the historical 2021 M06 probe on the first four samples. '
               'Configure dataset/upstream paths through the existing WILDFIRE_* environment.')
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
                             'bn-activation-audit.json).')
    args = parser.parse_args(argv)
    args.checkpoints = [
        (1, args.run_root / args.t1_run / 'result/checkpoint.pt'),
        (5, args.run_root / args.t5_run / 'result/checkpoint.pt'),
    ]
    if args.output is None:
        args.output = args.run_root / 'three-directions-analysis/bn-activation-audit.json'
    return args


def run(args):
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Slurm allocation required for the archived BN activation audit.')

    import torch
    from torch import nn
    from reproductions.cross_history.data import setup, evaluation_dataset
    from reproductions.cross_history.run_three_directions import (
        make_forecaster, BankForecaster, apply_bn_bank,
    )

    setup()
    torch.set_num_threads(2)
    reports = {}
    for h, checkpoint in args.checkpoints:
        s = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model = make_forecaster(h, s['hyper_parameters'], 'control')
        model.load_state_dict(s['state_dict'])
        model.cuda().eval()
        dataset = evaluation_dataset(h, 2021, 'M06')
        packed = torch.stack([dataset[i][0] for i in range(4)]).cuda()
        records = {}
        current = {}
        hooks = []

        def hook(name):
            def capture(module, inputs, output):
                current[name] = dict(
                    input_max=float(inputs[0].abs().max()),
                    output_max=float(output.abs().max()),
                    output_rms=float(output.square().mean().sqrt()))
            return capture

        for name, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                hooks.append(m.register_forward_hook(hook(name)))
        with torch.no_grad():
            for variant, bank in [('original', None), ('common', s['common']),
                                  ('conditional', s['banks'][1])]:
                if bank is not None:
                    apply_bn_bank(model, bank)
                current = {}
                logits = model(packed)
                records[variant] = dict(logit_min=float(logits.min()),
                                        logit_max=float(logits.max()), layers=current)
            direct = logits.clone()
            routed = BankForecaster(model, s['banks']).eval()(packed)
            torch.testing.assert_close(direct, routed)
            records['direct_vs_routed_max_error'] = float((direct-routed).abs().max())
        for hook_handle in hooks:
            hook_handle.remove()
        reports[str(h)] = records
        print(h, {k: {a: v for a, v in r.items() if a != 'layers'}
                  if isinstance(r, dict) else r for k, r in records.items()}, flush=True)
        del model, packed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2) + '\n')


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':
    main()
