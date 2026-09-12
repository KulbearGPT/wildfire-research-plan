"""Bounded, matched probes of normalization, input fusion and distillation."""
import argparse
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import setup, base_dataset, PairedDataset, evaluation_dataset, MULTI_FEATURES
from .models import Forecaster, initial_payload, make_base
from .architectures import checkpoint_architecture, canonical_architecture
from .three_directions import (
    MomentCollector, apply_bn_bank, ShallowInputStem,
    observable_routes, bernoulli_teacher_kl,
)
from reproductions.wsts_fast_track.processed_reliability import PROCESSED_DYNAMIC_NON_FIRE
from reproductions.wsts_fast_track.evaluate_missingness import evaluate_batches

SCENARIOS = ('M00', 'M01', 'M06', 'M07')
DEFAULT_MANIFEST = Path('docs/experiments/three_directions_manifest.json')


def validate_source(payload, record):
    for key in ('history', 'seed', 'method', 'block_fraction'):
        if payload.get(key) != record.get(key):
            raise ValueError(f'teacher {key} mismatch')
    checkpoint_architecture(payload, expected=record['architecture'])


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class StemForecaster(nn.Module):
    def __init__(self, model, stem):
        super().__init__()
        self.model, self.stem = model, stem

    def forward(self, packed):
        channels = self.model.channels
        x = packed[:, :, :channels]
        batch, history, _, height, width = x.shape
        encoded = self.stem(x.reshape(batch * history, channels, height, width))
        encoded = encoded.reshape(batch, history, channels, height, width)
        return self.model(torch.cat((encoded, packed[:, :, channels:]), dim=2))

    def compute_loss(self, logits, target):
        return self.model.compute_loss(logits, target)


class RoutedTeacher(nn.Module):
    def __init__(self, models):
        super().__init__()
        if len(models) != 4:
            raise ValueError('clean, fire, mild, severe teachers required')
        self.models = nn.ModuleList(models)
        self.eval().requires_grad_(False)

    def forward(self, packed):
        routes = observable_routes(packed)
        output = None
        for index, model in enumerate(self.models):
            selected = routes == index
            if selected.any():
                logits = model(packed[selected])
                if output is None:
                    output = logits.new_empty((len(packed), *logits.shape[1:]))
                output[selected] = logits
        return output

    def compute_loss(self, logits, target):
        return self.models[0].compute_loss(logits, target)


class BankForecaster(nn.Module):
    def __init__(self, model, banks):
        super().__init__()
        self.model, self.banks = model, banks

    def forward(self, packed):
        block = packed[:, -1, -2].flatten(1).any(1)
        output = None
        for index in (0, 1):
            selected = block == bool(index)
            if selected.any():
                apply_bn_bank(self.model, self.banks[index])
                logits = self.model(packed[selected])
                if output is None:
                    output = logits.new_empty((len(packed), *logits.shape[1:]))
                output[selected] = logits
        return output

    def compute_loss(self, logits, target):
        return self.model.compute_loss(logits, target)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_forecaster(history, hparams, arm):
    model = Forecaster(make_base(history, hparams), history, 'control')
    if arm in ('mixed', 'typed'):
        columns = tuple(range(40)) if history == 1 else MULTI_FEATURES
        dynamic = set(PROCESSED_DYNAMIC_NON_FIRE) | {38, 39}
        static = tuple(i for i, column in enumerate(columns) if column not in dynamic)
        model = StemForecaster(model, ShallowInputStem(len(columns), static, arm))
    return model


def load_source(record):
    path = Path(record['checkpoint'])
    if path.stat().st_size != record['bytes'] or file_hash(path) != record['sha256']:
        raise ValueError(f'teacher checkpoint changed: {path}')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    validate_source(payload, record)
    model = make_forecaster(payload['history'], payload['hyper_parameters'], 'control')
    model.load_state_dict(payload['state_dict'], strict=True)
    return model, payload['hyper_parameters']


def load_teachers(records):
    return RoutedTeacher([load_source(records[role])[0]
                          for role in ('clean', 'fire', 'mild', 'severe')])


def evaluate(model, args, output, name):
    model.eval()
    results = {}
    for scenario in SCENARIOS:
        dataset = evaluation_dataset(args.history, args.year, scenario)
        loader = torch.utils.data.DataLoader(dataset, batch_size=min(args.batch_size, 16),
            num_workers=args.workers, pin_memory=True)
        result = evaluate_batches(model, loader, device=torch.device('cuda'))
        expected = {2021: 3181, 2022: 2856, 2023: 2102}[args.year]
        if result['sample_count'] != expected:
            raise ValueError(f'{scenario}: unexpected evaluation population')
        results[scenario] = result
        (output / f'{name}-{scenario}.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(dict(arm=name, scenario=scenario, metrics=result)), flush=True)
        del loader, dataset
    summary = dict(history=args.history, architecture=canonical_architecture(args.history),
                   seed=args.seed, year=args.year, method=f'three_{name}', results=results)
    if name.startswith('bn_'):
        summary['bn_forward_mode'] = args.bn_forward_mode
    (output / f'{name}-summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def train_loader(args, *, calibration=False):
    dataset = PairedDataset(base_dataset(args.history), args.history)
    if calibration:
        generator = torch.Generator().manual_seed(args.seed)
        count = min(args.calibration_samples, len(dataset))
        if count != args.calibration_samples:
            raise ValueError('calibration budget exceeds available training examples')
        indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
        shuffle=not calibration, generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
        drop_last=not calibration)


def bn_diagnostic(args, records, metadata):
    model, hparams = load_source(records['fire'])
    model.cuda().eval().requires_grad_(False)
    original = copy.deepcopy(model.state_dict())
    seed_everything(args.seed)
    start = time.monotonic()
    loader = train_loader(args, calibration=True)
    collector = MomentCollector(model, batch_statistics=args.bn_forward_mode == 'batch_stats')
    samples = [0, 0]
    with torch.no_grad():
        for number, (packed, _, _) in enumerate(loader):
            packed = packed.cuda()
            block = packed[:, -1, -2].flatten(1).any(1)
            for bank in (0, 1):
                selected = block == bool(bank)
                if selected.any():
                    collector.bank = bank
                    model(packed[selected])
                    samples[bank] += int(selected.sum())
            if args.smoke:
                break
    common, banks = collector.finalize()
    calibration_seconds = time.monotonic() - start
    # Only the normalization used during collection changes; no weights,
    # running buffers or counters may change in either calibration mode.
    for key, tensor in model.state_dict().items():
        if not torch.equal(tensor, original[key]):
            raise RuntimeError(f'calibration collection mutated {key}')
    del loader
    saved = dict(metadata, hyper_parameters=hparams,
                 state_dict={k: v.cpu() for k, v in original.items()},
                 common=common, banks=banks, bank_samples=samples,
                 calibration_seconds=calibration_seconds)
    torch.save(saved, args.output / 'checkpoint.pt')
    if args.smoke:
        reloaded = torch.load(args.output / 'checkpoint.pt', map_location='cpu', weights_only=False)
        for scenario in SCENARIOS:
            packed, target = evaluation_dataset(args.history, 2021, scenario)[0]
            packed = packed[None].cuda()
            for variants, restored in (
                ({0: common, 1: common}, {0: reloaded['common'], 1: reloaded['common']}),
                (banks, reloaded['banks']),
            ):
                with torch.no_grad():
                    result = BankForecaster(model, variants).eval()(packed)
                    again = BankForecaster(model, restored).eval()(packed)
                assert result.shape[-2:] == target.shape[-2:] and torch.isfinite(result).all()
                torch.testing.assert_close(result, again, rtol=0, atol=0)
        return dict(bank_samples=samples, bn_layers=len(common), reload_max_difference=0.,
                    scenarios=list(SCENARIOS), variants=['common', 'conditional'],
                    bn_forward_mode=args.bn_forward_mode,
                    calibration_seconds=calibration_seconds)
    evaluate(model, args, args.output, 'bn_original')
    apply_bn_bank(model, common)
    evaluate(model, args, args.output, 'bn_common')
    evaluate(BankForecaster(model, banks), args, args.output, 'bn_conditional')
    return dict(bank_samples=samples, bn_layers=len(common), calibration_seconds=calibration_seconds)


def learned_probe(args, records, metadata):
    payload, initial = initial_payload(args.history)
    hparams = payload['hyper_parameters']
    model = make_forecaster(args.history, hparams, args.arm)
    core = model.model if isinstance(model, StemForecaster) else model
    core.base.load_state_dict(payload['state_dict'], strict=True)
    del payload
    teacher = load_teachers(records).cuda() if args.arm == 'distill' else None
    model.cuda()
    metadata.update(initial_checkpoint=initial, initial_sha256=file_hash(initial),
        parameters=sum(p.numel() for p in model.parameters()),
        teacher_parameters=(sum(p.numel() for p in teacher.parameters()) if teacher else 0))
    (args.output / 'started.json').write_text(json.dumps(metadata, indent=2))
    # Initialization and teacher construction cannot shift training RNG.
    seed_everything(args.seed)
    loader = train_loader(args)
    seed_everything(args.seed)
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    start = time.monotonic()
    for step in range(1, (1 if args.smoke else args.steps) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, kl_sum = 0., 0.
        for micro in range(64 // args.batch_size):
            try:
                packed, target, _ = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                packed, target, _ = next(iterator)
            packed, target = packed.cuda(), target.cuda().long()
            if args.smoke and micro == 0:
                model.eval()
                with torch.no_grad():
                    expected = core(packed)
                    actual = model(packed)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                model.train()
            logits = model(packed).squeeze(1)
            loss = model.compute_loss(logits, target)
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(packed).squeeze(1)
                kl = bernoulli_teacher_kl(logits, teacher_logits)
                loss = loss + .1 * kl
                kl_sum += kl.item() / (64 // args.batch_size)
            if not torch.isfinite(loss):
                raise ValueError('non-finite training loss')
            (loss / (64 // args.batch_size)).backward()
            loss_sum += loss.item() / (64 // args.batch_size)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('non-finite training gradient')
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 100 == 0:
            print(json.dumps(dict(step=step, loss=loss_sum, teacher_kl=kl_sum,
                seconds=time.monotonic() - start, lr=optimizer.param_groups[0]['lr'],
                peak_gpu_bytes=torch.cuda.max_memory_allocated())), flush=True)
    metadata.update(train_seconds=time.monotonic()-start,
                    completed_steps=step, peak_gpu_bytes=torch.cuda.max_memory_allocated())
    torch.save(dict(metadata, hyper_parameters=hparams, state_dict=model.cpu().state_dict()),
               args.output / 'checkpoint.pt')
    model.cuda()
    del loader, iterator, optimizer, scheduler
    if args.smoke:
        packed, target = evaluation_dataset(args.history, 2021, 'M06')[0]
        model.eval()
        with torch.no_grad():
            result = model(packed[None].cuda())
        saved = torch.load(args.output / 'checkpoint.pt', map_location='cpu', weights_only=False)
        reloaded = make_forecaster(args.history, hparams, args.arm)
        reloaded.load_state_dict(saved['state_dict'], strict=True)
        reloaded.cuda().eval()
        with torch.no_grad():
            again = reloaded(packed[None].cuda())
        torch.testing.assert_close(result, again, rtol=0, atol=0)
        assert result.shape[-2:] == target.shape[-2:]
        return dict(reload_max_difference=0., loss=loss_sum,
                    peak_gpu_bytes=metadata['peak_gpu_bytes'])
    evaluate(model, args, args.output, args.arm)
    if teacher is not None:
        del model
        torch.cuda.empty_cache()
        evaluate(teacher, args, args.output, 'teacher')
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', type=int, choices=(1, 5), required=True)
    parser.add_argument('--arm', choices=('bn', 'control', 'mixed', 'typed', 'distill'), required=True)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--calibration-samples', type=int, default=4096)
    parser.add_argument('--bn-forward-mode', choices=('fixed', 'batch_stats'), default='fixed')
    parser.add_argument('--year', type=int, choices=(2021, 2022, 2023), default=2021)
    parser.add_argument('--evaluate-only', type=Path)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('GPU Slurm allocation required')
    if args.batch_size <= 0 or 64 % args.batch_size or args.steps < 1:
        raise ValueError('positive steps and physical batch dividing effective batch 64 required')
    if args.year != 2021 and args.evaluate_only is None:
        raise ValueError('training and calibration are restricted to training years / 2021 selection')
    torch.set_num_threads(max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', '4')) - args.workers))
    setup()
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    records = json.loads(args.manifest.read_text())['sources'][str(args.history)][str(args.seed)]
    metadata = dict(history=args.history, architecture=canonical_architecture(args.history),
        arm=args.arm, seed=args.seed, steps=args.steps, physical_batch=args.batch_size,
        effective_batch=64, lr=.001, schedule='cosine-to-zero', teacher_kl_weight=.1,
        calibration_samples=args.calibration_samples, source_records=records,
        bn_forward_mode=args.bn_forward_mode,
        job=os.environ['SLURM_JOB_ID'], smoke=args.smoke)
    (args.output / 'started.json').write_text(json.dumps(metadata, indent=2))
    if args.evaluate_only:
        saved = torch.load(args.evaluate_only, map_location='cpu', weights_only=False)
        checkpoint_architecture(saved, expected=canonical_architecture(args.history))
        if any(saved[key] != getattr(args, key) for key in ('history', 'seed', 'arm')):
            raise ValueError('evaluation checkpoint mismatch')
        if saved.get('smoke'):
            raise ValueError('smoke checkpoint cannot supply formal evaluation')
        if args.arm == 'bn' and saved.get('bn_forward_mode', 'fixed') != args.bn_forward_mode:
            raise ValueError('BN evaluation calibration mode mismatch')
        model = make_forecaster(args.history, saved['hyper_parameters'], args.arm)
        model.load_state_dict(saved['state_dict'], strict=True)
        model.cuda()
        if args.arm == 'bn':
            evaluate(model, args, args.output, 'bn_original')
            apply_bn_bank(model, saved['common'])
            evaluate(model, args, args.output, 'bn_common')
            evaluate(BankForecaster(model, saved['banks']), args, args.output, 'bn_conditional')
        else:
            evaluate(model, args, args.output, args.arm)
        outcome = dict(evaluated_checkpoint=str(args.evaluate_only))
    elif args.arm == 'bn':
        outcome = bn_diagnostic(args, records, metadata)
    else:
        outcome = learned_probe(args, records, metadata)
    (args.output / ('smoke.json' if args.smoke else 'completed.json')).write_text(
        json.dumps(dict(status='pass', **outcome), indent=2))


if __name__ == '__main__':
    main()
