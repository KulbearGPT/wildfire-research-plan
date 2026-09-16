"""Execute one fixed three-direction variant inside a GPU Slurm allocation."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from reproductions.cross_history.data import setup, base_dataset, PairedDataset, evaluation_dataset
from reproductions.cross_history.models import Forecaster, make_base
from reproductions.wsts_fast_track.evaluate_missingness import evaluate_batches
from .methods import StatisticsModel, condition_ids, midpoint_state, bernoulli_kl, teacher_ids
from .sources import resolve_sources

SCENARIOS = ('M00', 'M01', 'M06', 'M07')
MODES = ('bn_shared', 'bn_conditional', 'merge', 'x14_shared',
         'student_control', 'student_distill')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load_source(record, history, seed):
    path = Path(record['checkpoint'])
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != record['sha256']:
        raise ValueError(f'checkpoint checksum mismatch: {path}')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    for key, expected in [('history', history), ('seed', seed), ('steps', 3000),
                          ('method', record['metadata']['method'])]:
        if payload.get(key) != expected:
            raise ValueError(f'checkpoint {key} mismatch: {path}')
    if payload.get('initial_checkpoint') != record['metadata']['initial_checkpoint']:
        raise ValueError('checkpoint initialization differs from source manifest')
    return payload


def make_model(payload, history):
    base = make_base(history, payload['hyper_parameters'])
    model = Forecaster(base, history, 'control')
    model.load_state_dict(payload['state_dict'], strict=True)
    return model


def loader_for_training(history, seed, workers):
    seed_all(seed)
    dataset = PairedDataset(base_dataset(history), history,
                           fire_probability=.3, block_probability=.3)
    return DataLoader(dataset, batch_size=16, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0, drop_last=True)


def calibrate(model, args):
    model.eval()
    for layer in model.statistic_layers():
        layer.begin_calibration()
    loader = loader_for_training(args.history, args.seed, args.workers)
    histogram = torch.zeros(4, dtype=torch.long)
    limit = 8 if args.smoke else 128
    actual_batches = 0
    with torch.no_grad():
        for batch, (packed, _, _) in enumerate(loader):
            if batch == limit:
                break
            histogram += torch.bincount(condition_ids(packed), minlength=4)
            logits = model(packed.cuda(non_blocking=True))
            if not torch.isfinite(logits).all():
                raise ValueError('non-finite calibration output')
            actual_batches += 1
    if actual_batches != limit:
        raise ValueError('calibration loader shorter than registered budget')
    for layer in model.statistic_layers():
        layer.finish_calibration()
    record = dict(batches=actual_batches, samples=16 * actual_batches,
                  condition_samples=histogram.tolist(), bn_layers=len(model.statistic_layers()),
                  min_layer_support=min(int(m.count.min()) for m in model.statistic_layers()))
    if not all(histogram.tolist()):
        raise ValueError('missing training calibration condition')
    return record


def teacher_reference(sources):
    result = {}
    for scenario, role in zip(SCENARIOS, ('erm', 'x22', 'mild', 'severe')):
        record = sources[role]
        if record['summary']:
            summary = json.loads(Path(record['summary']).read_text())
            if summary['year'] != 2021:
                raise ValueError('teacher reference must use screening year')
            result[scenario] = summary['results'][scenario]
        else:
            # Original T5 seed0 evaluation stopped after M00; that completed
            # scene is sufficient for the route's ERM teacher reference.
            result[scenario] = json.loads((Path(record['checkpoint']).parent / f'{scenario}.json').read_text())
    return result


def train_student(model, payloads, args, metadata):
    teachers = []
    if args.mode == 'student_distill':
        teachers = [make_model(payloads[role], args.history).cuda().eval().requires_grad_(False)
                    for role in ('erm', 'x22', 'mild', 'severe')]
    loader = loader_for_training(args.history, args.seed, args.workers)
    # Construction of four frozen teachers must not shift student dropout RNG.
    seed_all(args.seed)
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    steps = 2 if args.smoke else 3000
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000)
    histogram = [0, 0, 0, 0]
    start = time.monotonic()
    with (args.output / 'train.jsonl').open('x') as log:
        for step in range(1, steps + 1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            supervised_total = kl_total = 0.
            for _ in range(4):
                try:
                    packed, target, _ = next(iterator)
                except StopIteration:
                    iterator = iter(loader); packed, target, _ = next(iterator)
                packed = packed.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True).long()
                logits = model(packed).squeeze(1)
                supervised = model.compute_loss(logits, target)
                kl = logits.new_zeros(())
                if teachers:
                    routes = teacher_ids(packed)
                    with torch.no_grad():
                        teacher_logits = torch.empty_like(logits)
                        for route, teacher in enumerate(teachers):
                            indices = torch.where(routes == route)[0]
                            histogram[route] += len(indices)
                            if len(indices):
                                teacher_logits[indices] = teacher(packed[indices]).squeeze(1)
                        if not torch.isfinite(teacher_logits).all():
                            raise ValueError('non-finite teacher logits')
                    kl = bernoulli_kl(logits, teacher_logits)
                loss = supervised + .1 * kl
                if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                    raise ValueError('non-finite student loss/logits')
                (loss / 4).backward()
                supervised_total += float(supervised.detach()) / 4
                kl_total += float(kl.detach()) / 4
            optimizer.step(); scheduler.step()
            if step == 1 or step % 100 == 0 or step == steps:
                row = dict(step=step, supervised_loss=supervised_total, kl=kl_total,
                           seconds=time.monotonic()-start, learning_rate=optimizer.param_groups[0]['lr'],
                           peak_gpu_bytes=torch.cuda.max_memory_allocated())
                log.write(json.dumps(row, allow_nan=False)+'\n'); log.flush()
                print(json.dumps(row), flush=True)
    metadata.update(updates=steps, effective_batch=64, physical_batch=16,
                    teacher_samples=histogram, training_seconds=time.monotonic()-start)
    del loader, iterator, optimizer, scheduler, teachers
    gc.collect(); torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', type=int, choices=(1, 5), required=True)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sources', type=Path,
                        default=os.environ.get('WILDFIRE_ROUTED_SOURCES',
                                               Path(__file__).with_name('sources.json')),
                        help='source manifest; artifact paths may be relative to this file')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--checkpoint', type=Path, help='evaluate an already fitted variant')
    parser.add_argument('--year', type=int, choices=(2021, 2022, 2023), default=2021)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('GPU Slurm allocation required')
    if args.year != 2021 and args.checkpoint is None:
        raise ValueError('fitting is restricted to train years and 2021 screening')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    setup()
    sources = resolve_sources(args.sources, args.history, args.seed)
    roles = ['x22']
    if args.mode in ('merge', 'x14_shared'):
        roles.append('x14')
    if args.mode == 'student_distill':
        roles += ['erm', 'mild', 'severe']
    payloads = {role: load_source(sources[role], args.history, args.seed) for role in roles}
    if len({p['initial_checkpoint'] for p in payloads.values()}) != 1:
        raise ValueError('source initializations differ')
    seed_all(args.seed)
    payload = payloads['x14'] if args.mode == 'x14_shared' else payloads['x22']
    model = make_model(payload, args.history)
    if args.mode == 'merge':
        model.load_state_dict(midpoint_state(payloads['x22']['state_dict'],
                                             payloads['x14']['state_dict']), strict=True)
    banks = 0 if args.mode.startswith('student_') else (4 if args.mode == 'bn_conditional' else 1)
    if banks:
        model = StatisticsModel(model, banks)
    model.cuda()
    metadata = dict(history=args.history, architecture='res18_unet' if args.history == 1 else 'res18_utae',
                    seed=args.seed, mode=args.mode, year=args.year, smoke=args.smoke,
                    job=os.environ['SLURM_JOB_ID'], source_commit=os.environ.get('WILDFIRE_SOURCE_COMMIT'),
                    source_checkpoints={r: sources[r] for r in roles}, bn_banks=banks,
                    parameters=sum(p.numel() for p in model.parameters()),
                    parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                    device=torch.cuda.get_device_name(), torch_version=torch.__version__)
    write_json(args.output / 'started.json', metadata)
    start = time.monotonic()
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if any(saved[key] != metadata[key] for key in ('history', 'seed', 'mode', 'bn_banks')) or saved['smoke']:
            raise ValueError('evaluation checkpoint metadata mismatch or smoke source')
        model.load_state_dict(saved['state_dict'], strict=True)
        metadata['evaluated_checkpoint'] = str(args.checkpoint)
    elif banks:
        metadata['calibration'] = calibrate(model, args)
        write_json(args.output / 'calibration.json', metadata['calibration'])
    else:
        train_student(model, payloads, args, metadata)
    if not args.checkpoint:
        metadata['fit_seconds'] = time.monotonic() - start
        cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(dict(metadata, state_dict=cpu_state), args.output / 'checkpoint.pt')
        metadata['checkpoint_bytes'] = (args.output / 'checkpoint.pt').stat().st_size
        del cpu_state
    results = {}
    model.eval()
    for scenario in SCENARIOS:
        dataset = evaluation_dataset(args.history, args.year, scenario)
        if args.smoke:
            dataset = Subset(dataset, range(2))
        loader = DataLoader(dataset, batch_size=16, num_workers=args.workers, pin_memory=True)
        scene_start = time.monotonic()
        metrics = evaluate_batches(model, loader, device=torch.device('cuda'))
        metrics['evaluation_seconds'] = time.monotonic()-scene_start
        results[scenario] = metrics
        write_json(args.output / f'{scenario}.json', metrics)
        print(json.dumps(dict(scenario=scenario, metrics=metrics)), flush=True)
        del loader, dataset
        gc.collect()
    metadata.update(results=results, total_seconds=time.monotonic()-start,
                    peak_gpu_bytes=torch.cuda.max_memory_allocated())
    if args.year == 2021:
        metadata['route_reference'] = teacher_reference(sources)
        metadata['route_checkpoint_bytes'] = sum(Path(sources[r]['checkpoint']).stat().st_size
                                                 for r in ('erm', 'x22', 'mild', 'severe'))
        metadata['original_x22'] = json.loads(Path(sources['x22']['summary']).read_text())['results']
        metadata['original_x14'] = json.loads(Path(sources['x14']['summary']).read_text())['results']
    write_json(args.output / ('smoke.json' if args.smoke else 'summary.json'), metadata)


if __name__ == '__main__':
    main()
