"""Small matched experiment driver; execute only inside a Slurm allocation."""
import argparse
import copy
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from .data import setup, base_dataset, PairedDataset, evaluation_dataset
from .models import initial_payload, make_base, Forecaster, feature_loss, transition_loss
from reproductions.wsts_fast_track.evaluate_missingness import evaluate_batches


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--history',type=int,choices=(1,5),required=True)
    p.add_argument('--method',choices=('control','balanced_corruption','context','context_adapter','distill','distill_block','risk','risk_strong','global_consistency','local_consistency','impact_consistency','transition','transition_decoupled','context_transition','missingness_experts'),required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--evaluate-only',type=Path)
    p.add_argument('--year',type=int,choices=(2021,2022,2023),default=2021)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('GPU Slurm allocation required')
    if 64 % a.batch_size:
        raise ValueError('physical batch must divide effective batch 64')
    if a.year != 2021 and not a.evaluate_only:
        raise ValueError('training screens are 2021-only')
    torch.set_num_threads(max(1,int(os.environ.get('SLURM_CPUS_PER_TASK','4'))-a.workers))
    setup()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    a.output.mkdir(parents=True,exist_ok=False)
    payload, initial = initial_payload(a.history)
    base = make_base(a.history,payload['hyper_parameters'])
    base.load_state_dict(payload['state_dict'],strict=True)
    model = Forecaster(base,a.history,a.method).cuda()
    if a.method == 'context_adapter':
        model.base.requires_grad_(False)
    corruption_probability = .5 if a.method == 'balanced_corruption' else .3
    metadata = dict(history=a.history,method=a.method,seed=a.seed,steps=a.steps,
                    physical_batch=a.batch_size,effective_batch=64,learning_rate=.001,
                    fire_dropout_probability=corruption_probability,
                    block_dropout_probability=corruption_probability,
                    initial_checkpoint=initial,job=os.environ['SLURM_JOB_ID'],
                    parameters=sum(x.numel() for x in model.parameters()),
                    trainable_parameters=sum(x.numel() for x in model.parameters() if x.requires_grad))
    (a.output/'started.json').write_text(json.dumps(metadata,indent=2))
    if a.evaluate_only:
        saved = torch.load(a.evaluate_only,map_location='cpu',weights_only=False)
        if saved['history'] != a.history or saved['method'] != a.method:
            raise ValueError('checkpoint method/history mismatch')
        model.load_state_dict(saved['state_dict'],strict=True)
    else:
        dataset = PairedDataset(base_dataset(a.history),a.history,
            fire_probability=corruption_probability,block_probability=corruption_probability)
        loader = torch.utils.data.DataLoader(dataset,batch_size=a.batch_size,shuffle=True,
            generator=torch.Generator().manual_seed(a.seed),num_workers=a.workers,
            pin_memory=True,persistent_workers=a.workers>0,drop_last=True)
        distill_teacher = copy.deepcopy(model).eval().requires_grad_(False) if a.method.startswith('distill') else None
        # Equal data and model RNG across methods; module construction must not
        # shift augmentation or temporal dropout relative to the control.
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        iterator = iter(loader)
        optimizer = torch.optim.AdamW((x for x in model.parameters() if x.requires_grad),lr=.001)
        start = time.monotonic()
        for step in range(1,(1 if a.smoke else a.steps)+1):
            model.train()
            if a.method == 'context_adapter': model.base.eval()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.
            for micro in range(64//a.batch_size):
                try: packed,target,clean = next(iterator)
                except StopIteration:
                    iterator = iter(loader); packed,target,clean = next(iterator)
                packed,target,clean = packed.cuda(),target.cuda().long(),clean.cuda()
                clean_logits = None
                if a.method in ('global_consistency','local_consistency','impact_consistency'):
                    zeros = clean.new_zeros(clean.shape[0],clean.shape[1],2,*clean.shape[-2:])
                    clean_logits = model(torch.cat((clean,zeros),dim=2)).squeeze(1)
                if a.smoke and micro == 0:
                    model.eval()
                    with torch.no_grad():
                        expected = base(packed[:,:,:model.channels])
                        actual = model(packed)
                        difference = (expected-actual).abs().max().item()
                    print(f'INITIAL_EQUIVALENCE_MAX={difference}',flush=True)
                    if difference > (1e-3 if a.method in ('transition','transition_decoupled','context_transition') else 1e-5):
                        raise RuntimeError('initial forward mismatch')
                    model.train()
                    if a.method == 'context_adapter': model.base.eval()
                logits,features,aux = model(packed,details=True)
                if a.method in ('risk','risk_strong'):
                    from torchvision.ops import sigmoid_focal_loss
                    raw_loss = sigmoid_focal_loss(logits.squeeze(1),target.float(),
                        alpha=1-float(base.hparams.pos_class_weight),gamma=2,reduction='none')
                    spatial = packed[:,-1,-2]
                    risk_weight = 1 + (4 if a.method == 'risk_strong' else 2)*spatial
                    loss = (raw_loss*risk_weight).sum()/risk_weight.sum()
                else:
                    loss = model.compute_loss(logits.squeeze(1),target)
                if clean_logits is not None:
                    loss = .5*(model.compute_loss(clean_logits,target)+loss)
                    teacher_logits = clean_logits.detach()
                    probability = teacher_logits.sigmoid()
                    per_pixel_kl = probability*(torch.nn.functional.logsigmoid(teacher_logits)-torch.nn.functional.logsigmoid(logits.squeeze(1)))
                    per_pixel_kl += (1-probability)*(torch.nn.functional.logsigmoid(-teacher_logits)-torch.nn.functional.logsigmoid(-logits.squeeze(1)))
                    if a.method == 'global_consistency':
                        consistency = per_pixel_kl.mean()
                    elif a.method == 'local_consistency':
                        invalid = packed[:,-1,-2:].amax(dim=1)
                        confidence = (2*probability-1).abs()
                        weight = invalid*(.5+.5*confidence)
                        consistency = (per_pixel_kl*weight).sum()/weight.sum().clamp_min(1.)
                    else:
                        # Counterfactual impact can extend outside the missing
                        # input support; normalize it independently per sample.
                        impact = (probability-logits.squeeze(1).sigmoid().detach()).abs()
                        mean_impact = impact.mean(dim=(-2,-1),keepdim=True)
                        weight = torch.where(mean_impact > 0,
                            impact/mean_impact.clamp_min(1e-6),torch.zeros_like(impact))
                        consistency = (per_pixel_kl*weight).mean()
                    loss = loss + .1*consistency
                if distill_teacher is not None:
                    with torch.no_grad():
                        tf = distill_teacher.features(clean)[-3:]
                    missing = packed[:,-1,-2:-1] if a.method == 'distill_block' else packed[:,-1,-1:]
                    loss = loss + .05*feature_loss(features,tf,missing,target,
                        spatial_only=a.method == 'distill_block')
                if aux is not None:
                    loss = loss + transition_loss(aux,clean,target)
                if not torch.isfinite(loss): raise RuntimeError('nonfinite training loss')
                (loss/(64//a.batch_size)).backward()
                loss_sum += loss.detach().item()/(64//a.batch_size)
            optimizer.step()
            if step == 1 or step % 100 == 0:
                print(json.dumps(dict(step=step,loss=loss_sum,seconds=time.monotonic()-start,
                    peak_gpu_bytes=torch.cuda.max_memory_allocated())),flush=True)
        metadata['state_dict'] = model.cpu().state_dict()
        metadata['hyper_parameters'] = dict(base.hparams)
        torch.save(metadata,a.output/'checkpoint.pt')
        model.cuda()
        if a.smoke:
            sample_x,sample_y = evaluation_dataset(a.history,2021,'M06')[0]
            model.eval()
            with torch.no_grad(): result = model(sample_x[None].cuda())
            assert result.shape[-2:] == sample_y.shape[-2:]
            (a.output/'smoke.json').write_text(json.dumps(dict(status='pass',history=a.history,method=a.method)))
            return
        del loader,iterator,optimizer,metadata['state_dict']
        if distill_teacher is not None: del distill_teacher
    results = {}
    for scenario in ('M00','M01','M06','M07'):
        dataset = evaluation_dataset(a.history,a.year,scenario)
        loader = torch.utils.data.DataLoader(dataset,batch_size=min(a.batch_size,16),
            num_workers=a.workers,pin_memory=True)
        metrics = evaluate_batches(model,loader,device=torch.device('cuda'))
        results[scenario] = metrics
        (a.output/f'{scenario}.json').write_text(json.dumps(metrics,indent=2))
        print(json.dumps(dict(scenario=scenario,metrics=metrics)),flush=True)
    (a.output/'summary.json').write_text(json.dumps(dict(history=a.history,method=a.method,
        seed=a.seed,year=a.year,results=results),indent=2))


if __name__ == '__main__': main()
