# Code lifecycle: archive_support. Original TD diagnostics; Slurm-only, including imports.
# Scope and settings: reproductions/three_directions/README.md; docs/research/method-inventory.json.
"""Read-only audit of final student checkpoints and their reported evidence."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import torch
from reproductions.cross_history.data import setup
from .run import load_source, make_model


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    torch.set_num_threads(1);setup()
    records=[]
    for run in args.run:
        summary=json.loads((run/'summary.json').read_text())
        checkpoint=run/'checkpoint.pt'
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        for key in ('history','seed','mode','source_commit','bn_banks','smoke','updates'):
            assert saved[key]==summary[key],key
        assert not saved['smoke'] and saved['updates']==3000 and saved['bn_banks']==0
        assert summary['year']==2021
        assert saved['mode'] in ('student_control','student_distill')
        source=load_source(saved['source_checkpoints']['x22'],saved['history'],saved['seed'])
        model=make_model(source,saved['history'])
        model.load_state_dict(saved['state_dict'],strict=True)
        assert all(torch.isfinite(v).all() for v in saved['state_dict'].values() if v.is_floating_point())
        changed=sum(not torch.equal(v,source['state_dict'][k]) for k,v in saved['state_dict'].items())
        assert changed>0
        assert sum(p.numel()*p.element_size() for p in model.parameters())==summary['parameter_bytes']
        assert checkpoint.stat().st_size==summary['checkpoint_bytes']
        assert set(summary['results'])=={'M00','M01','M06','M07'}
        for metrics in summary['results'].values():
            assert metrics['sample_count']==3181 and metrics['pixel_count']==52117504
            assert all(math.isfinite(v) for v in metrics.values())
        digest=hashlib.sha256()
        with checkpoint.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
        records.append(dict(run=str(run),history=saved['history'],seed=saved['seed'],mode=saved['mode'],
            updates=saved['updates'],strict_load=True,all_state_finite=True,changed_state_tensors=changed,
            checkpoint_sha256=digest.hexdigest(),checkpoint_bytes=checkpoint.stat().st_size))
    args.output.write_text(json.dumps(dict(job=os.environ['SLURM_JOB_ID'],records=records),indent=2)+'\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
