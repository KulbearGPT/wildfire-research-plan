"""Read-only sanity audit after the conditional-BN screen regressed sharply."""
import copy
import json
import os
from pathlib import Path
import torch
from .methods import StatisticsModel, condition_ids, teacher_ids
from .run import load_source, make_model
from reproductions.cross_history.data import setup, evaluation_dataset


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    torch.set_num_threads(1)
    setup()
    sources=json.loads(Path(__file__).with_name('sources.json').read_text())
    output=[]
    for history,job in [(1,21787044),(5,21787050)]:
        source=load_source(sources[f't{history}-s0']['x22'],history,0)
        model=make_model(source,history).eval()
        samples=[]
        for scenario in ('M00','M01','M06','M07'):
            dataset=evaluation_dataset(history,2021,scenario)
            samples.extend(dataset[i][0] for i in (0,100,1000))
        packed=torch.stack(samples)
        groups=condition_ids(packed).tolist()
        assert groups==[0]*3+[1]*3+[2]*6,groups
        routes=teacher_ids(packed).tolist()
        assert routes==[0]*3+[1]*3+[2]*3+[3]*3,routes
        with torch.no_grad():expected=model(packed)
        wrapped=StatisticsModel(model,4).eval()
        with torch.no_grad():initial=wrapped(packed)
        initial_error=float((initial-expected).abs().max())
        assert initial_error<1e-5,initial_error
        checkpoint=Path(f'/project/6085198/kulbear/wildfire/runs/three-directions-{job}/result/checkpoint.pt')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        wrapped.load_state_dict(saved['state_dict'],strict=True)
        for name,param in wrapped.named_parameters():
            assert torch.equal(param,source['state_dict'][name.removeprefix('forecaster.')]),name
        with torch.no_grad():after=wrapped(packed)
        assert torch.isfinite(after).all()
        output.append(dict(history=history,groups=groups,teacher_routes=routes,
            original_wrapper_max_error=initial_error,all_parameters_unchanged=True,
            calibration_samples=saved['calibration']['samples'],
            group_samples=saved['calibration']['condition_samples'],
            scenes=[dict(scenario=s,original_logit_min=float(expected[3*i:3*i+3].min()),
                original_logit_max=float(expected[3*i:3*i+3].max()),
                calibrated_logit_min=float(after[3*i:3*i+3].min()),
                calibrated_logit_max=float(after[3*i:3*i+3].max()))
                for i,s in enumerate(('M00','M01','M06','M07'))]))
    print(json.dumps(output,indent=2,allow_nan=False))


if __name__=='__main__':main()
