"""Read paired numerical evidence and apply the pre-registered seed-0 gates."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean

SCENARIOS=('M00','M01','M06','M07')
PRIMARY=SCENARIOS[1:]
MODES=('bn_shared','bn_conditional','merge','x14_shared','student_control','student_distill')


def values(metrics):
    if set(metrics) != set(SCENARIOS):
        raise ValueError('all four scenarios are required')
    result={}
    for scene in SCENARIOS:
        value=float(metrics[scene]['avg_precision'])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('invalid AP')
        if metrics[scene]['sample_count'] != 3181:
            raise ValueError('incorrect 2021 evaluation population')
        result[scene]=value
    return result


def delta(candidate, control):
    per_scene={s:candidate[s]-control[s] for s in SCENARIOS}
    return dict(scenarios=per_scene,primary=mean(per_scene[s] for s in PRIMARY),
                block=mean(per_scene[s] for s in ('M06','M07')),
                worst_scene=min(per_scene.values()))


def costs(row):
    result={key:row[key] for key in ('checkpoint_bytes','route_checkpoint_bytes',
        'parameter_bytes','training_seconds','fit_seconds','total_seconds',
        'updates','peak_gpu_bytes') if key in row}
    if result.get('route_checkpoint_bytes') and 'checkpoint_bytes' in result:
        result['checkpoint_to_route_ratio']=result['checkpoint_bytes']/result['route_checkpoint_bytes']
    if all('evaluation_seconds' in m for m in row['results'].values()):
        result['evaluation_seconds']=sum(m['evaluation_seconds'] for m in row['results'].values())
    return result


def assess(rows):
    indexed={}
    for row in rows:
        if row['smoke'] or row['year'] != 2021 or row['seed'] != 0 or row['history'] not in (1,5):
            raise ValueError('screen requires non-smoke 2021 seed0 T1/T5 runs')
        key=row['history'],row['mode']
        if key in indexed or row['mode'] not in MODES:
            raise ValueError('duplicate or unknown run')
        indexed[key]=dict(ap=values(row['results']), original=values(row['original_x22']),
                          route=values(row['route_reference']), path=row.get('path'), cost=costs(row))
    output={}
    for direction,required in [('N',('bn_conditional','bn_shared')),
                               ('W',('merge','bn_shared','x14_shared')),
                               ('D',('student_distill','student_control'))]:
        cells=[]; missing=[]
        for history in (1,5):
            absent=[mode for mode in required if (history,mode) not in indexed]
            if absent:
                missing.append(dict(history=history,modes=absent));continue
            candidate=indexed[history,required[0]]
            control=indexed[history,required[1]]
            if candidate['original'] != control['original'] or candidate['route'] != control['route']:
                raise ValueError('paired source reference metrics differ')
            nearest=delta(candidate['ap'],control['ap'])
            original=delta(candidate['ap'],candidate['original'])
            route=delta(candidate['ap'],candidate['route'])
            if direction=='N':
                passed=(nearest['primary']>=.005 and original['primary']>=.005
                        and original['scenarios']['M00']>=-.010)
            elif direction=='W':
                passed=(nearest['primary']>=.005 and original['primary']>0
                        and original['scenarios']['M00']>=-.010)
            else:
                passed=(route['primary']>=-.005 and route['worst_scene']>=-.010
                        and nearest['primary']>0)
            cell=dict(history=history,ap=candidate['ap'],vs_nearest=nearest,
                      vs_original_x22=original,vs_retained_route=route,pass_gate=passed,
                      candidate_cost=candidate['cost'],control_cost=control['cost'])
            if direction=='W':
                cell['vs_recalibrated_x14']=delta(candidate['ap'],indexed[history,'x14_shared']['ap'])
            cells.append(cell)
        output[direction]=dict(status='incomplete' if missing else 'complete',missing=missing,
                               cells=cells,screen_pass=not missing and all(c['pass_gate'] for c in cells))
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rows=[]
    for run in args.run:
        path=run/'summary.json' if run.is_dir() else run
        row=json.loads(path.read_text());row['path']=str(path);rows.append(row)
    result=assess(rows)
    result['runs']=[r['path'] for r in rows]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
