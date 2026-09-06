"""Compare matched cross-history summaries without loading data or models."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

SCENARIOS = ('M00','M01','M06','M07')
PRIMARY = ('M01','M06','M07')
BLOCK = ('M06','M07')


def load(path: Path):
    payload = json.loads(path.read_text())
    values = {name: float(payload['results'][name]['avg_precision']) for name in SCENARIOS}
    return payload, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control',type=Path,action='append',required=True)
    parser.add_argument('--candidate',type=Path,action='append',required=True)
    parser.add_argument('--routed-spatial',action='store_true')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if len(args.control) != len(args.candidate):
        raise ValueError('control and candidate counts differ')
    rows=[]
    for control_path,candidate_path in zip(args.control,args.candidate):
        cmeta,control=load(control_path); mmeta,candidate=load(candidate_path)
        keys=('history','seed','year')
        if any(cmeta.get(k) != mmeta.get(k) for k in keys):
            raise ValueError('matched history/seed/year differ')
        effective = dict(candidate)
        if args.routed_spatial:
            effective['M00']=control['M00']; effective['M01']=control['M01']
        delta={name:effective[name]-control[name] for name in SCENARIOS}
        rows.append(dict(history=cmeta['history'],seed=cmeta['seed'],year=cmeta['year'],
            control=control,candidate=candidate,effective=effective,delta=delta,
            primary_delta=mean(delta[x] for x in PRIMARY),
            block_delta=mean(delta[x] for x in BLOCK)))
    result=dict(method=load(args.candidate[0])[0]['method'],routed_spatial=args.routed_spatial,
        rows=rows,mean_primary_delta=mean(r['primary_delta'] for r in rows),
        std_primary_delta=pstdev(r['primary_delta'] for r in rows),
        mean_block_delta=mean(r['block_delta'] for r in rows),
        worst_clean_delta=min(r['delta']['M00'] for r in rows),
        screen_pass=all(r['primary_delta'] >= .005 and r['delta']['M00'] >= -.01 for r in rows))
    rendered=json.dumps(result,indent=2,sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(rendered+'\n')


if __name__ == '__main__': main()
