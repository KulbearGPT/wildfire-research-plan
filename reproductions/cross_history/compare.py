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


def aggregate(rows, history, year):
    selected = [r for r in rows if r['history'] == history and r['year'] == year]
    return dict(history=history,year=year,seeds=sorted(r['seed'] for r in selected),
        primary_delta=mean(r['primary_delta'] for r in selected),
        block_delta=mean(r['block_delta'] for r in selected),
        clean_delta=mean(r['delta']['M00'] for r in selected))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control',type=Path,action='append',required=True)
    parser.add_argument('--candidate',type=Path,action='append',required=True)
    routing = parser.add_mutually_exclusive_group()
    routing.add_argument('--routed-spatial',action='store_true')
    routing.add_argument('--routed-fire',action='store_true')
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
        if args.routed_fire:
            effective['M00']=control['M00']
            effective['M06']=control['M06']; effective['M07']=control['M07']
        delta={name:effective[name]-control[name] for name in SCENARIOS}
        rows.append(dict(history=cmeta['history'],seed=cmeta['seed'],year=cmeta['year'],
            control=control,candidate=candidate,effective=effective,delta=delta,
            primary_delta=mean(delta[x] for x in PRIMARY),
            block_delta=mean(delta[x] for x in BLOCK)))
    methods={load(path)[0]['method'] for path in args.candidate}
    if len(methods) != 1:
        raise ValueError('candidate methods differ')
    grouped=[aggregate(rows,history,year) for history in sorted({r['history'] for r in rows})
             for year in sorted({r['year'] for r in rows})
             if any(r['history'] == history and r['year'] == year for r in rows)]
    screen=[r for r in rows if r['year'] == 2021 and r['seed'] == 0]
    screen_pass=(len(screen) == 2 and {r['history'] for r in screen} == {1,5}
                 and all(r['primary_delta'] >= .005 and r['delta']['M00'] >= -.01 for r in screen))
    confirmation=[g for g in grouped if g['year'] == 2021 and {0,1,2}.issubset(g['seeds'])]
    confirmation_pass=(len(confirmation) == 2 and {g['history'] for g in confirmation} == {1,5}
                       and all(g['primary_delta'] > 0 and g['clean_delta'] >= -.01 for g in confirmation))
    heldout=[g for g in grouped if g['year'] in (2022,2023)]
    heldout_pass=(len(heldout) == 4 and {(g['history'],g['year']) for g in heldout}
                  == {(1,2022),(1,2023),(5,2022),(5,2023)}
                  and all(g['primary_delta'] > 0 and g['clean_delta'] >= -.01 for g in heldout))
    mean_primary=mean(r['primary_delta'] for r in rows)
    result=dict(method=methods.pop(),routed_spatial=args.routed_spatial,
        routed_fire=args.routed_fire,
        rows=rows,mean_primary_delta=mean_primary,
        std_primary_delta=pstdev(r['primary_delta'] for r in rows),
        mean_block_delta=mean(r['block_delta'] for r in rows),
        worst_clean_delta=min(r['delta']['M00'] for r in rows),
        magnitude_target_met=mean_primary >= .02,
        grouped=grouped,screen_pass=screen_pass,
        confirmation_pass=confirmation_pass,heldout_pass=heldout_pass,
        goal_evidence_pass=confirmation_pass and heldout_pass)
    rendered=json.dumps(result,indent=2,sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(rendered+'\n')


if __name__ == '__main__': main()
