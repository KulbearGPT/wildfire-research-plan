"""Compose observable FireDrop and spatial specialists from saved summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

from .compare import BLOCK, PRIMARY, SCENARIOS, aggregate, load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', type=Path, action='append', required=True)
    parser.add_argument('--fire', type=Path, action='append', required=True)
    parser.add_argument('--spatial', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if len({len(args.control),len(args.fire),len(args.spatial)}) != 1:
        raise ValueError('control, fire, and spatial summary counts differ')

    rows=[]
    component_methods={'fire':set(),'spatial':set()}
    for control_path,fire_path,spatial_path in zip(args.control,args.fire,args.spatial):
        cmeta,control=load(control_path)
        fmeta,fire=load(fire_path)
        smeta,spatial=load(spatial_path)
        keys=('history','seed','year')
        if any(cmeta.get(k) != fmeta.get(k) or cmeta.get(k) != smeta.get(k) for k in keys):
            raise ValueError('matched history/seed/year differ')
        component_methods['fire'].add(fmeta['method'])
        component_methods['spatial'].add(smeta['method'])
        effective=dict(M00=control['M00'],M01=fire['M01'],
                       M06=spatial['M06'],M07=spatial['M07'])
        delta={name:effective[name]-control[name] for name in SCENARIOS}
        rows.append(dict(history=cmeta['history'],seed=cmeta['seed'],year=cmeta['year'],
            control=control,fire=fire,spatial=spatial,effective=effective,delta=delta,
            primary_delta=mean(delta[x] for x in PRIMARY),
            block_delta=mean(delta[x] for x in BLOCK)))
    if any(len(methods) != 1 for methods in component_methods.values()):
        raise ValueError('component methods differ across matched rows')

    grouped=[aggregate(rows,history,year) for history in sorted({r['history'] for r in rows})
             for year in sorted({r['year'] for r in rows})
             if any(r['history'] == history and r['year'] == year for r in rows)]
    confirmation=[g for g in grouped if g['year'] == 2021 and {0,1,2}.issubset(g['seeds'])]
    heldout=[g for g in grouped if g['year'] in (2022,2023)]
    result=dict(
        component_methods={name:methods.pop() for name,methods in component_methods.items()},
        rows=rows,grouped=grouped,
        mean_primary_delta=mean(r['primary_delta'] for r in rows),
        std_primary_delta=pstdev(r['primary_delta'] for r in rows),
        mean_block_delta=mean(r['block_delta'] for r in rows),
        magnitude_target_met=mean(r['primary_delta'] for r in rows) >= .02,
        confirmation_pass=(len(confirmation) == 2 and
            {g['history'] for g in confirmation} == {1,5} and
            all(g['primary_delta'] > 0 and g['clean_delta'] >= -.01 for g in confirmation)),
        heldout_pass=(len(heldout) == 4 and
            {(g['history'],g['year']) for g in heldout} ==
            {(1,2022),(1,2023),(5,2022),(5,2023)} and
            all(g['primary_delta'] > 0 and g['clean_delta'] >= -.01 for g in heldout)),
    )
    rendered=json.dumps(result,indent=2,sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(rendered+'\n')


if __name__ == '__main__': main()
