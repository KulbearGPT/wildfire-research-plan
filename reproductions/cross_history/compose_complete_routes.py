"""Compose clean, FireDrop, mild-block, and severe-block predictions."""
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
    parser.add_argument('--mild', type=Path, action='append', required=True)
    parser.add_argument('--severe', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if len({len(args.control), len(args.fire), len(args.mild), len(args.severe)}) != 1:
        raise ValueError('control, fire, mild, and severe summary counts differ')

    rows = []
    component_methods = None
    for control_path, fire_path, mild_path, severe_path in zip(
            args.control, args.fire, args.mild, args.severe):
        cmeta, control = load(control_path)
        fmeta, fire = load(fire_path)
        mmeta, mild = load(mild_path)
        smeta, severe = load(severe_path)
        keys = ('history', 'seed', 'year')
        if any(cmeta.get(key) != meta.get(key)
               for key in keys for meta in (fmeta, mmeta, smeta)):
            raise ValueError('matched history/seed/year differ')
        if mmeta.get('method') != 'block_specialist' or mmeta.get('block_fraction') != .25:
            raise ValueError('mild input must be a fixed-0.25 block specialist')
        if smeta.get('method') != 'block_specialist' or smeta.get('block_fraction') != .5:
            raise ValueError('severe input must be a fixed-0.50 block specialist')
        methods = dict(control=cmeta.get('method'), fire=fmeta.get('method'),
                       mild=mmeta.get('method'), severe=smeta.get('method'))
        if component_methods is None:
            component_methods = methods
        elif component_methods != methods:
            raise ValueError('component methods differ across matched rows')
        effective = dict(M00=control['M00'], M01=fire['M01'],
                         M06=mild['M06'], M07=severe['M07'])
        delta = {name: effective[name] - control[name] for name in SCENARIOS}
        rows.append(dict(history=cmeta['history'], seed=cmeta['seed'],
            year=cmeta['year'], control=control, fire=fire, mild=mild,
            severe=severe, effective=effective, delta=delta,
            primary_delta=mean(delta[name] for name in PRIMARY),
            block_delta=mean(delta[name] for name in BLOCK)))

    grouped = [aggregate(rows, history, year)
        for history in sorted({row['history'] for row in rows})
        for year in sorted({row['year'] for row in rows})
        if any(row['history'] == history and row['year'] == year for row in rows)]
    confirmation = [group for group in grouped
        if group['year'] == 2021 and {0, 1, 2}.issubset(group['seeds'])]
    heldout = [group for group in grouped if group['year'] in (2022, 2023)]
    mean_primary = mean(row['primary_delta'] for row in rows)
    result = dict(method='complete_observable_route',
        component_methods=component_methods, rows=rows, grouped=grouped,
        mean_primary_delta=mean_primary,
        std_primary_delta=pstdev(row['primary_delta'] for row in rows),
        mean_block_delta=mean(row['block_delta'] for row in rows),
        worst_clean_delta=min(row['delta']['M00'] for row in rows),
        magnitude_target_met=mean_primary >= .02,
        confirmation_pass=(len(confirmation) == 2
            and {group['history'] for group in confirmation} == {1, 5}
            and all(group['primary_delta'] > 0 and group['clean_delta'] >= -.01
                    for group in confirmation)),
        heldout_pass=(len(heldout) == 4
            and {(group['history'], group['year']) for group in heldout}
                == {(1, 2022), (1, 2023), (5, 2022), (5, 2023)}
            and all(group['primary_delta'] > 0 and group['clean_delta'] >= -.01
                    for group in heldout)))
    result['goal_evidence_pass'] = result['confirmation_pass'] and result['heldout_pass']
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n')


if __name__ == '__main__':
    main()
