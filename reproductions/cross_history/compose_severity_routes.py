"""Compose fixed-25% and fixed-50% BlockDrop specialists by observed severity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

from .compare import BLOCK, PRIMARY, SCENARIOS, aggregate, load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', type=Path, action='append', required=True)
    parser.add_argument('--mild', type=Path, action='append', required=True,
                        help='summary from the fixed-0.25 specialist')
    parser.add_argument('--severe', type=Path, action='append', required=True,
                        help='summary from the fixed-0.50 specialist')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if len({len(args.control), len(args.mild), len(args.severe)}) != 1:
        raise ValueError('control, mild, and severe summary counts differ')

    rows = []
    for control_path, mild_path, severe_path in zip(
            args.control, args.mild, args.severe):
        cmeta, control = load(control_path)
        mmeta, mild = load(mild_path)
        smeta, severe = load(severe_path)
        keys = ('history', 'seed', 'year')
        if any(cmeta.get(k) != mmeta.get(k) or cmeta.get(k) != smeta.get(k)
               for k in keys):
            raise ValueError('matched history/seed/year differ')
        if mmeta.get('method') != 'block_specialist' or mmeta.get('block_fraction') != .25:
            raise ValueError('mild input must be a fixed-0.25 block_specialist')
        if smeta.get('method') != 'block_specialist' or smeta.get('block_fraction') != .5:
            raise ValueError('severe input must be a fixed-0.50 block_specialist')
        effective = dict(M00=control['M00'], M01=control['M01'],
                         M06=mild['M06'], M07=severe['M07'])
        delta = {name: effective[name] - control[name] for name in SCENARIOS}
        rows.append(dict(history=cmeta['history'], seed=cmeta['seed'],
            year=cmeta['year'], control=control, mild=mild, severe=severe,
            effective=effective, delta=delta,
            primary_delta=mean(delta[x] for x in PRIMARY),
            block_delta=mean(delta[x] for x in BLOCK)))

    grouped = [aggregate(rows, history, year)
        for history in sorted({r['history'] for r in rows})
        for year in sorted({r['year'] for r in rows})
        if any(r['history'] == history and r['year'] == year for r in rows)]
    screen = [r for r in rows if r['year'] == 2021 and r['seed'] == 0]
    confirmation = [g for g in grouped
        if g['year'] == 2021 and {0, 1, 2}.issubset(g['seeds'])]
    heldout = [g for g in grouped if g['year'] in (2022, 2023)]
    result = dict(
        method='severity_factorized_block_specialist',
        rows=rows, grouped=grouped,
        mean_primary_delta=mean(r['primary_delta'] for r in rows),
        std_primary_delta=pstdev(r['primary_delta'] for r in rows),
        mean_block_delta=mean(r['block_delta'] for r in rows),
        worst_clean_delta=min(r['delta']['M00'] for r in rows),
        magnitude_target_met=mean(r['primary_delta'] for r in rows) >= .02,
        screen_pass=(len(screen) == 2 and {r['history'] for r in screen} == {1, 5}
            and all(r['primary_delta'] >= .005 and r['delta']['M00'] >= -.01
                    for r in screen)),
        confirmation_pass=(len(confirmation) == 2
            and {g['history'] for g in confirmation} == {1, 5}
            and all(g['primary_delta'] > 0 and g['clean_delta'] >= -.01
                    for g in confirmation)),
        heldout_pass=(len(heldout) == 4
            and {(g['history'], g['year']) for g in heldout}
                == {(1, 2022), (1, 2023), (5, 2022), (5, 2023)}
            and all(g['primary_delta'] > 0 and g['clean_delta'] >= -.01
                    for g in heldout)),
    )
    result['goal_evidence_pass'] = result['confirmation_pass'] and result['heldout_pass']
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n')


if __name__ == '__main__':
    main()
