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
    parser.add_argument('--mixed', type=Path, action='append', required=True,
                        help='summary from the mixed-0.25/0.50 X14 specialist')
    parser.add_argument('--mild', type=Path, action='append', required=True,
                        help='summary from the fixed-0.25 specialist')
    parser.add_argument('--severe', type=Path, action='append', required=True,
                        help='summary from the fixed-0.50 specialist')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if len({len(args.control), len(args.mixed), len(args.mild), len(args.severe)}) != 1:
        raise ValueError('control, mixed, mild, and severe summary counts differ')

    rows = []
    for control_path, mixed_path, mild_path, severe_path in zip(
            args.control, args.mixed, args.mild, args.severe):
        cmeta, control = load(control_path)
        xmeta, mixed = load(mixed_path)
        mmeta, mild = load(mild_path)
        smeta, severe = load(severe_path)
        keys = ('history', 'seed', 'year')
        if any(cmeta.get(k) != xmeta.get(k) or cmeta.get(k) != mmeta.get(k)
               or cmeta.get(k) != smeta.get(k)
               for k in keys):
            raise ValueError('matched history/seed/year differ')
        if xmeta.get('method') != 'block_specialist' or xmeta.get('block_fraction') is not None:
            raise ValueError('mixed input must be the mixed-severity X14 block_specialist')
        if mmeta.get('method') != 'block_specialist' or mmeta.get('block_fraction') != .25:
            raise ValueError('mild input must be a fixed-0.25 block_specialist')
        if smeta.get('method') != 'block_specialist' or smeta.get('block_fraction') != .5:
            raise ValueError('severe input must be a fixed-0.50 block_specialist')
        effective = dict(M00=control['M00'], M01=control['M01'],
                         M06=mild['M06'], M07=severe['M07'])
        mixed_effective = dict(M00=control['M00'], M01=control['M01'],
                               M06=mixed['M06'], M07=mixed['M07'])
        delta = {name: effective[name] - control[name] for name in SCENARIOS}
        mechanism_delta = {
            name: effective[name] - mixed_effective[name] for name in SCENARIOS
        }
        rows.append(dict(history=cmeta['history'], seed=cmeta['seed'],
            year=cmeta['year'], control=control, mixed=mixed, mild=mild,
            severe=severe, effective=effective, delta=delta,
            mechanism_delta=mechanism_delta,
            primary_delta=mean(delta[x] for x in PRIMARY),
            block_delta=mean(delta[x] for x in BLOCK),
            mechanism_block_delta=mean(mechanism_delta[x] for x in BLOCK)))

    grouped = [aggregate(rows, history, year)
        for history in sorted({r['history'] for r in rows})
        for year in sorted({r['year'] for r in rows})
        if any(r['history'] == history and r['year'] == year for r in rows)]
    screen = [r for r in rows if r['year'] == 2021 and r['seed'] == 0]
    confirmation = [g for g in grouped
        if g['year'] == 2021 and {0, 1, 2}.issubset(g['seeds'])]
    heldout = [g for g in grouped if g['year'] in (2022, 2023)]
    mechanism_grouped = [dict(history=history, year=year,
        seeds=sorted(r['seed'] for r in selected),
        block_delta=mean(r['mechanism_block_delta'] for r in selected))
        for history in sorted({r['history'] for r in rows})
        for year in sorted({r['year'] for r in rows})
        if (selected := [r for r in rows
                         if r['history'] == history and r['year'] == year])]
    mechanism_confirmation = [g for g in mechanism_grouped
        if g['year'] == 2021 and {0, 1, 2}.issubset(g['seeds'])]
    mechanism_heldout = [g for g in mechanism_grouped if g['year'] in (2022, 2023)]
    result = dict(
        method='severity_factorized_block_specialist',
        rows=rows, grouped=grouped, mechanism_grouped=mechanism_grouped,
        mean_primary_delta=mean(r['primary_delta'] for r in rows),
        std_primary_delta=pstdev(r['primary_delta'] for r in rows),
        mean_block_delta=mean(r['block_delta'] for r in rows),
        worst_clean_delta=min(r['delta']['M00'] for r in rows),
        magnitude_target_met=mean(r['primary_delta'] for r in rows) >= .02,
        screen_pass=(len(screen) == 2 and {r['history'] for r in screen} == {1, 5}
            and all(r['primary_delta'] >= .005 and r['delta']['M00'] >= -.01
                    for r in screen)),
        mechanism_screen_pass=(len(screen) == 2
            and {r['history'] for r in screen} == {1, 5}
            and all(r['mechanism_block_delta'] > 0 for r in screen)),
        confirmation_pass=(len(confirmation) == 2
            and {g['history'] for g in confirmation} == {1, 5}
            and all(g['primary_delta'] > 0 and g['clean_delta'] >= -.01
                    for g in confirmation)),
        mechanism_confirmation_pass=(len(mechanism_confirmation) == 2
            and {g['history'] for g in mechanism_confirmation} == {1, 5}
            and all(g['block_delta'] > 0 for g in mechanism_confirmation)),
        mechanism_heldout_pass=(len(mechanism_heldout) == 4
            and {(g['history'], g['year']) for g in mechanism_heldout}
                == {(1, 2022), (1, 2023), (5, 2022), (5, 2023)}
            and all(g['block_delta'] > 0 for g in mechanism_heldout)),
        heldout_pass=(len(heldout) == 4
            and {(g['history'], g['year']) for g in heldout}
                == {(1, 2022), (1, 2023), (5, 2022), (5, 2023)}
            and all(g['primary_delta'] > 0 and g['clean_delta'] >= -.01
                    for g in heldout)),
    )
    result['goal_evidence_pass'] = (result['confirmation_pass']
        and result['heldout_pass'] and result['mechanism_confirmation_pass']
        and result['mechanism_heldout_pass'])
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n')


if __name__ == '__main__':
    main()
