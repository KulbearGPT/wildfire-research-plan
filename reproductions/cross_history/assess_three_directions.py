"""Compose the preregistered seed-zero screens; no model/data dependencies."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean

SCENARIOS = ('M00', 'M01', 'M06', 'M07')


def difference(candidate, control):
    delta = {s: candidate[s] - control[s] for s in SCENARIOS}
    return dict(scenarios=delta, primary=mean(delta[s] for s in SCENARIOS[1:]),
                block=mean(delta[s] for s in SCENARIOS[2:]), clean=delta['M00'])


def assess(rows):
    cells = {}
    bn_modes = set()
    for row in rows:
        if row.get('year') != 2021 or row.get('seed') != 0 or row.get('smoke', False):
            raise ValueError('screen requires nonsmoke 2021 seed-zero results')
        history = row['history']
        if history not in (1, 5):
            raise ValueError('history must be 1 or 5')
        expected = 'res18_unet' if history == 1 else 'res18_utae'
        if row.get('architecture', expected) != expected:
            raise ValueError('screen requires canonical architecture')
        key = (history, row['method'].removeprefix('three_'))
        if key[1].startswith('bn_'):
            bn_modes.add(row.get('bn_forward_mode', 'fixed'))
            if len(bn_modes) > 1:
                raise ValueError('cannot mix BN calibration modes')
        if key in cells:
            raise ValueError(f'duplicate screen cell {key}')
        ap = {}
        for scenario in SCENARIOS:
            value = row['results'][scenario]
            if value['sample_count'] != 3181:
                raise ValueError('wrong evaluation population')
            score = value['avg_precision']
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('invalid AP')
            ap[scenario] = score
        cells[key] = ap
    required = dict(normalization=('bn_original', 'bn_common', 'bn_conditional'),
                    fusion=('control', 'mixed', 'typed'),
                    distillation=('control', 'teacher', 'distill'))
    results = {}
    for direction, arms in required.items():
        missing = [(h, arm) for h in (1, 5) for arm in arms if (h, arm) not in cells]
        result = dict(status='incomplete' if missing else 'screen_complete',
                      missing=missing, screen_pass=False, comparisons=[])
        for history in (1, 5):
            if any((history, arm) not in cells for arm in arms):
                continue
            first, second, candidate = [cells[(history, arm)] for arm in arms]
            versus_first = difference(candidate, first)
            versus_second = difference(candidate, second)
            clean_ok = versus_first['clean'] >= -.010
            if direction == 'normalization':
                passed = (versus_second['primary'] > 0 and versus_first['primary'] > 0
                          and clean_ok and versus_second['clean'] >= -.010)
            elif direction == 'fusion':
                passed = (versus_second['primary'] >= .005 and versus_first['primary'] > 0
                          and clean_ok and versus_second['clean'] >= -.010)
            else:
                passed = (versus_second['primary'] >= -.003
                          and versus_first['primary'] > 0 and clean_ok)
            result['comparisons'].append(dict(history=history, candidate=arms[2],
                references=list(arms[:2]), versus_first=versus_first,
                versus_second=versus_second, pass_gate=passed))
        if not missing:
            result['screen_pass'] = all(c['pass_gate'] for c in result['comparisons'])
        results[direction] = result
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('summaries', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.summaries:
        completed = json.loads((path.parent / 'completed.json').read_text())
        started = json.loads((path.parent / 'started.json').read_text())
        if completed.get('status') != 'pass' or started.get('smoke'):
            raise ValueError('only completed nonsmoke jobs supply evidence')
        if started['arm'] == 'bn':
            if sum(completed['bank_samples']) != 4096:
                raise ValueError('incomplete calibration budget')
        elif completed.get('completed_steps') != 3000:
            raise ValueError('incomplete training budget')
        row = json.loads(path.read_text())
        for key in ('history', 'seed', 'architecture'):
            if row[key] != started[key]:
                raise ValueError(f'summary/started {key} mismatch')
        row['source'] = str(path.resolve())
        rows.append(row)
    output = dict(rows=rows, assessment=assess(rows))
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps(output['assessment'], indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
