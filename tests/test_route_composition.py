"""Small synthetic metric checks; run on a CPU Slurm allocation."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproductions.cross_history import compose_complete_routes, compose_severity_routes


MODULES = (compose_complete_routes, compose_severity_routes)
SCENARIOS = ('M00', 'M01', 'M06', 'M07')


def group(history=1, seed=0, year=2021):
    roles = {'control': ('control', None, .2), 'fire': ('cosine_erm', None, .23),
             'mixed': ('block_specialist', None, .21),
             'mild': ('block_specialist', .25, .24),
             'severe': ('block_specialist', .5, .25)}
    count = {2021: 3181, 2022: 2856, 2023: 2102}[year]
    return {role: dict(history=history, seed=seed, year=year,
                architecture={1: 'res18_unet', 5: 'res18_utae'}[history],
                method=method, block_fraction=fraction,
                results={s: dict(avg_precision=ap, sample_count=count,
                                 pixel_count=count * 64 * 64) for s in SCENARIOS})
            for role, (method, fraction, ap) in roles.items()}


def compose(module, groups, *, mainline=True, existing_output=False):
    roles = ('control', 'fire', 'mild', 'severe') if module is MODULES[0] else (
        'control', 'mixed', 'mild', 'severe')
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        argv = ['compose', *(['--mainline'] if mainline else [])]
        for index, row in enumerate(groups):
            for role in roles:
                path = root / f'{index}-{role}.json'
                path.write_text(json.dumps(row[role]))
                argv += ['--' + role, str(path)]
        output = root / 'report.json'
        if existing_output:
            output.write_text('preserved result')
        argv += ['--output', str(output)]
        try:
            with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()):
                module.main()
        finally:
            if existing_output:
                assert output.read_text() == 'preserved result'
        return json.loads(output.read_text())


class CompositionTests(unittest.TestCase):
    def test_complete_matrix_preserves_routing_and_gates(self):
        rows = [group(h, s, y) for h in (1, 5) for s in (0, 1, 2)
                for y in (2021, 2022, 2023)]
        for module in MODULES:
            with self.subTest(module=module.__name__):
                report = compose(module, rows)
                self.assertTrue(report['goal_evidence_pass'])
                self.assertEqual(len(report['rows']), 18)
                self.assertEqual(report['summary_contract'], 'mainline')
                self.assertFalse(report['training_provenance_verified'])
                self.assertAlmostEqual(report['mean_block_delta'], .045)
                expected = .04 if module is MODULES[0] else .03
                self.assertAlmostEqual(report['mean_primary_delta'], expected)
                self.assertEqual(report['worst_clean_delta'], 0)

    def test_incomplete_historical_test_seeds_cannot_pass(self):
        rows = [group(h, s, 2021) for h in (1, 5) for s in (0, 1, 2)]
        rows += [group(h, 0, y) for h in (1, 5) for y in (2022, 2023)]
        for module in MODULES:
            for mainline in (False, True):
                with self.subTest(module=module.__name__, mainline=mainline):
                    report = compose(module, rows, mainline=mainline)
                    self.assertTrue(report['confirmation_pass'])
                    self.assertFalse(report['heldout_pass'])
                    self.assertFalse(report['goal_evidence_pass'])
                    if module is MODULES[1]:
                        self.assertFalse(report['mechanism_heldout_pass'])

    def test_duplicate_cells_rejected_in_both_modes(self):
        for module in MODULES:
            for mainline in (False, True):
                with self.subTest(module=module.__name__, mainline=mainline):
                    with self.assertRaisesRegex(ValueError, 'duplicate'):
                        compose(module, [group(), group()], mainline=mainline)

    def test_bad_role_and_protocol_fields_rejected(self):
        for module in MODULES:
            for field, value in [('method', 'cosine_erm'), ('block_fraction', .25),
                                 ('architecture', 'swin_unet'), ('architecture', None),
                                 ('seed', 1), ('history', True)]:
                rows = [group()]
                rows[0]['control'][field] = value
                with self.subTest(module=module.__name__, field=field, value=value):
                    with self.assertRaises(ValueError):
                        compose(module, rows)
        rows = [group()]
        rows[0]['fire']['method'] = 'fire_specialist'
        with self.assertRaisesRegex(ValueError, 'cosine_erm'):
            compose(MODULES[0], rows)
        # Historical X8+X17 remains available via the broad composer.
        report = compose(MODULES[0], rows, mainline=False)
        self.assertEqual(report['component_methods']['fire'], 'fire_specialist')

    def test_invalid_metrics_and_populations_rejected(self):
        changes = [('avg_precision', float('nan')), ('avg_precision', float('inf')),
                   ('avg_precision', -.1), ('avg_precision', 1.1),
                   ('avg_precision', True), ('sample_count', 2312),
                   ('sample_count', None), ('pixel_count', 1), ('pixel_count', 0)]
        for module in MODULES:
            for key, value in changes:
                rows = [group()]
                rows[0]['mild']['results']['M06'][key] = value
                with self.subTest(module=module.__name__, field=key, value=value):
                    with self.assertRaises(ValueError):
                        compose(module, rows)

    def test_seed_outside_recorded_protocol_rejected(self):
        for module in MODULES:
            with self.assertRaisesRegex(ValueError, 'seed'):
                compose(module, [group(seed=3)])

    def test_archive_summaries_without_population_remain_readable(self):
        rows = [group()]
        for metadata in rows[0].values():
            metadata.pop('architecture')
            for metrics in metadata['results'].values():
                metrics.pop('sample_count')
                metrics.pop('pixel_count')
        for module in MODULES:
            report = compose(module, copy.deepcopy(rows), mainline=False)
            self.assertEqual(report['summary_contract'], 'archive-compatible')
            self.assertFalse(report['goal_evidence_pass'])

    def test_mainline_output_is_not_overwritten(self):
        for module in MODULES:
            with self.assertRaises(FileExistsError):
                compose(module, [group()], existing_output=True)


if __name__ == '__main__':
    unittest.main()
