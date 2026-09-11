"""Lightweight tests; no dataset, model, network, or GPU required."""
import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('tutorial_results', Path(__file__).with_name('results.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class ResultsTests(unittest.TestCase):
    def test_reads_lightning_table(self):
        table = '\n'.join(f'│ test_{key} │ {value} │' for key, value in
                          [('AP', .5), ('f1', .4), ('iou', .3), ('precision', .6), ('recall', .2), ('loss', .01)])
        self.assertEqual(module.parse_metrics(table)['AP'], .5)
    def test_missing_or_nonfinite_metric_fails(self):
        for text in ('test_AP 0.5', 'test_AP nan', 'test_AP 0.5\ntest_AP nan'):
            with self.assertRaises(ValueError):
                module.parse_metrics(text)
    def test_nonfinite_final_table_does_not_reuse_an_older_score(self):
        table = '\n'.join(f'test_{key} 0.5' for key in ('AP', 'f1', 'iou', 'precision', 'recall', 'loss'))
        with self.assertRaises(ValueError):
            module.parse_metrics(table + '\ntest_AP nan')
    def test_complete_fold_aggregation(self):
        rows = [{'mode': 'weight', 'fold': i, 'metrics': {'AP': i / 12}} for i in range(12)]
        result = module.aggregate(rows, 'weight')
        self.assertAlmostEqual(result['mean_AP'], 5.5 / 12)
        self.assertAlmostEqual(result['population_std_AP'], math.sqrt(143 / 12) / 12)
    def test_missing_duplicate_or_mixed_folds_fail(self):
        rows = [{'mode': 'weight', 'fold': i, 'metrics': {'AP': .5}} for i in range(12)]
        for bad in (rows[:-1], rows + [rows[0]], [dict(r, mode='full') if i == 0 else r for i, r in enumerate(rows)]):
            with self.assertRaises(ValueError):
                module.aggregate(bad, 'weight')
    def test_out_of_range_ap_fails(self):
        rows = [{'mode': 'weight', 'fold': i, 'metrics': {'AP': .5}} for i in range(12)]
        rows[0]['metrics']['AP'] = 1.1
        with self.assertRaises(ValueError):
            module.aggregate(rows, 'weight')

if __name__ == '__main__':
    unittest.main()
