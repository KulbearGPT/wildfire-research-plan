import importlib
import importlib.util
import unittest


class AssessmentTests(unittest.TestCase):
    def assess(self, rows):
        name = 'reproductions.cross_history.assess_three_directions'
        self.assertIsNotNone(importlib.util.find_spec(name), 'assessment module missing')
        return importlib.import_module(name).assess(rows)

    def rows(self):
        values = dict(bn_original=.5, bn_common=.5, bn_conditional=.506,
                      control=.5, mixed=.501, typed=.51, teacher=.514, distill=.513)
        return [dict(history=h, seed=0, year=2021, method='three_'+arm,
                     results={s:dict(avg_precision=value, sample_count=3181)
                              for s in ('M00','M01','M06','M07')})
                for h in (1,5) for arm,value in values.items()]

    def test_requires_both_histories_and_attribution_control(self):
        rows = self.rows()
        self.assertTrue(all(x['screen_pass'] for x in self.assess(rows).values()))
        incomplete = self.assess([r for r in rows if r['history']==1])
        self.assertTrue(all(x['status']=='incomplete' for x in incomplete.values()))
        for row in rows:
            if row['method']=='three_mixed':
                for metric in row['results'].values():metric['avg_precision']=.509
        self.assertFalse(self.assess(rows)['fusion']['screen_pass'])

    def test_distillation_cannot_pass_by_only_matching_teacher(self):
        rows = self.rows()
        for row in rows:
            if row['method']=='three_control':
                for metric in row['results'].values():metric['avg_precision']=.515
        self.assertFalse(self.assess(rows)['distillation']['screen_pass'])

    def test_rejects_invalid_or_duplicate_evidence(self):
        rows = self.rows()
        with self.assertRaises(ValueError):self.assess(rows+[rows[0]])
        for field,bad in [('avg_precision',float('nan')),('sample_count',42)]:
            rows = self.rows(); rows[0]['results']['M00'][field]=bad
            with self.assertRaises(ValueError):self.assess(rows)


if __name__=='__main__':unittest.main()
