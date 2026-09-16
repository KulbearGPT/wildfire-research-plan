import copy
import math
import unittest
from reproductions.three_directions.report import assess

SCENES=('M00','M01','M06','M07')

def rows():
    result=[]
    for history in (1,5):
        for mode,value in [('bn_shared',.506),('bn_conditional',.513),('merge',.513),
                           ('x14_shared',.51),('student_control',.51),('student_distill',.514)]:
            result.append(dict(history=history,seed=0,year=2021,mode=mode,smoke=False,
                results={s:dict(avg_precision=value,sample_count=3181) for s in SCENES},
                original_x22={s:dict(avg_precision=.5,sample_count=3181) for s in SCENES},
                route_reference={s:dict(avg_precision=.515,sample_count=3181) for s in SCENES}))
    return result

class ReportTests(unittest.TestCase):
    def test_three_paired_passes(self):
        output=assess(rows())
        self.assertTrue(all(output[d]['screen_pass'] for d in ('N','W','D')))

    def test_one_history_is_not_complete(self):
        output=assess([r for r in rows() if r['history']==1])
        self.assertTrue(all(output[d]['status']=='incomplete' for d in ('N','W','D')))

    def test_distillation_must_beat_no_kl_student(self):
        data=rows()
        for r in data:
            if r['mode']=='student_control':
                for metric in r['results'].values():metric['avg_precision']=.52
        self.assertFalse(assess(data)['D']['screen_pass'])

    def test_scene_drop_cannot_hide_in_mean(self):
        data=rows()
        for r in data:
            if r['mode']=='student_distill':r['results']['M00']['avg_precision']=.49
        self.assertFalse(assess(data)['D']['screen_pass'])

    def test_rejects_nan_smoke_and_wrong_population(self):
        for field,value in [('nan',float('nan')),('smoke',True),('population',42)]:
            data=rows()
            if field=='nan':data[0]['results']['M01']['avg_precision']=value
            elif field=='smoke':data[0]['smoke']=value
            else:data[0]['results']['M01']['sample_count']=value
            with self.assertRaises(ValueError):assess(data)

    def test_distillation_reports_measured_storage_and_training_cost(self):
        data=rows()
        for r in data:
            r.update(checkpoint_bytes=128,route_checkpoint_bytes=512,
                     parameter_bytes=100,training_seconds=120,updates=3000)
        cell=assess(data)['D']['cells'][0]
        self.assertEqual(cell['candidate_cost']['checkpoint_to_route_ratio'],.25)
        self.assertEqual(cell['candidate_cost']['training_seconds'],120)
        self.assertEqual(cell['control_cost']['updates'],3000)

    def test_rejects_duplicate_runs(self):
        data=rows()
        with self.assertRaises(ValueError):assess(data+[copy.deepcopy(data[0])])

if __name__=='__main__':unittest.main()
