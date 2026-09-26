"""Synthetic checks for the paper aggregation; execute inside CPU Slurm."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('paper_build',Path(__file__).resolve().parents[1]/'scripts/paper/build_materials.py')
build=importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def rows():
    out=[]
    components={'ERM':(.5,.2,.3,.1),'X22':(.51,.23,.31,.12),
                'X14_RAW':(.45,.1,.32,.13),'X14_ROUTE':(.5,.2,.32,.13),
                'X17_MILD':(.45,.1,.33,.125),'X17_SEVERE':(.42,.05,.30,.14),
                'X17_ROUTE':(.5,.2,.33,.14),'X22_X17_HISTORICAL':(.51,.23,.33,.14)}
    for method,values in components.items():
        for h in (1,5):
            for seed in (0,1,2):
                for year in (2021,2022,2023):
                    for scenario,value in zip(build.SCENARIOS,values):
                        out.append(dict(method=method,history=h,seed=seed,year=year,scenario=scenario,
                                        ap=value+seed*.001,source_role="candidate",provenance_notes=[]))
    return out


class PaperAggregationTests(unittest.TestCase):
    def test_current_and_historical_clean_remain_distinct(self):
        records,index=build.add_routes(rows())
        metrics=build.seed_metrics(records,index)
        current=next(r for r in metrics if r['method']=='X22_X17_ERM_CLEAN')
        historical=next(r for r in metrics if r['method']=='X22_X17_HISTORICAL')
        self.assertAlmostEqual(current['clean_delta'],0)
        self.assertAlmostEqual(historical['clean_delta'],.01)
        self.assertAlmostEqual(current['primary'],historical['primary'])
        self.assertAlmostEqual(current['primary_delta'],(.03+.03+.04)/3)

    def test_mixed_control_is_composed_from_actual_components(self):
        _,index=build.add_routes(rows())
        self.assertAlmostEqual(index['X22_X14_ERM_CLEAN',5,2,2023,'M01']['ap'],.232)
        self.assertAlmostEqual(index['X22_X14_ERM_CLEAN',5,2,2023,'M07']['ap'],.132)
        derived=index['X22_X14_ERM_CLEAN',5,2,2023,'M07']
        self.assertEqual(derived['source_role'],'candidate')
        self.assertEqual(derived['component_method'],'X14_RAW')

    def test_duplicate_or_missing_cells_rejected(self):
        data=rows()
        with self.assertRaisesRegex(ValueError,'Duplicate'):build.add_routes(data+[data[0]])
        with self.assertRaises((ValueError,KeyError)):build.add_routes(data[1:])

    def test_historical_route_cannot_silently_change_clean_definition(self):
        data=rows()
        next(r for r in data if r['method']=='X22_X17_HISTORICAL' and r['scenario']=='M00')['ap']=.5
        with self.assertRaisesRegex(ValueError,'Historical route'):build.add_routes(data)

    def test_mean_ap_and_sample_sd_are_not_pooled_or_population_sd(self):
        records,index=build.add_routes(rows())
        tables,_=build.summarize(build.seed_metrics(records,index))
        item=next(r for r in tables if r['method']=='ERM' and r['history']==1 and r['year']==2021 and r['metric']=='primary')
        self.assertAlmostEqual(item['mean'],.201)
        self.assertAlmostEqual(item['sd'],.001)
        self.assertEqual(item['n'],3)


if __name__=='__main__':unittest.main()
