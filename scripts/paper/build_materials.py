#!/usr/bin/env python3
# Code lifecycle: active_support. Paper evidence aggregation and figure packaging.
# Scope and settings: paper/materials/README.md; docs/CODE_LIFECYCLE.md.
"""Build an author-facing paper bundle from immutable historical JSON evidence."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import socket
from statistics import mean, stdev
import subprocess
import sys
import zipfile

SCENARIOS = ('M00', 'M01', 'M06', 'M07')
METRICS = ('primary', 'block', 'clean', 'fire', 'mild', 'severe')
MAIN = ('ERM', 'X22', 'X14_ROUTE', 'X17_ROUTE', 'X22_X14_ERM_CLEAN',
        'X22_X17_ERM_CLEAN', 'X22_X17_HISTORICAL')
LABELS = {'ERM': 'Fresh ERM', 'X22': 'Cosine ERM (X22)',
          'X14_ROUTE': 'Mixed block route (X14)', 'X17_ROUTE': 'Fixed block route (X17)',
          'X22_X14_ERM_CLEAN': 'X22 + mixed X14',
          'X22_X17_ERM_CLEAN': 'X22 + X17 (ERM clean)',
          'X22_X17_HISTORICAL': 'X22 + X17 (X22 clean)'}


def write_csv(path, rows, fields=None):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def check_index(rows):
    index = {}
    for row in rows:
        key = tuple(row[k] for k in ('method', 'history', 'seed', 'year', 'scenario'))
        if key in index:
            raise ValueError(f'Duplicate normalized evidence cell: {key}')
        if row['scenario'] not in SCENARIOS or not math.isfinite(row['ap']) or not 0 <= row['ap'] <= 1:
            raise ValueError(f'Invalid metric cell: {key}')
        index[key] = row
    return index


def add_routes(rows):
    index = check_index(rows)
    routes = {
        'X22_X14_ERM_CLEAN': ('ERM', 'X22', 'X14_RAW', 'X14_RAW'),
        'X22_X17_ERM_CLEAN': ('ERM', 'X22', 'X17_MILD', 'X17_SEVERE'),
    }
    for route, roles in routes.items():
        for h in (1, 5):
            for seed in (0, 1, 2):
                for year in (2021, 2022, 2023):
                    for scenario, role in zip(SCENARIOS, roles):
                        original = index[(role, h, seed, year, scenario)]
                        rows.append(dict(original, method=route, component_method=role,
                            evidence_level='new_summary_composition_of_existing_evaluations',
                            provenance_notes=[*original.get('provenance_notes', []),
                                'Derived without retraining or executing an inference router.']))
    index = check_index(rows)
    expected = {(h, s, y, scenario) for h in (1, 5) for s in (0, 1, 2)
                for y in (2021, 2022, 2023) for scenario in SCENARIOS}
    for method in MAIN:
        actual = {key[1:] for key in index if key[0] == method}
        if actual != expected:
            raise ValueError(f'Incomplete main comparison for {method}: {expected-actual}')
    for h, seed, year, scenario in expected:
        now = index[('X22_X17_ERM_CLEAN', h, seed, year, scenario)]['ap']
        historical = index[('X22_X17_HISTORICAL', h, seed, year, scenario)]['ap']
        expected_old = index[('X22', h, seed, year, scenario)]['ap'] if scenario == 'M00' else now
        if not math.isclose(historical, expected_old, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f'Historical route definition differs at {(h,seed,year,scenario)}')
    return rows, index


def metrics_from(values):
    return dict(primary=mean(values[s] for s in ('M01','M06','M07')),
                block=mean(values[s] for s in ('M06','M07')),
                clean=values['M00'], fire=values['M01'], mild=values['M06'], severe=values['M07'])


def seed_metrics(rows, index):
    cells = defaultdict(dict)
    for row in rows:
        if row['seed'] is not None:
            cells[tuple(row[k] for k in ('method','history','seed','year'))][row['scenario']] = row['ap']
    out = []
    for (method, h, seed, year), values in sorted(cells.items()):
        if set(values) != set(SCENARIOS):
            raise ValueError(f'Incomplete four-scenario result: {(method,h,seed,year)}')
        record = dict(method=method, history=h, seed=seed, year=year, **metrics_from(values))
        # Do not attach the canonical ERM comparator to different historical protocols.
        if not method.startswith(('HIST_', 'RF_', 'TD_')):
            control = metrics_from({s: index[('ERM',h,seed,year,s)]['ap'] for s in SCENARIOS})
            record.update({k+'_delta': record[k]-control[k] for k in METRICS})
        out.append(record)
    return out


def summarize(records):
    groups = defaultdict(list)
    for r in records:
        groups[(r['method'],r['history'],r['year'])].append(r)
    tables, effects = [], []
    for (method, h, year), group in sorted(groups.items()):
        seeds = sorted(r['seed'] for r in group)
        for metric in METRICS:
            values = [r[metric] for r in group]
            record = dict(method=method, history=h, year=year, metric=metric,
                          mean=mean(values), sd=stdev(values) if len(values)>1 else None,
                          n=len(values), seeds=seeds,
                          split_role='selection' if year==2021 else 'historical_test')
            tables.append(record)
            if all(metric+'_delta' in r for r in group):
                delta = [r[metric+'_delta'] for r in group]
                effects.append(dict(method=method, history=h, year=year, metric=metric,
                    mean_delta=mean(delta), sd_delta=stdev(delta) if len(delta)>1 else None,
                    n=len(delta), seeds=seeds, comparator='matched_fresh_ERM',
                    split_role=record['split_role']))
    return tables, effects


def latex_escape(value):
    return str(value).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('%',r'\%').replace('&',r'\&').replace('#',r'\#')


def latex_tables(root, table, effects):
    by = {(r['method'],r['history'],r['year'],r['metric']):r for r in table}
    out = [r'\begin{tabular}{lrrrrrr}', r'\toprule',
           r'Method & T1/2021 & T1/2022 & T1/2023 & T5/2021 & T5/2022 & T5/2023 \\', r'\midrule']
    for method in MAIN:
        values = []
        for h in (1, 5):
            for year in (2021,2022,2023):
                r = by[method,h,year,'primary']
                values.append(f"${100*r['mean']:.2f}\\pm{100*r['sd']:.2f}$")
        out.append(latex_escape(LABELS[method])+' & '+' & '.join(values)+r' \\')
    out += [r'\bottomrule',r'\end{tabular}']
    (root/'tables/main_results.tex').write_text('\n'.join(out)+'\n')
    table_md=['# Main results','', 'Primary AP ×100, mean ± sample SD over three seeds. 2021 is selection data; 2022/2023 are historical test years.','',
              '| Method | T1/2021 | T1/2022 | T1/2023 | T5/2021 | T5/2022 | T5/2023 |',
              '|---|---|---|---|---|---|---|']
    for method in MAIN:
        values=[f"{100*by[method,h,y,'primary']['mean']:.2f} ± {100*by[method,h,y,'primary']['sd']:.2f}" for h in (1,5) for y in (2021,2022,2023)]
        table_md.append('| '+LABELS[method]+' | '+' | '.join(values)+' |')
    (root/'tables/main_results.md').write_text('\n'.join(table_md)+'\n')
    lines=[r'\begin{tabular}{llrrr}',r'\toprule',r'Setting & Year & Primary & Block & Clean \\',r'\midrule']
    for h in (1,5):
        for year in (2021,2022,2023):
            cells=[]
            for metric in ('primary','block','clean'):
                r=next(x for x in effects if x['method']=='X22_X17_ERM_CLEAN' and x['history']==h and x['year']==year and x['metric']==metric)
                cells.append(f"${100*r['mean_delta']:+.2f}\\pm{100*r['sd_delta']:.2f}$")
            lines.append(f'T{h} & {year} & '+' & '.join(cells)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (root/'tables/paired_effects.tex').write_text('\n'.join(lines)+'\n')


def archive_tables(evidence, root):
    catalogue=[]
    for item in evidence['inventory']:
        state=item.get('lifecycle',{})
        catalogue.append(dict(id=item['id'],name=item.get('name'),family=item.get('family'),
            maintenance_state=state.get('state'),evidence_status=item.get('status'),
            evidence=item.get('evidence'),comparator=item.get('comparator'),
            setting=item.get('evidence_scope'),source_document=item.get('source_document'),
            recipe=state.get('recipe_ref'),reason=state.get('reason')))
    write_csv(root/'tables/supplementary_catalogue.csv',catalogue)
    comparisons=[]
    for artifact in evidence['archived_comparisons']:
        payload=artifact['payload']
        for row in payload.get('rows',[]):
            if not isinstance(row,dict):continue
            comparisons.append(dict(artifact=artifact['artifact'],source_id=artifact['source_id'],
                method=payload.get('method'),history=row.get('history'),seed=row.get('seed'),year=row.get('year'),
                primary_delta=row.get('primary_delta'),block_delta=row.get('block_delta'),
                clean_delta=row.get('delta',{}).get('M00'),mechanism_block_delta=row.get('mechanism_block_delta'),
                comparator='See original source payload and linked experiment ledger',raw_row=row))
    write_csv(root/'tables/archive_comparison_rows.csv',comparisons)


def build(args):
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    (root/'data').mkdir(); (root/'tables').mkdir(); (root/'code').mkdir()
    evidence=json.loads((args.evidence/'evidence.json').read_text())
    shutil.copytree(args.evidence/'sources',root/'sources')
    shutil.copy2(args.evidence/'evidence.json',root/'data/evidence.json')
    for source in evidence['sources']:
        content=(root/source['source_rel']).read_bytes()
        if hashlib.sha256(content).hexdigest()!=source['sha256']:
            raise ValueError(f"Evidence hash mismatch: {source['source_rel']}")
    rows,index=add_routes([dict(r) for r in evidence['rows']])
    write_csv(root/'data/scenario_rows.csv',rows)
    metrics=seed_metrics(rows,index)
    write_csv(root/'data/metrics_by_seed.csv',metrics)
    tables,effects=summarize(metrics)
    write_csv(root/'tables/main_results.csv',[r for r in tables if r['method'] in MAIN])
    write_csv(root/'tables/all_normalized_results.csv',tables)
    write_csv(root/'tables/paired_effects.csv',effects)
    attribution=[]
    metric_index={(r['method'],r['history'],r['seed'],r['year']):r for r in metrics}
    for h in (1,5):
        for seed in (0,1,2):
            for year in (2021,2022,2023):
                a,b=(metric_index[m,h,seed,year] for m in ('X17_ROUTE','X14_ROUTE'))
                attribution.append(dict(history=h,seed=seed,year=year,block_delta=a['block']-b['block'],
                                        primary_delta=a['primary']-b['primary']))
    write_csv(root/'tables/attribution.csv',attribution)
    historical=[]
    for comparison,method,control in [('D1_KL_vs_ERM','HIST_D1_KL','HIST_D1_ERM'),
                                      ('D12_vs_D2_STD','HIST_D12_SARP','HIST_D2_STD'),
                                      ('D12_vs_ERM','HIST_D12_SARP','HIST_D1_ERM')]:
        for year in (2021,2022,2023):
            a,b=(metric_index[m,1,0,year] for m in (method,control))
            for metric in ('primary','block','clean'):
                historical.append(dict(comparison=comparison,year=year,metric=metric,
                    delta=a[metric]-b[metric],history=1,seed=0,method=method,comparator=control,
                    evidence_scope='Historical single-run T1 chain; not mainline confirmation'))
    write_csv(root/'tables/historical_chain_effects.csv',historical)
    costs=[]
    for method,count in zip(MAIN,(1,1,2,3,3,4,3)):
        values=[r for r in metrics if r['method']==method]
        costs.append(dict(method=method,models=count,continuation_steps=3000*count,
            primary_delta_mean=mean(r['primary_delta'] for r in values),
            cost_basis='Analytical recipe count per history/seed; excludes shared foundation and research controls; not measured runtime'))
    write_csv(root/'tables/costs.csv',costs)
    overall=[]
    for method in MAIN:
        for label,years in [('all_years',(2021,2022,2023)),('selection',(2021,)),('historical_test',(2022,2023))]:
            selected=[r for r in metrics if r['method']==method and r['year'] in years]
            overall.append(dict(method=method,scope=label,n_setting_seed_year_rows=len(selected),
                                **{key:mean(r[key] for r in selected) for key in (*METRICS,*(m+'_delta' for m in METRICS))}))
    write_csv(root/'tables/overall_effects.csv',overall)
    derived=[]
    for h in (1,5):
        for year in (2021,2022,2023):
            for metric in METRICS:
                differences=[metric_index['X22_X17_ERM_CLEAN',h,s,year][metric]-metric_index['X22_X14_ERM_CLEAN',h,s,year][metric]
                             for s in (0,1,2)]
                derived.append(dict(history=h,year=year,metric=metric,mean_delta=mean(differences),
                    sd_delta=stdev(differences),n=3,comparison='X22_X17_ERM_CLEAN minus X22_X14_ERM_CLEAN'))
    write_csv(root/'tables/system_attribution.csv',derived)
    # Verify original published rounded values from independent normalized rows.
    known={'X22':.012009,'X17_ROUTE':.008250,'X22_X17_ERM_CLEAN':.014642}
    for method,target in known.items():
        actual=next(r['primary_delta'] for r in overall if r['method']==method and r['scope']=='all_years')
        if abs(actual-target)>5e-7:raise ValueError(f'Ledger reconciliation failed: {method} {actual}')
    frozen=[]
    for h in (1,5):
        for year in (2021,2022,2023):
            keys=[('FROZEN',h,None,year,s) for s in SCENARIOS]
            if not all(k in index for k in keys):continue
            baseline=metrics_from({s:index[k]['ap'] for k,s in zip(keys,SCENARIOS)})
            for method in MAIN:
                for seed in (0,1,2):
                    r=metric_index[method,h,seed,year]
                    frozen.append(dict(method=method,history=h,seed=seed,year=year,
                        **{key+'_delta':r[key]-baseline[key] for key in METRICS},
                        comparator='one frozen B3/B5 checkpoint per setting; same reference reused across seeds'))
    write_csv(root/'tables/frozen_reference.csv',frozen)
    if len(frozen)!=126:raise ValueError('All six frozen reference cells are required')
    frozen_gain=mean(r['primary_delta'] for r in frozen if r['method']=='X22_X17_ERM_CLEAN')
    if abs(frozen_gain-.034527)>5e-7:raise ValueError(f'Frozen-reference reconciliation failed: {frozen_gain}')
    archive_tables(evidence,root)
    latex_tables(root,tables,effects)
    for path in (args.repo/'paper/materials').glob('*.md'):shutil.copy2(path,root/path.name)
    for path in (args.repo/'scripts/paper').glob('*.py'):shutil.copy2(path,root/'code'/path.name)
    repository_sources=[]
    for name in ('docs/CODE_LIFECYCLE.md','docs/experiments/t1_t5_innovations.md',
                 'docs/experiments/quantitative_reliability_ledger.md',
                 'docs/experiments/rejected_experiments.md',
                 'docs/experiments/three_directions_results.md','docs/experiments/three-directions-results.md',
                 'docs/research/evaluation-population.md','docs/research/positive-signals.md',
                 'docs/research/negative-results.md','docs/research/method-inventory.json',
                 'reproductions/wsts_fast_track/evaluate_missingness.py',
                 'reproductions/wsts_fast_track/missingness.py','reproductions/cross_history/run.py',
                 'reproductions/cross_history/data.py','reproductions/cross_history/compose_complete_routes.py'):
        source=args.repo/name
        dest=root/'repository'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
        repository_sources.append(dict(repository_path=name,package_path=dest.relative_to(root).as_posix(),
                                       sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),bytes=dest.stat().st_size))
    write_csv(root/'data/repository_source_manifest.csv',repository_sources)
    write_csv(root/'data/source_manifest.csv',evidence['sources'])
    (root/'data/gaps.json').write_text(json.dumps(evidence['gaps'],indent=2)+'\n')
    from render_figures import render_all
    figures=render_all(root)
    build_preview(root,figures,overall,evidence)
    commit=os.environ.get('WILDFIRE_SOURCE_COMMIT') or subprocess.check_output(['git','-C',str(args.repo),'rev-parse','HEAD'],text=True).strip()
    import matplotlib
    import numpy
    result=dict(schema_version=1,source_commit=commit,evidence_source_commit=evidence['source_commit'],
                job_id=os.environ['SLURM_JOB_ID'],node=socket.gethostname(),
                python=sys.version,matplotlib_version=matplotlib.__version__,numpy_version=numpy.__version__,
                scientific_scope='Existing evaluations only; no retraining or inference',
                normalized_scenario_rows=len(rows),catalogue_methods=len(evidence['inventory']),
                original_source_records=len(evidence['sources']),figures=len(figures),
                mainline_seed_coverage='2 histories x 3 seeds x 3 years x 4 scenarios for each main method',
                standard_deviation='Sample SD (ddof=1) over three seeds within each setting/year',
                route_discrepancy='Historical M00=X22; derived current-route M00=ERM. Preserved separately.',
                anonymous_submission_ready=False,checks=['source hashes','unique cells','full mainline grid','historical route identity','ledger primary gains','frozen reference gain'])
    (root/'BUILD_REPORT.json').write_text(json.dumps(result,indent=2)+'\n')
    (root/'REPRODUCE.md').write_text('''# Regenerate these materials

This author bundle contains the source JSONs, full-precision CSVs and plotting code.
No weights or training dataset are required to regenerate the figures.
Use Python 3.10+ with matplotlib 3.7.2 and numpy 1.26.4 (the verified build environment).
Within a Slurm CPU allocation, run `python code/render_figures.py .` from the
bundle directory to regenerate all figures. The original evidence collector and
full builder are included for future repository updates. Their CLI help documents
the source/output arguments. Source paths in the manifests are historical identities;
the `source_rel` files are self-contained in this bundle. Repository-relative paths cited in the prose are preserved beneath `repository/`.
Scenario AP is in [0,1]; paper plots/tables display AP x100 or differences in AP points. Seed standard
deviation is not a confidence interval and does not establish statistical significance.
The HTML and PDF previews are author material, not an anonymized submission.
''')
    manifest=[]
    for path in sorted(root.rglob('*')):
        if path.is_file():manifest.append(dict(path=path.relative_to(root).as_posix(),bytes=path.stat().st_size,
                                               sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (root/'checksums.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=root.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(root.rglob('*')):
            if path.is_file():z.write(path,arcname=str(Path(root.name)/path.relative_to(root)))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC validation failed')
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix('.zip.sha256').write_text(f'{digest}  {archive.name}\n')
    print(json.dumps(dict(result,archive=str(archive),archive_bytes=archive.stat().st_size,sha256=digest),indent=2))


def build_preview(root,figures,overall,evidence):
    # Compile the review PDF from the vector figures, preserving text and lines.
    text = [r'\documentclass[10pt,letterpaper]{article}',
            r'\usepackage[margin=0.65in]{geometry}',
            r'\usepackage{graphicx,booktabs,mathptmx,hyperref}',
            r'\hypersetup{colorlinks=true,urlcolor=blue}',
            r'\setlength{\parindent}{0pt}', r'\setlength{\parskip}{7pt}',
            r'\begin{document}', r'{\LARGE Wildfire forecasting: paper materials}\par',
            r'{\large X22 + X17 and necessary controls}\par',
            'Author working package. Existing results and explicit compositions; no new training or model evaluation.',
            r'\section*{Evidence and reporting}',
            'Primary is mean AP over M01/M06/M07. Tables display AP $\\times100$; plots show AP points. '
            'Variability is sample SD over three continuation seeds within each setting/year, not a confidence interval.',
            '2021 is validation and selection data. 2022/2023 are historical test years already inspected during research.',
            'The original complete-route artifact uses X22 for M00. The current route uses ERM. '
            'The two variants share missing-scenario scores, but differ in clean AP and retained model count.',
            f"Recovered {len(evidence['sources'])} source records and retained all {len(evidence['inventory'])} method inventory records.",
            r'\section*{Main results}', r'\begin{center}\footnotesize\input{tables/main_results.tex}\end{center}',
            'Mean $\\pm$ sample SD over three seeds. T1 and T5 differ in architecture and features as well as history. '
            'The two X22+X17 rows have identical primary values by construction.',
            r'\section*{Interpretation}',
            'The mainline supports a system result. Cosine decay is an established schedule; fixed-severity factorization '
            'has failed closest-control cells. Use the mixed X14 control and preserve those failures.',
            'Analytical continuation counts exclude shared foundation training and evaluation. '
            'This package does not measure inference latency or certify operational robustness.',
            r'\section*{Included material}',
            'Full-precision CSVs, source JSONs and checksums, vector figures, LaTeX tables, English experimental prose, '
            'the complete archive catalogue, and regeneration scripts are included. Local provenance makes this an '
            'author copy; prepare a separate anonymized submission copy.']
    for item in figures:
        name=item.get('stem') or item.get('name') or item.get('id')
        text += [r'\clearpage', r'\begin{center}',
                 r'\includegraphics[width=\linewidth,height=0.72\textheight,keepaspectratio]{figures/'+name+'.pdf}',
                 r'\end{center}', latex_escape(item.get('caption',''))]
    text += [r'\end{document}']
    (root/'report.tex').write_text('\n'.join(text)+'\n')
    log=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','report.tex'],
                       cwd=root,capture_output=True,text=True)
    (root/'report-build.log').write_text(log.stdout+log.stderr)
    if log.returncode:
        raise RuntimeError('PDF review compilation failed; inspect report-build.log')
    for extension in ('aux','out','log'):
        (root/f'report.{extension}').unlink(missing_ok=True)
    body=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Wildfire paper materials</title>',
          '<style>body{max-width:1100px;margin:45px auto;font:16px/1.6 Georgia,serif;color:#17232c;padding:0 24px}img{width:100%;height:auto}article{border-top:1px solid #ddd;margin-top:35px;padding-top:20px}a{color:#006a92}code{font-size:14px}</style>',
          '<h1>Wildfire forecasting: paper materials</h1><p>Author working package. Existing evaluation evidence, explicit compositions, and preserved archived results.</p>',
          '<p><a href="report.pdf">PDF overview</a> · <a href="README.md">Readme</a> · <a href="tables/main_results.md">Main table</a> · <a href="CLAIMS_AND_LIMITATIONS.md">Claims and limits</a></p>',
          '<p><strong>Source correction:</strong> the historical complete route uses X22 for M00; the current mainline uses ERM. These are separate variants. Primary AP on missing scenarios is identical.</p>']
    for item in figures:
        name=item.get('stem') or item.get('name') or item.get('id')
        body.append(f'<article><img src="figures/{name}.svg" alt="{html.escape(name)}"><p>{html.escape(item.get("caption",""))}</p><a href="figures/{name}.pdf">Vector PDF</a> · <a href="figures/{name}.svg">Editable SVG</a> · <a href="figures/{name}.png">PNG preview</a></article>')
    body.append('</html>')
    (root/'index.html').write_text('\n'.join(body))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Slurm CPU allocation required for aggregation, figures and ZIP')
    build(args)


if __name__=='__main__':main()
