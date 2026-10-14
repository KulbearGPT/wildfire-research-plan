#!/usr/bin/env python3
"""Generate high-level experiment walkthroughs; no executable recipes are created."""
from html import escape
import json
from course_shell import ROOT, shell, descriptive_names, migrate_links


def slug(ident):
    return ident.lower().replace('+','-plus-')


def fallback(item):
    scenario = item['kind'] == 'scenario'
    if scenario:
        return dict(question=f'What changes under the {item["name"].lower()}?',
          prepare=['Identify the fixed model checkpoint, target dates and evaluation samples.', 'Keep the input arrays, target labels and missingness definition separately traceable.'],
          intervention=[item['explanation'], 'Apply only this input condition; retain the same target and model weights.'],
          comparison=['Use the same evaluator and sample identities across the declared input conditions.', 'Report this condition separately before including it in any predeclared scenario mean.'],
          deliver=['Record the checkpoint, input condition, target population and metric artifact.', 'Explain why this evaluation scenario is not itself a new training method.'])
    if item['id']=='official-fold-2':
        return dict(question='Can you complete the official Fold 2 training and evaluation contract?',prepare=['Pin the official source, environment and original four-year data.', 'Train on 2018/2020, validate on 2019 and test on 2021.'],intervention=['Pass setup, data conversion and the independent smoke gate.', 'Train Res18-U-Net T1 All for 10,000 updates using the official tutorial’s selection rule.'],comparison=['Separate your trained Fold 2 from released-weight evaluation.', 'Do not compare one fold as if it were a twelve-fold aggregate.'],deliver=['Preserve configurations, completed updates, checkpoint and original metrics.', 'Use lesson 4 for the ordered preparation and completion checklist.'])
    if item['id']=='control':
        return dict(question='What changes under ordinary matched ERM continuation?',prepare=['Identify the matching T1/B3 or T5/B5 foundation bytes and training-only statistics.', 'For the first classroom exercise use T1, seed 0 and 2021 validation.'],intervention=['Initialize model weights from the foundation and create a fresh optimizer.', 'Keep shared stochastic FireDrop/BlockDrop and use the constant-rate, 3,000-update, batch-64 continuation; select the final checkpoint.'],comparison=['Evaluate frozen foundation and continuation with the same four conditions and sample identities.', 'Use this control for a later cosine comparison only when the entire contract matches.'],deliver=['Complete the A/B table and retain original metric and job-completion records.', 'Separate the continuation recipe effect from any later schedule effect.'])
    return dict(question=f'How does {item["name"].lower()} anchor its recorded comparison?',prepare=['Read the linked source to identify this historical control’s foundation, data and budget.', 'Keep it distinct from similarly named controls in other experimental families.'],intervention=[item['explanation'], 'Prepare the declared ordinary-training branch without introducing the candidate mechanism.'],comparison=['Match the original candidate’s initialization, inputs, training and evaluation contract.', 'Do not substitute a current continuation score for this historical control.'],deliver=['Provide the control’s configuration, provenance and matched result table.', 'State which mechanism the comparison can and cannot isolate.'])


def main():
    names=json.loads((ROOT/'site-content/en/experiment-names.json').read_text())['entries']
    walkthroughs={x['id']:x for x in json.loads((ROOT/'site-content/en/experiment-walkthroughs.json').read_text())['entries']}
    inventory={x['id']:x for x in json.loads((ROOT/'docs/research/method-inventory.json').read_text())['methods']}
    if not set(inventory).issubset(walkthroughs):raise ValueError('Missing inventory walkthrough')
    for item in names:
        ident=item['id']; plan=walkthroughs.get(ident) or fallback(item); record=inventory.get(ident,{})
        lifecycle=record.get('lifecycle',{}).get('state','scenario' if item['kind']=='scenario' else 'teaching_control')
        status=record.get('status','No new result is claimed here')
        review=any('evidence review' in x or 'not a single training recipe' in x for x in plan['intervention'])
        def listing(lines):return '<ol>'+''.join('<li>'+escape(line)+'</li>' for line in lines)+'</ol>'
        blocks=[('Question',f'<h2>State the question before choosing a result.</h2><p class="lead">{escape(plan["question"])}</p><p>{escape(item["explanation"])}</p><p class="done"><strong>Deliver:</strong> a one-sentence question and an outcome that would contradict it.</p>'),
                ('Prepare','<h2>Gather the exact assets and protocol.</h2>'+listing(plan['prepare'])+'<p class="done"><strong>Deliver:</strong> an asset sheet with data, statistics, checkpoint identities and the comparison contract.</p>'),
                ('Intervention','<h2>Specify what changes, and trace its implementation.</h2>'+listing(plan['intervention'])+'<p class="done"><strong>Deliver:</strong> a diagram of the changed input, model component, objective or sampling rule.</p>')]
        if review:
            execution='<h2>Reconstruct the evidence without inventing a run.</h2><ol><li>Locate the individual records grouped under this entry.</li><li>Separate completed measurements, failures, invalid comparisons and unrun ideas.</li><li>Match every reported value to its original population and artifact.</li></ol><p class="done"><strong>Deliver:</strong> a provenance table, not a fabricated training result.</p>'
        elif item['kind']=='scenario':
            execution='<h2>Evaluate the same fixed prediction contract.</h2><ol><li>Apply the specified input condition to the declared samples.</li><li>Keep weights and inference state fixed; do not train during evaluation.</li><li>Score against the same targets and retain the per-condition output record.</li></ol><p class="done"><strong>Deliver:</strong> a traceable evaluation record for this condition.</p>'
        else:
            execution='<h2>Run the declared plan, then verify the outputs.</h2><ol><li>Finalize the experiment scope and resource budget. For archived methods, distinguish historical reproduction from a newly proposed experiment.</li><li>Use the separately prepared implementation recipe; first smoke-check an unfamiliar loading, update and checkpoint path.</li><li>Start formal candidate and control branches from their declared initializations. Keep their budgets and checkpoint rules fixed.</li><li>Check terminal job status, actual updates and saved artifacts; then evaluate the fixed checkpoints on matched samples and conditions.</li></ol><p class="note">This page is a high-level experiment guide. It does not supply a new runnable script or certify that an archived recipe is ready to execute. Installation, data and model work run through Slurm.</p><p class="done"><strong>Deliver:</strong> actual completion records and original outputs, or a clearly labelled preparation-only plan.</p>'
        if ident in {'M00','M01','M06','M07'}:
            execution += '<p><a href="../../handbook/student-real-data/index.html">Inspect this condition on real wildfire observations</a>. Gray in the display marks hidden input, while the target remains unchanged.</p>'
        blocks.append(('Execute / inspect',execution))
        blocks.append(('Compare','<h2>Use the intended comparator.</h2>'+listing(plan['comparison'])+'<details><summary>Historical scope and evidence</summary><p><strong>Scope:</strong> '+escape(record.get('evidence_scope','Use the linked source contract.'))+'</p><p><strong>Recorded evidence:</strong> '+'<a href="../../'+escape(item['source'],quote=True)+'">Read the original evidence with its setting and comparator</a>. Historical values are not the expected outcome of your classroom run.'+'</p></details><p class="done"><strong>Deliver:</strong> a complete matched table, including regressions and missing results.</p>'))
        blocks.append(('Deliver','<h2>Write what the experiment supports.</h2>'+listing(plan['deliver'])+'<div class="actions"><a class="button" href="worksheet.md" download>Download experiment worksheet</a><button class="secondary js-only" id="print-lesson" type="button">Print all steps</button></div><p class="done"><strong>Deliver:</strong> your result or proposal, one supported conclusion, one limitation and the next comparison.</p>'))
        support = '<p><a href="../../handbook/student-start/index.html">Readiness</a> · <a href="../../handbook/student-report/index.html">Output-to-report example</a> · <a href="../../handbook/student-help/index.html">If you are stuck</a></p>'
        blocks[1] = (blocks[1][0], blocks[1][1] + support)
        core = {'control', 'X22', 'X14', 'X17', 'X22+X17', 'B3'}
        if ident in core:
            if ident == 'B3':
                contract = 'Input: corrected T1 data, training-only statistics and the foundation encoder initialization. Action: train B3 only, verify completion, export the validation-selected checkpoint, then evaluate frozen B3 on 2021. Output: completion record, checkpoint/checksum and four-condition summary. Proceed only when the exported checkpoint matches its record.'
                recipe = '../../handbook/baselines/index.html#t1-classroom-path'
                label = 'B3-only preparation'
            else:
                roles = {'control':'one constant-rate ERM branch; compare with frozen B3',
                         'X22':'one cosine branch; reuse ERM only if its full contract matches',
                         'X14':'one mixed-severity block branch; compare with matched ERM',
                         'X17':'two fixed-severity branches (25% and 50%); compare each with the mixed specialist at that severity',
                         'X22+X17':'the matched ERM, cosine and two fixed experts for the routed score, plus the mixed specialist for attribution'}
                contract = 'First classroom setting: T1, seed 0, verified B3. Prepare ' + roles[ident] + '. Start each continuation independently from B3 with a fresh optimizer, batch 64 and 3,000 updates; select the final checkpoint. Evaluate every component on the same 2021 samples and all four conditions. Bring back checkpoint.pt, summary.json, completion logs and source/configuration identities for each trained branch. Proceed only after the expected branches and matching records exist; a proposal alone does not complete the practical.'
                recipe = '../../handbook/method-recipes/index.html'
                label = 'Select only the required recipe roles'
            blocks[3] = (blocks[3][0], '<h2>Turn this experiment into a checkable handover.</h2><p>' + contract + '</p><p><a href="'+recipe+'">'+label+'</a>. Agree on resource limits before running. Use separate smoke outputs; do not continue a smoke checkpoint.</p><p><a href="../../handbook/student-bridge/index.html#handover">Input asset sheet</a> → <a href="../../handbook/student-report/index.html#files">Output field checks</a> → <a href="../../handbook/student-report/index.html#review">Report acceptance criteria</a></p>')
        blocks[-1] = (blocks[-1][0], blocks[-1][1] + '<p><a href="../../handbook/student-report/index.html#model-report">Read a completed teaching report</a>; replace synthetic evidence with your actual artifacts for a practical.</p>')
        panels=[]
        for i,(label,body) in enumerate(blocks,1):
            prev=f'<a href="#step{i-1}">← Previous step</a>' if i>1 else '<a href="../index.html">← Experiment directory</a>'
            nxt=f'<a class="button" href="#step{i+1}">Next step →</a>' if i<len(blocks) else '<a class="button" href="../index.html">Back to experiment directory →</a>'
            panels.append(f'<section class="step-panel" id="step{i}"><p class="step-kicker">Experiment guide · {i} / {len(blocks)}</p>{body}<div class="pager">{prev}{nxt}</div></section>')
        intro=f'<header class="lesson-header"><p class="eyebrow">High-level experiment guide · {escape(item["kind"])}</p><h1>{escape(item["name"])}</h1><p class="source">Record ID: {escape(ident)} · Lifecycle: {escape(lifecycle)} · Evidence status: {escape(status)}</p><p>Work through the preparation, intervention, comparison and report. Keep the original identifier when locating code or results.</p></header>'
        nav='<nav class="step-nav" aria-label="Experiment steps">'+''.join(f'<a href="#step{i}">{i} · {label}</a>' for i,(label,_) in enumerate(blocks,1))+'</nav>'
        output=ROOT/'experiments'/slug(ident)/'index.html';output.parent.mkdir(parents=True,exist_ok=True)
        body=descriptive_names('\n'.join(panels),'../../')
        output.write_text(migrate_links(shell(item['name'],body,prefix='../../',page='experiment-'+slug(ident),intro=intro,step_links=nav,source='../../'+item['source']),output,output))
        worksheet=f'# {item["name"]} ({ident})\n\nQuestion: {plan["question"]}\n\n'
        for label,key in [('Preparation','prepare'),('Intervention','intervention'),('Comparison','comparison'),('Report','deliver')]:
            worksheet+=f'## {label}\n\n'+''.join(f'- [ ] {line}\n' for line in plan[key])+'\n'
        worksheet+='## My record\n\n- Data / statistics / initialization identities:\n- Source version and intended differences:\n- Budget and selection/evaluation contract:\n- Job IDs, terminal states and actual artifact paths (if run):\n- Result table or preparation-only status:\n- Supported conclusion and limitation:\n- Next comparison:\n\nDo not fill missing measurements with teaching values.\n'
        (output.parent/'worksheet.md').write_text(worksheet)
    print(f'Rendered {len(names)} high-level experiment/scenario guides and worksheets.')

if __name__=='__main__':main()
