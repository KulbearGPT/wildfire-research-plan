#!/usr/bin/env python3
"""Build eight static lessons and the descriptive experiment catalog."""
from pathlib import Path
from html import escape
import json
import re
from course_shell import ROOT, Sections, lessons, shell, migrate_links
from course_flow import lesson_readings
from course_visuals import explorer, contract_lab, route_lab
from importlib.util import spec_from_file_location, module_from_spec


def main():
    course = lessons()
    for lesson in course:
        slug = lesson['slug']
        path = ROOT / f'site-content/en/lessons/{slug}.html'
        text = path.read_text()
        sections = Sections(text, lambda a: 'step-panel' in a.get('class', '').split()).items
        if not sections:
            raise ValueError(f'No steps in {path}')
        nav = '<details class="lesson-contents"><summary id="section-label">Contents · choose a step</summary><nav class="step-nav" aria-label="Lesson steps">' + ''.join(
            f'<a href="#{a["id"]}">{i} · {escape(a["data-title"])}</a>'
            for i, (a, _, _, _) in enumerate(sections, 1)) + '</nav></details>'
        content = []
        for i, (attrs, body, _, _) in enumerate(sections):
            # Replace the static lesson figure with an in-place guided explorer.
            views = {'forecast':'timeline','prediction':'channels','evaluation':'persistence','missingness':'missingness'}
            if slug in views and '<!-- real-data-start -->' in body:
                body = re.sub(r'<!-- real-data-start -->.*?<!-- real-data-end -->',
                              lambda _: explorer(slug, views[slug]), body, flags=re.S)
            # The explorer already provides the binary-mask view.
            body = re.sub(r'<!-- real-mask-start -->.*?<!-- real-mask-end -->', '', body, flags=re.S)
            if (slug, attrs['id']) == ('schedule','step3'):
                body = body[:-len('</section>')] + contract_lab() + '</section>'
            if (slug, attrs['id']) == ('specialists','step2'):
                body = body[:-len('</section>')] + route_lab() + '</section>'
            body = lesson_readings(body, slug, attrs['id'])
            prev = ('../../index.html#course', '← Course directory') if i == 0 else (f'#{sections[i-1][0]["id"]}', '← Previous step')
            if i + 1 < len(sections):
                next_link = (f'#{sections[i+1][0]["id"]}', 'Next step →')
            elif lesson['number'] < 8:
                next_lesson = course[lesson['number']]
                next_link = (f'../{next_lesson["slug"]}/index.html', f'Next: {next_lesson["short"]} →')
            else:
                next_link = ('../../index.html#course', 'Return to the course →')
            ending = ''
            if i == len(sections) - 1:
                download = 'report-template.md' if slug == 'erm' else 'worksheet.md'
                ending = f'''<div class="actions"><a class="button secondary" href="{download}" download>Download your worksheet</a><button class="secondary js-only" type="button" id="print-lesson">Print all steps</button></div>
<label class="completion js-only"><input type="checkbox" id="lesson-complete"> I have reviewed the concepts. Save this reminder locally.</label><p class="source">This is a personal reminder, not a grade or proof that a cluster experiment finished.</p>'''
            pager = f'<div class="pager"><a href="{prev[0]}">{escape(prev[1])}</a><a class="button" href="{next_link[0]}">{escape(next_link[1])}</a></div>'
            content.append(body[:-len('</section>')] + ending + pager + '</section>')
        pre = 'Start here with basic Python and CNN concepts.' if lesson['number'] == 1 else f'Before this lesson: <a href="../{course[lesson["number"]-2]["slug"]}/index.html">{escape(course[lesson["number"]-2]["short"])}</a>.'
        intro = f'''<header class="lesson-header"><p class="eyebrow">Lesson {lesson['number']:02} · Learn by doing</p><h1>{escape(lesson['title'])}</h1><p class="lead">{escape(lesson['goal'])}</p><div class="meta"><span>{escape(lesson['time'])}</span><span>{len(sections)} steps</span><span>Queue and training time are separate</span></div><details><summary>Before you start &amp; what you will produce</summary><p>{pre}</p><p><strong>Your deliverable:</strong> {escape(lesson['deliverable'])}</p><p>Follow the next-step button. Extra reading is grouped at the end of each step; practical instructions remain available there. Browser activities use teaching examples and labelled real exports.</p></details></header>'''
        output = ROOT / f'lessons/{slug}/index.html'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(migrate_links(shell(lesson['title'], '\n'.join(content), prefix='../../', page=slug, current=slug,
                                step_links=nav, intro=intro, source=f'../../site-content/en/lessons/{slug}.html'),output,output))
    spec=spec_from_file_location('experiment_guides',ROOT/'scripts/build-experiment-guides.py')
    module=module_from_spec(spec); spec.loader.exec_module(module); module.main()
    support_spec=spec_from_file_location("student_support", ROOT/"scripts/build-student-support.py")
    support=module_from_spec(support_spec); support_spec.loader.exec_module(support); support.main()
    build_catalog()
    build_home(course)
    print(f'Rendered {len(course)} lessons and the experiment name catalog.')


def build_catalog():
    names = json.loads((ROOT / 'site-content/en/experiment-names.json').read_text())['entries']
    inventory = {m['id']:m for m in json.loads((ROOT / 'docs/research/method-inventory.json').read_text())['methods']}
    cards = []
    core_cards = []
    core = {'control','official-fold-2','B3','B5','X14','X17','X22','X22+X17','M00','M01','M06','M07'}
    for item in names:
        evidence = inventory.get(item['id'], {})
        lifecycle = evidence.get('lifecycle', {}).get('state', '')
        status = evidence.get('status', '')
        search = ' '.join(str(item.get(k,'')) for k in ['id','name','name_zh','explanation','kind']) + ' ' + evidence.get('family','') + ' ' + lifecycle + ' ' + status
        source = item['source']
        # Sources are repository paths, not executable aliases or remote redirects.
        if source.startswith('/') or '..' in Path(source.split('#')[0]).parts:
            raise ValueError(source)
        target = item['id'].lower().replace('+','-plus-')
        destination = core_cards if item['id'] in core else cards
        destination.append(f'''<article class="resource experiment-card" id="{escape(item['id'],quote=True)}" data-experiment data-search="{escape(search.lower(),quote=True)}"><span class="pill">{escape(item['kind'])}</span><h2>{escape(item['name'])}</h2><p class="experiment-id">Record ID: <code>{escape(item['id'])}</code></p><p>{escape(item['explanation'])}</p><p class="source">{escape(lifecycle)}{(' · Evidence status: '+escape(status)) if status else ''}</p><a class="button secondary" href="{target}/index.html">Work through the experiment →</a></article>''')
    intro = '''<header class="lesson-header"><p class="eyebrow">Reference library · Experiment names</p><h1>Remember the idea.<br>Keep the record ID.</h1><p class="lead">Find what an experiment changes, then trace it back to its evidence.</p><p>Names describe mechanisms, not success. Machine commands and historical records retain their original identifiers. A scenario is an input condition, not a trained method.</p></header>'''
    controls = f'''<div class="search-box js-only"><label for="experiment-search">Search by name, ID, mechanism, family or status</label><input id="experiment-search" type="search" placeholder="Try cosine, X14, distillation or archived"><p class="source" id="experiment-count" aria-live="polite">{len(names)} entries</p></div>'''
    output = ROOT / 'experiments/index.html'; output.parent.mkdir(exist_ok=True)
    output.write_text(shell('Experiment name finder', controls+'<h2>Course foundations and mainline</h2><div class="resource-grid">'+''.join(core_cards)+'</div><details id="historical-experiments"><summary>All other experiments and historical investigations</summary><div class="resource-grid">'+''.join(cards)+'</div></details>', prefix='../',page='experiments',intro=intro,source='../site-content/en/experiment-names.json'))


def build_home(course):
    cards=[]
    for item in course:
        cards.append(f'''<article class="lesson-card" id="lesson{item['number']}"><div class="number">{item['number']:02}</div><div><span class="pill live">Interactive lesson · {escape(item['time'])}</span><h3>{escape(item['short'])}</h3><p>{escape(item['goal'])}</p><p class="deliver"><strong>You will produce:</strong> {escape(item['deliverable'])}</p><a class="button secondary" href="lessons/{item['slug']}/index.html">Start lesson {item['number']} →</a><a class="lesson-resume js-only" data-resume-page="{item['slug']}" href="lessons/{item['slug']}/index.html">Resume saved step →</a></div></article>''')
    intro='''<header class="hero"><p class="eyebrow">A practical course in research</p><h1>From a fire map<br>to a fair experiment.</h1><p class="lead">Eight lessons. One learning path. Explore an example, make a decision, then produce something you can explain.</p><div class="actions"><a class="button" href="lessons/forecast/index.html">Start lesson 1 →</a><a class="button" href="lessons/erm/index.html" data-continue>Try the ERM comparison →</a></div><div class="meta"><span>Before you start: basic Python + CNN concepts</span><span>Browser activities + separate Slurm practicals</span></div></header>'''
    content='<section class="section"><h2>Before you begin</h2><p><a href="handbook/student-start/index.html">Choose your study or practical route</a> · <a href="handbook/student-real-data/index.html">Explore real wildfire observations</a> · <a href="student-record.md" download>Keep one course record</a></p><p>Start with lessons 1–3. Before cluster work, check access and assets. Without compute access, continue the labelled teaching exercises.</p></section>'+'<section id="course" class="section"><h2>Your course directory</h2><p>Follow the lessons in order. Each ends with a completion criterion, an exit task and a downloadable worksheet. Cluster work is separate from browser exercises.</p>'+''.join(cards)+'</section>'
    resources=(ROOT/'site-content/en/course-resources.html').read_text()
    (ROOT/'index.html').write_text(migrate_links(shell('Learn to build a trustworthy experiment',content+resources,prefix='',page='home',intro=intro,source='site-content/en/course-lessons.json'),ROOT/'index.html',ROOT/'index.html'))

if __name__ == '__main__':
    main()
