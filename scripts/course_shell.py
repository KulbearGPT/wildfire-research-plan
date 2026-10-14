"""Static course layout shared by lessons and reference pages (stdlib only)."""
from html import escape
from html.parser import HTMLParser
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def lessons():
    return json.loads((ROOT / 'site-content/en/course-lessons.json').read_text())


class Sections(HTMLParser):
    """Locate complete outer section elements without reparsing their contents."""
    def __init__(self, text, match):
        super().__init__(convert_charrefs=False)
        self.text, self.match, self.items = text, match, []
        self.offsets = [0]
        for line in text.splitlines(keepends=True):
            self.offsets.append(self.offsets[-1] + len(line))
        self.depth, self.start, self.attrs = 0, None, None
        self.feed(text)

    def position(self):
        row, col = self.getpos()
        return self.offsets[row - 1] + col

    def handle_starttag(self, tag, attrs):
        if tag != 'section':
            return
        if self.start is not None:
            self.depth += 1
        elif self.match(dict(attrs)):
            self.start, self.attrs, self.depth = self.position(), dict(attrs), 1

    def handle_endtag(self, tag):
        if tag == 'section' and self.start is not None:
            self.depth -= 1
            if self.depth == 0:
                end = self.position() + len('</section>')
                self.items.append((self.attrs, self.text[self.start:end], self.start, end))
                self.start = None


def directory(prefix, current=''):
    items = []
    for item in lessons():
        active = ' aria-current="page"' if item['slug'] == current else ''
        items.append(f'<li><a href="{prefix}lessons/{item["slug"]}/index.html"{active}>'
                     f'{item["number"]:02} &nbsp; {escape(item["short"] )}</a></li>')
    return '<ol class="directory">' + ''.join(items) + '</ol>'


# The sidebar follows learning order; supporting groups are lookup material.
HANDBOOK_NAV = [
    ('Prepare environment', [('student-start', 'Readiness and access'), ('start-here', 'Preparation checklist'), ('reproduce', 'Project environment')]),
    ('Prepare data and weights', [('student-bridge', 'Official-to-project handover'), ('data-preparation', 'Project data'), ('baselines', 'Foundation checkpoints')]),
    ('Run an experiment', [('res18-baseline-slurm', 'Official Fold 2'), ('matched-erm-experiment-design', 'Matched ERM comparison'), ('method-recipes', 'Continuation recipes')]),
    ('Check results', [('evaluation-population', 'Evaluation population'), ('artifacts', 'Required artifacts'), ('student-report', 'From outputs to a report')]),
    ('Troubleshoot', [('student-help', 'Find the next recovery action'), ('setup-recovery', 'Environment and job recovery')]),
]
REFERENCE_NAV = [
    ('Concepts and examples', [('handbook/student-real-data/index.html', 'Real-data examples'), ('handbook/student-case/index.html', 'Cedar counting exercise'), ('library/foundations/index.html', 'Datasets and research foundations'), ('handbook/architectures/index.html', 'Model architectures')]),
    ('Experiment lookup', [('experiments/index.html', 'Experiment names and records')]),
    ('Research and historical records', [('library/reproduction/index.html', 'Reproduction background and results'), ('handbook/research-roadmap/index.html', 'Research roadmap'), ('handbook/positive-signals/index.html', 'Positive signals'), ('handbook/negative-results/index.html', 'Negative and incomplete results'), ('handbook/qualification/index.html', 'Execution qualification'), ('handbook/diagnostics-reproduction/index.html', 'Reproduction diagnostics'), ('handbook/legacy-p00/index.html', 'Historical P00 protocol'), ('handbook/project-b0-slurm/index.html', 'Project B0 protocol'), ('handbook/teachers/index.html', 'Instructor notes')]),
]


def course_navigation(prefix, page, title):
    lesson_page = page in {item['slug'] for item in lessons()}
    if lesson_page:
        path = f'lessons/{page}/index.html'
    elif page.startswith('guide-'):
        path = 'handbook/'+page[6:].lower().replace('_','-')+'/index.html'
    elif page.startswith('student-'):
        path = f'handbook/{page}/index.html'
    elif page.startswith('reference-'):
        path = 'library/'+('foundations' if page == 'reference-related-work' else 'reproduction')+'/index.html'
    elif page == 'experiments':
        path = 'experiments/index.html'
    elif page.startswith('experiment-'):
        path = 'experiments/'+page[11:]+'/index.html'
    else:
        path = 'index.html'
    handbook = [(label, [(f'handbook/{slug}/index.html', name) for slug,name in entries]) for label,entries in HANDBOOK_NAV]
    references = [(label,list(entries)) for label,entries in REFERENCE_NAV]
    if page.startswith('experiment-'):
        references[1][1].append((path,title))
    handbook_active = any(path == href for _,entries in handbook for href,_ in entries)
    reference_active = any(path == href for _,entries in references for href,_ in entries)
    def groups(groups):
        parts = []
        for label,entries in groups:
            rows = ''.join(f'<li><a href="{prefix}{href}"'+(' aria-current="page"' if href==path else '')+f'>{escape(name)}</a></li>' for href,name in entries)
            parts.append(f'<li class="nav-category"><span>{escape(label)}</span><ul class="directory">{rows}</ul></li>')
        return '<ul class="resource-directory">'+''.join(parts)+'</ul>'
    primary_open = ' open' if lesson_page or page=='home' else ''
    return f'''<nav aria-label="Course navigation"><h2 class="sidebar-title">Course navigation</h2>
<details class="nav-group" data-directory data-nav-group="lessons"{primary_open}><summary>Learning path</summary>{directory(prefix,page if lesson_page else '')}</details>
<details class="nav-group" data-nav-group="handbook"{' open' if handbook_active else ''}><summary>Practical handbook</summary><p class="nav-hint">Use when needed for a practical; not extra lessons.</p>{groups(handbook)}</details>
<details class="nav-group" data-nav-group="reference"{' open' if reference_active else ''}><summary>Reference library</summary><p class="nav-hint">Look up a concept or record; no reading order required.</p>{groups(references)}</details></nav>'''


def shell(title, content, *, prefix, page, current='', step_links='', source='', intro=''):
    progress_html = '<div class="js-only"><p class="progress-note" id="progress-status" aria-live="polite"></p><button type="button" class="reset" id="reset-progress">Restart this lesson</button></div>' if step_links else ''
    in_lesson = page in {item['slug'] for item in lessons()}
    task_location = ''
    body_class = 'lesson-page' if in_lesson else 'reference-page'
    navigation = course_navigation(prefix, page, title)
    src = f'<a href="{escape(source, quote=True)}">Maintained source</a> · ' if source else ''
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)} · Wildfire Lab</title><link rel="stylesheet" href="{prefix}assets/learning.css">
<script src="{prefix}assets/course-map.js" defer></script><script src="{prefix}assets/course-state.js" defer></script><script src="{prefix}assets/visual-lab.js" defer></script><script src="{prefix}assets/learning.js" defer></script><script src="{prefix}assets/lesson-demos.js" defer></script><script src="{prefix}assets/course-route.js" defer></script></head>
<body class="{body_class}" data-page="{page}" data-root="{prefix}"><a class="skip" href="#main">Skip to content</a>
<header class="topbar"><a class="brand" href="{prefix}index.html">WILDFIRE <span>/ LAB</span></a>
<nav aria-label="Main navigation"><a href="{prefix}index.html#course">Course directory</a></nav></header>
<div class="layout"><aside class="sidebar">{navigation}{progress_html}</aside>
<main class="content" id="main"><aside class="course-return" data-course-return hidden><a data-course-return-link href="{prefix}index.html#course">Return to your lesson</a><span>Supporting material · your lesson is waiting</span></aside>{intro}{task_location}{step_links}
<noscript><p class="note">All sections are shown. Examples, answers and downloads remain readable without JavaScript. Use your browser’s Back button after reading a resource; the lesson-return button requires JavaScript.</p></noscript>
{content}<p class="support-return"><a class="button" data-course-return-link hidden href="{prefix}index.html#course">Return to your task</a></p><footer class="footer"><details><summary>Sources and course resources</summary>{src}<a href="{prefix}index.html#course">Course directory</a> · <a href="{prefix}experiments/index.html">Experiment names</a></details>
Browser demonstrations are teaching examples, not new model results. Run model and data work through Slurm.</footer></main></div></body></html>
'''


def descriptive_names(fragment, prefix):
    """Define each historical ID once in prose; preserve code, links and attributes."""
    import re
    entries = json.loads((ROOT / 'site-content/en/experiment-names.json').read_text())['entries']
    mapping = {e['id']: e['name'] for e in entries if e['id'] != 'control'}
    pattern = re.compile(r'(?<![\w-])(' + '|'.join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)) + r')(?![\w-])')

    class Names(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=False)
            self.parts, self.stack, self.seen = [], [], set()
        def handle_starttag(self, tag, attrs):
            self.parts.append(self.get_starttag_text())
            if tag in ('pre','code','a','script','style'):
                self.stack.append(tag)
        def handle_endtag(self, tag):
            self.parts.append(f'</{tag}>')
            if self.stack and self.stack[-1] == tag:
                self.stack.pop()
        def handle_startendtag(self, tag, attrs):
            self.parts.append(self.get_starttag_text())
        def handle_data(self, text):
            def replace(match):
                ident = match.group(0)
                if ident in self.seen:
                    return ident
                self.seen.add(ident)
                return f'<span class="experiment-name">{escape(mapping[ident])} <small>({escape(ident)})</small></span>'
            self.parts.append(text if self.stack else pattern.sub(replace,text))
        def handle_entityref(self, name): self.parts.append(f'&{name};')
        def handle_charref(self, name): self.parts.append(f'&#{name};')
        def handle_comment(self, data): self.parts.append(f'<!--{data}-->')
    parser = Names(); parser.feed(fragment)
    return ''.join(parser.parts)


def descriptive_title(text):
    import re
    names=json.loads((ROOT/'site-content/en/experiment-names.json').read_text())['entries']
    names={x['id']:x['name'] for x in names if x['id']!='control'}
    pattern=re.compile(r'(?<![\w-])('+'|'.join(re.escape(k) for k in sorted(names,key=len,reverse=True))+r')(?![\w-])')
    return pattern.sub(lambda m: names[m.group(0)]+' ('+m.group(0)+')',text)


def reference_page(title, sections, *, prefix, page, source, description, current=''):
    """Render (id, title, content) chapters in the same progressively enhanced shell."""
    title = descriptive_title(title)
    nav = '<details open data-default-open class="reference-toc" id="TOC"><summary id="section-label">Contents · choose a section</summary><nav class="step-nav" aria-label="Reference sections">' + ''.join(
        f'<a href="#{escape(ident,quote=True)}">{i+1} · {escape(descriptive_title(label))}</a>' for i,(ident,label,_) in enumerate(sections)) + '</nav></details>'
    content = []
    for i, (ident, label, fragment) in enumerate(sections):
        prev = f'<a href="#{sections[i-1][0]}">← Previous section</a>' if i else f'<a href="{prefix}index.html#handbook">← Handbook</a>'
        nxt = f'<a class="button" href="#{sections[i+1][0]}">Read next: {escape(descriptive_title(sections[i+1][1]))} →</a>' if i+1<len(sections) else f'<a class="button" href="{prefix}index.html#course">Back to the course →</a>'
        content.append(f'<section class="step-panel" id="{escape(ident,quote=True)}"><p class="step-kicker">Reference · {i+1} / {len(sections)}</p>{fragment}<div class="reference-pager">{prev}<a class="button" data-course-return-link hidden href="{prefix}index.html#course">Return to your task</a><span data-reference-next>{nxt}</span></div></section>')
    intro = f'<header class="lesson-header reference-intro"><p class="eyebrow">Practical handbook &amp; reference library</p><h1>{escape(title)}</h1><p class="lead">{escape(description)}</p><p>Use the contents to jump to the section you need. Follow the linked lesson for the learning sequence.</p><button type="button" class="secondary js-only" id="print-lesson">Print the complete reference</button></header>'
    body = descriptive_names('\n'.join(content),prefix)
    import re
    def fold_code(match):
        block = match.group(0)
        count = block.count('\n') + 1
        if count < 13:
            return block
        return f'<details class="code-disclosure"><summary>Show complete code · {count} rendered lines</summary>{block}</details>'
    body = re.sub(r'<pre\b[^>]*>.*?</pre>',fold_code,body,flags=re.S)
    return shell(title, '<div class="reference-body">'+body+'</div>', prefix=prefix,page=page,current=current,step_links=nav,intro=intro,source=source)


def public_path(path):
    """Map retired teaching URLs to their new location at build time only."""
    path = path.resolve()
    try:
        rel = path.relative_to(ROOT)
    except ValueError:
        return path
    if rel.parts[:2] == ('guides','docs') and path.suffix == '.html':
        return ROOT / 'handbook' / path.stem.lower().replace('_','-') / 'index.html'
    if rel.as_posix() in ('related-work','related-work/index.html'):
        return ROOT / 'library/foundations/index.html'
    if rel.as_posix() in ('baseline-reproduction','baseline-reproduction/index.html'):
        return ROOT / 'library/reproduction/index.html'
    return path


def migrate_links(fragment, old_output, new_output):
    """Rewrite local HTML attributes, never scientific command text."""
    import re, os
    from urllib.parse import urlsplit, unquote
    def rewrite(match):
        attr, quote, value = match.group(1), match.group(2), match.group(3)
        parsed = urlsplit(value)
        if parsed.scheme or parsed.netloc or not parsed.path:
            return match.group(0)
        target = public_path(old_output.parent / unquote(parsed.path))
        relative = os.path.relpath(target,new_output.parent).replace(os.sep,'/')
        return attr + '=' + quote + escape(relative,quote=True) + ('?'+parsed.query if parsed.query else '') + ('#'+parsed.fragment if parsed.fragment else '') + quote
    return re.sub(r'''\b(href|src)=(['"])(.*?)\2''',rewrite,fragment)
