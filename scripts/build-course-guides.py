#!/usr/bin/env python3
"""Render the English onboarding Markdown as offline-readable course pages.

Requires pandoc (verified with 2.18). Run from any directory; no model imports.
"""
from pathlib import Path
import os
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [ROOT / name for name in (
    'docs/START_HERE.md', 'docs/research-roadmap.md',
    'docs/tutorials/res18-baseline-slurm.md', 'docs/tutorials/project-b0-slurm.md',
)] + sorted((ROOT / 'docs/research').glob('*.md'))
OUTPUTS = {p.resolve(): ROOT / 'guides' / p.relative_to(ROOT).with_suffix('.html') for p in SOURCES}
TEMPLATE = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>$pagetitle$ · Wildfire Forecasting Course</title><link rel="stylesheet" href="$stylesheet$"></head>
<body><nav aria-label="Course navigation"><a href="$home$">Course home</a><a href="$start$">Start here</a><a href="$lecture$">Baseline reproduction</a></nav>
<main class="guide"><p class="source">Generated from the maintained English guide. <a href="$source$">View source</a>. Follow the commands for your selected experimental protocol.</p>
$if(toc)$<details><summary>On this page</summary><div id="TOC">$toc$</div></details>$endif$
$body$
</main><footer>Wildfire research methods · <a href="$home$">Back to course home</a></footer></body></html>
'''


def main():
    with tempfile.TemporaryDirectory(prefix='wildfire-course-') as temp:
        template = Path(temp) / 'page.html'
        template.write_text(TEMPLATE)
        for source, output in OUTPUTS.items():
            def relative(path):
                return os.path.relpath(path, output.parent).replace(os.sep, '/')

            def link(match):
                target = match.group(2)
                if re.match(r'[a-zA-Z][a-zA-Z0-9+.-]*:', target) or target.startswith('#'):
                    return match.group(0)
                path, sep, fragment = target.partition('#')
                resolved = (source.parent / path).resolve()
                return match.group(1) + relative(OUTPUTS.get(resolved, resolved)) + (sep + fragment if sep else '') + ')'

            english = ROOT / 'site-content/en' / source.relative_to(ROOT)
            if not english.is_file():
                raise FileNotFoundError(f'Missing English website source: {english}')
            text = english.read_text()
            if re.search(r'[\u3400-\u9fff]', text):
                raise ValueError(f'Untranslated text in English website source: {english}')
            # Resolve links against the canonical document; preserve English code blocks.
            chunks = re.split(r'(^```[^\n]*\n.*?^```[^\n]*$)', text, flags=re.M | re.S)
            content = ''.join(chunk if chunk.startswith('```') else re.sub(r'(\]\()([^\s)]+)\)', link, chunk) for chunk in chunks)
            title = next(line[2:] for line in content.splitlines() if line.startswith('# '))
            output.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['pandoc', '--from=gfm', '--to=html5', '--standalone', '--toc',
                '--template', str(template), '--metadata', 'pagetitle=' + title,
                '--metadata', 'stylesheet=' + relative(ROOT / 'assets/course.css'),
                '--metadata', 'home=' + relative(ROOT / 'index.html'),
                '--metadata', 'start=' + relative(OUTPUTS[(ROOT / 'docs/START_HERE.md').resolve()]),
                '--metadata', 'lecture=' + relative(ROOT / 'baseline-reproduction/index.html'),
                '--metadata', 'source=' + relative(english), '--output', str(output)],
                input=content, text=True, check=True)
    print(f'Rendered {len(OUTPUTS)} course guides.')


if __name__ == '__main__':
    main()
