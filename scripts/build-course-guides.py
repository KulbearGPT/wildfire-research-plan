#!/usr/bin/env python3
"""Render maintained English Markdown in the shared static course shell.

Requires Pandoc. Scientific code blocks and original heading anchors are kept.
No model imports, network calls or experiment execution.
"""
from pathlib import Path
from html import unescape
import os
import re
import subprocess
from course_shell import ROOT, reference_page, migrate_links

SOURCES = [ROOT / name for name in (
    'docs/START_HERE.md', 'docs/research-roadmap.md',
    'docs/tutorials/res18-baseline-slurm.md', 'docs/tutorials/project-b0-slurm.md',
    'docs/tutorials/matched-erm-experiment-design.md',
)] + sorted((ROOT / 'docs/research').glob('*.md'))
OUTPUTS = {p.resolve(): ROOT / 'handbook' / p.stem.lower().replace('_','-') / 'index.html' for p in SOURCES}


def plain(text):
    return unescape(re.sub('<[^>]+>', '', text)).strip()


def main():
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
        text = english.read_text()
        if re.search(r'[\u3400-\u9fff]', text):
            raise ValueError(f'Untranslated text in {english}')
        chunks = re.split(r'(^```[^\n]*\n.*?^```[^\n]*$)', text, flags=re.M | re.S)
        markdown = ''.join(chunk if chunk.startswith('```') else re.sub(r'(\]\()([^\s)]+)\)', link, chunk) for chunk in chunks)
        fragment = subprocess.run(['pandoc','--from=gfm','--to=html5'],input=markdown,text=True,check=True,capture_output=True).stdout
        heading = re.search(r'<h1([^>]*)>(.*?)</h1>',fragment,re.S)
        title = plain(heading.group(2))
        # Keep the original title anchor, but only one visible page H1.
        title_anchor = re.search(r'id="([^"]+)"',heading.group(1))
        fragment = fragment[:heading.start()] + (f'<span id="{title_anchor.group(1)}"></span>' if title_anchor else '') + fragment[heading.end():]
        chunks = re.split(r'(?=<h2(?:\s|>))',fragment)
        sections=[]
        for i, chunk in enumerate(chunks):
            if not chunk.strip():continue
            h = re.search(r'<h2[^>]*>(.*?)</h2>',chunk,re.S)
            label = plain(h.group(1)) if h else 'Before you begin'
            if not h:chunk='<h2>Before you begin</h2>'+chunk
            sections.append((f'chapter-{i+1}',label,chunk))
        slug=source.stem
        description = 'Read one section at a time. Keep the exact settings and record the artifacts for your chosen protocol.'
        if slug in ('positive-signals','negative-results','qualification','research-roadmap'):
            description='Historical evidence and context. These records support interpretation; they are not a required sequence of new experiments.'
        current = {'res18-baseline-slurm':'fold2','matched-erm-experiment-design':'erm','method-recipes':'specialists','data-preparation':'missingness'}.get(slug,'')
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(migrate_links(reference_page(title,sections,prefix=relative(ROOT)+'/',page='guide-'+slug,source=relative(english),description=description,current=current),output,output))
    print(f'Rendered {len(OUTPUTS)} course guides with the shared reference shell.')

if __name__ == '__main__':
    main()
