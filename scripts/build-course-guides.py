#!/usr/bin/env python3
"""Render the maintained onboarding Markdown as offline-readable course pages.

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
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>$pagetitle$ · 野火预测课程</title><link rel="stylesheet" href="$stylesheet$"></head>
<body><nav aria-label="课程导航"><a href="$home$">课程首页</a><a href="$start$">入门路线</a><a href="$lecture$">基线复现课</a></nav>
<main class="guide"><p class="source">本页由维护中的 Markdown 生成。<a href="$source$">查看源文档</a>。操作命令以所选实验协议为准。</p>
$if(toc)$<details><summary>本页目录</summary><div id="TOC">$toc$</div></details>$endif$
$body$
</main><footer>Wildfire research methods · <a href="$home$">返回课程首页</a></footer></body></html>
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

            # Keep shell examples byte-for-byte; rewrite only prose links.
            chunks = re.split(r'(^```[^\n]*\n.*?^```[^\n]*$)', source.read_text(), flags=re.M | re.S)
            content = ''.join(chunk if chunk.startswith('```') else re.sub(r'(\]\()([^\s)]+)\)', link, chunk) for chunk in chunks)
            title = next(line[2:] for line in content.splitlines() if line.startswith('# '))
            output.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['pandoc', '--from=gfm', '--to=html5', '--standalone', '--toc',
                '--template', str(template), '--metadata', 'pagetitle=' + title,
                '--metadata', 'stylesheet=' + relative(ROOT / 'assets/course.css'),
                '--metadata', 'home=' + relative(ROOT / 'index.html'),
                '--metadata', 'start=' + relative(OUTPUTS[(ROOT / 'docs/START_HERE.md').resolve()]),
                '--metadata', 'lecture=' + relative(ROOT / 'baseline-reproduction/index.html'),
                '--metadata', 'source=' + relative(source), '--output', str(output)],
                input=content, text=True, check=True)
    print(f'Rendered {len(OUTPUTS)} course guides.')


if __name__ == '__main__':
    main()
