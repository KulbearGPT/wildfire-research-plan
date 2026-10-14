#!/usr/bin/env python3
"""Build student handover pages and explicitly synthetic teaching downloads."""
import json
from course_shell import ROOT, Sections, reference_page


def main():
    source = ROOT / 'site-content/en/student-support'
    for page in json.loads((source / 'pages.json').read_text()):
        slug = page['slug']
        fragment = (source / f'{slug}.html').read_text()
        sections = [(a['id'], a['data-title'], body[body.index('>') + 1:-len('</section>')])
                    for a, body, _, _ in Sections(fragment, lambda a: 'data-title' in a).items]
        output = ROOT / f'handbook/student-{slug}/index.html'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(reference_page(page['title'], sections, prefix='../../',
            page=f'student-{slug}', source=f'../../site-content/en/student-support/{slug}.html',
            description='Follow the action, check your evidence, then return to the linked lesson.'))
    # All downloadable examples are maintained beside the page sources.
    for name, slug in [('cedar.json', 'case'), ('teaching-outputs.json', 'report')]:
        (ROOT / f'handbook/student-{slug}/{name}').write_text((source / name).read_text())
    print('Rendered 5 student support pages and 2 synthetic teaching downloads.')


if __name__ == '__main__':
    main()
