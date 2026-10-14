#!/usr/bin/env python3
"""Build historical lecture references without changing original evidence."""
import re
from html import unescape
from course_shell import ROOT, Sections, reference_page, migrate_links

for slug,title,lesson in [('related-work','Research foundations and related work','forecast'),('baseline-reproduction','From a paper claim to a verified baseline','fold2')]:
    source=ROOT/f'site-content/en/reference/{slug}.html'
    text=source.read_text()
    sections=[]
    for attrs,body,_,_ in Sections(text,lambda a:'reference-section' in a.get('class','').split()).items:
        h=re.search(r'<h2[^>]*>(.*?)</h2>',body,re.S)
        label=unescape(re.sub('<[^>]+>','',h.group(1)))
        # Drop the source container; the common shell owns section navigation.
        body=body[body.index('>')+1:-len('</section>')]
        sections.append((attrs['id'],label,body))
    output=ROOT/'library'/('foundations' if slug == 'related-work' else 'reproduction')/'index.html'
    output.parent.mkdir(parents=True,exist_ok=True)
    sections=[(ident,label,migrate_links(body,ROOT/slug/'index.html',output)) for ident,label,body in sections]
    output.write_text(reference_page(title,sections,prefix='../../',page='reference-'+slug,source=f'../../site-content/en/reference/{slug}.html',description=('Original figures and explanations. Literature retains its 18 August 2026 review scope; this redesign is not a new literature survey.' if slug == 'related-work' else 'Historical reproduction published 13 August 2026. These measurements are preserved records, not newly run experiments.'),current=lesson))
    print(f'Rendered {slug}: {len(sections)} reference chapters.')
