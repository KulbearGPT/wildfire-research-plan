#!/usr/bin/env python3
"""Check generated course markup, anchors and retirement (stdlib; no models)."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
from collections import Counter
r=Path(__file__).resolve().parents[1]
void={'area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'}
class Page(HTMLParser):
 def __init__(self,text,strict=False):
  super().__init__();self.ids=[];self.links=[];self.labels=[];self.stack=[];self.errors=[];self.strict=strict;self.pre=[];self.active=None;self.feed(text)
  if strict and self.stack:self.errors.append(f'unclosed {self.stack}')
 def handle_starttag(self,t,attrs):
  a=dict(attrs)
  if 'id' in a:self.ids.append(a['id'])
  if t=='pre':self.active=''
  for k in ['href','src']:
   if k in a:self.links.append(a[k])
  if 'for' in a:self.labels.append(a['for'])
  if 'aria-labelledby' in a:self.labels.extend(a['aria-labelledby'].split())
  if t not in void:self.stack.append(t)
 def handle_endtag(self,t):
  if t in void:return
  if t=='pre':self.pre.append(self.active);self.active=None
  if self.strict and (not self.stack or self.stack[-1]!=t):self.errors.append(f'unbalanced {t}: {self.stack[-3:]}')
  if self.stack and self.stack[-1]==t:self.stack.pop()
 def handle_data(self,data):
  if self.active is not None:self.active+=data
files=[r/'index.html',*sorted((r/'experiments').rglob('*.html')),*sorted((r/'lessons').glob('*/index.html')),*sorted((r/'handbook').rglob('*.html')),*sorted((r/'library').rglob('*.html'))]
parsed={p:Page(p.read_text(),strict=True) for p in files}
errors=[];count=0
for p,page in parsed.items():
 name=str(p.relative_to(r));errors += [name+': '+e for e in page.errors]
 for k,v in Counter(page.ids).items():
  if v>1:errors.append(f'{name}: duplicate ID {k}')
 for label in page.labels:
  if label not in page.ids:errors.append(f'{name}: missing label {label}')
 for url in page.links:
  u=urlsplit(url)
  if u.scheme or u.netloc:continue
  target=(p.parent/unquote(u.path)).resolve() if u.path else p
  if target.is_dir():target/='index.html'
  count+=1
  if not target.exists():errors.append(f'{name}: missing {url}');continue
  if u.fragment and target.suffix=='.html':
   other=parsed.get(target) or Page(target.read_text())
   if unquote(u.fragment) not in other.ids:errors.append(f'{name}: missing anchor {url}')
# Retirement is intentional: no landing stubs, redirects, or active links remain.
retired=[r/'related-work/index.html',r/'baseline-reproduction/index.html',*list((r/'guides').rglob('*.html'))]
for path in retired:
 if path.exists():errors.append(f'Retired page still exists: {path.relative_to(r)}')
for p in files:
 if 'http-equiv="refresh"' in p.read_text().lower():errors.append(f'Redirect found: {p.relative_to(r)}')
print(f'{len(files)} pages; {count} local links/assets; {len(errors)} errors')
print('\n'.join(errors));raise SystemExit(bool(errors))
