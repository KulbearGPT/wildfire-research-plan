#!/usr/bin/env python3
"""Export real generated DOM structure for the dependency-free JS flow harness."""
from html.parser import HTMLParser
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
class Tree(HTMLParser):
    def __init__(self):
        super().__init__();self.root={'tag':'document','attrs':{},'children':[]};self.stack=[self.root]
    def handle_starttag(self,tag,attrs):
        node={'tag':tag,'attrs':dict(attrs),'children':[]};self.stack[-1]['children'].append(node)
        if tag not in {'area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'}:self.stack.append(node)
    def handle_endtag(self,tag):
        for i in range(len(self.stack)-1,0,-1):
            if self.stack[i]['tag']==tag:self.stack=self.stack[:i];break
    def handle_data(self,text):
        if text.strip():self.stack[-1]['children'].append(text)
paths=['index.html']+[f'lessons/{s}/index.html' for s in ['forecast','prediction','evaluation','fold2','missingness','erm','schedule','specialists']]+['handbook/student-bridge/index.html','handbook/method-recipes/index.html']
pages={}
for path in paths:
    t=Tree();t.feed((ROOT/path).read_text());pages[path]=t.root
scripts={name:(ROOT/'assets'/name).read_text() for name in ['course-map.js','course-state.js','visual-lab.js','learning.js','lesson-demos.js','course-route.js']}
print(json.dumps({'pages':pages,'scripts':scripts}))
