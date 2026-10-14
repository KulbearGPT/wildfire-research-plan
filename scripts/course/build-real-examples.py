#!/usr/bin/env python3
"""Publish the verified real-data export into maintained teaching fragments.

Run after reviewing the CPU export. No missing-image placeholders are published.
"""
import argparse
import json
import shutil
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--export',type=Path,required=True)
    a=parser.parse_args()
    manifest=json.loads((a.export/'manifest.json').read_text())
    if not manifest['events']:raise ValueError('No reviewed events')
    assets=ROOT/'assets/real-examples'
    # Validate paths before copying public files. Keep only named plot exports.
    for event in manifest['events']:
        if len(event['figures'])!=5:raise ValueError('Incomplete figure collection')
        for fig in event['figures']:
            path=Path(fig['path'])
            if path.is_absolute() or '..' in path.parts:raise ValueError(path)
            source=a.export/path
            if not source.is_file() or source.read_bytes()[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError(source)
            if fig.get('sha256'):
                import hashlib
                if hashlib.sha256(source.read_bytes()).hexdigest()!=fig['sha256']:raise ValueError('Figure checksum mismatch: '+str(path))
    assets.mkdir(exist_ok=True)
    for event in manifest['events']:
        for fig in event['figures']:
            dest=assets/fig['path'];dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(a.export/fig['path'],dest)
    shutil.copyfile(a.export/'manifest.json',assets/'manifest.json')
    captions={
      'timeline':'Orange pixels are active-fire proxy detections on each date; pale pixels have no positive detection. Changes do not describe the full burned perimeter. The last panel is the next-day target, not an input.',
      'fire-mask':'Left: the stored active-fire feature. In the original WSTS encoding shown here, positive values encode detection time after preprocessing, not fire intensity or model probability. Middle: the observed input mask produced by value > 0. Right: the same rule applied to the next day creates the target. Zero here follows the benchmark encoding; it is not proof of perfect visibility.',
      'channels':'Three genuine raw channels on the same date and grid. Color scales are independent and clipped at their 2nd/98th percentiles for display only. Values use the stored dataset encoding; do not interpret the raw NDVI colorbar as an already normalized index. These panels are not natural-color satellite photographs.',
      'missingness':'Real observations with controlled corruption. Gray marks deliberately hidden fire input; pale marks observed background. M06/M07 also hide dynamic non-fire features, while static terrain and the target remain unchanged. The masks come from the project corruption function; gray is a display overlay, not the numerical model input.',
      'persistence':'This forecast copies the previous observed mask with no learning. Blue is a correct fire prediction, orange a false alarm and purple a missed fire pixel. Pale pixels are true negatives. This is one patch, not a neural-model result or full-population AP.'}
    def figure(event,name):
        fig=next(f for f in event['figures'] if f['name']==name)
        url='../../assets/real-examples/'+fig['path']
        return f'<figure class="real-example"><a href="{url}"><img src="{url}" loading="lazy" alt="{escape(fig["title"])} for {escape(event["id"])} on {event["dates"][-2]} and {event["dates"][-1]}"></a><figcaption><strong>{escape(event["id"])} · input {event["dates"][-2]} → target {event["dates"][-1]}.</strong> {captions[name]} <a href="{url}">Open full size</a>.</figcaption><details><summary>Magnify the pixels</summary><p>Scroll across the enlarged view to inspect the same source pixels. The full patch above preserves context.</p><div class="real-magnified" tabindex="0" role="region" aria-label="Enlarged figure; scroll to inspect"><img src="{url}" loading="lazy" alt="Enlarged {escape(fig["title"])} for {escape(event["id"])}"></div></details></figure>'
    first=manifest['events'][0]
    parts=['<section id="read" data-title="Read a real patch"><h2>Read a real patch before interpreting the model.</h2><p>These are real 2021 WSTS source observations, not the fictional Cedar arrays. Inspect the date, feature and legend before comparing panels. Candidate event records were fixed from the inventory; within each, the first qualifying consecutive-date window was selected for visible fire. These are dataset event records, not necessarily independent fires; the two displayed crops show similar, potentially overlapping structures. They are teaching cases, not a representative performance sample.</p><p>All views use a 128×128 center crop, source pixel coordinates and nearest-neighbor display. No geographic basemap or complete fire perimeter is implied.</p><p><a href="../../assets/real-examples/manifest.json">View provenance and crop identities</a> · <a href="'+manifest['source_record']+'">Original dataset record</a></p></section>']
    for event in manifest['events']:
        counts=event.get('persistence_counts')
        text=f'<section id="{event["id"]}" data-title="{event["id"]}"><h2>{event["id"]}: {event["dates"][-2]} → {event["dates"][-1]}</h2><p>Source: <code>{event["source_file"]}</code>. Dates and pixel axes are preserved in the panels. These are full center crops, not zoomed fire perimeters.</p>'
        for name in captions:
            text+=figure(event,name)
        if counts:
            text+=f'<p>Persistence counts for this crop: TP {counts["tp"]}, FP {counts["fp"]}, FN {counts["fn"]}, TN {counts["tn"]}. Their denominator is this one crop, not the project evaluation population.</p>'
        else:
            text+='<p>Use this panel to locate error types. Exact confusion counts were not retained by the interrupted export, so no numerical score is inferred from the rendered image.</p>'
        text+='<p class="done">Point to one observed fire pixel, one hidden input region, and one persistence error. Explain why they describe different things.</p></section>'
        parts.append(text)
    parts.append('<section id="return" data-title="Use these views"><h2>Carry the observations back to your experiment.</h2><p><a href="../../lessons/forecast/index.html">Lesson 1: date and target</a> · <a href="../../lessons/prediction/index.html">Lesson 2: feature channels</a> · <a href="../../lessons/evaluation/index.html">Lesson 3: errors</a> · <a href="../../lessons/missingness/index.html">Lesson 5: missing observations</a></p><p>The first baseline here is persistence. ERM and cosine need their own verified model predictions before any qualitative superiority can be claimed. The small synthetic arithmetic exercises remain optional aids, separate from these real observations.</p></section>')
    source=ROOT/'site-content/en/student-support'
    (source/'real-data.html').write_text('\n'.join(parts))
    pages=json.loads((source/'pages.json').read_text())
    pages=[p for p in pages if p['slug']!='real-data']+[{'slug':'real-data','title':'Real wildfire observations: look, compare, explain'}]
    (source/'pages.json').write_text(json.dumps(pages,indent=2)+'\n')
    # Insert one complete real figure at the relevant lesson step, plus the gallery link.
    import re
    targets={'forecast':('step1','timeline'),'prediction':('step1','channels'),'evaluation':('step1','persistence'),'missingness':('step1','missingness')}
    for slug,(step,name) in targets.items():
        path=ROOT/f'site-content/en/lessons/{slug}.html'
        text=path.read_text()
        text=re.sub(r'<!-- real-data-start -->.*?<!-- real-data-end -->','',text,flags=re.S)
        block='<!-- real-data-start --><h3>Look at a real event</h3>'+figure(first,name)+'<p><a href="../../handbook/student-real-data/index.html">Compare the real-data views</a> · '+escape(first['id'])+' · '+first['dates'][-2]+' → '+first['dates'][-1]+'</p><!-- real-data-end -->'
        pattern=r'(<section\b[^>]*\bid="'+step+r'"[^>]*>.*?</h2>)'
        text,n=re.subn(pattern,lambda m:m.group(1)+block,text,count=1,flags=re.S)
        if n!=1:raise ValueError(slug)
        path.write_text(text)
    # The fire-mask panel belongs beside the binary-channel explanation.
    path=ROOT/'site-content/en/lessons/prediction.html'
    text=path.read_text();marker='<!-- real-mask-start -->'
    text=re.sub(r'<!-- real-mask-start -->.*?<!-- real-mask-end -->','',text,flags=re.S)
    pos=text.index('</section>')
    text=text[:pos]+marker+figure(first,'fire-mask')+'<!-- real-mask-end -->'+text[pos:];path.write_text(text)
    print(f'Prepared {sum(len(e["figures"]) for e in manifest["events"])} real-data figures, gallery and lesson figures. Rebuild the course to publish HTML.')

if __name__=='__main__':main()
