"""Reusable browser interactions based on existing, labelled image exports."""
from html import escape
import json
from course_shell import ROOT

VIEWS = {
    'timeline': ('Dates and target', 'Which of the three dates is the forecast target?', 'The rightmost date, 2021-02-17. The preceding observations may enter the model; that next-day mask is the answer used for supervision or scoring.'),
    'fire-mask': ('Observation to binary mask', 'Does a larger raw active-fire value mean a stronger fire?', 'No. This channel encodes detection time after preprocessing. The binary rule value > 0 records a positive detection, not its intensity.'),
    'channels': ('Feature channels', 'Which panel is much coarser than the terrain, and why should you notice?', 'The wind-speed panel has large blocks. Features aligned to one grid can still have different source resolutions. Resampling does not create new fine-scale observations.'),
    'missingness': ('Missing inputs', 'Does a gray cell mean the model observed no fire there?', 'No. Gray is a display overlay for deliberately hidden input. Pale pixels represent the benchmark background encoding. The target is not hidden by this experiment.'),
    'persistence': ('Forecast errors', 'What do purple pixels tell you about copying yesterday’s mask?', 'They are missed next-day positives under persistence. Orange marks false alarms and blue marks correct positive predictions. This demonstrates an error pattern, not neural-model performance.')}


def explorer(slug, view):
    events = json.loads((ROOT/'assets/real-examples/manifest.json').read_text())['events']
    ident = 'real-lab-'+slug
    options = ''.join(f'<option value="{escape(e["id"])}">{escape(e["id"])} · 16 → 17 Feb 2021</option>' for e in events)
    choices = ''.join(f'<option value="{key}"'+(' selected' if key==view else '')+f'>{value[0]}</option>' for key,value in VIEWS.items())
    prompt, answer = VIEWS[view][1:]
    instructions = {
        'timeline': 'Read the dates from left to right. Point to the last legal input, then the next-day target. Stop when you can explain which mask must stay outside the input.',
        'channels': 'Compare the terrain and wind panels on the same patch. Find a coarse block in the wind panel, then explain why a shared grid does not imply equal source resolution.',
        'missingness': 'Compare the complete input, gray hidden regions and target. Point out one hidden region without treating it as observed background.',
        'persistence': 'Locate one purple missed-fire region and one orange false alarm. Explain why simply copying yesterday can make both errors.',
        'fire-mask': 'Compare one positive observation with its binary-mask pixel. Explain why the mask records detection rather than fire intensity.'}

    path = '../../assets/real-examples/'+events[0]['id']+'/'+view+'.png'
    return f'''<div class="visual-lab" data-real-lab data-initial-view="{view}">
<p class="eyebrow">Explore real observations · no model runs in the browser</p>
<h3>Your observation task</h3><p class="guided-observation">{instructions[view]}</p>

<p data-lab-prompt class="lab-question">{prompt}</p>
<div class="lab-viewport" tabindex="0" role="region" aria-label="Real observation figure; scroll when magnified"><img data-lab-image src="{path}" alt="{VIEWS[view][0]} for {events[0]['id']}, input 2021-02-16 and target 2021-02-17" loading="lazy"></div>
<p data-lab-caption class="source">{events[0]['id']} · input 2021-02-16 → target 2021-02-17 · {VIEWS[view][0]}. These dataset records may overlap; they are not independent fire cases.</p>
<details data-lab-answer><summary>Check my interpretation</summary><p data-lab-explanation>{answer}</p></details>
<details class="lab-explore" id="{ident}-explore"><summary>After this task: explore other views</summary><p>Optional exploration. Return to the task view whenever you want to continue the lesson.</p><div class="lab-controls js-only"><label for="{ident}-event">Dataset record<select id="{ident}-event" data-lab-event>{options}</select></label><label for="{ident}-view">View<select id="{ident}-view" data-lab-view>{choices}</select></label><label for="{ident}-zoom">Magnification<input id="{ident}-zoom" data-lab-zoom type="range" min="1" max="4" step="0.5" value="1"></label></div><button type="button" class="secondary js-only" data-lab-reset>Return to the task view</button></details>
<details open data-default-open><summary>Read the figure legends and limits</summary><p class="note">Mask panels: orange means detected fire, pale means encoded background, and gray means deliberately hidden input. Error panels use their own hit/false-alarm/miss legend. Feature panels show raw stored values with independent 2nd–98th percentile color limits, not natural-color photos. Zoom changes display size only. No probability values are inferred from PNG colors; exact metric counts are unavailable in this partial export.</p></details>
</div>'''


def contract_lab():
    return '''<div class="visual-lab" data-contract-lab><p class="eyebrow">Design exercise · no measured scores</p><h3>Build a comparison that can answer the schedule question.</h3><p>ERM is fixed: independently load B3, train 3,000 updates and select the final checkpoint. Configure the cosine branch.</p>
<div class="lab-controls js-only"><label for="contract-start">Candidate starting weights<select id="contract-start"><option value="b3">Same frozen B3</option><option value="erm">ERM final checkpoint</option></select></label><label for="contract-budget">Candidate updates<select id="contract-budget"><option value="3000">3,000</option><option value="6000">6,000</option></select></label><label for="contract-selection">Candidate checkpoint<select id="contract-selection"><option value="final">Final update</option><option value="best">Best validation score</option></select></label></div>
<div class="comparison-branches"><div class="branch-root">Same B3 foundation</div><div class="pair"><article class="score-card"><h4>Control</h4><p>B3 → fresh optimizer → constant rate</p><strong>3,000 updates · final checkpoint</strong></article><article class="score-card" data-contract-card><h4>Candidate</h4><p id="contract-path">B3 → fresh optimizer → cosine rate</p><strong id="contract-summary">3,000 updates · final checkpoint</strong></article></div></div>
<p class="feedback" id="contract-feedback" aria-live="polite">These three choices match. Still verify data, masks, seed, objective, augmentation and optimizer settings before attributing any result.</p><p class="done">Try making the comparison invalid in three different ways, then repair it. A matched plan can still produce a negative result.</p></div>'''


def route_lab():
    return '''<div class="visual-lab" data-route-lab><h3>Trace the selected expert and count what you trained.</h3><p class="eyebrow">Protocol routing · no automatic outage detector</p><div class="route-models"><div data-route-node="M00" class="route-node selected">ERM<br><small>Complete input</small></div><div data-route-node="M01" class="route-node">Cosine<br><small>No fire history</small></div><div data-route-node="M06" class="route-node">25% expert<br><small>Mild blocks</small></div><div data-route-node="M07" class="route-node">50% expert<br><small>Severe blocks</small></div></div><p id="route-active-count" aria-live="polite">Selected now: ERM. One model is used for this condition; the complete route retains four trained models.</p><label class="js-only" for="route-cost">Compare training scope<select id="route-cost"><option value="mixed">One mixed-severity expert</option><option value="fixed">Two fixed-severity experts</option><option value="route" selected>Complete four-model route</option><option value="attribution">Complete route plus mixed control</option></select></label><div class="budget-meter" aria-hidden="true"><span id="route-cost-bar" style="width:80%"></span></div><p id="route-cost-readout" aria-live="polite">4 models × 3,000 = 12,000 continuation updates.</p><p>Totals are for one history and seed, excluding foundation training and evaluation. Inference routing selects one model; it does not erase the cost of training and storing the others.</p></div>'''
