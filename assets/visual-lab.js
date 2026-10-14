/* Inspect existing data exports and reason about experiment design. No inference. */
(() => {
  'use strict';
  const views = {
    timeline:['Dates and target','Which of the three dates is the forecast target?','The rightmost date, 2021-02-17. The preceding observations may enter the model; that next-day mask is the answer used for supervision or scoring.'],
    'fire-mask':['Observation to binary mask','Does a larger raw active-fire value mean a stronger fire?','No. This channel encodes detection time after preprocessing. The binary rule value > 0 records a positive detection, not its intensity.'],
    channels:['Feature channels','Which panel is much coarser than the terrain, and why should you notice?','The wind-speed panel has large blocks. Features aligned to one grid can still have different source resolutions. Resampling does not create new fine-scale observations.'],
    missingness:['Missing inputs','Does a gray cell mean the model observed no fire there?','No. Gray is a display overlay for deliberately hidden input. Pale pixels represent the benchmark background encoding. The target is not hidden by this experiment.'],
    persistence:['Forecast errors','What do purple pixels tell you about copying yesterday’s mask?','They are missed next-day positives under persistence. Orange marks false alarms and blue marks correct positive predictions. This demonstrates an error pattern, not neural-model performance.']
  };
  function assess(start,budget,selection) {
    const issues=[];
    if(start!=='b3') issues.push('Starting from ERM adds its prior training; the branches no longer share their starting weights.');
    if(budget!=='3000') issues.push('The candidate gets twice the continuation budget; the schedule is not the only change.');
    if(selection!=='final') issues.push('Best-validation selection differs from ERM’s final-step rule.');
    return issues;
  }
  const costs={mixed:1,fixed:2,route:4,attribution:5};
  window.CourseVisuals={assess,costs};
  document.querySelectorAll('[data-real-lab]').forEach(lab=>{
    const q=selector=>lab.querySelector(selector);
    function refresh(){
      const event=q('[data-lab-event]').value;
      const view=q('[data-lab-view]').value;
      if(!['fire_24935867','fire_24935874'].includes(event)||!views[view])return;
      const info=views[view];
      q('[data-lab-image]').src=`${document.body.dataset.root || ''}assets/real-examples/${event}/${view}.png`;
      q('[data-lab-image]').alt=`${info[0]} for ${event}, input 2021-02-16 and target 2021-02-17`;
      q('[data-lab-prompt]').textContent=info[1];
      q('[data-lab-explanation]').textContent=info[2];
      q('[data-lab-answer]').open=false;
      q('[data-lab-caption]').textContent=`${event} · input 2021-02-16 → target 2021-02-17 · ${info[0]}. These dataset records may overlap; they are not independent fire cases.`;
      q('[data-lab-zoom]').value='1';zoom();
    }
    function zoom(){
      const level=Math.max(1,Math.min(4,Number(q('[data-lab-zoom]').value)||1));
      q('[data-lab-image]').style.width=`${100*level}%`;
      q('[data-lab-zoom]').setAttribute('aria-valuetext',`${level} times magnification`);
    }
    q('[data-lab-event]').addEventListener('change',refresh);
    q('[data-lab-view]').addEventListener('change',refresh);
    q('[data-lab-zoom]').addEventListener('input',zoom);
    q('[data-lab-reset]').addEventListener('click', () => {
      q('[data-lab-event]').value='fire_24935867';
      q('[data-lab-view]').value=lab.dataset.initialView;
      refresh();
      q('.lab-explore').open=false;
      window.CourseState?.save();
    });
    refresh();
  });
  const get=id=>document.getElementById(id);
  if(get('contract-start')) {
    function update(){
      const start=get('contract-start').value,budget=get('contract-budget').value,selection=get('contract-selection').value;
      const issues=assess(start,budget,selection);
      get('contract-path').textContent=`${start==='b3'?'B3':'ERM final checkpoint'} → fresh optimizer → cosine rate`;
      get('contract-summary').textContent=`${Number(budget).toLocaleString('en-US')} updates · ${selection==='final'?'final checkpoint':'best-validation checkpoint'}`;
      get('contract-feedback').textContent=issues.length?`${issues.length} comparison problem${issues.length===1?'':'s'}: ${issues.join(' ')}`:'These three choices match. Still verify data, masks, seed, objective, augmentation and optimizer settings before attributing any result.';
      document.querySelector('[data-contract-card]').classList.toggle('contract-invalid',issues.length>0);
    }
    ['contract-start','contract-budget','contract-selection'].forEach(id=>get(id).addEventListener('change',update));update();
  }
  if(get('route-cost')){
    const names={M00:'ERM',M01:'cosine',M06:'25% expert',M07:'50% expert'};
    function route(){
      const scenario=get('route-scenario').value;
      document.querySelectorAll('[data-route-node]').forEach(n=>{
        n.classList.toggle('selected',n.dataset.routeNode===scenario);
        n.setAttribute('aria-current',n.dataset.routeNode===scenario?'true':'false');
      });
      get('route-active-count').textContent=`Selected now: ${names[scenario]}. One model is used for this condition; the complete route retains four trained models.`;
    }
    function cost(){
      const count=costs[get('route-cost').value]||4;
      get('route-cost-readout').textContent=`${count} model${count===1?'':'s'} × 3,000 = ${(count*3000).toLocaleString('en-US')} continuation updates.`;
      get('route-cost-bar').style.width=`${count*20}%`;
    }
    get('route-scenario').addEventListener('change',route);
    get('route-cost').addEventListener('change',cost);route();cost();
  }
})();
