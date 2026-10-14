/* Continuous reading with anchor navigation and a saved reading position. */
(() => {
  'use strict';
  const slugs = ['forecast','prediction','evaluation','fold2','missingness','erm','schedule','specialists'];
  const page = document.body.dataset.page || 'home';
  const prefix = document.body.dataset.root || '';
  const panels = Array.from(document.querySelectorAll('.step-panel'));
  document.documentElement.classList.add('enhanced');
  let storageAvailable = true;
  function read(key) {
    try {
      const value = JSON.parse(window.localStorage.getItem(key) || '{}');
      return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
    } catch (_) { storageAvailable=false; return {}; }
  }
  function write(key,value) {
    try { window.localStorage.setItem(key,JSON.stringify(value)); }
    catch (_) { storageAvailable=false; }
  }
  function reveal(target) {
    for (let n=target;n;n=n.parentElement) if (n.tagName==='DETAILS') n.open=true;
  }
  function targetFromHash() {
    try { return document.getElementById(decodeURIComponent(window.location.hash.slice(1))); }
    catch (_) { return null; }
  }
  if (!panels.length) {
    const update=()=>{
      reveal(targetFromHash());
      const resume=document.querySelector('[data-continue]');
      const saved=read('wildfire-course-last-v1');
      if(resume && window.CourseMap?.[saved.page]?.[saved.step]) {
        resume.href=`${prefix}lessons/${saved.page}/index.html#${saved.step}`;
        resume.textContent=`Continue learning: ${window.CourseMap[saved.page][saved.step].title} →`;
      }
    };
    update();window.addEventListener('hashchange',update);window.addEventListener('pageshow',update);
    return;
  }
  const key=`wildfire-${page}-lesson-v1`;
  let stored=read(key);
  const valid=id=>panels.some(p=>p.id===id);
  const savedStep=valid(stored.step)?stored.step:panels[0].id;
  let current=savedStep, printing=false, detailStates=[], scrollPending=false;
  window.CourseReading = {current:()=>current};
  const links=Array.from(document.querySelectorAll('.step-nav a'));
  const progress=document.getElementById('progress-status');
  const completion=document.getElementById('lesson-complete');
  panels.forEach(p=>{p.hidden=false});
  function save(id) {
    if (!valid(id)) return;
    current=id; stored.step=id; stored.complete=stored.complete===true;
    write(key,stored);
    if(slugs.includes(page)) write('wildfire-course-last-v1',{page,step:id});
    if(progress) progress.textContent=`Reading section ${panels.findIndex(p=>p.id===id)+1} of ${panels.length}${stored.complete?' · Reviewed':''}. ${storageAvailable?'Reading position saved on this browser.':'Progress is temporary; browser storage is unavailable.'}`;
    if(completion) completion.checked=stored.complete;
    links.forEach(link=>{
      if(link.getAttribute('href')===`#${id}`) link.setAttribute('aria-current','location');
      else link.removeAttribute('aria-current');
    });
  }
  function navigate(focus=false) {
    const target=targetFromHash();reveal(target);
    const panel=target?.closest('.step-panel');
    if(panel) save(panel.id);
    else if(!window.location.hash) save(current);
    if(focus && target) {
      const heading=target===panel?panel.querySelector('h2'):target;
      heading?.setAttribute('tabindex','-1');heading?.focus({preventScroll:true});
      target.scrollIntoView({block:'start'});
    }
  }
  save(savedStep);navigate();
  // Resume only an existing saved section, after deferred interactions settle.
  if(!window.location.hash && valid(stored.step) && savedStep!==panels[0].id) {
    window.history.replaceState(null,'',`#${savedStep}`);
    document.addEventListener('DOMContentLoaded',()=>document.getElementById(savedStep).scrollIntoView({block:'start'}));
  }
  function observeReading() {
    if(printing || scrollPending) return;
    scrollPending=true;
    window.requestAnimationFrame(()=>{
      scrollPending=false;if(printing)return;
      let selected=panels[0];
      for(const p of panels) {if(p.getBoundingClientRect().top<=160) selected=p;else break;}
      if(selected.id!==current) {
        save(selected.id);
        window.history.replaceState(null,'',`#${selected.id}`);
      }
    });
  }
  window.addEventListener('scroll',observeReading,{passive:true});
  window.addEventListener('hashchange',()=>navigate(true));
  window.addEventListener('pageshow',event=>{if(event?.persisted)save(current);else navigate()});
  if(completion)completion.addEventListener('change',()=>{stored.complete=completion.checked;save(current)});
  document.getElementById('reset-progress')?.addEventListener('click',()=>{
    window.CourseState?.reset();stored={complete:false,step:panels[0].id};save(panels[0].id);
    window.history.replaceState(null,'',`#${panels[0].id}`);navigate(true);
  });
  function preparePrint(){
    if(printing)return;printing=true;window.CourseState?.printing(true);
    detailStates=Array.from(document.querySelectorAll('main details')).map(n=>[n,n.open]);
    detailStates.forEach(([n])=>{n.open=true});
  }
  function restorePrint(){
    if(!printing)return;detailStates.forEach(([n,open])=>{n.open=open});
    printing=false;window.CourseState?.printing(false);
  }
  window.addEventListener('beforeprint',preparePrint);window.addEventListener('afterprint',restorePrint);
  document.getElementById('print-lesson')?.addEventListener('click',()=>{preparePrint();window.print()});
})();
