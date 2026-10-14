/* Static content is complete; this adds section navigation and local reminders. */
(() => {
  'use strict';
  const slugs = ['forecast', 'prediction', 'evaluation', 'fold2', 'missingness', 'erm', 'schedule', 'specialists'];
  const page = document.body.dataset.page || 'home';
  const prefix = document.body.dataset.root || '';
  const panels = Array.from(document.querySelectorAll('.step-panel'));
  const directory = document.querySelector('[data-directory]');
  if (directory && window.matchMedia('(max-width: 720px)').matches) directory.open = false;
  document.documentElement.classList.add('enhanced');
  let storageAvailable = true;
  function read(key) {
    try {
      const item = JSON.parse(window.localStorage.getItem(key) || '{}');
      return item && typeof item === 'object' && !Array.isArray(item) ? item : {};
    } catch (_) { storageAvailable = false; return {}; }
  }
  function write(key, value) {
    try { window.localStorage.setItem(key, JSON.stringify(value)); }
    catch (_) { storageAvailable = false; }
  }
  if (!panels.length) {
    function revealAnchor() {
      let id; try { id = decodeURIComponent(window.location.hash.slice(1)); } catch (_) { return; }
      let target = document.getElementById(id);
      if (!target) return;
      let parent = target;
      while (parent) { if (parent.tagName === 'DETAILS') parent.open = true; parent = parent.parentElement; }
    }
    revealAnchor();
    window.addEventListener('hashchange', revealAnchor);
    document.querySelectorAll('[data-resume-page]').forEach(link => {
      const slug = link.dataset.resumePage;
      const item = read(slug === 'erm' ? 'wildfire-erm-lesson-v1' : `wildfire-${slug}-lesson-v1`);
      if (slugs.includes(slug) && /^step[1-7]$/.test(item.step)) link.href = `${prefix}lessons/${slug}/index.html#${item.step}`;
      else link.hidden = true;
    });
    const resume = document.querySelector('[data-continue]');
    let saved = read('wildfire-course-last-v1');
    if (!slugs.includes(saved.page)) {
      const legacy = read('wildfire-erm-lesson-v1');
      saved = typeof legacy.step === 'string' ? { page: 'erm', step: legacy.step } : {};
    }
    if (resume && slugs.includes(saved.page) && /^step[1-7]$/.test(saved.step)) {
      resume.href = `${prefix}lessons/${saved.page}/index.html#${saved.step}`;
      resume.textContent = `Continue lesson ${slugs.indexOf(saved.page) + 1} →`;
    }
    return;
  }
  const storageKey = page === 'erm' ? 'wildfire-erm-lesson-v1' : `wildfire-${page}-lesson-v1`;
  let stored = read(storageKey);
  const valid = id => panels.some(panel => panel.id === id);
  if (!valid(stored.step)) stored.step = panels[0].id;
  stored.complete = stored.complete === true;
  const stepLinks = Array.from(document.querySelectorAll('.step-nav a'));
  const progress = document.getElementById('progress-status');
  const completion = document.getElementById('lesson-complete');
  let current = panels[0].id;
  let printing = false;
  let detailStates = [];

  function save() {
    stored.step = current;
    write(storageKey, stored);
    if (slugs.includes(page)) write('wildfire-course-last-v1', { page, step: current });
    if (progress) progress.textContent = `Section ${panels.findIndex(p => p.id === current) + 1} of ${panels.length}${stored.complete ? ' · Reviewed' : ''}. ${storageAvailable ? 'Position saved on this browser.' : 'Progress is temporary; browser storage is unavailable.'}`;
    if (completion) completion.checked = stored.complete;
  }
  function show(id, focus = false, target = null) {
    current = valid(id) ? id : panels[0].id;
    panels.forEach(panel => { panel.hidden = !printing && panel.id !== current; });
    stepLinks.forEach(link => {
      if (link.getAttribute('href') === `#${current}`) link.setAttribute('aria-current', 'step');
      else link.removeAttribute('aria-current');
    });
    const selected = stepLinks.find(link => link.getAttribute('href') === `#${current}`);
    const label = document.getElementById('section-label');
    if (label && selected) label.textContent = `Contents · ${selected.textContent}`;
    save();
    if (focus) {
      const panel = panels.find(p => p.id === current);
      const heading = target && target !== panel ? target : panel.querySelector('h2');
      if (heading) { heading.setAttribute('tabindex', '-1'); heading.focus({ preventScroll: true }); heading.scrollIntoView({ block: 'start' }); }
    }
  }
  function followLocation(focus) {
    let hash;
    try { hash = decodeURIComponent(window.location.hash.slice(1)); }
    catch (_) { hash = ''; }
    const target = document.getElementById(hash);
    if (target) {
      let ancestor = target;
      while (ancestor) { if (ancestor.tagName === 'DETAILS') ancestor.open = true; ancestor = ancestor.parentElement; }
    }
    const panel = target && target.closest('.step-panel');
    if (panel) show(panel.id, focus, target);
    else if (!hash) show(panels[0].id, focus);
    else if (!target) show(panels[0].id, false);
    // Known anchors outside sections (e.g. Skip to content) stay outside them.
  }
  show(panels[0].id);
  followLocation(false);
  window.addEventListener('hashchange', () => followLocation(true));
  if (completion) completion.addEventListener('change', () => { stored.complete = completion.checked; save(); });
  const reset = document.getElementById('reset-progress');
  if (reset) reset.addEventListener('click', () => {
    stored = { complete: false, step: panels[0].id };
    try { window.localStorage.removeItem(storageKey); } catch (_) { storageAvailable = false; }
    show(panels[0].id, true);
    window.location.hash = panels[0].id;
  });
  function preparePrint() {
    if (printing) return;
    printing = true;
    detailStates = Array.from(document.querySelectorAll('main details')).map(detail => [detail, detail.open]);
    detailStates.forEach(([detail]) => { detail.open = true; });
    panels.forEach(panel => { panel.hidden = false; });
  }
  function restorePrint() {
    if (!printing) return;
    printing = false;
    detailStates.forEach(([detail, open]) => { detail.open = open; });
    show(current);
  }
  window.addEventListener('beforeprint', preparePrint);
  window.addEventListener('afterprint', restorePrint);
  const print = document.getElementById('print-lesson');
  if (print) print.addEventListener('click', () => { preparePrint(); window.print(); });
})();
