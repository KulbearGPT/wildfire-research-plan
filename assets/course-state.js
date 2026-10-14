/* Restore lesson controls before rendering, then replay their normal UI handlers. */
(() => {
  'use strict';
  const page = document.body.dataset.page;
  const key = `wildfire-interactions-${page}-v1`;
  let available = true, saved = {};
  try { saved = JSON.parse(localStorage.getItem(key) || '{}'); } catch (_) { available = false; }
  if (!saved || typeof saved !== 'object' || Array.isArray(saved)) saved = {};
  const controls = Array.from(document.querySelectorAll('main select[id], main input[id]'))
    .filter(n => n.id !== 'lesson-complete' && n.type !== 'search');
  const details = Array.from(document.querySelectorAll('main details'));
  const detailKey = n => n.id || `${n.closest('.step-panel')?.id || 'intro'}:${details.indexOf(n)}`;
  const defaults = controls.map(n => [n, n.type === 'checkbox' ? n.checked : n.value]);
  const detailDefaults = details.map(n => [n, n.open]);
  const valid = (n, value) => {
    if (n.tagName === 'SELECT') return Array.from(n.options).some(o => o.value === value);
    if (n.type === 'checkbox') return typeof value === 'boolean';
    if (n.type === 'range') return typeof value === 'string' && Number.isFinite(Number(value)) && Number(value) >= Number(n.min) && Number(value) <= Number(n.max);
    return false;
  };
  function set(n, value) { if (n.type === 'checkbox') n.checked = value; else n.value = value; }
  if (saved.controls && typeof saved.controls === 'object') controls.forEach(n => {
    if (valid(n, saved.controls[n.id])) set(n, saved.controls[n.id]);
  });
  const restored = controls.map(n => [n, n.type === 'checkbox' ? n.checked : n.value]);
  let restoring = true, printing = false;
  function persist() {
    if (restoring || printing) return;
    const snapshot = {controls: {}, details: {}, prediction: saved.prediction};
    controls.forEach(n => { snapshot.controls[n.id] = n.type === 'checkbox' ? n.checked : n.value; });
    details.forEach(n => { snapshot.details[detailKey(n)] = n.open; });
    saved = snapshot;
    try { localStorage.setItem(key, JSON.stringify(saved)); } catch (_) { available = false; }
  }
  function replay(values) {
    // Select changes can reset zoom; restore range values after those changes.
    [...values.filter(([n]) => n.tagName === 'SELECT'), ...values.filter(([n]) => n.tagName !== 'SELECT')].forEach(([n,v]) => {
      set(n,v); n.dispatchEvent(new Event(n.type === 'range' ? 'input' : 'change', {bubbles:true}));
    });
  }
  window.CourseState = {
    available: () => available,
    save: persist,
    printing(value) { printing = value; },
    reset() {
      restoring = true; saved = {}; replay(defaults);
      detailDefaults.forEach(([n,v]) => { n.open = v; });
      const feedback = document.getElementById('prediction-feedback');
      if (feedback) feedback.textContent = '';
      restoring = false; persist();
    }
  };
  document.addEventListener('DOMContentLoaded', () => {
    // An explicit practical deep link takes priority over a saved study mode.
    replay(restored.map(([n,v]) => [n, n.id === 'erm-mode' ? n.value : v]));
    if (saved.prediction === 'yes' || saved.prediction === 'no') document.querySelector(`[data-prediction="${saved.prediction}"]`)?.click();
    details.forEach(n => { if (typeof saved.details?.[detailKey(n)] === 'boolean') n.open = saved.details[detailKey(n)]; });
    // A direct anchor must stay visible even if this disclosure was saved closed.
    try {
      let target = document.getElementById(decodeURIComponent(window.location.hash.slice(1)));
      while (target) { if (target.tagName === 'DETAILS') target.open = true; target = target.parentElement; }
    } catch (_) {}
    restoring = false;
    controls.forEach(n => { n.addEventListener('change',persist); n.addEventListener('input',persist); });
    details.forEach(n => n.addEventListener('toggle',persist));
    document.querySelectorAll('[data-prediction]').forEach(n => n.addEventListener('click', () => { saved.prediction=n.dataset.prediction; persist(); }));
  });
  window.addEventListener('pagehide',persist);
})();
