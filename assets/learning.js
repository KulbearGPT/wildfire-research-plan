/* Progressive enhancement: the complete lesson remains readable without JS. */
(() => {
  'use strict';
  const storageKey = 'wildfire-erm-lesson-v1';
  const validStep = value => typeof value === 'string' && /^step[1-7]$/.test(value);
  let stored = {};
  let storageAvailable = true;
  try {
    const value = JSON.parse(window.localStorage.getItem(storageKey) || '{}');
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      stored = { step: validStep(value.step) ? value.step : 'step1', complete: value.complete === true };
    }
  } catch (_) { storageAvailable = false; }

  const directory = document.querySelector('[data-directory]');
  if (directory && window.matchMedia('(max-width: 720px)').matches) directory.open = false;
  document.documentElement.classList.add('enhanced');

  if (document.body.dataset.page !== 'erm') {
    const resume = document.querySelector('[data-continue]');
    if (resume && validStep(stored.step)) {
      resume.href = `lessons/erm/index.html#${stored.step}`;
      resume.textContent = stored.complete ? 'Revisit the ERM lesson →' : 'Continue the ERM lesson →';
    }
    return;
  }

  const panels = Array.from(document.querySelectorAll('.step-panel'));
  const stepLinks = Array.from(document.querySelectorAll('.step-nav a'));
  const progress = document.getElementById('progress-status');
  const completion = document.getElementById('lesson-complete');
  let current = 'step1';
  let printing = false;
  let detailStates = [];

  function updateProgress() {
    progress.textContent = `Step ${current.slice(-1)} of 7${stored.complete ? ' · Concepts reviewed' : ''}. ${storageAvailable ? 'Position saved on this browser.' : 'Progress is temporary; browser storage is unavailable.'}`;
    completion.checked = stored.complete === true;
  }
  function save() {
    stored.step = current;
    try { window.localStorage.setItem(storageKey, JSON.stringify(stored)); }
    catch (_) { storageAvailable = false; }
    updateProgress();
  }
  function showStep(id, focus) {
    current = validStep(id) ? id : 'step1';
    panels.forEach(panel => { panel.hidden = !printing && panel.id !== current; });
    stepLinks.forEach(link => {
      if (link.getAttribute('href') === `#${current}`) link.setAttribute('aria-current', 'step');
      else link.removeAttribute('aria-current');
    });
    save();
    if (focus) {
      const heading = document.querySelector(`#${current} h2`);
      heading.focus({ preventScroll: true });
      document.querySelector('.lesson-header').scrollIntoView({ block: 'start' });
    }
  }
  function locationStep() {
    const hash = window.location.hash.slice(1);
    return validStep(hash) ? hash : 'step1';
  }
  // An explicit URL wins over stored progress. The home page offers resume.
  showStep(locationStep(), false);
  window.addEventListener('hashchange', () => showStep(locationStep(), true));
  completion.addEventListener('change', () => { stored.complete = completion.checked; save(); });
  document.getElementById('reset-progress').addEventListener('click', () => {
    stored = { complete: false, step: 'step1' };
    try { window.localStorage.removeItem(storageKey); }
    catch (_) { storageAvailable = false; }
    // Other lessons or sites' saved data are never cleared.
    if (window.location.hash === '#step1') showStep('step1', true);
    else window.location.hash = 'step1';
  });

  document.querySelectorAll('[data-prediction]').forEach(button => {
    button.addEventListener('click', () => {
      document.getElementById('prediction-feedback').textContent = button.dataset.prediction === 'no'
        ? 'Exactly: the additional training could explain the gain. Reveal what ordinary continuation achieved.'
        : 'The total score improved, but the schedule and training budget both changed. We still need ordinary continuation as a control.';
      document.getElementById('missing-control').open = true;
    });
  });

  const examples = {
    loses: { a: 0.40, b: 0.46, c: 0.44, text: 'The candidate improves over B3, but ordinary continuation performs better. This case does not support a positive schedule effect.' },
    wins: { a: 0.40, b: 0.46, c: 0.49, text: 'The candidate beats the matched ERM control by 0.03 in this teaching case. In a real experiment, one seed would still be a pilot, not a confirmed gain.' },
    negative: { a: 0.40, b: 0.38, c: 0.37, text: 'Both continuations lose to B3, and the candidate also loses to ERM. A valid negative result is still useful evidence; do not extend training after seeing it.' }
  };
  const signed = value => `${value < 0 ? '−' : '+'}${Math.abs(value).toFixed(2)}`;
  document.getElementById('score-case').addEventListener('change', event => {
    const item = examples[event.target.value] || examples.loses;
    document.querySelectorAll('[data-score]').forEach(node => { node.textContent = item[node.dataset.score].toFixed(2); });
    const delta = { continuation: item.b - item.a, schedule: item.c - item.b, total: item.c - item.a };
    document.querySelectorAll('[data-delta]').forEach(node => { node.textContent = signed(delta[node.dataset.delta]); });
    document.getElementById('score-interpretation').textContent = item.text;
  });
  const repairText = {
    budget: ['Correct. Match 3,000 updates and starting weights; the intended difference is the schedule.', 'The training budgets differ. A higher score or AP per update cannot isolate the schedule.'],
    start: ['Correct. Both branches load the same B3 bytes independently.', 'A seed or filename cannot undo the additional training in ERM’s final checkpoint.'],
    checkpoint: ['Correct. Use the predeclared final-step selection rule for both continuations.', 'Selection changes the comparison. Do not choose a favorable rule or use test scores to select checkpoints.']
  };
  document.querySelectorAll('[data-repair]').forEach(select => {
    select.addEventListener('change', () => {
      document.getElementById(`feedback-${select.dataset.repair}`).textContent = select.value
        ? repairText[select.dataset.repair][select.value === 'right' ? 0 : 1] : '';
    });
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
    showStep(current, false);
  }
  window.addEventListener('beforeprint', preparePrint);
  window.addEventListener('afterprint', restorePrint);
  document.getElementById('print-lesson').addEventListener('click', () => { preparePrint(); window.print(); });
})();
