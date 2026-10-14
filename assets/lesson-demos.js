/* Browser-only teaching examples. No model is trained here. */
(() => {
  'use strict';
  if (document.getElementById('score-case')) {
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
  }
  document.querySelectorAll('[data-quiz], [data-repair]').forEach(select => {
    select.addEventListener('change', () => {
      const feedbackId = select.dataset.repair ? `feedback-${select.dataset.repair}` : `${select.id}-feedback`;
      const feedback = document.getElementById(feedbackId);
      const option = select.selectedOptions[0];
      feedback.textContent = select.value === '' ? '' : `${select.value === select.dataset.answer ? 'Correct. ' : 'Reconsider. '}${option.dataset.reason} Open the reasoning below to compare every option.`;
    });
  });
  const byId = id => document.getElementById(id);
  function bind(id, event, fn) {
    const control = byId(id);
    if (control) { control.addEventListener(event, () => fn(control.value)); fn(control.value); }
  }
  function cells(id, values, hidden = []) {
    const nodes = byId(id).querySelectorAll('span');
    nodes.forEach((node, i) => {
      node.textContent = hidden.includes(i) ? '?' : String(values[i]);
      node.classList.toggle('obscured', hidden.includes(i));
      node.classList.toggle('positive', !hidden.includes(i) && values[i] === 1);
    });
  }
  bind('forecast-day', 'change', value => {
    const day = Number(value);
    byId('forecast-timeline').querySelectorAll('span').forEach((node, i) => {
      const n = i + 1;
      node.textContent = `Day ${n} · ${n < day ? 'past' : n === day ? 'last input' : n === day + 1 ? 'target' : 'future'}`;
      node.classList.toggle('selected', n === day);
      node.classList.toggle('target', n === day + 1);
    });
    byId('forecast-readout').textContent = `With Day ${day} as the last input, predict Day ${day + 1}. The Day ${day + 1} fire label cannot enter the input.`;
  });
  bind('head-weight', 'input', value => {
    const weight = Number(value), logit = .8 * weight + .2 - .5;
    const probability = 1 / (1 + Math.exp(-logit));
    byId('weight-value').textContent = weight.toFixed(1);
    byId('logit-value').textContent = logit.toFixed(2);
    byId('probability-value').textContent = probability.toFixed(3);
    byId('head-readout').textContent = `At threshold 0.5, this pixel is predicted as ${probability > .5 ? 'fire' : 'background'}. Changing a teaching weight is not training.`;
  });
  const probability = [.05,.90,.70,.10,.25,.80,.40,.30,.15,.60,.55,.20,.02,.35,.12,.01];
  const truth = probability.map((_, i) => [1,5,6,10].includes(i) ? 1 : 0);
  bind('metric-threshold', 'input', value => {
    const threshold = Number(value), mask = probability.map(p => p > threshold ? 1 : 0);
    let tp = 0, fp = 0, fn = 0, tn = 0;
    mask.forEach((v, i) => { if (truth[i]) { if (v) tp++; else fn++; } else { if (v) fp++; else tn++; } });
    const ratio = (n,d) => d ? (n/d).toFixed(3) : 'undefined';
    cells('metric-mask', mask);
    byId('metric-counts').textContent = `τ=${threshold.toFixed(2)} · TP=${tp} · FP=${fp} · FN=${fn} · TN=${tn}`;
    byId('metric-values').textContent = `Precision=${ratio(tp,tp+fp)} · Recall=${ratio(tp,tp+fn)} · F1=${ratio(2*tp,2*tp+fp+fn)} · IoU=${ratio(tp,tp+fp+fn)}`;
  });
  bind('mask-scenario', 'change', value => {
    const blocks = value === 'M06' ? [0,1,4,5] : value === 'M07' ? [0,1,4,5,8,9,12,13] : [];
    cells('mask-fire', truth, value === 'M01' ? Array.from({length:16},(_,i)=>i) : blocks);
    cells('mask-dynamic', truth.map((_,i)=>(.2+(i%4)*.1).toFixed(1)), blocks);
    const texts = {M00:'Complete inputs.', M01:'All fire history hidden; other dynamic and static inputs retained.', M06:'25% of the illustrative dynamic grid hidden; static inputs retained.', M07:'50% of the illustrative dynamic grid hidden; static inputs retained.'};
    byId('mask-readout').textContent = `${value}: ${texts[value]} The target remains unchanged.`;
  });
  bind('schedule-progress', 'input', value => {
    const steps = Number(value), multiplier = .5 * (1 + Math.cos(Math.PI * steps / 3000));
    byId('cosine-value').textContent = `${multiplier.toFixed(3)}×`;
    byId('cosine-bar').style.width = `${multiplier*100}%`;
    byId('schedule-readout').textContent = `After ${steps.toLocaleString('en-US')} of 3,000 updates: cosine multiplier = ${multiplier.toFixed(3)}.`;
  });
  bind('route-scenario', 'change', value => {
    const routes = {M00:['Matched ERM continuation','The clean score is ERM’s score because the route selects ERM here.'], M01:['Cosine-schedule continuation (X22)','This role supplies the no-fire-history prediction.'], M06:['25% block specialist (X17)','The mild fixed-severity specialist supplies this block condition.'], M07:['50% block specialist (X17)','The severe fixed-severity specialist supplies this block condition.']};
    byId('route-model').textContent = `${routes[value][0]} → ${value}`;
    byId('route-explanation').textContent = routes[value][1];
  });
  const search = byId('experiment-search');
  if (search) search.addEventListener('input', () => {
    const query = search.value.trim().toLowerCase();
    const archive = byId('historical-experiments');
    if (archive && query) archive.open = true;
    const cards = Array.from(document.querySelectorAll('[data-experiment]'));
    let shown = 0;
    cards.forEach(card => { card.hidden = !card.dataset.search.includes(query); if (!card.hidden) shown++; });
    byId('experiment-count').textContent = `${shown} of ${cards.length} entries${shown ? '' : ' · Try a name, ID, family or scenario.'}`;
  });
})();
