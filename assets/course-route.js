/* A bounded lesson return route travels in the URL, independently of storage. */
(() => {
  'use strict';
  const steps = {forecast:5,prediction:5,evaluation:5,fold2:6,missingness:5,erm:7,schedule:5,specialists:5};
  function parse(value) {
    const m = /^([a-z]+)\.(step([1-7]))$/.exec(value || '');
    return m && steps[m[1]] && Number(m[3]) <= steps[m[1]] ? {slug:m[1],step:m[2]} : null;
  }
  function returnURL(root, route) {
    return new URL(`lessons/${route.slug}/index.html#${route.step}`,root).href;
  }
  window.CourseRoute = {parse,returnURL};
  const page = document.body.dataset.page;
  const root = new URL(document.body.dataset.root || './',window.location.href);
  const inherited = parse(new URL(window.location.href).searchParams.get('via'));
  const banner = document.querySelector('[data-course-return]');
  if (!steps[page] && inherited && banner) {
    const link = banner.querySelector('[data-course-return-link]');
    link.href = returnURL(root,inherited);
    link.textContent = `← Return to lesson ${Object.keys(steps).indexOf(inherited.slug)+1}, ${inherited.step.replace('step','step ')}`;
    banner.hidden = false;
    document.documentElement.classList.add('with-course-return');
  }
  function refresh() {
    let step = 'step1';
    try {
      const target = document.getElementById(decodeURIComponent(window.location.hash.slice(1)));
      const panel = target && target.closest('.step-panel');
      if (panel) step = panel.id;
    } catch (_) { /* malformed hashes use the first step */ }
    const route = steps[page] ? parse(`${page}.${step}`) : inherited;
    document.querySelectorAll('a[href]').forEach(link => {
      const raw = link.getAttribute('href');
      if (!raw || raw.startsWith('#') || link.hasAttribute('download') || link.hasAttribute('data-course-return-link')) return;
      let url; try {url = new URL(raw,window.location.href);} catch (_) {return;}
      if (!['http:','https:','file:'].includes(url.protocol)) return;
      const inside = url.origin === root.origin && url.pathname.startsWith(root.pathname);
      const relative = inside ? url.pathname.slice(root.pathname.length) : '';
      if (route && /^(handbook|library|experiments)\/.+\.html$/.test(relative)) {
        // Main lessons always supply the current step, even after a back navigation.
        const parent = link.closest('.step-panel');
        const selected = steps[page] && parent ? parse(`${page}.${parent.id}`) : route;
        url.searchParams.set('via',`${selected.slug}.${selected.step}`);
        link.href = url.href;
      } else if ((!inside || /\.(png|jpg|jpeg|pdf|md|json|csv|py)$/.test(url.pathname)) && !link.closest('.pager')) {
        link.target = '_blank';
        link.rel = 'noopener';
      }
    });
  }
  refresh();
  window.addEventListener('hashchange',refresh);
  window.addEventListener('pageshow',refresh);
})();
