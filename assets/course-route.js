/* Carry the originating task through supporting pages, including nested reading. */
(() => {
  'use strict';
  const map = window.CourseMap || {};
  function parse(value) {
    const m = /^([a-z]+)\.(step[0-9]+(?:-[a-z-]+)?)$/.exec(value || '');
    return m && Object.hasOwn(map,m[1]) && Object.hasOwn(map[m[1]],m[2]) ? {slug:m[1],step:m[2]} : null;
  }
  const returnURL = (root,route) => new URL(`lessons/${route.slug}/index.html#${route.step}`,root).href;
  window.CourseRoute = {parse,returnURL};
  const page = document.body.dataset.page;
  const root = new URL(document.body.dataset.root || './',window.location.href);
  const inherited = parse(new URL(window.location.href).searchParams.get('via'));
  if (!map[page] && inherited) {
    const task = map[inherited.slug][inherited.step];
    document.querySelectorAll('[data-course-return-link]').forEach(link => {
      link.href = returnURL(root,inherited);
      link.textContent = `Return to your task: ${task.title} →`;
      link.hidden = false;
    });
    const banner = document.querySelector('[data-course-return]');
    if (banner) banner.hidden = false;
    document.documentElement.classList.add('with-course-return');
    document.querySelectorAll('[data-reference-next]').forEach(n => {
      const link = n.querySelector('a');
      if (link) { link.classList.remove('button'); link.textContent = 'Optional: '+link.textContent; }
    });
  }
  function refresh() {
    let panel;
    try { panel = document.getElementById(decodeURIComponent(window.location.hash.slice(1)))?.closest('.step-panel'); } catch (_) {}
    if (!panel) panel = Array.from(document.querySelectorAll('.step-panel')).find(p => !p.hidden);
    const route = map[page] ? parse(`${page}.${panel?.id || 'step1'}`) : inherited;
    document.querySelectorAll('a[href]').forEach(link => {
      const raw = link.getAttribute('href');
      if (!raw || raw.startsWith('#') || link.hasAttribute('download') || link.hasAttribute('data-course-return-link')) return;
      let url; try { url = new URL(raw,window.location.href); } catch (_) { return; }
      if (!['http:','https:','file:'].includes(url.protocol)) return;
      const inside = url.origin === root.origin && url.pathname.startsWith(root.pathname);
      const relative = inside ? url.pathname.slice(root.pathname.length) : '';
      if (route && /^(handbook|library|experiments)\/.+\.html$/.test(relative)) {
        const parent = link.closest('.step-panel');
        const selected = map[page] && parent ? parse(`${page}.${parent.id}`) : route;
        if (selected) { url.searchParams.set('via',`${selected.slug}.${selected.step}`); link.href=url.href; }
      } else if ((!inside || /\.(png|jpg|jpeg|pdf|md|json|csv|py)$/.test(url.pathname)) && !link.closest('.pager')) {
        link.target = '_blank'; link.rel = 'noopener';
      }
    });
  }
  refresh();
  window.addEventListener('hashchange',refresh);
  window.addEventListener('pageshow',refresh);
})();
