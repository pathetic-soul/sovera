// Shared helpers. Loaded first; every later panel script depends on these.
// Plain globals rather than ES modules: no bundler, no import map, no build
// step (AGENTS.md §7) — load order in index.html is the whole dependency graph.

const $ = id => document.getElementById(id);

// Escapes text interpolated into innerHTML. Audit payloads, router reasons,
// model output and the firewall ruleset all pass through here — none of it is
// trusted markup.
const esc = s => String(s).replace(/[&<>"']/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const reducedMotion = () =>
  window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// Comma-separated attachment inputs (router + agent panels). The §9.2 image
// override is unreachable if this list is ever hardcoded empty.
const splitList = id => $(id).value.split(',').map(s => s.trim()).filter(Boolean);

// One decision card shared by the router panel and the agent run channel.
// The reason string IS the explanation rendered while a model swaps (§4.2.3).
function routeCardHTML(d) {
  return '<div class="rcrow">'
    + `<span class="rmodel">${esc(d.model_id)}</span>`
    + (d.swap_required
        ? '<span class="rswap">model swap</span>'
        : '<span class="rhot">already resident</span>')
    + `<span class="rmeta">${esc(d.modality)} · ~${d.est_ctx} ctx</span></div>`
    + `<div class="rreason">${esc(d.reason)}</div>`;
}

// Console clock: one text node per second, no compositing. A demo runs for
// minutes and a clock makes liveness legible without inventing motion.
(function () {
  const el = $('clock');
  if (!el) return;
  const tick = () => {
    const d = new Date();
    const p = n => String(n).padStart(2, '0');
    el.textContent = `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
  };
  tick();
  setInterval(tick, 1000);
})();

