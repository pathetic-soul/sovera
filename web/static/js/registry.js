// The registry panel (AGENTS.md §8.1, §14.3). "Reload registry" is leg 3: edit
// config/models.yaml, press this, and a new model is routed to with no code
// change and no restart. Rejected models render with their reason, because an
// over-budget model being refused is as much the demo as one being accepted.

function renderRegistry(s) {
  $('profile').textContent = `profile ${s.profile}, budget ${s.budget_gb} GB, fallback ${s.fallback}`;
  $('modeltbl').innerHTML = s.models.map(m =>
    `<tr><td>${esc(m.id)}</td><td class="dim">${esc(m.ref)}</td><td>${m.device}</td>`
    + `<td>${m.vram_gb}</td><td class="dim">${esc(m.routes.join(' '))}</td></tr>`
  ).join('');
  $('rejected').innerHTML = s.rejected.length
    ? '<h2 style="margin-top:12px">Rejected</h2>' + s.rejected.map(r =>
        `<div class="rec egress">${esc(r.model_id)} — ${esc(r.reason)}</div>`).join('')
    : '';
}

async function reloadRegistry() {
  $('reloadmsg').textContent = 'reloading…';
  const r = await (await fetch('/api/registry/reload', {method: 'POST'})).json();
  $('reloadmsg').className = r.ok ? 'ok' : 'bad';
  $('reloadmsg').textContent = r.ok
    ? `${r.accepted.length} accepted, ${r.rejected.length} rejected`
    : `keeping previous registry — ${r.error}`;
  if (r.snapshot) renderRegistry(r.snapshot);
}

fetch('/api/registry').then(r => r.json()).then(renderRegistry);
