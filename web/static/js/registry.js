// The registry panel (§8.1, §14.3). "Reload registry" is leg 3: edit
// config/models.yaml, press this, and a new model is routed to with no code
// change and no restart. Rejected models render with their reason, because an
// over-budget model being refused is as much the demo as one being accepted.

function renderRegistry(s) {
  $('profile').textContent = `${s.profile} · budget ${s.budget_gb} GB · fallback ${s.fallback}`;
  $('modeltbl').innerHTML = s.models.map(m => {
    // Each model's VRAM claim drawn against the profile budget — the 5.2 GB
    // ceiling is the binding constraint (§4.1), so it is drawn, not stated.
    const pct = Math.min(100, 100 * m.vram_gb / (s.budget_gb || 1));
    return `<tr><td class="mid">${esc(m.id)}</td><td class="dim mref">${esc(m.ref)}</td>`
      + `<td><span class="dev ${m.device}">${m.device}</span></td>`
      + `<td class="vram"><span class="vbar"><i style="width:${pct}%"></i></span>`
      + `<span class="vnum">${m.vram_gb} GB</span></td>`
      + `<td class="dim mroutes">${esc(m.routes.join(' '))}</td></tr>`;
  }).join('');
  $('rejected').innerHTML = s.rejected.length
    ? '<h3 class="shead bad">rejected — visible, not silent</h3>'
      + s.rejected.map(r =>
          `<div class="rec reject"><b>${esc(r.model_id)}</b> — ${esc(r.reason)}</div>`).join('')
    : '';
}

async function reloadRegistry() {
  const btn = $('reloadbtn');
  if (btn.disabled) return;
  btn.disabled = true;
  $('reloadmsg').textContent = 'reloading…';
  $('reloadmsg').className = 'dim';
  try {
    const r = await (await fetch('/api/registry/reload', { method: 'POST' })).json();
    $('reloadmsg').className = r.ok ? 'ok' : 'bad';
    $('reloadmsg').textContent = r.ok
      ? `${r.accepted.length} accepted · ${r.rejected.length} rejected`
      : `keeping previous registry: ${r.error}`;
    if (r.snapshot) renderRegistry(r.snapshot);
  } finally {
    // A failed reload must not leave the demo's leg-3 button dead on stage.
    btn.disabled = false;
  }
}

fetch('/api/registry').then(r => r.json()).then(renderRegistry);

