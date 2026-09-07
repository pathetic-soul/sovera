// The router panel (§9.2). Deterministic, loads nothing — which is why the
// decision can render here before the weights it selected finish loading
// (§4.2.3 swap masking).

async function doRoute() {
  const text = $('prompt').value.trim();
  if (!text) return;
  const attachments = splitList('attach');
  $('decision').hidden = false;
  $('decision').className = 'routecard';
  $('decision').innerHTML = '<div class="rreason dim">routing…</div>';

  const d = await (await fetch('/api/route', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, attachments })
  })).json();

  $('decision').className = 'routecard ok';
  $('decision').innerHTML = routeCardHTML(d);

  const ranked = Object.entries(d.scores).sort((a, b) => b[1] - a[1]);
  const top = ranked[0][1] || 1;
  $('scoretbl').innerHTML = ranked.map(([k, v]) =>
    `<tr${k === d.task_type ? ' class="win"' : ''}>`
    + `<td class="sk">${k}</td><td class="sv">${v.toFixed(3)}</td>`
    + `<td><span class="sbar"><i style="width:${Math.max(1, 100 * v / top)}%"></i></span></td></tr>`
  ).join('');
}

