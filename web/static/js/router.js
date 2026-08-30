// The router panel (AGENTS.md §9.2). The `reason` string is rendered verbatim
// because it IS the explanation shown while a model swaps (§4.2.3) — a stall
// with a rationale on screen reads as an explanation, not a hang.

async function doRoute() {
  const text = $('prompt').value.trim();
  if (!text) return;
  const attachments = $('attach').value.split(',').map(s => s.trim()).filter(Boolean);
  $('decision').textContent = 'routing…';
  const d = await (await fetch('/api/route', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text, attachments})
  })).json();

  // The reason string IS the explanation shown during a model swap (AGENTS.md 4.2.3).
  $('decision').className = 'ok';
  $('decision').innerHTML = `<b>${esc(d.model_id)}</b>`
    + (d.swap_required ? ' <span class="warn">[swap]</span>' : '')
    + `<br><span class="dim">${esc(d.reason)}</span>`;

  const ranked = Object.entries(d.scores).sort((a, b) => b[1] - a[1]);
  const top = ranked[0][1] || 1;
  $('scoretbl').innerHTML = ranked.map(([k, v]) =>
    `<tr><td style="width:9em">${k}</td><td style="width:4em">${v.toFixed(3)}</td>`
    + `<td><div style="height:8px;width:${Math.max(1, 100 * v / top)}%;`
    + `background:${k === d.task_type ? 'var(--ok)' : 'var(--line)'}"></div></td></tr>`
  ).join('');
}
