// The agent panel (AGENTS.md §14.4, §14.5). One websocket per run, bidirectional
// because the §2.4 human gate suspends the loop until the reviewer answers on
// the same channel. Approve/Reject are wired to that reply.

let agentWs = null;

fetch('/api/backend').then(r => r.json()).then(b => {
  $('backend').className = b.ok ? 'ok' : 'bad';
  $('backend').textContent = b.ok ? `ollama up — ${b.note}` : `ollama down — ${b.note}`;
  $('autoapprove').checked = !!b.auto_approve;
  renderGateBanner();
});

// The §2.4 gate is a compliance control, so arming it off must never be quiet.
function renderGateBanner() {
  const on = $('autoapprove').checked;
  $('gatebanner').style.display = on ? '' : 'none';
  $('gatebanner').textContent = on
    ? '⚠ auto mode — writes and sandboxed code run without human approval. '
      + 'The audit log records these as granted_by: auto, not as human sign-off. '
      + 'Turn this off for the demo.'
    : '';
}

function addStep(cls, html) {
  const div = document.createElement('div');
  div.className = 'step ' + cls;
  div.innerHTML = html;
  $('trace').appendChild(div);
  $('trace').scrollTop = $('trace').scrollHeight;
  return div;
}

function runAgent() {
  const text = $('task').value.trim();
  if (!text || (agentWs && agentWs.readyState === WebSocket.OPEN)) return;
  $('trace').innerHTML = '';
  $('agentroute').textContent = '';
  $('budget').textContent = '';
  $('agentstat').textContent = 'connecting…';
  $('agentstat').className = 'warn';
  $('runbtn').disabled = true;

  const ws = new WebSocket(`ws://${location.host}/ws/agent`);
  agentWs = ws;
  ws.onopen = () => {
    $('agentstat').textContent = 'running';
    ws.send(JSON.stringify({text, attachments: [], auto_approve: $('autoapprove').checked}));
  };
  ws.onmessage = e => onAgentEvent(JSON.parse(e.data));
  ws.onclose = () => {
    $('runbtn').disabled = false;
    if ($('agentstat').textContent === 'running') $('agentstat').textContent = 'disconnected';
  };
}

function onAgentEvent(ev) {
  const d = ev.data;
  if (ev.type === 'route') {
    // Rendered before the first token is requested, so the 2-5 s model load
    // reads as an explanation rather than a stall (AGENTS.md 4.2.3).
    $('agentroute').className = 'ok';
    $('agentroute').innerHTML = `<b>${esc(d.model_id)}</b>`
      + (d.swap_required ? ' <span class="warn">[loading — swap]</span>'
                         : ' <span class="dim">[loading]</span>')
      + `<br><span class="dim">${esc(d.reason)}</span>`;
    return;
  }

  if (ev.type === 'approval_request') {
    const div = addStep('gate',
      `<b>step ${d.n}</b> <span class="tag">${esc(d.tool)}</span> `
      + (d.auto ? `<span class="warn">auto-approved (granted_by: auto)</span>`
                : `<span class="warn">awaiting human approval</span>`)
      + `<div class="dim">${esc(d.thought)}</div>`
      + `<div class="obs">${esc(JSON.stringify(d.args, null, 1))}</div>`
      + (d.auto ? '' : `<p style="margin:8px 0 0"><button class="go" data-yes>✓ Approve</button> `
                       + `<button data-no>✕ Reject</button></p>`));
    // Auto-granted steps never suspend the loop, so there is nothing to answer.
    if (d.auto) return;
    $('agentstat').textContent = 'waiting for approval';
    const answer = ok => {
      agentWs.send(JSON.stringify({approve: ok}));
      div.querySelector('p').innerHTML =
        ok ? '<span class="ok">approved</span>' : '<span class="bad">rejected</span>';
      $('agentstat').textContent = 'running';
    };
    div.querySelector('[data-yes]').onclick = () => answer(true);
    div.querySelector('[data-no]').onclick = () => answer(false);
    return;
  }

  if (ev.type === 'step' || ev.type === 'denied') {
    const links = (d.artifacts || []).map(a =>
      `<a href="/api/artifact?path=${encodeURIComponent(a)}">${esc(a)}</a>`).join(' ');
    addStep(ev.type === 'denied' ? 'deny' : '',
      `<b>step ${d.n}</b> <span class="tag">${esc(d.tool || '—')}</span> `
      + (d.repaired ? '<span class="warn tag">repaired</span> ' : '')
      + `<span class="dim">${d.tokens_used} tok</span>`
      + `<div class="dim">${esc(d.thought)}</div>`
      + `<div class="obs">${esc(d.observation || '')}</div>`
      + (links ? `<div>📄 ${links}</div>` : ''));
    return;
  }

  if (ev.type === 'final') {
    addStep('done', `<b>answer</b><div class="obs">${esc(d.answer)}</div>`);
    $('agentstat').className = d.halted ? 'warn' : 'ok';
    $('agentstat').textContent = d.halted ? 'halted' : 'done';
    $('budget').textContent = `${d.tokens_used} / ${d.max_tokens ?? '?'} tokens · ${d.steps ?? ''} steps`;
    return;
  }

  if (ev.type === 'error') {
    addStep('deny', `<b class="bad">error</b><div class="obs">${esc(d.error)}</div>`);
    $('agentstat').className = 'bad';
    $('agentstat').textContent = 'failed';
  }
}
