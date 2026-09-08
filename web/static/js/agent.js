// The agent panel (§14.4, §14.5). One websocket per run, bidirectional because
// the §2.4 human gate suspends the loop until the reviewer answers here.

let agentWs = null;

fetch('/api/backend').then(r => r.json()).then(b => {
  $('backend').className = 'pill led ' + (b.ok ? 'ok' : 'bad');
  $('backend').textContent = (b.ok ? 'ollama up' : 'ollama down') + (b.note ? ` · ${b.note}` : '');
  if (b.resident) setResident(b.resident);
  $('autoapprove').checked = !!b.auto_approve;
  renderGateBanner();
});

// Header pill: which model holds VRAM right now (§4.2.1 — one at a time).
function setResident(id) {
  $('resident').className = 'pill on';
  $('resident').textContent = 'resident ' + id;
}

// The §2.4 gate is a compliance control, so arming it off must never be quiet.
function renderGateBanner() {
  const on = $('autoapprove').checked;
  $('gatebanner').hidden = !on;
  $('gatebanner').textContent = on
    ? '⚠ auto mode: writes and sandboxed code run without human approval. '
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
  $('agentroute').hidden = true;
  $('budget').textContent = '';
  $('budgetbar').style.width = '0%';
  $('agentstat').textContent = 'connecting…';
  $('agentstat').className = 'stat warn';
  $('runbtn').disabled = true;
  $('stopbtn').hidden = false;
  $('stopbtn').disabled = false;

  const ws = new WebSocket(`ws://${location.host}/ws/agent`);
  agentWs = ws;
  ws.onopen = () => {
    $('agentstat').textContent = 'running';
    // A hardcoded [] here made §9.2's image override unreachable (test_web.py).
    const attachments = splitList('agentattach');
    ws.send(JSON.stringify({ text, attachments, auto_approve: $('autoapprove').checked }));
  };
  ws.onmessage = e => onAgentEvent(JSON.parse(e.data));
  ws.onclose = () => {
    $('runbtn').disabled = false;
    $('stopbtn').hidden = true;
    if ($('agentstat').textContent === 'running') {
      $('agentstat').textContent = 'disconnected';
      $('agentstat').className = 'stat bad';
    }
  };
}

// The operator kill switch. Sent on the same channel as approve/reject —
// core/api/agent.py's reader task demultiplexes it from an approval answer.
// A stop mid-approval is answered as a denial server-side, so the pending
// button prompt (if any) is cleared here rather than left stale on screen.
function stopAgent() {
  if (!agentWs || agentWs.readyState !== WebSocket.OPEN) return;
  agentWs.send(JSON.stringify({ stop: true }));
  $('stopbtn').disabled = true;
  $('agentstat').textContent = 'stopping…';
  $('agentstat').className = 'stat warn';
  const pending = $('trace').querySelector('.gate .gateactions');
  if (pending) pending.remove();
}

function onAgentEvent(ev) {
  const d = ev.data;

  if (ev.type === 'route') {
    // Rendered before the first token is requested, so the 2-5 s model load
    // reads as an explanation rather than a stall (§4.2.3).
    setResident(d.model_id);
    $('agentroute').hidden = false;
    $('agentroute').className = 'routecard ok';
    $('agentroute').innerHTML = routeCardHTML(d);
    return;
  }

  if (ev.type === 'approval_request') {
    const div = addStep('gate',
      `<div class="gateline"><span class="gmark">approval required</span>`
      + `<span class="tag">${esc(d.tool)}</span><span class="dim">step ${d.n}</span></div>`
      + `<div class="sthought">${esc(d.thought)}</div>`
      + `<div class="sargs">${esc(JSON.stringify(d.args, null, 1))}</div>`
      + (d.auto
          ? '<div class="ga"><span class="warn">auto-approved · recorded as granted_by: auto, never as human sign-off</span></div>'
          : '<div class="ga"><button class="go" data-yes>✓ Approve</button>'
            + '<button class="denybtn" data-no>✕ Reject</button></div>'));

    // Auto-granted steps never suspend the loop, so there is nothing to answer.
    if (d.auto) return;
    $('agentstat').textContent = 'waiting for approval';
    $('agentstat').className = 'stat warn';
    const answer = ok => {
      agentWs.send(JSON.stringify({ approve: ok }));
      div.querySelector('.ga').innerHTML =
        ok ? '<span class="ok">✓ approved — proceeding</span>'
           : '<span class="bad">✕ rejected — nothing was written</span>';
      $('agentstat').textContent = 'running';
      $('agentstat').className = 'stat ok';
    };
    div.querySelector('[data-yes]').onclick = () => answer(true);
    div.querySelector('[data-no]').onclick = () => answer(false);
    return;
  }

  if (ev.type === 'step' || ev.type === 'denied') {
    const links = (d.artifacts || []).map(a =>
      `<a href="/api/artifact?path=${encodeURIComponent(a)}">${esc(a)}</a>`).join(' ');
    const div = addStep(ev.type === 'denied' ? 'deny' : '',
      `<div class="sline"><span class="tag">${esc(d.tool || '-')}</span>`
      + (d.repaired ? '<span class="tag rep">repaired</span>' : '')
      + `<span class="stok">${d.tokens_used} tok</span></div>`
      + `<div class="sthought">${esc(d.thought)}</div>`
      + `<div class="sobs">${esc(d.observation || '')}</div>`
      + (links ? `<div class="slinks">▸ ${links}</div>` : ''));
    // Long observations clip with a fade; one click opens them.
    const obs = div.querySelector('.sobs');
    if (obs.scrollHeight > obs.clientHeight + 4) {
      obs.classList.add('clip');
      obs.title = 'click to expand';
      obs.addEventListener('click', () => obs.classList.toggle('open'));
    }
    return;
  }

  if (ev.type === 'final') {
    addStep('done', `<div class="gateline"><span class="gmark ok">answer</span></div>`
      + `<div class="sobs answer">${esc(d.answer)}</div>`);
    $('agentstat').className = 'stat ' + (d.halted ? 'warn' : 'ok');
    $('agentstat').textContent = d.halted ? 'halted' : 'done';
    const used = d.tokens_used || 0, cap = d.max_tokens || 0;
    $('budgetbar').style.width = cap ? `${Math.min(100, 100 * used / cap)}%` : '0%';
    $('budget').textContent =
      `${used.toLocaleString()} / ${cap ? cap.toLocaleString() : '?'} tokens · ${d.steps ?? '?'} steps`;
    return;
  }

  if (ev.type === 'error') {
    addStep('deny', `<div class="gateline"><span class="gmark bad">error</span></div>`
      + `<div class="sobs">${esc(d.error)}</div>`);
    $('agentstat').className = 'stat bad';
    $('agentstat').textContent = 'failed';
  }
}
