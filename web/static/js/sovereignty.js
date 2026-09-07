// The sovereignty panel: drop counter, drop feed, audit stream, red button,
// chain verification. The drop monitor is the graded claim; it reconnects on
// close because a blank rail mid-demo looks like a failure of the thesis
// rather than of a websocket.

// The counter is the thesis as a number, so a change in it is the one state
// change worth animating: counts up to the new value over ~600ms so a jump
// from 0 to 7 reads as packets being killed one after another.
let dropsShown = 0;
function setDrops(target) {
  const el = $('drops');
  el.className = 'big ' + (target > 0 ? 'ok' : 'dim');
  if (reducedMotion() || target === dropsShown || target - dropsShown > 500) {
    dropsShown = target; el.textContent = target; return;
  }
  const from = dropsShown, t0 = performance.now();
  dropsShown = target;
  (function step(now) {
    const k = Math.min(1, (now - t0) / 600);
    const eased = 1 - Math.pow(1 - k, 3);            // ease-out cubic
    el.textContent = Math.round(from + (target - from) * eased);
    if (k < 1) requestAnimationFrame(step);
    else el.textContent = target;                     // land exactly on the value
  })(t0);
}

// Audit kinds map onto the console's semantic palette; unknown kinds stay dim.
const KIND_CLASS = {
  egress_attempt: 'k-bad', error: 'k-bad',
  approval: 'k-warn', model_call: 'k-focus', tool_call: 'k-ok',
  file_write: 'k-ok', file_read: '',
};

function render(m) {
  setDrops(m.total_drops);
  $('mon').textContent = m.monitor_ok
    ? 'watching the Windows Firewall drop log, live'
    : 'monitor unavailable: ' + m.monitor_note;
  $('mon').className = m.monitor_ok ? 'mon dim' : 'mon warn';

  if (m.new_drops.length) {
    const tb = $('droptbl');
    if (tb.querySelector('.dim')) tb.innerHTML = '';
    for (const d of m.new_drops) {
      const tr = tb.insertRow(0);
      tr.className = 'flash';
      tr.innerHTML = `<td class="dim">${d.ts}</td><td class="dim">${d.proto}</td>`
        + `<td>${d.dst}</td><td><span class="port">${d.dport}</span></td>`;
    }
    while (tb.rows.length > 40) tb.deleteRow(-1);
  }

  $('audit').innerHTML = m.audit.slice().reverse().map(r => {
    const kc = KIND_CLASS[r.kind] ?? '';
    return `<div class="rec${r.kind === 'egress_attempt' ? ' egress' : ''}">`
      + `<div class="recline"><b class="seq">#${r.seq}</b>`
      + `<span class="kind ${kc}">${r.kind}</span>`
      + `<span class="rmeta">${r.ts.slice(11, 19)} · ${r.hash.slice(0, 8)}</span></div>`
      + `<div class="payload">${esc(JSON.stringify(r.payload))}</div></div>`;
  }).join('');
}

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws/sovereignty`);
  ws.onmessage = e => render(JSON.parse(e.data));
  ws.onclose = () => { $('mon').textContent = 'disconnected, retrying'; setTimeout(connect, 2000); };
}
connect();

async function fire() {
  const btn = $('firebtn');
  const label = btn.querySelector('.rlabel');
  if (btn.disabled) return;               // the probe takes ~4s; two clicks fired two probes
  const labelText = label.textContent;
  btn.disabled = true;
  btn.classList.add('firing');            // pulses while the firewall runs out the clock
  label.textContent = 'Reaching for api.openai.com…';
  $('result').textContent = 'attempting…';
  $('result').className = 'verdict warn';
  try {
    const r = await (await fetch('/api/egress-test', { method: 'POST' })).json();
    const p = r.payload;
    $('result').className = p.contained ? 'verdict ok' : 'verdict bad';
    $('result').textContent = p.contained
      ? `BLOCKED at ${p.stage} after ${p.elapsed_s}s — ${p.error}`
      : `!! CONTAINMENT FAILED: reached ${p.resolved}`;
  } catch (e) {
    $('result').className = 'verdict bad';
    $('result').textContent = `probe failed to run: ${e}`;
  } finally {
    // finally, not after the await: a thrown fetch must not leave the single
    // most important control on the page permanently dead mid-demo.
    btn.disabled = false;
    btn.classList.remove('firing');
    label.textContent = labelText;
  }
}

async function checkChain() {
  const r = await (await fetch('/api/audit/verify')).json();
  $('chain').className = 'pill led ' + (r.ok ? 'ok' : 'bad');
  $('chain').textContent = r.ok ? 'chain: intact' : `chain: BROKEN at seq ${r.broken_at}`;
}
checkChain();

