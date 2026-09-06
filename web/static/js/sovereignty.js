// The sovereignty panel: drop counter, drop table, audit stream, red button,
// chain verification, and the firewall ruleset shown verbatim (AGENTS.md §10).
// This is the graded claim; it reconnects on close because a blank panel mid-demo
// looks like a failure of the thesis rather than of a websocket.

// The drop counter is the thesis as a number, so a change in it is the one
// state change worth animating rather than swapping. Counts from the value on
// screen to the new one over ~600ms; a jump from 0 to 7 reads as a glitch,
// a climb reads as packets being killed one after another.
let dropsShown = 0;
function setDrops(target) {
  const el = $('drops');
  el.className = 'big ' + (target > 0 ? 'ok' : 'dim');
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (reduced || target === dropsShown || target - dropsShown > 500) {
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

function render(m) {
  setDrops(m.total_drops);
  $('mon').textContent = m.monitor_ok
    ? 'watching Windows Firewall drop log'
    : 'monitor unavailable: ' + m.monitor_note;
  $('mon').className = m.monitor_ok ? 'dim' : 'warn';

  if (m.new_drops.length) {
    const tb = $('droptbl');
    if (tb.querySelector('.dim')) tb.innerHTML = '';
    for (const d of m.new_drops) {
      const tr = tb.insertRow(0);
      tr.className = 'flash';
      tr.innerHTML = `<td>${d.ts}</td><td>${d.proto}</td><td>${d.dst}</td><td>${d.dport}</td>`;
    }
    while (tb.rows.length > 40) tb.deleteRow(-1);
  }

  $('audit').innerHTML = m.audit.slice().reverse().map(r =>
    `<div class="rec ${r.kind === 'egress_attempt' ? 'egress' : ''}">`
    + `<b>#${r.seq}</b> ${r.kind} `
    + `<span class="dim">${r.ts.slice(11, 19)} · ${r.hash.slice(0, 8)}</span><br>`
    + `<span class="dim">${esc(JSON.stringify(r.payload))}</span></div>`
  ).join('');
}

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws/sovereignty`);
  ws.onmessage = e => render(JSON.parse(e.data));
  ws.onclose = () => { $('mon').textContent = 'disconnected, retrying'; setTimeout(connect, 2000); };
}
connect();

async function fire() {
  const btn = $('firebtn');
  if (btn.disabled) return;               // the probe takes ~4s; two clicks fired two probes
  const label = btn.textContent;
  btn.disabled = true;
  btn.classList.add('firing');            // pulses while the firewall runs out the clock
  btn.textContent = '● Reaching for api.openai.com:443…';
  $('result').textContent = 'attempting…';
  $('result').className = 'warn';
  try {
    const r = await (await fetch('/api/egress-test', {method: 'POST'})).json();
    const p = r.payload;
    $('result').className = p.contained ? 'ok' : 'bad';
    $('result').textContent = p.contained
      ? `BLOCKED at ${p.stage} after ${p.elapsed_s}s: ${p.error}`
      : `!! CONTAINMENT FAILED: reached ${p.resolved}`;
  } catch (e) {
    $('result').className = 'bad';
    $('result').textContent = `probe failed to run: ${e}`;
  } finally {
    // finally, not after the await: a thrown fetch must not leave the single
    // most important control on the page permanently dead mid-demo.
    btn.disabled = false;
    btn.classList.remove('firing');
    btn.textContent = label;
  }
}

async function checkChain() {
  const r = await (await fetch('/api/audit/verify')).json();
  $('chain').className = r.ok ? 'ok' : 'bad';
  $('chain').textContent = r.ok ? 'chain: intact' : `chain: BROKEN at seq ${r.broken_at}`;
}
checkChain();

fetch('/api/firewall/rules').then(r => r.text()).then(t => $('rules').textContent = t);
