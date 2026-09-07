// The tool roster panel — /api/tools renders the same specs the agent sees in
// its system prompt, including which tools the §2.4 human gate covers. The
// gate is a compliance feature, so "which actions need sign-off" is rendered,
// never implied.

fetch('/api/tools').then(r => r.json()).then(({ tools }) => {
  $('toolgrid').innerHTML = tools.map(t =>
    `<div class="tool${t.requires_approval ? ' gated' : ''}">`
    + `<div class="trow"><b>${esc(t.name)}</b>`
    + (t.requires_approval
        ? '<span class="tgate" title="runs only after a human approves (§2.4)">needs sign-off</span>'
        : '<span class="topen">ungated</span>')
    + '</div>'
    + `<div class="tdesc">${esc(t.description)}</div>`
    + '</div>'
  ).join('');
});
