// The firewall ruleset, verbatim (§10.1) — paraphrasing it would weaken the
// claim, so the API serves the file text and this only adds line numbers and
// a minimal PowerShell tint. Escaping happens before any highlight span, so
// the ruleset can never inject markup into its own display.

fetch('/api/firewall/rules').then(r => r.text()).then(text => {
  const lines = text.replace(/\r/g, '').split('\n');
  $('rules').innerHTML = lines.map((line, i) => {
    const num = String(i + 1).padStart(3, '0');
    const safe = esc(line);
    const trimmed = line.trim();
    // Whole-line comments dim; cmdlet names tinted. Nothing else — a clever
    // tokenizer is a way to accidentally unescape something.
    const body = trimmed.startsWith('#')
      ? safe
      : safe.replace(/\b((?:Set|New|Get|Remove|Enable|Disable|Out)-[A-Za-z]+\b)/g,
          '<span class="kw">$1</span>');
    const cls = trimmed.startsWith('#') ? ' comment' : '';
    return `<div class="rl${cls}"><span class="ln">${num}</span><span class="lc">${body || ' '}</span></div>`;
  }).join('');
});
