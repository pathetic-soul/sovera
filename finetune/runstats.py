"""Live training stats: an append-only JSONL plus a self-refreshing dashboard.

Why this exists. A QLoRA run on a 6 GB card takes hours, and the two things
that decide whether it is worth finishing — is the loss actually falling, and
did we quietly cross the §4.1 VRAM ceiling into shared-memory swap — are both
invisible in a scrolling log until it is far too late to act. §17 records the
Windows failure mode that makes this acute: overcommitting VRAM does not raise
OOM, it just makes every step take minutes, so a run that has gone wrong looks
exactly like a run that is merely slow.

Design, in one line: the trainer appends facts, the dashboard is a pure
function of those facts, regenerated on every append.

That ordering matters. `metrics.jsonl` is the source of truth and is never
rewritten, so a crashed run leaves a complete record rather than a corrupt
file, and the dashboard can be thrown away and rebuilt from it at any time.

**Self-contained by construction (§2.1).** The HTML embeds its own data, CSS
and SVG — no CDN, no fonts, no fetch(). This matters beyond neatness: a
dashboard that loaded anything over the network would be a page in this repo
that phones home, which is the one thing the project promises never to ship.
It also means the file works from `file://` with the firewall armed, which is
when you will actually be reading it.

Not gitignored by accident — `finetune/runs/` is build-time output. The stats
are reproducible by re-running training; the code that produces them is source.

    python -m finetune.runstats --demo    # render a sample dashboard and exit
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "finetune" / "runs"

VRAM_CEILING_GB = 5.2  # AGENTS.md §4.1
REFRESH_SECONDS = 15

# From the validated reference palette. Single-series charts throughout, so no
# categorical pair ever shares a plot and the CVD adjacency gates are moot; the
# title names the series, which is why no chart here carries a legend.
_CSS = """
*{box-sizing:border-box}
.viz-root{
  color-scheme:light;
  --surface-1:#fcfcfb; --plane:#f9f9f7;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
  --series-1:#2a78d6;
  --good:#0ca30c; --warning:#fab219; --critical:#d03b3b;
}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{
  color-scheme:dark;
  --surface-1:#1a1a19; --plane:#0d0d0d;
  --text-primary:#fff; --text-secondary:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
  --series-1:#3987e5;
}}
:root[data-theme="dark"] .viz-root{
  color-scheme:dark;
  --surface-1:#1a1a19; --plane:#0d0d0d;
  --text-primary:#fff; --text-secondary:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
  --series-1:#3987e5;
}
body{margin:0;background:var(--plane)}
.viz-root{
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
  color:var(--text-primary); padding:24px; max-width:1180px; margin:0 auto;
}
h1{font-size:19px;margin:0 0 2px;font-weight:650;letter-spacing:-0.01em}
.sub{color:var(--text-secondary);font-size:13px;margin-bottom:16px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 18px}
.chip{font-size:11.5px;color:var(--text-secondary);background:var(--surface-1);
  border:1px solid var(--border);border-radius:999px;padding:3px 9px}
.chip b{color:var(--text-primary);font-weight:600}
.badge{display:inline-block;font-size:11.5px;font-weight:650;border-radius:999px;
  padding:3px 10px;color:#fff;vertical-align:2px;margin-left:8px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:18px}
.tile{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.tile .k{font-size:11.5px;color:var(--muted);margin-bottom:5px}
.tile .v{font-size:25px;font-weight:640;letter-spacing:-0.02em;line-height:1.08}
.tile .n{font-size:11.5px;color:var(--text-secondary);margin-top:3px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:12px}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:14px 12px 8px}
.card h2{font-size:13px;margin:0 0 1px;padding-left:4px;font-weight:620}
.card .cap{font-size:11.5px;color:var(--muted);margin:0 0 6px;padding-left:4px}
.wrap{overflow-x:auto}
svg{display:block;width:100%;height:auto}
.tt{position:fixed;pointer-events:none;opacity:0;transition:opacity .09s;
  background:var(--surface-1);border:1px solid var(--border);border-radius:7px;
  padding:6px 9px;font-size:12px;box-shadow:0 3px 12px rgba(0,0,0,.14);z-index:9}
.tt b{font-variant-numeric:tabular-nums}
table{border-collapse:collapse;width:100%;font-size:12px;
  font-variant-numeric:tabular-nums;margin-top:4px}
th,td{text-align:right;padding:4px 8px;border-bottom:1px solid var(--grid)}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600}
details{margin-top:16px}
summary{cursor:pointer;font-size:12.5px;color:var(--text-secondary);padding:5px 0}
.foot{color:var(--muted);font-size:11.5px;margin-top:18px}
"""

# Pause the auto-reload while a pointer is over the page, so a hover tooltip is
# actually readable. Without this the reload fires mid-inspection and the thing
# you were reading vanishes — the dashboard would be refreshing itself out of
# usefulness.
_JS = """
(function(){
  var ms=%d*1000,t=null,armed=true;
  function arm(){if(t)clearTimeout(t);if(armed)t=setTimeout(function(){location.reload()},ms)}
  document.addEventListener('mouseenter',function(){armed=false;if(t)clearTimeout(t)},true);
  document.addEventListener('mouseleave',function(){armed=true;arm()},true);
  arm();
  var tip=document.querySelector('.tt');
  document.querySelectorAll('[data-tip]').forEach(function(el){
    el.addEventListener('mousemove',function(e){
      tip.innerHTML=el.getAttribute('data-tip');tip.style.opacity=1;
      var x=e.clientX+14,y=e.clientY+14;
      if(x+180>window.innerWidth)x=e.clientX-190;
      tip.style.left=x+'px';tip.style.top=y+'px';
    });
    el.addEventListener('mouseleave',function(){tip.style.opacity=0});
  });
})();
"""


def _esc(text: Any) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _nice_ticks(lo: float, hi: float, count: int = 4) -> list[float]:
    """Round tick values, so the axis reads 0.5 / 1.0 / 1.5 rather than 0.5137."""
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / count
    mag = 10 ** (len(str(int(abs(raw)))) - 1 if abs(raw) >= 1 else -3)
    for mult in (1, 2, 2.5, 5, 10, 20, 25, 50, 100):
        step = mult * mag
        if step >= raw:
            break
    start = (int(lo / step)) * step
    ticks = []
    value = start
    while value <= hi + step * 0.5:
        if value >= lo - step * 0.5:
            ticks.append(round(value, 6))
        value += step
    return ticks or [lo, hi]


def line_chart(
    points: Sequence[tuple[float, float]],
    *,
    y_label: str,
    fmt: str = "{:.4f}",
    ref: float | None = None,
    ref_label: str = "",
    width: int = 560,
    height: int = 190,
) -> str:
    """One series, inline SVG, no dependencies.

    Downsamples above ~400 points: a 3000-step run cannot resolve 3000 marks in
    560 px, and emitting them only makes the file bigger and the hover targets
    narrower than the pointer.
    """
    if not points:
        return '<p class="cap">no data yet</p>'
    if len(points) > 400:
        stride = len(points) // 400 + 1
        points = list(points[::stride]) + [points[-1]]

    pad_l, pad_r, pad_t, pad_b = 54, 14, 12, 26
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    if ref is not None:
        y_hi = max(y_hi, ref)
        y_lo = min(y_lo, ref)
    if y_hi - y_lo < 1e-9:
        y_hi, y_lo = y_hi + 1, y_lo - 1
    span = y_hi - y_lo
    y_lo, y_hi = y_lo - span * 0.08, y_hi + span * 0.08

    def sx(x: float) -> float:
        return pad_l + (0 if x_hi == x_lo else (x - x_lo) / (x_hi - x_lo)) * plot_w

    def sy(y: float) -> float:
        return pad_t + (1 - (y - y_lo) / (y_hi - y_lo)) * plot_h

    out = [f'<svg viewBox="0 0 {width} {height}" role="img" '
           f'aria-label="{_esc(y_label)} over training steps">']

    for tick in _nice_ticks(y_lo, y_hi):
        y = sy(tick)
        if not (pad_t - 1 <= y <= pad_t + plot_h + 1):
            continue
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
                   f'stroke="var(--grid)" stroke-width="1"/>')
        out.append(f'<text x="{pad_l - 7}" y="{y + 3.5:.1f}" text-anchor="end" '
                   f'font-size="10.5" fill="var(--muted)">{fmt.format(tick)}</text>')

    out.append(f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" '
               f'y2="{pad_t + plot_h}" stroke="var(--axis)" stroke-width="1"/>')
    for x in ({xs[0], xs[len(xs) // 2], xs[-1]} if len(xs) > 2 else set(xs)):
        out.append(f'<text x="{sx(x):.1f}" y="{height - 8}" text-anchor="middle" '
                   f'font-size="10.5" fill="var(--muted)">{int(x)}</text>')

    if ref is not None:
        y = sy(ref)
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
                   f'stroke="var(--critical)" stroke-width="2" stroke-dasharray="5 4"/>')
        out.append(f'<text x="{pad_l + plot_w}" y="{y - 5:.1f}" text-anchor="end" '
                   f'font-size="10.5" font-weight="600" fill="var(--critical)">'
                   f'{_esc(ref_label)}</text>')

    path = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f} {sy(y):.1f}"
                    for i, (x, y) in enumerate(points))
    out.append(f'<path d="{path}" fill="none" stroke="var(--series-1)" stroke-width="2" '
               f'stroke-linejoin="round" stroke-linecap="round"/>')

    # Only the last point gets a mark and a direct label — a number on every
    # point is the anti-pattern, and the latest value is the one being watched.
    lx, ly = sx(xs[-1]), sy(ys[-1])
    out.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="3.5" fill="var(--series-1)" '
               f'stroke="var(--surface-1)" stroke-width="2"/>')

    # Invisible hit columns: hover targets must be bigger than the mark.
    band = max(plot_w / max(len(points), 1), 6)
    for x, y in points:
        tip = f"step <b>{int(x)}</b> &middot; {_esc(y_label)} <b>{fmt.format(y)}</b>"
        out.append(f'<rect x="{sx(x) - band / 2:.1f}" y="{pad_t}" width="{band:.1f}" '
                   f'height="{plot_h}" fill="transparent" data-tip="{tip}"/>')

    out.append("</svg>")
    return "".join(out)


def _fmt_hms(seconds: float) -> str:
    if seconds <= 0 or seconds != seconds:  # NaN-safe
        return "--"
    seconds = int(seconds)
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


class RunLog:
    """Append facts; the dashboard follows.

    One instance per training run. Safe to construct when nothing will ever be
    logged (a probe, a crash before step 1): the dashboard renders an empty
    state rather than failing, because a run that died early is exactly when
    you want to look at it.
    """

    def __init__(self, model: str, config: dict[str, Any], runs_dir: Path = RUNS_DIR) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{stamp}-{model}"
        self.dir = runs_dir / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.dir / "metrics.jsonl"
        self.html_path = self.dir / "dashboard.html"
        self.model = model
        self.config = dict(config)
        self.started = time.time()
        self.status = "running"
        self.summary: dict[str, Any] = {}
        self.rows: list[dict[str, Any]] = []
        (self.dir / "config.json").write_text(
            json.dumps({"model": model, "run_id": self.run_id, **self.config}, indent=2),
            encoding="utf-8",
        )
        self.render()

    def log(self, **metrics: Any) -> None:
        row = {"t": round(time.time() - self.started, 2), **metrics}
        self.rows.append(row)
        with self.metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        self.render()

    def finish(self, status: str, **summary: Any) -> None:
        self.status = status
        self.summary = summary
        (self.dir / "summary.json").write_text(
            json.dumps({"status": status, "elapsed_s": round(time.time() - self.started, 1),
                        **summary}, indent=2),
            encoding="utf-8",
        )
        self.render()
        write_index(self.dir.parent)

    # ---------------- rendering ----------------

    def _tiles(self) -> str:
        last = self.rows[-1] if self.rows else {}
        step = int(last.get("step", 0))
        total = int(last.get("total_steps", 0) or self.config.get("total_steps", 0))
        elapsed = time.time() - self.started
        per_step = elapsed / step if step else 0.0
        eta = per_step * (total - step) if total and step else 0.0
        peak = max((r.get("vram_gb", 0) or 0) for r in self.rows) if self.rows else 0.0

        if peak == 0:
            vram_color, vram_note = "var(--muted)", "not yet measured"
        elif peak > VRAM_CEILING_GB:
            vram_color, vram_note = "var(--critical)", f"over the {VRAM_CEILING_GB} GB ceiling"
        elif peak > VRAM_CEILING_GB - 0.4:
            vram_color, vram_note = "var(--warning)", "close to the ceiling"
        else:
            vram_color, vram_note = "var(--good)", f"under the {VRAM_CEILING_GB} GB ceiling"

        pct = f"{100.0 * step / total:.0f}%" if total else "--"
        loss = last.get("loss")
        first_loss = next((r.get("loss") for r in self.rows if r.get("loss") is not None), None)
        if loss is not None and first_loss:
            delta = f"from {first_loss:.3f} at start"
        else:
            delta = "waiting for step 1"

        tiles = [
            ("Progress", pct, f"step {step} of {total or '?'}", None),
            ("Loss", f"{loss:.4f}" if loss is not None else "--", delta, None),
            ("Peak VRAM", f"{peak:.2f} GB" if peak else "--", vram_note, vram_color),
            ("Seconds / step", f"{per_step:.1f}" if per_step else "--", "wall clock", None),
            ("Elapsed", _fmt_hms(elapsed), "since start", None),
            ("ETA", _fmt_hms(eta), "at current rate", None),
        ]
        out = ['<div class="tiles">']
        for key, value, note, color in tiles:
            style = f' style="color:{color}"' if color else ""
            out.append(f'<div class="tile"><div class="k">{_esc(key)}</div>'
                       f'<div class="v"{style}>{_esc(value)}</div>'
                       f'<div class="n">{_esc(note)}</div></div>')
        out.append("</div>")
        return "".join(out)

    def _series(self, key: str) -> list[tuple[float, float]]:
        return [(float(r["step"]), float(r[key])) for r in self.rows
                if r.get(key) is not None and r.get("step") is not None]

    def _table(self) -> str:
        if not self.rows:
            return ""
        cols = ["step", "loss", "grad_norm", "lr", "vram_gb"]
        head = "".join(f"<th>{c}</th>" for c in cols)
        body = []
        for r in self.rows[-40:]:
            cells = []
            for c in cols:
                v = r.get(c)
                if v is None:
                    cells.append("<td>--</td>")
                elif c == "lr":
                    cells.append(f"<td>{v:.2e}</td>")
                elif c == "step":
                    cells.append(f"<td>{int(v)}</td>")
                else:
                    cells.append(f"<td>{v:.4f}</td>")
            body.append("<tr>" + "".join(cells) + "</tr>")
        return (f'<details><summary>Table view - last {min(40, len(self.rows))} logged steps '
                f'(identity never rests on color alone)</summary>'
                f'<div class="wrap"><table><thead><tr>{head}</tr></thead>'
                f'<tbody>{"".join(body)}</tbody></table></div></details>')

    def render(self) -> None:
        badge = {"running": "#2a78d6", "done": "#0ca30c",
                 "failed": "#d03b3b", "stopped": "#fab219"}.get(self.status, "#898781")
        chips = "".join(
            f'<span class="chip">{_esc(k)} <b>{_esc(v)}</b></span>'
            for k, v in self.config.items()
        )
        charts = [
            ("Training loss", "Lower is better. A flat curve after warmup means the run "
             "is not learning and is worth killing early.",
             line_chart(self._series("loss"), y_label="loss", fmt="{:.3f}")),
            ("VRAM", f"Against the §4.1 ceiling. On Windows, crossing it does not raise "
             f"OOM (§17) - it silently spills to shared memory and steps take minutes.",
             line_chart(self._series("vram_gb"), y_label="VRAM GB", fmt="{:.1f}",
                        ref=VRAM_CEILING_GB, ref_label=f"{VRAM_CEILING_GB} GB ceiling")),
            ("Gradient norm", "Clipped at 0.3. Spikes here precede loss divergence.",
             line_chart(self._series("grad_norm"), y_label="grad norm", fmt="{:.2f}")),
            ("Learning rate", "Cosine schedule after warmup. Confirms the scheduler ran.",
             line_chart(self._series("lr"), y_label="lr", fmt="{:.1e}")),
        ]
        cards = "".join(
            f'<div class="card"><h2>{_esc(title)}</h2><p class="cap">{cap}</p>'
            f'<div class="wrap">{svg}</div></div>'
            for title, cap, svg in charts
        )
        refresh = (f'<meta http-equiv="refresh" content="{REFRESH_SECONDS}">'
                   if self.status == "running" else "")
        html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
{refresh}<title>{_esc(self.run_id)}</title><style>{_CSS}</style></head>
<body><div class="viz-root">
<h1>{_esc(self.model)} - QLoRA<span class="badge" style="background:{badge}">
{_esc(self.status)}</span></h1>
<div class="sub">{_esc(self.run_id)} &middot; updated {datetime.now():%H:%M:%S}
{' &middot; auto-refreshing, paused while hovering' if self.status == 'running' else ''}</div>
<div class="chips">{chips}</div>
{self._tiles()}
<div class="grid">{cards}</div>
{self._table()}
<p class="foot">Source of truth is metrics.jsonl beside this file; this page is
regenerated from it on every append. Self-contained - no network, no CDN (§2.1).</p>
</div><div class="tt"></div><script>{_JS % REFRESH_SECONDS}</script></body></html>"""
        self.html_path.write_text(html, encoding="utf-8")


def write_index(runs_dir: Path = RUNS_DIR) -> Path:
    """One page listing every run, so the history is visible across the project."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for d in sorted(runs_dir.iterdir(), reverse=True):
        if not d.is_dir() or not (d / "config.json").exists():
            continue
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        summary: dict[str, Any] = {}
        if (d / "summary.json").exists():
            summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        status = summary.get("status", "running")
        rows = []
        if (d / "metrics.jsonl").exists():
            rows = [json.loads(x) for x in
                    (d / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        last_loss = next((r["loss"] for r in reversed(rows) if r.get("loss") is not None), None)
        peak = max((r.get("vram_gb", 0) or 0) for r in rows) if rows else 0
        badge = {"running": "#2a78d6", "done": "#0ca30c",
                 "failed": "#d03b3b", "stopped": "#fab219"}.get(status, "#898781")
        entries.append(
            f'<tr><td><a href="{_esc(d.name)}/dashboard.html">{_esc(d.name)}</a></td>'
            f'<td style="text-align:left"><span class="badge" style="background:{badge}">'
            f'{_esc(status)}</span></td>'
            f'<td>{_esc(cfg.get("lora_r", "--"))}</td><td>{_esc(cfg.get("max_len", "--"))}</td>'
            f'<td>{_esc(cfg.get("traces", "--"))}</td>'
            f'<td>{f"{last_loss:.4f}" if last_loss is not None else "--"}</td>'
            f'<td>{f"{peak:.2f}" if peak else "--"}</td>'
            f'<td>{_fmt_hms(summary.get("elapsed_s", 0))}</td></tr>'
        )
    body = "".join(entries) or '<tr><td colspan="8">no runs yet</td></tr>'
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QLoRA runs</title><style>{_CSS}</style></head>
<body><div class="viz-root"><h1>QLoRA runs</h1>
<div class="sub">Every fine-tuning run in this project, newest first.</div>
<div class="wrap"><table><thead><tr><th>run</th><th style="text-align:left">status</th>
<th>r</th><th>max_len</th><th>traces</th><th>last loss</th><th>peak GB</th><th>elapsed</th>
</tr></thead><tbody>{body}</tbody></table></div>
<p class="foot">finetune/runs/ is gitignored: reproducible build-time output.</p>
</div></body></html>"""
    path = runs_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def main() -> int:
    import argparse
    import math
    import random

    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="render a sample dashboard")
    args = parser.parse_args()
    if not args.demo:
        print(f"runs dir: {RUNS_DIR}\nindex: {write_index()}")
        return 0

    log = RunLog("demo", {"lora_r": 8, "max_len": 1024, "traces": 612, "total_steps": 120})
    for step in range(1, 121):
        log.log(step=step, total_steps=120,
                loss=2.6 * math.exp(-step / 45) + 0.35 + random.uniform(-0.04, 0.04),
                grad_norm=0.3 * math.exp(-step / 70) + random.uniform(0, 0.05),
                lr=2e-4 * (1 + math.cos(math.pi * step / 120)) / 2,
                vram_gb=4.9 + random.uniform(-0.08, 0.12))
    log.finish("done", peak_gb=5.02)
    print(f"demo -> {log.html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
