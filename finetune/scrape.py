"""Polite, provenance-recording web collection for the training corpus.

Build time only (AGENTS.md §2.1, §3). Nothing here runs at demo time — it sits
in the same window as `docker build` and `finetune.stage`, before
`sovereignty/firewall.ps1 -Enable` arms containment.

    python -m finetune.scrape --check          # robots + reachability, fetch nothing
    python -m finetune.scrape --source oisd    # fetch one source
    python -m finetune.scrape --all

WHY THIS IS MOSTLY POLICY AND NOT MOSTLY SCRAPING
-------------------------------------------------
Scrapling handles the hard part of retrieval well, so the code that matters
here is the code that decides *whether to fetch at all*. Three rules, enforced
rather than documented:

1.  **robots.txt is fail-closed.** If it cannot be read, the host is treated as
    disallowed. This is not pedantry — mrpl.co.in currently serves 503 on
    /robots.txt, and the standard says an unavailable robots.txt means "do not
    crawl". A scraper that defaults to "allowed" when it cannot check would
    quietly crawl a site that never granted permission.

2.  **Allowlisted hosts only.** This is a collector for named sources, not a
    crawler. A generic crawl would drag in text of unknown provenance, and
    `data/sources.yaml` exists precisely so every row in the corpus can name
    its origin and licence. Untraceable text is a liability in a project handed
    to employers, not an asset.

3.  **Every fetch records provenance.** URL, HTTP status, SHA-256, fetch
    timestamp and the licence assessment go into a manifest beside the content.
    A document whose licence cannot be stated is downloaded to
    `data/external/scraped/` and marked `publishable: false`, so it can inform
    build-time work without ever being republished.

WHAT IS DELIBERATELY NOT COLLECTED
----------------------------------
The full text of API 510 / 570 / 653 and equivalent standards. They are the
most on-topic documents imaginable and they are **copyrighted and sold** by
API. `tools/calc.py` cites clause numbers, which is fair reference; mirroring
clause text into a training corpus is not. The formulas themselves are facts
and are already in code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.robotparser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "external" / "scraped"

USER_AGENT = "SovereignWorkbench-ResearchBot (SIH 2026 PS 26117; build-time corpus collection)"

# Politeness floor. Honoured even when robots.txt declares nothing, and raised
# to whatever a host asks for.
# Measured: oisd.gov.in refused 9 of 12 requests at a 2 s delay with
# "could not connect" — the polite floor was still too fast for it. 5 s
# holds. Being slow is the cost of being welcome.
DEFAULT_DELAY_S = 5.0
TIMEOUT_S = 30
MAX_PAGES_PER_SOURCE = 40

# Never follow a link into paid or account-gated territory, whatever a source's
# follow pattern says. OISD and API both sell their standards; buying a PDF and
# then training on it would not make it redistributable either.
EXCLUDE = r"(purchase|buy|cart|checkout|login|signin|register|payment)"


@dataclass(frozen=True)
class Source:
    """One named collection target."""

    id: str
    host: str
    seeds: tuple[str, ...]
    # Only follow links whose URL matches this, so a seed page cannot walk the
    # whole site.
    follow: str
    licence: str
    publishable: bool
    why: str
    want_pdf: bool = True


SOURCES: dict[str, Source] = {
    "oisd": Source(
        id="oisd",
        host="https://www.oisd.gov.in",
        seeds=("https://www.oisd.gov.in/en-in/IncidentGuidelines",
               "https://www.oisd.gov.in/en-in/Publication",
               "https://www.oisd.gov.in/en-in/standards-for-public-comments"),
        # Freely published material only. OISD SELLS its standards
        # (/en-in/purchase-of-standards), so the standards themselves are
        # excluded on the same grounds as API 510: paid, copyrighted text does
        # not belong in a training corpus. Drafts released for public comment,
        # incident guidelines and publications are published for reuse.
        follow=r"/(IncidentGuidelines|Publication|public-comments|assets/upload)",
        licence="Government of India publication — Government Open Data Licence (GODL-India)",
        publishable=False,   # GODL permits reuse; republishing as a dataset still needs review
        why=("Oil Industry Safety Directorate — the Indian regulator whose standards "
             "govern refinery inspection and safety practice. This is the closest "
             "thing to an on-domain public corpus that exists, and nothing "
             "equivalent appears on Kaggle."),
    ),
    "ongc": Source(
        id="ongc",
        host="https://www.ongcindia.com",
        seeds=("https://www.ongcindia.com/web/eng/investors/annual-reports",),
        follow=r"/(investor|annual|report|disclosure)",
        licence="public company disclosure",
        publishable=False,
        why=("MRPL's parent company. Annual reports carry refinery throughput, "
             "turnaround and capacity language absent from MRPL's quarterly "
             "financial statements."),
    ),
}


# --------------------------------------------------------------------------
# politeness
# --------------------------------------------------------------------------

@dataclass
class RobotsVerdict:
    allowed: bool
    delay: float
    reason: str


def check_robots(host: str) -> RobotsVerdict:
    """Read robots.txt and decide. FAIL CLOSED.

    urllib's RobotFileParser already returns False when it cannot read the
    file, but it does so silently and for several different reasons. This
    reports which reason, because "the site said no" and "the site was down"
    call for different follow-ups even though both must block the fetch now.
    """
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(urljoin(host, "/robots.txt"))
    try:
        rp.read()
    except Exception as exc:
        return RobotsVerdict(False, DEFAULT_DELAY_S,
                             f"robots.txt unreadable ({type(exc).__name__}) — treating as disallowed")

    allowed = rp.can_fetch(USER_AGENT, host + "/")
    declared = rp.crawl_delay(USER_AGENT)
    delay = max(DEFAULT_DELAY_S, float(declared) if declared else 0.0)

    if not allowed:
        return RobotsVerdict(False, delay,
                             "robots.txt disallows this agent, or was unavailable "
                             "(a 4xx means allow-all; a 5xx means disallow-all)")
    return RobotsVerdict(True, delay, f"allowed, crawl delay {delay:.1f}s")


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------

@dataclass
class Fetched:
    url: str
    status: int
    content_type: str
    body: bytes
    sha256: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if self.body and not self.sha256:
            self.sha256 = hashlib.sha256(self.body).hexdigest()


def _fetcher() -> Any:
    from scrapling.fetchers import Fetcher

    return Fetcher


def fetch(url: str) -> Fetched:
    """One request through Scrapling, with everything that can throw contained."""
    Fetcher = _fetcher()
    try:
        page = Fetcher.get(url, timeout=TIMEOUT_S, stealthy_headers=True)
    except Exception as exc:
        return Fetched(url, 0, "", b"", note=f"{type(exc).__name__}: {str(exc)[:120]}")

    status = int(getattr(page, "status", 0) or 0)
    body = getattr(page, "body", b"") or b""
    if isinstance(body, str):
        body = body.encode("utf-8", "replace")
    ctype = ""
    headers = getattr(page, "headers", {}) or {}
    for k, v in dict(headers).items():
        if str(k).lower() == "content-type":
            ctype = str(v)
            break
    return Fetched(url, status, ctype, body)


def links_on(url: str, pattern: str) -> list[str]:
    """Absolute in-scope links from one page."""
    Fetcher = _fetcher()
    try:
        page = Fetcher.get(url, timeout=TIMEOUT_S, stealthy_headers=True)
    except Exception:
        return []

    out: list[str] = []
    host = "{0.scheme}://{0.netloc}".format(urlparse(url))
    try:
        anchors = page.css("a::attr(href)")
    except Exception:
        return []

    for href in anchors:
        href = str(href).strip()
        if not href or href.startswith(("#", "javascript:", "mailto:")):
            continue
        absolute = urljoin(url, href)
        # Never leave the host we checked robots.txt for.
        if not absolute.startswith(host):
            continue
        if re.search(EXCLUDE, absolute, re.I):
            continue
        if re.search(pattern, absolute, re.I) or absolute.lower().endswith(".pdf"):
            out.append(absolute)
    # Preserve order, drop duplicates.
    return list(dict.fromkeys(out))


# --------------------------------------------------------------------------

@dataclass
class Report:
    source: Source
    verdict: RobotsVerdict
    saved: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def collect(source: Source, limit: int = MAX_PAGES_PER_SOURCE) -> Report:
    verdict = check_robots(source.host)
    report = Report(source=source, verdict=verdict)
    if not verdict.allowed:
        return report

    dest = OUT_DIR / source.id
    dest.mkdir(parents=True, exist_ok=True)

    targets: list[str] = []
    for seed in source.seeds:
        targets.append(seed)
        targets.extend(links_on(seed, source.follow))
        time.sleep(verdict.delay)

    targets = list(dict.fromkeys(targets))[:limit]

    for url in targets:
        if source.want_pdf is False and url.lower().endswith(".pdf"):
            continue
        got = fetch(url)
        time.sleep(verdict.delay)
        if got.status != 200 and "Connection" in got.note:
            time.sleep(verdict.delay * 2)      # backing off, not retrying harder
            got = fetch(url)
            time.sleep(verdict.delay)
        if got.status != 200 or not got.body:
            report.skipped.append(f"{url} -> {got.note or got.status or 'no response'}")
            continue

        suffix = ".pdf" if "pdf" in got.content_type.lower() or url.lower().endswith(".pdf") else ".html"
        name = re.sub(r"[^A-Za-z0-9]+", "-", urlparse(url).path).strip("-").lower()[:80] or "index"
        path = dest / f"{name}{suffix}"
        path.write_bytes(got.body)

        report.saved.append({
            "url": url,
            "file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "status": got.status,
            "content_type": got.content_type,
            "bytes": len(got.body),
            "sha256": got.sha256,
            "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "licence": source.licence,
            "publishable": source.publishable,
        })

    if report.saved:
        manifest = dest / "manifest.json"
        manifest.write_text(json.dumps({
            "source": source.id,
            "host": source.host,
            "licence": source.licence,
            "publishable": source.publishable,
            "user_agent": USER_AGENT,
            "robots": verdict.reason,
            "documents": report.saved,
        }, indent=2), encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="report robots.txt verdicts and fetch nothing")
    ap.add_argument("--source", choices=sorted(SOURCES))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=MAX_PAGES_PER_SOURCE)
    args = ap.parse_args()

    chosen = (list(SOURCES.values()) if args.all or args.check
              else [SOURCES[args.source]] if args.source else [])
    if not chosen:
        ap.error("pass --check, --source <id> or --all")

    if args.check:
        print(f"user-agent: {USER_AGENT}\n")
        for src in chosen:
            v = check_robots(src.host)
            mark = "ALLOWED" if v.allowed else "BLOCKED"
            print(f"[{mark:>7}] {src.id:<6} {src.host}")
            print(f"           {v.reason}")
            print(f"           licence: {src.licence}")
        print("\nNothing was fetched.")
        return 0

    rc = 0
    for src in chosen:
        report = collect(src, args.limit)
        if not report.verdict.allowed:
            print(f"[{src.id}] REFUSED — {report.verdict.reason}")
            rc = 1
            continue
        print(f"[{src.id}] {len(report.saved)} documents, {len(report.skipped)} skipped "
              f"-> data/external/scraped/{src.id}/")
        for s in report.skipped[:5]:
            print(f"    skip {s[:110]}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
