"""The UI is vendored. This test is what makes that a fact rather than a claim.

AGENTS.md §2.1: "All frontend assets are vendored locally. No `<script
src="https://…">`. No Google Fonts. No Tailwind CDN." A reviewer cannot verify
that by reading a slide, and a single pasted <script> tag would void it silently.
So it is asserted over every file the browser is served.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"

ASSETS = sorted(p for p in WEB.rglob("*") if p.suffix in {".html", ".css", ".js"})

# Anything that would make the browser open a socket to somewhere that is not us.
_EXTERNAL = re.compile(
    r"""(https?:)?//(?!\s)                 # scheme-relative or absolute
        (?!127\.0\.0\.1|localhost)         # except loopback
        [a-z0-9.-]+\.[a-z]{2,}""",
    re.I | re.X,
)
_IMPORT_FROM_URL = re.compile(r"""import\s+.*?from\s+['"]https?://""", re.I | re.S)

# Browsers normalise excess leading slashes in an authority-taking scheme —
# `////cdn.example.com` resolves exactly like `//cdn.example.com` — so it is
# just as forbidden under §2.1. Collapse runs of 3+ slashes to the canonical
# two *before* scanning: otherwise a match on the 3rd/4th slash finds its own
# extra slashes sitting right before it and is wrongly excluded below as if
# it followed a `//` line-comment marker.
_MULTI_SLASH = re.compile(r"/{3,}")


def _external_hits(text: str) -> list[str]:
    text = _MULTI_SLASH.sub("//", text)
    # `//` inside a JS line comment is not a URL; strip the obvious case first.
    return [m.group(0) for m in _EXTERNAL.finditer(text)
            if not text[:m.start()].rstrip().endswith("//")]


def test_there_are_assets_to_check() -> None:
    """Guard against the glob silently matching nothing and the suite passing."""
    assert len(ASSETS) >= 6, f"expected the split UI, found {[p.name for p in ASSETS]}"


@pytest.mark.parametrize("path", ASSETS, ids=lambda p: p.name)
def test_no_external_references(path: Path) -> None:
    """§2.1 — no CDN, no webfont, no remote script."""
    hits = _external_hits(path.read_text(encoding="utf-8"))
    assert not hits, f"{path.name} references an external host: {hits}"


@pytest.mark.parametrize(
    ("text", "expect_hit"),
    [
        pytest.param('<script src="////cdn.example.com/x.js"></script>', True,
                      id="multi-slash-bypass"),
        pytest.param("// http://example.com", False, id="dead-js-comment"),
        pytest.param("ws://127.0.0.1:8080/ws", False, id="loopback"),
    ],
)
def test_external_reference_bypass_regression(text: str, expect_hit: bool) -> None:
    """Regression for a review finding: a run of 3+ leading slashes evaded
    detection because its own extra slashes satisfied the "preceded by a
    `//` comment" exclusion. §2.1 forbids the multi-slash form exactly as
    much as the two-slash one; the negative controls (a dead comment, our
    own loopback) must stay unflagged."""
    assert bool(_external_hits(text)) is expect_hit


@pytest.mark.parametrize("path", ASSETS, ids=lambda p: p.name)
def test_no_remote_es_module_import(path: Path) -> None:
    assert not _IMPORT_FROM_URL.search(path.read_text(encoding="utf-8"))


def test_index_loads_every_script_locally() -> None:
    """Every <script src> and <link href> must be an absolute local path."""
    html = (WEB / "index.html").read_text(encoding="utf-8")
    srcs = re.findall(r"""<(?:script|link)[^>]*?(?:src|href)=["']([^"']+)["']""", html, re.I)
    assert srcs, "index.html loads no external files — did the split happen?"
    for src in srcs:
        assert src.startswith("/static/"), f"non-local asset reference: {src}"


def test_no_build_step_was_introduced() -> None:
    """§7: no npm build step. A package.json under web/ means one appeared."""
    assert not (WEB / "package.json").exists()
    assert not (WEB / "node_modules").exists()
