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


def test_there_are_assets_to_check() -> None:
    """Guard against the glob silently matching nothing and the suite passing."""
    assert len(ASSETS) >= 6, f"expected the split UI, found {[p.name for p in ASSETS]}"


@pytest.mark.parametrize("path", ASSETS, ids=lambda p: p.name)
def test_no_external_references(path: Path) -> None:
    """§2.1 — no CDN, no webfont, no remote script."""
    text = path.read_text(encoding="utf-8")
    # `//` inside a JS line comment is not a URL; strip the obvious case first.
    hits = [m.group(0) for m in _EXTERNAL.finditer(text)
            if not text[:m.start()].rstrip().endswith("//")]
    assert not hits, f"{path.name} references an external host: {hits}"


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
