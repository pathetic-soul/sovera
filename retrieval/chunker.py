"""Markdown -> chunks, respecting heading boundaries (AGENTS.md §9.4).

CPU only, 0 GB VRAM. §9.4: "Chunk: 400-600 tokens, 15% overlap, respect
heading boundaries." Tokens are approximated at ~4 chars/token, the same
rule of thumb `core/router.py:estimate_ctx` already uses, so a size chosen
here means the same thing it means everywhere else in this codebase.
"""

from __future__ import annotations

import re
from typing import NamedTuple

TARGET_CHARS = 2000  # ~500 tokens at ~4 chars/token, the middle of the 400-600 band
OVERLAP_CHARS = 300  # 15% of TARGET_CHARS
_HEADING = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)


class Chunk(NamedTuple):
    doc_id: str
    section: str
    text: str


def chunk(text: str, doc_id: str) -> list[Chunk]:
    """Split on markdown headings first; only sub-split a section that alone
    exceeds the target size, so a short table or paragraph never gets cut
    mid-thought for a limit it was never close to."""
    sections = _split_by_heading(text)
    out: list[Chunk] = []
    for heading, body in sections:
        body = body.strip()
        if not body:
            continue
        if len(body) <= TARGET_CHARS:
            out.append(Chunk(doc_id, heading, body))
            continue
        for piece in _split_oversized(body):
            out.append(Chunk(doc_id, heading, piece))
    return out


def _split_by_heading(text: str) -> list[tuple[str, str]]:
    """(heading, body) pairs. Text before the first heading gets "" as its
    heading rather than being dropped — a report's title-block metadata
    (equipment tag, inspector, date) usually lives there."""
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [("", text)]
    out: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        out.append(("", text[: matches[0].start()]))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((m.group(1).strip(), text[m.end() : end]))
    return out


def _split_oversized(body: str) -> list[str]:
    """Pack paragraphs up to TARGET_CHARS, repeating the tail OVERLAP_CHARS of
    each piece as the head of the next — the overlap AGENTS.md §9.4 asks for,
    so a fact split across a paragraph boundary still appears whole in at
    least one chunk."""
    paras = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
    pieces: list[str] = []
    current = ""
    for para in paras:
        if current and len(current) + len(para) + 2 > TARGET_CHARS:
            pieces.append(current)
            current = current[-OVERLAP_CHARS:] + "\n\n" + para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current.strip():
        pieces.append(current)
    return pieces or [body]
