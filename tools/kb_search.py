"""Hybrid retrieval over the reference corpus (AGENTS.md §9.4, leg 7, §8.3).

CPU only, 0 GB VRAM — the dense half reuses the router's already-staged
bge-small-en-v1.5 encoder (core/embed.py), so this tool costs nothing beyond
what §9.2 already pays.

No approval gate: like fs_read and ocr_read, this only reads and changes
nothing (§2.4 gates writes and execution).

Indexes `data/corpus/inbox/` (120 synthetic UT inspection reports, generated
by `finetune/corpus_inspection.py` for training/eval) and `data/corpus/filings/`
(38 real MRPL SEBI disclosures, machine-extracted from PDF). That is the
reference library this tool searches ACROSS; it is distinct from `workspace/`,
which `fs_read` reads one document at a time from what the user staged for a
specific task. Building the index (chunk + BM25 + encode) is paid once, on the
first call — subsequent calls reuse the module-level singleton, the same
lazy-singleton shape as `core.embed.get_encoder`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from retrieval.chunker import Chunk, chunk
from retrieval.hybrid import ScoredChunk, search
from retrieval.index import Index
from tools.base import RunContext, Tool, ToolResult

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIRS = (ROOT / "data" / "corpus" / "inbox", ROOT / "data" / "corpus" / "filings")
MAX_CHARS = 6000  # same context-budget cap as fs_read.py (§8.4)

_INDEX: Index | None = None


def _load_chunks() -> list[Chunk]:
    out: list[Chunk] = []
    for directory in CORPUS_DIRS:
        for path in sorted(directory.glob("*.md")):
            out.extend(chunk(path.read_text(encoding="utf-8", errors="replace"), path.stem))
    return out


def get_index() -> Index:
    """Process-wide singleton, built once. §11 forbids other global mutable
    state; this one is read-only after construction, like `core.embed`'s."""
    global _INDEX
    if _INDEX is None:
        _INDEX = Index.build(_load_chunks())
    return _INDEX


class KbSearch(Tool):
    name = "kb_search"
    description = (
        "Search the reference corpus (inspection reports, MRPL filings) for "
        "relevant passages. Hybrid lexical+semantic, finds exact tags "
        "(V-2301) and paraphrased matches alike. Use before answering "
        "questions the current workspace document does not cover."
    )
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "what to find, e.g. corrosion rate limits for CDU vessels",
            },
        },
        "required": ["query"],
    }
    requires_approval = False

    def run(self, args: dict[str, Any], ctx: RunContext) -> ToolResult:
        query = str(args["query"]).strip()
        if not query:
            return ToolResult(ok=False, output="", error="empty query")

        results: list[ScoredChunk] = search(get_index(), query, k=5)
        if not results:
            ctx.audit.append("tool_call", {"tool": self.name, "ok": True, "hits": 0})
            return ToolResult(ok=True, output="[no matching passages in the corpus]")

        parts = []
        for r in results:
            heading = f" — {r.chunk.section}" if r.chunk.section else ""
            parts.append(f"[{r.chunk.doc_id}{heading}]\n{r.chunk.text}")
        text = "\n\n---\n\n".join(parts)
        truncated = len(text) > MAX_CHARS
        if truncated:
            text = text[:MAX_CHARS] + f"\n…[truncated, {len(text)} chars total]"

        ctx.audit.append(
            "tool_call",
            {"tool": self.name, "ok": True, "query": query[:200], "hits": len(results),
             "sources": [r.chunk.doc_id for r in results], "truncated": truncated},
        )
        return ToolResult(ok=True, output=text)
