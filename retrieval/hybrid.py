"""RRF fusion over BM25 + dense rankings (AGENTS.md §9.4).

"Hybrid BM25 + dense with RRF fusion is mandatory, not optional. Pure vector
search fails on refinery equipment tags (10-P-101A, V-2301, PSV-4402).
Lexical matching is what makes this usable on their documents."

No cross-encoder rerank here (§9.4 asks for top-30 -> top-5 via a reranker
model): that is a second staged CPU model this leg does not add. RRF fusion
of two independent rankings is the fallback that ships without one — see
`tools/kb_search.py` for where that cut is spent back to the user.
"""

from __future__ import annotations

from typing import NamedTuple

from retrieval.chunker import Chunk
from retrieval.index import Index

RRF_K = 60  # the constant from the original Reciprocal Rank Fusion paper


class ScoredChunk(NamedTuple):
    chunk: Chunk
    score: float


def search(index: Index, query: str, k: int = 5) -> list[ScoredChunk]:
    """Fuse the two rankings and return the top `k` chunks.

    RRF: score(d) = sum over rankers of 1 / (RRF_K + rank(d)). Rank-based, not
    score-based, is the point — it needs no calibration between BM25's
    unbounded scores and dense cosine's [-1, 1] range, which is exactly the
    mismatch `core/routing/features.py:_standardise` documents for the router's
    own hybrid scorer.
    """
    if not index.chunks:
        return []
    fused = [0.0] * len(index.chunks)
    for ranking in (index.bm25_rank(query), index.dense_rank(query)):
        for rank, idx in enumerate(ranking):
            fused[idx] += 1.0 / (RRF_K + rank + 1)
    order = sorted(range(len(index.chunks)), key=lambda i: -fused[i])
    return [ScoredChunk(index.chunks[i], fused[i]) for i in order[:k]]
