"""BM25 + dense indices over a fixed chunk list (AGENTS.md §9.4).

CPU only. The dense half reuses `core.embed.Encoder` (bge-small-en-v1.5,
already staged for the §9.2 router) rather than adding a second model —
one small encoder, two consumers, 0 extra GB of the 5.2 GB VRAM budget
since it never touches the GPU.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from core.embed import get_encoder
from retrieval.chunker import Chunk

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-]*")


def _tokenize(text: str) -> list[str]:
    """Lowercase word-ish tokens. Hyphens kept: equipment tags like `V-2301`
    and `PSV-4402` are exactly what §9.4 says pure vector search loses, and
    BM25 only catches them if the tokenizer does not split them apart."""
    return _TOKEN.findall(text.lower())


@dataclass
class Index:
    chunks: list[Chunk]
    bm25: BM25Okapi | None  # None only for an empty corpus — see build()
    vectors: np.ndarray  # (n_chunks, DIM), L2-normalised

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "Index":
        # BM25Okapi divides by corpus_size while indexing (rank_bm25 has no
        # guard of its own), so an empty corpus — a fresh clone with
        # data/corpus/ missing, or this being called in a unit test with no
        # documents — must never reach it.
        bm25 = BM25Okapi([_tokenize(c.text) for c in chunks]) if chunks else None
        vectors = get_encoder().encode([c.text for c in chunks], max_length=256)
        return cls(chunks=chunks, bm25=bm25, vectors=vectors)

    def bm25_rank(self, query: str) -> list[int]:
        """Chunk indices, best lexical match first."""
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        return list(np.argsort(-scores))

    def dense_rank(self, query: str) -> list[int]:
        """Chunk indices, best cosine match first. Vectors are pre-normalised
        (core/embed.py), so the dot product against a normalised query vector
        is the cosine similarity directly."""
        q = get_encoder().encode([query], max_length=64)[0]
        scores = self.vectors @ q
        return list(np.argsort(-scores))
