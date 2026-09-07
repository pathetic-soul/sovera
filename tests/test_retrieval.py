"""Chunking and RRF fusion (AGENTS.md §9.4, leg 7)."""

from __future__ import annotations

from retrieval.chunker import chunk
from retrieval.hybrid import search
from retrieval.index import Index

# --- chunker -----------------------------------------------------------------

def test_chunk_splits_on_headings() -> None:
    text = "intro text\n\n## Findings\nsome findings\n\n## Recommendation\nfix it"
    chunks = chunk(text, "doc1")
    sections = [c.section for c in chunks]
    assert sections == ["", "Findings", "Recommendation"]
    assert chunks[0].text.strip() == "intro text"


def test_chunk_keeps_a_short_section_whole() -> None:
    text = "## Background\nshort paragraph, well under the target size."
    chunks = chunk(text, "doc1")
    assert len(chunks) == 1
    assert chunks[0].doc_id == "doc1"


def test_chunk_splits_an_oversized_section_with_overlap() -> None:
    # Three ~900-char paragraphs under one heading, target is 2000 -> should
    # pack two per piece and split, carrying the §9.4 overlap into piece two.
    para = "Reading " * 130  # ~1040 chars
    text = f"## Thickness readings\n{para}\n\n{para}\n\n{para}"
    chunks = chunk(text, "doc1")
    assert len(chunks) >= 2
    assert all(c.section == "Thickness readings" for c in chunks)
    # the tail of piece one reappears at the head of piece two
    overlap = chunks[0].text[-300:]
    assert overlap[:50] in chunks[1].text


def test_chunk_keeps_pre_heading_text() -> None:
    """A report's title-block metadata usually lives before the first heading
    and must not be silently dropped."""
    text = "Equipment tag: V-2301\n\n## Findings\nbody"
    chunks = chunk(text, "doc1")
    assert chunks[0].section == ""
    assert "V-2301" in chunks[0].text


# --- hybrid search (RRF fusion) -----------------------------------------------

def test_search_finds_an_exact_tag_lexically() -> None:
    """§9.4's whole justification for BM25+dense: a pure vector search can
    miss an exact equipment tag that lexical matching catches directly."""
    docs = [
        chunk("The corrosion rate on grid S7 exceeded the design allowance.", "a"),
        chunk("Equipment tag V-2301 failed its UT survey at grid S7, 8.9 mm.", "b"),
        chunk("General safety procedures for confined space entry apply here.", "c"),
    ]
    index = Index.build([c for group in docs for c in group])
    results = search(index, "V-2301", k=2)
    assert results
    assert results[0].chunk.doc_id == "b"


def test_search_returns_empty_for_an_empty_index() -> None:
    index = Index.build([])
    assert search(index, "anything") == []


def test_search_respects_k() -> None:
    docs = [chunk(f"document number {i} about pressure vessels", f"d{i}") for i in range(10)]
    index = Index.build([c for group in docs for c in group])
    assert len(search(index, "pressure vessel", k=3)) == 3
