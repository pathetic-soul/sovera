"""The three scorers: lexical, dense, and the hybrid that ships (§9.2).

All three satisfy `Scorer` (core/routing/types.py) and none of them inherits
from another — the shared machinery is in `features.py`, not in a base class.

VRAM: 0 GB, all three. The lexical scorer holds ~110 sparse centroids in RAM.
The dense scorer loads a 33M-parameter encoder on **CPU** (`core/embed.py`) and
costs ~0.9 s once at construction, ~22 ms per query. That is the one design
decision in this file worth defending out loud: putting a 33M encoder on the
GPU to save 20 ms would spend the scarce resource (§4.1's 5.2 GB, with an 8B
model resident) to save the abundant one.

MEASURED, held out on 66 prompts, against the §13 gate of 90%:

    ExemplarScorer (lexical)   75.8%
    HybridScorer   (w=0.8)     80.3%   <- ships

TRAIN LOO-CV at w=0.8: lexical 74.5%, dense 84.5%, hybrid 85.5%.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

from core.routing.features import _features, _l2, _standardise
from core.routing.types import Row


class ExemplarScorer:
    """Nearest-exemplar cosine over TF-IDF vectors, weighted to favour intent.

    §9.2 specifies embedding similarity. This is lexical instead, for the reason
    §9.4 already gives: refinery language is dominated by codes and tags that
    dense models blur together. It also costs 0 GB of RAM and needs no staged
    weights. Accuracy is measured in tests/test_router.py — if that number ever
    drops below the §13 gate, swap this class for a bge-m3 ONNX encoder and
    leave every caller untouched.

    Class-IDF, not just document-IDF, is the one choice here that earned its
    keep. Every class talks about thickness, inspection, reports and API codes —
    those nouns are the topic, not the task. A term appearing in all 11 classes
    is damped to ~1.0; one appearing in a single class ("handwritten", "draft",
    "summarise") is amplified. Without it the router reads the subject matter
    and ignores what the user actually asked for.

    Aggregation (class centroid) and the `@start` weighting were both selected
    by leave-one-out CV on the training split only — never against the held-out
    set. Nearest-exemplar scored 10 points worse (too brittle at ~10 exemplars
    per class), and up-weighting the opening phrase was worse still.

    MEASURED: 75.8% held out (74.5% TRAIN LOO), against the >=90% gate in §13.

    NO LONGER THE DEFAULT. The prediction two paragraphs up came true and the
    fix was the one named there: `DenseScorer` below is that encoder, and
    `HybridScorer` blends the two for 80.3% held out. This class stays for two
    reasons that are not sentiment — it is half of the hybrid, and it is the
    fallback when staged weights are missing (§2.1), where it keeps the router
    alive at a known, measured cost. See AGENTS.md §16.
    """

    def __init__(self, rows: Iterable[Row]) -> None:
        pairs = list(rows)
        if not pairs:
            raise ValueError("no exemplars: cannot build a router")
        docs = [_features(p) for p, _ in pairs]
        labels = [label for _, label in pairs]
        classes = sorted(set(labels))
        n_docs, n_classes = len(docs), len(classes)

        doc_freq = Counter(term for doc in docs for term in set(doc))
        class_freq: Counter[str] = Counter()
        for cls in classes:
            terms: set[str] = set()
            for doc, label in zip(docs, labels):
                if label == cls:
                    terms |= set(doc)
            class_freq.update(terms)

        self.idf = {
            term: (math.log((n_docs + 1) / (freq + 1)) + 1.0)
            * (math.log(n_classes / class_freq[term]) + 1.0)
            for term, freq in doc_freq.items()
        }
        centroids: dict[str, dict[str, float]] = {}
        for doc, label in zip(docs, labels):
            bucket = centroids.setdefault(label, {})
            for term, count in Counter(doc).items():
                bucket[term] = bucket.get(term, 0.0) + self._weight(term, count)
        self.centroids = {label: _l2(vec) for label, vec in centroids.items()}
        self.labels = tuple(classes)

    def _weight(self, term: str, count: int) -> float:
        return (1.0 + math.log(count)) * self.idf.get(term, 1.0)

    def score(self, text: str, allowed: Iterable[str] | None = None) -> dict[str, float]:
        query = _l2(
            {t: self._weight(t, c) for t, c in Counter(_features(text)).items() if t in self.idf}
        )
        labels = tuple(allowed) if allowed is not None else self.labels
        return {
            label: round(
                sum(w * self.centroids[label].get(term, 0.0) for term, w in query.items()), 4
            )
            for label in labels
            if label in self.centroids
        }


class DenseScorer:
    """Nearest-centroid cosine over frozen sentence embeddings (§9.2).

    Deliberately mirrors `ExemplarScorer`'s shape: class centroids rather than
    nearest-exemplar, because LOO-CV on the training split picked centroids for
    the lexical scorer for a reason that applies here too — at ~16 exemplars per
    class, a single odd neighbour swings the answer.

    What this buys over lexical matching is paraphrase. "Work out the corrosion
    rate", "compute the wall loss per year" and "what's the metal loss annually"
    share almost no tokens, so TF-IDF sees three unrelated requests; the encoder
    sees one intent. That is precisely the failure mode behind the 16 held-out
    misses, and why §9.2 asked for embeddings first.

    What it costs: ~0.9 s of load at construction and ~22 ms per query, both on
    CPU, both measured. 0 GB VRAM.
    """

    def __init__(self, rows: Iterable[Row]) -> None:
        from core.embed import get_encoder

        pairs = list(rows)
        if not pairs:
            raise ValueError("no exemplars: cannot build a router")
        vectors = get_encoder().encode([p for p, _ in pairs])
        buckets: dict[str, list[int]] = {}
        for i, (_, label) in enumerate(pairs):
            buckets.setdefault(label, []).append(i)
        self.centroids: dict[str, list[float]] = {}
        for label, idx in buckets.items():
            centroid = vectors[idx].mean(axis=0)
            norm = float((centroid @ centroid) ** 0.5) or 1.0
            self.centroids[label] = (centroid / norm).tolist()
        self.labels = tuple(sorted(buckets))

    def score(self, text: str, allowed: Iterable[str] | None = None) -> dict[str, float]:
        from core.embed import get_encoder

        query = get_encoder().encode([text])[0]
        labels = tuple(allowed) if allowed is not None else self.labels
        return {
            label: round(float(sum(q * c for q, c in zip(query, self.centroids[label]))), 4)
            for label in labels
            if label in self.centroids
        }


class HybridScorer:
    """Lexical + dense, blended on standardised scores. The shipped router.

    Neither half is redundant, and the reason is domain-specific. Refinery
    language is full of tags and codes — V-2301, API 510, PSV, MOC — that a
    dense model blurs toward their neighbours and TF-IDF nails exactly (that is
    the argument `ExemplarScorer` makes for going lexical in the first place,
    and it was right). Paraphrased intent is the mirror image: TF-IDF misses it,
    the encoder catches it. They fail on different prompts, which is the only
    condition under which an ensemble is worth its complexity.

    `weight` is the share given to the dense half. It is selected by
    leave-one-out CV on the training split only, never against held-out — see
    core/routing/split.py for why that distinction is the whole point of
    quoting an accuracy number at all.
    """

    def __init__(self, rows: Iterable[Row], weight: float = 0.5) -> None:
        pairs = list(rows)
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"weight must be in [0, 1], got {weight}")
        self.weight = weight
        self.lexical = ExemplarScorer(pairs)
        self.dense = DenseScorer(pairs)
        self.labels = self.lexical.labels

    def score(self, text: str, allowed: Iterable[str] | None = None) -> dict[str, float]:
        lex = _standardise(self.lexical.score(text, allowed))
        dense = _standardise(self.dense.score(text, allowed))
        w = self.weight
        return {
            label: round((1.0 - w) * lex.get(label, 0.0) + w * dense.get(label, 0.0), 4)
            for label in lex.keys() | dense.keys()
        }
