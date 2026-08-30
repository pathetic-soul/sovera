"""Text -> features, and the vector arithmetic the scorers share (§9.2).

CPU only, 0 GB VRAM. Featurising one prompt is a regex split and two zips —
microseconds, and it runs before any model is loaded, which is what makes
§4.2.3's swap-masking possible.

Nothing here knows about task types or models. It turns strings into bags of
terms and normalises numbers, and that is the whole of it — which is why the
lexical scorer, the dense scorer and the hybrid blend can all be built on top
without any of them importing each other.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

from core.routing.types import Modality

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_EXT = {".pdf"}


def _features(text: str) -> list[str]:
    """Unigrams, bigrams, and the opening phrase as its own feature.

    Bigrams carry the domain: 'approval note', 'title block', 'thickness
    reading' mean far more than their halves. The `@start` features exist
    because an English work request front-loads its intent — "Summarise…",
    "Calculate…", "Draft…", "What is…" — and that first word is the single most
    reliable signal of task_type, while everything after it is subject matter
    shared with every other route.
    """
    toks = re.findall(r"[a-z0-9]+", text.lower())
    feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    if toks:
        feats.append(f"@start:{toks[0]}")
    if len(toks) > 1:
        feats.append(f"@start2:{toks[0]}_{toks[1]}")
    return feats


def _l2(vec: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {k: v / norm for k, v in vec.items()}


def _standardise(scores: dict[str, float]) -> dict[str, float]:
    """Z-score across the candidate classes, so two scorers can be added up.

    This is the only non-obvious step in the hybrid and it is load-bearing.
    Lexical cosine over sparse TF-IDF lands around 0.1-0.5 with a wide spread;
    bge cosine lands around 0.6-0.9 with a narrow one, because every sentence
    in one domain is genuinely similar under a dense model. Blending the raw
    numbers therefore does not blend the opinions — it just lets whichever
    scorer has the larger spread win every time, and tuning the weight only
    changes which one that is.

    Standardising per query throws away the absolute scale and keeps the shape:
    each scorer contributes *how much it prefers this class over its own
    alternatives*, measured in its own standard deviations. A flat scorer (no
    opinion) contributes ~0 and lets the other decide, which is the behaviour
    you want at the exact moment one of them is out of its depth.
    """
    if not scores:
        return {}
    values = list(scores.values())
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    sd = math.sqrt(var)
    if sd < 1e-9:  # every class scored identically: no information to add
        return {k: 0.0 for k in scores}
    return {k: (v - mean) / sd for k, v in scores.items()}


def detect_modality(attachments: Iterable[str] | None) -> Modality:
    kinds = {
        "image" if Path(a).suffix.lower() in IMAGE_EXT
        else "pdf" if Path(a).suffix.lower() in PDF_EXT
        else "other"
        for a in (attachments or [])
    }
    kinds.discard("other")
    if kinds == {"image"}:
        return "image"
    if kinds == {"pdf"}:
        return "pdf"
    if kinds:
        return "mixed"
    return "text"
