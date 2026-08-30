"""Deterministic task_type + modality detection (AGENTS.md §9.2, §2.3).

No LLM decides anything here. Same input, same output, every time — which is
what lets us quote a router accuracy number in §13 instead of a vibe.

CPU only, 0 GB VRAM. The lexical path is a dict lookup per token; the hybrid
adds ~22 ms of CPU encoding. Both are well under the 50 ms budget in §9.2.

This module is the facade: it wires the hard overrides (`overrides.py`) to a
scorer (`scorers.py`) and returns a `Classification` (`types.py`). The one
decision it owns is which scorer to build when nobody names one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from core.embed import EncoderUnavailable
from core.routing.features import detect_modality
from core.routing.overrides import allowed_routes
from core.routing.scorers import ExemplarScorer, HybridScorer
from core.routing.types import Classification, Row, Scorer

ROOT = Path(__file__).resolve().parents[2]
EXEMPLARS = ROOT / "config" / "routing_exemplars.jsonl"

# Selected by leave-one-out CV on the training split only (finetune/tune_router.py).
# TRAIN LOO: lexical 74.5%, dense 84.5%, hybrid 85.5% at this value.
ROUTER_WEIGHT = 0.8


def load_exemplars(path: Path = EXEMPLARS) -> list[Row]:
    rows: list[Row] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                obj = json.loads(line)
                rows.append((obj["prompt"], obj["task_type"]))
    return rows


class Classifier:
    def __init__(self, scorer: Scorer | None = None) -> None:
        self.scorer = scorer or self._default_scorer()

    @staticmethod
    def _default_scorer() -> Scorer:
        """Hybrid when the encoder is staged, lexical when it is not.

        The fallback is not defensive padding — it is the difference between a
        workbench that degrades and one that will not start. Staged weights are
        a build-time artifact (§2.1), so a fresh clone, a machine where the
        stage step has not run yet, or a partial copy would otherwise take the
        router down and the whole loop with it. Losing ~5 points of accuracy is
        recoverable; refusing to classify anything is not.
        """
        rows = load_exemplars()
        try:
            return HybridScorer(rows, weight=ROUTER_WEIGHT)
        except EncoderUnavailable:
            import warnings

            warnings.warn(
                "router encoder not staged; falling back to the lexical scorer "
                "(~5 points less accurate). Run: python -m core.embed --stage",
                RuntimeWarning, stacklevel=2,
            )
            return ExemplarScorer(rows)

    def classify(self, text: str, attachments: Iterable[str] | None = None) -> Classification:
        """Hard overrides constrain the candidate set; the scorer picks within it.

        Constraining rather than short-circuiting keeps the decision explainable:
        an attached drawing narrows the field to the three vision routes, and the
        scorer still has to choose between drawing_qa and handwriting.
        """
        modality = detect_modality(attachments)
        allowed, override = allowed_routes(text, modality)
        scores = self.scorer.score(text, allowed)
        best = max(scores, key=lambda k: scores[k]) if scores else "qa"
        return Classification(
            task_type=best,  # type: ignore[arg-type]
            modality=modality,
            scores=scores,
            override=override,
        )
