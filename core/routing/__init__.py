"""Routing: how a task becomes a model choice (AGENTS.md §9.2, §2.3, §13).

    text + attachments
        -> overrides.allowed_routes   which routes are even permissible
        -> scorers.*.score            which of those the language points at
        -> classifier.Classification  task_type + modality + scores + why
        -> core/router.RouteDecision  + registry lookup = a model id

Deterministic end to end. No LLM is consulted, no weights are loaded on the
lexical path, and the whole thing runs *before* Ollama is asked for anything —
which is what lets the UI render the decision and its rationale while a model is
still loading (§4.2.3), turning a 2-5 s swap into an explanation rather than a
stall.

This was one 384-line module against §11's 400-line cap, holding five concerns.
Split by concern so legs 6-7 (new modalities, new routes) extend one file each:

    types.py       TaskType, Modality, Classification, the Scorer protocol
    features.py    text -> terms, L2, z-score, attachment -> modality
    overrides.py   the §9.2 hard rules — reviewable by a domain expert alone
    scorers.py     lexical, dense, hybrid
    classifier.py  the facade, and which scorer to build by default
    split.py       train/held-out and the §13 accuracy measurement

Import from `core.routing` for the public surface; import a submodule directly
only for the internals (`_standardise`, the scorer classes) that the tuner and
the tests legitimately reach for.
"""

from core.routing.classifier import EXEMPLARS, ROUTER_WEIGHT, Classifier, load_exemplars
from core.routing.features import detect_modality
from core.routing.overrides import DOC_ROUTES, VISION_ROUTES, allowed_routes
from core.routing.scorers import DenseScorer, ExemplarScorer, HybridScorer
from core.routing.split import (
    ACCURACY_GATE,
    HELD_OUT,
    ROWS,
    TRAIN,
    held_out_accuracy,
    split,
)
from core.routing.types import Classification, Modality, Row, Scorer, TaskType

__all__ = [
    "ACCURACY_GATE",
    "Classification",
    "Classifier",
    "DOC_ROUTES",
    "DenseScorer",
    "EXEMPLARS",
    "ExemplarScorer",
    "HELD_OUT",
    "HybridScorer",
    "Modality",
    "ROUTER_WEIGHT",
    "ROWS",
    "Row",
    "Scorer",
    "TRAIN",
    "TaskType",
    "VISION_ROUTES",
    "allowed_routes",
    "detect_modality",
    "held_out_accuracy",
    "load_exemplars",
    "split",
]
