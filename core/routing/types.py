"""The routing vocabulary: task types, modalities, and what a scorer must do.

AGENTS.md §8.2. Separated from the classifier so that `core/router.py`,
`core/registry.py`'s route strings and the tuner can all agree on the type of a
task without importing an encoder. Importing this module costs nothing — no
weights, no exemplars, no numpy.

`TaskType` is the contract §8.2 names, and it is not free-form: `models.yaml`
declares `routes:` using these exact strings, and `Registry.models_for(route)`
matches on them. Adding a task type means adding an exemplar class, a route on
some model, and a member here — in that order, or the router will happily
select a route no model can serve and fall through to `fallback`.
"""

from __future__ import annotations

from typing import Iterable, Literal, Protocol

from pydantic import BaseModel

TaskType = Literal[
    "plan", "qa", "summarize", "approval_note", "code_write", "code_fix",
    "scan_understanding", "drawing_qa", "handwriting", "calc", "spreadsheet",
]
Modality = Literal["text", "image", "pdf", "mixed"]

# (prompt, task_type) — the shape of one labelled exemplar.
Row = tuple[str, str]


class Classification(BaseModel):
    task_type: TaskType
    modality: Modality
    scores: dict[str, float]
    override: str | None = None


class Scorer(Protocol):
    """What `Classifier` needs from a scorer. Three implementations satisfy it.

    The Protocol rather than a base class is what lets `tests/test_router.py`
    and `finetune/tune_router.py` swap the lexical, dense and hybrid scorers
    through the same `Classifier` without any of them inheriting anything — and
    it is why `core/embed.py` can be absent at runtime without the router
    failing to start (§2.1 stages weights at build time; the lexical scorer is
    the fallback).
    """

    def score(self, text: str, allowed: Iterable[str] | None = None) -> dict[str, float]:
        ...
