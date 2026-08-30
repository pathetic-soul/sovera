"""The §9.2 hard overrides — which routes stay on the table, and why.

CPU only, 0 GB VRAM, four regex searches per prompt.

These run *before* any scorer and they short-circuit nothing: they constrain the
candidate set and hand it to the scorer, which still has to choose within it.
That distinction is what keeps a decision explainable — an attached drawing
narrows the field to the three vision routes, and the scorer still picks between
`drawing_qa` and `handwriting`, so the `reason` string in the UI can show both
the override and the margin.

Kept in its own module because these rules are the part of routing a domain
expert can review without reading any machine learning. A refinery engineer can
be handed this file and asked "is that what you meant by a test?".
"""

from __future__ import annotations

import re

from core.routing.types import Modality, TaskType

VISION_ROUTES: tuple[TaskType, ...] = ("scan_understanding", "drawing_qa", "handwriting")
DOC_ROUTES: tuple[TaskType, ...] = ("approval_note", "summarize")

# "test" is NOT a run-verb here. In a refinery, "PSV test date", "hydrotest" and
# "test certificate" are everywhere; §9.2's literal rule sends all of them to the
# coder. We require a run/execute/debug verb AND a code noun to co-occur.
_RUN_VERB = re.compile(r"\b(run|execute|debug|traceback)\b", re.I)
_CODE_NOUN = re.compile(
    r"\b(script|code|program|python|snippet|function|pytest|unit tests?|stack trace)\b", re.I
)
_REPAIR = re.compile(
    r"\b(fix|fixed|repair|debug|broken|breaks|crash(es|ed)?|bug|wrong|error|fails?|failing|hangs?|patch|correct)\b",
    re.I,
)
_XLSX = re.compile(r"\.xlsx\b|\bexcel\b|\bspreadsheet\b", re.I)
_DOCX = re.compile(r"\.docx\b|\bword document\b", re.I)


def allowed_routes(text: str, modality: Modality) -> tuple[tuple[str, ...] | None, str | None]:
    """The §9.2 hard overrides: which routes stay on the table, and why.

    Extracted from `Classifier.classify` so that offline tuning scores prompts
    under the *same* constraints the live router applies. A tuning harness that
    reimplemented these rules would drift from them silently, and every accuracy
    number measured after that point would be measuring a router that does not
    exist. Returns `(None, None)` when nothing is overridden, meaning all routes
    remain candidates.
    """
    if modality in ("image", "pdf", "mixed"):
        return VISION_ROUTES, f"{modality} attached -> vision routes only"
    if _RUN_VERB.search(text) and _CODE_NOUN.search(text):
        allowed = ("code_fix", "code_write") if _REPAIR.search(text) else ("code_write",)
        return allowed, "run/execute + code noun -> code routes only"
    if _XLSX.search(text):
        return ("spreadsheet",), "explicit .xlsx/spreadsheet -> spreadsheet deliverable"
    if _DOCX.search(text):
        return DOC_ROUTES, "explicit .docx -> document deliverable routes only"
    return None, None
