"""The routing endpoint — deterministic, and it loads nothing.

AGENTS.md §2.3, §9.2. 0 GB VRAM, ~0.2 ms lexical / ~22 ms hybrid, all CPU.

No model is loaded and no LLM is consulted here, which is exactly what makes
§4.2.3's swap-masking possible: the decision and its rationale render while
Ollama is still loading the weights the decision selected.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from core.router import Router

router = APIRouter()


class RouteRequest(BaseModel):
    text: str
    attachments: list[str] = Field(default_factory=list)


@router.post("/api/route")
def route(req: RouteRequest, request: Request) -> dict[str, Any]:
    """Deterministic (§2.3): no model is loaded and no LLM is consulted here."""
    task_router: Router = request.app.state.router
    decision = task_router.route(req.text, req.attachments, request.app.state.resident)
    request.app.state.audit.append("model_call", {"phase": "route", **decision.model_dump()})
    request.app.state.resident = decision.model_id
    return decision.model_dump()
