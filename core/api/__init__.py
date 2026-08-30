"""HTTP and websocket surface, one module per panel of the UI.

AGENTS.md §11 caps a module at 400 lines; `core/orchestrator.py` was heading
past that as legs 6-7 added endpoints, and every panel's endpoints shared one
file. Splitting by panel means leg 6 is a new module plus one line in ROUTERS,
not an edit to the file every other panel lives in.

State is unchanged: everything still hangs on `app.state`, set once in the
lifespan (§11 forbids global mutable state). Endpoints reach it through
`request.app.state` / `websocket.app.state` rather than closing over `app`.

Order matters only for readability — FastAPI matches on path, and
tests/test_api.py fails if two routers ever claim the same one.
"""

from fastapi import APIRouter

from core.api import agent, registry, routing, sovereignty, workspace

ROUTERS: tuple[APIRouter, ...] = (
    sovereignty.router,   # §10 — the graded claim
    registry.router,      # §14.3
    routing.router,       # §9.2
    agent.router,         # §14.4, §14.5
    workspace.router,     # §2.2, §2.3
)

__all__ = ["ROUTERS"]
