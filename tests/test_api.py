"""The API surface is assembled from routers, and the assembly is asserted.

A split into routers is only safe if the resulting app exposes the same paths.
This test is the net: it lists the routes the UI actually calls and fails if any
of them stops being served, which is the one way this refactor could break the
demo without breaking a unit test.

Coverage is asked of the composed app through Starlette's own dispatch contract
(`route.matches(scope)`) rather than by walking `app.routes` for a `.path`
attribute: this FastAPI version wraps an included router's routes lazily
(`fastapi.routing._IncludedRouter`), so the top-level list no longer flattens to
plain leaf routes the way it did before the router split — but every route,
wrapped or not, still answers `matches()`, because that is the same contract
Starlette itself uses to dispatch a real request. Duplicate-path detection is
therefore checked at the source instead (each router's own, un-included
`.routes`), where every entry is still a plain, comparable route object.
"""

from __future__ import annotations

from starlette.routing import Match
from starlette.types import Scope

from core.api import ROUTERS
from core.orchestrator import app

EXPECTED_PATHS = {
    "/",
    "/api/firewall/rules",
    "/api/egress-test",
    "/api/registry",
    "/api/registry/reload",
    "/api/route",
    "/api/tools",
    "/api/backend",
    "/api/artifact",
    "/api/audit/verify",
    "/ws/agent",
    "/ws/sovereignty",
}


def _is_served(path: str, *, websocket: bool) -> bool:
    scope: Scope = {"type": "websocket" if websocket else "http", "method": "GET", "path": path}
    return any(route.matches(scope)[0] != Match.NONE for route in app.routes)


def test_every_endpoint_the_ui_calls_is_served() -> None:
    missing = {p for p in EXPECTED_PATHS if not _is_served(p, websocket=p.startswith("/ws/"))}
    assert not missing, f"routes lost in the router split: {sorted(missing)}"


def test_no_duplicate_paths() -> None:
    """Two routers registering the same path is a silent shadowing bug."""
    declared = [r.path for r in app.routes if hasattr(r, "path")]
    for api_router in ROUTERS:
        declared += [r.path for r in api_router.routes]
    duplicates = {p for p in declared if declared.count(p) > 1}
    assert not duplicates, f"path registered twice: {sorted(duplicates)}"
