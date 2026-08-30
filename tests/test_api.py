"""The API surface is assembled from routers, and the assembly is asserted.

A split into routers is only safe if the resulting app exposes the same paths.
This test is the net: it lists the routes the UI actually calls and fails if any
of them stops being served, which is the one way this refactor could break the
demo without breaking a unit test.
"""

from __future__ import annotations

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


def test_every_endpoint_the_ui_calls_is_served() -> None:
    served = {getattr(r, "path", "") for r in app.routes}
    missing = EXPECTED_PATHS - served
    assert not missing, f"routes lost in the router split: {sorted(missing)}"


def test_no_duplicate_paths() -> None:
    """Two routers registering the same path is a silent shadowing bug."""
    paths = [getattr(r, "path", "") for r in app.routes if getattr(r, "path", "") != ""]
    duplicates = {p for p in paths if paths.count(p) > 1}
    assert not duplicates, f"path registered twice: {sorted(duplicates)}"
