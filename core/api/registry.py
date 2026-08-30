"""Registry endpoints — the §14.3 live model addition.

AGENTS.md §8.1, §14.3. CPU only, 0 GB VRAM: this reads YAML, it never loads
weights. Reloading is the whole of leg 3 — editing config/models.yaml and
pressing this button is the entire change, with no code edit and no restart.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from core.registry import Registry, ReloadResult

router = APIRouter()


@router.get("/api/registry")
def registry_snapshot(request: Request) -> dict[str, Any]:
    registry: Registry = request.app.state.registry
    return registry.snapshot()


@router.post("/api/registry/reload")
def registry_reload(request: Request) -> dict[str, Any]:
    """§14.3 — the live model addition. Editing models.yaml is the whole change."""
    registry: Registry = request.app.state.registry
    result: ReloadResult = registry.reload()
    request.app.state.audit.append(
        "approval", {"event": "registry_reload", **result.model_dump()}
    )
    return {**result.model_dump(), "snapshot": registry.snapshot()}
