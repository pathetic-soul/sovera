"""Reading back what the agent produced: artifacts and the audit chain.

AGENTS.md §2.2, §2.3. CPU only, 0 GB VRAM.

Grouped together because both are the same question from a reviewer's side —
"show me what it did, and prove the record was not edited afterwards."
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from tools.base import JailBreak, resolve_in_jail

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "workspace"

router = APIRouter()


@router.get("/api/artifact")
def artifact(path: str) -> FileResponse:
    """Download a deliverable. Jailed by the same function the tools use, so a
    crafted query string cannot read outside workspace/ (§2.3)."""
    try:
        target = resolve_in_jail(WORKSPACE, path)
    except JailBreak as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"no such artifact: {path}")
    return FileResponse(target, filename=target.name)


@router.get("/api/audit/verify")
def audit_verify(request: Request) -> dict[str, Any]:
    ok, broken = request.app.state.audit.verify()
    return {"ok": ok, "broken_at": broken}
