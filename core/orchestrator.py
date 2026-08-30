"""FastAPI app — sovereignty panel, registry, router, agent (§10, §14.1-5).

Legs 1-3 touch no weights: routing is deterministic and happens before any
model is loaded, which is what makes §4.2.3's swap-masking possible — the
decision and its rationale render while Ollama is still loading.

Legs 4-5 are the first VRAM spend. One model resident at a time (§4.2.1); the
agent resolves its route once per run so a run never swaps mid-flight.

Loopback bind only. Run: `python -m core.orchestrator`
"""

from __future__ import annotations

import asyncio
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from backends.ollama_backend import OllamaBackend
from core.agent import Agent, AgentStep
from core.audit import AuditLog
from core.net_guard import attempt_egress, install_guard
from core.registry import Registry, ReloadResult
from core.router import Router
from sovereignty.monitor import DropWatcher
from tools.base import JailBreak, Tool, resolve_in_jail
from tools.registry import build_tools, tool_specs

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "web" / "index.html"
FIREWALL_RULES = ROOT / "sovereignty" / "firewall.ps1"
WORKSPACE = ROOT / "workspace"
SEED_CORPUS = ROOT / "data" / "corpus" / "inbox"
EGRESS_TARGET = ("api.openai.com", 443)


def _seed_workspace() -> None:
    """workspace/ is gitignored (it is agent-writable), so a fresh clone has no
    documents to demo against. Copy the sample corpus in once, never overwrite:
    anything the operator edited on the machine wins."""
    inbox = WORKSPACE / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    for src in SEED_CORPUS.glob("*.md"):
        if not (inbox / src.name).exists():
            shutil.copy2(src, inbox / src.name)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    audit = AuditLog(WORKSPACE / ".audit" / "audit.jsonl", uuid.uuid4().hex[:12])
    app.state.audit = audit
    app.state.watcher = DropWatcher()
    app.state.registry = Registry()
    app.state.router = Router(app.state.registry)
    # Which model Ollama currently holds, for swap_required (§4.2.3). The agent
    # updates it after each run.
    app.state.resident = None

    _seed_workspace()
    app.state.backend = OllamaBackend(audit)
    # The roster and the reason `calc` is a fourth tool against §14.4's "three
    # tools only" both live in tools/registry.py, which is also what the gate
    # runner and the trace tests build from. One list, four consumers.
    tools: dict[str, Tool] = build_tools()
    app.state.tools = tools
    app.state.agent = Agent(app.state.backend, app.state.router, tools, WORKSPACE, audit)

    install_guard(audit)
    audit.append("approval", {"event": "session_start", "leg": "agent_and_sandbox"})
    yield


app = FastAPI(title="Sovereign Workbench", lifespan=lifespan)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX)


@app.get("/api/firewall/rules", response_class=PlainTextResponse)
def firewall_rules() -> str:
    """§10.1 — the ruleset is displayed verbatim in the UI, not paraphrased."""
    return FIREWALL_RULES.read_text(encoding="utf-8")


@app.post("/api/egress-test")
def egress_test() -> dict[str, Any]:
    """Sync on purpose: attempt_egress blocks, FastAPI runs it in a threadpool."""
    host, port = EGRESS_TARGET
    return attempt_egress(host, port, app.state.audit)


class RouteRequest(BaseModel):
    text: str
    attachments: list[str] = Field(default_factory=list)


@app.get("/api/registry")
def registry_snapshot() -> dict[str, Any]:
    registry: Registry = app.state.registry
    return registry.snapshot()


@app.post("/api/registry/reload")
def registry_reload() -> dict[str, Any]:
    """§14.3 — the live model addition. Editing models.yaml is the whole change."""
    registry: Registry = app.state.registry
    result: ReloadResult = registry.reload()
    app.state.audit.append("approval", {"event": "registry_reload", **result.model_dump()})
    return {**result.model_dump(), "snapshot": registry.snapshot()}


@app.post("/api/route")
def route(req: RouteRequest) -> dict[str, Any]:
    """Deterministic (§2.3): no model is loaded and no LLM is consulted here."""
    router: Router = app.state.router
    decision = router.route(req.text, req.attachments, app.state.resident)
    app.state.audit.append("model_call", {"phase": "route", **decision.model_dump()})
    app.state.resident = decision.model_id
    return decision.model_dump()


@app.get("/api/tools")
def tools_snapshot() -> dict[str, Any]:
    """§14.4 — the roster, and which of them the human gate covers (§2.4)."""
    tools: dict[str, Tool] = app.state.tools
    return {"tools": tool_specs(tools)}


@app.get("/api/backend")
def backend_status() -> dict[str, Any]:
    """Is Ollama up? Rendered in the UI so a dead daemon is caught before the
    demo, not as a stack trace mid-run."""
    ok, note = app.state.backend.available()
    return {"ok": ok, "note": note, "resident": app.state.resident}


@app.get("/api/artifact")
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


@app.websocket("/ws/agent")
async def agent_ws(ws: WebSocket) -> None:
    """One run per connection (§14.4, §14.5).

    A websocket rather than SSE because the human gate is bidirectional: the
    loop stops at an approval_request and cannot continue until the reviewer
    answers on the same channel. Approval over a separate POST would need
    run-id bookkeeping to serve one operator on one laptop.
    """
    await ws.accept()
    agent: Agent = app.state.agent

    async def approve(step: AgentStep) -> bool:
        # Safe to block here: the client has just been sent approval_request
        # and the run is suspended until it replies.
        reply = await ws.receive_json()
        return bool(reply.get("approve"))

    try:
        req = await ws.receive_json()
        task = str(req.get("text", "")).strip()
        attachments = [str(a) for a in (req.get("attachments") or [])]
        if not task:
            await ws.send_json({"type": "error", "data": {"error": "empty task"}})
            return

        async for event in agent.run(task, attachments, approve, app.state.resident):
            if event.type == "route":
                app.state.resident = event.data["model_id"]
            await ws.send_json(event.model_dump())
        await ws.send_json({"type": "done", "data": {}})
    except WebSocketDisconnect:
        return


@app.get("/api/audit/verify")
def audit_verify() -> dict[str, Any]:
    ok, broken = app.state.audit.verify()
    return {"ok": ok, "broken_at": broken}


@app.websocket("/ws/sovereignty")
async def sovereignty(ws: WebSocket) -> None:
    await ws.accept()
    watcher: DropWatcher = app.state.watcher
    audit: AuditLog = app.state.audit
    available, why = watcher.available()
    try:
        while True:
            new = watcher.poll()
            await ws.send_json(
                {
                    "monitor_ok": available,
                    "monitor_note": why,
                    "total_drops": watcher.total,
                    "new_drops": new,
                    "audit": [r.model_dump() for r in audit.tail(15)],
                }
            )
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")
