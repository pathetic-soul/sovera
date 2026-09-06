"""The agent run channel, the tool roster, and backend health (§14.4, §14.5).

GPU: this module holds no VRAM itself, but `/ws/agent` is what drives the
backend, so it is the endpoint that decides how much VRAM is in use. One model
per run — the route resolves once, up front, so a single run never swaps
(§4.2.1).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from core.agent import Agent, AgentStep
from tools.base import Tool
from tools.registry import tool_specs

router = APIRouter()


@router.get("/api/tools")
def tools_snapshot(request: Request) -> dict[str, Any]:
    """§14.4 — the roster, and which of them the human gate covers (§2.4)."""
    tools: dict[str, Tool] = request.app.state.tools
    return {"tools": tool_specs(tools)}


@router.get("/api/backend")
def backend_status(request: Request) -> dict[str, Any]:
    """Is Ollama up? Rendered in the UI so a dead daemon is caught before the
    demo, not as a stack trace mid-run."""
    ok, note = request.app.state.backend.available()
    return {
        "ok": ok,
        "note": note,
        "resident": request.app.state.resident,
        # So the panel's auto-approve box opens in the state the server is
        # actually configured for, rather than claiming a gate that is not armed.
        "auto_approve": request.app.state.settings.agent.auto_approve,
    }


@router.websocket("/ws/agent")
async def agent_ws(ws: WebSocket) -> None:
    """One run per connection (§14.4, §14.5).

    A websocket rather than SSE because the human gate is bidirectional: the
    loop stops at an approval_request and cannot continue until the reviewer
    answers on the same channel. Approval over a separate POST would need
    run-id bookkeeping to serve one operator on one laptop.
    """
    await ws.accept()
    agent: Agent = ws.app.state.agent

    async def approve(step: AgentStep) -> bool:
        # Safe to block here: the client has just been sent approval_request
        # and the run is suspended until it replies.
        reply = await ws.receive_json()
        return bool(reply.get("approve"))

    try:
        req = await ws.receive_json()
        task = str(req.get("text", "")).strip()
        attachments = [str(a) for a in (req.get("attachments") or [])]
        # Absent means "use the configured default" (§2.4), not "auto-approve".
        raw_auto = req.get("auto_approve")
        auto_approve = None if raw_auto is None else bool(raw_auto)
        if not task:
            await ws.send_json({"type": "error", "data": {"error": "empty task"}})
            return

        async for event in agent.run(
            task, attachments, approve, ws.app.state.resident, auto_approve
        ):
            if event.type == "route":
                ws.app.state.resident = event.data["model_id"]
            await ws.send_json(event.model_dump())
        await ws.send_json({"type": "done", "data": {}})
    except WebSocketDisconnect:
        return
