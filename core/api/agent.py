"""The agent run channel, the tool roster, and backend health (§14.4, §14.5).

GPU: this module holds no VRAM itself, but `/ws/agent` is what drives the
backend, so it is the endpoint that decides how much VRAM is in use. One model
per run — the route resolves once, up front, so a single run never swaps
(§4.2.1).
"""

from __future__ import annotations

import asyncio
import subprocess
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from core.agent import Agent, AgentStep
from tools.base import RunContext, Tool
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


def _kill_container(name: str) -> None:
    """`docker kill` by name, best-effort. Runs in a worker thread (called
    from the async kill handler via to_thread) so a hung docker CLI cannot
    block the websocket's own message loop. Failure just means the container
    already exited on its own — nothing to audit, nothing to raise."""
    try:
        subprocess.run(["docker", "kill", name], capture_output=True, timeout=10, check=False)
    except (subprocess.TimeoutExpired, OSError):
        pass


@router.websocket("/ws/agent")
async def agent_ws(ws: WebSocket) -> None:
    """One run per connection (§14.4, §14.5).

    A websocket rather than SSE because the human gate is bidirectional: the
    loop stops at an approval_request and cannot continue until the reviewer
    answers on the same channel. Approval over a separate POST would need
    run-id bookkeeping to serve one operator on one laptop.

    The operator kill switch adds a second, unsolicited message the client
    can send at any time — not just when an approval is pending — so incoming
    messages are demultiplexed by a background reader task into an approval
    queue and a stop event, rather than `approve()` being the only thing that
    ever calls `ws.receive_json()`.
    """
    await ws.accept()
    agent: Agent = ws.app.state.agent
    ctx = RunContext(workspace=agent.workspace, audit=agent.audit, session_id=agent.audit.session_id)
    stop_event = asyncio.Event()
    approvals: asyncio.Queue[bool] = asyncio.Queue()

    async def reader() -> None:
        """The sole reader of this websocket. `approve()` and the kill switch
        both consume from here instead of racing each other on `receive_json`.

        `finally: stop_event.set()` guarantees a client disconnect while
        `approve()` is pending cannot hang the run forever — without it,
        neither `approvals.get()` nor `stop_event.wait()` would ever resolve,
        since nothing else here would notice the socket died.
        """
        try:
            while True:
                msg = await ws.receive_json()
                if msg.get("stop"):
                    stop_event.set()
                    ctx.audit.append("approval", {"event": "operator_kill_requested"})
                    if ctx.active_container:
                        await asyncio.to_thread(_kill_container, ctx.active_container)
                    continue
                await approvals.put(bool(msg.get("approve")))
        finally:
            stop_event.set()

    async def approve(step: AgentStep) -> bool:
        # A stop that arrives while an approval is pending must not leave the
        # loop waiting forever for an answer that is never coming; treat it
        # as a denial so the loop's own next-iteration stop check can exit.
        get_answer = asyncio.ensure_future(approvals.get())
        wait_stop = asyncio.ensure_future(stop_event.wait())
        done, _ = await asyncio.wait(
            {get_answer, wait_stop}, return_when=asyncio.FIRST_COMPLETED
        )
        if get_answer in done:
            wait_stop.cancel()
            return get_answer.result()
        get_answer.cancel()
        return False

    reader_task: asyncio.Task[None] | None = None
    try:
        # This one receive happens before the reader task exists — the reader
        # must not start consuming until the initial task payload is claimed,
        # or the two would race on the same socket and could hand the task
        # payload to the wrong side.
        req = await ws.receive_json()
        task = str(req.get("text", "")).strip()
        attachments = [str(a) for a in (req.get("attachments") or [])]
        # Absent means "use the configured default" (§2.4), not "auto-approve".
        raw_auto = req.get("auto_approve")
        auto_approve = None if raw_auto is None else bool(raw_auto)
        if not task:
            await ws.send_json({"type": "error", "data": {"error": "empty task"}})
            return

        reader_task = asyncio.ensure_future(reader())
        async for event in agent.run(
            task, attachments, approve, ws.app.state.resident, auto_approve,
            stop_event, ctx,
        ):
            if event.type == "route":
                ws.app.state.resident = event.data["model_id"]
            await ws.send_json(event.model_dump())
        await ws.send_json({"type": "done", "data": {}})
    except WebSocketDisconnect:
        return
    finally:
        if reader_task is not None:
            reader_task.cancel()
