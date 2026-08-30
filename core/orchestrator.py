"""FastAPI composition root — build the app, own the lifespan, mount the rest.

AGENTS.md §10, §14.1-5. Loopback bind only. Run: `python -m core.orchestrator`

Everything this module does is wiring. The endpoints live in `core/api/`, one
module per panel; the containment layers live in `sovereignty/`, `core/net_guard.py`
and `tools/py_sandbox.py`. What stays here is the single place where the audit
log, the registry, the router, the backend, the tools and the agent are
constructed — one lifespan, one set of objects on `app.state`, no globals (§11).

Legs 1-3 touch no weights: routing is deterministic and happens before any model
is loaded, which is what makes §4.2.3's swap-masking possible. Legs 4-5 are the
first VRAM spend; one model resident at a time (§4.2.1), and the agent resolves
its route once per run so a run never swaps mid-flight.
"""

from __future__ import annotations

import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backends.ollama_backend import OllamaBackend
from core.agent import Agent
from core.api import ROUTERS
from core.audit import AuditLog
from core.net_guard import install_guard
from core.registry import Registry
from core.router import Router
from core.settings import load_settings
from sovereignty.monitor import DropWatcher
from tools.base import Tool
from tools.registry import build_tools

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
INDEX = WEB / "index.html"
STATIC = WEB / "static"
WORKSPACE = ROOT / "workspace"
SEED_CORPUS = ROOT / "data" / "corpus" / "inbox"


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
    app.state.settings = load_settings()
    app.state.watcher = DropWatcher()
    app.state.registry = Registry()
    app.state.router = Router(app.state.registry)
    # Which model Ollama currently holds, for swap_required (§4.2.3). The agent
    # updates it after each run.
    app.state.resident = None

    _seed_workspace()
    app.state.backend = OllamaBackend(audit)
    tools: dict[str, Tool] = build_tools()
    app.state.tools = tools
    app.state.agent = Agent(
        app.state.backend, app.state.router, tools, WORKSPACE, audit,
        settings=app.state.settings.agent,
    )

    install_guard(audit)
    audit.append("approval", {"event": "session_start", "leg": "agent_and_sandbox"})
    yield


app = FastAPI(title="Sovereign Workbench", lifespan=lifespan)

for api_router in ROUTERS:
    app.include_router(api_router)

# Vendored locally, never a CDN (§2.1). StaticFiles reads off disk and opens no
# socket; tests/test_web.py asserts nothing in web/ references an external host.
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX)


if __name__ == "__main__":
    settings = load_settings()
    uvicorn.run(app, host=settings.server.host, port=settings.server.port, log_level="info")
