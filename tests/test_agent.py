"""Agent loop: caps, repair, the human gate, and audit coverage (§8.4, §2.4, §2.2).

No model is loaded here. The backend is scripted, which is the only way to test
loop behaviour deterministically — a real 4B would make these tests flaky and
they would stop meaning anything.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from backends.base import BackendError, Completion, LLMBackend, Message
from core.agent import Agent, AgentEvent, AgentStep
from core.audit import AuditLog
from core.registry import Registry
from core.router import Router
from core.settings import AgentSettings
from tools.base import RunContext, Tool, ToolResult
from tools.fs_read import FsRead

MAX_STEPS = AgentSettings().max_steps


class ScriptedBackend(LLMBackend):
    """Replays canned replies. Raises if the loop asks for more than scripted —
    an over-run is a bug we want loud, not a hang."""

    def __init__(self, replies: list[str], tokens: int = 100) -> None:
        self.replies = list(replies)
        self.tokens = tokens
        self.calls: list[list[Message]] = []

    def chat(
        self,
        ref: str,
        messages: list[Message],
        *,
        max_ctx: int,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> Completion:
        self.calls.append(list(messages))
        if not self.replies:
            raise AssertionError("agent asked for more turns than the script provides")
        return Completion(
            text=self.replies.pop(0), model=ref,
            prompt_tokens=self.tokens, eval_tokens=0, ms=1,
        )


class EchoTool(Tool):
    name = "echo"
    description = "Echo a message back."
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"msg": {"type": "string", "description": "e.g. hello"}},
        "required": ["msg"],
    }
    requires_approval = False

    def run(self, args: dict[str, Any], ctx: RunContext) -> ToolResult:
        return ToolResult(ok=True, output=f"echo: {args['msg']}")


class GatedTool(EchoTool):
    name = "gated"
    requires_approval = True


def call(tool: str, **args: Any) -> str:
    return json.dumps({"thought": "t", "tool": tool, "args": args})


def finish(answer: str = "done") -> str:
    return json.dumps({"thought": "t", "answer": answer})


@pytest.fixture
def build(tmp_path: Path):  # type: ignore[no-untyped-def]
    def _build(replies: list[str], tools: list[Tool] | None = None, tokens: int = 100):  # type: ignore[no-untyped-def]
        ws = tmp_path / "workspace"
        (ws / "inbox").mkdir(parents=True, exist_ok=True)
        audit = AuditLog(ws / ".audit" / "audit.jsonl", "test")
        backend = ScriptedBackend(replies, tokens)
        registry = Registry()
        chosen = tools if tools is not None else [EchoTool()]
        agent = Agent(backend, Router(registry), {t.name: t for t in chosen}, ws, audit)
        return agent, backend, audit, ws

    return _build


def drain(agent: Agent, task: str, **kw: Any) -> list[AgentEvent]:
    """Run the loop to completion synchronously.

    asyncio.run here instead of pytest-asyncio: the whole need is "call an
    async generator from a sync test", and that is three lines, not a plugin.
    """

    async def go() -> list[AgentEvent]:
        return [ev async for ev in agent.run(task, **kw)]

    return asyncio.run(go())


def first_event(agent: Agent, task: str) -> AgentEvent:
    async def go() -> AgentEvent:
        return await anext(agent.run(task))

    return asyncio.run(go())


def kinds(events: list[AgentEvent]) -> list[str]:
    return [e.type for e in events]


# --- the happy path ---------------------------------------------------------

def test_reads_a_file_then_answers(build: Any) -> None:
    agent, _, _, ws = build([call("fs_read", path="inbox/r.md"), finish("8.9 mm")],
                            tools=[FsRead()])
    (ws / "inbox" / "r.md").write_text("min thickness 8.9 mm", encoding="utf-8")

    events = drain(agent, "What is the minimum thickness?")
    assert kinds(events) == ["route", "step", "final"]
    assert "8.9 mm" in events[1].data["observation"]
    assert events[-1].data["answer"] == "8.9 mm"


def test_route_is_emitted_before_any_model_call(build: Any) -> None:
    """§4.2.3 — the rationale must render while the weights are still loading,
    so it cannot wait on the first token."""
    agent, backend, _, _ = build([finish()])
    first = first_event(agent, "Summarise the inspection report")
    assert first.type == "route"
    assert backend.calls == [], "a model was called before the route was shown"
    assert first.data["reason"]


# --- caps (§8.4) ------------------------------------------------------------

def test_step_cap_stops_and_reports_partial(build: Any) -> None:
    agent, _, _, _ = build([call("echo", msg="again")] * (MAX_STEPS + 4))
    events = drain(agent, "loop forever")
    assert sum(1 for e in events if e.type == "step") == MAX_STEPS
    assert events[-1].data["halted"] is True
    assert "step cap" in events[-1].data["answer"]


def test_token_cap_stops_the_run(build: Any) -> None:
    # tokens/call chosen so the 1,000,000 cap (config/runtime.yaml) trips
    # within the 8 scripted replies, rather than exhausting the script first.
    agent, _, _, _ = build([call("echo", msg="x")] * 8, tokens=150_000)
    events = drain(agent, "burn tokens")
    assert events[-1].data["halted"] is True
    assert "token cap" in events[-1].data["answer"]
    assert sum(1 for e in events if e.type == "step") < MAX_STEPS


def test_observations_are_truncated_before_re_entering_context(build: Any) -> None:
    agent, backend, _, ws = build([call("fs_read", path="inbox/big.md"), finish()],
                                  tools=[FsRead()])
    (ws / "inbox" / "big.md").write_text("y" * 80_000, encoding="utf-8")
    drain(agent, "read it")

    last_context = "".join(m.content for m in backend.calls[-1])
    assert len(last_context) < 20_000, "an 8k window cannot absorb this (§4.2.5)"


# --- repair (§8.4, §12.6) ---------------------------------------------------

def test_one_repair_attempt_recovers_bad_json(build: Any) -> None:
    agent, backend, _, _ = build(["I think I should echo that!", finish("recovered")])
    events = drain(agent, "say hi")
    assert events[-1].type == "final"
    assert events[-1].data["answer"] == "recovered"
    assert len(backend.calls) == 2
    assert "rejected" in backend.calls[1][-1].content


def test_invalid_args_are_repaired_not_executed(build: Any) -> None:
    """§2.3 — reject, repair, retry; never execute unvalidated args."""
    agent, backend, _, _ = build([call("echo", wrong="x"), call("echo", msg="ok"), finish()])
    events = drain(agent, "echo something")
    step = next(e for e in events if e.type == "step")
    assert step.data["observation"] == "echo: ok"
    assert step.data["repaired"] is True
    assert "msg" in backend.calls[1][-1].content


def test_two_bad_replies_fail_loudly(build: Any) -> None:
    agent, _, _, _ = build(["not json", "still not json"])
    events = drain(agent, "confuse it")
    assert events[-1].type == "error"
    assert "twice" in events[-1].data["error"]


def test_unknown_tool_is_survivable(build: Any) -> None:
    agent, _, _, _ = build([call("teleport", msg="x"), call("echo", msg="ok"), finish()])
    events = drain(agent, "try a bad tool")
    assert events[-1].type == "final"


# --- the human gate (§2.4) --------------------------------------------------

def test_approved_tool_runs(build: Any) -> None:
    agent, _, audit, _ = build([call("gated", msg="ok"), finish()], tools=[GatedTool()])

    async def yes(step: AgentStep) -> bool:
        return True

    events = drain(agent, "do the gated thing", approve=yes)
    assert kinds(events) == ["route", "approval_request", "step", "final"]
    assert events[2].data["observation"] == "echo: ok"
    assert any(r.kind == "approval" and r.payload["granted"] for r in audit.tail())


def test_denied_tool_does_not_run(build: Any) -> None:
    ran: list[str] = []

    class Recording(GatedTool):
        def run(self, args: dict[str, Any], ctx: RunContext) -> ToolResult:
            ran.append(args["msg"])
            return ToolResult(ok=True, output="should not happen")

    agent, _, audit, _ = build(
        [call("gated", msg="nope"), finish("stopped")], tools=[Recording()]
    )

    async def no(step: AgentStep) -> bool:
        return False

    events = drain(agent, "do the gated thing", approve=no)
    assert ran == [], "a denied tool executed"
    assert "denied" in kinds(events)
    assert any(r.kind == "approval" and not r.payload["granted"] for r in audit.tail())


def test_no_approver_means_denied(build: Any) -> None:
    """§2.4 is a gate, not a default-yes. Forgetting to wire the UI must fail
    closed."""
    agent, _, _, _ = build([call("gated", msg="x"), finish()], tools=[GatedTool()])
    events = drain(agent, "do it")
    assert "denied" in kinds(events)


# --- auto mode: the §2.4 relaxation, and the honesty it owes the log ---------

def test_auto_approve_runs_the_gated_tool_without_an_approver(build: Any) -> None:
    agent, _, _, _ = build([call("gated", msg="ok"), finish()], tools=[GatedTool()])
    events = drain(agent, "do it", auto_approve=True)
    assert "denied" not in kinds(events)
    assert events[2].data["observation"] == "echo: ok"


def test_auto_approve_is_audited_as_auto_never_as_human(build: Any) -> None:
    """The whole basis for allowing this: the chain must not imply a human
    signed off when nobody did."""
    agent, _, audit, _ = build([call("gated", msg="ok"), finish()], tools=[GatedTool()])
    drain(agent, "do it", auto_approve=True)
    approvals = [r for r in audit.tail() if r.kind == "approval" and "granted_by" in r.payload]
    assert [r.payload["granted_by"] for r in approvals] == ["auto"]
    assert all(r.payload["granted"] for r in approvals)


def test_human_approval_is_audited_as_human(build: Any) -> None:
    agent, _, audit, _ = build([call("gated", msg="ok"), finish()], tools=[GatedTool()])

    async def yes(step: AgentStep) -> bool:
        return True

    drain(agent, "do it", approve=yes)
    approvals = [r for r in audit.tail() if r.kind == "approval" and "granted_by" in r.payload]
    assert [r.payload["granted_by"] for r in approvals] == ["human"]


def test_auto_approve_is_off_by_default(build: Any) -> None:
    """§2.4 stays armed unless someone explicitly relaxes it."""
    from core.settings import AgentSettings

    assert AgentSettings().auto_approve is False
    agent, _, _, _ = build([call("gated", msg="x"), finish()], tools=[GatedTool()])
    assert "denied" in kinds(drain(agent, "do it"))


def test_auto_approve_still_announces_the_gate(build: Any) -> None:
    """An unattended grant must be visible in the trace, not silent."""
    agent, _, _, _ = build([call("gated", msg="ok"), finish()], tools=[GatedTool()])
    events = drain(agent, "do it", auto_approve=True)
    gate = next(e for e in events if e.type == "approval_request")
    assert gate.data["auto"] is True


# --- failure surfaces -------------------------------------------------------

def test_backend_down_is_reported_not_swallowed(build: Any) -> None:
    class Dead(LLMBackend):
        def chat(self, ref: str, messages: list[Message], *, max_ctx: int,
                 temperature: float = 0.2, json_mode: bool = False) -> Completion:
            raise BackendError("ollama unreachable at http://127.0.0.1:11434")

    agent, _, _, _ = build([finish()])
    agent.backend = Dead()
    events = drain(agent, "anything")
    assert events[-1].type == "error"
    assert "unreachable" in events[-1].data["error"]


def test_a_raising_tool_becomes_an_observation(build: Any) -> None:
    class Broken(EchoTool):
        def run(self, args: dict[str, Any], ctx: RunContext) -> ToolResult:
            raise ValueError("kaboom")

    agent, _, audit, _ = build([call("echo", msg="x"), finish("carried on")],
                               tools=[Broken()])
    events = drain(agent, "trigger a bug")
    step = next(e for e in events if e.type == "step")
    assert "kaboom" in (step.data["observation"] or "")
    assert events[-1].data["answer"] == "carried on"
    assert any(r.kind == "error" for r in audit.tail())


def test_failed_tool_output_reaches_the_model(build: Any) -> None:
    """Regression: the loop used to observe only the error label, so a sandbox
    failure arrived as "exit 1" with the traceback discarded. The coder then
    resubmitted byte-identical code until the step cap. A traceback is
    repairable; a label is not."""

    class Failing(EchoTool):
        def run(self, args: dict[str, Any], ctx: RunContext) -> ToolResult:
            return ToolResult(
                ok=False,
                output="exit_code: 1\nstderr:\nModuleNotFoundError: No module named 'numpy'",
                error="exit 1",
            )

    agent, backend, _, _ = build(
        [call("echo", msg="x"), finish("fixed it")], tools=[Failing()]
    )
    events = drain(agent, "run something that fails")

    step = next(e for e in events if e.type == "step")
    assert "numpy" in (step.data["observation"] or ""), "traceback was dropped"
    assert "numpy" in backend.calls[-1][-1].content, "model never saw the traceback"


def test_multiline_code_in_json_is_parsed_not_rejected(build: Any) -> None:
    """Regression: strict JSON forbids literal control characters inside
    strings, so a model writing multi-line Python for py_sandbox produced
    unparseable output and burned the whole run on a repair. It was the one
    failure in an otherwise 7/8 measured run."""
    nl = chr(10)
    reply = ('{"thought": "compute it", "tool": "echo", "args": {"msg": "a = 1'
             + nl + 'print(a)"}}')
    agent, _, _, _ = build([reply, finish("done")])
    events = drain(agent, "run some code")

    step = next(e for e in events if e.type == "step")
    assert step.data["tool"] == "echo"
    assert step.data["repaired"] is False, "should parse first time, not need repair"
    assert events[-1].type == "final"
