"""The agent loop — ReAct with a step cap, a token cap and a human gate.

AGENTS.md §8.4, §12, §14.4, §14.5. Written by hand in ~250 lines rather than
imported from LangChain (§7 forbids it) because on 6 GB the token budget has to
be controllable and the loop has to be readable by a judge.

VRAM: this module holds none itself, but it is what drives `backends/` and so
it is what decides how much VRAM is in use. One model per run — the route is
resolved once, up front, so a single run never triggers a swap (§4.2.1). The
route event is yielded before the first token is requested, which is what makes
the 2-5 s cold load in §4.2.3 read as an explanation instead of a stall.

Budget: 8 steps, 20k cumulative tokens, observations truncated to ~1500 tokens
before re-entering context. All three exist because an 8k window (§4.2.5) has
no room to absorb a runaway loop.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Literal

from pydantic import BaseModel, Field

from backends.base import BackendError, LLMBackend, Message
from core.audit import AuditLog
from core.router import RouteDecision, Router
from core.settings import AgentSettings
from tools.base import RunContext, Tool, ToolResult, validate_args

# §8.4's budget now lives in config/runtime.yaml (§11: config over constants).
# The values are unchanged; what changed is that a judge can alter them without
# a code edit. `AgentSettings()` carries the charter defaults, so a caller that
# passes nothing behaves exactly as this module did before.

# Approval callback: given the pending step, return True to let it run.
Approver = Callable[["AgentStep"], Awaitable[bool]]


class AgentStep(BaseModel):
    n: int
    thought: str
    tool: str | None = None
    args: dict[str, Any] | None = None
    observation: str | None = None
    tokens_used: int = 0
    repaired: bool = False
    artifacts: list[str] = Field(default_factory=list)


class AgentEvent(BaseModel):
    type: Literal["route", "step", "approval_request", "denied", "final", "error"]
    data: dict[str, Any]


SYSTEM = """You are an offline engineering assistant at an Indian oil refinery (MRPL).
You have no internet access. You work in small steps using tools.

Reply with ONE JSON object and nothing else.
To use a tool:  {"thought": "why", "tool": "name", "args": {...}}
To finish:      {"thought": "why", "answer": "the reply for the engineer"}

Rules:
- One tool per reply. Use only the parameters listed. Never nest objects.
- Read a document before answering questions about it. Do not guess its contents.
- If a scanned image or photo is attached, call ocr_read on it before answering.
  Never guess printed text from the image description alone.
- If the document does not state something, write UNKNOWN. Never invent a
  thickness reading, a date, an equipment tag or a code clause.
- For ANY number you must calculate, call calc. Never do arithmetic in
  your head and never re-derive a standard formula: calc has the
  inspection formulas built in and shows its working.
- When asked for a note or a document, call doc_write, then finish.

Tools:
{tools}

Example:
{"thought": "I must read the report before summarising it.", "tool": "fs_read", "args": {"path": "inbox/UT-2024-114.md"}}
{"thought": "I have the findings and have drafted the note.", "tool": "doc_write", "args": {"filename": "approval-note-V-2301", "title": "Approval Note - V-2301", "body": "## Background\\n- UT minimum 8.1 mm"}}"""


def _tool_lines(tools: dict[str, Tool]) -> str:
    """Compact tool card. §12.3 caps the system prompt at ~300 tokens, so each
    tool gets one signature line plus its (<=200 char) description."""
    out: list[str] = []
    for tool in tools.values():
        props: dict[str, Any] = tool.schema.get("properties", {})
        params = ", ".join(f"{k}: {v.get('description', '')}" for k, v in props.items())
        gate = " [needs human approval]" if tool.requires_approval else ""
        out.append(f"- {tool.name}({params}){gate}\n  {tool.description}")
    return "\n".join(out)


def _extract_json(raw: str) -> dict[str, Any] | None:
    """Ollama's format=json usually gives clean JSON, but a 4B still sometimes
    wraps it in a fence or prepends a sentence. Try strict, then the first
    balanced object. §12.6 — the repair path is written before it is needed.

    `strict=False` is load-bearing, not a nicety. Strict JSON forbids literal
    control characters inside strings, and a model writing multi-line Python
    for py_sandbox emits exactly that: a real newline inside `{"code": "..."}`.
    Measured — it was the single failure in an otherwise 7/8 run, and it became
    *more* likely once the training data was fixed to use real newlines. The
    relaxed parse recovers the same object; refusing it burns a repair call and
    then the whole run for a purely cosmetic violation.
    """
    for candidate in (raw.strip(), *re.findall(r"\{.*\}", raw, re.S)):
        for strict in (True, False):
            try:
                parsed = json.loads(candidate, strict=strict)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…[truncated {len(text) - limit} chars]"


class Agent:
    def __init__(
        self,
        backend: LLMBackend,
        router: Router,
        tools: dict[str, Tool],
        workspace: Path,
        audit: AuditLog,
        settings: AgentSettings | None = None,
    ) -> None:
        self.backend = backend
        self.router = router
        self.tools = tools
        self.workspace = workspace
        self.audit = audit
        # Default rather than required: tests/test_agent.py and finetune/gate.py
        # construct an Agent directly and must keep working unchanged.
        self.settings = settings or AgentSettings()

    async def run(
        self,
        task: str,
        attachments: list[str] | None = None,
        approve: Approver | None = None,
        resident: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Stream the run. Yields one event per step so the UI renders a live
        plan trace (§8.4) instead of a spinner."""
        approver: Approver = approve or _deny_by_default
        ctx = RunContext(
            workspace=self.workspace, audit=self.audit, session_id=self.audit.session_id
        )

        decision = self.router.route(task, attachments, resident)
        spec = self.router.registry.models[decision.model_id]
        yield AgentEvent(type="route", data=decision.model_dump())

        messages = [
            Message(role="system", content=SYSTEM.replace("{tools}", _tool_lines(self.tools))),
            Message(role="user", content=task),
        ]
        spent = 0
        step_n = 0

        while step_n < self.settings.max_steps:
            step_n += 1
            if spent >= self.settings.max_tokens:
                yield self._stop(
                    f"token cap {self.settings.max_tokens} reached after {step_n - 1} steps", spent
                )
                return

            try:
                obj, used, repaired = await self._decide(spec.ref, messages, spec.max_ctx)
            except BackendError as exc:
                yield AgentEvent(type="error", data={"error": str(exc), "step": step_n})
                return
            except _Unrepairable as exc:
                # §8.4: one repair attempt, then fail loudly. Not silently
                # retried — a loop that keeps asking a 4B the same question
                # burns the token budget and lands nowhere.
                spent += exc.tokens
                yield AgentEvent(
                    type="error",
                    data={"error": str(exc), "step": step_n, "tokens_used": spent},
                )
                return
            spent += used

            thought = str(obj.get("thought", "")).strip()

            if "answer" in obj and not obj.get("tool"):
                yield AgentEvent(
                    type="final",
                    data={
                        "answer": str(obj["answer"]),
                        "steps": step_n - 1,
                        "tokens_used": spent,
                        "model": spec.id,
                    },
                )
                return

            name = str(obj.get("tool", ""))
            raw_args = obj.get("args")
            args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
            step = AgentStep(
                n=step_n, thought=thought, tool=name, args=args,
                tokens_used=spent, repaired=repaired,
            )
            tool = self.tools.get(name)
            if tool is None:
                # Recoverable in-loop: tell it what exists and spend a step.
                step.observation = (
                    f"no tool named {name!r}. Available: {', '.join(self.tools)}"
                )
                yield AgentEvent(type="step", data=step.model_dump())
                _record(messages, obj, step.observation)
                continue

            if tool.requires_approval:
                yield AgentEvent(
                    type="approval_request",
                    data={**step.model_dump(), "description": tool.description},
                )
                granted = await approver(step)
                self.audit.append(
                    "approval",
                    {"tool": name, "granted": granted, "step": step_n, "args": args},
                )
                if not granted:
                    step.observation = (
                        "The human reviewer declined this action. Do not retry it. "
                        "Either finish with what you have, or take a different approach."
                    )
                    yield AgentEvent(type="denied", data=step.model_dump())
                    _record(messages, obj, step.observation)
                    continue

            result = await asyncio.to_thread(_invoke, tool, args, ctx)
            step.observation = _truncate(_observe(result), self.settings.observation_chars)
            step.artifacts = result.artifacts
            yield AgentEvent(type="step", data=step.model_dump())
            _record(messages, obj, step.observation)

        yield self._stop(f"step cap {self.settings.max_steps} reached", spent)

    async def _decide(
        self, ref: str, messages: list[Message], max_ctx: int
    ) -> tuple[dict[str, Any], int, bool]:
        """One model call, plus at most one repair (§8.4).

        Returns (parsed object, tokens spent, whether a repair was needed).
        """
        completion = await asyncio.to_thread(
            self.backend.chat, ref, messages,
            max_ctx=max_ctx, temperature=self.settings.temperature, json_mode=True,
        )
        used = completion.total_tokens
        obj = _extract_json(completion.text)
        problem = self._defect(obj)
        if problem is None:
            assert obj is not None
            return obj, used, False

        repair = messages + [
            Message(role="assistant", content=completion.text[:1000]),
            Message(
                role="user",
                content=(
                    f"That reply was rejected: {problem}\n"
                    'Reply with ONE JSON object only: {"thought": "...", "tool": "...", '
                    '"args": {...}} or {"thought": "...", "answer": "..."}'
                ),
            ),
        ]
        second = await asyncio.to_thread(
            self.backend.chat, ref, repair,
            max_ctx=max_ctx, temperature=self.settings.temperature, json_mode=True,
        )
        used += second.total_tokens
        obj = _extract_json(second.text)
        still = self._defect(obj)
        if still is not None:
            raise _Unrepairable(
                f"model produced invalid output twice: {still}", used
            )
        assert obj is not None
        return obj, used, True

    def _defect(self, obj: dict[str, Any] | None) -> str | None:
        """Why this reply is unusable, phrased for the model to act on."""
        if obj is None:
            return "not valid JSON"
        if "answer" in obj and not obj.get("tool"):
            return None
        name = obj.get("tool")
        if not isinstance(name, str) or not name:
            return 'no "tool" and no "answer" field'
        tool = self.tools.get(name)
        if tool is None:
            return f"unknown tool {name!r}; available: {', '.join(self.tools)}"
        args = obj.get("args")
        if not isinstance(args, dict):
            return f'"args" must be a JSON object of parameters for {name}'
        # §2.3: arguments are schema-validated before execution, never after.
        return validate_args(tool.schema, args)

    def _stop(self, why: str, spent: int) -> AgentEvent:
        self.audit.append("error", {"event": "agent_halted", "why": why, "tokens": spent})
        return AgentEvent(type="final", data={"answer": f"Stopped: {why}. Partial work above.",
                                              "halted": True, "tokens_used": spent})


class _Unrepairable(RuntimeError):
    def __init__(self, message: str, tokens: int) -> None:
        super().__init__(message)
        self.tokens = tokens


def _invoke(tool: Tool, args: dict[str, Any], ctx: RunContext) -> ToolResult:
    """Runs in a worker thread. Tool bugs become failed observations rather than
    killing the run — §11 forbids a bare except, so this catches Exception and
    audits it with the tool name attached."""
    try:
        return tool.run(args, ctx)
    except Exception as exc:  # noqa: BLE001 - audited and returned, never swallowed
        ctx.audit.append(
            "error", {"tool": tool.name, "error": f"{type(exc).__name__}: {exc}"}
        )
        return ToolResult(ok=False, output="", error=f"{type(exc).__name__}: {exc}")


def _observe(result: ToolResult) -> str:
    """What the model actually sees after a tool runs.

    On failure the tool's *output* matters far more than its error label: a
    traceback is repairable, "exit 1" is not. Returning only the label is what
    made the coder resubmit a byte-identical numpy import until the step cap,
    burning 18k of the 20k token budget (§8.4) and learning nothing.
    """
    if result.ok:
        return result.output
    label = (result.error or "the tool failed").strip()
    body = result.output.strip()
    return f"{label}\n{body}" if body else label


def _record(messages: list[Message], call: dict[str, Any], observation: str) -> None:
    """Append the model's own call and the resulting observation.

    §12.9 — small models lose the middle, so the observation goes in as a fresh
    user turn at the end of the context rather than being merged upward.
    """
    messages.append(Message(role="assistant", content=json.dumps(call)))
    messages.append(Message(role="user", content=f"Observation:\n{observation}"))


async def _deny_by_default(step: AgentStep) -> bool:
    """No approver wired means no approval. §2.4 is a gate, not a default-yes."""
    return False
