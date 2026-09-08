"""Tool contract, filesystem jail, and argument validation (AGENTS.md §8.3, §2.3).

CPU only, 0 GB VRAM.

Three things live here because all of them must have exactly one implementation:

1. `resolve_in_jail` — §2.3 says any write outside `workspace/` is rejected. If
   each tool implemented its own check, one tool forgetting `..` handling would
   be a hole in the jail. Every tool that touches a path calls this.
2. `validate_args` — §2.3 forbids executing unvalidated args. Deliberately a
   small flat-schema checker rather than the `jsonschema` package: §8.3 caps
   schemas at 5 flat params with no nesting (4B models fail on nested objects),
   so the full JSON Schema spec is ~40 lines of value we would never use.
3. `Tool` — the ABC itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.audit import AuditLog


class ToolResult(BaseModel):
    ok: bool
    output: str
    # Paths relative to workspace/, surfaced in the UI as download links.
    artifacts: list[str] = Field(default_factory=list)
    error: str | None = None


@dataclass
class RunContext:
    """Per-run state handed to every tool. Not a Pydantic model: it carries the
    live AuditLog, and §8 asks for Pydantic on *contracts*, not on plumbing."""

    workspace: Path
    audit: AuditLog
    session_id: str
    # Set by py_sandbox.py for the duration of a container run, cleared after.
    # Lets the kill switch (core/api/agent.py) `docker kill` the right
    # container by name instead of waiting out its own 60s timeout.
    active_container: str | None = None


class JailBreak(RuntimeError):
    """A path escaped workspace/. §2.3 makes this fatal, never a warning."""


def resolve_in_jail(workspace: Path, candidate: str) -> Path:
    """Resolve `candidate` inside `workspace`, or raise.

    Rejects absolute paths, drive letters and `..` traversal alike by resolving
    first and comparing afterwards — string-prefix checks on the raw argument
    are what jail escapes are made of. Symlinks resolve too, so a link planted
    inside workspace/ pointing at C:\\Windows does not get you out either.

    `candidate.strip()` — a stray leading space (easy to pick up pasting a
    path into the UI) used to resolve to a silently different, nonexistent
    directory (`" inbox/x.png"` -> `workspace/ inbox/x.png`) with no jail
    rejection and no diagnostic beyond a plain "no such file". Whitespace is
    never a real part of a path here, so trimming it costs nothing.
    """
    root = workspace.resolve()
    raw = Path(candidate.strip())
    if raw.is_absolute() or raw.drive:
        raise JailBreak(f"absolute paths are refused: {candidate!r}")
    target = (root / raw).resolve()
    if target != root and root not in target.parents:
        raise JailBreak(f"{candidate!r} resolves outside the workspace jail")
    return target


def validate_args(schema: dict[str, Any], args: dict[str, Any]) -> str | None:
    """Return None if valid, else a message the model can act on (§8.4 repair).

    The message is written *for a 4B to read*: it names the parameter and the
    expected type, because "ValidationError at $.properties" repairs nothing.
    """
    props: dict[str, Any] = schema.get("properties", {})
    required: list[str] = schema.get("required", [])

    for key in required:
        if key not in args:
            hint = props.get(key, {}).get("description", "")
            return f"missing required parameter {key!r}. {hint}".strip()

    unknown = set(args) - set(props)
    if unknown:
        return (
            f"unknown parameter(s) {sorted(unknown)}. "
            f"Allowed: {sorted(props)}"
        )

    kinds: dict[str, type | tuple[type, ...]] = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
    }
    for key, value in args.items():
        want = props[key].get("type", "string")
        expected = kinds.get(want, str)
        # bool is a subclass of int in Python; an LLM sending true for an
        # integer param is a real mistake, not a widening conversion.
        if want in ("integer", "number") and isinstance(value, bool):
            return f"parameter {key!r} must be {want}, got boolean"
        if not isinstance(value, expected):
            return f"parameter {key!r} must be {want}, got {type(value).__name__}"
        enum = props[key].get("enum")
        if enum and value not in enum:
            return f"parameter {key!r} must be one of {enum}, got {value!r}"
    return None


def nearby_files(workspace: Path, limit: int = 20) -> str:
    """Name what actually exists in the jail (§12.6): a small model — or a
    user typing from memory — that got a filename wrong can only fix it if
    the error names the real ones. Shared by fs_read and ocr_read rather than
    kept as fs_read's own private helper, since ocr_read hit the identical
    "guessed wrong, told nothing" gap it was already built to close."""
    root = workspace.resolve()
    names = sorted(
        str(p.relative_to(root)).replace("\\", "/")
        for p in root.rglob("*")
        if p.is_file() and ".audit" not in p.parts
    )[:limit]
    return f"Files available: {', '.join(names)}" if names else "The workspace is empty."


class Tool(ABC):
    name: str
    description: str  # <= 200 chars (§8.3) — small models cannot read essays
    schema: dict[str, Any]  # flat, <= 5 params, example in every description
    requires_approval: bool  # §2.4 human gate

    @abstractmethod
    def run(self, args: dict[str, Any], ctx: RunContext) -> ToolResult: ...

    def spec(self) -> dict[str, Any]:
        """What the agent puts in the system prompt and the UI renders."""
        return {
            "name": self.name,
            "description": self.description,
            "schema": self.schema,
            "requires_approval": self.requires_approval,
        }
