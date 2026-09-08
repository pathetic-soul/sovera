"""Read a text file from inside the workspace jail (AGENTS.md §8.3, §2.3).

CPU only, 0 GB VRAM. No approval gate: reading inside the jail cannot change
anything, and §2.4 gates *writes and execution*. Gating reads would add a click
to the demo that defends nothing.

The truncation cap here is a context-budget control, not a convenience. An
unbounded observation re-entering an 8k window (§4.2.5) is how a 6 GB agent
loop dies; §8.4 caps observations at 1500 tokens and this is the first line of
that defence.
"""

from __future__ import annotations

from typing import Any

from tools.base import JailBreak, RunContext, Tool, ToolResult, nearby_files, resolve_in_jail

MAX_CHARS = 6000  # ~1500 tokens at the ~4 chars/token used throughout (§8.4)


class FsRead(Tool):
    name = "fs_read"
    description = (
        "Read a UTF-8 text file from the workspace. Returns the text, truncated "
        "if long. Use it before answering questions about a document."
    )
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "workspace-relative path, e.g. inbox/UT-2024-114.md",
            }
        },
        "required": ["path"],
    }
    requires_approval = False

    def run(self, args: dict[str, Any], ctx: RunContext) -> ToolResult:
        rel = str(args["path"])
        try:
            target = resolve_in_jail(ctx.workspace, rel)
        except JailBreak as exc:
            ctx.audit.append(
                "file_read", {"path": rel, "ok": False, "error": str(exc)}
            )
            return ToolResult(ok=False, output="", error=str(exc))

        if not target.is_file():
            listing = nearby_files(ctx.workspace)
            msg = f"no such file: {rel}"
            ctx.audit.append("file_read", {"path": rel, "ok": False, "error": msg})
            return ToolResult(ok=False, output="", error=f"{msg}. {listing}")

        text = target.read_text(encoding="utf-8", errors="replace")
        truncated = len(text) > MAX_CHARS
        if truncated:
            text = text[:MAX_CHARS] + f"\n…[truncated, {len(text)} chars total]"

        ctx.audit.append(
            "file_read",
            {
                "path": rel,
                "ok": True,
                "bytes": target.stat().st_size,
                "truncated": truncated,
            },
        )
        return ToolResult(ok=True, output=text)
