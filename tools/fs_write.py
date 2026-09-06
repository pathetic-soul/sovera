"""Write a text file inside the workspace jail (AGENTS.md §8.3, §2.3, §2.4).

CPU only, 0 GB VRAM. Companion to fs_read.py: same jail, but a write is exactly
what §2.4's human gate exists for, so unlike fs_read this requires approval.

doc_write.py already covers the flagship .docx deliverable; this tool is for
everything else an agent step might need to save — a .txt note, a .py script,
a .csv, a .json result — without inventing a new gated tool per file type.
"""

from __future__ import annotations

from typing import Any

from tools.base import JailBreak, RunContext, Tool, ToolResult, resolve_in_jail

MAX_CHARS = 50_000  # generous relative to fs_read's 6000 cap: this is input the
# model already composed, not an observation being re-read into its own context.


class FsWrite(Tool):
    name = "fs_write"
    description = (
        "Write a UTF-8 text file into the workspace (creates parent folders). "
        "Overwrites if it exists. Needs human approval."
    )
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "workspace-relative path, e.g. out/notes.txt",
            },
            "content": {
                "type": "string",
                "description": "the full file content to write",
            },
        },
        "required": ["path", "content"],
    }
    requires_approval = True

    def run(self, args: dict[str, Any], ctx: RunContext) -> ToolResult:
        rel = str(args["path"])
        content = str(args["content"])[:MAX_CHARS]

        try:
            target = resolve_in_jail(ctx.workspace, rel)
        except JailBreak as exc:
            ctx.audit.append("file_write", {"path": rel, "ok": False, "error": str(exc)})
            return ToolResult(ok=False, output="", error=str(exc))

        target.parent.mkdir(parents=True, exist_ok=True)
        intent = ctx.audit.append(
            "file_write", {"phase": "intent", "path": rel, "content_chars": len(content)}
        )
        target.write_text(content, encoding="utf-8")
        size = target.stat().st_size
        ctx.audit.append(
            "file_write", {"phase": "result", "path": rel, "ok": True, "bytes": size},
            ref=intent.seq,
        )
        return ToolResult(ok=True, output=f"Wrote {rel} ({size} bytes).", artifacts=[rel])
