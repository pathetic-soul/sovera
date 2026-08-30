"""Run generated Python in a network-less container (AGENTS.md §9.3, §14.5).

CPU only, 0 GB VRAM — the GPU wrote the code, the container merely runs it.
Wall clock is container start (~0.5-1.5 s on Docker Desktop) plus the script,
hard-capped at 60 s.

`--network none` is doing double duty and that is the point of leg 5: it is a
security control *and* a sovereignty artifact. It is the third of the three
containment layers in the order a packet meets them — Windows Firewall, then
core/net_guard.py, then this. The argv is audited verbatim and rendered in the
UI so the flag is evidence a judge can read, not a claim in a slide.

INVARIANT: there is no host-execution fallback. If the daemon is down or the
image is missing, this tool fails and says so. Running model-generated code on
the host because Docker was inconvenient would void §9.3 entirely, and it is
exactly the shortcut that looks harmless at 2am the night before a demo.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from tools.base import RunContext, Tool, ToolResult, resolve_in_jail

IMAGE = "sandbox-py:local"
TIMEOUT_S = 60
ARTIFACT_SUBDIR = "out/sandbox"
MAX_CAPTURE = 4000  # chars of stdout/stderr kept; §8.4 truncates observations

# §9.3 verbatim. Kept as data, not a formatted string, so the UI can show the
# exact flags the container ran with.
HARDENING = (
    "--rm",
    "--network", "none",
    "--read-only",
    "--tmpfs", "/work:rw,size=256m,exec",
    "--memory", "2g",
    "--cpus", "2",
    "--pids-limit", "128",
    "--security-opt", "no-new-privileges",
    "--cap-drop", "ALL",
)


class PySandbox(Tool):
    name = "py_sandbox"
    description = (
        "Run a Python 3 script offline (no network, 60s, standard library only "
        "- no numpy/pandas). Returns stdout, stderr, exit code. /out is kept."
    )
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "complete Python source, e.g. print(sum(range(10)))",
            },
            "purpose": {
                "type": "string",
                "description": "one line on what it computes, e.g. remaining life from UT readings",
            },
        },
        "required": ["code"],
    }
    requires_approval = True

    def run(self, args: dict[str, Any], ctx: RunContext) -> ToolResult:
        code = str(args["code"])
        purpose = str(args.get("purpose", ""))

        problem = _preflight()
        if problem:
            ctx.audit.append(
                "tool_call",
                {"tool": self.name, "ok": False, "stage": "preflight", "error": problem},
            )
            return ToolResult(ok=False, output="", error=problem)

        artifact_dir = resolve_in_jail(ctx.workspace, ARTIFACT_SUBDIR)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        before = {p.name for p in artifact_dir.iterdir()}

        with tempfile.TemporaryDirectory(prefix="sovereign-src-") as src:
            # The script is mounted read-only at /src and run with /work (tmpfs)
            # as the cwd, so the container can write scratch but cannot rewrite
            # the code it was handed.
            Path(src, "main.py").write_text(code, encoding="utf-8")
            argv = [
                "docker", "run", *HARDENING,
                "--mount", f"type=bind,source={src},target=/src,readonly",
                "--mount", f"type=bind,source={artifact_dir},target=/out",
                "-w", "/work",
                IMAGE,
                "python", "/src/main.py",
            ]
            intent = ctx.audit.append(
                "tool_call",
                {
                    "phase": "intent",
                    "tool": self.name,
                    "purpose": purpose,
                    "code_chars": len(code),
                    "argv": argv,
                    "network": "none",
                },
            )
            try:
                proc = subprocess.run(
                    argv, capture_output=True, text=True, timeout=TIMEOUT_S, check=False
                )
            except subprocess.TimeoutExpired:
                ctx.audit.append(
                    "tool_call",
                    {"phase": "result", "tool": self.name, "ok": False, "timeout_s": TIMEOUT_S},
                    ref=intent.seq,
                )
                return ToolResult(
                    ok=False,
                    output="",
                    error=f"killed after {TIMEOUT_S}s. Make the script terminate.",
                )

        produced = sorted(
            f"{ARTIFACT_SUBDIR}/{p.name}"
            for p in artifact_dir.iterdir()
            if p.name not in before
        )
        stdout = proc.stdout[:MAX_CAPTURE]
        stderr = proc.stderr[:MAX_CAPTURE]
        ok = proc.returncode == 0

        ctx.audit.append(
            "tool_call",
            {
                "phase": "result",
                "tool": self.name,
                "ok": ok,
                "exit_code": proc.returncode,
                "artifacts": produced,
            },
            ref=intent.seq,
        )
        return ToolResult(
            ok=ok,
            output=_transcript(proc.returncode, stdout, stderr),
            artifacts=produced,
            # §12.6: hand the traceback back so the coder model can repair it.
            error=None if ok else f"exit {proc.returncode}",
        )


def _transcript(code: int, stdout: str, stderr: str) -> str:
    parts = [f"exit_code: {code}"]
    parts.append(f"stdout:\n{stdout}" if stdout.strip() else "stdout: (empty)")
    if stderr.strip():
        parts.append(f"stderr:\n{stderr}")
    return "\n".join(parts)


def _preflight() -> str | None:
    """Say precisely which of the three failure modes it is, with the fix.

    A generic "sandbox unavailable" on stage is unrecoverable; naming the
    missing piece is the difference between a 10-second save and a dead demo.
    """
    if shutil.which("docker") is None:
        return "docker is not on PATH. The sandbox will not run code on the host (§9.3)."
    try:
        info = subprocess.run(
            ["docker", "image", "inspect", IMAGE],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return f"docker did not respond ({type(exc).__name__}). Is Docker Desktop running?"
    if info.returncode == 0:
        return None
    detail = (info.stderr or "").strip().splitlines()
    hint = detail[0][:200] if detail else ""
    if "daemon" in hint.lower() or "pipe" in hint.lower():
        return f"Docker daemon is not running — start Docker Desktop. ({hint})"
    return (
        f"image {IMAGE} is missing. Build it once, offline-capable: "
        f"docker build -t {IMAGE} sandbox/  ({hint})"
    )
