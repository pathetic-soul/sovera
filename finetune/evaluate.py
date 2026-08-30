"""Measure agent-loop behaviour on the held-out split (AGENTS.md §13).

This produces a NEW metric. It is not the §13 router-accuracy gate and must
never be quoted as it — the router is core/routing/, a deterministic
lexical scorer with no model in it (§2.3). What is measured here is how well a
given Ollama model drives the loop in core/agent.py:

  json_valid   first reply parses as one JSON object, with no repair
  tool_match   picked the tool the trace expected (or finished when it should)
  args_valid   arguments pass the same validate_args() the loop enforces (§2.3)
  unknown      said UNKNOWN where the document genuinely does not answer (§12.7)

json_valid is the one with a direct cost attached: every failure is a repair
call in core/agent.py, which on a 6 GB budget is a second full forward pass out
of a 20k token ceiling (§8.4).

Runs against whatever ref you point it at, so the same harness gives the
before and after number:

    python -m finetune.evaluate qwen3:4b-instruct
    python -m finetune.evaluate sovereign-driver:q4_K_M --limit 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from backends.base import BackendError, Message
from backends.ollama_backend import OllamaBackend
from core.agent import TEMPERATURE, _extract_json
from core.audit import AuditLog
from tools.base import Tool, validate_args
from tools.doc_write import DocWrite
from tools.fs_read import FsRead
from tools.py_sandbox import PySandbox

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "finetune" / "data" / "eval.jsonl"
MAX_CTX = 8192  # the §5 floor the loop actually serves with

TOOLS: dict[str, Tool] = {t.name: t for t in (FsRead(), PySandbox(), DocWrite())}


class Score:
    def __init__(self) -> None:
        self.n = 0
        self.json_valid = 0
        self.tool_match = 0
        self.args_valid = 0
        self.unknown_total = 0
        self.unknown_hit = 0

    def row(self, name: str) -> str:
        def pct(hit: int, total: int) -> str:
            return f"{100.0 * hit / total:6.1f}%" if total else "     --"

        return (
            f"{name:<26} n={self.n:<4} "
            f"json_valid {pct(self.json_valid, self.n)}  "
            f"tool_match {pct(self.tool_match, self.n)}  "
            f"args_valid {pct(self.args_valid, self.n)}  "
            f"unknown {pct(self.unknown_hit, self.unknown_total)}"
            f" ({self.unknown_total})"
        )


def prediction_points(messages: list[dict[str, str]]) -> list[tuple[list[Message], str]]:
    """Every assistant turn is one prediction, given everything before it."""
    points: list[tuple[list[Message], str]] = []
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        prefix = [Message(role=m["role"], content=m["content"]) for m in messages[:i]]
        points.append((prefix, msg["content"]))
    return points


def judge(predicted: dict[str, Any] | None, expected: dict[str, Any], score: Score) -> None:
    score.n += 1
    want_tool = expected.get("tool")
    wants_unknown = "UNKNOWN" in str(expected.get("answer", ""))
    if wants_unknown:
        score.unknown_total += 1

    if predicted is None:
        return
    score.json_valid += 1

    got_tool = predicted.get("tool")
    if (got_tool or None) == (want_tool or None):
        score.tool_match += 1
        if want_tool:
            tool = TOOLS.get(str(want_tool))
            args = predicted.get("args")
            if tool and isinstance(args, dict) and validate_args(tool.schema, args) is None:
                score.args_valid += 1
        else:
            # Finishing correctly needs no arguments to validate.
            score.args_valid += 1

    if wants_unknown and "UNKNOWN" in str(predicted.get("answer", "")):
        score.unknown_hit += 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ref", help="ollama tag, e.g. qwen3:4b-instruct")
    parser.add_argument("--limit", type=int, default=0, help="first N traces only")
    parser.add_argument("--label", default="", help="name for the results row")
    args = parser.parse_args()

    rows = [json.loads(line) for line in EVAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]

    # A throwaway log: §2.2 wants every model call audited, and an eval run is
    # still model calls. Kept out of the demo chain so it cannot pollute it.
    audit = AuditLog(ROOT / "workspace" / ".audit" / "eval.jsonl", "eval")
    backend = OllamaBackend(audit)
    ok, note = backend.available()
    if not ok:
        print(f"ollama unavailable: {note}")
        return 1

    score = Score()
    for i, row in enumerate(rows, 1):
        for prefix, expected_raw in prediction_points(row["messages"]):
            expected = json.loads(expected_raw)
            try:
                completion = backend.chat(
                    args.ref, prefix, max_ctx=MAX_CTX,
                    temperature=TEMPERATURE, json_mode=True,
                )
            except BackendError as exc:
                print(f"backend error: {exc}")
                return 1
            judge(_extract_json(completion.text), expected, score)
        if i % 10 == 0:
            print(f"  ...{i}/{len(rows)} traces")

    print()
    print(score.row(args.label or args.ref))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
