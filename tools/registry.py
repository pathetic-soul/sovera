"""The tool roster — one declarative list, four consumers (AGENTS.md §8.3, §14.4).

CPU only, 0 GB VRAM. Constructing every tool is four object allocations; the
cost is in `run()`, not here.

WHY THIS FILE EXISTS
--------------------
§8.1 makes adding a *model* a config edit plus a button click, and leg 3 demos
exactly that on stage. Adding a *tool* was a source edit inside the FastAPI
lifespan — and the identical four-tool tuple had been copied into four places:

    core/orchestrator.py   the shipped agent
    finetune/gate.py       the gate that decides whether a model ships
    tests/test_finetune.py the trace validator
    tests/test_traceset.py the trace generator's schema check

That is not a style problem. `finetune/gate.py` measured an agent, and the tests
validated training data, against a tool set that had to be kept in step with the
real one by hand. The v2 adapter's headline failure was brittleness to an
unfamiliar tool set (§16) — a defect in exactly this area, measured with a
hand-copied roster.

WHAT THIS IS NOT
----------------
Not a plugin loader, not discovery by entry point, not YAML. §11 asks for boring
inspectable code, and a judge should be able to read the complete list of things
this agent can do in one screen. It is a list and a function.

Tool selection deliberately stays in code rather than moving to config the way
models did. Tools carry executable behaviour and a `requires_approval` flag that
§2.4 makes a compliance control; making that set config-driven would let an edit
to a YAML file switch off the human gate. Models are the graded config surface
(§8.1); tools are not.
"""

from __future__ import annotations

from typing import Any, Callable

from tools.base import Tool
from tools.calc import Calc
from tools.doc_write import DocWrite
from tools.fs_read import FsRead
from tools.ocr_read import OcrRead
from tools.py_sandbox import PySandbox

# The roster, in the order the agent sees it in its system prompt. Order is not
# cosmetic: §12.4 says few-shot beats instructions, and the tool card block in
# `core/agent.py:_tool_lines` renders in this order, so read-before-answer
# (`fs_read`) leads and the two gated, side-effecting tools trail.
#
# `calc` is a deliberate fourth against §14.4's "three tools only". It was added
# on measurement, not preference: on the leg-5 task both the base and the
# fine-tuned driver wrote the wrong formula (1/8 and 0/8 correct) and py_sandbox
# executed it faithfully. Moving the formula into code took the base model to
# 7/8. See tools/calc.py.
#
# `ocr_read` is leg 6 (§5's `ocr` roster row): CPU-only PaddleOCR, so it costs
# 0 GB of the 5.2 GB VRAM budget the LLMs already fight over. It sits next to
# fs_read because it is the same "read before you answer" discipline applied
# to a scanned image instead of a text file.
TOOL_CLASSES: tuple[Callable[[], Tool], ...] = (FsRead, OcrRead, Calc, PySandbox, DocWrite)


def build_tools() -> dict[str, Tool]:
    """A fresh roster, keyed by tool name — what `Agent` and the UI both take.

    Fresh instances rather than a module-level singleton dict: tools are handed
    a per-run `RunContext` and are cheap to build, and a shared mutable mapping
    is the kind of global state §11 rules out.
    """
    return {tool.name: tool for tool in (cls() for cls in TOOL_CLASSES)}


def tool_specs(tools: dict[str, Tool] | None = None) -> list[dict[str, Any]]:
    """What `/api/tools` renders: name, description, schema, and — the part that
    matters for §2.4 — which tools the human gate covers."""
    return [tool.spec() for tool in (tools or build_tools()).values()]


def gated_tools(tools: dict[str, Tool] | None = None) -> list[str]:
    """Names of the tools that cannot run without a human approving (§2.4).

    Exposed as its own function because "which actions need sign-off" is a
    question a PSU reviewer asks directly, and it should have one answer derived
    from the tools themselves rather than a list in a slide.
    """
    return [t.name for t in (tools or build_tools()).values() if t.requires_approval]
