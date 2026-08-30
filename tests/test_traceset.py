"""The shipping training corpus must not regress into the one that failed.

finetune/traceset.py replaced finetune/dataset.py after three adapters trained
on the old corpus all scored worse than their own base model. Profiling named
four structural causes, and each one is asserted here as a number rather than
described in a comment — a corpus can drift back into any of them silently, and
the only symptom would be another wasted training run.

No model and no GPU: static analysis of generated JSON.
"""

from __future__ import annotations

import json
from typing import Any

from finetune.traceset import ALL_TOOLS, FAMILIES, build, profile, split
from tools.base import validate_args
from tools.registry import build_tools

REAL_TOOLS = build_tools()

TRACES, TALLY = build(1200)
PROFILE = profile(TRACES)


def assistant_turns(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [json.loads(m["content"]) for m in trace["messages"]
            if m["role"] == "assistant"]


# --- structure --------------------------------------------------------------

def test_every_assistant_turn_is_one_flat_json_object() -> None:
    for trace in TRACES:
        for turn in assistant_turns(trace):
            assert isinstance(turn, dict)
            assert ("tool" in turn) != ("answer" in turn), turn
            assert turn.get("thought"), "every turn needs a thought"
            for value in turn.get("args", {}).values():
                assert not isinstance(value, (dict, list)), \
                    "§8.3 caps schemas at flat params; 4B models fail on nesting"


def test_traces_open_correctly_and_end_on_an_answer() -> None:
    for trace in TRACES:
        roles = [m["role"] for m in trace["messages"]]
        assert roles[0] == "system"
        assert roles[1] == "user"
        assert roles[-1] == "assistant"
        assert "answer" in assistant_turns(trace)[-1], "a trace must finish"
        # Strict alternation after the system turn, or the label mask in
        # qlora.py lands on the wrong tokens.
        for i, role in enumerate(roles[2:], 2):
            assert role == ("assistant" if i % 2 == 0 else "user"), roles


def test_no_trace_calls_a_tool_it_was_not_offered() -> None:
    """The corpus contains tools this repo has not built, on purpose — that is
    how tool-schema generalisation is taught. It is only safe because a tool
    never appears in a trace unless it is also printed in that trace's own
    system prompt. If that invariant breaks, the corpus starts teaching
    hallucinated tool names."""
    for trace in TRACES:
        offered = {n for n in ALL_TOOLS if f"- {n}(" in trace["messages"][0]["content"]}
        for turn in assistant_turns(trace):
            if "tool" in turn:
                assert turn["tool"] in offered, \
                    f"{turn['tool']} called but not offered in this trace"


def test_calls_to_real_tools_pass_the_loops_own_validator() -> None:
    """§2.3 — the loop schema-validates before executing. Training data that
    fails the same check teaches the model to produce rejected calls."""
    checked = 0
    for trace in TRACES:
        for turn in assistant_turns(trace):
            name = turn.get("tool")
            if name in REAL_TOOLS:
                problem = validate_args(REAL_TOOLS[name].schema, turn["args"])
                assert problem is None, f"{name}: {problem}"
                checked += 1
    assert checked > 100, "suspiciously few real-tool calls in the sample"


# --- the four defects that sank the previous corpus -------------------------

def test_first_action_is_not_dominated_by_fs_read() -> None:
    """DEFECT 1: the old corpus opened with fs_read in 600 of 612 traces (98%),
    so the model learned that step one is always a read rather than when to
    read. Anything above ~85% is that failure returning."""
    firsts = PROFILE["first_action"]
    share = firsts.get("fs_read", 0) / len(TRACES)
    assert share < 0.85, f"fs_read opens {share:.1%} of traces (old corpus: 98%)"
    assert firsts.get("<answer>", 0) > 0, "no trace answers without a tool"
    assert firsts.get("calc", 0) > 0, "no trace opens by calculating"


def test_tool_sets_vary_across_traces() -> None:
    """DEFECT 2: the old corpus built its system prompt once from four tools.
    That is measurably what made v2 refuse to use a newly offered calc tool."""
    assert PROFILE["distinct_tool_sets"] > 50, \
        f"only {PROFILE['distinct_tool_sets']} distinct tool sets (old corpus: 1)"


def test_hard_paths_are_represented() -> None:
    """DEFECT 4: the old corpus had no failed call, no ambiguity, no refusal
    and no multi-document synthesis — the loop's difficult branches had zero
    supervision."""
    for family in ("repair", "clarify", "unknown", "multi_doc"):
        assert TALLY.get(family, 0) > 0, f"family {family} generated nothing"


def test_every_family_generates_and_survives_the_split() -> None:
    """A family that silently drops to zero removes a whole behaviour from the
    corpus, and stratified eval stops representing it."""
    train, ev = split(TRACES)
    for name, _, _ in FAMILIES:
        assert TALLY.get(name, 0) > 0, f"family {name} generated nothing"
    for part, label in ((train, "train"), (ev, "eval")):
        families = {t["_family"] for t in part}
        missing = {n for n, _, _ in FAMILIES} - families
        assert not missing, f"{label} split is missing {missing}"


def test_prompts_are_not_heavily_repeated() -> None:
    """Repetition is what drives training loss to 0.0016 without teaching
    anything. The old corpus recycled prompts hard; this asserts the floor."""
    ratio = PROFILE["unique_user_prompts"] / len(TRACES)
    assert ratio > 0.45, f"only {ratio:.0%} of user prompts are unique"


def test_thoughts_are_task_specific_not_copied_boilerplate() -> None:
    thoughts = [t["thought"] for tr in TRACES for t in assistant_turns(tr)]
    assert len(set(thoughts)) > len(thoughts) * 0.3


# --- ground truth -----------------------------------------------------------

def test_derivations_go_through_a_calculation_tool() -> None:
    """A sandbox guarantees the arithmetic, not the formula — measured, when a
    fine-tuned model computed 5.70/0.2982 in a running sandbox and was wrong in
    7 of 8 trials. Derivation traces must reach for calc."""
    derive = [t for t in TRACES if t["_family"] in ("inspection_derive", "calc_only")]
    assert derive, "no derivation traces generated"
    for trace in derive:
        tools_used = [t["tool"] for t in assistant_turns(trace) if "tool" in t]
        assert "calc" in tools_used, "a derivation that never calculates"


def test_unknown_traces_answer_unknown_and_invent_nothing() -> None:
    for trace in [t for t in TRACES if t["_family"] == "unknown"]:
        final = assistant_turns(trace)[-1]["answer"]
        assert "UNKNOWN" in final, "§12.7 requires an explicit UNKNOWN"


def test_clarify_traces_ask_rather_than_guess() -> None:
    for trace in [t for t in TRACES if t["_family"] == "clarify"]:
        turns = assistant_turns(trace)
        assert len(turns) == 1, "a clarification should not also do the work"
        assert "?" in turns[0]["answer"], "a clarification must ask something"
