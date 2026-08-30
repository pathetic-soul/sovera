"""The training data and the eval harness must both be trustworthy.

These are cheap checks on expensive mistakes. A trace that teaches a malformed
tool call trains the model to emit exactly what core/agent.py rejects, and a
scoring bug puts a wrong number on a slide. Neither failure is visible by
reading a training curve.

No model and no GPU: this is all static analysis of generated JSON.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from finetune.dataset import GENERATORS, build
from finetune.evaluate import Score, judge
from finetune.grounding_eval import GRID, QUESTIONS, close, verify_ground_truth
from tools.base import Tool, validate_args
from tools.registry import build_tools

TOOLS: dict[str, Tool] = build_tools()
TRACES = build(120)


def assistant_turns(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        json.loads(m["content"])
        for m in trace["messages"]
        if m["role"] == "assistant"
    ]


# --- the training data ------------------------------------------------------

def test_every_assistant_turn_is_one_json_object() -> None:
    for trace in TRACES:
        for msg in trace["messages"]:
            if msg["role"] == "assistant":
                parsed = json.loads(msg["content"])
                assert isinstance(parsed, dict)


def test_every_tool_call_would_pass_the_loops_own_validator() -> None:
    """§2.3 — the loop schema-validates before executing. Training data that
    fails the same check teaches the model to produce rejected calls."""
    checked = 0
    for trace in TRACES:
        for turn in assistant_turns(trace):
            name = turn.get("tool")
            if not name:
                continue
            assert name in TOOLS, f"unknown tool in training data: {name}"
            problem = validate_args(TOOLS[name].schema, turn["args"])
            assert problem is None, f"{name}: {problem}"
            checked += 1
    assert checked > 50, "suspiciously few tool calls in the sample"


def test_every_turn_is_a_tool_call_or_an_answer_never_both() -> None:
    for trace in TRACES:
        for turn in assistant_turns(trace):
            assert ("tool" in turn) != ("answer" in turn), turn
            assert turn.get("thought"), "every turn needs a thought"


def test_traces_alternate_and_end_on_an_answer() -> None:
    for trace in TRACES:
        roles = [m["role"] for m in trace["messages"]]
        assert roles[0] == "system"
        assert roles[-1] == "assistant"
        assert "answer" in assistant_turns(trace)[-1], "a trace must finish"


def test_thoughts_are_task_specific_not_copied_boilerplate() -> None:
    """Leg 4 showed the 8B copying the few-shot thought verbatim, which makes
    the live plan trace look canned. Distinct tasks must yield distinct thoughts."""
    thoughts = [t["thought"] for tr in TRACES for t in assistant_turns(tr)]
    assert len(set(thoughts)) > len(thoughts) * 0.3


def test_arithmetic_is_routed_through_calc_not_the_sandbox() -> None:
    """This assertion was inverted, on measurement.

    It used to require arithmetic to go to py_sandbox. That taught the wrong
    thing: py_sandbox executes whatever formula the model writes, and the
    fine-tuned model wrote `interval / rate` in 7 of 8 trials (19.11 years
    instead of 5.03). calc has the formula fixed in code, so derivations go
    there and py_sandbox keeps only genuine scripting work.
    """
    with_calc = sum(
        1 for tr in TRACES
        if any(t.get("tool") == "calc" for t in assistant_turns(tr))
    )
    assert with_calc / len(TRACES) > 0.25, "calc must dominate the arithmetic traces"


def test_sandbox_code_is_runnable_and_actually_does_something() -> None:
    """Compiling is not enough, and assuming it was cost a whole training run.

    The generator used to join lines with a literal backslash-n instead of a
    newline, so every "script" was a single line beginning with `#`. That
    compiles perfectly — it is a valid Python file containing one comment — and
    an earlier version of this test passed on it. The model duly learned to
    emit one-line comments, the sandbox returned exit 0 with empty stdout, and
    the model then invented the numbers it claimed to have computed.

    `ast.parse(...).body` is the check that catches it: a comment-only file has
    an EMPTY body. Also assert a print, because a script that computes and
    reports nothing is useless as a training example.
    """
    import ast

    checked = 0
    for trace in TRACES:
        for turn in assistant_turns(trace):
            if turn.get("tool") != "py_sandbox":
                continue
            code = turn["args"]["code"]
            assert "\\n" not in code, f"literal backslash-n in code: {code[:80]!r}"
            assert "\n" in code, f"single-line script, probably a comment: {code[:80]!r}"
            body = ast.parse(code).body
            assert body, f"code parses to an empty body (all comments): {code[:80]!r}"
            assert "print(" in code, f"code produces no output: {code[:80]!r}"
            checked += 1
    assert checked > 10


def test_generator_weights_are_all_positive() -> None:
    assert all(weight > 0 for _, weight in GENERATORS)


# --- the grounding harness --------------------------------------------------

def test_ground_truth_matches_the_corpus_document() -> None:
    """Guards the table in grounding_eval.py against the corpus drifting."""
    verify_ground_truth()


def test_derived_tolerance_accepts_rounding_but_rejects_arithmetic_errors() -> None:
    assert close(9.167, (9.4 + 8.9 + 9.2) / 3, "derived")   # honest rounding
    assert close(0.298, 1.7 / 5.70, "derived")
    assert not close(9.5, (9.4 + 8.9 + 9.2) / 3, "derived")  # the miss observed
    assert not close(5.4, 4.8, "derived")                    # coder's wrong sum


def test_lookup_tolerance_is_exact() -> None:
    """A cell printed in the document must match it, not merely come close."""
    assert close(11.0, 11.0, "lookup")
    assert close(5.7, 5.70, "lookup")
    assert not close(11.1, 11.0, "lookup")   # the adjacent grid
    assert not close(10.0, 10.5, "lookup")   # the integer-rounding bug


@pytest.mark.parametrize("grid", sorted(GRID))
def test_no_two_distinct_readings_are_confusable(grid: str) -> None:
    """A wrong-row answer must not score as correct.

    Exact duplicates are excluded deliberately: S6 and N2 both read 9.4 mm in
    2024, so answering either returns a numerically correct value and no false
    credit is possible. What must never happen is two *different* readings
    landing inside the tolerance of one another — that is what would let a
    misread row pass, and it is why the lookup tolerance is 0.005 rather than
    the 2% this originally used.
    """
    mine = GRID[grid][1]
    confusable = [
        g for g in GRID
        if g != grid and GRID[g][1] != mine and close(GRID[g][1], mine, "lookup")
    ]
    assert not confusable, f"{grid} ({mine}) is confusable with {confusable}"


def test_questions_are_balanced_and_labelled() -> None:
    kinds = [kind for _, _, kind in QUESTIONS]
    assert set(kinds) == {"lookup", "derived"}
    assert kinds.count("lookup") >= 8 and kinds.count("derived") >= 8


# --- the loop-behaviour scorer ----------------------------------------------

def test_judge_credits_a_correct_tool_call() -> None:
    score = Score()
    judge({"tool": "fs_read", "args": {"path": "inbox/a.md"}},
          {"tool": "fs_read", "args": {"path": "inbox/a.md"}}, score)
    assert (score.json_valid, score.tool_match, score.args_valid) == (1, 1, 1)


def test_judge_rejects_unparseable_output() -> None:
    score = Score()
    judge(None, {"tool": "fs_read", "args": {"path": "x"}}, score)
    assert (score.n, score.json_valid, score.tool_match) == (1, 0, 0)


def test_judge_separates_wrong_tool_from_bad_args() -> None:
    wrong_tool = Score()
    judge({"tool": "doc_write", "args": {}}, {"tool": "fs_read", "args": {"path": "x"}}, wrong_tool)
    assert (wrong_tool.tool_match, wrong_tool.args_valid) == (0, 0)

    bad_args = Score()
    judge({"tool": "fs_read", "args": {"nope": 1}}, {"tool": "fs_read", "args": {"path": "x"}}, bad_args)
    assert (bad_args.tool_match, bad_args.args_valid) == (1, 0)


def test_judge_tracks_unknown_honesty_only_where_it_applies() -> None:
    score = Score()
    judge({"answer": "UNKNOWN - not stated"}, {"answer": "UNKNOWN - not stated"}, score)
    judge({"answer": "8.9 mm"}, {"answer": "8.9 mm"}, score)
    assert (score.unknown_total, score.unknown_hit) == (1, 1)

    missed = Score()
    judge({"answer": "probably 12 mm"}, {"answer": "UNKNOWN - not stated"}, missed)
    assert (missed.unknown_total, missed.unknown_hit) == (1, 0)


def test_answering_when_a_tool_was_expected_scores_zero() -> None:
    score = Score()
    judge({"answer": "8.9 mm"}, {"tool": "fs_read", "args": {"path": "x"}}, score)
    assert score.tool_match == 0


# --- label masking (needs the staged tokenizer) ------------------------------

TOKENIZER_DIR = pathlib.Path(__file__).resolve().parents[1] / "models" / "hf" / "driver"
needs_tokenizer = pytest.mark.skipif(
    not (TOKENIZER_DIR / "tokenizer.json").exists(),
    reason="run `python -m finetune.stage driver` first",
)


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))


@needs_tokenizer
def test_every_trace_encodes(tokenizer: Any) -> None:
    """encode() returns None on any misalignment, so a silent API change shows
    up as an empty training set. transformers 5.x returning a BatchEncoding
    instead of list[int] did exactly that and dropped all 612 traces."""
    from finetune.qlora import encode

    encoded = [encode(tokenizer, t["messages"]) for t in TRACES]
    assert all(e is not None for e in encoded)


@needs_tokenizer
def test_loss_falls_only_on_assistant_turns(tokenizer: Any) -> None:
    """The one bug that looks like a healthy training run and produces a
    useless model. Decode the supervised spans and check they are the model's
    own JSON replies — not the system prompt, not the observations."""
    from finetune.qlora import encode

    for trace in TRACES[:20]:
        encoded = encode(tokenizer, trace["messages"])
        assert encoded is not None
        spans, start = [], None
        for i, label in enumerate(encoded["labels"]):
            if label != -100 and start is None:
                start = i
            elif label == -100 and start is not None:
                spans.append((start, i))
                start = None
        if start is not None:
            spans.append((start, len(encoded["labels"])))

        expected = sum(1 for m in trace["messages"] if m["role"] == "assistant")
        assert len(spans) == expected, "one supervised span per assistant turn"
        for a, b in spans:
            text = tokenizer.decode(encoded["input_ids"][a:b])
            assert text.lstrip().startswith("{"), f"supervised non-JSON: {text[:80]!r}"
            assert "Observation:" not in text, "observation leaked into the loss"


@needs_tokenizer
def test_labels_align_with_inputs(tokenizer: Any) -> None:
    from finetune.qlora import encode

    encoded = encode(tokenizer, TRACES[0]["messages"])
    assert encoded is not None
    assert len(encoded["input_ids"]) == len(encoded["labels"])
    for token, label in zip(encoded["input_ids"], encoded["labels"]):
        assert label in (-100, token), "a label must be the token or masked"


# --- held-out split selection ------------------------------------------------


def test_eval_split_is_disjoint_from_train() -> None:
    """The whole point of a held-out loss. If eval.jsonl ever gets regenerated
    from the same pool as train.jsonl, best-checkpoint selection silently
    starts choosing the most memorised adapter instead of the best one."""
    import json
    import pathlib as _p

    data = _p.Path(__file__).resolve().parents[1] / "finetune" / "data"
    prompts = {}
    for name in ("train.jsonl", "eval.jsonl"):
        with (data / name).open(encoding="utf-8") as fh:
            prompts[name] = {
                next(m["content"] for m in json.loads(line)["messages"]
                     if m["role"] == "user")
                for line in fh if line.strip()
            }
    assert not prompts["train.jsonl"] & prompts["eval.jsonl"]


def test_eval_stride_spreads_across_the_file_and_caps_at_n() -> None:
    """qlora.py subsamples eval.jsonl by stride, not by head slice: the file is
    written grouped by trace kind, so eval_rows[:n] would score one composition
    bucket and report it as the held-out loss."""
    rows = list(range(453))
    for n in (32, 96, 453, 1000):
        stride = max(1, len(rows) // n)
        picked = rows[::stride][:n]
        assert len(picked) == min(n, len(rows))
        assert len(set(picked)) == len(picked)
        # spans the file rather than clustering at the front
        assert picked[-1] > len(rows) * 0.8
