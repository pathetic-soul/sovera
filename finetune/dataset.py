"""Build the QLoRA training set for `driver` (AGENTS.md §12, §8.4).

What this teaches, honestly
---------------------------
Format and behaviour, not knowledge. Template-generated traces can reliably
teach a 4B to emit well-formed agent-loop JSON, pick the right tool, fill a
flat schema, write a thought that refers to *this* task, and say UNKNOWN when
the document does not answer the question. That is exactly the §12 failure
surface the repair path in core/agent.py exists to paper over.

It will NOT fix the leg-5 grounding gap (§16) — reading the correct row out of
a markdown table and getting the sign of a subtraction right is reasoning
ability, not output style, and no amount of synthetic formatting data supplies
it. The one adjacent thing it can teach is the *habit* of quoting the source
row before computing, which makes a wrong read visible instead of silent.

Why this does not touch the §13 router gate
-------------------------------------------
The gate measures core/routing/, a lexical TF-IDF scorer with no model in
it — §2.3 keeps routing deterministic and no LLM decides a route. Fine-tuning
`driver` cannot move that number in either direction. The refinery phrasings in
config/routing_exemplars.jsonl are reused here only as realistic *task text*;
the labels are ignored, and nothing here is trained to classify.

The eval split below is a separate, new held-out set for agent-loop behaviour.
Report its numbers as their own metric, never as the router number.

    python -m finetune.dataset            # writes finetune/data/{train,eval}.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator

from core.agent import SYSTEM, _tool_lines
from tools.base import Tool
from tools.calc import Calc
from tools.doc_write import DocWrite
from tools.fs_read import FsRead
from tools.py_sandbox import PySandbox

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "finetune" / "data"
EXEMPLARS = ROOT / "config" / "routing_exemplars.jsonl"

SEED = 26117  # the problem statement number, so a rerun reproduces the split
EVAL_FRACTION = 0.15

TOOLS: dict[str, Tool] = {t.name: t for t in (FsRead(), Calc(), PySandbox(), DocWrite())}
SYSTEM_TEXT = SYSTEM.replace("{tools}", _tool_lines(TOOLS))

# §15 vocabulary. Tags must survive intact (§9.4) so they appear verbatim in
# both the task text and the expected arguments.
TAGS = [
    "V-2301", "10-P-101A", "PSV-4402", "E-1502", "T-3301", "C-2101",
    "D-4405", "P-2205B", "HX-1103", "V-1201", "10-P-304B", "PSV-1130",
    "E-2204", "T-1102", "V-3402", "C-3301", "D-1205", "HX-2401",
]
UNITS = ["CDU", "VDU", "FCC", "HGU", "SRU", "CCR", "DHDS"]
METHODS = ["UT thickness survey", "MPI", "DPI", "radiography"]
CODES = ["API 570", "API 510", "API 653"]
# Things a real inspection report genuinely may not state — the UNKNOWN lever
# in §12.7 only means anything if the honest answer is sometimes "not stated".
ABSENT_FIELDS = [
    "the welder qualification number", "the PWHT soak temperature",
    "the coating specification", "the vendor purchase order number",
    "the radiographer's certification expiry", "the hydrotest pressure",
    "the insulation type", "the gasket material",
]


def doc_name(tag: str, year: int, seq: int) -> str:
    return f"inbox/UT-{year}-{seq:03d}-{tag}.md"


def _thought(text: str) -> str:
    """Task-specific by construction.

    The 8B copied the few-shot thought verbatim during leg 4 ("I must read the
    report before summarising it."), which makes the live plan trace look
    canned. Every thought here names the actual tag or file.
    """
    return text


def _call(tool: str, args: dict[str, Any], thought: str) -> str:
    return json.dumps({"thought": _thought(thought), "tool": tool, "args": args})


def _answer(text: str, thought: str) -> str:
    return json.dumps({"thought": _thought(thought), "answer": text})


def _turns(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": SYSTEM_TEXT}]
    for role, content in pairs:
        msgs.append({"role": role, "content": content})
    return msgs


def _report_stub(tag: str, unit: str, code: str, tmin: float, tmeas: float) -> str:
    """A compact stand-in for a read observation. Shaped like the real corpus
    document so the model sees the same layout it will meet at inference."""
    return (
        f"# INSPECTION REPORT\n**Equipment tag:** {tag}\n**Unit:** {unit}\n"
        f"**Inspection code:** {code}\n\n## Thickness readings (mm)\n"
        f"| Grid | 2019 | 2024 |\n|---|---|---|\n"
        f"| S1 | {tmeas + 2.1:.1f} | {tmeas + 1.8:.1f} |\n"
        f"| S7 | {tmeas + 1.7:.1f} | {tmeas:.1f} |\n\n"
        f"**t-min (retirement limit):** {tmin:.1f} mm\n"
        f"Minimum measured thickness: **{tmeas:.1f} mm at grid S7**.\n"
    )


def gen_read_then_answer(rng: random.Random) -> list[dict[str, str]]:
    tag, unit = rng.choice(TAGS), rng.choice(UNITS)
    code = rng.choice(CODES)
    path = doc_name(tag, rng.choice([2023, 2024, 2025]), rng.randint(1, 320))
    tmin, tmeas = rng.uniform(6.0, 8.0), rng.uniform(8.2, 11.0)
    obs = _report_stub(tag, unit, code, tmin, tmeas)
    return _turns(
        ("user", f"What is the minimum measured wall thickness for {tag}? It is in {path}."),
        ("assistant", _call("fs_read", {"path": path},
                            f"The thickness for {tag} is in {path}; I must read it, not recall it.")),
        ("user", f"Observation:\n{obs}"),
        ("assistant", _answer(
            f"{tmeas:.1f} mm, at grid S7 (source: \"Minimum measured thickness: "
            f"{tmeas:.1f} mm at grid S7\"). The retirement limit is {tmin:.1f} mm, "
            f"so there is {tmeas - tmin:.1f} mm of margin remaining.",
            f"The report states the minimum for {tag} directly; I will quote the row it came from.")),
    )


def gen_read_then_draft(rng: random.Random) -> list[dict[str, str]]:
    tag, unit = rng.choice(TAGS), rng.choice(UNITS)
    code = rng.choice(CODES)
    path = doc_name(tag, 2024, rng.randint(1, 320))
    tmin, tmeas = rng.uniform(6.0, 8.0), rng.uniform(8.2, 11.0)
    obs = _report_stub(tag, unit, code, tmin, tmeas)
    body = (
        f"## Background\\n- {code} inspection of {tag} in the {unit}\\n"
        f"## Findings\\n- Minimum thickness {tmeas:.1f} mm at grid S7 against a "
        f"{tmin:.1f} mm retirement limit\\n"
        f"## Recommendation\\n- Re-survey grid S7 at the next opportunity"
    )
    return _turns(
        ("user", f"Read {path} and draft an approval note for the {tag} recommendations."),
        ("assistant", _call("fs_read", {"path": path},
                            f"I need the {tag} findings before I can draft anything.")),
        ("user", f"Observation:\n{obs}"),
        ("assistant", _call("doc_write", {
            "filename": f"approval-note-{tag}",
            "title": f"Approval Note - {tag} {code} inspection",
            "body": body,
        }, f"I have the {tag} findings and can now draft the note for review.")),
        ("user", "Observation:\nWrote out/approval-note.docx (36112 bytes). "
                 "Open it to review before signing."),
        ("assistant", _answer(
            f"Drafted the approval note for {tag} covering the {code} findings and the "
            f"grid S7 recommendation. It is in workspace/out/ and needs your sign-off.",
            "The note is written; the human reviewer signs it, not me.")),
    )


def gen_read_then_compute(rng: random.Random) -> list[dict[str, str]]:
    """Teaches the *habit* of restating the source values in the code itself.

    This is the only lever this dataset has on the §16 grounding gap: it does
    not make the model read the table correctly, but a script that names the
    grid and the year it took each number from makes a misread visible on the
    plan trace instead of silent in the arithmetic.
    """
    tag = rng.choice(TAGS)
    path = doc_name(tag, 2024, rng.randint(1, 320))
    tmin, tmeas = rng.uniform(6.0, 8.0), rng.uniform(8.2, 11.0)
    prev, years = tmeas + 1.7, 5.7
    obs = _report_stub(tag, rng.choice(UNITS), rng.choice(CODES), tmin, tmeas)
    code_lines = "\n".join([
        f"# {tag} grid S7, from {path}",
        f"t_2019 = {prev:.1f}   # grid S7, 2019 column",
        f"t_2024 = {tmeas:.1f}   # grid S7, 2024 column",
        f"t_min  = {tmin:.1f}   # stated retirement limit",
        f"years  = {years}",
        "rate = (t_2019 - t_2024) / years",
        "remaining = (t_2024 - t_min) / rate   # margin above t_min, not below",
        "print('corrosion rate mm/yr:', round(rate, 3))",
        "print('remaining life yr:', round(remaining, 2))",
    ])
    rate = (prev - tmeas) / years
    remaining = (tmeas - tmin) / rate
    return _turns(
        ("user", f"Compute the corrosion rate and remaining life for grid S7 of {tag} using {path}."),
        ("assistant", _call("fs_read", {"path": path},
                            f"I must take the {tag} grid S7 readings from the report, not assume them.")),
        ("user", f"Observation:\n{obs}"),
        ("assistant", _call("py_sandbox", {
            "code": code_lines,
            "purpose": f"corrosion rate and remaining life for {tag} grid S7",
        }, f"I will restate each {tag} value with the grid and year it came from, then compute.")),
        ("user", f"Observation:\nexit_code: 0\nstdout:\ncorrosion rate mm/yr: "
                 f"{rate:.3f}\nremaining life yr: {remaining:.2f}\n"),
        ("assistant", _answer(
            f"Corrosion rate {rate:.3f} mm/yr and remaining life {remaining:.2f} years for "
            f"{tag} grid S7, from the 2019 reading {prev:.1f} mm, the 2024 reading "
            f"{tmeas:.1f} mm and the {tmin:.1f} mm retirement limit.",
            "The figures are computed from the quoted rows, so they can be checked.")),
    )


def gen_unknown(rng: random.Random) -> list[dict[str, str]]:
    """§12.7 — the single most effective anti-hallucination lever on a 4B, and
    the one that matters most for inspection data."""
    tag = rng.choice(TAGS)
    field = rng.choice(ABSENT_FIELDS)
    path = doc_name(tag, 2024, rng.randint(1, 320))
    obs = _report_stub(tag, rng.choice(UNITS), rng.choice(CODES),
                       rng.uniform(6.0, 8.0), rng.uniform(8.2, 11.0))
    return _turns(
        ("user", f"What is {field} for {tag}? Check {path}."),
        ("assistant", _call("fs_read", {"path": path},
                            f"I need to check whether {path} states this for {tag}.")),
        ("user", f"Observation:\n{obs}"),
        ("assistant", _answer(
            f"UNKNOWN — {path} does not state {field} for {tag}. The report covers "
            f"thickness readings and the retirement limit only. Ask the inspection "
            f"section for the source record rather than inferring it.",
            f"The report does not contain {field}; saying UNKNOWN is correct here.")),
    )


def gen_missing_file(rng: random.Random) -> list[dict[str, str]]:
    """Recovery from a bad guess, using the file list fs_read returns (§12.6)."""
    tag = rng.choice(TAGS)
    wrong = doc_name(tag, 2024, rng.randint(1, 320))
    right = doc_name(tag, 2024, rng.randint(400, 999))
    return _turns(
        ("user", f"Summarise the latest inspection findings for {tag}."),
        ("assistant", _call("fs_read", {"path": wrong},
                            f"I will look for the {tag} report in the inbox.")),
        ("user", f"Observation:\nno such file: {wrong}. Files available: {right}"),
        ("assistant", _call("fs_read", {"path": right},
                            f"That path was wrong; the listing shows the {tag} report is at {right}.")),
        ("user", f"Observation:\n{_report_stub(tag, rng.choice(UNITS), rng.choice(CODES), 7.4, 8.9)}"),
        ("assistant", _answer(
            f"{tag}: minimum measured thickness 8.9 mm at grid S7 against a 7.4 mm "
            f"retirement limit, leaving 1.5 mm of margin.",
            f"I have the {tag} report now and can summarise what it states.")),
    )


def gen_denied(rng: random.Random) -> list[dict[str, str]]:
    """§2.4 — a denial is final. The model must not retry the same write."""
    tag = rng.choice(TAGS)
    return _turns(
        ("user", f"Draft an approval note for the {tag} recommendations."),
        ("assistant", _call("doc_write", {
            "filename": f"approval-note-{tag}",
            "title": f"Approval Note - {tag}",
            "body": f"## Recommendation\\n- Re-survey {tag} at the next shutdown",
        }, f"I will draft the {tag} note for the reviewer to approve.")),
        ("user", "Observation:\nThe human reviewer declined this action. Do not retry it. "
                 "Either finish with what you have, or take a different approach."),
        ("assistant", _answer(
            f"Understood — no file was written. Here is the {tag} note in full for you to "
            f"paste or amend:\n\nRecommendation: re-survey {tag} at the next shutdown.",
            "The reviewer declined the write, so I will not retry it and will answer here instead.")),
    )


def gen_derived_via_sandbox(rng: random.Random) -> list[dict[str, str]]:
    """Never do arithmetic in your head — call the sandbox.

    This generator exists because of a measurement, not a hunch.
    finetune/grounding_eval.py on the real corpus table gives:

        driver (qwen3:4b-instruct)   lookup 10/10 100%   derived 4/8  50%
        coder  (qwen2.5-coder:7b)    lookup 10/10 100%   derived 2/8  25%

    Both models find the right cells every time and then get the sum wrong —
    `coder` quoted "S6: 1.5, S7: 1.7, S8: 1.6" and answered 5.4 instead of 4.8.
    Retrieval is not the problem; mental arithmetic is, and it is worse on the
    larger model. Fine-tuning cannot install a working adder into a 4B, but it
    can teach the habit of sending every derivation to py_sandbox, which is
    exact. That is the whole point of having built leg 5.
    """
    tag = rng.choice(TAGS)
    path = doc_name(tag, 2024, rng.randint(1, 320))
    grids = ["S6", "S7", "S8"]
    losses = [round(rng.uniform(0.9, 2.1), 1) for _ in grids]
    readings = [round(rng.uniform(8.2, 11.5), 1) for _ in grids]
    kind = rng.choice(["mean_reading", "total_loss", "mean_loss", "margin"])
    tmin = round(rng.uniform(6.0, 7.8), 1)

    if kind == "mean_reading":
        question = f"What is the mean of the 2024 readings at grids {', '.join(grids)} for {tag}?"
        values, expr, unit = readings, "sum(vals) / len(vals)", "mm"
    elif kind == "total_loss":
        question = f"What is the total thickness loss across grids {', '.join(grids)} for {tag}?"
        values, expr, unit = losses, "sum(vals)", "mm"
    elif kind == "mean_loss":
        question = f"What is the mean thickness loss across grids {', '.join(grids)} for {tag}?"
        values, expr, unit = losses, "sum(vals) / len(vals)", "mm"
    else:
        question = f"How much margin above t-min remains at the thinnest grid of {tag}?"
        values, expr, unit = [min(readings), tmin], "vals[0] - vals[1]", "mm"

    result = eval(expr, {"vals": values})  # noqa: S307 - fixed expressions above
    listed = ", ".join(f"'{g}': {v}" for g, v in zip(grids, values)) if kind != "margin" \
        else f"'t_meas': {values[0]}, 't_min': {values[1]}"
    code = "\n".join([
        f"# {tag}, values read from {path}",
        f"vals = {values}   # {listed}",
        f"answer = {expr}",
        "print(round(answer, 4))",
    ])
    return _turns(
        ("user", f"{question} The report is {path}."),
        ("assistant", _call("fs_read", {"path": path},
                            f"I need the {tag} readings from {path} before computing anything.")),
        ("user", f"Observation:\n{_report_stub(tag, rng.choice(UNITS), rng.choice(CODES), tmin, min(readings))}"),
        ("assistant", _call("py_sandbox", {
            "code": code,
            "purpose": f"{kind.replace('_', ' ')} for {tag}",
        }, "This needs arithmetic, so I will compute it in the sandbox rather than in my head.")),
        ("user", f"Observation:\nexit_code: 0\nstdout:\n{round(result, 4)}\n"),
        ("assistant", _answer(
            f"{round(result, 3)} {unit}, computed in the sandbox from the values "
            f"{values} taken from {path}.",
            "The sandbox returned an exact figure, so I can report it as checked.")),
    )


def gen_derived_via_calc(rng: random.Random) -> list[dict[str, str]]:
    """Arithmetic goes to `calc`, which has the formula fixed in code.

    This replaces an earlier generator that sent arithmetic to py_sandbox.
    That was measured and it was wrong: py_sandbox executes whatever formula
    the model writes, and the fine-tuned model wrote `interval / rate` in 7 of
    8 trials, getting 19.11 years instead of 5.03. The sandbox guarantees the
    arithmetic, not the formula.

    calc takes named parameters, so the model supplies the numbers and chooses
    which formula applies; it cannot invent the algebra. Trained here so the
    adapter reaches for the tool that exists rather than the one it remembers.
    """
    tag = rng.choice(TAGS)
    path = doc_name(tag, 2024, rng.randint(1, 320))
    t_2019 = round(rng.uniform(9.8, 12.4), 1)
    t_2024 = round(t_2019 - rng.uniform(0.6, 2.0), 1)
    tmin = round(rng.uniform(6.0, 7.8), 1)
    years = 5.70
    obs = _report_stub(tag, rng.choice(UNITS), rng.choice(CODES), tmin, t_2024)

    rate = (t_2019 - t_2024) / years
    remaining = (t_2024 - tmin) / rate

    return _turns(
        ("user", f"Using {path}, compute the corrosion rate and remaining life for {tag} grid S7."),
        ("assistant", _call("fs_read", {"path": path},
                            f"I need the {tag} grid S7 readings from {path} before computing anything.")),
        ("user", f"Observation:\n{obs}"),
        ("assistant", _call("calc", {
            "expression": f"corrosion_rate(t_previous={t_2019}, t_current={t_2024}, years={years})",
        }, f"calc has the API 510 rate formula; I will not do this arithmetic myself.")),
        ("user", f"Observation:\ncorrosion_rate  [API 510 7.1]\n  formula:     (t_previous - t_current) / years\n"
                 f"  substituted: ({t_2019} - {t_2024}) / {years}\n  result:      {rate:.4f} mm/year"),
        ("assistant", _call("calc", {
            "expression": f"remaining_life(t_current={t_2024}, t_min={tmin}, corrosion_rate={rate:.4f})",
        }, f"Now the remaining life for {tag}, again with the fixed formula rather than my own.")),
        ("user", f"Observation:\nremaining_life  [API 510 7.1]\n  formula:     (t_current - t_min) / corrosion_rate\n"
                 f"  substituted: ({t_2024} - {tmin}) / {rate:.4f}\n  result:      {remaining:.4f} years"),
        ("assistant", _answer(
            f"Corrosion rate {rate:.4f} mm/year and remaining life {remaining:.2f} years for {tag} grid S7, "
            f"from the 2019 reading {t_2019} mm, the 2024 reading {t_2024} mm and the {tmin} mm retirement "
            f"limit. Both figures come from calc, which shows its substitution.",
            "calc returned both figures with their working, so they can be checked.")),
    )


def gen_lookup_needs_no_tool(rng: random.Random) -> list[dict[str, str]]:
    """A value printed in the document is read, not computed.

    Lookup already measures 100%; this exists so the calc habit does not
    over-generalise into calling a calculator to restate a number.
    """
    tag = rng.choice(TAGS)
    path = doc_name(tag, 2024, rng.randint(1, 320))
    tmin = round(rng.uniform(6.0, 7.8), 1)
    tmeas = round(rng.uniform(8.2, 11.0), 1)
    obs = _report_stub(tag, rng.choice(UNITS), rng.choice(CODES), tmin, tmeas)
    return _turns(
        ("user", f"What is the retirement limit for {tag}? It is in {path}."),
        ("assistant", _call("fs_read", {"path": path},
                            f"The {tag} retirement limit is stated in {path}.")),
        ("user", f"Observation:\n{obs}"),
        ("assistant", _answer(
            f"{tmin} mm (source: \"t-min (retirement limit): {tmin} mm\"). The report states it "
            f"directly, so no calculation is needed.",
            "This value is printed in the report; reading it is enough.")),
    )


GENERATORS = [
    (gen_read_then_answer, 4),
    (gen_read_then_draft, 3),
    (gen_read_then_compute, 2),     # py_sandbox for real scripting, not sums
    (gen_derived_via_calc, 7),      # arithmetic goes to calc: the measured fix
    (gen_lookup_needs_no_tool, 3),  # guard against over-calling calc
    (gen_unknown, 3),
    (gen_missing_file, 2),
    (gen_denied, 2),
]


def build(total: int = 720) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    weighted: list[Any] = []
    for fn, weight in GENERATORS:
        weighted += [fn] * weight
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    guard = 0
    while len(rows) < total and guard < total * 40:
        guard += 1
        messages = rng.choice(weighted)(rng)
        key = json.dumps(messages, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"messages": messages})
    return rows


def _user_prompt(row: dict[str, Any]) -> str:
    """The first user turn — the unit the split must be taken on."""
    return str(next(m["content"] for m in row["messages"] if m["role"] == "user"))


def main() -> int:
    rows = build()
    rng = random.Random(SEED)

    # Split on the USER PROMPT, not on the trace.
    #
    # build() dedupes whole traces, but the generators emit several traces per
    # prompt — 3,040 traces over 1,822 distinct prompts. Shuffling traces and
    # cutting therefore put the same question on both sides: 136 prompts, 36.5%
    # of the eval set, appeared verbatim in train.
    #
    # That is not a cosmetic leak. eval.jsonl exists so qlora.py can keep the
    # best checkpoint by held-out loss; scored against prompts the model has
    # already trained on, "best" selects for memorisation — which is the exact
    # failure that produced loss 0.0016 and three adapters worse than base.
    #
    # Grouping first also makes the split honest per prompt rather than per
    # trace, so a prompt's variants stay together and the eval fraction is
    # measured in prompts.
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_user_prompt(row), []).append(row)
    keys = sorted(groups)               # sorted first: dict order must not leak in
    rng.shuffle(keys)
    cut = int(len(keys) * (1 - EVAL_FRACTION))
    train_rows = [r for k in keys[:cut] for r in groups[k]]
    eval_rows = [r for k in keys[cut:] for r in groups[k]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train_rows), ("eval", eval_rows)):
        path = OUT_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in part:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        turns = sum(len(r["messages"]) for r in part)
        print(f"[dataset] {name}: {len(part)} traces, {turns} messages -> {path}")
    return 0


def _cli() -> int:
    """Guarded entry point.

    THIS MODULE IS SUPERSEDED BY finetune/traceset.py.

    It also used to be dangerous: there was no argument parsing at all, so
    `main()` ran on *any* invocation and rewrote finetune/data/{train,eval}.jsonl
    unconditionally. `python -m finetune.dataset --help` silently destroyed a
    curated 2,469-trace corpus and replaced it with this generator's 612 —
    which is exactly the output measured to produce adapters worse than base.

    The generators below are still imported by tests as a regression fixture,
    so the file stays. Writing over the real corpus now takes an explicit flag.
    """
    import argparse

    ap = argparse.ArgumentParser(
        description="SUPERSEDED by finetune/traceset.py. Kept for its generators.")
    ap.add_argument("--force-overwrite", action="store_true",
                    help="really rewrite finetune/data/{train,eval}.jsonl with the OLD "
                         "612-trace corpus. You almost certainly want traceset instead.")
    args = ap.parse_args()

    if not args.force_overwrite:
        print(
            "finetune.dataset is superseded by finetune.traceset and writes nothing." "\n"
            "  python -m finetune.traceset                    # the current corpus" "\n"
            "  python -m finetune.dataset --force-overwrite   # the old one, if you must"
        )
        return 0
    return main()


if __name__ == "__main__":
    raise SystemExit(_cli())
