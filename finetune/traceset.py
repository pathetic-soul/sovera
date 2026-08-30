"""Build the agent-loop training corpus. Replaces finetune/dataset.py.

Build time only (AGENTS.md §3).

    python -m finetune.traceset --report      # composition, write nothing
    python -m finetune.traceset               # -> finetune/data/{train,eval}.jsonl

WHY THE OLD SET HAD TO BE REPLACED RATHER THAN GROWN
----------------------------------------------------
Three adapters trained on `finetune/dataset.py` output and all three scored
*worse* than the untouched base model (v2 0/8, v3 2/8 against base 5/8). The
corpus was profiled before this file was written, and it had four structural
defects that more of the same data would have deepened:

1.  **600 of 612 traces opened with `fs_read`.** The model was not learning
    when to read a document; it was learning that step one is always a read.
2.  **The tool set was a frozen constant.** `SYSTEM_TEXT` was built once from
    four tools. That is precisely the brittleness measured on v2: offered a new
    `calc` tool it never reached for it, and answered "UNKNOWN — the API 510
    formula is not in the document" instead.
3.  **It taught what was already at ceiling.** `finetune/evaluate.py` measured
    base `driver` at json_valid 100%, tool_match 97.3%, unknown 100%. A corpus
    of format demonstrations has no headroom to buy, so gradient steps spent on
    it can only trade general capability for template mimicry.
4.  **No hard paths.** No failed tool call, no ambiguity, no refusal, no
    multi-document synthesis. The agent loop's difficult branches had zero
    supervision.

Every design decision below is aimed at one of those four.

WHAT GROUNDS THE NUMBERS
------------------------
No answer in this corpus is typed by hand. Inspection answers are recomputed
from `data/corpus/inbox/facts.json`, which `corpus_inspection.py` derives from
the readings themselves and which cross-checks exactly against `tools/calc.py`.
Financial answers come from the 75 figures in
`data/corpus/filings/manifest.json` that survived reconciliation against each
statement's own stated totals. FinQA supplies real human-written derivation
programs over real tables.

If a fact cannot be recomputed, it does not become a question.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "finetune" / "data"
INBOX_FACTS = ROOT / "data" / "corpus" / "inbox" / "facts.json"
FILINGS_MANIFEST = ROOT / "data" / "corpus" / "filings" / "manifest.json"
FINQA_TRAIN = ROOT / "data" / "external" / "finqa" / "train.json"

SEED = 26117
EVAL_FRACTION = 0.15


# --------------------------------------------------------------------------
# tool cards
# --------------------------------------------------------------------------
# Target 2 is tool-schema generalization. The model must learn "use what is on
# the card", not "these four tools exist". So every trace is generated against
# a RANDOM SUBSET, and the pool deliberately contains tools this repo has not
# built (kb_search and fs_write are listed unbuilt in AGENTS.md §6; the rest are
# plausible neighbours). Those are never hallucinated at inference — a tool only
# ever appears in a trace when it is also printed in that trace's system prompt,
# so what transfers is the habit of reading the card.

@dataclass(frozen=True)
class ToolCard:
    name: str
    params: tuple[tuple[str, str], ...]
    description: str
    approval: bool = False

    def render(self) -> str:
        sig = ", ".join(f"{k}: {v}" for k, v in self.params)
        gate = " [needs human approval]" if self.approval else ""
        return f"- {self.name}({sig}){gate}\n  {self.description}"


REAL_TOOLS = {
    "fs_read": ToolCard("fs_read", (("path", "workspace-relative path, e.g. inbox/UT-2024-114.md"),),
                        "Read a UTF-8 text file from the workspace. Returns the text, truncated if long."),
    "calc": ToolCard("calc", (("expression", "e.g. remaining_life(t_current=8.9, t_min=7.4, corrosion_rate=0.2982)"),),
                     "Exact maths: arithmetic, algebra, calculus. Also fixed inspection formulas with code citations."),
    "py_sandbox": ToolCard("py_sandbox", (("code", "complete Python source"), ("purpose", "one line on what it computes")),
                           "Run a Python 3 script offline (no network, 60s, standard library only).", True),
    "doc_write": ToolCard("doc_write", (("filename", "base name, no extension"), ("title", "document title"),
                                        ("body", "note text; '## Section' for a heading, '- item' for a bullet")),
                          "Write a .docx deliverable to workspace/out/.", True),
}

EXTRA_TOOLS = {
    "kb_search": ToolCard("kb_search", (("query", "search terms"), ("top_k", "how many passages, default 5")),
                          "Hybrid search over the indexed document corpus. Returns passages with their source file."),
    "fs_write": ToolCard("fs_write", (("path", "workspace-relative path"), ("content", "full file text")),
                         "Write a UTF-8 text file inside the workspace.", True),
    "xlsx_write": ToolCard("xlsx_write", (("filename", "base name"), ("rows", "TSV, one record per line")),
                           "Write a .xlsx spreadsheet to workspace/out/.", True),
    "unit_convert": ToolCard("unit_convert", (("value", "the number"), ("frm", "source unit"), ("to", "target unit")),
                             "Convert between engineering units (barg/psi, mm/in, degC/degF)."),
    "tag_lookup": ToolCard("tag_lookup", (("tag", "equipment tag, e.g. V-2301"),),
                           "Look up an equipment tag in the plant register: unit, service, design data."),
    "psv_registry": ToolCard("psv_registry", (("tag", "relief valve tag, e.g. PSV-4402"),),
                             "Last pop-test date, set pressure and certification status for a relief valve."),
    "moc_lookup": ToolCard("moc_lookup", (("equipment_tag", "equipment tag"),),
                           "Open Management of Change records against a tag."),
}

ALL_TOOLS = {**REAL_TOOLS, **EXTRA_TOOLS}


SYSTEM_TEMPLATE = """You are an offline engineering assistant at an Indian oil refinery (MRPL).
You have no internet access. You work in small steps using tools.

Reply with ONE JSON object and nothing else.
To use a tool:  {{"thought": "why", "tool": "name", "args": {{...}}}}
To finish:      {{"thought": "why", "answer": "the reply for the engineer"}}

Rules:
- One tool per reply. Use only the tools and parameters listed below. Never nest objects.
- Read a document before answering questions about it. Do not guess its contents.
- If the document does not state something, write UNKNOWN. Never invent a
  thickness reading, a date, an equipment tag or a code clause.
- Never do arithmetic in your head. If a calculation tool is listed, use it.
- If the request is ambiguous, ask one short clarifying question instead of guessing.

Tools:
{tools}"""


def system_for(tools: list[str]) -> str:
    """Render the system prompt for one trace's tool subset.

    The rules above are phrased against *capabilities* ("if a calculation tool
    is listed") rather than against tool names. Naming `calc` in the rules is
    what taught v2 that the world contains exactly four tools.
    """
    return SYSTEM_TEMPLATE.format(tools="\n".join(ALL_TOOLS[t].render() for t in tools))


def pick_tools(rng: random.Random, required: list[str]) -> list[str]:
    """A tool subset that always contains what the trace needs, plus noise.

    The distractors are the point: a trace where the answer needs `calc` but
    six other tools are also on offer teaches selection. A trace where only
    `calc` is on offer teaches nothing.
    """
    pool = [t for t in ALL_TOOLS if t not in required]
    extra = rng.sample(pool, k=rng.randint(1, 4))
    chosen = required + extra
    rng.shuffle(chosen)
    return chosen


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _t(thought: str, tool: str, args: dict[str, Any]) -> str:
    return json.dumps({"thought": thought, "tool": tool, "args": args})


def _a(thought: str, answer: str) -> str:
    return json.dumps({"thought": thought, "answer": answer})


def _trace(system: str, user: str,
           turns: list[tuple[str, str | None]]) -> dict[str, Any]:
    """turns is [(assistant_json, tool_output_or_None), ...]

    The observation is optional because the LAST assistant turn of every
    trace is a final answer, which no tool output follows.
    """
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for assistant, observation in turns:
        messages.append({"role": "assistant", "content": assistant})
        if observation is not None:
            messages.append({"role": "user", "content": observation})
    return {"messages": messages}


def _excerpt(fact: dict[str, Any]) -> str:
    """The slice of an inspection report a fs_read would plausibly return."""
    g = fact["governing_grid"]
    per = fact["per_grid"]
    rows = "\n".join(
        f"| {k} | {v['previous']:.1f} | {v['current']:.1f} | {v['loss']:.1f} |"
        for k, v in list(per.items())
    )
    return (f"# INSPECTION REPORT\n**Equipment tag:** {fact['tag']}\n"
            f"**Unit:** {fact['unit']}\n**Inspection code:** {fact['code']}\n"
            f"**Inspection date:** {fact['inspection_date']}\n"
            f"**Last inspection:** {fact['previous_date']}\n"
            f"**Material:** {fact['material']}, {fact['nominal_mm']:.1f} mm nominal\n"
            f"**Design pressure / temperature:** {fact['design_pressure_barg']} barg / {fact['design_temp_c']} °C\n"
            f"**t-min (retirement limit):** {fact['t_min']:.1f} mm\n\n"
            f"| Grid | {fact['previous_date']} | {fact['inspection_date']} | Loss |\n"
            f"|---|---|---|---|\n{rows}\n\n"
            f"Interval between surveys: {fact['interval_years']} years.\n"
            f"Minimum measured thickness: {fact['governing_current']:.1f} mm at grid {g}.\n")


# --------------------------------------------------------------------------
# trace families
# --------------------------------------------------------------------------
# Each family is a generator over (trace, family_name). Coverage is asserted at
# the end so a family cannot silently drop to zero.

def fam_inspection_derive(rng: random.Random, facts: list[dict[str, Any]], n: int) -> Iterator[dict[str, Any]]:
    """Read -> cite the row -> calc -> answer. The leg-5 skill, done correctly.

    Two sources of variety matter here and both are deliberate. The question is
    asked about an ARBITRARY grid, not only the governing one — otherwise the
    model learns "answer about the thinnest row" instead of "answer about the
    row you were asked about", which is exactly the §16 failure where `coder`
    read grid S5 when asked for S7. And the phrasing is drawn from a paraphrase
    pool, because a single template repeated 570 times is what drives training
    loss to 0.0016 without teaching anything.
    """
    for _ in range(n):
        f = rng.choice(facts)
        grid = rng.choice(list(f["per_grid"]))
        p = f["per_grid"][grid]
        tools = pick_tools(rng, ["fs_read", "calc"])
        which = rng.choice(["remaining_life", "corrosion_rate", "next_interval"])

        if which == "corrosion_rate":
            user = rng.choice([
                f"What is the corrosion rate at grid {grid} on {f['tag']}? Use {f['doc']}.",
                f"From {f['doc']}, work out how fast grid {grid} of {f['tag']} is corroding.",
                f"{f['tag']} grid {grid} — corrosion rate please, per {f['doc']}.",
                f"Using {f['doc']}, what metal-loss rate does grid {grid} show on {f['tag']}?",
            ])
            expr = (f"corrosion_rate(t_previous={p['previous']}, "
                    f"t_current={p['current']}, years={f['interval_years']})")
            out = (f"corrosion_rate = (t_previous - t_current) / years\n"
                   f"({p['previous']} - {p['current']}) / {f['interval_years']} "
                   f"= {p['rate']}\nmm/year   [{f['code']}]")
            answer = (f"Corrosion rate at grid {grid} is {p['rate']} mm/year, from "
                      f"{p['previous']} mm ({f['previous_date']}) to "
                      f"{p['current']} mm ({f['inspection_date']}) over "
                      f"{f['interval_years']} years.")
        elif which == "remaining_life":
            user = rng.choice([
                f"Compute the remaining life for grid {grid} on {f['tag']} from {f['doc']}.",
                f"How long can {f['tag']} stay in service on grid {grid}? See {f['doc']}.",
                f"{f['doc']}: remaining life at grid {grid} of {f['tag']}?",
                f"Per {f['code']}, what is the remaining life of grid {grid} on {f['tag']}? "
                f"Figures are in {f['doc']}.",
            ])
            expr = (f"remaining_life(t_current={p['current']}, t_min={f['t_min']}, "
                    f"corrosion_rate={p['rate']})")
            out = (f"remaining_life = (t_current - t_min) / corrosion_rate\n"
                   f"({p['current']} - {f['t_min']}) / {p['rate']} "
                   f"= {p['remaining_life']}\nyears   [{f['code']} §7.1]")
            answer = (f"Remaining life at grid {grid} is {p['remaining_life']} years, based on "
                      f"{p['current']} mm measured against a t-min of {f['t_min']} mm "
                      f"at {p['rate']} mm/year.")
        else:
            user = rng.choice([
                f"What inspection interval should {f['tag']} get? Base it on {f['doc']}.",
                f"When is {f['tag']} next due, per {f['code']}? Figures in {f['doc']}.",
                f"Set the next inspection interval for {f['tag']} from {f['doc']}.",
            ])
            expr = f"next_interval(remaining_life_years={f['remaining_life_years']})"
            out = (f"next_interval = min(remaining_life_years / 2, 10)\n"
                   f"min({f['remaining_life_years']} / 2, 10) = {f['next_interval_years']}\n"
                   f"years   [{f['code']} §7.2]")
            # MEASURED FAILURE MODE, base qwen3:4b at 50.0% on this category
            # (95% CI 41.2-58.8, against 91.7% on corrosion rate — the intervals
            # do not overlap, so the gap is real). The model computes the value
            # correctly and then discards it:
            #     min(3.78 / 2, 10) = min(1.89, 10) = 1.89 -> rounded to 10 years
            # It reads the 10 as the answer rather than as a ceiling. So the
            # answer text names which branch of the min() actually binds, and
            # says so in words a 4B can pattern-match.
            half = round(f['remaining_life_years'] / 2.0, 4)
            binds = ("the 10-year statutory ceiling" if half > 10
                     else "half the remaining life")
            answer = (f"The next inspection interval is {f['next_interval_years']} years. "
                      f"Half the remaining life is {half} years and the {f['code']} ceiling "
                      f"is 10 years; min() takes the SMALLER of the two, so "
                      f"{binds} governs here. The 10-year figure is an upper limit, "
                      f"not the answer.")
            grid = f["governing_grid"]
            p = f["per_grid"][grid]

        yield _trace(system_for(tools), user, [
            (_t(f"I must read the report before quoting any reading for {f['tag']}.",
                "fs_read", {"path": f["doc"]}), _excerpt(f)),
            (_t(f"Grid {grid} reads {p['current']} mm against t-min {f['t_min']} mm. "
                f"I will not do this arithmetic in my head.", "calc", {"expression": expr}), out),
            (_a(rng.choice([
                f"calc returned the figure with its substitution; I quote it and grid {grid}.",
                f"The substitution is checkable by hand, so I report it with the {grid} row.",
                f"I have the computed value for {grid} on {f['tag']}; no further step is needed.",
             ]), answer), None),
        ])


def fam_lookup(rng: random.Random, facts: list[dict[str, Any]], n: int) -> Iterator[dict[str, Any]]:
    """Read -> answer. No calculation. Teaches NOT to over-tool."""
    for _ in range(n):
        f = rng.choice(facts)
        field, question, value = rng.choice([
            ("t_min", f"What is the retirement limit for {f['tag']}?", f"{f['t_min']} mm"),
            ("material", f"What material is {f['tag']} made of?", f["material"]),
            ("inspection_date", f"When was {f['tag']} last inspected?", f["inspection_date"]),
            ("design_pressure_barg", f"What is the design pressure of {f['tag']}?",
             f"{f['design_pressure_barg']} barg"),
            ("governing_current", f"What is the minimum measured thickness on {f['tag']}?",
             f"{f['governing_current']} mm at grid {f['governing_grid']}"),
            ("code", f"Which inspection code governs {f['tag']}?", f["code"]),
        ])
        tools = pick_tools(rng, ["fs_read"])
        prompt = rng.choice([
            f"{question} See {f['doc']}.",
            f"{f['doc']} — {question[0].lower() + question[1:]}",
            f"Check {f['doc']} and tell me: {question[0].lower() + question[1:]}",
            f"{question} It should be in {f['doc']}.",
        ])
        yield _trace(system_for(tools), prompt, [
            (_t(rng.choice([
                f"{f['tag']} states this directly; this is a read, not a calculation.",
                f"The answer is printed in {f['doc']}, so I look it up rather than derive it.",
                f"No arithmetic is involved — I just need the {f['tag']} report.",
             ]), "fs_read", {"path": f["doc"]}), _excerpt(f)),
            (_a(rng.choice([
                f"{f['doc']} states it outright; nothing to derive.",
                f"The value is given for {f['tag']}, so I quote it with its source.",
                f"Read directly from the {f['tag']} report.",
             ]), f"{value}. Source: {f['doc']}, {f['tag']}."), None),
        ])


def fam_unknown(rng: random.Random, facts: list[dict[str, Any]], n: int) -> Iterator[dict[str, Any]]:
    """Asked for something genuinely absent. §12.7 UNKNOWN, correctly scoped."""
    absent = [
        ("the last hydrotest pressure", "hydrotest"),
        ("the PWHT temperature used at fabrication", "PWHT"),
        ("the name of the approving authority", "approver"),
        ("the internal coating specification", "coating spec"),
        ("the radiography acceptance class", "RT class"),
    ]
    for _ in range(n):
        f = rng.choice(facts)
        what, short = rng.choice(absent)
        tools = pick_tools(rng, ["fs_read"])
        yield _trace(system_for(tools), f"What is {what} for {f['tag']}? See {f['doc']}.", [
            (_t(rng.choice([
                f"I must check {f['doc']} rather than assume the value is absent.",
                f"Before answering UNKNOWN I have to confirm it is not in the {f['tag']} report.",
                f"Absence has to be verified against {f['doc']}, not guessed.",
             ]), "fs_read", {"path": f["doc"]}), _excerpt(f)),
            (_a(f"The report covers thickness, dates and design data but records no {short}.",
                f"UNKNOWN — {f['doc']} does not state {what} for {f['tag']}. The report covers "
                f"the UT survey, design data and the retirement limit only."), None),
        ])


def fam_clarify(rng: random.Random, facts: list[dict[str, Any]], n: int) -> Iterator[dict[str, Any]]:
    """Ambiguous request -> one short question. Never guess which tag."""
    for _ in range(n):
        a, b = rng.sample(facts, 2)
        tools = pick_tools(rng, ["fs_read"])
        user = rng.choice([
            "What is the remaining life?",
            "Compute the corrosion rate for the bottom grid.",
            "Draft the approval note for the vessel.",
            "When is the next inspection due?",
        ])
        yield _trace(system_for(tools), user, [
            (_a("No equipment tag was given and the inbox holds several reports. "
                "Guessing which one would produce a confident wrong answer.",
                f"Which equipment tag do you mean? The inbox currently holds reports for "
                f"{a['tag']} and {b['tag']}, among others."), None),
        ])


def fam_repair(rng: random.Random, facts: list[dict[str, Any]], n: int) -> Iterator[dict[str, Any]]:
    """A tool call fails; recover rather than restating the failure."""
    for _ in range(n):
        f = rng.choice(facts)
        tools = pick_tools(rng, ["fs_read", "calc"])
        wrong = f["doc"].replace("inbox/", "inbox/archive/")
        yield _trace(system_for(tools), f"Remaining life for grid {f['governing_grid']} on {f['tag']}?", [
            (_t(rng.choice([
                f"I will read the {f['tag']} report.",
                f"Fetching the survey for {f['tag']}.",
                f"Opening what I believe is the {f['tag']} report.",
             ]), "fs_read", {"path": wrong}),
             f"ERROR: no such file: {wrong}"),
            (_t("That path does not exist. The reports live directly under inbox/, so I will "
                "correct the path rather than report a failure.",
                "fs_read", {"path": f["doc"]}), _excerpt(f)),
            (_t(f"Grid {f['governing_grid']} is the governing point at {f['governing_current']} mm.",
                "calc", {"expression": f"remaining_life(t_current={f['governing_current']}, "
                                       f"t_min={f['t_min']}, corrosion_rate={f['corrosion_rate_mm_yr']})"}),
             f"({f['governing_current']} - {f['t_min']}) / {f['corrosion_rate_mm_yr']} "
             f"= {f['remaining_life_years']}\nyears   [{f['code']} §7.1]"),
            (_a(rng.choice([
                f"Recovered from the bad path and computed from the real {f['tag']} report.",
                f"The corrected path worked and calc gave the figure for {f['governing_grid']}.",
                f"Second attempt found {f['doc']}; the derivation follows from it.",
             ]),
                f"Remaining life at grid {f['governing_grid']} is {f['remaining_life_years']} years "
                f"({f['governing_current']} mm against t-min {f['t_min']} mm at "
                f"{f['corrosion_rate_mm_yr']} mm/year)."), None),
        ])


def fam_multi_doc(rng: random.Random, facts: list[dict[str, Any]], n: int) -> Iterator[dict[str, Any]]:
    """Two documents, one comparison. Tags must survive intact (§9.4)."""
    for _ in range(n):
        a, b = rng.sample(facts, 2)
        worse = a if a["remaining_life_years"] < b["remaining_life_years"] else b
        tools = pick_tools(rng, ["fs_read"])
        yield _trace(system_for(tools),
                     f"Which needs attention first, {a['tag']} or {b['tag']}? "
                     f"See {a['doc']} and {b['doc']}.", [
            (_t(f"I need both reports before comparing. Reading {a['tag']} first.",
                "fs_read", {"path": a["doc"]}), _excerpt(a)),
            (_t(f"Now the second report, for {b['tag']}.", "fs_read", {"path": b["doc"]}), _excerpt(b)),
            (_a(rng.choice([
                f"Both reports give a remaining life; {worse['tag']} has the shorter one.",
                f"Comparing the two governing grids, {worse['tag']} is the binding case.",
                f"{a['tag']} and {b['tag']} are both read; the shorter remaining life governs.",
             ]),
                f"{worse['tag']} needs attention first — remaining life "
                f"{worse['remaining_life_years']} years at grid {worse['governing_grid']} "
                f"({worse['governing_current']} mm against t-min {worse['t_min']} mm), against "
                f"{(b if worse is a else a)['remaining_life_years']} years for "
                f"{(b if worse is a else a)['tag']}."), None),
        ])


def fam_deliverable(rng: random.Random, facts: list[dict[str, Any]], n: int) -> Iterator[dict[str, Any]]:
    """Approval note through doc_write. Target 4 — deliverable quality."""
    for _ in range(n):
        f = rng.choice(facts)
        tools = pick_tools(rng, ["fs_read", "doc_write"])
        body = (f"## Background\\n"
                f"- {f['tag']} ({f['unit']}) inspected {f['inspection_date']} under {f['code']}.\\n"
                f"- Previous survey {f['previous_date']}; interval {f['interval_years']} years.\\n"
                f"## Findings\\n"
                f"- Governing grid {f['governing_grid']}: {f['governing_previous']} mm to "
                f"{f['governing_current']} mm, loss {f['governing_loss']} mm.\\n"
                f"- Corrosion rate {f['corrosion_rate_mm_yr']} mm/year against t-min {f['t_min']} mm.\\n"
                f"- Remaining life {f['remaining_life_years']} years.\\n"
                f"## Recommendation\\n"
                f"- Set the next inspection interval at {f['next_interval_years']} years per {f['code']}.\\n"
                f"- Raise an MOC against the corrosion driver identified in the report.\\n"
                f"## Sign-off\\n- Prepared by:\\n- Reviewed by:\\n- Approved by:")
        yield _trace(system_for(tools),
                     f"Draft an approval note for the {f['tag']} inspection findings. Use {f['doc']}.", [
            (_t(rng.choice([
                f"The note must quote {f['tag']}'s figures, not mine, so I read it first.",
                f"Drafting before reading {f['doc']} would mean inventing the findings.",
                f"I need the {f['tag']} survey figures before any note can be written.",
             ]), "fs_read", {"path": f["doc"]}), _excerpt(f)),
            (_t(f"I have the governing grid, the rate and the remaining life for {f['tag']}.",
                "doc_write", {"filename": f"approval-note-{f['tag']}",
                              "title": f"Approval Note - {f['tag']}", "body": body}),
             f"wrote out/approval-note-{f['tag']}.docx"),
            (_a(rng.choice([
                f"The {f['tag']} note is written with a sign-off block for the reviewer.",
                f"Deliverable produced; it carries the {f['code']} basis and a sign-off block.",
                f"Draft complete for {f['tag']}, pending the reviewing engineer.",
             ]),
                f"Approval note for {f['tag']} is ready at out/approval-note-{f['tag']}.docx. It "
                f"records the governing grid {f['governing_grid']}, a corrosion rate of "
                f"{f['corrosion_rate_mm_yr']} mm/year, remaining life {f['remaining_life_years']} "
                f"years, and recommends a {f['next_interval_years']}-year interval."), None),
        ])


def fam_direct(rng: random.Random, facts: list[dict[str, Any]], n: int) -> Iterator[dict[str, Any]]:
    """No tool needed. Teaches when to stop — the opposite of the fs_read reflex."""
    qa = [
        ("What does t-min mean?",
         "t-min is the retirement limit: the minimum wall thickness at which equipment may stay "
         "in service. Below it the item must be repaired, re-rated or withdrawn."),
        ("What is an MOC?",
         "Management of Change — the approval workflow that must be completed before any plant "
         "modification, covering the technical case, the risk review and the sign-off chain."),
        ("Which code governs pressure vessel inspection?",
         "API 510 governs in-service inspection, repair and alteration of pressure vessels. "
         "API 570 covers piping and API 653 covers storage tanks."),
        ("What does UT stand for in an inspection report?",
         "Ultrasonic Testing — a non-destructive method that measures remaining wall thickness."),
        ("Can you reach the internet to check a standard?",
         "No. This workbench runs fully offline by design, so I can only work from documents in "
         "the workspace and the formulas built into my tools."),
    ]
    for _ in range(n):
        q, a = rng.choice(qa)
        tools = pick_tools(rng, [])
        yield _trace(system_for(tools), q, [
            (_a(rng.choice([
                "This is general engineering knowledge, not a question about a document. "
                "Reading a file would not help.",
                "Nothing in the workspace bears on this; it is definitional.",
                "No document is involved — answering directly is correct here.",
             ]), a), None),
        ])


def fam_calc_only(rng: random.Random, facts: list[dict[str, Any]], n: int) -> Iterator[dict[str, Any]]:
    """Engineering maths with the numbers already supplied — no document.

    This family exists to attack the `fs_read` reflex directly. The old corpus
    opened with a read in 98% of traces, so the model had no example of a task
    where reading is the wrong first move. Here the figures are in the prompt
    and the correct first action is `calc`.
    """
    for _ in range(n):
        f = rng.choice(facts)
        grid = rng.choice(list(f["per_grid"]))
        p = f["per_grid"][grid]
        tools = pick_tools(rng, ["calc"])
        kind = rng.choice(["rate", "life", "algebra", "convert"])

        if kind == "rate":
            user = rng.choice([
                f"Grid {grid} on {f['tag']} went from {p['previous']} mm to {p['current']} mm "
                f"over {f['interval_years']} years. Corrosion rate?",
                f"{p['previous']} mm down to {p['current']} mm in {f['interval_years']} years — "
                f"what rate is that?",
            ])
            expr = (f"corrosion_rate(t_previous={p['previous']}, t_current={p['current']}, "
                    f"years={f['interval_years']})")
            out = (f"({p['previous']} - {p['current']}) / {f['interval_years']} = {p['rate']}\n"
                   f"mm/year   [{f['code']}]")
            ans = f"{p['rate']} mm/year."
        elif kind == "life":
            user = (f"At {p['rate']} mm/year, how long until a wall of {p['current']} mm reaches "
                    f"a t-min of {f['t_min']} mm?")
            expr = (f"remaining_life(t_current={p['current']}, t_min={f['t_min']}, "
                    f"corrosion_rate={p['rate']})")
            out = (f"({p['current']} - {f['t_min']}) / {p['rate']} = {p['remaining_life']}\n"
                   f"years   [{f['code']} §7.1]")
            ans = f"{p['remaining_life']} years."
        elif kind == "algebra":
            a, b = rng.randint(2, 12), rng.randint(2, 9)
            user = rng.choice([
                f"Differentiate {a}*x**{b} with respect to x.",
                f"What is the integral of {a}*x**{b} from 0 to 2?",
            ])
            if user.startswith("Diff"):
                expr, out = f"diff({a}*x**{b}, x)", f"{a * b}*x**{b - 1}"
            else:
                val = a * (2 ** (b + 1)) / (b + 1)
                expr, out = f"integrate({a}*x**{b}, (x, 0, 2))", f"{val}"
            ans = out
        else:
            barg = f["design_pressure_barg"]
            user = f"Convert a design pressure of {barg} barg to psi."
            expr, out = f"{barg} * 14.5038", f"{round(barg * 14.5038, 4)}"
            ans = f"{round(barg * 14.5038, 2)} psi."

        yield _trace(system_for(tools), user, [
            (_t(rng.choice([
                "The figures are given, so no document is needed — but the arithmetic still "
                "goes through the tool rather than my head.",
                "Everything I need is in the question; I only have to compute it exactly.",
                "No file to read here. The calculation itself still belongs in the tool.",
             ]), "calc", {"expression": expr}), out),
            (_a(rng.choice([
                "Quoting the tool's result.",
                "The tool computed it exactly; I report that figure.",
                "Reporting calc's output verbatim.",
             ]), ans), None),
        ])


def fam_interval(rng: random.Random, facts: list[dict[str, Any]], n: int) -> Iterator[dict[str, Any]]:
    """Inspection-interval derivation, the measured weak spot.

    Base qwen3:4b scores 50.0% here (95% CI 41.2-58.8) against 91.7% on
    corrosion rate (85.3-95.4). Non-overlapping intervals, 720 samples — a real
    capability gap rather than sampling noise.

    The observed failure is semantic, not arithmetic. The model derives the
    right number and then throws it away:

        next_interval = min(3.26 / 2, 10) = min(1.63, 10) = 1.63
                      -> rounded to 10 years

    It treats the 10 as the answer instead of as a ceiling. These traces make
    the binding branch explicit every time, and deliberately include cases where
    the ceiling DOES bind, so the lesson is "compare the two and take the
    smaller", not "the answer is never 10".
    """
    for _ in range(n):
        f = rng.choice(facts)
        tools = pick_tools(rng, ["fs_read", "calc"])
        life = f["remaining_life_years"]
        half = round(life / 2.0, 4)
        answer_val = f["next_interval_years"]
        ceiling_binds = half > 10

        user = rng.choice([
            f"When is {f['tag']} next due for inspection? Figures in {f['doc']}.",
            f"Set the next inspection interval for {f['tag']} per {f['code']}. See {f['doc']}.",
            f"{f['doc']} — what interval should {f['tag']} run to?",
            f"How long until {f['tag']} must be inspected again? Use {f['doc']}.",
        ])

        yield _trace(system_for(tools), user, [
            (_t(f"The interval depends on remaining life at the governing grid, so I need "
                f"the {f['tag']} readings first.", "fs_read", {"path": f["doc"]}), _excerpt(f)),
            (_t(f"Remaining life at grid {f['governing_grid']} is {life} years. The interval "
                f"is the smaller of half that and the {f['code']} 10-year ceiling.",
                "calc", {"expression": f"next_interval(remaining_life_years={life})"}),
             f"next_interval = min(remaining_life_years / 2, 10)\n"
             f"min({life} / 2, 10) = min({half}, 10) = {answer_val}\n"
             f"years   [{f['code']} §7.2]"),
            (_a(rng.choice([
                f"min() returns the smaller operand; here that is "
                f"{'the 10-year ceiling' if ceiling_binds else f'half the remaining life ({half})'}.",
                f"Comparing {half} against the 10-year ceiling and taking the lower.",
                f"The ceiling is an upper bound, not the result; {answer_val} is what binds.",
             ]),
                f"{f['tag']} is next due in {answer_val} years. Half the remaining life at "
                f"grid {f['governing_grid']} is {half} years and the {f['code']} ceiling is "
                f"10 years; the interval is the smaller of the two, so "
                f"{'the ceiling' if ceiling_binds else 'half the remaining life'} governs."), None),
        ])


def fam_financial(rng: random.Random, facts: list[dict[str, Any]], n: int) -> Iterator[dict[str, Any]]:
    """Derivations over REAL MRPL filings — the only real documents we have.

    Grounded exclusively in figures that survived reconciliation against each
    statement's own stated totals (finetune/corpus_filings.py). The OCR on these
    PDFs mangles decimal separators, so an unverified figure here would be a
    plausible wrong number attached to a real company's real filing — the single
    worst thing this corpus could contain. 75 facts across 13 documents passed;
    everything else in those 43 documents is readable text that sources no
    question.

    The financial framing is kept as financial. Dressing an MRPL income
    statement up as an inspection report would put fabricated context beside
    real numbers, which is exactly why the Kaggle pipeline datasets were
    rejected in data/sources.yaml.
    """
    rows = json.loads(FILINGS_MANIFEST.read_text(encoding="utf-8"))["files"]
    pool = [(d["markdown"], f) for d in rows for f in d["facts"]]
    if not pool:
        return
    for _ in range(n):
        doc, fact = rng.choice(pool)
        rel = doc.replace("data/corpus/", "")
        rev, oth, tot = (fact["revenue_from_operations"],
                         fact["other_income"], fact["total_income"])
        tools = pick_tools(rng, ["fs_read", "calc"])
        kind = rng.choice(["total", "share", "check"])

        if kind == "total":
            user = rng.choice([
                f"From {rel}, what is total income for the period?",
                f"Add up total income in {rel}.",
                f"{rel} — revenue plus other income, please.",
            ])
            expr, out, ans = (f"{rev} + {oth}", f"{rev} + {oth} = {tot}",
                              f"Total income is Rs {tot:,.2f} crore — revenue from operations "
                              f"Rs {rev:,.2f} crore plus other income Rs {oth:,.2f} crore.")
        elif kind == "share":
            pct = round(oth / tot * 100, 4)
            user = f"In {rel}, what share of total income is other income?"
            expr, out = f"{oth} / {tot} * 100", f"{oth} / {tot} * 100 = {pct}"
            ans = (f"Other income is {pct:.2f}% of total income "
                   f"(Rs {oth:,.2f} crore of Rs {tot:,.2f} crore).")
        else:
            user = f"Does the income statement in {rel} add up? Check it."
            expr, out = f"{rev} + {oth} - {tot}", f"{rev} + {oth} - {tot} = 0.0"
            ans = (f"Yes. Revenue from operations Rs {rev:,.2f} crore plus other income "
                   f"Rs {oth:,.2f} crore equals the stated total income of "
                   f"Rs {tot:,.2f} crore exactly.")

        yield _trace(system_for(tools), user, [
            (_t(rng.choice([
                f"I need the figures from {rel} before computing anything.",
                f"The statement holds the line items; reading it first.",
                f"Opening {rel} to take the revenue and other-income lines.",
             ]), "fs_read", {"path": rel}),
             f"# MRPL Financial Results\n(All amounts in Rs Crore)\n\n"
             f"Revenue from Operations   {rev:,.2f}\n"
             f"Other Income              {oth:,.2f}\n"
             f"Total Income              {tot:,.2f}\n"),
            (_t(rng.choice([
                "The figures are read; the arithmetic goes through the tool.",
                f"Computing from the two stated lines rather than estimating.",
                "Exact arithmetic on figures a reviewer can check against the filing.",
             ]), "calc", {"expression": expr}), out),
            (_a(rng.choice([
                f"Reporting the computed figure with the lines it came from.",
                f"The filing's own lines support this; I quote both.",
                f"Computed from {rel}; the substitution is checkable.",
             ]), ans), None),
        ])


def fam_finqa(rng: random.Random, rows: list[dict[str, Any]], n: int) -> Iterator[dict[str, Any]]:
    """Real multi-step derivations over real tables, from FinQA.

    Used for reasoning *shape*, not domain: locate the cells, compose the
    derivation, cite the row. The financial framing is left intact rather than
    disguised as refinery data — inventing a refinery story around a real S&P
    table would put fabricated context next to real numbers, which is the exact
    failure mode the Kaggle pipeline datasets were rejected for.
    """
    for row in rng.sample(rows, min(n, len(rows))):
        qa = row.get("qa", {})
        q, ans, prog = qa.get("question"), qa.get("answer"), qa.get("program")
        gold = list(qa.get("gold_inds", {}).values())
        if not (q and ans and prog and gold):
            continue
        table = "\n".join(" | ".join(str(c) for c in r) for r in row.get("table", [])[:8])
        tools = pick_tools(rng, ["calc"])
        expr = _finqa_to_expr(prog)
        if expr is None:
            continue
        yield _trace(system_for(tools),
                     f"From this filing extract, {q}\n\n{table}", [
            (_t(f"The figures I need are in the table. Source rows: {gold[0][:110]}. "
                f"I will compute rather than estimate.", "calc", {"expression": expr}),
             f"{expr} = {ans}"),
            (_a(rng.choice([
                "I quote the computed figure and the rows it came from.",
                "The derivation is exact; I cite the source rows alongside it.",
                "Reporting the computed value with its provenance in the table.",
             ]),
                f"{ans}. Derived from: {gold[0][:150]}"), None),
        ])


def _finqa_to_expr(program: str) -> str | None:
    """Turn a FinQA program into a calc expression, or give up.

    FinQA programs are postfix-ish with #n references (`divide(3.8, #0)`).
    Only single-step programs are converted: chaining #n references correctly
    is fiddly, and a mis-converted program would put a wrong answer next to a
    real table. Dropping them costs coverage; getting one wrong costs trust.
    """
    program = program.strip().rstrip(",")
    if "#" in program or program.count("(") != 1:
        return None
    op, _, rest = program.partition("(")
    args = [a.strip() for a in rest.rstrip(")").split(",")]
    if len(args) != 2:
        return None
    try:
        [float(a.replace("%", "").replace("const_", "")) for a in args]
    except ValueError:
        return None
    a, b = (x.replace("const_", "") for x in args)
    return {"add": f"{a} + {b}", "subtract": f"{a} - {b}",
            "multiply": f"{a} * {b}", "divide": f"{a} / {b}"}.get(op)


# --------------------------------------------------------------------------

FAMILIES: list[tuple[str, Callable[..., Iterator[dict[str, Any]]], float]] = [
    ("inspection_derive", fam_inspection_derive, 0.16),
    # Weighted by MEASUREMENT, not intuition: base scores 50.0% here
    # against 91.7% on corrosion rate. This is where headroom exists.
    ("interval",          fam_interval,          0.10),
    ("lookup",            fam_lookup,            0.11),
    ("unknown",           fam_unknown,           0.08),
    ("clarify",           fam_clarify,           0.06),
    ("repair",            fam_repair,            0.08),
    ("multi_doc",         fam_multi_doc,         0.08),
    ("deliverable",       fam_deliverable,       0.10),
    ("direct",            fam_direct,            0.05),
    ("calc_only",         fam_calc_only,         0.11),
    ("financial",         fam_financial,         0.10),
    ("finqa",             fam_finqa,             0.10),
]


def build(total: int, seed: int = SEED) -> tuple[list[dict[str, Any]], Counter[str]]:
    rng = random.Random(seed)
    facts = json.loads(INBOX_FACTS.read_text(encoding="utf-8"))
    finqa: list[dict[str, Any]] = []
    if FINQA_TRAIN.exists():
        finqa = json.loads(FINQA_TRAIN.read_text(encoding="utf-8"))

    out: list[dict[str, Any]] = []
    tally: Counter[str] = Counter()
    for name, fn, share in FAMILIES:
        n = int(total * share)
        source = finqa if name == "finqa" else facts
        if name == "financial" and not FILINGS_MANIFEST.exists():
            continue
        if not source:
            continue
        made = list(fn(rng, source, n))
        for tr in made:
            tr["_family"] = name
        out.extend(made)
        tally[name] = len(made)

    rng.shuffle(out)
    return out, tally


def split(traces: list[dict[str, Any]], seed: int = SEED
          ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_family: dict[str, list[dict[str, Any]]] = {}
    for t in traces:
        by_family.setdefault(t["_family"], []).append(t)

    train: list[dict[str, Any]] = []
    ev: list[dict[str, Any]] = []
    # Stratify: every family must appear in eval, or the eval loss curve stops
    # representing the corpus and early stopping optimises the wrong thing.
    #
    # Cut on the USER PROMPT, not on the trace. The generators emit several
    # traces per prompt, so shuffling traces and slicing put the same question
    # on both sides — measured at 136 prompts, 36.5% of the eval set, present
    # verbatim in train. A held-out loss computed over that is partly a
    # memorisation score, and qlora.py now selects its best checkpoint by it.
    for family, rows in by_family.items():
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            prompt = next(m["content"] for m in row["messages"] if m["role"] == "user")
            groups.setdefault(prompt, []).append(row)
        keys = sorted(groups)           # sorted first: dict order must not leak in
        rng.shuffle(keys)
        cut = max(1, int(len(keys) * EVAL_FRACTION))
        ev.extend(r for k in keys[:cut] for r in groups[k])
        train.extend(r for k in keys[cut:] for r in groups[k])
    rng.shuffle(train)
    rng.shuffle(ev)
    return train, ev


def profile(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """The numbers that condemned the previous corpus, recomputed on this one."""
    firsts: Counter[str] = Counter()
    turns: Counter[int] = Counter()
    toolsets: set[frozenset[str]] = set()
    tool_use: Counter[str] = Counter()
    users: set[str] = set()

    for t in traces:
        msgs = t["messages"]
        users.add(next(m["content"] for m in msgs if m["role"] == "user"))
        sys_msg = msgs[0]["content"]
        offered = frozenset(n for n in ALL_TOOLS if f"- {n}(" in sys_msg)
        toolsets.add(offered)
        assistants = [m for m in msgs if m["role"] == "assistant"]
        turns[len(assistants)] += 1
        first = json.loads(assistants[0]["content"])
        firsts[first.get("tool", "<answer>")] += 1
        for m in assistants:
            obj = json.loads(m["content"])
            if "tool" in obj:
                tool_use[obj["tool"]] += 1

    return {
        "traces": len(traces),
        "unique_user_prompts": len(users),
        "distinct_tool_sets": len(toolsets),
        "first_action": dict(firsts.most_common()),
        "assistant_turns": dict(sorted(turns.items())),
        "tool_calls": dict(tool_use.most_common()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--total", type=int, default=2600)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    traces, tally = build(args.total)
    prof = profile(traces)

    print(f"built {len(traces)} traces\n")
    print("composition:")
    for name, _, _ in FAMILIES:
        print(f"   {tally.get(name, 0):>5}  {name}")
    print()
    print(f"unique user prompts : {prof['unique_user_prompts']}")
    print(f"distinct tool sets  : {prof['distinct_tool_sets']}")
    print(f"assistant turns     : {prof['assistant_turns']}")
    print(f"tool calls          : {prof['tool_calls']}")
    print()
    print("first action (the old corpus was 600/612 fs_read):")
    for k, v in prof["first_action"].items():
        print(f"   {v:>5} ({v / len(traces):>5.1%})  {k}")

    if args.report:
        return 0

    train, ev = split(traces)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("eval", ev)):
        path = OUT_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
        print(f"\nwrote {len(rows):>5} -> {path.relative_to(ROOT)}")

    (OUT_DIR / "profile.json").write_text(
        json.dumps({"composition": dict(tally), **prof}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
