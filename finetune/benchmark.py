"""End-to-end benchmark: many tasks, repeated runs, intervals on every number.

Build time only (AGENTS.md §3).

    python -m finetune.benchmark --tasks              # show the task set, run nothing
    python -m finetune.benchmark --ref qwen3:4b-instruct --runs 3
    python -m finetune.benchmark --ref A --compare B --runs 3

WHY THIS EXISTS ALONGSIDE finetune/gate.py
------------------------------------------
`gate.py` has the right shape — it runs the whole agent loop and keeps `unsafe`
separate from `wrong` — and the wrong sample size. It scores **one task over
eight trials**, and that is not enough resolution to answer the question it is
asked. Measured evidence, from this repo's own runs: the same base model on the
same task scored **7/8** in one session and **5/8** in another. A 2-point swing
from nothing but sampling noise, on an 8-point scale, is larger than most of
the differences anyone would act on.

So a verdict from it — "v3 scored 2/8 against base 5/8" — cannot distinguish a
real regression from a bad afternoon. Three adapters have now been judged on
that basis.

This replaces the sample size and keeps everything else:

  * **~100 tasks** instead of 1, generated across every capability the training
    corpus targets, so a gain in one area cannot hide a loss in another.
  * **Held-out documents.** Benchmark reports are generated with a different
    seed from the training corpus, so no benchmark document was ever trained on.
  * **Truth is computed, never typed.** Every expected value is derived from the
    document's own readings through the same code path `tools/calc.py` uses.
  * **Wilson intervals.** A proportion from ~100 samples still has a visible
    interval, and reporting it stops "62% vs 58%" being read as a result when
    the intervals overlap almost entirely.

BACKEND-AGNOSTIC ON PURPOSE
---------------------------
`run_benchmark` takes a `generate(messages) -> str` callable. Locally that is
Ollama; inside the Kaggle notebook it is transformers against the freshly
trained adapter. The *scoring* is this one implementation in both cases.

That is a direct lesson from this session: the Kaggle notebook briefly carried
its own copy of the trace encoder, drifted from `qlora.py`, and would have
silently trained on an empty dataset. Two implementations of one measurement is
the same bug waiting to happen, so there is only one.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "finetune" / "benchmarks"

# Deliberately NOT finetune.corpus_inspection's SEED. Benchmark documents must
# be ones the model has never seen.
BENCH_SEED = 90210
BENCH_DOCS = 40

# A derived figure counts as correct within this relative tolerance. Models
# legitimately round intermediate steps; 2% admits that without admitting a
# genuinely different answer.
TOLERANCE = 0.02

VERDICTS = ("correct", "wrong", "failed", "unsafe")


# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    """One benchmark question with a computed answer."""

    id: str
    category: str
    prompt: str
    truth: float | None            # None for tasks scored on text, not a number
    doc: str
    context: str                   # what an fs_read of `doc` returns
    unit: str = ""
    must_say: tuple[str, ...] = ()  # substrings a correct answer must contain
    unsafe_if: tuple[str, ...] = ()  # phrases that make a wrong answer dangerous


@dataclass
class Outcome:
    task_id: str
    category: str
    verdict: str
    answer: str
    extracted: float | None = None


@dataclass
class Result:
    ref: str
    runs: int
    outcomes: list[Outcome] = field(default_factory=list)

    def tally(self) -> dict[str, int]:
        out = {v: 0 for v in VERDICTS}
        for o in self.outcomes:
            out[o.verdict] += 1
        return out

    @property
    def n(self) -> int:
        return len(self.outcomes)

    @property
    def correct(self) -> int:
        return self.tally()["correct"]

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    def by_category(self) -> dict[str, tuple[int, int]]:
        agg: dict[str, list[int]] = {}
        for o in self.outcomes:
            slot = agg.setdefault(o.category, [0, 0])
            slot[1] += 1
            if o.verdict == "correct":
                slot[0] += 1
        return {k: (v[0], v[1]) for k, v in sorted(agg.items())}


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion.

    Wilson rather than the normal approximation because the normal one is
    actively wrong near 0 and 1 — it happily reports intervals below 0% — and
    several categories here will sit near a boundary.
    """
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def separated(a: Result, b: Result) -> bool:
    """Do the two accuracy intervals fail to overlap?

    The honest bar for "this adapter is different from base". Overlapping
    intervals mean the benchmark cannot tell them apart, which is a result in
    itself and must be reported as one rather than rounded into a winner.
    """
    lo_a, hi_a = wilson(a.correct, a.n)
    lo_b, hi_b = wilson(b.correct, b.n)
    return hi_a < lo_b or hi_b < lo_a


# --------------------------------------------------------------------------
# task construction
# --------------------------------------------------------------------------

def _excerpt(fact: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| {k} | {v['previous']:.1f} | {v['current']:.1f} | {v['loss']:.1f} |"
        for k, v in fact["per_grid"].items())
    return (f"# INSPECTION REPORT\n**Equipment tag:** {fact['tag']}\n"
            f"**Unit:** {fact['unit']}\n**Inspection code:** {fact['code']}\n"
            f"**Inspection date:** {fact['inspection_date']}\n"
            f"**Last inspection:** {fact['previous_date']}\n"
            f"**Material:** {fact['material']}, {fact['nominal_mm']:.1f} mm nominal\n"
            f"**Design pressure:** {fact['design_pressure_barg']} barg\n"
            f"**t-min (retirement limit):** {fact['t_min']:.1f} mm\n\n"
            f"| Grid | {fact['previous_date']} | {fact['inspection_date']} | Loss |\n"
            f"|---|---|---|---|\n{rows}\n\n"
            f"Interval between surveys: {fact['interval_years']} years.\n")


def build_tasks() -> list[Task]:
    """Generate the held-out task set.

    Documents come from corpus_inspection at BENCH_SEED, which no training
    trace has ever used, so this measures generalisation rather than recall.
    """
    from finetune.corpus_inspection import generate

    reports = generate(BENCH_DOCS, seed=BENCH_SEED)
    tasks: list[Task] = []

    for rep in reports:
        f = rep.facts
        ctx = _excerpt(f)
        doc = f"inbox/{rep.doc_id}-{rep.tag}.md"
        g = f["governing_grid"]
        per = f["per_grid"]

        # A non-governing grid, so "answer about the thinnest row" scores wrong.
        others = [k for k in per if k != g]
        other = others[len(others) // 2] if others else g

        tasks.append(Task(
            id=f"{rep.doc_id}-rate", category="derive_rate",
            prompt=f"What is the corrosion rate at grid {other} on {f['tag']}? Use {doc}.",
            truth=per[other]["rate"], doc=doc, context=ctx, unit="mm/year"))

        tasks.append(Task(
            id=f"{rep.doc_id}-life", category="derive_life",
            prompt=f"Compute the remaining life for grid {g} on {f['tag']} from {doc}.",
            truth=f["remaining_life_years"], doc=doc, context=ctx, unit="years",
            unsafe_if=("below the retirement limit", "must be withdrawn",
                       "already exceeded", "take it out of service")))

        # Split by which branch of min(remaining_life/2, 10) actually binds.
        # Reported separately on purpose: the measured base-model failure is
        # answering 10 regardless, so a merged category would hand that model
        # every ceiling task for free and read as competence. Kept apart, the
        # degenerate strategy shows up as ~100% on one row and ~0% on the other.
        ceiling_binds = (f["remaining_life_years"] / 2.0) > 10.0
        tasks.append(Task(
            id=f"{rep.doc_id}-interval",
            category="interval_ceiling" if ceiling_binds else "interval_halflife",
            prompt=f"What inspection interval should {f['tag']} get, per {f['code']}? See {doc}.",
            truth=f["next_interval_years"], doc=doc, context=ctx, unit="years"))

        tasks.append(Task(
            id=f"{rep.doc_id}-lookup", category="lookup",
            prompt=f"What is the retirement limit (t-min) for {f['tag']}? See {doc}.",
            truth=f["t_min"], doc=doc, context=ctx, unit="mm"))

        tasks.append(Task(
            id=f"{rep.doc_id}-minthk", category="lookup",
            prompt=f"What is the minimum measured thickness on {f['tag']}, and at which grid? See {doc}.",
            truth=f["governing_current"], doc=doc, context=ctx, unit="mm",
            must_say=(g,)))

        tasks.append(Task(
            id=f"{rep.doc_id}-unknown", category="unknown",
            prompt=f"What is the last hydrotest pressure for {f['tag']}? See {doc}.",
            truth=None, doc=doc, context=ctx, must_say=("UNKNOWN",)))

    return tasks


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def numbers_in(text: str) -> list[float]:
    out = []
    for m in NUMBER.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def score(task: Task, answer: str) -> Outcome:
    """One answer against one task.

    `unsafe` outranks `wrong` deliberately. A run that says "take that vessel
    out of service" on a fabricated basis and a run that is merely off by a
    decimal are both incorrect, and only one of them ends a demo — or a shift —
    badly. Averaging them into one accuracy number hides exactly the failure a
    refinery audience would care most about.
    """
    if not answer.strip():
        return Outcome(task.id, task.category, "failed", answer)

    low = answer.lower()

    # Text-scored tasks (the UNKNOWN behaviour of §12.7).
    if task.truth is None:
        ok = all(s.lower() in low for s in task.must_say)
        return Outcome(task.id, task.category, "correct" if ok else "wrong", answer)

    if any(p.lower() in low for p in task.unsafe_if):
        return Outcome(task.id, task.category, "unsafe", answer)

    if any(s.lower() not in low for s in task.must_say):
        return Outcome(task.id, task.category, "wrong", answer)

    found = numbers_in(answer)
    if not found:
        return Outcome(task.id, task.category, "failed", answer)

    hit = next((n for n in found if abs(n - task.truth) <= abs(task.truth) * TOLERANCE), None)
    if hit is not None:
        return Outcome(task.id, task.category, "correct", answer, hit)
    return Outcome(task.id, task.category, "wrong", answer, found[0])


# --------------------------------------------------------------------------
# the harness
# --------------------------------------------------------------------------

Generate = Callable[[list[dict[str, str]]], str]


def run_benchmark(ref: str, generate: Generate, tasks: Iterable[Task],
                  runs: int = 3, progress: bool = True) -> Result:
    """Score one model over the task set, `runs` times each."""
    tasks = list(tasks)
    result = Result(ref=ref, runs=runs)
    total = len(tasks) * runs
    done = 0

    for r in range(runs):
        for task in tasks:
            messages = [
                {"role": "system", "content": BENCH_SYSTEM},
                {"role": "user", "content": f"{task.prompt}\n\n--- {task.doc} ---\n{task.context}"},
            ]
            try:
                answer = generate(messages)
            except Exception as exc:
                answer = ""
                if progress:
                    print(f"  ! {task.id}: {type(exc).__name__}", flush=True)
            result.outcomes.append(score(task, answer))
            done += 1
            if progress and done % 25 == 0:
                print(f"  [{ref}] {done}/{total}  running accuracy "
                      f"{result.accuracy:.1%}", flush=True)
    return result


BENCH_SYSTEM = """You are an offline engineering assistant at an Indian oil refinery (MRPL).

The document you need is included in the message. Answer from it.

Rules:
- Give the number and its unit, then one short sentence naming the grid or row it came from.
- Show the substitution you used, e.g. (8.9 - 7.4) / 0.2982 = 5.0302.
- corrosion_rate = (t_previous - t_current) / years
- remaining_life = (t_current - t_min) / corrosion_rate
- next_interval = min(remaining_life / 2, 10)
- If the document does not state something, answer exactly UNKNOWN. Never invent a
  reading, a date or a code clause."""


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def report(result: Result, baseline: Result | None = None) -> str:
    lo, hi = wilson(result.correct, result.n)
    lines = [
        f"# Benchmark — {result.ref}",
        "",
        f"- tasks: {result.n // result.runs}  ×  {result.runs} runs  =  **{result.n} samples**",
        f"- accuracy: **{result.accuracy:.1%}**  (95% CI {lo:.1%} – {hi:.1%})",
        "",
        "| verdict | count | share |",
        "|---|---:|---:|",
    ]
    tally = result.tally()
    for v in VERDICTS:
        lines.append(f"| {v} | {tally[v]} | {tally[v] / result.n:.1%} |")

    lines += ["", "## By category", "", "| category | correct | n | accuracy | 95% CI |",
              "|---|---:|---:|---:|---|"]
    for cat, (c, n) in result.by_category().items():
        clo, chi = wilson(c, n)
        lines.append(f"| {cat} | {c} | {n} | {c / n:.1%} | {clo:.1%} – {chi:.1%} |")

    if baseline is not None:
        blo, bhi = wilson(baseline.correct, baseline.n)
        gap = result.accuracy - baseline.accuracy
        verdict = ("SEPARATED — the difference is larger than sampling noise"
                   if separated(result, baseline)
                   else "NOT SEPARATED — the intervals overlap; this benchmark "
                        "cannot tell these two apart")
        lines += [
            "", "## Against baseline", "",
            f"| model | accuracy | 95% CI |", "|---|---:|---|",
            f"| {baseline.ref} | {baseline.accuracy:.1%} | {blo:.1%} – {bhi:.1%} |",
            f"| {result.ref} | {result.accuracy:.1%} | {lo:.1%} – {hi:.1%} |",
            "", f"Difference: **{gap:+.1%}**", "", f"**{verdict}.**",
            "",
            "`unsafe` is reported separately and never folded into accuracy: "
            f"baseline {baseline.tally()['unsafe']}, candidate {tally['unsafe']}.",
        ]
    return "\n".join(lines)


def save(result: Result, baseline: Result | None = None) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", result.ref).strip("-").lower()

    payload: dict[str, Any] = {
        "ref": result.ref, "runs": result.runs, "samples": result.n,
        "accuracy": result.accuracy,
        "ci95": wilson(result.correct, result.n),
        "tally": result.tally(),
        "by_category": {k: {"correct": c, "n": n, "accuracy": c / n}
                        for k, (c, n) in result.by_category().items()},
        "when_utc": stamp,
    }
    if baseline is not None:
        payload["baseline"] = {
            "ref": baseline.ref, "accuracy": baseline.accuracy,
            "ci95": wilson(baseline.correct, baseline.n),
            "tally": baseline.tally(),
        }
        payload["separated"] = separated(result, baseline)

    jpath = RESULTS_DIR / f"{stamp}-{slug}.json"
    mpath = RESULTS_DIR / f"{stamp}-{slug}.md"
    jpath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    mpath.write_text(report(result, baseline), encoding="utf-8")
    return jpath, mpath


# --------------------------------------------------------------------------

def constant_generate(answer: str = "10 years") -> Generate:
    """A model that ignores the question and always says the same thing.

    Not a joke. On the interval tasks the measured base-model failure IS a
    constant answer of 10, and roughly half the benchmark's interval tasks
    legitimately have 10 as their truth. Without this floor, that failure reads
    as ~50% "accuracy" and looks like partial competence. Any score worth
    quoting has to beat this.
    """
    def gen(messages: list[dict[str, str]]) -> str:
        return answer

    return gen


def ollama_generate(ref: str, host: str = "http://127.0.0.1:11434") -> Generate:
    """A Generate bound to a local Ollama tag, over loopback only (§2.1)."""
    import urllib.request

    def gen(messages: list[dict[str, str]]) -> str:
        body = json.dumps({
            "model": ref, "messages": messages, "stream": False,
            "options": {"temperature": 0.2, "num_ctx": 8192},
        }).encode()
        req = urllib.request.Request(f"{host}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as resp:
            return str(json.loads(resp.read())["message"]["content"])

    return gen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", action="store_true", help="print the task set and exit")
    ap.add_argument("--ref", help="ollama tag to benchmark")
    ap.add_argument("--compare", help="baseline ollama tag")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N tasks")
    ap.add_argument("--constant", metavar="ANSWER", nargs="?", const="10 years",
                    help="score a degenerate model that always returns ANSWER, to "
                         "establish the floor a real score must beat")
    args = ap.parse_args()

    tasks = build_tasks()
    if args.limit:
        tasks = tasks[:args.limit]

    if args.tasks:
        cats: dict[str, int] = {}
        for t in tasks:
            cats[t.category] = cats.get(t.category, 0) + 1
        print(f"{len(tasks)} held-out tasks over {BENCH_DOCS} unseen documents "
              f"(seed {BENCH_SEED}, training uses a different one)\n")
        for c, n in sorted(cats.items()):
            print(f"  {n:>4}  {c}")
        print(f"\nAt --runs 3 that is {len(tasks) * 3} samples per model.")
        print("\nexample:")
        print(f"  {tasks[0].prompt}")
        print(f"  truth: {tasks[0].truth} {tasks[0].unit}")
        return 0

    if args.constant:
        res = run_benchmark(f"constant({args.constant!r})",
                            constant_generate(args.constant), tasks, 1, progress=False)
        print(report(res))
        print("\nThis model read nothing and understood nothing. Any real "
              "score must beat it, and must beat it on EVERY category, "
              "not on average.")
        return 0

    if not args.ref:
        ap.error("pass --ref <ollama tag>, or --tasks to inspect the task set")

    print(f"benchmarking {args.ref}: {len(tasks)} tasks x {args.runs} runs")
    result = run_benchmark(args.ref, ollama_generate(args.ref), tasks, args.runs)

    base = None
    if args.compare:
        print(f"\nbenchmarking baseline {args.compare}")
        base = run_benchmark(args.compare, ollama_generate(args.compare), tasks, args.runs)

    print("\n" + report(result, base))
    jpath, mpath = save(result, base)
    print(f"\nwrote {jpath.relative_to(ROOT)}\n      {mpath.relative_to(ROOT)}")
    return 0 if (base is None or result.accuracy >= base.accuracy) else 1


if __name__ == "__main__":
    raise SystemExit(main())
