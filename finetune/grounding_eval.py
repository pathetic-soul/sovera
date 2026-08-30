"""Measure the §16 grounding gap directly, on the real corpus table.

finetune/evaluate.py measures whether the model drives the loop correctly, and
`driver` already scores at ceiling there. This measures the thing that actually
failed in leg 5: reading the correct cell out of a 12-row markdown table and
getting the arithmetic right afterwards.

Two question kinds, deliberately separated, because they fail for different
reasons and only one of them is plausibly fixable by fine-tuning:

  lookup   the answer is a single cell, printed verbatim in the document.
           Failure here is a retrieval/attention problem.
  derived  the answer needs two cells and one formula. This is where leg 5
           broke: `coder` took grid S5's 11.7 instead of S7's 10.6, then
           inverted (t_meas - t_min) into (t_min - t_meas) and reported a
           negative remaining life as fact.

Ground truth is taken from data/corpus/inbox/UT-2024-114-V-2301.md by hand and
asserted against the file at startup, so the two cannot silently drift apart.

    python -m finetune.grounding_eval qwen3:4b-instruct
    python -m finetune.grounding_eval qwen2.5-coder:7b-instruct-q4_K_M
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backends.base import BackendError, Message
from backends.ollama_backend import OllamaBackend
from core.agent import _extract_json
from core.audit import AuditLog

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data" / "corpus" / "inbox" / "UT-2024-114-V-2301.md"
MAX_CTX = 8192

# The table, as the document states it: grid -> (2019, 2024, loss)
GRID = {
    "S1": (11.8, 11.5, 0.3), "S2": (11.9, 11.6, 0.3), "S3": (11.7, 11.4, 0.3),
    "S4": (11.6, 11.0, 0.6), "S5": (11.7, 11.1, 0.6), "S6": (10.9, 9.4, 1.5),
    "S7": (10.6, 8.9, 1.7), "S8": (10.8, 9.2, 1.6), "H1": (13.7, 13.5, 0.2),
    "H2": (13.8, 13.6, 0.2), "N1": (9.9, 9.5, 0.4), "N2": (9.8, 9.4, 0.4),
}
INTERVAL = 5.70
TMIN = 7.4


def rate(grid: str) -> float:
    y2019, y2024, _ = GRID[grid]
    return (y2019 - y2024) / INTERVAL


def remaining(grid: str) -> float:
    return (GRID[grid][1] - TMIN) / rate(grid)


# (question, expected value, kind)
QUESTIONS: list[tuple[str, float, str]] = [
    ("What is the 2024-11-18 thickness reading at grid S4, in mm?", 11.0, "lookup"),
    ("What is the 2019-03-06 thickness reading at grid S7, in mm?", 10.6, "lookup"),
    ("What is the thickness loss at grid S6, in mm?", 1.5, "lookup"),
    ("What is the 2024-11-18 reading at grid H2, in mm?", 13.6, "lookup"),
    ("What is the 2019-03-06 reading at grid N1, in mm?", 9.9, "lookup"),
    ("What is the retirement limit (t-min) for this vessel, in mm?", 7.4, "lookup"),
    ("How many years elapsed between the two surveys?", 5.70, "lookup"),
    ("What is the lowest 2024 thickness reading anywhere on the vessel, in mm?", 8.9, "lookup"),
    ("What is the design pressure, in barg?", 10.5, "lookup"),
    ("What is the nominal shell thickness, in mm?", 12.0, "lookup"),
    ("What is the corrosion rate at grid S7, in mm per year?", rate("S7"), "derived"),
    ("What is the remaining life at grid S7, in years, before it reaches t-min?",
     remaining("S7"), "derived"),
    ("How much margin above t-min remains at grid S7, in mm?", 8.9 - TMIN, "derived"),
    ("What is the corrosion rate at grid S4, in mm per year?", rate("S4"), "derived"),
    ("What is the remaining life at grid S4, in years?", remaining("S4"), "derived"),
    ("What is the mean of the 2024 readings at grids S6, S7 and S8, in mm?",
     (9.4 + 8.9 + 9.2) / 3, "derived"),
    ("What is the total thickness loss across grids S6, S7 and S8 combined, in mm?",
     1.5 + 1.7 + 1.6, "derived"),
    ("What is the mean thickness loss across grids S6, S7 and S8, in mm?",
     (1.5 + 1.7 + 1.6) / 3, "derived"),
]

# §12 in practice. The first draft of this prompt said `<number>` and "a plain
# number with no units", and the 4B read that as "integer": it answered 10 for
# "10.5 barg" and 5 for a sum its own source field spelled out as 4.8. It also
# said "not stated" for every derived question, because the draft told it to
# answer only what the document states — and a corrosion rate is computed, not
# printed. Both were faults in the instrument, not the model. Hence the
# explicit decimal examples and the explicit licence to compute.
PROMPT = """You are reading an inspection report. Use ONLY the numbers in this document.

Reply with one JSON object:
{"value": <decimal number>, "source": "<the row or line you used>"}

Rules for `value`:
- Give a DECIMAL number, for example 8.9 or 0.298 or 11.0.
- Keep at least one decimal place. Never round to a whole number.
- No units, no text, no ranges.
- You MAY calculate it from numbers in the document (subtract, divide, average).
- Only if the numbers needed are genuinely absent, reply
  {"value": null, "source": "not stated"}.

DOCUMENT:
{doc}

QUESTION: {q}

Reply with the JSON object only."""


def close(got: float, want: float, kind: str = "derived") -> bool:
    """Tolerance depends on what is being asked, and this distinction is load-bearing.

    A `lookup` answer is a cell printed verbatim in the document, so it must
    match exactly to the stated precision. An earlier version of this function
    allowed 2% everywhere, which was wrong and would have flattered the score:
    adjacent grids in this table are ~0.9% apart (S1 11.5, S2 11.6, S4 11.0,
    S5 11.1), so a confident answer from the *wrong row* scored as correct.
    tests/test_finetune.py pins this — it asserts no two grids are mutually
    acceptable under the lookup tolerance.

    A `derived` answer is computed, so honest rounding (0.298 for 0.29825) has
    to pass while a real arithmetic error (9.5 for 9.167, 3.6% out) must not.
    """
    if kind == "lookup":
        return abs(got - want) <= 0.005
    return abs(got - want) <= max(0.01, abs(want) * 0.02)


def verify_ground_truth() -> None:
    text = REPORT.read_text(encoding="utf-8")
    for grid, (y19, y24, loss) in GRID.items():
        row = f"| {grid} |"
        line = next((ln for ln in text.splitlines() if ln.startswith(row)), None)
        assert line is not None, f"grid {grid} missing from {REPORT.name}"
        for value in (y19, y24, loss):
            assert f"{value}" in line, f"{grid}: {value} not in {line!r}"
    # Literal strings, not f-formatted floats: f"{5.70}" renders as "5.7" and
    # would not match the document's "5.70 years".
    assert "7.4 mm" in text, "t-min not stated as expected"
    assert "5.70 years" in text, "survey interval not stated as expected"
    assert "10.5 barg" in text and "12.0 mm nominal shell" in text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ref")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    verify_ground_truth()
    doc = REPORT.read_text(encoding="utf-8")
    audit = AuditLog(ROOT / "workspace" / ".audit" / "eval.jsonl", "grounding")
    backend = OllamaBackend(audit)
    ok, note = backend.available()
    if not ok:
        print(f"ollama unavailable: {note}")
        return 1

    tally: dict[str, list[int]] = {"lookup": [0, 0], "derived": [0, 0]}
    misses: list[str] = []

    for question, expected, kind in QUESTIONS:
        prompt = PROMPT.replace("{doc}", doc).replace("{q}", question)
        try:
            completion = backend.chat(
                args.ref, [Message(role="user", content=prompt)],
                max_ctx=MAX_CTX, temperature=0.1, json_mode=True,
            )
        except BackendError as exc:
            print(f"backend error: {exc}")
            return 1
        parsed = _extract_json(completion.text) or {}
        raw = parsed.get("value")
        tally[kind][1] += 1
        got: float | None
        try:
            got = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            got = None

        if got is not None and close(got, expected, kind):
            tally[kind][0] += 1
        else:
            misses.append(
                f"  [{kind}] {question}\n"
                f"      expected {expected:.4g}, got {raw!r}"
                f"  src={str(parsed.get('source'))[:70]!r}"
            )

    label = args.label or args.ref
    print()
    print(f"=== grounding: {label} ===")
    for kind in ("lookup", "derived"):
        hit, total = tally[kind]
        pct = 100.0 * hit / total if total else 0.0
        print(f"  {kind:<8} {hit}/{total}  {pct:5.1f}%")
    hit = sum(v[0] for v in tally.values())
    total = sum(v[1] for v in tally.values())
    print(f"  {'TOTAL':<8} {hit}/{total}  {100.0*hit/total:5.1f}%")
    if misses:
        print("\n  misses:")
        print("\n".join(misses))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
