"""Bring a Kaggle benchmark result back into the repo, with its claims checked.

Build time only (AGENTS.md §3).

    python -m finetune.import_benchmark <path-to-benchmark.json>

WHY THIS IS NOT `copy benchmark.json finetune/benchmarks/`
---------------------------------------------------------
A result file arriving from another machine is an assertion until something
verifies it. This recomputes every derived number from the raw tallies before
storing anything:

  * accuracy must equal correct / samples
  * the Wilson interval must match a local recomputation
  * `separated` must match a local recomputation from both tallies
  * the four verdict counts must sum to the sample count

If any of those disagree, the import is refused. That matters because
`separated` is the field the ship/no-ship decision reads, and it is exactly the
field a stale notebook cell or a hand-edited file would get wrong. This project
has already shipped three adapters on numbers nobody re-derived.

The imported file is written into `finetune/benchmarks/` alongside locally
produced runs, with `source` recording where it came from, so a reader can tell
a Kaggle run from a laptop run without guessing.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from finetune.benchmark import RESULTS_DIR, VERDICTS, wilson

TOLERANCE = 1e-6


def _close(a: float, b: float, tol: float = TOLERANCE) -> bool:
    return abs(a - b) <= tol


def verify(payload: dict[str, Any]) -> list[str]:
    """Recompute every derived claim. Returns the problems found."""
    problems: list[str] = []

    for side in ("base", "candidate"):
        block = payload.get(side)
        if not isinstance(block, dict):
            problems.append(f"missing '{side}' block")
            continue

        tally = block.get("tally", {})
        samples = block.get("samples")
        if not isinstance(samples, int) or samples <= 0:
            problems.append(f"{side}: samples is not a positive integer")
            continue

        counted = sum(int(tally.get(v, 0)) for v in VERDICTS)
        if counted != samples:
            problems.append(
                f"{side}: verdict counts sum to {counted} but samples is {samples}")

        correct = int(tally.get("correct", 0))
        stated = float(block.get("accuracy", -1))
        if not _close(stated, correct / samples):
            problems.append(
                f"{side}: accuracy {stated:.6f} != correct/samples "
                f"{correct}/{samples} = {correct / samples:.6f}")

        lo, hi = wilson(correct, samples)
        got = block.get("ci95") or [None, None]
        if len(got) != 2 or not _close(float(got[0]), lo, 1e-4) or not _close(float(got[1]), hi, 1e-4):
            problems.append(
                f"{side}: ci95 {got} != locally recomputed ({lo:.4f}, {hi:.4f})")

    if not problems:
        b, c = payload["base"], payload["candidate"]
        blo, bhi = wilson(int(b["tally"]["correct"]), int(b["samples"]))
        clo, chi = wilson(int(c["tally"]["correct"]), int(c["samples"]))
        local = chi < blo or bhi < clo
        if bool(payload.get("separated")) != local:
            problems.append(
                f"separated={payload.get('separated')} but local recomputation "
                f"says {local} (base {blo:.3f}-{bhi:.3f}, candidate {clo:.3f}-{chi:.3f})")
    return problems


def summarise(payload: dict[str, Any]) -> str:
    b, c = payload["base"], payload["candidate"]
    blo, bhi = wilson(int(b["tally"]["correct"]), int(b["samples"]))
    clo, chi = wilson(int(c["tally"]["correct"]), int(c["samples"]))
    gap = float(c["accuracy"]) - float(b["accuracy"])
    lines = [
        f"source     : {payload.get('source', 'unknown')}",
        f"samples    : {b['samples']} per model",
        "",
        f"  base      {b['ref'][:44]:<44} {b['accuracy']:>6.1%}  "
        f"[{blo:.1%} - {bhi:.1%}]",
        f"  candidate {c['ref'][:44]:<44} {c['accuracy']:>6.1%}  "
        f"[{clo:.1%} - {chi:.1%}]",
        "",
        f"  difference {gap:+.1%}",
        f"  unsafe     base {b['tally'].get('unsafe', 0)}, "
        f"candidate {c['tally'].get('unsafe', 0)}",
        "",
    ]
    if payload.get("separated"):
        lines.append("  SEPARATED — the difference exceeds sampling noise.")
        lines.append("  " + ("Candidate is better." if gap > 0
                             else "Candidate is WORSE. Do not ship it."))
    else:
        lines.append("  NOT SEPARATED — the intervals overlap. This benchmark cannot")
        lines.append("  tell the two models apart; the difference is not evidence.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="benchmark.json downloaded from the Kaggle notebook")
    ap.add_argument("--markdown", help="benchmark.md to import alongside it")
    ap.add_argument("--force", action="store_true",
                    help="import even if verification fails (records the failures)")
    args = ap.parse_args()

    src = Path(args.path)
    if not src.exists():
        print(f"no such file: {src}")
        return 2

    payload = json.loads(src.read_text(encoding="utf-8"))
    problems = verify(payload)

    if problems:
        print(f"VERIFICATION FAILED — {len(problems)} problem(s):")
        for p in problems:
            print(f"  ! {p}")
        if not args.force:
            print("\nRefusing to import. The 'separated' field drives the ship/no-ship\n"
                  "decision, so a result whose own arithmetic does not reconcile is\n"
                  "worse than no result. Re-run the notebook, or pass --force.")
            return 1
        payload["verification_failures"] = problems
    else:
        print("verified: accuracy, Wilson intervals and `separated` all recompute locally\n")

    print(summarise(payload))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    payload["imported_utc"] = stamp
    payload["imported_from"] = str(src)
    out = RESULTS_DIR / f"{stamp}-kaggle-import.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(RESULTS_DIR.parents[1])}")

    if args.markdown:
        md = Path(args.markdown)
        if md.exists():
            dest = RESULTS_DIR / f"{stamp}-kaggle-import.md"
            shutil.copy2(md, dest)
            print(f"      {dest.relative_to(RESULTS_DIR.parents[1])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
