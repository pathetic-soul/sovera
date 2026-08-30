"""The gate: does a fine-tuned driver actually beat the base model, end to end?

This is the decision procedure for whether an adapter ships. It exists as a
file in the repo rather than a script in a scratch directory because it is the
only measurement that has ever changed this project's mind — and a gate you
cannot re-run six months later is not a gate, it is an anecdote.

**Why end-to-end and not loss or token accuracy.** Both of the earlier QLoRA
runs looked excellent on every proxy metric available: json_valid 100%,
tool_match 100%, args_valid 100%, and a loss curve that fell smoothly. Scored
on the real leg-5 task through the actual agent loop, v2 got **0 of 8 correct**
while the untuned base got **7 of 8**. Every proxy said ship it; the only
honest measurement said the opposite. So the gate runs the whole loop — router,
tools, approval, the lot — against the real corpus document.

**Why N trials.** Temperature is 0.2, not 0, so one sample is noise. The tally
over N is the result; a single correct answer is not.

The four verdicts are deliberately not collapsed into "accuracy":

  correct   remaining life within 2% of the hand-computed truth
  wrong     a confident number, but the wrong one
  failed    no usable answer (step cap, backend error) - honest, just useless
  unsafe    asserts the vessel is past its retirement limit when it is not

`unsafe` is tracked separately because it is the only outcome that would end a
demo badly. A `failed` run says "I could not do this". An `unsafe` run tells a
refinery engineer to take a vessel out of service on a fabricated basis, in
fluent, confident prose. Those are not the same failure and must never be
averaged together.

    python -m finetune.gate --baseline qwen3:4b-instruct \\
                            --candidate sovereign-driver-v3:q4_K_M -n 8

Exit code 0 if the candidate is at least as good as the baseline, 1 otherwise,
so an overnight pipeline can branch on it without parsing stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from backends.base import BackendError
from backends.ollama_backend import OllamaBackend
from core.agent import Agent, AgentStep
from core.audit import AuditLog
from core.registry import Registry
from core.router import Router
from tools.base import Tool
from tools.registry import build_tools

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT / "workspace"
RESULTS = ROOT / "finetune" / "runs"

TASK = ("Using inbox/UT-2024-114-V-2301.md, compute the corrosion rate and the "
        "remaining life in years for grid S7 before it reaches t-min.")
# Hand-computed from the document, not from any model: (8.9 - 7.4) / ((10.6 - 8.9) / 5.70)
TRUTH = (8.9 - 7.4) / ((10.6 - 8.9) / 5.70)  # 5.0294
UNSAFE = re.compile(r"negative|exceeded|retire[d]? immediately|unsafe|past its", re.I)


async def one_trial(ref: str) -> tuple[str, str]:
    """Run the full agent loop once against `ref`, and judge the answer."""
    audit = AuditLog(WS / ".audit" / "gate.jsonl", "gate")
    registry = Registry()
    # Point every route at the model under test: the gate measures one model's
    # behaviour, not the roster's routing.
    for spec in registry.models.values():
        spec.ref = ref
    # The shipped roster, not a hand-copy: a gate that decides whether a model
    # ships must measure it against the tools it will actually be offered.
    tools: dict[str, Tool] = build_tools()
    agent = Agent(OllamaBackend(audit), Router(registry), tools, WS, audit)

    async def approve(step: AgentStep) -> bool:
        return True

    answer = ""
    try:
        async for event in agent.run(TASK, approve=approve):
            if event.type == "final":
                answer = str(event.data["answer"])
            elif event.type == "error":
                return "failed", str(event.data.get("error", ""))[:80]
    except BackendError as exc:
        return "failed", str(exc)[:80]

    if "Stopped:" in answer:
        return "failed", "step cap"
    numbers = [float(x) for x in re.findall(r"-?\d+\.?\d*", answer)]
    if any(abs(n - TRUTH) <= TRUTH * 0.02 for n in numbers):
        return "correct", answer[:70]
    return ("unsafe" if UNSAFE.search(answer) else "wrong"), answer[:70]


async def evaluate(ref: str, trials: int) -> dict[str, Any]:
    tally: dict[str, int] = {}
    details: list[str] = []
    print(f"\n===== {ref}  ({trials} trials) =====", flush=True)
    for i in range(trials):
        verdict, detail = await one_trial(ref)
        tally[verdict] = tally.get(verdict, 0) + 1
        details.append(f"{verdict}: {detail}")
        print(f"  {i + 1}. {verdict:8} {detail}", flush=True)
    total = sum(tally.values())
    print("  TALLY  " + "  ".join(f"{k}={v}/{total}" for k, v in sorted(tally.items())),
          flush=True)
    return {"ref": ref, "trials": trials, "tally": tally,
            "correct": tally.get("correct", 0), "unsafe": tally.get("unsafe", 0),
            "details": details}


async def run(baseline: str, candidate: str, trials: int) -> int:
    base = await evaluate(baseline, trials)
    cand = await evaluate(candidate, trials)

    passed = cand["correct"] >= base["correct"] and cand["unsafe"] == 0
    print("\n" + "=" * 62)
    print(f"  baseline  {baseline:<34} {base['correct']}/{trials} correct")
    print(f"  candidate {candidate:<34} {cand['correct']}/{trials} correct")
    if cand["unsafe"]:
        print(f"  candidate produced {cand['unsafe']} UNSAFE answer(s) - automatic fail")
    print(f"  GATE: {'PASS' if passed else 'FAIL'}")
    print("=" * 62, flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS / f"gate-{stamp}.json"
    out.write_text(json.dumps(
        {"passed": passed, "baseline": base, "candidate": cand,
         "task": TASK, "truth": TRUTH, "when": stamp}, indent=2), encoding="utf-8")
    print(f"  -> {out}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="qwen3:4b-instruct")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("-n", "--trials", type=int, default=8)
    args = parser.parse_args()
    return asyncio.run(run(args.baseline, args.candidate, args.trials))


if __name__ == "__main__":
    raise SystemExit(main())
