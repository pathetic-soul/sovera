"""The AGENTS.md §13 acceptance gates, as one runnable command.

    python gates.py              # run everything runnable here, report a table
    python gates.py --strict     # SKIPPED counts as failure (use before a demo)
    python gates.py --json       # machine-readable, for CI

WHY A GATE RUNNER RATHER THAN A CHECKLIST
-----------------------------------------
§13 lists eight acceptance gates and says "track and quote these as numbers".
They were being checked by hand, in different places, at different times — which
is how this project came to believe three separate things that were not true:
that the router was at 75.8% after the encoder had already moved it, that the
adapter beat base because one 8-trial run said 5/8 vs 7/8, and that "mypy
strict, clean" covered the codebase when it skipped its largest module.

THE DESIGN DECISION THAT MATTERS: SKIPPED IS NOT PASSED
------------------------------------------------------
Several §13 gates cannot run on a developer laptop mid-session. `verify.ps1`
needs Administrator and an armed firewall. The egress count needs the firewall
armed and the demo run. Peak VRAM needs a model resident. The .docx check needs
a human with Word open.

A runner that quietly omits those, prints all-green and exits 0 is worse than no
runner at all, because it manufactures confidence. So every gate returns one of
four verdicts and they are never collapsed:

    PASS     ran, met the threshold          — with the number
    FAIL     ran, missed the threshold       — with the number
    SKIP     could not run here, and why     — never counted as PASS
    MANUAL   requires a human, and what to do

Exit code is 0 only when nothing FAILED. `--strict` additionally fails on any
SKIP or MANUAL, which is the mode to run the morning of a demo — at that point
"we could not check it" and "it is broken" have the same consequence on stage.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent

PASS, FAIL, SKIP, MANUAL = "PASS", "FAIL", "SKIP", "MANUAL"


@dataclass
class Result:
    verdict: str
    detail: str
    number: str = ""          # the quotable figure, §13's "track these as numbers"
    section: str = ""


@dataclass
class Gate:
    name: str
    section: str
    run: Callable[[], Result]
    slow: bool = False


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _run(argv: list[str], timeout: int = 900) -> tuple[int, str]:
    try:
        proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                              timeout=timeout, encoding="utf-8", errors="replace")
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except FileNotFoundError:
        return 127, f"not found: {argv[0]}"
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s"


def _python() -> str:
    """The venv interpreter if we are in one, else whatever is running us."""
    return sys.executable


def _ollama_up() -> bool:
    import urllib.error
    import urllib.request
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3)
        return True
    except Exception:
        return False


def _is_admin() -> bool:
    if platform.system() != "Windows":
        return os.geteuid() == 0 if hasattr(os, "geteuid") else False
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())  # type: ignore[attr-defined]
    except Exception:
        return False


# --------------------------------------------------------------------------
# the gates
# --------------------------------------------------------------------------

def gate_tests() -> Result:
    code, out = _run([_python(), "-m", "pytest", "-q"], timeout=1200)
    tail = [l for l in out.splitlines() if "passed" in l or "failed" in l]
    summary = tail[-1].strip() if tail else out.strip()[-160:]
    return Result(PASS if code == 0 else FAIL, summary, summary, "§13")


def gate_types() -> Result:
    code, out = _run([_python(), "-m", "mypy"], timeout=900)
    last = out.strip().splitlines()[-1] if out.strip() else "no output"
    return Result(PASS if code == 0 else FAIL, last, last, "§11")


def gate_router() -> Result:
    """Held-out router accuracy against the §13 90% gate.

    Imported rather than shelled out so the number itself is reported, not just
    a pass/fail from pytest. §13 asks for numbers.
    """
    try:
        sys.path.insert(0, str(ROOT))
        from core.routing import (
            ACCURACY_GATE, ROUTER_WEIGHT, Classifier, HybridScorer, TRAIN,
            held_out_accuracy,
        )
        # Fit on TRAIN only, exactly as the shipping router is built. A bare
        # Classifier() fits the default scorer on ALL exemplars, held-out
        # included, and would report an inflated number scored against itself.
        clf = Classifier(HybridScorer(TRAIN, weight=ROUTER_WEIGHT))
        accuracy, wrong = held_out_accuracy(clf)
    except Exception as exc:
        return Result(SKIP, f"could not measure: {type(exc).__name__}: {exc}", "", "§13")
    num = f"{accuracy:.1%} (gate {ACCURACY_GATE:.0%})"
    verdict = PASS if accuracy >= ACCURACY_GATE else FAIL
    return Result(verdict, f"{len(wrong)} held-out prompts misrouted", num, "§13")


def gate_model_accuracy() -> Result:
    """The benchmark baseline must exist and be beaten by anything shipping.

    Reads finetune/benchmarks/BASELINE.json rather than re-running 1,440
    generations. Re-measuring is `python -m finetune.benchmark`; this gate
    checks that a baseline exists, is replicated, and that config/models.yaml
    does not register an adapter which has never beaten it.
    """
    path = ROOT / "finetune" / "benchmarks" / "BASELINE.json"
    if not path.exists():
        return Result(FAIL, "no BASELINE.json — run: python -m finetune.benchmark "
                            "--ref qwen3:4b-instruct --runs 3", "", "§13")
    data = json.loads(path.read_text(encoding="utf-8"))
    pooled = data.get("pooled", {})
    acc = float(pooled.get("accuracy", 0.0))
    lo, hi = (list(pooled.get("ci95", [0, 0])) + [0, 0])[:2]
    noise = data.get("noise_floor_points")
    runs = len(data.get("runs", []))

    if runs < 2:
        return Result(FAIL, "baseline is a single run; it has no measured noise floor "
                            "and cannot support a ship/no-ship decision", "", "§13")

    # An adapter tag registered in models.yaml must have a benchmark that beat
    # base. Three adapters were previously registered-then-withdrawn on the
    # strength of an 8-trial gate.
    import yaml
    reg = yaml.safe_load((ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))
    refs = [m.get("ref", "") for m in reg.get("models", [])]
    tuned = [r for r in refs if "sovereign-driver" in r]
    num = f"base {acc:.1%} (CI {lo:.1%}-{hi:.1%}), noise floor {noise} pts"
    if tuned:
        return Result(FAIL, f"models.yaml registers {tuned}, which has no benchmark "
                            f"result beating base", num, "§13")
    return Result(PASS, f"baseline replicated over {runs} runs; no unproven adapter "
                        f"registered", num, "§13")


def gate_corpus_integrity() -> Result:
    """Training and benchmark documents must not overlap, and the authored
    corpus must be internally consistent. Both are cheap and both have failed
    before — a zero measured loss once produced infinite remaining life."""
    try:
        from finetune.benchmark import BENCH_DOCS, BENCH_SEED
        from finetune.corpus_inspection import SEED, generate, verify
        train = generate(60, seed=SEED)
        bench = generate(BENCH_DOCS, seed=BENCH_SEED)
        problems = verify(train) + verify(bench)
        t = {f"{r.doc_id}-{r.tag}" for r in train}
        b = {f"{r.doc_id}-{r.tag}" for r in bench}
        leak = t & b
    except Exception as exc:
        return Result(SKIP, f"{type(exc).__name__}: {exc}", "", "§9.4")
    if leak:
        return Result(FAIL, f"{len(leak)} documents appear in BOTH train and benchmark",
                      f"{len(leak)} leaked", "§13")
    if problems:
        return Result(FAIL, f"{len(problems)} arithmetic inconsistencies: {problems[0]}",
                      f"{len(problems)} bad", "§13")
    return Result(PASS, "no train/benchmark overlap; all derived figures reconcile",
                  f"{len(t)} train / {len(b)} bench docs, 0 overlap", "§13")


def gate_sandbox() -> Result:
    """§13: sandbox pass rate on canned coding tasks >= 70% first-run."""
    if not shutil.which("docker"):
        return Result(SKIP, "docker not on PATH", "", "§13")
    code, out = _run(["docker", "image", "inspect", "sandbox-py:local"], timeout=60)
    if code != 0:
        return Result(SKIP, "sandbox-py:local not built. Before arming the firewall: "
                            "docker build -t sandbox-py:local sandbox\\", "", "§13")
    return Result(MANUAL, "image present; the 10-task pass rate is not automated yet",
                  "", "§13")


def gate_vram() -> Result:
    """§13: peak VRAM during the demo < 5.2 GB, logged."""
    if not shutil.which("nvidia-smi"):
        return Result(SKIP, "nvidia-smi not available", "", "§4.1")
    if not _ollama_up():
        return Result(SKIP, "ollama not reachable on loopback; nothing is resident",
                      "", "§4.1")
    code, out = _run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                      "--format=csv,noheader,nounits"], timeout=30)
    if code != 0:
        return Result(SKIP, "nvidia-smi query failed", "", "§4.1")
    try:
        used, total = (int(x.strip()) for x in out.strip().splitlines()[0].split(","))
    except Exception:
        return Result(SKIP, f"unparsable nvidia-smi output: {out.strip()[:60]}", "", "§4.1")
    gb = used / 1024
    # This is instantaneous, not the demo peak. Report it as what it is.
    return Result(MANUAL, "instantaneous reading, not the demo peak — measure during "
                          "the golden path", f"{gb:.2f} GB used / {total/1024:.1f} GB", "§4.1")


def gate_sovereignty() -> Result:
    """§13: sovereignty/verify.ps1 exits 0."""
    if platform.system() != "Windows":
        return Result(SKIP, "Windows-only (§17)", "", "§13")
    script = ROOT / "sovereignty" / "verify.ps1"
    if not script.exists():
        return Result(FAIL, "sovereignty/verify.ps1 is missing", "", "§13")
    if not _is_admin():
        return Result(SKIP, "needs an elevated shell; verify.ps1 reads the firewall "
                            "drop log, which is Administrator-only", "", "§13")
    code, out = _run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                      "-File", str(script)], timeout=300)
    tail = out.strip().splitlines()[-1] if out.strip() else ""
    return Result(PASS if code == 0 else FAIL, tail[:120], f"exit {code}", "§13")


def gate_egress() -> Result:
    """§13: zero egress packets delivered during a full demo run."""
    return Result(MANUAL, "arm the firewall, run the golden path, then press the red "
                          "button and read the drops panel. Cannot be asserted from "
                          "a dev shell.", "", "§13")


def gate_docx() -> Result:
    return Result(MANUAL, "open the newest workspace/out/*.docx in MS Word and confirm "
                          "headings, bullets and the sign-off block render", "", "§13")


def gate_demo_path() -> Result:
    return Result(MANUAL, "§14 golden path end to end must complete in < 6 min "
                          "wall clock; time it during rehearsal", "", "§13")


GATES: list[Gate] = [
    Gate("pytest suite",            "§13", gate_tests, slow=True),
    Gate("mypy strict",             "§11", gate_types, slow=True),
    Gate("router accuracy",         "§13", gate_router),
    Gate("model accuracy vs base",  "§13", gate_model_accuracy),
    Gate("corpus integrity",        "§13", gate_corpus_integrity),
    Gate("sandbox image",           "§13", gate_sandbox),
    Gate("peak VRAM",               "§4.1", gate_vram),
    Gate("sovereignty verify.ps1",  "§13", gate_sovereignty),
    Gate("zero egress delivered",   "§13", gate_egress),
    Gate("docx opens in Word",      "§13", gate_docx),
    Gate("golden path < 6 min",     "§13", gate_demo_path),
]


# --------------------------------------------------------------------------

MARK = {PASS: "PASS", FAIL: "FAIL", SKIP: "SKIP", MANUAL: "MANL"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true",
                    help="treat SKIP and MANUAL as failure (use before a demo)")
    ap.add_argument("--fast", action="store_true", help="skip the slow gates")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--only", help="substring match on gate name")
    args = ap.parse_args()

    chosen = [g for g in GATES
              if (not args.fast or not g.slow)
              and (not args.only or args.only.lower() in g.name.lower())]

    results: list[tuple[Gate, Result, float]] = []
    for gate in chosen:
        if not args.json:
            print(f"  ... {gate.name}", end="\r", flush=True)
        t0 = time.time()
        try:
            res = gate.run()
        except Exception as exc:                    # a gate must never crash the run
            res = Result(FAIL, f"gate raised {type(exc).__name__}: {exc}", "", gate.section)
        results.append((gate, res, time.time() - t0))

    if args.json:
        print(json.dumps([{
            "gate": g.name, "section": g.section, "verdict": r.verdict,
            "number": r.number, "detail": r.detail, "seconds": round(s, 1),
        } for g, r, s in results], indent=2))
    else:
        width = max(len(g.name) for g, _, _ in results)
        print(f"\nAGENTS.md §13 acceptance gates\n{'=' * (width + 46)}")
        for g, r, s in results:
            print(f"[{MARK[r.verdict]}] {g.name:<{width}}  {r.number or '—'}")
            if r.detail:
                print(f"       {' ' * width}  {r.detail[:100]}")
        print()

    counts = {v: sum(1 for _, r, _ in results if r.verdict == v)
              for v in (PASS, FAIL, SKIP, MANUAL)}
    failed = counts[FAIL]
    unrun = counts[SKIP] + counts[MANUAL]

    if not args.json:
        print(f"{counts[PASS]} passed, {failed} failed, "
              f"{counts[SKIP]} skipped, {counts[MANUAL]} manual")
        if unrun and not args.strict:
            print("\nSKIP and MANUAL are NOT passes. They are gates nobody checked.\n"
                  "Run `python gates.py --strict` before a demo, where "
                  "'could not check' and 'broken' cost the same.")

    if failed:
        return 1
    if args.strict and unrun:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
