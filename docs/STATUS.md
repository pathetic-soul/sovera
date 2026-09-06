# Project status — Sovereign Workbench

**SIH 2026 · PS 26117 (MRPL)** · reporting date **2026-08-29** · owner: build team

Single source of truth for where the project stands. Regenerate the gate table
with `python gates.py` before every status update — do not hand-edit the numbers.

---

## 1. Executive summary

**RAG: 🟡 AMBER.** Five of eight demo legs are built, measured and demonstrable.
The containment thesis — the graded claim — is fully built and provable on
stage. Two things hold the status short of green: one acceptance gate is failing
on a known, understood metric (router accuracy), and **the entire codebase is
uncommitted with zero git history**, which is the largest single risk in the
project and the cheapest to close.

| Dimension | Score | Note |
|---|---|---|
| Scope delivery | 🟢 62% | legs 1–5 of 8 built and verified; legs 6–8 open |
| Quality | 🟢 | 250 tests pass, 1 deliberate xfail; mypy strict clean over 54 files |
| Technical risk | 🟡 | router gate failing 86.7% vs 90%; grounding gap closed by `tools/calc.py` |
| Delivery risk | 🔴 | no version-control history; golden path never timed end to end |
| Evidence quality | 🟢 | every headline claim has a measured number and a way to reproduce it |

**Pitch position:** the thesis is defensible today. What is not yet de-risked is
the *performance of the demo*, not the *truth of the claim*.

---

## 2. Acceptance gates — AGENTS.md §13

Run `python gates.py` (add `--strict` on demo morning, where "could not check"
and "broken" cost the same). Verdicts are never collapsed: **SKIP is not PASS.**

| Gate | Verdict | Number | Note |
|---|---|---|---|
| `pytest` suite | ✅ PASS | 250 passed, 1 xfailed | 44 s |
| `mypy --strict` | ✅ PASS | clean, 52 source files | includes `finetune/`, the largest module |
| Router accuracy on held-out | ❌ **FAIL** | **86.7% vs 90% gate** | 10 of 75 misrouted — risk R3 |
| Model accuracy vs base | ✅ PASS | base 85.5% (CI 83.6–87.2), noise floor 1.53 pts | no unproven adapter registered |
| Corpus integrity | ✅ PASS | 60 train / 40 bench docs, 0 overlap | derived figures reconcile |
| Sandbox image built | ⚠️ SKIP | — | `docker build -t sandbox-py:local sandbox\` **before arming the firewall** |
| Peak VRAM < 5.2 GB | 🔶 MANUAL | 4.6 GB on `writer`, the largest model | re-measure *during* the golden path, not at idle |
| `sovereignty/verify.ps1` | ❌ FAIL (expected off-demo) | exit 1, 4 failures | fails because the firewall is disarmed; must exit 0 on stage |
| Zero egress delivered | 🔶 MANUAL | — | arm firewall → golden path → red button → read the drops panel |
| `.docx` opens in Word | 🔶 MANUAL | — | open newest `workspace/out/*.docx`; confirm headings + sign-off block |
| Golden path < 6 min | 🔶 MANUAL | — | **never timed end to end.** Risk R2 |

**4 passed · 2 failed · 1 skipped · 4 manual.**

Four manual gates have never been executed. That is not the same as passing
them: on current evidence the demo has not been rehearsed as a whole.

> Fixed today: the router-accuracy gate was throwing `AttributeError` and
> reporting SKIP, so the project's single failing headline metric was silently
> unmeasured by its own gate runner. It now fits on the training split exactly
> as the shipping router does and reports 80.3% FAIL.

---

## 3. Delivery status — the eight demo legs (§14)

| # | Leg | State | Evidence |
|---|---|---|---|
| 1 | Sovereignty panel, red button, audit log | ✅ Done | firewall drop rows + hash-chained JSONL, tamper check live |
| 2 | Registry + deterministic router, 2 models | ✅ Done | 0.05 ms/decision vs a 50 ms budget; accuracy below gate |
| 3 | Live model addition | ✅ Done | verified 3 ways: accepted · over-budget rejected · bad YAML safe |
| 4 | Agent loop, 6 tools, `.docx` deliverable | ✅ Done | 4.6 GB peak, ~157 s cold; human gate wired both ways (auto-approve opt-in, audited as `granted_by: auto`) |
| 5 | Coding task in `--network none` sandbox | ✅ Done | `Errno 101 Network is unreachable` from inside the container |
| 6 | Multimodal — handwriting / drawing title block | ⬜ Not started | `vision` (qwen3-vl:4b) already staged and routed to |
| 7 | Ingest + hybrid retrieval over 10–15 docs | ⬜ Not started | `ingest/` and `retrieval/` packages do not exist yet |
| 8 | Cable pull | ⬜ Not started | trivial once 1–7 hold |

**Scope freeze: end of week 3** (§14). Project start ≈ 2026-08-24 by earliest
file timestamp, putting freeze around **2026-09-13** — confirm against the real
SIH calendar. After freeze: bug fixes and rehearsal only.

---

## 4. Risk register

Scored **probability (1–5) × impact (1–5)**. Response bands: >18 avoid ·
12–18 mitigate · 8–12 transfer/contain · <8 accept.

| ID | Risk | P | I | Score | Response | Action |
|---|---|---|---|---|---|---|
| **R1** | **Zero git commits.** Weeks of work exist in one folder on one laptop. One bad delete, one disk fault, one cloud-sync conflict and the project is gone. | 4 | 5 | **20** | **Avoid** | Commit today, push to a private remote before the next work session. |
| **R2** | Golden path never timed end to end against the 6-minute §13 gate. The first full run would happen on stage. | 4 | 4 | **16** | Mitigate | Full dress rehearsal with a stopwatch, this week. Non-negotiable before freeze. |
| **R3** | Router accuracy 86.7% vs the 90% gate — the one §13 number a judge can ask about. | 5 | 3 | **15** | Mitigate | Corpus expanded 176 → 209 exemplars across the six weakest classes (2026-08-31); misses fell 13 → 10 but the gate is still not met. Continue expanding — no single confused pair dominates, so broad coverage is still the lever. |
| **R4** | Venue laptop may not run as Administrator; the drop-log monitor needs read access to `pfirewall.log`. Without it the sovereignty panel — the graded claim — degrades on stage. | 3 | 5 | **15** | Mitigate | Rehearse the exact elevated launch sequence; pre-agree an ACL relaxation on that one file as fallback. |
| **R5** | Thermal throttle after ~20 min of sustained inference on the RTX 4050 laptop. A silent killer mid-demo. | 3 | 4 | **12** | Mitigate | 25-minute sustained-load soak test with clocks logged. Currently untested. |
| **R7** | Sandbox image not built on the demo machine. It is the only step that needs a network, and §2.1 forbids doing it at demo time. | 2 | 5 | **10** | Mitigate | Build it in the same pre-flight window as model staging. `gates.py` checks for it. |
| **R6** | Doc drift. §16 previously recorded router accuracy as 75.8% with the encoder undecided; both were stale until the 2026-08-31 corpus-expansion task refreshed §16 to 86.7% and resolved the encoder decision. README's test count was also fixed. | 4 | 2 | 8 | Contain | Quote `gates.py`, never a hand-typed number. Keep §16 moving in lockstep with every future number change, so this risk does not reopen. |
| **R8** | Model arithmetic on tabular inspection data — a wrong thickness figure is the one error a refinery audience will catch. | 2 | 4 | 8 | **Contained** | `tools/calc.py` moved the base model from 1/8 to 7/8 correct. Route every derived figure through `calc`/`py_sandbox`, never mental math. |
| **R9** | Legs 6–8 unbuilt with scope freeze approaching. | 3 | 2 | 6 | Accept | The charter is explicit: *ship three legs deep, not six shallow.* Do not chase 6–8 at the cost of rehearsal. |

**Top three actions, in order:** commit the repo (R1) → dress-rehearse and time
the golden path (R2) → expand router exemplars (R3).

---

## 5. What is genuinely strong

Worth protecting under time pressure, because this is what differentiates the
submission:

1. **The claim is physical, not promised.** Three independent containment layers
   in the order a packet meets them — Windows Firewall, then `net_guard`, then
   `--network none` on the sandbox. The red button is deliberately routed around
   our own Python so the *firewall* is what kills it; being blocked by our own
   code would prove nothing.
2. **Tamper-evident audit.** Hash-chained JSONL. Edit one byte and
   `audit verify` names the first bad record by sequence number, live on stage.
3. **Honest measurement discipline.** The router blend weight was tuned by
   leave-one-out CV on the training split alone, and the held-out set was scored
   once, afterwards. The failing gate is recorded as a strict xfail so it cannot
   be quietly forgotten. `gates.py` refuses to let SKIP masquerade as PASS.
4. **A negative result kept and published.** The QLoRA adapter did not work
   (0/8 either way); `tools/calc.py` did (7/8). Both are documented. Most
   hackathon submissions bury this — keeping it is evidence of judgement.
5. **The scaling answer already exists.** `config/profiles/{6gb,16gb,120gb}.yaml`
   — same code, different registry. That is the answer to "will this run on our
   real server?"

---

## 6. Repository map

```
SIH 26117/
├─ AGENTS.md            the charter — invariants, contracts, gates. Read first.
├─ README.md            install + the demo runbooks
├─ gates.py             the §13 acceptance gates as one runnable command
├─ docs/                management + pitch layer (this folder)
│  ├─ STATUS.md         this file
│  ├─ MODULARITY.md     code shape: what to split, in what order, and why not
│  └─ superpowers/plans/2026-08-31-remaining-work.md
│                       the executable plan for M2/M5/M6 + router corpus
│  ├─ ROADMAP.md        legs 1–8, scope freeze, what ships
│  ├─ DEMO-DAY.md       pre-flight checklist and the stage script
│  └─ Sovereign-Workbench-Architecture-Report.pdf
├─ core/                orchestrator · registry · router · agent · audit · net_guard
│  └─ routing/          types · features · overrides · scorers · classifier · split
├─ backends/            LLMBackend ABC + Ollama over loopback
├─ tools/               registry + fs_read · py_sandbox · doc_write · calc — all jailed to workspace/
├─ sovereignty/         firewall.ps1 · verify.ps1 · monitor.py
├─ config/              models.yaml (the ONLY place models are named) + profiles + exemplars
├─ data/corpus/         refinery SOPs, API 510 UT reports, filings
├─ finetune/            build-time only: QLoRA, benchmarks, corpus curation
├─ tests/               250 tests, 1 deliberate xfail
├─ sandbox/Dockerfile   the network-less execution image
└─ web/index.html       the panel — vanilla JS, zero dependencies
```

**Deliberately untracked** (`.gitignore`): `models/` (31 GB of weights),
`workspace/` (agent output), `tools_ext/` (205 MB vendored llama.cpp),
`finetune/runs/`, `.venv/`, caches. Build artefacts are reproducible; only
sources are tracked.
