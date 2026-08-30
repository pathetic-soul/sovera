# Sovereign Workbench

Air-gapped agentic AI workbench for MRPL. SIH 2026, PS 26117.
Design constraints and contracts live in [AGENTS.md](AGENTS.md) — read that first.
Project status, risks and the demo-day script live in [docs/](docs/).

> Not an AI we *promise* is offline. An AI that *physically cannot* reach the
> internet, and shows you the blocked packets.

**Status:** demo legs 1–5 of 8 (§14) — sovereignty panel, red button, audit log,
model registry, deterministic router, live model addition, the agent loop with
three tools, and the container sandbox.

Legs 4–5 are the first real VRAM spend. **Measured peak 4.6 GB** against the
5.2 GB ceiling (§4.1), on the RTX 4050, with `writer` (qwen3:8b-q4_K_M)
resident — the largest model in the roster. That closes one of the §16 blockers
for this model; the rest still need `ollama ps` figures.

**Router accuracy: 86.7%** on held-out prompts, against the §13 gate of 90%.
The dense encoder §9.2 originally specified has landed, and it closed most of
the gap the lexical scorer had plateaued against; the exemplar corpus was then
widened from 176 to 209 rows across the six weakest classes:

| scorer | TRAIN LOO-CV | held-out (75 prompts) |
|---|---|---|
| lexical TF-IDF (was shipping) | 74.5% | 75.8% (66 prompts, pre-expansion) |
| dense `bge-small-en-v1.5` | 84.5% | — |
| **hybrid, w=0.8 (ships now)** | **85.5%** | **86.7%** |

The encoder runs on **CPU** — §9.2 specifies it, and it is the right call
anyway: 14.4 ms median / 17.2 ms p95 per classification against a 50 ms budget,
for **0 GB VRAM** while an 8B model is resident. Weights are staged at build
time (`python -m core.embed --stage`); if they are absent the router falls back
to the lexical scorer with a warning rather than failing to start.

How the number was obtained, because it is the part that makes it worth
quoting: the blend weight was chosen by leave-one-out CV on the **training
split only**, and the held-out set was scored **once**, afterwards. Tuning
against held-out would have bought a prettier number and destroyed its meaning.

**Still short of the gate.** 10 of 75 held-out prompts miss, spread across the
same six weak classes (`qa`, `code_write`, `plan`, `summarize`, `calc`,
`scan_understanding`) with no single confusion dominating — `code_write` ->
`calc` is the only pair that repeats (2 of 10). The corpus expansion from 176
to 209 exemplars cut misses from 13 to 10 without closing the gate; the next
lever is more of the same, not tuning. Tracked in [AGENTS.md §16](AGENTS.md);
recorded in the suite as a strict xfail so the gap cannot be quietly forgotten.

**The gain is real generalisation, not a moved denominator.** Scored against
exactly the same 66 held-out prompts used before the corpus expansion — none
of them moved into TRAIN under the new split — accuracy is **86.4%, 9 misses**
(was 13). The 75-prompt figure above is the current shipping gate number; this
one exists to answer the sharpest question a judge can ask about it.

---

## Install

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

Python 3.11+ (3.13.9 here). Deps: fastapi, uvicorn, pydantic.
No CDN, no npm, no webfont — the UI is one vendored HTML file.

## Run

```powershell
.venv\Scripts\python -m core.orchestrator      # http://127.0.0.1:8080
```

Loopback bind only, never `0.0.0.0`.

## Tests

```powershell
.venv\Scripts\python -m pytest -q              # 182 passed, 1 xfailed
.venv\Scripts\python -m mypy                   # strict, clean
.venv\Scripts\python gates.py                  # all §13 acceptance gates, one table
.venv\Scripts\python gates.py --strict         # run this the morning of the demo
```

The one xfail is the router accuracy gap below, recorded deliberately.

**The sandbox image is built once, before the firewall is armed** — it is the
only step in the project that needs a network, and §2.1 forbids doing it at
demo time:

```powershell
docker build -t sandbox-py:local sandbox\
```

---

## Demo runbook — leg 1

Run these in order. Steps 1 and 2 need an **Administrator** PowerShell.

**1. Arm containment.** Cuts this machine off the internet until step 6.

```powershell
.\sovereignty\firewall.ps1 -Enable
```

Sets outbound to default-DENY on all firewall profiles, allows the local subnet,
and turns on drop logging to `%windir%\system32\LogFiles\Firewall\pfirewall.log`.
Loopback is never filtered by Windows Firewall, so Ollama and this app keep working.

**2. Prove it on stage.**

```powershell
.\sovereignty\verify.ps1        # exits 0 only if contained
```

Asserts: outbound default-DENY, drop logging on, log present, zero established
external sockets, Ollama loopback-only, audit chain intact.

**3. Start the app** and open `http://127.0.0.1:8080`.

The app must be able to read `pfirewall.log`. That is Administrator-only by
default — run it elevated, or relax the ACL on that one file. Decide before
rehearsal; if the monitor can't read the log the panel says so in yellow rather
than silently reporting zero.

**4. Press the red button.** "Attempt external call (api.openai.com:443)".

The attempt deliberately bypasses `core/net_guard.py` so the **firewall** is what
kills it — being blocked by our own Python would prove nothing. Expect:

- panel: `BLOCKED at connect after 4.0s — TimeoutError`
- recent-drops table: a new row flashing red with the real destination IP and port
- audit log: two linked records (intent, then result) with a red left border

If it ever says `!! CONTAINMENT FAILED`, the firewall is not armed. That is the
check being honest, not a bug — with step 1 skipped it reports exactly that.

**5. Verify the chain.** Click *verify chain* in the header, or:

```powershell
.venv\Scripts\python -m core.audit verify workspace\.audit\audit.jsonl
```

To show tamper-evidence live: edit one byte of any `payload` in the JSONL and
re-run. It reports `FAIL chain broken at seq N` and names the first bad record.

**6. Disarm** when you're done.

```powershell
.\sovereignty\firewall.ps1 -Disable
```

---

## Demo runbook — legs 2 and 3, routing

Nothing here needs the firewall or admin rights. Start the app and use the two
right-hand panels.

**Two tasks, two models (§14.2).** Type a summary task and then a coding task.
The decision line under the box is the `reason` field verbatim — it is what
covers the 2–5 s model swap in §4.2.3, so the delay reads as an explanation
rather than a stall:

```
writer  summarize (0.32 vs scan_understanding 0.05) · image absent · 0.5k ctx · -> writer [cold start]
coder   code_write (0.36 vs calc 0.06) · image absent · 0.5k ctx · -> coder [swap from writer] · requires py_sandbox
vision  drawing_qa (0.30 vs handwriting 0.11) · image attached · 1.5k ctx · -> vision [swap from writer]
```

No LLM is consulted and no weights are touched: **0.05 ms per decision** against
the 50 ms budget in §9.2. Same input, same output, every time.

**Live model addition (§14.3).** Paste a block into `config/models.yaml`, click
*Reload registry*, route to it. No code change, no restart. Three things worth
showing, in this order:

| Paste this | What happens |
|---|---|
| a model that fits, `priority: 0`, `routes: [summarize]` | accepted, and the next summary task routes to it |
| the same model with `vram_gb: 42.0` | **rejected**, with `claims 42.0 GB, profile '6gb' budget is 5.2 GB` shown in red |
| a deliberate YAML typo | reload refuses, the error is shown, and the previous registry keeps serving |

The third one matters more than it looks: a typo pasted on stage cannot brick
the app.

---

## Demo runbook — legs 4 and 5, the agent

Needs Ollama running and the sandbox image built. Use the full-width **Agent**
panel. `workspace/inbox/` is seeded from `data/corpus/` on first start, so a
fresh clone has a document to work on.

**Leg 4 — scanned report to `.docx` (§14.4).** Paste:

> Read inbox/UT-2024-114-V-2301.md and draft an approval note for the V-2301
> recommendations.

What to point at, in order:

1. The route line renders **before any weights load** — that is what covers the
   cold start (§4.2.3): `approval_note (0.43 vs plan 0.05) · -> writer [cold start]`
2. Step 1 `fs_read` — the model reads the report rather than inventing it.
3. Step 2 stops on an amber **approval gate**. Nothing is written until a human
   clicks. This is §2.4 and it is a compliance feature, not friction.
4. Approve. A `.docx` link appears; it opens in Word with real headings, real
   bullets and a Prepared/Reviewed/Approved sign-off block.

**Show the gate refusing.** Run it again and click *Reject*. No file is
written, and the model falls back to answering in the panel. The audit log
records `granted: false`.

**Leg 5 — code in a network-less container (§14.5).** Paste:

> Run a Python script that computes the average wall loss in mm across grids
> S6, S7 and S8 from inbox/UT-2024-114-V-2301.md.

The route line now reads `-> coder [swap from writer]`. After approval the
container runs with `--network none`, and the full argv is in the audit log
verbatim — the flag is evidence, not a claim:

```
docker run --rm --network none --read-only --tmpfs /work:rw,size=256m,exec
  --memory 2g --cpus 2 --pids-limit 128 --security-opt no-new-privileges --cap-drop ALL
```

**The sandbox is the third containment layer.** Ask it to phone home:

> Run a Python script that opens a socket to 1.1.1.1 on port 443 and prints
> whether it connected.

It prints `OSError [Errno 101] Network is unreachable` — from inside the
container, with the host firewall not even involved.

Measured on this machine: leg 4 ≈ 157 s cold (8B, includes model load), leg 5
≈ 12 s once `coder` is warm. Peak VRAM 4.6 GB. Token spend 4–6k of the 20k cap.

### Known gap in leg 5 — now measured

`finetune/grounding_eval.py` puts a number on it: 18 questions against the real
corpus table, split into cells you can look up and values you must derive.

| model | lookup | derived | total |
|---|---|---|---|
| `driver` qwen3:4b-instruct | **10/10 - 100%** | 4/8 - 50% | 77.8% |
| `coder` qwen2.5-coder:7b | **10/10 - 100%** | 2/8 - 25% | 66.7% |

Retrieval is not the problem. Both models find the correct cell **every time**.
Mental arithmetic is, and it is *worse on the larger model*: `coder` quoted
`S6: 1.5, S7: 1.7, S8: 1.6` in its own source field and answered 5.4 instead
of 4.8.

**FIXED by `tools/calc.py`.** Measured over 8 trials on the same task:
the base model goes from **1/8 to 7/8 correct** once the formulas live in code
instead of in the model's head. Every result prints its substitution
(`(8.9 - 7.4) / 0.2982 = 5.0302`) with an API 510 clause next to it, so an
engineer can check it by hand. A QLoRA adapter trained on the same problem
scored 0/8 either way — see [finetune/README.md](finetune/README.md).

**Demo consequence:** ask for figures the report states (thickness, t-min,
dates, tags) and they are right every time. Ask for a computed figure and let
the agent run it through `py_sandbox`, which is exact. Do not let a model do
arithmetic in its head on stage.

See [finetune/README.md](finetune/README.md) for the adapter aimed at this.

### Original note

The loop is sound; the 7B's *arithmetic* is not always. On a harder framing —
"corrosion rate and remaining life for grid S7" — `coder` picked grid S5's
2019 reading instead of S7's and inverted a subtraction, then reported a
negative remaining life as "already exceeded by 3.05 years". The sandbox ran
exactly what it was given, and the wrong number was confidently narrated.

Do not put that task in the demo script until it is fixed. The fix is §9.4
grounding (make the model quote the source row it used) and/or the `calc` tool
in §6, neither of which is in legs 4–5. Tracked as an open question.

## What leg 1 contains

| File | Role |
|---|---|
| `core/audit.py` | hash-chained append-only JSONL + `audit verify` CLI |
| `core/net_guard.py` | patches `socket.connect`; loopback and local subnet only |
| `core/orchestrator.py` | FastAPI composition root: app factory, lifespan, `/static` mount, `include_router` over `core/api/` |
| `core/api/` | one router per UI panel — sovereignty, registry, routing, agent, workspace |
| `core/settings.py` | `config/runtime.yaml` loader — agent caps, server bind, egress-probe target |
| `sovereignty/monitor.py` | incremental tail of the firewall drop log |
| `sovereignty/firewall.ps1` | the egress control, shown verbatim in the UI |
| `sovereignty/verify.ps1` | pre-demo assertions |
| `core/registry.py` | loads/validates/hot-reloads `models.yaml`, enforces the VRAM budget |
| `core/routing/` | deterministic task_type + modality detection |
| `core/router.py` | `RouteDecision` with a human-readable rationale |
| `config/routing_exemplars.jsonl` | 209 hand-labelled refinery prompts, 11 task types |
| `web/index.html` + `web/static/` | the panel — HTML shell + vendored CSS/JS, no dependencies |

Three independent containment layers, in the order a packet meets them:
the firewall, then `net_guard`, then `--network none` on the sandbox (leg 5).

**Windows deviations from the charter's Linux assumptions** (nftables → Windows
Firewall, drop counters → drop log) are recorded in [AGENTS.md §17](AGENTS.md).

## What legs 4 and 5 add

| File | Role |
|---|---|
| `backends/base.py` | `LLMBackend` ABC — `Message`, `Completion` |
| `backends/ollama_backend.py` | Ollama over loopback via stdlib `urllib`; audits every call |
| `core/agent.py` | the ReAct loop: 8 steps, 20k tokens, one repair, human gate |
| `tools/base.py` | `Tool` ABC, the workspace jail, flat-schema arg validation |
| `tools/fs_read.py` | jailed read, truncated to the context budget |
| `tools/doc_write.py` | `.docx` deliverable via python-docx |
| `tools/py_sandbox.py` | `docker run --network none`, no host fallback, ever |
| `sandbox/Dockerfile` | the sandbox image — stdlib only, non-root, pre-built |
| `data/corpus/inbox/` | a realistic API 510 UT report, seeded into `workspace/` |

## Next

Leg 6 (§14): multimodal — a handwritten inspection note or a drawing title
block through `vision`, answered with the source region highlighted. The
`vision` model (`qwen3-vl:4b`) is already staged and routed to.

Two things worth settling first:

- The **encoder question** in §16 — router accuracy is still the one §13 number
  failing, and legs 4–5 did not touch it.
- The **grounding gap** in leg 5 above. It is the difference between a demo
  that impresses and one that gets a wrong thickness figure questioned on
  stage.
#   s o v e r a 
 
 #   s o v e r a 
 
 