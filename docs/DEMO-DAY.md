# Demo day — pre-flight and stage script

**SIH 2026 · PS 26117 (MRPL)** · Sovereign Workbench

The README holds the full runbooks per leg. This is the compressed version: what
to do before you walk in, what to say on stage, in what order, and what to do
when something breaks.

**The single claim being defended:**

> Not an AI we *promise* is offline. An AI that *physically cannot* reach the
> internet — and shows you the blocked packets.

Everything else is supporting evidence.

---

## 1. Pre-flight — the night before

Everything here needs a network. **All of it must be done before the firewall is
armed**; §2.1 forbids any of it at demo time.

- [ ] `docker build -t sandbox-py:local sandbox\` — the sandbox image
- [ ] `ollama list` — confirm every `ref` in `config/models.yaml` resolves. A bad
      tag is invisible until the first call 404s, on stage.
- [ ] `python -m core.embed --stage` — the router's encoder weights
- [ ] Warm each model once so the first stage run is not a cold pull
- [ ] `python gates.py --strict` — read every non-green line and decide about it
- [ ] Battery charged, power brick packed, laptop on a hard surface (see thermals)

## 2. Pre-flight — the morning of

- [ ] Open an **Administrator** PowerShell. Confirm the drop-log monitor can read
      `%windir%\system32\LogFiles\Firewall\pfirewall.log`. If it cannot, the
      sovereignty panel says so in yellow rather than lying — but fix it first.
- [ ] `.\sovereignty\firewall.ps1 -Enable`
- [ ] `.\sovereignty\verify.ps1` — **must exit 0.** If it does not, do not start.
- [ ] `python gates.py --strict` once more, contained
- [ ] One silent full run of the golden path. Stopwatch it. Target: under 6 min.
- [ ] Clear `workspace/out/` so the `.docx` produced on stage is visibly new

---

## 3. Stage script

Timings are measured on the demo machine: leg 4 ≈ 157 s cold (includes model
load), leg 5 ≈ 12 s warm.

### Act 1 — the claim, made physical *(~90 s)*

Open the panel. Point at the sovereignty status, then press **the red button**
("Attempt external call — api.openai.com:443").

**Say:** *"Every vendor says their AI is private. We are going to try to reach
the internet, live, and let you watch it fail."*

Expect on screen:

- `BLOCKED at connect after 4.0s — TimeoutError`
- a new row in the recent-drops table, flashing red, with the real destination
  **IP and port**
- two linked audit records — intent, then result — with a red left border

**The line that matters:** *"That attempt deliberately bypasses our own Python.
If our own code blocked it, that would prove nothing. The **firewall** killed
it, and that is a packet you can see."*

Then click **verify chain** in the header. Optionally edit one byte of a
`payload` in the JSONL and re-run: it reports `FAIL chain broken at seq N` and
names the first bad record.

### Act 2 — routing, and why it is deterministic *(~60 s)*

Type a summary task, then a coding task. Read the decision line under the box
aloud — it is the `reason` field verbatim:

```
writer  summarize (0.32 vs scan_understanding 0.05) · image absent · 0.5k ctx · -> writer [cold start]
coder   code_write (0.36 vs calc 0.06) · image absent · 0.5k ctx · -> coder [swap from writer] · requires py_sandbox
```

**Say:** *"No LLM is consulted to make that decision and no weights are touched.
0.05 milliseconds. Same input, same output, every time — because a routing
decision you cannot reproduce is a routing decision you cannot audit."*

### Act 3 — live model addition *(~45 s, the "is this hard-coded?" answer)*

Paste a block into `config/models.yaml`, click **Reload registry**. Three things,
in this order:

| Paste | What happens |
|---|---|
| a model that fits, `priority: 0`, `routes: [summarize]` | accepted; the next summary task routes to it |
| the same model with `vram_gb: 42.0` | **rejected** — `claims 42.0 GB, profile '6gb' budget is 5.2 GB`, in red |
| a deliberate YAML typo | reload refuses, the error is shown, **the previous registry keeps serving** |

The third matters most: a typo pasted on stage cannot brick the app.

### Act 4 — real work, real deliverable *(~157 s — narrate the cold start)*

> Read inbox/UT-2024-114-V-2301.md and draft an approval note for the V-2301
> recommendations.

1. The route line renders **before any weights load** — that covers the cold
   start. Read it out; it turns a stall into an explanation.
2. Step 1 `fs_read` — the model reads the report rather than inventing it.
3. Step 2 stops on an amber **approval gate**. *"Nothing is written until a human
   clicks. That is a compliance feature, not friction."*
4. Approve → a `.docx` link appears → **open it in Word.** Real headings, real
   bullets, a Prepared / Reviewed / Approved sign-off block.

If time allows, run it again and click **Reject**: no file is written, the model
finishes in chat, and the audit log records `granted: false`.

### Act 5 — the third containment layer *(~12 s warm)*

> Run a Python script that computes the average wall loss in mm across grids
> S6, S7 and S8 from inbox/UT-2024-114-V-2301.md.

Route line reads `-> coder [swap from writer]`. Then show the audited argv
verbatim — the flag is evidence, not a claim:

```
docker run --rm --network none --read-only --tmpfs /work:rw,size=256m,exec
  --memory 2g --cpus 2 --pids-limit 128 --security-opt no-new-privileges --cap-drop ALL
```

Then ask it to phone home from inside the container:

> Run a Python script that opens a socket to 1.1.1.1 on port 443 and prints
> whether it connected.

`OSError [Errno 101] Network is unreachable` — from inside the container, with
the host firewall not even involved.

**Close on:** *"Three independent layers, in the order a packet meets them: the
firewall, then our egress guard, then a container with no network device at all."*

---

## 4. Numbers to quote — and only these

Every figure below is measured and reproducible with `python gates.py`. Do not
quote anything that is not on this list.

| Claim | Number | Source |
|---|---|---|
| Peak VRAM, largest model resident | **4.6 GB** against a 5.2 GB ceiling | `writer` qwen3:8b-q4_K_M, num_ctx 8192 |
| Routing decision latency | **0.05 ms** against a 50 ms budget | deterministic, no LLM |
| Encoder classification latency | **14.4 ms median / 17.2 ms p95**, on CPU, **0 GB VRAM** | bge-small-en-v1.5 |
| Router accuracy | **86.7%** held out, 10 of 75 misrouted — *gate is 90%, we are short* | see §5 below |
| Router accuracy, same 66 held-out prompts as before the corpus expansion | **86.4%**, 9 of 66 misrouted (was 13) — none of the 66 moved into TRAIN | independent generalisation check, not the §13 gate number |
| Test suite | **225 passed, 1 deliberate xfail** | `pytest -q` |
| Type checking | **mypy strict, clean, 52 files** | includes the largest module |
| Grounding — lookup questions | **10/10, both models** | `finetune/grounding_eval.py` |
| Arithmetic, before and after `tools/calc.py` | **1/8 → 7/8** over 8 trials | same task, same model |
| Leg 4 wall clock | ~157 s cold, including model load | measured |
| Leg 5 wall clock | ~12 s once `coder` is warm | measured |
| Token spend, full demo | 4–6k of a 20k cap | measured |

---

## 5. Hard questions, and the honest answers

**"Your router accuracy is below your own gate."**
Yes — 86.7% against a 90% gate, and it is written down in our own test suite as a
strict xfail so it cannot be quietly forgotten. The dense encoder took it from
75.8% to 80.3%; a corpus expansion on 2026-08-31 (176 → 209 exemplars across
the six weakest classes) cut the misses from 13 of 66 to 10 of 75. There is no
longer one dominant confusion to point to — the 10 remaining misses land on
the same six weak classes, spread across nine distinct confusion pairs, with
only `code_write` → `calc` repeating (2 of 10). The lever is still corpus
size, not tuning. And we tuned the blend weight on the training split only,
scoring held-out once at the end; tuning against held-out would have bought a
prettier number and destroyed its meaning.

**"Did fine-tuning help?"**
No, and we kept the result. The QLoRA adapter scored 0/8 with and without the
calc tool; the base model plus `tools/calc.py` scored 7/8. Fine-tuning on a fixed
tool set also made the model brittle — offered a tool it had not seen in
training, it refused to use it. We shipped the tool and not the adapter.

**"Will this run on our real server?"**
`config/profiles/` ships `6gb.yaml`, `16gb.yaml` and `120gb.yaml`. Same code,
different registry. The 6 GB profile is the constraint we chose to prove it
under, not the ceiling.

**"How do I know the audit log is real?"**
Hash-chained. Edit one byte and `audit verify` names the first bad record by
sequence number. Happy to do it now.

**"What about arithmetic errors?"**
Real, measured, and contained. Both models retrieve the right cell from the
table 10 times out of 10; mental arithmetic was the failure, and it was *worse*
on the larger model. `tools/calc.py` puts the formulas in code with an API 510
citation and prints the substitution, so an engineer can check it by hand. Do not
ask the model to do arithmetic in its head — that is a design rule, not a
workaround.

---

## 6. If something breaks on stage

| Symptom | Say this, do this |
|---|---|
| `!! CONTAINMENT FAILED` in the panel | The firewall is not armed. That message is the check being honest, not a bug — point that out, then arm it. |
| Drop panel shows zero and a yellow warning | The app cannot read `pfirewall.log`. It says so rather than reporting a false zero. Relaunch elevated. |
| Docker daemon down, `py_sandbox` fails | There is deliberately **no host-execution fallback.** The tool fails and names the fix. Say that: silently running untrusted code outside the container is the failure mode we refused to build. |
| A model swap stalls | Read the route line aloud. It rendered before any weights loaded and it explains exactly what is happening. |
| A figure looks wrong | Do not defend it. Re-run it through `calc`/`py_sandbox` live, which prints the substitution. |

**Never** improvise a task that asks for a derived figure the demo script has not
been rehearsed on. A wrong thickness number is the one error a refinery audience
will catch.
