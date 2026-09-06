# AGENTS.md — Sovereign On-Premise Agentic AI Workbench

> **Read this file completely before writing any code.**
> It is the single source of truth for constraints, contracts, and scope.
> If a request conflicts with anything in `## INVARIANTS`, refuse and say why.
> Filename aliases: also usable as `CLAUDE.md`, `.cursorrules`, `GEMINI.md`, `.github/copilot-instructions.md`.

---

## 0. How to use this file

| You are… | Read at minimum |
|---|---|
| Writing code | §1, §2, §4, §5, §7, §8, §11, §14 |
| Designing a component | §2, §6, §8, §9 |
| Writing prompts / templates | §2, §5, §12 |
| Adding a model | §5, §8.1 |
| Doing anything network-related | §2, §10, §17 — **stop and re-read** |
| Deciding whether to build something | §14 |

**Behavioural rules for you, the agent:**

1. **Never** add a dependency that phones home at import or runtime. Check before adding.
2. **Never** hardcode a model name outside `config/models.yaml`.
3. **Never** widen the sandbox, the firewall, or the filesystem jail "temporarily for testing."
4. Assume **5.2 GB of usable VRAM**, not 6. Every design must state its VRAM cost.
5. Prefer boring, inspectable code over clever abstractions. Judges read source.
6. If a task would take >200 lines, propose the interface first and wait for approval.
7. When unsure between two designs, pick the one that is **easier to demo live**.

---

## 1. Mission

**SIH 2026, Problem Statement 26117** (MRPL — refinery / PSU context).

Build a **self-hosted, air-gapped agentic AI workbench** that runs entirely on one
GPU workstation, routes tasks across multiple open-weight models, executes multi-step
agentic work with local tools, understands scanned/handwritten/visual documents, and
produces **real file deliverables** — not chat replies.

**The actual thesis being defended:**

> Not an AI we *promise* is offline. An AI that *physically cannot* reach the internet,
> and shows you the blocked packets.

Containment must be **demonstrable**, not asserted. That is the graded claim.

**Primary user:** a refinery/PSU engineer or officer doing routine confidential knowledge
work — inspection reports, approval notes, MOC paperwork, internal calculations, small
scripts, vendor correspondence. Today they either do it manually or paste it into a
public chatbot. We are replacing the second behaviour.

---

## 2. INVARIANTS

These are non-negotiable. Violating any of them invalidates the entire project.

### 2.1 Sovereignty

- **No process in this system may make an outbound network call.** Ever. Not for
  telemetry, not for model download at runtime, not for a font, not for a CDN-hosted
  JS library, not for `pip install` at runtime.
- All frontend assets are **vendored locally**. No `<script src="https://…">`.
  No Google Fonts. No Tailwind CDN. Vendor into `web/static/`.
- Disable telemetry explicitly for every library used (HF, Gradio, Chroma, etc.) via
  env vars set in `config/env.sovereign`, even if the firewall already blocks it.
  **Defence in depth: firewall + app config + container isolation.**
- Model weights are pre-staged on disk. Runtime never downloads.
- Any code path that *could* egress must be routed through `core/net_guard.py`,
  which logs the attempt and raises.

### 2.2 Auditability

- Every model invocation, tool call, file read, and file write emits an
  `AuditRecord` (§8.5) to an append-only, hash-chained JSONL log.
- The audit log is written **before** the action, and updated with the result after.
- No action bypasses the audit log. If you find one, that's a bug.

### 2.3 Determinism where it matters

- **Routing is deterministic.** No LLM decides which model to use. See §9.2.
- Tool arguments are schema-validated before execution. Reject, repair, retry — never
  execute unvalidated args.
- Any file write outside `workspace/` is rejected by the filesystem jail.

### 2.4 Human gate

- The agent **proposes** file writes and code execution. A human approves.
  This is a compliance feature for a PSU, not a limitation. Do not remove it to
  "make the demo smoother."

---

## 3. Non-goals (explicitly out of scope)

Do not build these. If asked, push back and cite this section.

- Multi-user auth, RBAC, SSO, LDAP integration
- Multi-GPU or distributed serving
- ~~Fine-tuning, LoRA training, RLHF, or any training loop~~ — **overridden by
  decision, 2026-08-26.** QLoRA adaptation of `driver` lives in `finetune/`.
  The non-goal stood on the reasoning that training is a distraction from a
  containment demo, and that reasoning is still mostly right: measurement
  showed the loop behaviour it was first aimed at was already at ceiling
  (json_valid 100%, tool_match 97.3%). It was re-aimed at the §16 grounding gap
  instead. The boundary that still holds absolutely: **training is build time
  only.** It runs before the firewall is armed, alongside `docker build` for
  the sandbox image. No code path in `core/`, `tools/` or `backends/` trains,
  adapts or downloads anything at demo time, so §2.1 is untouched. If you are
  reading this during a demo, nothing here is running.
- Full P&ID symbol recognition / drawing vectorisation (**research problem — we scope
  to title-block + tag extraction + drawing Q&A, and we say so out loud**)
- Voice interface, mobile app
- Kubernetes, Helm, service mesh
- A beautiful frontend. Functional and legible only.
- Cloud fallback of any kind, even behind a flag

---

## 4. Hardware — the binding constraint

**Demo machine (fixed, confirmed): HP Omen laptop — RTX 4050 Laptop GPU, 6 GB GDDR6,
64 GB system RAM.**

This is the *entire* design constraint. Read this section twice.

### 4.1 VRAM budget

```
Physical VRAM                       6.0 GB
- display / compositor / driver    -0.5 GB
- CUDA context + fragmentation     -0.3 GB
─────────────────────────────────────────
USABLE FOR MODEL + KV CACHE         5.2 GB   ← design against this number
```

### 4.2 Consequences (all mandatory)

1. **The GPU is single-tenant for LLMs. Exactly one LLM resident at any moment.**
   Set `OLLAMA_MAX_LOADED_MODELS=1`.
2. **Embeddings, reranking, and classical OCR run on CPU.** 64 GB RAM is the
   asset here — use it. GPU time is reserved for generation only.
3. **Model switching costs 2–5 s** when weights are warm in the OS page cache
   (64 GB RAM makes this reliable after first load). Cold from NVMe: 8–20 s.
   - Pre-warm all demo models once at boot so they are page-cached.
   - **Mask the swap in the UI** — render the routing decision + rationale during load.
     The delay becomes an explanation, not a stall.
4. **KV cache quantisation is mandatory.** `OLLAMA_KV_CACHE_TYPE=q8_0` and
   `OLLAMA_FLASH_ATTENTION=1`. Roughly halves KV memory.
5. **Context ceilings are hard.** See per-model `max_ctx` in `models.yaml`. Never
   exceed. The retrieval layer must respect the ceiling and truncate/rerank, not hope.
6. Long documents are handled by **map-reduce chunking**, not long context.
7. Close Chrome, Discord, Slack, and any Electron app before the demo. They eat VRAM.

### 4.3 Required env

```bash
# config/env.sovereign
OLLAMA_MAX_LOADED_MODELS=1
OLLAMA_NUM_PARALLEL=1
OLLAMA_KEEP_ALIVE=10m
OLLAMA_FLASH_ATTENTION=1
OLLAMA_KV_CACHE_TYPE=q8_0
OLLAMA_HOST=127.0.0.1:11434        # loopback only, never 0.0.0.0
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_HUB_DISABLE_TELEMETRY=1
DO_NOT_TRACK=1
ANONYMIZED_TELEMETRY=False          # chroma
GRADIO_ANALYTICS_ENABLED=False
```

---

## 5. Model roster — LOCKED for 6 GB

> **Every size below must be re-verified with `ollama ps` / `nvidia-smi` on the actual
> machine before being trusted.** Quant sizes drift between releases. If a model does
> not fit, drop one tier — do not reduce context below the stated floor.

| Role id | Model (Q4_K_M unless noted) | ~VRAM | Device | Ctx floor | Used for |
|---|---|---|---|---|---|
| `driver` | Qwen3-4B-Instruct | ~2.6 GB | GPU | 8k | Planning, tool-call orchestration, routing fallback |
| `writer` | Qwen3-8B | ~4.9 GB | GPU | 8k | Approval notes, summaries, long-form prose |
| `coder` | Qwen2.5-Coder-7B-Instruct | ~4.4 GB | GPU | 8k | Code generation + repair |
| `vision` | Qwen3-VL-4B | ~3.2 GB | GPU | 8k | Handwriting, drawings, photos, visual Q&A |
| `ocr` | PaddleOCR-VL-1.5 / PP-OCRv6 | ~1.5 GB | **CPU** | — | Printed scan → text + layout |
| `embed` | bge-m3 (ONNX int8) | ~1.2 GB | **CPU** | — | Dense retrieval |
| `rerank` | bge-reranker-v2-m3 | ~1.2 GB | **CPU** | — | Top-k reranking |

**Stretch (only if measured to fit):** Gemma 4 9B as a combined `writer`+`vision` role,
reported to run around 6 GB with vision and tool-calling. Treat as *unproven* until
measured on this exact laptop. Do not build the demo around it.

**Scaling story for the presentation (do not implement, just document):**
`config/profiles/` ships `6gb.yaml` (demo), `16gb.yaml` (gpt-oss-20b class), and
`120gb.yaml` (gpt-oss-120b class). Same code, different registry. This *is* the answer
to "deployable on a mid-range GPU, scalable to their real server."

---

## 6. Repo layout

```
sovereign-workbench/
├── AGENTS.md                     ← this file
├── README.md                     ← install + demo runbook
├── config/
│   ├── models.yaml               ← ONLY place models are named
│   ├── profiles/{6gb,16gb,120gb}.yaml
│   ├── env.sovereign
│   ├── routing_exemplars.jsonl   ← labelled prompts for the router
│   └── runtime.yaml              ← agent caps, server bind, egress-probe target
├── core/
│   ├── orchestrator.py           ← FastAPI composition root: app factory, lifespan, /static mount
│   ├── api/                      ← one router per UI panel
│   │   ├── sovereignty.py        ← firewall rules, red button, drop stream
│   │   ├── registry.py           ← registry read + hot reload
│   │   ├── routing.py            ← /api/route
│   │   ├── agent.py              ← agent websocket, tools, backend health
│   │   └── workspace.py          ← artifact download, audit verify
│   ├── settings.py               ← config/runtime.yaml loader, validated at load
│   ├── registry.py               ← loads/validates/hot-reloads models.yaml
│   ├── routing/                  ← task_type + modality detection
│   │   ├── types.py              ← TaskType, Modality, Scorer protocol
│   │   ├── features.py           ← text → terms, modality from attachments
│   │   ├── overrides.py          ← the §9.2 hard rules
│   │   ├── scorers.py            ← lexical · dense · hybrid
│   │   ├── classifier.py         ← the facade
│   │   └── split.py              ← train/held-out + the §13 accuracy measure
│   ├── router.py                 ← RouteDecision (deterministic)
│   ├── planner.py                ← task → Plan
│   ├── agent.py                  ← ReAct loop, step cap, repair
│   ├── audit.py                  ← hash-chained JSONL
│   └── net_guard.py              ← egress interception + logging
├── backends/
│   ├── base.py                   ← LLMBackend ABC
│   ├── ollama_backend.py
│   └── llamacpp_backend.py       ← fallback if Ollama misbehaves
├── tools/
│   ├── base.py                   ← Tool ABC + JSON schema
│   ├── registry.py               ← the tool roster, one list, four consumers
│   ├── fs_read.py  fs_write.py   ← jailed to workspace/
│   ├── py_sandbox.py             ← docker --network none
│   ├── doc_write.py              ← docx / xlsx / pptx
│   ├── kb_search.py              ← hybrid retrieval
│   └── calc.py                   ← sympy, shows steps
├── ingest/
│   ├── pipeline.py               ← deskew → layout → OCR/VLM → JSON
│   ├── layout.py                 ← Surya / PaddleOCR layout
│   └── extractors.py             ← tag numbers, dates, findings
├── retrieval/
│   ├── index.py                  ← BM25 + vector
│   ├── hybrid.py                 ← RRF fusion + rerank
│   └── chunker.py
├── sovereignty/
│   ├── firewall.ps1              ← egress control (see §17)
│   ├── verify.ps1                ← pre-demo assertion script
│   └── monitor.py                ← drop-log watcher → websocket
├── web/                          ← UI: index.html + static/{app.css,js/*.js}, all vendored
├── workspace/                    ← the ONLY writable path for the agent
├── models/                       ← pre-staged weights (gitignored)
├── data/corpus/                  ← refinery SOPs, sample reports
└── tests/
```

---

## 7. Stack

- Python 3.11+, FastAPI, Uvicorn (loopback bind only)
- Ollama primary backend; llama.cpp server as fallback (`backends/` abstracts both)
- Pydantic v2 for every contract in §8
- LanceDB (embedded, no server) **or** Qdrant in Docker — prefer LanceDB for fewer
  moving parts on one laptop
- `rank_bm25` for lexical, RRF fusion with dense
- `python-docx`, `openpyxl`, `python-pptx` for deliverables
- Docker for the code sandbox
- Windows Firewall for egress control (§17)
- Frontend: plain HTML + vanilla JS + vendored CSS. **No npm build step, no CDN.**
- `structlog` for logging, `pytest` for tests

**Forbidden dependencies:** LangChain, LlamaIndex, CrewAI, AutoGen, or any framework
that hides the agent loop. We need the loop inspectable and the token budget
controllable on 6 GB. Write ~300 lines of loop yourself.

---

## 8. Contracts

All contracts are Pydantic models. Do not pass raw dicts between components.

### 8.1 `config/models.yaml`

The **only** file that names a model. Adding a model = editing this file + clicking
"Reload registry" in the UI. No code change. This is a graded requirement — protect it.

```yaml
version: 1
profile: 6gb
models:
  - id: driver
    backend: ollama
    ref: qwen3:4b-instruct-q4_K_M
    device: gpu
    vram_gb: 2.6
    max_ctx: 8192
    capabilities: [reasoning, tool_call, planning]
    routes: [plan, qa, classify]
    priority: 1

  - id: coder
    backend: ollama
    ref: qwen2.5-coder:7b-instruct-q4_K_M
    device: gpu
    vram_gb: 4.4
    max_ctx: 8192
    capabilities: [code_generation, code_repair, tool_call]
    routes: [code_write, code_fix, code_explain]
    priority: 1
    requires_tools: [py_sandbox]

  - id: vision
    backend: ollama
    ref: qwen3-vl:4b-q4_K_M
    device: gpu
    vram_gb: 3.2
    max_ctx: 8192
    capabilities: [vision, ocr_reasoning]
    routes: [scan_understanding, drawing_qa, handwriting]
    priority: 1
```

`registry.py` must: validate against a schema, **reject any model whose `vram_gb`
exceeds the profile budget**, support hot reload, and expose `models_for(route)`.

### 8.2 `RouteDecision`

```python
class RouteDecision(BaseModel):
    task_type: Literal["plan","qa","summarize","approval_note","code_write",
                       "code_fix","scan_understanding","drawing_qa",
                       "handwriting","calc","spreadsheet"]
    modality:  Literal["text","image","pdf","mixed"]
    model_id:  str
    reason:    str        # human-readable, RENDERED IN THE UI
    scores:    dict[str, float]
    est_ctx:   int
    swap_required: bool
```

`reason` is user-facing. Write it like:
`"code_write (0.91 vs qa 0.12) · image absent · 3.1k ctx → coder [swap from driver]"`

### 8.3 `Tool`

```python
class Tool(ABC):
    name: str
    description: str          # <= 200 chars, small models can't read essays
    schema: dict              # JSON Schema, <= 5 params, flat, no nesting
    requires_approval: bool

    def run(self, args: dict, ctx: RunContext) -> ToolResult: ...
```

Rules: flat schemas only (4B models fail on nested objects), ≤5 params, every param
gets a one-line description with an example value, no optional params if avoidable.

### 8.4 `AgentStep` / loop

```python
class AgentStep(BaseModel):
    n: int
    thought: str
    tool: str | None
    args: dict | None
    observation: str | None
    tokens_used: int
```

Loop rules:
- Hard cap **64 steps** (the `max_steps` schema ceiling), hard cap **1,000,000
  cumulative tokens** (`config/runtime.yaml`, raised from the original 8/20k
  charter values so long multi-document tasks don't stop early — see
  `core/settings.py`). Exceed → stop, report partial.
- Invalid tool args → one grammar-constrained repair attempt → then fail loudly.
- Observations truncated to 1500 tokens before re-entering context (6 GB, remember).
- Every step is audited and streamed to the UI as a visible plan trace.

### 8.5 `AuditRecord`

```python
class AuditRecord(BaseModel):
    seq: int
    ts: str                   # ISO8601
    session_id: str
    kind: Literal["model_call","tool_call","file_read","file_write",
                  "egress_attempt","approval","error"]
    payload: dict
    prev_hash: str            # sha256 of previous record
    hash: str                 # sha256 of this record with `hash` excluded
```

Append-only JSONL at `workspace/.audit/audit.jsonl`. `audit verify` CLI walks the
chain and reports the first break. Frame as DPDP Act 2023 / CERT-In audit readiness
in the presentation.

**Two resolved ambiguities, implemented in `core/audit.py`:**

1. *Digest scope.* An earlier draft specified `sha256(prev_hash + canonical_json(payload))`,
   which leaves `seq`, `ts` and `kind` outside the digest and therefore rewritable
   without breaking the chain. We hash the **whole record minus `hash`**. `prev_hash`
   is a field, so the link to the predecessor survives.
2. *Write-before / update-after.* §2.2 wants the outcome recorded after the fact, but
   an append-only chain cannot mutate a record. An action emits **two** records: an
   intent record, then a result record whose payload carries `ref: <intent seq>`.

---

## 9. Component specs

### 9.1 Ingest pipeline

`scan → deskew/denoise/upscale → layout detect → per-region route → structured JSON`

- Printed regions → `ocr` (PaddleOCR-VL, CPU)
- Handwritten / drawing / photo regions → `vision` (Qwen3-VL, GPU)
- Tables → layout model → markdown table, **never** freeform LLM transcription
- Output schema for inspection reports:
  `{equipment_tag, unit, inspection_date, inspector, method, findings[],
    thickness_readings[], recommendation, next_due}`
- **OCR quality caps RAG quality.** A bad OCR pass poisons the index permanently.
  Store the raw OCR text alongside the structured JSON so extraction can be re-run.

### 9.2 Router (deterministic — no LLM)

```
1. Hard overrides (highest priority, short-circuit):
   - image/PDF-with-images attached      → modality=image
   - request mentions run/execute/test   → code_* + py_sandbox required
   - explicit output format (.docx/.xlsx)→ deliverable route
2. Embedding similarity vs config/routing_exemplars.jsonl (CPU, <50 ms)
3. Registry lookup: models_for(task_type), filter by vram budget, sort by priority
4. Emit RouteDecision with reason + scores
```

Build `routing_exemplars.jsonl` with **≥100 hand-labelled prompts** in refinery
language. Report router accuracy as a number on a slide — almost no other team will
have one.

### 9.3 Sandbox

```
docker run --rm --network none --read-only
  --tmpfs /work:rw,size=256m,exec
  --memory 2g --cpus 2 --pids-limit 128
  --security-opt no-new-privileges --cap-drop ALL
  -v "$ARTIFACT_DIR":/out
  sandbox-py:local python /work/main.py
```

`--network none` is doing double duty: security control **and** sovereignty artifact.
Timeout 60 s. Capture stdout/stderr/exit code/artifacts. Never mount the host repo.

### 9.4 Retrieval

- **Hybrid BM25 + dense with RRF fusion is mandatory, not optional.**
  Pure vector search fails on refinery equipment tags (`10-P-101A`, `V-2301`,
  `PSV-4402`). Lexical matching is what makes this usable on their documents.
- Chunk: 400–600 tokens, 15% overlap, respect heading boundaries.
- Preserve `{doc_id, page, section, bbox}` on every chunk — needed for citation spans.
- Rerank top-30 → top-5 (CPU). Then truncate to the model's `max_ctx` floor.
- Every generated claim in an approval note carries a source span. Track
  **grounding rate** as a metric.

### 9.5 Deliverables

Templates in `templates/` with placeholder tokens, filled by `python-docx`.
Never generate raw XML. Output to `workspace/out/` with a timestamped filename.
The `.docx` must open cleanly in MS Word — test it, don't assume.

---

## 10. Sovereignty enforcement

**Build this in week 1. It is the highest score-per-effort item in the project.**

1. **Host firewall** — outbound default-DENY; allow loopback and the local subnet
   only. Ruleset lives in `sovereignty/firewall.ps1` and is displayed verbatim in
   the UI at `/api/firewall/rules`. See §17 for why this is PowerShell, not nftables.
2. **Live monitor** — tail the firewall drop log, stream over websocket to a
   sovereignty panel. Always visible, never hidden behind a tab.
3. **The red button** — a UI control labelled *"Attempt external call"* that tries to
   reach `api.openai.com:443`. It fails, the drop appears in the panel with its
   destination IP, and an `egress_attempt` record flashes red in the audit log.
   **This is the single most memorable thing in the demo. Build it early, polish it.**
   The button deliberately bypasses `net_guard` so the *firewall* is what blocks it —
   being blocked by our own Python would prove nothing.
4. **Cable pull** — mid-demo, physically disconnect ethernet and disable Wi-Fi.
   Keep working. Say nothing.
5. **`sovereignty/verify.ps1`** — pre-demo assertion script: firewall active, drop
   logging on, zero established external sockets, Ollama bound to loopback, audit
   chain intact. Run it on stage; show it pass.

---

## 11. Coding conventions

- Type hints everywhere. `mypy --strict` on `core/` and `tools/`.
- No bare `except`. Catch specific, log with `structlog`, re-raise or return typed error.
- No global mutable state except the registry singleton. App state lives on `app.state`.
- Every module ≤400 lines. Split beyond that.
- Docstrings state **VRAM and latency implications** for anything touching the GPU.
- Config over constants. Anything a judge might ask you to change live goes in YAML.
- Comments explain *why*, not *what*.
- Deliberate shortcuts get a `# ponytail:` comment naming the ceiling and the upgrade path.
- Commit style: `feat(router): deterministic exemplar matching`

---

## 12. Prompting small models (4B–8B) — read before writing any prompt

Prompts written for frontier models fail on a 4B. Rules:

1. **One task per call.** Never "summarise this and also draft the note and also
   extract tags." Decompose in the planner.
2. **Constrain the output format.** Use GBNF grammars (llama.cpp) or Outlines for
   anything parsed. JSON-in-prose from a 4B is a coin flip.
3. **Short system prompts.** ≤300 tokens. Long system prompts eat the context budget
   you don't have and degrade instruction-following at this size.
4. **Few-shot > instructions.** Two concrete examples beat a paragraph of rules.
5. **No nested tool schemas.** Flat, ≤5 params, examples in the descriptions.
6. **Always plan for repair.** Assume malformed output; write the repair path first.
7. **Give the model an out.** Explicit `"if the document does not state this, output
   UNKNOWN"` — this is the single most effective anti-hallucination lever on small
   models, and it matters enormously for inspection data.
8. Temperature 0.1–0.3 for extraction and code; 0.6 for prose. Never above 0.8.
9. Put retrieved context **before** the instruction, and repeat the instruction after
   it. Small models lose the middle.

---

## 13. Testing & acceptance

| Test | Gate |
|---|---|
| `pytest tests/` | green before any merge |
| Router accuracy on held-out prompts | ≥ 90% — **currently 86.7%, FAILING**, see §16 |
| Golden demo path (§14) end to end | < 6 min wall clock |
| `sovereignty/verify.ps1` | exits 0 |
| Egress packets during a full demo run | **0 delivered**, drops counted and shown |
| Generated `.docx` opens in MS Word | manual, every build |
| Peak VRAM during demo | < 5.2 GB, logged |
| Sandbox pass rate on 10 canned coding tasks | ≥ 70% first-run |

Track and quote these as **numbers** in the presentation.

---

## 14. Golden demo path — build in this order

Everything else is secondary. If a task doesn't serve this path, defer it.

1. **Sovereignty panel + red button + audit log.** ✅ built — `core/audit.py`,
   `core/net_guard.py`, `core/orchestrator.py`, `sovereignty/`, `web/index.html`
2. **Registry + deterministic router + 2 models.** ✅ built — `core/registry.py`,
   `core/routing/`, `core/router.py`, 176 labelled exemplars.
   Accuracy is below gate; see §16 before quoting a number on a slide.
3. **Live model addition.** ✅ built — `POST /api/registry/reload`. Verified three
   ways: a fitting model is accepted and routed to, an over-budget model is
   rejected with a visible reason, and malformed YAML leaves the previous
   registry serving. Zero code change in all three.
4. **Agent loop** — built with 3 tools (`fs_read`, `py_sandbox`, `doc_write`);
   now 6, after `calc` (§16, measured), `ocr_read` (leg 6) and `fs_write`.
   ✅ built — `core/agent.py`, `backends/`, `tools/`. Inspection report →
   findings → approval note → `.docx` that opens in Word. The §2.4 gate is
   wired both ways: approve writes the file, reject writes nothing and the
   model finishes in chat. 4.6 GB peak, ~157 s cold including model load.
   Input is a text report, not a scan — OCR arrives with leg 6.
5. **Coding task** run and verified in the sandbox, showing `--network none`.
   ✅ built — `tools/py_sandbox.py`, `sandbox/Dockerfile`. Verified from inside
   the container: `OSError [Errno 101] Network is unreachable`, `/src`
   read-only, host filesystem absent, artifacts captured from `/out`. The full
   `docker run` argv is audited verbatim so the flag is evidence, not a claim.
   There is deliberately **no host-execution fallback** — if the daemon is down
   the tool fails and names the fix. See the grounding gap in §16 before
   choosing which coding task to demo.
6. **Multimodal**: handwritten inspection note or a drawing title block → structured
   answer with the source region highlighted.
7. **Ingest + hybrid retrieval** over 10–15 realistic refinery documents
   (public API 570/510/653 inspection formats, PSU tender PDFs, MOC templates).
   Realistic corpus beats synthetic every time.
8. **Cable pull.**

**Scope freeze: end of week 3.** After that, only bug fixes and rehearsal.
Ship three legs deep, not six shallow.

---

## 15. Domain glossary

Use this vocabulary in prompts, templates, and the UI. It is what makes the system
look built *for MRPL* rather than generic.

| Term | Meaning |
|---|---|
| **P&ID** | Piping & Instrumentation Diagram |
| **MOC** | Management of Change — the approval workflow for any plant modification |
| **Approval Note** | Internal document seeking sign-off; our flagship deliverable |
| **NDT** | Non-Destructive Testing (UT, RT, MPI, DPI) |
| **UT thickness** | Ultrasonic wall-thickness reading; tracked against retirement limits |
| **API 570 / 510 / 653** | Inspection codes for piping / pressure vessels / storage tanks |
| **Equipment tag** | `10-P-101A`, `V-2301`, `PSV-4402` — **must survive retrieval intact** |
| **Work Permit** | Safety clearance document required before field work |
| **SOP** | Standard Operating Procedure |
| **Unit** | A plant section (CDU, VDU, FCC, HGU, SRU) |

---

## 16. Open questions

Do not silently invent answers. Flag these when they block you.

- [ ] Final measured VRAM for each model on the actual RTX 4050 — **blocking §5**
      Partially answered by legs 4-5: `writer` (qwen3:8b-q4_K_M), the largest
      model in the roster, peaks at **4.6 GB** with `num_ctx=8192` — inside the
      5.2 GB ceiling, so the roster's claims are plausible. `coder` and
      `driver` still need `ollama ps` figures. Also note `config/models.yaml`
      carried two `ref` values that no longer matched `ollama list`
      (`qwen3:4b-instruct-q4_K_M`, `qwen3-vl:4b-q4_K_M`); refs must be
      re-verified whenever the roster changes, since nothing catches a bad tag
      until the first call 404s.
- [ ] **Grounding on tabular data — new, found in leg 5.** Asked for the
      corrosion rate and remaining life of grid S7, `coder` read grid S5's 2019
      value out of the markdown table instead of S7's, inverted the remaining
      life subtraction, and then narrated the resulting negative number as
      "already exceeded by 3.05 years". The loop, the sandbox and the audit
      trail all behaved correctly — the model simply read the wrong row and was
      confident about it. Simpler framings (average wall loss across S6-S8) come
      out right, so this is task difficulty, not a broken pipeline.
      §12.7's UNKNOWN lever does not help here: the value *is* in the document,
      just on another row. **Keep this task out of the demo script until
      fixed** — a wrong thickness figure is the one error a refinery audience
      will catch.

      **Now measured** (`python -m finetune.grounding_eval <ref>`, 18 questions
      against the real corpus table, 10 lookup + 8 derived):

      | model | lookup | derived | total |
      |---|---|---|---|
      | `driver` qwen3:4b-instruct | **10/10 · 100%** | 4/8 · 50% | 77.8% |
      | `coder` qwen2.5-coder:7b | **10/10 · 100%** | 2/8 · 25% | 66.7% |

      The diagnosis was half wrong. Retrieval is not the failure — both models
      find the correct cell **every time**. Arithmetic is, and it is *worse on
      the larger model*: `coder` quoted `S6: 1.5, S7: 1.7, S8: 1.6` in its own
      source field and then answered 5.4 instead of 4.8.

      That changes the fix. Quoting the source row does not help if the row was
      already right. The lever is to stop the model doing arithmetic in its
      head and route every derivation through `py_sandbox`, which is exact and
      which leg 5 already built. Being pursued two ways: as a learned habit
      (`finetune/`, 42% of traces demonstrate it) and, if that does not
      transfer, as a `calc` tool (§6) that makes the arithmetic path
      non-optional rather than learned. The second is smaller and more
      reliable; prefer it if the adapter disappoints.

      **RESOLVED, 2026-08-26.** Both were built and measured, 8 trials each on
      the real document:

      | | without `calc` | with `calc` |
      |---|---|---|
      | base `qwen3:4b-instruct` | 1/8 correct | **7/8 correct** |
      | fine-tuned `sovereign-driver-v2` | 0/8 | **0/8** |

      `tools/calc.py` is the answer; the adapter is not. The formulas now live
      in code with an API 510 / ASME VIII citation and a printed substitution,
      so the model chooses *which* formula to apply and supplies the numbers,
      but cannot invent the algebra.

      Two findings worth carrying forward:

      1. **A sandbox guarantees the arithmetic, not the formula.** The
         fine-tuned model reliably called py_sandbox with running code that
         computed `5.70 / 0.2982` — the survey interval over the rate — and got
         19.11 years in 7 of 8 trials. Every layer worked except the algebra.
      2. **Fine-tuning on a fixed tool set makes a model brittle to new tools.**
         v2 was trained when three tools existed. Offered `calc`, it does not
         use it, and instead answers "UNKNOWN — the API 510 formula is not in
         the document". Since §6 still lists kb_search and fs_write as unbuilt,
         that rigidity is disqualifying on its own.
- [ ] Does Gemma 4 9B genuinely fit in 5.2 GB with vision enabled?
- [ ] Ollama vs llama.cpp for swap latency on this machine — measure both
- [ ] Real sample inspection report to build extraction schema against
- [ ] Venue power/thermals: does the laptop throttle after 20 min of sustained inference?
      **Test this. A thermal throttle mid-demo is a silent killer.**
- [ ] Does the venue laptop run the demo as Administrator? The drop-log monitor needs
      read access to `pfirewall.log`. Rehearse the exact launch sequence.
- [ ] **Router accuracy is 86.7% against a 90% gate — blocking §13.** The
      encoder decision this bullet used to track is resolved: `bge-small-en-v1.5`
      runs on CPU as the dense half of `HybridScorer` (w=0.8), which took
      held-out accuracy from the lexical-only plateau of 75.8% to 80.3%. The
      exemplar corpus was then widened 176 → 209 rows across the six weakest
      classes (2026-08-31), taking held-out to 86.7% and cutting misses from 13
      of 66 to 10 of 75 — no single confusion pair dominates; `code_write` ->
      `calc` is the only one that repeats. This is still not a tuning problem:
      the blend weight was chosen by leave-one-out CV on the training split
      alone and the held-out set was scored once. The remaining lever is more
      corpus, not a different scorer.
      `tests/test_router.py` records the gap as a **strict xfail**, so the day
      the gate is finally met the suite fails until the xfail is deleted.

---

## 17. Platform deviations — Windows demo machine

The charter was drafted against Linux. The demo machine is **Windows 11**, and the
only WSL distribution present is Docker Desktop's utility VM, which has no general
Linux userland. `nftables` therefore has nowhere to run. Recorded here rather than
silently substituted, per §16.

| §10 as drafted | Substitute in this repo | Note |
|---|---|---|
| `nftables` OUTPUT DROP | `Set-NetFirewallProfile -DefaultOutboundAction Block` | `sovereignty/firewall.ps1` |
| allow `lo` + local subnet | outbound allow rule for `LocalSubnet` | loopback is never filtered by Windows Firewall, so Ollama and Uvicorn are unaffected |
| drop-chain packet counters | Windows Firewall drop **log** (`pfirewall.log`) | no counter API exists; the log is richer — it names the destination IP and port of each dead packet |
| `sovereignty/verify.sh` | `sovereignty/verify.ps1` | same assertions |
| `nftables.rules` shown in UI | `firewall.ps1` shown in UI | still verbatim, still `/api/firewall/rules` |

Docker is present (29.6.1) so §9.3 needs no substitution — the sandbox works as
specified. Python is 3.13.9 rather than 3.11; nothing in the stack objects.

### Overcommitting VRAM does not fail on Windows — it gets slow

**Measured, 2026-08-26.** This changes how §4 should be read on this machine.

On Linux, exceeding VRAM raises a CUDA OOM and you find out immediately. On
Windows the display driver (WDDM) silently spills the excess into *shared
system memory* instead. Nothing raises. The work completes. It is simply
20-25x slower.

Caught while fine-tuning: a run sitting at 5896/6141 MiB took **844 s per
step** and reported a 53-hour ETA, with no error anywhere. Freeing 0.78 GB
(§16, the fp32 vocab recast) took the same run to **36 s per step** — a 23x
speedup from one allocation change, with identical output. The only symptom of
a VRAM fault on this platform is the clock.

Two consequences, and the second is the one that can cost a demo:

1. **Never trust "it ran" as evidence that it fit.** Log
   `torch.cuda.mem_get_info()` / `ollama ps` and compare against §4.1. A
   measurement that does not report headroom has not measured anything.
2. **A mid-demo VRAM squeeze will not crash — it will crawl.** §4.2.7 says to
   close Chrome, Discord and Electron apps before the demo. That instruction is
   now load-bearing rather than housekeeping: if one of them takes 800 MB while
   `writer` is resident, inference does not fail over to anything, it just
   starts taking minutes per answer on stage, and the cause is invisible.
   Check free VRAM as part of `sovereignty/verify.ps1`'s pre-demo pass.

**Demo consequence:** the monitor needs read access to `pfirewall.log`, which is
normally Administrator-only. Either run the app elevated or relax the ACL on that one
file. Decide before rehearsal, not on stage.

**Presentation framing:** this deviation is a strength, not an apology — the same
containment argument holds on the platform PSU desktops actually run. Say that out loud.

---

*Last updated for: SIH 2026 · PS 26117 · demo target RTX 4050 Laptop 6 GB · Windows 11.*
