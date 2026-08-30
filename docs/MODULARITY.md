# Modularity plan — Sovereign Workbench

**SIH 2026 · PS 26117 (MRPL)** · drafted 2026-08-31 · **M1–M6 executed 2026-08-31**

> **Status: 6 of 6 items done.** M1 (one router split), M3 (tool registry),
> M4 (`core/routing/` package), M2 (API router split), M5
> (`config/runtime.yaml` + `core/settings.py`) and M6 (`web/static/` split) are
> all in the tree. M1/M3/M4 were verified against an unchanged baseline: **184
> passed / 1 xfailed**, **mypy strict clean over 45 files**, **router accuracy
> 80.3%, 13 held-out prompts misrouted** — identical before and after (see
> each item below for that historical snapshot). Current state, after all six:
> **221 passed / 1 xfailed**, **mypy strict clean over 52 files**, router
> accuracy since moved to **86.7%** by a later, separate corpus-expansion
> commit — not by this modularity work, which changed no scoring code.

`STATUS.md` says where the project stands, `ROADMAP.md` says what ships, this
file says how the code is *shaped* and what to change about that. It exists
because legs 6–8 add two new packages (`ingest/`, `retrieval/`) and at least
three new tools, and several files in the current tree would absorb that growth
by getting worse rather than by getting more parts.

## The rule this plan is subordinate to

> AGENTS.md §14: Everything else is secondary. If a task doesn't serve the
> golden demo path, defer it.

Refactoring does not serve the golden path. **Nothing in this file outranks R1
(commit the repo) or R2 (time the golden path).** It is sequenced to run *after*
those, and every item is scoped so it can be abandoned mid-way without leaving
the demo broken.

---

## 1. What is already well-factored

Worth stating, because the honest finding is that this codebase is in better
shape than most at this stage and the refactor surface is small.

| Property | Evidence |
|---|---|
| **Acyclic import graph** | `core/` depends on nothing first-party except itself; `tools/` depends only on `core.audit`; `backends/` only on `core.audit`. No cycles. |
| **One seam per swappable thing** | `LLMBackend` ABC (`backends/base.py`), `Tool` ABC (`tools/base.py`), `Scorer` Protocol (`core/routing/types.py`). Each has a real second implementation or a documented one. |
| **Build-time code is quarantined** | `finetune/` (~5,000 lines) is imported by nothing in `core/`, `tools/` or `backends/`. It lives behind the `[curate]` extra, so a demo install never pulls torch. This is what keeps §2.1 auditable. |
| **Single implementation of each safety invariant** | `resolve_in_jail` and `validate_args` exist exactly once (`tools/base.py`), and every tool calls them. A jail with four copies is a jail with four holes. |
| **Config-over-code where it is graded** | `config/models.yaml` is the only file naming a model, enforced by §8.1 and demonstrated by leg 3. |

Do not "improve" any of the above. They are the load-bearing structure.

---

## 2. Findings — ranked by (risk retired + leg-6/7 enablement) ÷ effort

### M1 · The router eval split is defined three times · ✅ **DONE**

`HELD_OUT` / `TRAIN` — the split that produces the single number a judge is most
likely to ask about — exists in three places:

| Location | Form |
|---|---|
| `tests/test_router.py:34-35` | `i % 3 == 0` / `i % 3 != 0`, the canonical copy |
| `finetune/tune_router.py:44-51` | `def split()`, a hand-reimplementation |
| `gates.py:` `gate_router()` | imports `TRAIN, HELD_OUT, held_out_accuracy` **from the test module** |

Two problems, and the second is worse than the first:

1. **A duplicated definition of a measurement split.** `tests/test_router.py:112`
   already contains a test whose entire job is to assert the two copies have not
   drifted — which is the codebase telling you, in its own words, that this
   should be one function.
2. **A layering inversion.** `gates.py` is production tooling (it is what
   `DEMO-DAY.md` tells you to run on stage morning). It imports from `tests/`,
   which has no `__init__.py` and is only importable because `gates.py` inserts
   ROOT on `sys.path`. The acceptance-gate runner should not stop working
   because someone reorganised the test suite.

**Done.** `core/routing/split.py` holds `split(rows)`, `TRAIN`, `HELD_OUT`,
`ROWS`, `ACCURACY_GATE` and `held_out_accuracy(clf)`. All three consumers import
it. `gates.py` no longer imports from `tests/`.

`test_split_matches_the_tuner` was deleted — with one definition there is nothing
left for it to catch — and replaced by `test_the_split_is_stratified_and_disjoint`,
which asserts the property the drift test was really protecting: every class
represented on both sides, nothing shared.

`held_out_accuracy` takes a `SupportsClassify` Protocol rather than a concrete
`Classifier`, so the tuner, the gate runner and the tests can measure any scorer
arrangement without this module importing a construction path.

Verified: accuracy 80.3%, 13 misrouted — unchanged.

---

### M2 · `core/orchestrator.py` is a composition root *and* every endpoint · ✅ **DONE**

230 lines today: lifespan wiring, sovereignty endpoints, registry endpoints, the
route endpoint, the agent websocket, the audit websocket, and artifact download.
It is under §11's 400-line cap, but legs 6 and 7 both add endpoints (ingest
upload, retrieval query, region highlight), and there is exactly one file for
them to land in.

**Fix:** split into `APIRouter` modules, leaving `orchestrator.py` as a pure
composition root (lifespan, app factory, `include_router`, `uvicorn.run`).

```
core/orchestrator.py       app factory + lifespan + mounts  (~70 lines)
core/api/sovereignty.py    /api/firewall/rules, /api/egress-test, /ws/sovereignty
core/api/registry.py       /api/registry, /api/registry/reload
core/api/routing.py        /api/route
core/api/agent.py          /ws/agent, /api/tools, /api/backend
core/api/artifacts.py      /api/artifact
core/api/audit.py          /api/audit/verify
```

State stays on `app.state` exactly as it does now (§11 forbids global mutable
state and this does not add any). Leg 6 becomes `core/api/ingest.py` plus one
`include_router` line, instead of an edit to the file every other panel lives in.

Effort: mechanical, ~1 hour. Risk: low, but it touches the file the whole demo
runs through — do it with the golden path already timed so you have a baseline
to re-verify against.

**Done.** `core/orchestrator.py` is now 104 lines: app factory, lifespan,
`StaticFiles` mount, and `include_router` over `core/api/ROUTERS`. The split
landed as five panel modules rather than the six sketched above — artifacts and
the audit endpoint share one file, `core/api/workspace.py`, because both answer
the same "what did the agent produce" question from a reviewer's side:

```
core/api/sovereignty.py    /api/firewall/rules, /api/egress-test, /ws/sovereignty
core/api/registry.py       /api/registry, /api/registry/reload
core/api/routing.py        /api/route
core/api/agent.py          /ws/agent, /api/tools, /api/backend
core/api/workspace.py      /api/artifact, /api/audit/verify
```

`tests/test_api.py` asserts the full route set (including the `/static` mount)
answers through the composed app rather than by walking `app.routes` for a
`.path` attribute, since an included router's routes are wrapped lazily in this
FastAPI version. State is still hung on `app.state`, set once in the lifespan —
no global mutable state was added.

---

### M3 · Tools are hardcoded in the lifespan; models are not · ✅ **DONE**

`core/orchestrator.py:78`:

```python
tools: dict[str, Tool] = {t.name: t for t in (FsRead(), Calc(), PySandbox(), DocWrite())}
```

§8.1 makes adding a **model** a config edit and a button click, and leg 3 demos
exactly that. Adding a **tool** is a source edit in the FastAPI lifespan. Legs 6
and 7 add `kb_search` and `fs_write` (both already named in §6), so this is
touched twice more.

**Done.** `tools/registry.py` holds `TOOL_CLASSES`, `build_tools()`,
`tool_specs()` and `gated_tools()`. It is *not* a plugin system — §11 says prefer
boring inspectable code, and a judge should be able to read the complete list of
things this agent can do in one screen. It is a list and three functions.

The fourth consumer turned out to matter more than expected: `finetune/gate.py`
— the gate that decides whether a model ships — was measuring an agent against a
hand-copied tool roster. Given that the v2 adapter's headline failure was
brittleness to an unfamiliar tool set (§16), measuring that with a hand-copy was
a real hazard, not just duplication.

Tool selection deliberately stayed in code rather than moving to YAML the way
models did. Tools carry executable behaviour and a `requires_approval` flag that
§2.4 makes a compliance control; making that set config-driven would let a YAML
edit switch off the human gate. Models are the graded config surface (§8.1);
tools are not. That reasoning is recorded in the module docstring so it is not
relitigated later.

---

### M4 · `core/classifier.py` is 384 lines against a 400-line cap · ✅ **DONE**

§11: *"Every module ≤400 lines. Split beyond that."* This file is 16 lines from
violating the project's own convention, and it currently holds five distinct
concerns: feature extraction, the §9.2 hard overrides, three scorer
implementations, score standardisation, and the `Classifier` facade.

**Fix:** promote to a package alongside M1's `split.py`:

```
core/routing/__init__.py     re-exports Classifier, TaskType, Modality
core/routing/features.py     _features, _l2, _standardise, detect_modality
core/routing/overrides.py    allowed_routes + the regexes (§9.2 hard rules)
core/routing/scorers.py      ExemplarScorer, DenseScorer, HybridScorer, Scorer
core/routing/classifier.py   Classifier, ROUTER_WEIGHT, load_exemplars
core/routing/split.py        M1
```

**Done, and the shim was skipped.** All five consumers (`core/router.py`,
`gates.py`, `tests/test_router.py`, `finetune/tune_router.py` ×2) are in-repo and
were updated in the same change, so a compatibility shim would have been dead on
arrival. `core/classifier.py` is deleted.

`types.py` was added beyond the plan: `TaskType`, `Modality`, `Classification`
and the `Scorer` protocol now live in a module that imports nothing heavier than
pydantic, so `core/router.py` can name a task type without pulling in an
encoder.

The docstrings in this file are unusually good — they carry the *measurements*
behind each design choice (why lexical, why the hybrid, why standardisation).
**Move them with their code, do not summarise them.** They are a large part of
why this repo reads as engineered rather than generated.

Verified: 221 passed (184 at the time), mypy clean, accuracy 80.3% / 13 misrouted — unchanged.
Largest file in the package is now `scorers.py` at ~190 lines, comfortably inside
the §11 cap with room for leg 6-7 routes.

---

### M5 · Runtime constants that §11 says belong in YAML · ✅ **DONE**

> §11: *"Config over constants. Anything a judge might ask you to change live
> goes in YAML."*

Currently hardcoded across modules:

| Constant | Location | Would a judge ask? |
|---|---|---|
| `MAX_STEPS = 8`, `MAX_TOKENS = 20_000` | `core/agent.py:922-923` | **Yes** — §8.4 caps are a headline design claim |
| `OBS_CHARS = 6000`, `TEMPERATURE = 0.2` | `core/agent.py:924-925` | Yes — §12.8 is a stated policy |
| `TIMEOUT_S = 60`, `IMAGE` | `tools/py_sandbox.py` | Plausibly |
| `EGRESS_TARGET`, port `8080` | `core/orchestrator.py:46,231` | Yes — "try a different host" is an obvious challenge |
| `MAX_CHARS = 6000` | `tools/fs_read.py` | No |

**Fix:** `config/runtime.yaml` + `core/settings.py` (a Pydantic `Settings` model,
loaded once, hung on `app.state`). Move the first four rows. Leave per-tool
implementation details alone — the test is "would a judge ask you to change this
live", not "is it a number".

This is a *demo asset*, not housekeeping: "change the step cap to 3 and re-run,
no restart" is the same trick leg 3 already plays with the registry, and it costs
one YAML file.

Effort: ~1 hour. Risk: low, but it touches the agent loop — do it after M2 so the
golden path has been re-timed once already.

**Done.** `config/runtime.yaml` + `core/settings.py` (a frozen, `extra="forbid"`
Pydantic `Settings` model) now own the first three rows — `agent.max_steps`,
`agent.max_tokens`, `agent.observation_chars`, `agent.temperature` — plus
`server.host`/`server.port` and `egress_probe.host`/`egress_probe.port`. `host`
is the one field validated at load (§2.1: loopback only, refused otherwise);
everything else is a demo knob, deliberately unvalidated so "change it live" is
a YAML edit. A speculative `sandbox:` block (image, timeout_s) was drafted and
then deleted rather than wired: nothing in `tools/py_sandbox.py` reads it, so
shipping it would have been a config key that silently does nothing — worse
than no key. `tools/py_sandbox.py`'s `IMAGE`/`TIMEOUT_S` and `tools/fs_read.py`'s
`MAX_CHARS` stay hardcoded, per the "would a judge ask to change this live"
test above.

---

### M6 · `web/index.html` is 342 lines of HTML + CSS + JS in one file · ✅ **DONE**

Six panels' worth of markup, an inline `<style>` block, and an inline `<script>`
holding the sovereignty websocket, the router panel, the registry panel and the
agent loop client. Legs 6 and 7 add two more panels.

**This is the item most likely to be argued about, so state the constraint
precisely.** §2.1 forbids a CDN, a webfont and an npm build step. It does **not**
forbid more than one local file. Splitting into `web/index.html` +
`web/static/app.css` + `web/static/js/{sovereignty,router,registry,agent}.js`,
served by `StaticFiles` off local disk, keeps every asset vendored and adds no
build step. The sovereignty claim is untouched.

**Fix:** mount `app.mount("/static", StaticFiles(directory=ROOT / "web" / "static"))`
and split by panel, using plain `<script src="/static/js/....js">` tags — no
modules, no bundler, no import maps.

Effort: ~1 hour, mostly cut-and-paste. Risk: low but *visible* — this is the
surface the judges look at. Re-open every panel after the change.

**Done.** `app.mount("/static", StaticFiles(directory=WEB / "static"))` in
`core/orchestrator.py`, and `web/index.html` now loads
`web/static/app.css` + `web/static/js/{util,router,registry,sovereignty,agent}.js`
via plain `<script src="/static/...">` tags — no modules, no bundler. Every
asset stays vendored: `tests/test_web.py` scans every `.html`/`.css`/`.js` file
for an external reference (a CDN, a webfont, a scheme-relative `//host`, and
the multi-slash and backslash bypass forms of the same) and fails the suite if
one appears, plus asserts no `package.json`/`node_modules` ever lands under
`web/`. No panel logic changed — this was a pure extraction.

---

### M7 · `finetune/` violates the 400-line cap eight times · **recommend: do not fix**

| File | Lines | Cap |
|---|---|---|
| `finetune/traceset.py` | 922 | 2.3× |
| `finetune/kaggle_kernel.py` | 764 | 1.9× |
| `finetune/benchmark.py` | 518 | 1.3× |
| `finetune/dataset.py` | 515 | 1.3× |
| `finetune/runstats.py` | 494 | 1.2× |
| `finetune/qlora.py` | 469 | 1.2× |
| `finetune/corpus_filings.py` | 449 | 1.1× |
| `finetune/corpus_inspection.py` | 404 | 1.0× |

Real violations of a real convention. **Leave them.** The reasoning:

- This is build-time-only code (§3 override) that runs before the firewall is
  armed and touches nothing at demo time.
- The line it protects — that `core/`, `tools/` and `backends/` never import it —
  already holds and is the boundary that matters for §2.1.
- The fine-tuning track produced a **negative result** and was closed
  (`ROADMAP.md`, "Explicitly not doing"). This is frozen code.
- ~4,500 lines of churn in a module nobody will run again, two weeks before
  scope freeze, on a project whose top risk is an untimed demo.

Record it here as a known, accepted deviation rather than pretending it is not
one. If a judge asks: *"build-time code, deliberately outside the module cap that
governs the demo path, and it is the module we already decided not to ship."*

---

### M8 · Charter modules that do not exist yet

`AGENTS.md §6` lists these; they are absent. Not defects — they are legs 6–7, and
`ROADMAP.md` already recommends not starting leg 7.

| Missing | Leg | Recommendation |
|---|---|---|
| `ingest/` (pipeline, layout, extractors) | 6–7 | Defer. Gate on measuring `vision` VRAM first. |
| `retrieval/` (index, hybrid, chunker) | 7 | **Do not start** — `ROADMAP.md` §Next is right. |
| `core/planner.py` | — | Not needed. The agent loop plans inline and it works. |
| `backends/llamacpp_backend.py` | — | Only if Ollama swap latency proves unacceptable (§16, unmeasured). |
| `tools/fs_write.py`, `tools/kb_search.py` | 6–7 | M3 makes these additive when they land. |

If any of these do get built, M2 and M3 are what make them one-file additions
instead of edits to shared files. **That is the argument for doing M2 and M3 at
all** — not tidiness.

---

## 3. Sequencing

Strictly ordered. Each step is committable on its own and leaves the demo working.

| # | Step | Gate before starting | Effort |
|---|---|---|---|
| 0 | **`git commit` + push to a private remote** | — | minutes |
| 1 | **Dress-rehearse and time the golden path** (R2) | step 0 done | ~1 h |
| 2 | ✅ M1 — one router split | `pytest` green, accuracy 80.3% recorded as the baseline | done |
| 3 | ✅ M3 — tool registry | — | done |
| 4 | ✅ M4 — `core/routing/` package | — | done |
| 5 | ✅ M2 — API routers | golden path timed once (step 1) | done |
| 6 | ✅ M5 — `config/runtime.yaml` | — | done |
| 7 | ✅ M6 — `web/static/` split | — | done |
| 8 | **Re-time the golden path; `python gates.py --strict`** | steps 2–7 | ~1 h |

Total ≈ 7 hours of refactor across steps 2–7, bracketed by two timed runs of the
demo. If the schedule tightens, **cut from the bottom**: steps 2–4 carry almost
all of the leg-6/7 enablement and almost none of the risk; step 5 onward touches
the demo path.

## 4. Verification — the same rule as everywhere else in this repo

A refactor is only correct if the numbers do not move. After each step:

```powershell
.venv\Scripts\python -m pytest -q      # 221 passed, 1 xfailed (184 at M1/M3/M4 time) — unchanged by this refactor
.venv\Scripts\python -m mypy           # strict, clean, 52 files (45 at M1/M3/M4 time)
.venv\Scripts\python gates.py          # router accuracy MUST still read 80.3% at M1/M3/M4 time;
                                        # 86.7% is a later, separate corpus-expansion commit
```

The file counts move as modules are split (38 -> 45 across M1/M3/M4, 45 -> 52
across M2/M5/M6); the *test count* and the *accuracy*, measured immediately
before and after each refactor step, must not.

Router accuracy shifting by even a tenth of a point after M1 or M4 means the
split or the scorer changed, not that it improved. Treat any movement as a bug
in the refactor and revert.

## 5. What this plan deliberately does not do

| Not doing | Why |
|---|---|
| Introduce a DI container, a service locator, or a plugin loader | §11: boring inspectable code. Judges read source. A registry dict is enough. |
| Move tool selection into YAML | Weakens the §2.4 approval gate for no graded benefit. Models are the graded config surface, not tools. |
| Add an abstraction over `AuditLog` | It has one implementation and §2.2 says it must stay the only path. An interface here invites a second one. |
| Split `finetune/` | M7. Frozen, build-time, negative result. |
| Add any dependency | §2.1 and rule 1. Every item above uses what is already installed. |
| Start `retrieval/` | `ROADMAP.md` already decided this. Five deep legs beat seven shallow ones. |
