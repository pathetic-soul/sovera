# Sovereign Workbench — Remaining Work Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the three open modularity items (M5 runtime config, M2 API routers, M6 web static split) and raise held-out router accuracy toward the §13 gate, without moving any measured number that is not the target.

**Architecture:** Three structural refactors on the FastAPI app, executed leaf-first so each lands on stable ground: a settings module the routers will consume, then the router split, then the UI split that the static mount enables. Then a data-only change to the routing corpus. Every task is verified against a frozen baseline — the refactors must leave **184 passed / 1 xfailed** and **80.3% router accuracy** untouched; only Task 4 is permitted to move the accuracy, and only upward.

**Tech Stack:** Python 3.11+ (3.13.9 on the demo machine), FastAPI, Uvicorn, Pydantic v2, PyYAML, pytest, mypy strict. No new dependencies — every task uses what is already installed.

**Spec:** [`docs/MODULARITY.md`](../../MODULARITY.md) (items M2, M5, M6) and [`docs/ROADMAP.md`](../../ROADMAP.md) (P0, P1). Charter: [`AGENTS.md`](../../../AGENTS.md).

---

## Scope note — this is one plan covering two independent subsystems

Per the writing-plans scope check, the honest reading is that Tasks 1–3 (app
structure) and Task 4 (routing corpus) are independent: neither imports the
other's changes, and either order works. They are in one document because they
share one verification harness and one freeze date. **If you split execution
across two workers, Task 4 can run in parallel with Tasks 1–3.**

Phase 0 below is **not** a code plan. Rehearsal, soak testing and firewall
launch sequencing have no test cycle and no commit, so forcing them into a TDD
task shape would misrepresent them. They are a gate, listed as a checklist.

---

## Global Constraints

Copied verbatim from `AGENTS.md`. Every task's requirements implicitly include
this section.

- **§2.1** No process may make an outbound network call. No CDN, no webfont, no
  `<script src="https://…">`. All frontend assets vendored into `web/static/`.
- **§2.1** Model weights are pre-staged on disk. Runtime never downloads.
- **§2.3** Routing is deterministic. No LLM decides which model to use.
- **§2.3** Any file write outside `workspace/` is rejected by the filesystem jail.
- **§2.4** The agent proposes file writes and code execution; a human approves.
  Do not remove the gate to make the demo smoother.
- **§4.1** Design against **5.2 GB usable VRAM**, not 6.0. Every design states its VRAM cost.
- **§7** Forbidden dependencies: LangChain, LlamaIndex, CrewAI, AutoGen. **No new dependency of any kind in this plan.**
- **§7** Frontend: plain HTML + vanilla JS + vendored CSS. **No npm build step, no CDN.**
- **§8.1** `config/models.yaml` is the ONLY file that names a model.
- **§11** Type hints everywhere. `mypy --strict` clean. No bare `except`.
- **§11** Every module ≤400 lines. Split beyond that.
- **§11** Docstrings state **VRAM and latency implications** for anything touching the GPU.
- **§11** Comments explain *why*, not *what*. Deliberate shortcuts get a `# ponytail:` comment.
- **§11** Commit style: `feat(router): deterministic exemplar matching`
- **§14** Scope freeze end of week 3 (~2026-09-13). After that, bug fixes and rehearsal only.

### The frozen baseline

Every task in Tasks 1–3 must leave these **exactly** as they are:

| Measurement | Value | Command |
|---|---|---|
| Test suite | `184 passed, 1 xfailed` | `.venv\Scripts\python -m pytest -q` |
| Type check | `Success: no issues found` | `.venv\Scripts\python -m mypy` |
| Router accuracy | `80.3% (gate 90%)`, 13 misrouted | `.venv\Scripts\python gates.py --only router` |

Source-file count in the mypy line **will** move as modules split (45 today).
The test count and the accuracy must not. Accuracy moving by even a tenth of a
point during Tasks 1–3 means the split or the scorer changed — treat it as a bug
in the refactor and revert, do not accept it as an improvement.

---

## Phase 0 — Delivery-risk gate (not code; do this first)

`docs/STATUS.md` scores these above every item in this plan. R2 is the reason
Tasks 1–3 are sequenced after them: a refactor with no timed baseline cannot be
shown to have preserved the demo.

- [ ] **R1 — repository is committed and pushed to a private remote.** Two
      commits exist (`da78bfd`, `7e3e9da`, both titled "first commit") with 243
      files staged on top. Confirm they contain what you expect, commit the
      remainder, push.
- [ ] **R2 — golden path timed end to end, on a stopwatch.** Arm firewall →
      `verify.ps1` → leg 4 (inspection report → approval note `.docx`) → leg 5
      (coding task in the sandbox) → red button → verify chain → disarm.
      **Record the wall-clock number.** §13 gate is 6 minutes. This is the
      baseline every later change is re-verified against.
- [ ] **R4 — elevated launch sequence rehearsed** on the machine that will be
      used. Confirm the drop-log monitor can read
      `%windir%\system32\LogFiles\Firewall\pfirewall.log`. Settle the ACL
      question now, not on stage.
- [ ] **R5 — 25-minute sustained-load soak test**, GPU clocks logged. A thermal
      throttle mid-demo is silent and unrecoverable (§16).
- [ ] **R7 — sandbox image built:** `docker build -t sandbox-py:local sandbox\`
      **before arming the firewall.** It is the only step that needs a network
      and §2.1 forbids it at demo time.

**Do not start Task 1 until R2 has produced a recorded time.**

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `config/runtime.yaml` | Runtime knobs a judge may ask to change live | 1 |
| `core/settings.py` | Load + validate `runtime.yaml`; loopback invariant | 1 |
| `tests/test_settings.py` | Settings validation, especially the bind guard | 1 |
| `core/api/__init__.py` | Package marker + router list | 2 |
| `core/api/sovereignty.py` | `/api/firewall/rules`, `/api/egress-test`, `/ws/sovereignty` | 2 |
| `core/api/registry.py` | `/api/registry`, `/api/registry/reload` | 2 |
| `core/api/routing.py` | `/api/route` | 2 |
| `core/api/agent.py` | `/ws/agent`, `/api/tools`, `/api/backend` | 2 |
| `core/api/workspace.py` | `/api/artifact`, `/api/audit/verify` | 2 |
| `core/orchestrator.py` | Composition root only: lifespan, app factory, mounts | 2 |
| `web/static/app.css` | All styling, vendored | 3 |
| `web/static/js/util.js` | `$`, `esc` — shared by every panel | 3 |
| `web/static/js/sovereignty.js` | Drops, audit stream, red button, chain, rules | 3 |
| `web/static/js/router.js` | Route panel | 3 |
| `web/static/js/registry.js` | Registry panel | 3 |
| `web/static/js/agent.js` | Agent trace + human gate | 3 |
| `web/index.html` | Markup only, plus local `<link>` / `<script src>` | 3 |
| `tests/test_web.py` | Sovereignty assertion: zero external asset references | 3 |
| `config/routing_exemplars.jsonl` | +33 exemplars for the six weak classes | 4 |

---

## Task 1: Runtime settings (M5)

Moved ahead of M2 relative to `docs/MODULARITY.md`'s ordering. `settings.py` is a
leaf with no first-party imports, and three of the routers Task 2 creates will
consume it — building it first means Task 2's split lands on a stable module
rather than moving literals twice.

**Files:**
- Create: `config/runtime.yaml`
- Create: `core/settings.py`
- Create: `tests/test_settings.py`
- Modify: `core/agent.py:922-925` (the four module constants)
- Modify: `core/orchestrator.py:46` (`EGRESS_TARGET`), `:231` (`uvicorn.run` host/port)
- Modify: `pyproject.toml` (nothing — `pyyaml` and `pydantic` are already deps)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `core.settings.Settings` — pydantic model with `.agent`, `.server`, `.egress_probe`, `.sandbox`
  - `core.settings.AgentSettings(max_steps: int, max_tokens: int, observation_chars: int, temperature: float)`
  - `core.settings.ServerSettings(host: str, port: int)`
  - `core.settings.EgressProbeSettings(host: str, port: int)`
  - `core.settings.SandboxSettings(image: str, timeout_s: int)`
  - `core.settings.load_settings(path: Path = RUNTIME_YAML) -> Settings`
  - `core.settings.NotLoopback` — exception raised when `server.host` is not loopback

### The one non-obvious requirement

`server.host` in YAML is a sovereignty hazard, not a convenience. §2.1 and §4.3
both require loopback-only bind (`OLLAMA_HOST=127.0.0.1`, "never `0.0.0.0`"). A
config file that can bind `0.0.0.0` hands an editor the ability to break the
central claim of the project with one character. `ServerSettings` therefore
**validates** the host and refuses anything non-loopback. That validator is the
first test written.

- [ ] **Step 1: Write the failing test**

Create `tests/test_settings.py`:

```python
"""Runtime settings, and the one field that is a sovereignty control (§2.1, §11).

§11 says config over constants for anything a judge might ask to change live.
`server.host` is the exception that proves the rule: it is in the file for
completeness, but it is validated, because a YAML edit that binds 0.0.0.0 would
void the containment claim the whole project rests on.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pydantic import ValidationError

from core.settings import (
    ROOT,
    RUNTIME_YAML,
    NotLoopback,
    ServerSettings,
    Settings,
    load_settings,
)


def test_defaults_match_the_charter() -> None:
    """§8.4 caps and §12.8 temperature, as shipped."""
    s = Settings()
    assert s.agent.max_steps == 8
    assert s.agent.max_tokens == 20_000
    assert s.agent.observation_chars == 6000
    assert s.agent.temperature == 0.2
    assert s.server.port == 8080


def test_the_shipped_yaml_loads_and_matches_the_defaults() -> None:
    """config/runtime.yaml must not silently disagree with the code defaults —
    a judge reading one and running the other would see two different systems."""
    loaded = load_settings(RUNTIME_YAML)
    assert loaded.agent.model_dump() == Settings().agent.model_dump()
    assert loaded.server.model_dump() == Settings().server.model_dump()


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_hosts_are_accepted(host: str) -> None:
    assert ServerSettings(host=host).host == host


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "10.0.0.4", ""])
def test_non_loopback_bind_is_refused(host: str) -> None:
    """§2.1: loopback only, never 0.0.0.0. A config edit must not be able to
    expose the workbench on the network.

    The expected type is `ValidationError`, not `NotLoopback`: pydantic v2
    catches any `ValueError` a field validator raises and re-raises it wrapped.
    `NotLoopback` survives as the `__cause__` of the wrapped error, which is
    what the second assertion checks — the specific type is still available to
    a caller that wants it, it is just not what propagates.
    """
    with pytest.raises(ValidationError, match="loopback"):
        ServerSettings(host=host)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10"])
def test_the_refusal_carries_the_specific_exception(host: str) -> None:
    """A reader of the traceback should see NotLoopback, not a generic error."""
    with pytest.raises(ValidationError) as excinfo:
        ServerSettings(host=host)
    causes = [e.__cause__ for e in (excinfo.value,)] + [
        err.get("ctx", {}).get("error") for err in excinfo.value.errors()
    ]
    assert any(isinstance(c, NotLoopback) for c in causes if c is not None)


def test_a_malformed_file_names_the_problem(tmp_path: Path) -> None:
    """§11 forbids a bare except; a bad config must fail loudly and legibly."""
    bad = tmp_path / "runtime.yaml"
    bad.write_text("agent: {max_steps: not-a-number}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="max_steps"):
        load_settings(bad)


def test_a_missing_file_falls_back_to_charter_defaults(tmp_path: Path) -> None:
    """A fresh clone with no runtime.yaml still starts, on the §8.4 values."""
    assert load_settings(tmp_path / "absent.yaml").agent.max_steps == 8


def test_yaml_is_the_only_place_the_caps_are_written() -> None:
    """§11: config over constants. If someone re-hardcodes a cap in the agent
    loop, this catches it — the literal must not reappear in core/agent.py.

    Resolved from `core.settings.ROOT`, not from a relative path: pytest's
    working directory is not guaranteed to be the repo root, and a test that
    silently reads nothing passes for the wrong reason.
    """
    source = (ROOT / "core" / "agent.py").read_text(encoding="utf-8")
    assert "MAX_STEPS = 8" not in source
    assert "MAX_TOKENS = 20_000" not in source


def test_runtime_yaml_is_valid_yaml_and_versioned() -> None:
    raw = yaml.safe_load(RUNTIME_YAML.read_text(encoding="utf-8"))
    assert raw["version"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_settings.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.settings'`

- [ ] **Step 3: Write `config/runtime.yaml`**

```yaml
# Runtime knobs. AGENTS.md §11: "Config over constants. Anything a judge might
# ask you to change live goes in YAML."
#
# This is NOT config/models.yaml. Models are named there and only there (§8.1).
# Nothing in this file names a model, a model ref, or a route.
#
# Changing a value here takes effect on the next start. The agent caps are the
# ones worth demonstrating: "set max_steps to 3 and watch the loop stop early"
# is the same trick leg 3 plays with the registry, at zero extra cost.

version: 1

# §8.4. An 8k context window (§4.2.5) has no room to absorb a runaway loop, so
# all three of these are load-bearing rather than defensive.
agent:
  max_steps: 8            # hard cap on ReAct iterations
  max_tokens: 20000       # cumulative budget across the whole run
  observation_chars: 6000 # ~1500 tokens at ~4 chars/token, before re-entry
  temperature: 0.2        # §12.8: extraction and tool selection, not prose

# §2.1, §4.3: loopback only, never 0.0.0.0. `host` is validated in
# core/settings.py and a non-loopback value is REFUSED at load, because a
# config edit must not be able to expose the workbench on the network.
server:
  host: 127.0.0.1
  port: 8080

# §10.3, the red button. A judge asking "try a different host" is answered by
# editing this, not by editing Python.
egress_probe:
  host: api.openai.com
  port: 443

# §9.3. The image is built once at build time; the timeout bounds a runaway script.
sandbox:
  image: sandbox-py:local
  timeout_s: 60
```

- [ ] **Step 4: Write `core/settings.py`**

```python
"""Runtime configuration — the knobs, and the one that is a safety control.

AGENTS.md §11: "Config over constants. Anything a judge might ask you to change
live goes in YAML." This module is that sentence, implemented.

CPU only, 0 GB VRAM. One YAML parse at startup, then an immutable object.

WHAT DOES NOT LIVE HERE
-----------------------
Model names, refs and routes. §8.1 makes `config/models.yaml` the only file that
names a model and leg 3 demos exactly that on stage; a second file that could
also name one would weaken a graded requirement. Nothing below references a model.

Tool selection also stays out — see `tools/registry.py` for why making the
`requires_approval` set config-driven would weaken the §2.4 human gate.

WHY `server.host` IS VALIDATED AND NOTHING ELSE IS
--------------------------------------------------
Every other value here is a performance or budget knob: set `max_steps` to 3 and
the loop stops early, which is a demo, not a hazard. `server.host` is different.
§2.1 and §4.3 both require a loopback bind, and the entire containment claim
assumes nothing is listening on the network. A YAML file that can bind 0.0.0.0
hands one character the power to void the thesis, so it is refused at load
rather than trusted.
"""

from __future__ import annotations

import ipaddress
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_YAML = ROOT / "config" / "runtime.yaml"

# Hostnames that resolve to the loopback interface. Checked by name as well as
# by address because uvicorn accepts both and "localhost" is what a reader types.
_LOOPBACK_NAMES = {"localhost"}


class NotLoopback(ValueError):
    """`server.host` was not a loopback address. §2.1 makes this fatal."""


class AgentSettings(BaseModel):
    """§8.4 loop budget. Defaults are the charter's values, not placeholders."""

    model_config = ConfigDict(frozen=True)

    max_steps: int = Field(default=8, gt=0, le=64)
    max_tokens: int = Field(default=20_000, gt=0)
    observation_chars: int = Field(default=6000, gt=0)
    temperature: float = Field(default=0.2, ge=0.0, le=0.8)  # §12.8: never above 0.8


class ServerSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    host: str = "127.0.0.1"
    port: int = Field(default=8080, gt=0, lt=65536)

    @field_validator("host")
    @classmethod
    def _must_be_loopback(cls, value: str) -> str:
        """§2.1: loopback only, never 0.0.0.0."""
        if value in _LOOPBACK_NAMES:
            return value
        try:
            if ipaddress.ip_address(value).is_loopback:
                return value
        except ValueError as exc:
            raise NotLoopback(
                f"server.host {value!r} is not an address this app will bind. "
                f"AGENTS.md §2.1 requires loopback: use 127.0.0.1 or localhost."
            ) from exc
        raise NotLoopback(
            f"server.host {value!r} is not loopback. AGENTS.md §2.1 forbids "
            f"binding the workbench to a reachable interface — use 127.0.0.1."
        )


class EgressProbeSettings(BaseModel):
    """§10.3, the red button target. Deliberately configurable: "try a
    different host" is a question judges ask, and it should be a YAML edit."""

    model_config = ConfigDict(frozen=True)

    host: str = "api.openai.com"
    port: int = Field(default=443, gt=0, lt=65536)


class SandboxSettings(BaseModel):
    """§9.3. `image` is built at build time; nothing here pulls it."""

    model_config = ConfigDict(frozen=True)

    image: str = "sandbox-py:local"
    timeout_s: int = Field(default=60, gt=0, le=600)


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: int = 1
    agent: AgentSettings = Field(default_factory=AgentSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    egress_probe: EgressProbeSettings = Field(default_factory=EgressProbeSettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)


def load_settings(path: Path = RUNTIME_YAML) -> Settings:
    """Read runtime.yaml, or fall back to the charter defaults if it is absent.

    Absent is fine — a fresh clone must start. Present-but-wrong is not: a
    malformed file is a mistake someone made, and starting on silently different
    values is how a demo behaves in a way nobody can explain.
    """
    if not path.exists():
        return Settings()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read {path}: {type(exc).__name__}: {exc}") from exc
    try:
        return Settings.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"invalid {path.name}: {exc}") from exc
```

- [ ] **Step 5: Run the settings tests**

Run: `.venv\Scripts\python -m pytest tests/test_settings.py -q`
Expected: PASS for every test **except** `test_yaml_is_the_only_place_the_caps_are_written`, which still FAILS because `core/agent.py` has not been changed yet.

- [ ] **Step 6: Move the agent caps onto the Agent instance**

In `core/agent.py`, replace the module constants:

```python
MAX_STEPS = 8
MAX_TOKENS = 20_000
OBS_CHARS = 6000  # ~1500 tokens at ~4 chars/token (§8.4)
TEMPERATURE = 0.2  # §12.8: extraction and tool selection, not prose
```

with:

```python
# §8.4's budget now lives in config/runtime.yaml (§11: config over constants).
# The values are unchanged; what changed is that a judge can alter them without
# a code edit. `AgentSettings()` carries the charter defaults, so a caller that
# passes nothing behaves exactly as this module did before.
```

Change `Agent.__init__` to accept and store settings:

```python
    def __init__(
        self,
        backend: LLMBackend,
        router: Router,
        tools: dict[str, Tool],
        workspace: Path,
        audit: AuditLog,
        settings: AgentSettings | None = None,
    ) -> None:
        self.backend = backend
        self.router = router
        self.tools = tools
        self.workspace = workspace
        self.audit = audit
        # Default rather than required: tests/test_agent.py and finetune/gate.py
        # construct an Agent directly and must keep working unchanged.
        self.settings = settings or AgentSettings()
```

Add the import at the top of `core/agent.py`:

```python
from core.settings import AgentSettings
```

Then replace every use inside the class body:

| Old | New |
|---|---|
| `while step_n < MAX_STEPS:` | `while step_n < self.settings.max_steps:` |
| `if spent >= MAX_TOKENS:` | `if spent >= self.settings.max_tokens:` |
| `f"token cap {MAX_TOKENS} reached…"` | `f"token cap {self.settings.max_tokens} reached…"` |
| `f"step cap {MAX_STEPS} reached"` | `f"step cap {self.settings.max_steps} reached"` |
| `temperature=TEMPERATURE` (both call sites in `_decide`) | `temperature=self.settings.temperature` |

`_truncate` is a module function, not a method, so give it the limit explicitly:

```python
def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…[truncated {len(text) - limit} chars]"
```

and its one call site becomes:

```python
            step.observation = _truncate(_observe(result), self.settings.observation_chars)
```

- [ ] **Step 7: Fix the one test that imports `MAX_STEPS`**

`tests/test_agent.py:1` imports `MAX_STEPS` from `core.agent`. Change that import
to source the value from settings instead:

```python
from core.agent import Agent, AgentEvent, AgentStep
from core.settings import AgentSettings

MAX_STEPS = AgentSettings().max_steps
```

- [ ] **Step 8: Run the full suite**

Run: `.venv\Scripts\python -m pytest -q`
Expected: `199 passed, 1 xfailed` — 184 baseline + 15 new (`test_settings.py` parametrises to 15: 1 + 1 + 3 + 4 + 2 + 1 + 1 + 1 + 1)

- [ ] **Step 9: Wire settings into the orchestrator**

In `core/orchestrator.py`, delete the `EGRESS_TARGET` constant and load settings
in `lifespan`:

```python
    app.state.settings = load_settings()
```

immediately after `app.state.audit = audit`. Then:

```python
    app.state.agent = Agent(
        app.state.backend, app.state.router, tools, WORKSPACE, audit,
        settings=app.state.settings.agent,
    )
```

and in `egress_test`:

```python
@app.post("/api/egress-test")
def egress_test() -> dict[str, Any]:
    """Sync on purpose: attempt_egress blocks, FastAPI runs it in a threadpool."""
    probe = app.state.settings.egress_probe
    return attempt_egress(probe.host, probe.port, app.state.audit)
```

and at the bottom:

```python
if __name__ == "__main__":
    settings = load_settings()
    uvicorn.run(app, host=settings.server.host, port=settings.server.port, log_level="info")
```

Add `from core.settings import load_settings` to the imports.

- [ ] **Step 10: Verify the app still boots and nothing moved**

```powershell
.venv\Scripts\python -c "from core.orchestrator import app; print('routes', len(app.routes))"
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m mypy
.venv\Scripts\python gates.py --only router
```

Expected: `routes 16` · `199 passed, 1 xfailed` · mypy clean · **`80.3% (gate 90%)`, 13 misrouted**

- [ ] **Step 11: Commit**

```bash
git add config/runtime.yaml core/settings.py tests/test_settings.py core/agent.py core/orchestrator.py tests/test_agent.py
git commit -m "feat(settings): move the §8.4 loop budget and egress probe to config/runtime.yaml"
```

---

## Task 2: API routers (M2)

**Files:**
- Create: `core/api/__init__.py`, `core/api/sovereignty.py`, `core/api/registry.py`, `core/api/routing.py`, `core/api/agent.py`, `core/api/workspace.py`
- Modify: `core/orchestrator.py` (reduce to a composition root)
- Modify: `pyproject.toml` (add `core.api` to `[tool.setuptools] packages`)

**Interfaces:**
- Consumes: `core.settings.load_settings`, `core.settings.Settings` (Task 1).
- Produces: `core.api.ROUTERS: tuple[APIRouter, ...]` — the ordered list `core/orchestrator.py` iterates over with `app.include_router`. Each module exposes a module-level `router: APIRouter`.

### The mechanical change that makes this work

Endpoints currently close over the module-global `app` to reach `app.state`. In
an `APIRouter` they reach it through the request instead — `request.app.state`
for HTTP, `websocket.app.state` for websockets. That is the only behavioural
difference, and it is why state placement does not change: everything still
lives on `app.state`, set in one lifespan, exactly as §11 requires.

- [ ] **Step 1: Write the failing test**

Create `tests/test_api.py`:

```python
"""The API surface is assembled from routers, and the assembly is asserted.

A split into routers is only safe if the resulting app exposes the same paths.
This test is the net: it lists the routes the UI actually calls and fails if any
of them stops being served, which is the one way this refactor could break the
demo without breaking a unit test.
"""

from __future__ import annotations

from core.orchestrator import app

EXPECTED_PATHS = {
    "/",
    "/api/firewall/rules",
    "/api/egress-test",
    "/api/registry",
    "/api/registry/reload",
    "/api/route",
    "/api/tools",
    "/api/backend",
    "/api/artifact",
    "/api/audit/verify",
    "/ws/agent",
    "/ws/sovereignty",
}


def test_every_endpoint_the_ui_calls_is_served() -> None:
    served = {getattr(r, "path", "") for r in app.routes}
    missing = EXPECTED_PATHS - served
    assert not missing, f"routes lost in the router split: {sorted(missing)}"


def test_no_duplicate_paths() -> None:
    """Two routers registering the same path is a silent shadowing bug."""
    paths = [getattr(r, "path", "") for r in app.routes if getattr(r, "path", "") != ""]
    duplicates = {p for p in paths if paths.count(p) > 1}
    assert not duplicates, f"path registered twice: {sorted(duplicates)}"
```

- [ ] **Step 2: Run it against the current app to confirm it passes BEFORE the split**

Run: `.venv\Scripts\python -m pytest tests/test_api.py -q`
Expected: PASS. This is deliberate — the test must be green on the old structure
so that a failure after the split is unambiguously caused by the split.

- [ ] **Step 3: Commit the net before changing anything**

```bash
git add tests/test_api.py
git commit -m "test(api): assert the served route set before splitting the orchestrator"
```

- [ ] **Step 4: Create `core/api/sovereignty.py`**

```python
"""Sovereignty endpoints — the firewall ruleset, the red button, the drop stream.

AGENTS.md §10. CPU only, 0 GB VRAM.

These three are the graded claim (§1), so they are grouped together rather than
scattered: a judge asking "show me the containment code" should be shown one
file, and `/api/egress-test` should sit next to the docstring explaining why it
deliberately bypasses `net_guard`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse

from core.audit import AuditLog
from core.net_guard import attempt_egress
from sovereignty.monitor import DropWatcher

ROOT = Path(__file__).resolve().parents[2]
FIREWALL_RULES = ROOT / "sovereignty" / "firewall.ps1"

router = APIRouter()


@router.get("/api/firewall/rules", response_class=PlainTextResponse)
def firewall_rules() -> str:
    """§10.1 — the ruleset is displayed verbatim in the UI, not paraphrased."""
    return FIREWALL_RULES.read_text(encoding="utf-8")


@router.post("/api/egress-test")
def egress_test(request: Request) -> dict[str, Any]:
    """Sync on purpose: attempt_egress blocks, FastAPI runs it in a threadpool."""
    probe = request.app.state.settings.egress_probe
    return attempt_egress(probe.host, probe.port, request.app.state.audit)


@router.websocket("/ws/sovereignty")
async def sovereignty(ws: WebSocket) -> None:
    await ws.accept()
    watcher: DropWatcher = ws.app.state.watcher
    audit: AuditLog = ws.app.state.audit
    available, why = watcher.available()
    try:
        while True:
            new = watcher.poll()
            await ws.send_json(
                {
                    "monitor_ok": available,
                    "monitor_note": why,
                    "total_drops": watcher.total,
                    "new_drops": new,
                    "audit": [r.model_dump() for r in audit.tail(15)],
                }
            )
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return
```

- [ ] **Step 5: Create `core/api/registry.py`**

```python
"""Registry endpoints — the §14.3 live model addition.

AGENTS.md §8.1, §14.3. CPU only, 0 GB VRAM: this reads YAML, it never loads
weights. Reloading is the whole of leg 3 — editing config/models.yaml and
pressing this button is the entire change, with no code edit and no restart.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from core.registry import Registry, ReloadResult

router = APIRouter()


@router.get("/api/registry")
def registry_snapshot(request: Request) -> dict[str, Any]:
    registry: Registry = request.app.state.registry
    return registry.snapshot()


@router.post("/api/registry/reload")
def registry_reload(request: Request) -> dict[str, Any]:
    """§14.3 — the live model addition. Editing models.yaml is the whole change."""
    registry: Registry = request.app.state.registry
    result: ReloadResult = registry.reload()
    request.app.state.audit.append(
        "approval", {"event": "registry_reload", **result.model_dump()}
    )
    return {**result.model_dump(), "snapshot": registry.snapshot()}
```

- [ ] **Step 6: Create `core/api/routing.py`**

```python
"""The routing endpoint — deterministic, and it loads nothing.

AGENTS.md §2.3, §9.2. 0 GB VRAM, ~0.2 ms lexical / ~22 ms hybrid, all CPU.

No model is loaded and no LLM is consulted here, which is exactly what makes
§4.2.3's swap-masking possible: the decision and its rationale render while
Ollama is still loading the weights the decision selected.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from core.router import Router

router = APIRouter()


class RouteRequest(BaseModel):
    text: str
    attachments: list[str] = Field(default_factory=list)


@router.post("/api/route")
def route(req: RouteRequest, request: Request) -> dict[str, Any]:
    """Deterministic (§2.3): no model is loaded and no LLM is consulted here."""
    task_router: Router = request.app.state.router
    decision = task_router.route(req.text, req.attachments, request.app.state.resident)
    request.app.state.audit.append("model_call", {"phase": "route", **decision.model_dump()})
    request.app.state.resident = decision.model_id
    return decision.model_dump()
```

- [ ] **Step 7: Create `core/api/agent.py`**

```python
"""The agent run channel, the tool roster, and backend health (§14.4, §14.5).

GPU: this module holds no VRAM itself, but `/ws/agent` is what drives the
backend, so it is the endpoint that decides how much VRAM is in use. One model
per run — the route resolves once, up front, so a single run never swaps
(§4.2.1).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from core.agent import Agent, AgentStep
from tools.base import Tool
from tools.registry import tool_specs

router = APIRouter()


@router.get("/api/tools")
def tools_snapshot(request: Request) -> dict[str, Any]:
    """§14.4 — the roster, and which of them the human gate covers (§2.4)."""
    tools: dict[str, Tool] = request.app.state.tools
    return {"tools": tool_specs(tools)}


@router.get("/api/backend")
def backend_status(request: Request) -> dict[str, Any]:
    """Is Ollama up? Rendered in the UI so a dead daemon is caught before the
    demo, not as a stack trace mid-run."""
    ok, note = request.app.state.backend.available()
    return {"ok": ok, "note": note, "resident": request.app.state.resident}


@router.websocket("/ws/agent")
async def agent_ws(ws: WebSocket) -> None:
    """One run per connection (§14.4, §14.5).

    A websocket rather than SSE because the human gate is bidirectional: the
    loop stops at an approval_request and cannot continue until the reviewer
    answers on the same channel. Approval over a separate POST would need
    run-id bookkeeping to serve one operator on one laptop.
    """
    await ws.accept()
    agent: Agent = ws.app.state.agent

    async def approve(step: AgentStep) -> bool:
        # Safe to block here: the client has just been sent approval_request
        # and the run is suspended until it replies.
        reply = await ws.receive_json()
        return bool(reply.get("approve"))

    try:
        req = await ws.receive_json()
        task = str(req.get("text", "")).strip()
        attachments = [str(a) for a in (req.get("attachments") or [])]
        if not task:
            await ws.send_json({"type": "error", "data": {"error": "empty task"}})
            return

        async for event in agent.run(task, attachments, approve, ws.app.state.resident):
            if event.type == "route":
                ws.app.state.resident = event.data["model_id"]
            await ws.send_json(event.model_dump())
        await ws.send_json({"type": "done", "data": {}})
    except WebSocketDisconnect:
        return
```

- [ ] **Step 8: Create `core/api/workspace.py`**

```python
"""Reading back what the agent produced: artifacts and the audit chain.

AGENTS.md §2.2, §2.3. CPU only, 0 GB VRAM.

Grouped together because both are the same question from a reviewer's side —
"show me what it did, and prove the record was not edited afterwards."
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from tools.base import JailBreak, resolve_in_jail

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "workspace"

router = APIRouter()


@router.get("/api/artifact")
def artifact(path: str) -> FileResponse:
    """Download a deliverable. Jailed by the same function the tools use, so a
    crafted query string cannot read outside workspace/ (§2.3)."""
    try:
        target = resolve_in_jail(WORKSPACE, path)
    except JailBreak as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"no such artifact: {path}")
    return FileResponse(target, filename=target.name)


@router.get("/api/audit/verify")
def audit_verify(request: Request) -> dict[str, Any]:
    ok, broken = request.app.state.audit.verify()
    return {"ok": ok, "broken_at": broken}
```

- [ ] **Step 9: Create `core/api/__init__.py`**

```python
"""HTTP and websocket surface, one module per panel of the UI.

AGENTS.md §11 caps a module at 400 lines; `core/orchestrator.py` was heading
past that as legs 6-7 added endpoints, and every panel's endpoints shared one
file. Splitting by panel means leg 6 is a new module plus one line in ROUTERS,
not an edit to the file every other panel lives in.

State is unchanged: everything still hangs on `app.state`, set once in the
lifespan (§11 forbids global mutable state). Endpoints reach it through
`request.app.state` / `websocket.app.state` rather than closing over `app`.

Order matters only for readability — FastAPI matches on path, and
tests/test_api.py fails if two routers ever claim the same one.
"""

from fastapi import APIRouter

from core.api import agent, registry, routing, sovereignty, workspace

ROUTERS: tuple[APIRouter, ...] = (
    sovereignty.router,   # §10 — the graded claim
    registry.router,      # §14.3
    routing.router,       # §9.2
    agent.router,         # §14.4, §14.5
    workspace.router,     # §2.2, §2.3
)

__all__ = ["ROUTERS"]
```

- [ ] **Step 10: Reduce `core/orchestrator.py` to a composition root**

Replace the whole file with:

```python
"""FastAPI composition root — build the app, own the lifespan, mount the rest.

AGENTS.md §10, §14.1-5. Loopback bind only. Run: `python -m core.orchestrator`

Everything this module does is wiring. The endpoints live in `core/api/`, one
module per panel; the containment layers live in `sovereignty/`, `core/net_guard.py`
and `tools/py_sandbox.py`. What stays here is the single place where the audit
log, the registry, the router, the backend, the tools and the agent are
constructed — one lifespan, one set of objects on `app.state`, no globals (§11).

Legs 1-3 touch no weights: routing is deterministic and happens before any model
is loaded, which is what makes §4.2.3's swap-masking possible. Legs 4-5 are the
first VRAM spend; one model resident at a time (§4.2.1), and the agent resolves
its route once per run so a run never swaps mid-flight.
"""

from __future__ import annotations

import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backends.ollama_backend import OllamaBackend
from core.agent import Agent
from core.api import ROUTERS
from core.audit import AuditLog
from core.net_guard import install_guard
from core.registry import Registry
from core.router import Router
from core.settings import load_settings
from sovereignty.monitor import DropWatcher
from tools.base import Tool
from tools.registry import build_tools

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
INDEX = WEB / "index.html"
STATIC = WEB / "static"
WORKSPACE = ROOT / "workspace"
SEED_CORPUS = ROOT / "data" / "corpus" / "inbox"


def _seed_workspace() -> None:
    """workspace/ is gitignored (it is agent-writable), so a fresh clone has no
    documents to demo against. Copy the sample corpus in once, never overwrite:
    anything the operator edited on the machine wins."""
    inbox = WORKSPACE / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    for src in SEED_CORPUS.glob("*.md"):
        if not (inbox / src.name).exists():
            shutil.copy2(src, inbox / src.name)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    audit = AuditLog(WORKSPACE / ".audit" / "audit.jsonl", uuid.uuid4().hex[:12])
    app.state.audit = audit
    app.state.settings = load_settings()
    app.state.watcher = DropWatcher()
    app.state.registry = Registry()
    app.state.router = Router(app.state.registry)
    # Which model Ollama currently holds, for swap_required (§4.2.3). The agent
    # updates it after each run.
    app.state.resident = None

    _seed_workspace()
    app.state.backend = OllamaBackend(audit)
    tools: dict[str, Tool] = build_tools()
    app.state.tools = tools
    app.state.agent = Agent(
        app.state.backend, app.state.router, tools, WORKSPACE, audit,
        settings=app.state.settings.agent,
    )

    install_guard(audit)
    audit.append("approval", {"event": "session_start", "leg": "agent_and_sandbox"})
    yield


app = FastAPI(title="Sovereign Workbench", lifespan=lifespan)

for api_router in ROUTERS:
    app.include_router(api_router)

# Vendored locally, never a CDN (§2.1). StaticFiles reads off disk and opens no
# socket; tests/test_web.py asserts nothing in web/ references an external host.
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX)


if __name__ == "__main__":
    settings = load_settings()
    uvicorn.run(app, host=settings.server.host, port=settings.server.port, log_level="info")
```

**Note:** the `StaticFiles` mount references `web/static/`, which Task 3 creates.
Create the directory now so the mount does not fail:

```bash
mkdir -p web/static/js
```

- [ ] **Step 11: Register the new package**

In `pyproject.toml`, under `[tool.setuptools]`, change this line:

```toml
packages = ["core", "core.routing", "sovereignty", "backends", "tools", "finetune"]
```

to:

```toml
packages = ["core", "core.api", "core.routing", "sovereignty", "backends", "tools", "finetune"]
```

`[tool.mypy]`'s `files = ["core", ...]` names a directory, so it picks up
`core/api/` with no change.

- [ ] **Step 12: Run the full suite and confirm nothing moved**

```powershell
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m mypy
.venv\Scripts\python gates.py --only router
.venv\Scripts\python -c "from core.orchestrator import app; print('routes', len(app.routes))"
```

Expected: `201 passed, 1 xfailed` (199 + 2 from `test_api.py`) · mypy clean · **`80.3%`, 13 misrouted** ·
`routes` ≥ 16 (the static mount adds one). `tests/test_api.py` is the gate that
matters here — if a path went missing it names which one.

- [ ] **Step 13: Smoke-test the running app**

```powershell
.venv\Scripts\python -m core.orchestrator
```

In a second shell:

```powershell
curl.exe http://127.0.0.1:8080/api/registry
curl.exe http://127.0.0.1:8080/api/tools
curl.exe http://127.0.0.1:8080/api/audit/verify
curl.exe -X POST http://127.0.0.1:8080/api/route -H "Content-Type: application/json" -d "{\"text\":\"Draft an approval note for the PSV-4402 spring\",\"attachments\":[]}"
```

Expected: JSON from all four; the route call returns `"model_id": "writer"`.
Then open `http://127.0.0.1:8080/` and confirm all six panels render.

- [ ] **Step 14: Commit**

```bash
git add core/api core/orchestrator.py pyproject.toml
git commit -m "refactor(api): split the orchestrator into one router per UI panel"
```

---

## Task 3: Vendored static assets (M6)

**Files:**
- Create: `web/static/app.css`, `web/static/js/util.js`, `web/static/js/sovereignty.js`, `web/static/js/router.js`, `web/static/js/registry.js`, `web/static/js/agent.js`
- Create: `tests/test_web.py`
- Modify: `web/index.html` (markup only)

**Interfaces:**
- Consumes: the `/static` mount added in Task 2, Step 10.
- Produces: no Python interface. The JS files share globals in load order —
  `util.js` defines `$` and `esc`, every later file uses them. Plain `<script src>`
  tags, no modules, no bundler, no import map.

### The constraint, stated precisely because it will be argued about

§2.1 forbids a CDN, a webfont and an npm build step. It does **not** forbid more
than one local file. Everything below is served by `StaticFiles` off local disk,
loads over loopback, and adds no build step. The sovereignty claim is untouched,
and `tests/test_web.py` makes that assertion mechanical rather than a promise.

- [ ] **Step 1: Write the failing test**

Create `tests/test_web.py`:

```python
"""The UI is vendored. This test is what makes that a fact rather than a claim.

AGENTS.md §2.1: "All frontend assets are vendored locally. No `<script
src="https://…">`. No Google Fonts. No Tailwind CDN." A reviewer cannot verify
that by reading a slide, and a single pasted <script> tag would void it silently.
So it is asserted over every file the browser is served.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"

ASSETS = sorted(p for p in WEB.rglob("*") if p.suffix in {".html", ".css", ".js"})

# Anything that would make the browser open a socket to somewhere that is not us.
_EXTERNAL = re.compile(
    r"""(https?:)?//(?!\s)                 # scheme-relative or absolute
        (?!127\.0\.0\.1|localhost)         # except loopback
        [a-z0-9.-]+\.[a-z]{2,}""",
    re.I | re.X,
)
_IMPORT_FROM_URL = re.compile(r"""import\s+.*?from\s+['"]https?://""", re.I | re.S)


def test_there_are_assets_to_check() -> None:
    """Guard against the glob silently matching nothing and the suite passing."""
    assert len(ASSETS) >= 6, f"expected the split UI, found {[p.name for p in ASSETS]}"


@pytest.mark.parametrize("path", ASSETS, ids=lambda p: p.name)
def test_no_external_references(path: Path) -> None:
    """§2.1 — no CDN, no webfont, no remote script."""
    text = path.read_text(encoding="utf-8")
    # `//` inside a JS line comment is not a URL; strip the obvious case first.
    hits = [m.group(0) for m in _EXTERNAL.finditer(text)
            if not text[:m.start()].rstrip().endswith("//")]
    assert not hits, f"{path.name} references an external host: {hits}"


@pytest.mark.parametrize("path", ASSETS, ids=lambda p: p.name)
def test_no_remote_es_module_import(path: Path) -> None:
    assert not _IMPORT_FROM_URL.search(path.read_text(encoding="utf-8"))


def test_index_loads_every_script_locally() -> None:
    """Every <script src> and <link href> must be an absolute local path."""
    html = (WEB / "index.html").read_text(encoding="utf-8")
    srcs = re.findall(r"""<(?:script|link)[^>]*?(?:src|href)=["']([^"']+)["']""", html, re.I)
    assert srcs, "index.html loads no external files — did the split happen?"
    for src in srcs:
        assert src.startswith("/static/"), f"non-local asset reference: {src}"


def test_no_build_step_was_introduced() -> None:
    """§7: no npm build step. A package.json under web/ means one appeared."""
    assert not (WEB / "package.json").exists()
    assert not (WEB / "node_modules").exists()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_web.py -q`
Expected: FAIL on `test_there_are_assets_to_check` (only `index.html` exists) and
on `test_index_loads_every_script_locally` (no `<script src>` yet — everything is inline).

- [ ] **Step 3: Extract the stylesheet**

Move the entire contents of the `<style>` block in `web/index.html` (from
`:root {` through `a { color:#79c0ff; }`) into `web/static/app.css`, prefixed with:

```css
/* Sovereign Workbench panel styling. AGENTS.md §2.1: vendored, no webfont,
   no CDN. The monospace stack is whatever the machine already has — Consolas
   on the demo laptop — precisely so nothing is fetched. */
```

- [ ] **Step 4: Extract `web/static/js/util.js`**

```javascript
// Shared helpers. Loaded first; every later panel script depends on these two.
// Plain globals rather than ES modules: no bundler, no import map, no build
// step (AGENTS.md §7), and load order in index.html is the whole dependency graph.

const $ = id => document.getElementById(id);

// Escapes text interpolated into innerHTML. Audit payloads, router reasons and
// model output all pass through here — none of them are trusted markup.
const esc = s => String(s).replace(/[<&]/g, c => c === '<' ? '&lt;' : '&amp;');
```

- [ ] **Step 5: Extract `web/static/js/sovereignty.js`**

Move `render`, `connect`, `fire`, `checkChain`, the `connect()` and `checkChain()`
calls and the `fetch('/api/firewall/rules')` line, verbatim, minus the `const $`
and `const esc` definitions (now in `util.js`). Prefix:

```javascript
// The sovereignty panel: drop counter, drop table, audit stream, red button,
// chain verification, and the firewall ruleset shown verbatim (AGENTS.md §10).
// This is the graded claim; it reconnects on close because a blank panel mid-demo
// looks like a failure of the thesis rather than of a websocket.
```

- [ ] **Step 6: Extract `web/static/js/router.js`**

Move `doRoute` verbatim. Prefix:

```javascript
// The router panel (AGENTS.md §9.2). The `reason` string is rendered verbatim
// because it IS the explanation shown while a model swaps (§4.2.3) — a stall
// with a rationale on screen reads as an explanation, not a hang.
```

- [ ] **Step 7: Extract `web/static/js/registry.js`**

Move `renderRegistry`, `reloadRegistry` and the `fetch('/api/registry')` line
verbatim. Prefix:

```javascript
// The registry panel (AGENTS.md §8.1, §14.3). "Reload registry" is leg 3: edit
// config/models.yaml, press this, and a new model is routed to with no code
// change and no restart. Rejected models render with their reason, because an
// over-budget model being refused is as much the demo as one being accepted.
```

- [ ] **Step 8: Extract `web/static/js/agent.js`**

Move `agentWs`, the `fetch('/api/backend')` block, `addStep`, `runAgent` and
`onAgentEvent` verbatim. Prefix:

```javascript
// The agent panel (AGENTS.md §14.4, §14.5). One websocket per run, bidirectional
// because the §2.4 human gate suspends the loop until the reviewer answers on
// the same channel. Approve/Reject are wired to that reply.
```

- [ ] **Step 9: Rewrite `web/index.html` as markup only**

Replace the `<style>…</style>` block with:

```html
<link rel="stylesheet" href="/static/app.css">
```

and replace the entire `<script>…</script>` block at the bottom with:

```html
<!-- Vendored locally, loaded in dependency order. No CDN, no webfont, no npm
     build step (AGENTS.md §2.1, §7). tests/test_web.py asserts this. -->
<script src="/static/js/util.js"></script>
<script src="/static/js/sovereignty.js"></script>
<script src="/static/js/router.js"></script>
<script src="/static/js/registry.js"></script>
<script src="/static/js/agent.js"></script>
```

Everything between `<body>` and the script tags stays exactly as it is.

- [ ] **Step 10: Run the web tests**

Run: `.venv\Scripts\python -m pytest tests/test_web.py -q`
Expected: PASS — 17 tests (7 assets x 2 parametrised checks, plus 3 singles).

- [ ] **Step 11: Run the full suite**

```powershell
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m mypy
.venv\Scripts\python gates.py --only router
```

Expected: `218 passed, 1 xfailed` (201 + 17 from `test_web.py`) · mypy clean · **`80.3%`, 13 misrouted**

If your count differs, reconcile it before moving on — a test that vanished
during a refactor is exactly what these counts exist to catch.

- [ ] **Step 12: Verify the UI in a browser — this is the step that cannot be skipped**

```powershell
.venv\Scripts\python -m core.orchestrator
```

Open `http://127.0.0.1:8080/` and confirm, panel by panel:

- [ ] Egress panel: drop counter renders, monitor status line present
- [ ] Red button fires and reports `BLOCKED at …` (or `monitor unavailable` if the firewall is disarmed — that is expected off-demo)
- [ ] Audit panel streams records
- [ ] `verify chain` in the header reports `chain: intact`
- [ ] Router panel: type a prompt, press Route, the decision and score bars render
- [ ] Registry panel: models table populated, `↻ Reload registry` reports accepted/rejected
- [ ] Agent panel: backend status line renders; run a task and confirm the trace, the approval buttons and the artifact link all work
- [ ] Firewall ruleset panel shows `firewall.ps1` verbatim

Open devtools and confirm the console is clean — a `ReferenceError` here means a
function moved into a file that loads after its caller.

- [ ] **Step 13: Commit**

```bash
git add web tests/test_web.py
git commit -m "refactor(web): split the panel into vendored static assets, asserted offline"
```

---

## Task 4: Router corpus expansion (P1)

**Files:**
- Modify: `config/routing_exemplars.jsonl` (append 33 rows)

**Interfaces:**
- Consumes: nothing. Independent of Tasks 1–3; can run in parallel.
- Produces: no code interface. Changes one measured number.

### Correcting the diagnosis before acting on it

`docs/ROADMAP.md` and `README.md` both state the misses are "dominated by
subject matter overriding intent", citing `code_write` → `calc`. **Measured, that
is not what is happening.** All 13 misses land on 13 *distinct* confusion pairs;
none repeats. Grouped by the class that was wanted:

| Wanted | Misses | Got instead |
|---|---|---|
| `qa` | 3 | `approval_note`, `plan`, `drawing_qa` |
| `code_write` | 3 | `calc`, `scan_understanding`, `spreadsheet` |
| `plan` | 2 | `handwriting`, `approval_note` |
| `summarize` | 2 | `plan`, `calc` |
| `calc` | 2 | `qa`, `summarize` |
| `scan_understanding` | 1 | `qa` |

The lever is therefore **broad coverage for the six weak source classes**, not
targeted disambiguation of one pair. At ~16 exemplars per class the centroids are
thin everywhere, which is exactly the shape a scattered confusion matrix takes.

### The discipline that makes or breaks this task

The split is `i % 3` **within each class**, computed at import from the file. Adding
rows therefore **changes which rows are held out**. Two consequences, both binding:

1. **Never add an exemplar copied from, or paraphrased from, a held-out prompt
   that currently misses.** That is tuning against the held-out set. It would buy
   a prettier number and destroy its meaning, which is the one thing
   `core/routing/split.py`'s docstring exists to prevent.
2. **Measure once, at the end.** Do not add three rows, check, add three more.
   Write all 33, then run the gate a single time and record whatever it says —
   including if it goes down.

The 13 miss prompts are listed in the table above for *diagnosis only*. Write
new prompts about different equipment, different units and different phrasings.

- [ ] **Step 1: Record the pre-change baseline**

```powershell
.venv\Scripts\python -c "from core.routing import ROWS; from collections import Counter; print(len(ROWS)); print(sorted(Counter(t for _, t in ROWS).items()))"
.venv\Scripts\python gates.py --only router
```

Expected: `176` exemplars · `80.3% (gate 90%)`, 13 misrouted. Write both down.

- [ ] **Step 2: Append the new exemplars**

Append these 33 lines to `config/routing_exemplars.jsonl`. One JSON object per
line, no trailing comma, file ends with a newline.

```jsonl
{"prompt": "Who is authorised to countersign a hot work permit in the SRU during a shutdown", "task_type": "qa"}
{"prompt": "What retirement thickness does API 510 set for a class 1 pressure vessel shell", "task_type": "qa"}
{"prompt": "Does the vendor manual allow a weld repair on the crude charge pump casing", "task_type": "qa"}
{"prompt": "What is the minimum hold time for a hydrotest on a class 300 line", "task_type": "qa"}
{"prompt": "Which NDT method does the SOP require for a suspected lamination in a tank shell", "task_type": "qa"}
{"prompt": "How long is a confined space entry certificate valid for in the CDU", "task_type": "qa"}
{"prompt": "Write a python script that reads the UT survey csv and flags any grid below retirement", "task_type": "code_write"}
{"prompt": "Write a program to parse the inspection log and list every overdue PSV", "task_type": "code_write"}
{"prompt": "Give me code that converts the thickness readings into a corrosion rate table", "task_type": "code_write"}
{"prompt": "Produce a script that renames drawing files to match the equipment tag convention", "task_type": "code_write"}
{"prompt": "Code something that pulls the next due dates out of the inspection register", "task_type": "code_write"}
{"prompt": "Write a small program to check the MOC list for records missing a closure date", "task_type": "code_write"}
{"prompt": "Set out the steps to bring the FCC regenerator back online after the turnaround", "task_type": "plan"}
{"prompt": "What order should we tackle the pending API 653 tank inspections in this quarter", "task_type": "plan"}
{"prompt": "Work up an approach for clearing the backlog of overdue PSV pop tests", "task_type": "plan"}
{"prompt": "Map out how we schedule the HGU reformer catalyst change against the shutdown window", "task_type": "plan"}
{"prompt": "Propose the sequence for commissioning the new VDU overhead exchanger", "task_type": "plan"}
{"prompt": "Condense this thirty page turnaround report into the findings that need action", "task_type": "summarize"}
{"prompt": "Pull out the key points from the vendor inspection report on the SRU condenser", "task_type": "summarize"}
{"prompt": "Boil the CUI survey down to the circuits that need attention this year", "task_type": "summarize"}
{"prompt": "Shorten this MOC pack into a paragraph the shift superintendent can read", "task_type": "summarize"}
{"prompt": "Reduce the last turnaround findings to a list of open recommendations", "task_type": "summarize"}
{"prompt": "Compute the corrosion rate for grid S4 from the 2019 and 2024 readings", "task_type": "calc"}
{"prompt": "What is the remaining life if the wall is 7.8 mm and t-min is 6.4 mm at 0.21 mm per year", "task_type": "calc"}
{"prompt": "Figure out the MAWP for this shell at the measured thickness", "task_type": "calc"}
{"prompt": "How many years until this circuit reaches retirement thickness at the current rate", "task_type": "calc"}
{"prompt": "Calculate the required minimum thickness for a 12 inch line at 18 barg", "task_type": "calc"}
{"prompt": "Read the scanned inspection certificate and tell me the test date and the inspector", "task_type": "scan_understanding"}
{"prompt": "Extract the equipment tag and next due date from this scanned register page", "task_type": "scan_understanding"}
{"prompt": "Pull the vessel serial number off the scanned nameplate photograph", "task_type": "scan_understanding"}
{"prompt": "What thickness values are printed on this scanned UT report sheet", "task_type": "scan_understanding"}
{"prompt": "Transcribe the readings from the scanned survey form into a list", "task_type": "scan_understanding"}
{"prompt": "Read off the certificate number and validity dates from this scanned work permit", "task_type": "scan_understanding"}
```

- [ ] **Step 3: Verify the file is well-formed before measuring**

```powershell
.venv\Scripts\python -c "from core.routing import ROWS; from collections import Counter; print(len(ROWS)); print(sorted(Counter(t for _, t in ROWS).items()))"
```

Expected: `209` exemplars, with `qa` +6, `code_write` +6, `plan` +5,
`summarize` +5, `calc` +5, `scan_understanding` +6. A `json.JSONDecodeError`
here means a malformed line — fix it before going further.

- [ ] **Step 4: Confirm no new exemplar duplicates an existing one**

```powershell
.venv\Scripts\python -c "from core.routing import ROWS; ps=[p for p,_ in ROWS]; d=[p for p in set(ps) if ps.count(p)>1]; print('duplicates:', d or 'none')"
```

Expected: `duplicates: none`

- [ ] **Step 5: Measure once, and record whatever it says**

```powershell
.venv\Scripts\python -m pytest -q
.venv\Scripts\python gates.py --only router
```

The suite must stay green — `test_corpus_is_big_enough`,
`test_every_task_type_is_represented` and
`test_the_split_is_stratified_and_disjoint` all read the file and will catch a
structural mistake.

**`test_router_accuracy_has_not_regressed` asserts a floor of 0.80.** If accuracy
lands below that, the corpus made things worse: revert the append, and record the
negative result in `docs/STATUS.md` rather than trying variations until one
sticks. That is the same discipline that produced the QLoRA negative result the
project already publishes.

**If accuracy reaches ≥90%,** `test_router_accuracy_meets_the_gate` is a
`strict=True` xfail and the suite will FAIL with `XPASS(strict)`. That is
intentional and is the signal to delete the xfail decorator, which is the last
step of this task.

- [ ] **Step 6: Update the quoted numbers wherever they appear**

The accuracy figure is quoted in four places. All must move together or R6 (doc
drift) reopens:

- `README.md` — the scorer table and the "13 of 66 held-out prompts miss" line
- `docs/STATUS.md` §2 gate table and §4 risk R3
- `docs/ROADMAP.md` P1 item
- `AGENTS.md` §16 router bullet and §13 gate row
- `tests/test_router.py` — the xfail `reason` string, and the floor in
  `test_router_accuracy_has_not_regressed` if it improved (ratchet up, never down)

- [ ] **Step 7: Commit**

```bash
git add config/routing_exemplars.jsonl tests/test_router.py README.md docs AGENTS.md
git commit -m "feat(router): expand exemplars for the six weak classes, 176 -> 209"
```

---

## Closing verification

After every task, before the freeze date:

- [ ] `.venv\Scripts\python -m pytest -q` — green
- [ ] `.venv\Scripts\python -m mypy` — clean
- [ ] `.venv\Scripts\python gates.py --strict` — read every non-green line and consciously accept or fix it
- [ ] **Re-time the golden path.** Compare against the Phase 0 R2 number. A refactor that slowed the demo is a refactor that failed, and this is the only way to know.
- [ ] Update `docs/MODULARITY.md` status header (M2, M5, M6 → done)
- [ ] Update `docs/STATUS.md` gate table and test counts from `gates.py` output — never hand-typed

---

## What this plan deliberately does not do

| Not doing | Why |
|---|---|
| Build `ingest/` or `retrieval/` (legs 6–7) | `docs/ROADMAP.md` decided this. Five deep legs with a rehearsed demo beats seven shallow ones with an untimed one (§14). |
| Split `finetune/` despite 8 files over the 400-line cap | `docs/MODULARITY.md` M7. Build-time only, frozen after a negative result, ~4,500 lines of churn in a module already decided against shipping. |
| Move tool selection into YAML | Tools carry `requires_approval`, which §2.4 makes a compliance control. A YAML edit must not be able to switch off the human gate. |
| Add any dependency | §2.1 and behavioural rule 1. Every task above uses what is already installed. |
| Tune the blend weight against held-out | It would buy a prettier number and destroy its meaning. Task 4 adds data and measures once. |
| Use ES modules or a bundler for the web split | §7: no npm build step. Plain `<script src>` in load order is the whole dependency graph. |
