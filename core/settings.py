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
