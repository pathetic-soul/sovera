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
    assert s.agent.max_steps == 64
    assert s.agent.max_tokens == 1_000_000
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
    assert load_settings(tmp_path / "absent.yaml").agent.max_steps == 64


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


def test_unknown_key_in_nested_model_is_rejected() -> None:
    """A typo like `max_step` for `max_steps` must fail loudly, not be
    silently dropped and leave the default in place unexplained."""
    from core.settings import AgentSettings

    with pytest.raises(ValidationError, match="max_step"):
        AgentSettings.model_validate({"max_step": 3})


def test_unknown_key_via_load_settings_is_rejected(tmp_path: Path) -> None:
    """The same typo, made in config/runtime.yaml, must abort the load
    rather than silently run on defaults nobody can explain."""
    bad = tmp_path / "runtime.yaml"
    bad.write_text("agent: {max_step: 3}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="max_step"):
        load_settings(bad)
