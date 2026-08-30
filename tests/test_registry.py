"""Registry validation, VRAM budget enforcement, and hot reload (§8.1)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.registry import PROFILES_DIR, Registry, RegistryError

GOOD = {
    "version": 1,
    "profile": "6gb",
    "fallback": "driver",
    "models": [
        {
            "id": "driver", "backend": "ollama", "ref": "qwen3:4b-instruct-q4_K_M",
            "device": "gpu", "vram_gb": 2.6, "max_ctx": 8192,
            "capabilities": ["reasoning"], "routes": ["plan", "qa"], "priority": 1,
        }
    ],
}


def write(path: Path, doc: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


def test_shipped_config_loads() -> None:
    reg = Registry()
    assert reg.profile.id == "6gb"
    assert reg.profile.vram_budget_gb == 5.2
    assert {"driver", "writer", "coder", "vision"} <= set(reg.models)
    assert reg.rejected == []


def test_every_shipped_model_fits_the_budget() -> None:
    """§4.2.1: one LLM resident at a time, so each must fit alone in 5.2 GB."""
    reg = Registry()
    for spec in reg.models.values():
        assert spec.vram_gb <= reg.profile.vram_budget_gb, spec.id
        assert spec.max_ctx >= 8192, f"{spec.id} is below the 8k context floor (§5)"


def test_models_for_and_resolve() -> None:
    reg = Registry()
    assert [m.id for m in reg.models_for("code_write")] == ["coder"]
    assert reg.models_for("nonexistent_route") == []
    spec, fell_back = reg.resolve("nonexistent_route")
    assert (spec.id, fell_back) == ("driver", True)


def test_oversize_model_is_rejected_with_a_reason(tmp_path: Path) -> None:
    doc = {**GOOD, "models": [*GOOD["models"], {  # type: ignore[misc]
        "id": "too_big", "backend": "ollama", "ref": "llama3.3:70b-q4_K_M",
        "device": "gpu", "vram_gb": 42.0, "max_ctx": 8192,
        "capabilities": ["reasoning"], "routes": ["qa"], "priority": 1,
    }]}
    reg = Registry(write(tmp_path / "m.yaml", doc), PROFILES_DIR)
    assert "too_big" not in reg.models
    assert reg.rejected[0].model_id == "too_big"
    assert "42.0 GB" in reg.rejected[0].reason and "5.2 GB" in reg.rejected[0].reason


def test_cpu_model_ignores_the_vram_budget(tmp_path: Path) -> None:
    """embed/rerank/ocr run on CPU (§4.2.2); the GPU budget does not apply."""
    doc = {**GOOD, "models": [*GOOD["models"], {  # type: ignore[misc]
        "id": "embed", "backend": "llamacpp", "ref": "bge-m3-onnx-int8",
        "device": "cpu", "vram_gb": 0.0, "max_ctx": 8192,
        "capabilities": ["embedding"], "routes": [], "priority": 1,
    }]}
    reg = Registry(write(tmp_path / "m.yaml", doc), PROFILES_DIR)
    assert "embed" in reg.models


def test_live_model_addition_needs_no_code_change(tmp_path: Path) -> None:
    """§14.3: paste a block into the yaml, click Reload, route to it."""
    path = write(tmp_path / "m.yaml", GOOD)
    reg = Registry(path, PROFILES_DIR)
    assert reg.models_for("handwriting") == []

    doc = {**GOOD, "models": [*GOOD["models"], {  # type: ignore[misc]
        "id": "vision", "backend": "ollama", "ref": "qwen3-vl:4b-q4_K_M",
        "device": "gpu", "vram_gb": 3.2, "max_ctx": 8192,
        "capabilities": ["vision"], "routes": ["handwriting"], "priority": 1,
    }]}
    write(path, doc)
    result = reg.reload()
    assert result.ok
    assert [m.id for m in reg.models_for("handwriting")] == ["vision"]


def test_broken_yaml_keeps_the_previous_registry(tmp_path: Path) -> None:
    """A typo pasted on stage must not brick the app."""
    path = write(tmp_path / "m.yaml", GOOD)
    reg = Registry(path, PROFILES_DIR)
    path.write_text("models: [ this is not: valid: yaml", encoding="utf-8")

    result = reg.reload()
    assert not result.ok and result.error
    assert "driver" in reg.models  # still serving


def test_unknown_fallback_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path / "m.yaml", {**GOOD, "fallback": "ghost"})
    with pytest.raises(RegistryError):
        Registry(path, PROFILES_DIR)


def test_priority_orders_candidates(tmp_path: Path) -> None:
    doc = {**GOOD, "models": [*GOOD["models"], {  # type: ignore[misc]
        "id": "preferred", "backend": "ollama", "ref": "x:q4", "device": "gpu",
        "vram_gb": 4.0, "max_ctx": 8192, "capabilities": [], "routes": ["qa"],
        "priority": 0,
    }]}
    reg = Registry(write(tmp_path / "m.yaml", doc), PROFILES_DIR)
    assert [m.id for m in reg.models_for("qa")] == ["preferred", "driver"]
