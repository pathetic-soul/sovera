"""Model registry — loads, validates and hot-reloads config/models.yaml (§8.1).

CPU only, no GPU implication: this reads YAML, it never loads weights. The
`vram_gb` figures here are *claims* from the config, checked against the profile
budget so a model that cannot fit is rejected before it is ever routed to.
Actual residency is Ollama's problem (§4.2.1, OLLAMA_MAX_LOADED_MODELS=1).

Reload is atomic in the sense that matters live on stage: a malformed or
over-budget file leaves the previous registry serving traffic and reports why.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

ROOT = Path(__file__).resolve().parents[1]
MODELS_YAML = ROOT / "config" / "models.yaml"
PROFILES_DIR = ROOT / "config" / "profiles"


class ModelSpec(BaseModel):
    id: str
    backend: Literal["ollama", "llamacpp"]
    ref: str
    # Upstream checkpoint `ref` was built from. Empty for models we did not
    # build. §8.1 keeps every model name in this one file, and the finetune
    # pipeline reads it from here rather than hardcoding a repo id (§2, rule 2).
    hf_ref: str = ""
    device: Literal["gpu", "cpu"]
    vram_gb: float
    max_ctx: int
    capabilities: list[str] = Field(default_factory=list)
    routes: list[str] = Field(default_factory=list)
    priority: int = 1
    requires_tools: list[str] = Field(default_factory=list)


class Profile(BaseModel):
    id: str
    vram_budget_gb: float
    note: str = ""


class RegistryFile(BaseModel):
    version: int
    profile: str
    fallback: str
    models: list[ModelSpec]


class Rejection(BaseModel):
    model_id: str
    reason: str


class ReloadResult(BaseModel):
    ok: bool
    profile: str = ""
    budget_gb: float = 0.0
    accepted: list[str] = Field(default_factory=list)
    rejected: list[Rejection] = Field(default_factory=list)
    error: str | None = None


class RegistryError(RuntimeError):
    """Raised only on first load. Later failures keep the previous registry."""


class Registry:
    """The one permitted singleton (§11). Construct once, reload in place."""

    def __init__(self, path: Path = MODELS_YAML, profiles_dir: Path = PROFILES_DIR) -> None:
        self.path = Path(path)
        self.profiles_dir = Path(profiles_dir)
        self.profile = Profile(id="unloaded", vram_budget_gb=0.0)
        self.fallback = ""
        self.models: dict[str, ModelSpec] = {}
        self.rejected: list[Rejection] = []
        result = self.reload()
        if not result.ok:
            raise RegistryError(result.error or "registry failed to load")

    def reload(self) -> ReloadResult:
        """Re-read from disk. On any failure the live registry is left untouched."""
        try:
            raw = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            return ReloadResult(ok=False, error=f"{type(exc).__name__}: {exc}")

        try:
            parsed = RegistryFile.model_validate(raw)
            profile = Profile.model_validate(
                yaml.safe_load(
                    (self.profiles_dir / f"{parsed.profile}.yaml").read_text(encoding="utf-8")
                )
            )
        except (ValidationError, OSError, yaml.YAMLError) as exc:
            return ReloadResult(ok=False, error=f"{type(exc).__name__}: {exc}")

        accepted: dict[str, ModelSpec] = {}
        rejected: list[Rejection] = []
        for spec in parsed.models:
            if spec.id in accepted:
                rejected.append(Rejection(model_id=spec.id, reason="duplicate id"))
            elif spec.device == "gpu" and spec.vram_gb > profile.vram_budget_gb:
                rejected.append(
                    Rejection(
                        model_id=spec.id,
                        reason=(
                            f"claims {spec.vram_gb} GB, profile '{profile.id}' "
                            f"budget is {profile.vram_budget_gb} GB"
                        ),
                    )
                )
            else:
                accepted[spec.id] = spec

        if not accepted:
            return ReloadResult(ok=False, error="every model was rejected")
        if parsed.fallback not in accepted:
            return ReloadResult(
                ok=False, error=f"fallback '{parsed.fallback}' is not an accepted model"
            )

        self.profile, self.fallback, self.models, self.rejected = (
            profile,
            parsed.fallback,
            accepted,
            rejected,
        )
        return ReloadResult(
            ok=True,
            profile=profile.id,
            budget_gb=profile.vram_budget_gb,
            accepted=list(accepted),
            rejected=rejected,
        )

    def models_for(self, route: str) -> list[ModelSpec]:
        """Models declaring `route`, best first. Empty list means use `fallback`."""
        matches = [m for m in self.models.values() if route in m.routes]
        return sorted(matches, key=lambda m: (m.priority, m.vram_gb))

    def resolve(self, route: str) -> tuple[ModelSpec, bool]:
        """(model, used_fallback) for a route. Never raises once loaded."""
        candidates = self.models_for(route)
        if candidates:
            return candidates[0], False
        return self.models[self.fallback], True

    def snapshot(self) -> dict[str, Any]:
        """What the UI renders on the registry panel."""
        return {
            "profile": self.profile.id,
            "budget_gb": self.profile.vram_budget_gb,
            "fallback": self.fallback,
            "models": [m.model_dump() for m in self.models.values()],
            "rejected": [r.model_dump() for r in self.rejected],
        }
