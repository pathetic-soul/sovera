"""Deterministic router — task in, RouteDecision out (AGENTS.md §8.2, §9.2).

No LLM is consulted. 0 GB VRAM, ~0.2 ms per decision: the router runs before
any weights are touched, which is exactly what makes §4.2.3's swap-masking work
— we can render the decision and its rationale while Ollama is still loading.
"""

from __future__ import annotations

from pydantic import BaseModel

from core.routing import Classifier, Modality, TaskType
from core.registry import Registry

# A VLM spends roughly this much context on one page-sized image. Rough on
# purpose: it only has to be good enough to pick a model and warn about
# truncation, and §9.4's reranker does the real budgeting later.
CTX_PER_IMAGE = 1024
CTX_PER_PDF_PAGE = 768
CTX_OVERHEAD = 512  # system prompt + tool schemas (§12.3 keeps this small)


class RouteDecision(BaseModel):
    task_type: TaskType
    modality: Modality
    model_id: str
    reason: str
    scores: dict[str, float]
    est_ctx: int
    swap_required: bool


def estimate_ctx(text: str, attachments: list[str] | None) -> int:
    """~4 chars per token, plus a flat allowance per attachment."""
    tokens = len(text) // 4 + CTX_OVERHEAD
    for name in attachments or []:
        tokens += CTX_PER_PDF_PAGE if name.lower().endswith(".pdf") else CTX_PER_IMAGE
    return tokens


class Router:
    def __init__(self, registry: Registry, classifier: Classifier | None = None) -> None:
        self.registry = registry
        self.classifier = classifier or Classifier()

    def route(
        self,
        text: str,
        attachments: list[str] | None = None,
        resident: str | None = None,
    ) -> RouteDecision:
        cls = self.classifier.classify(text, attachments)
        spec, used_fallback = self.registry.resolve(cls.task_type)
        est_ctx = estimate_ctx(text, attachments)

        notes: list[str] = []
        if cls.override:
            notes.append(cls.override)
        if used_fallback:
            notes.append(f"no model declares '{cls.task_type}', fell back to {spec.id}")
        if est_ctx > spec.max_ctx:
            notes.append(
                f"est_ctx exceeds max_ctx {spec.max_ctx}, retrieval must truncate (AGENTS.md 4.2.5)"
            )
        if spec.requires_tools:
            notes.append(f"requires {', '.join(spec.requires_tools)}")

        return RouteDecision(
            task_type=cls.task_type,
            modality=cls.modality,
            model_id=spec.id,
            reason=self._reason(cls.task_type, cls.scores, cls.modality, est_ctx, spec.id, resident, notes),
            scores=cls.scores,
            est_ctx=est_ctx,
            swap_required=resident is not None and resident != spec.id,
        )

    @staticmethod
    def _reason(
        task: str,
        scores: dict[str, float],
        modality: str,
        est_ctx: int,
        model_id: str,
        resident: str | None,
        notes: list[str],
    ) -> str:
        """User-facing (§8.2). Rendered verbatim in the UI, so it has to read
        like a sentence an engineer would accept, not a debug dump."""
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        head = f"{task} ({ranked[0][1]:.2f}"
        if len(ranked) > 1:
            head += f" vs {ranked[1][0]} {ranked[1][1]:.2f}"
        head += ")"

        mod = "image absent" if modality == "text" else f"{modality} attached"
        if resident is None:
            target = f"-> {model_id} [cold start]"
        elif resident == model_id:
            target = f"-> {model_id} [resident]"
        else:
            target = f"-> {model_id} [swap from {resident}]"

        parts = [head, mod, f"{est_ctx / 1000:.1f}k ctx", target]
        if notes:
            parts.append("; ".join(notes))
        return " · ".join(parts)
