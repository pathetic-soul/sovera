"""LLM backend contract (AGENTS.md §7, §8).

Two implementations are planned — Ollama primary, llama.cpp if Ollama misbehaves
on swap latency (§16) — so the ABC is the seam that keeps `core/agent.py` from
learning either one's wire format.

GPU: everything behind `chat()` is where the VRAM actually goes. One resident
model at a time (§4.2.1); a call whose `ref` differs from the resident model
costs the 2-5 s swap in §4.2.3, which the router's rationale is there to mask.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str
    # Base64 image payloads for VLM routes (§5 `vision`). Empty on text routes.
    images: list[str] = Field(default_factory=list)


class Completion(BaseModel):
    text: str
    model: str
    prompt_tokens: int
    eval_tokens: int
    ms: int

    @property
    def total_tokens(self) -> int:
        """Charged against the 20k cumulative cap in §8.4."""
        return self.prompt_tokens + self.eval_tokens


class BackendError(RuntimeError):
    """Backend unreachable or refused. Never swallowed — §2.2 wants it audited."""


class LLMBackend(ABC):
    @abstractmethod
    def chat(
        self,
        ref: str,
        messages: list[Message],
        *,
        max_ctx: int,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> Completion:
        """One turn. `json_mode` constrains output to JSON — §12.2, because
        JSON-in-prose from a 4B is a coin flip."""
