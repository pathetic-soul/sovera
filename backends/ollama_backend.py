"""Ollama over loopback HTTP (AGENTS.md §7).

stdlib `urllib` on purpose. `requests`/`httpx` would add a dependency to reach
127.0.0.1, and every dependency is one more thing to audit for §2.1 phone-home
behaviour. net_guard already permits loopback, so this needs no exemption.

GPU: this module is the only place VRAM is spent. `num_ctx` is passed on every
call and clamped to the registry's `max_ctx`, because Ollama will happily
allocate a KV cache past 5.2 GB and OOM the card mid-demo (§4.2.5).
Latency: 2-5 s extra on the first call after a model swap (§4.2.3), then
roughly 15-30 tok/s for a 4B and 8-15 tok/s for a 7-8B on the RTX 4050.

Every call emits an intent record and a result record (§2.2, §8.5). The backend
is the choke point rather than the agent, so no future caller can bypass it.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from backends.base import BackendError, Completion, LLMBackend, Message
from core.audit import AuditLog

DEFAULT_HOST = "http://127.0.0.1:11434"
# Generous: a cold 8B load from NVMe is 8-20 s (§4.2.3) before a token appears.
DEFAULT_TIMEOUT = 300.0


class OllamaBackend(LLMBackend):
    def __init__(
        self,
        audit: AuditLog,
        host: str = DEFAULT_HOST,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.audit = audit
        self.host = host.rstrip("/")
        self.timeout = timeout

    def available(self) -> tuple[bool, str]:
        """Is the daemon up? Rendered in the UI so a dead Ollama is visible
        before the demo starts rather than as a stack trace on stage."""
        try:
            body = self._post("/api/tags", {}, method="GET")
        except BackendError as exc:
            return False, str(exc)
        tags = [m["name"] for m in body.get("models", [])]
        return True, f"{len(tags)} models staged"

    def chat(
        self,
        ref: str,
        messages: list[Message],
        *,
        max_ctx: int,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": ref,
            "messages": [m.model_dump(exclude_defaults=True) for m in messages],
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": max_ctx},
            # Measured on this machine: qwen3-vl:4b (Ollama's "thinking" build,
            # `ollama show --modelfile` -> RENDERER/PARSER qwen3-vl-thinking)
            # spent 2173 eval tokens and 104s producing one ~300-char reply to
            # the ReAct loop's one-JSON-object prompt — the hidden <think> block
            # is uncapped by §8.4's budget and would blow the 20k cap in ~9
            # steps. `think: False` drops that call to ~1s; Ollama ignores the
            # field for models that don't support hybrid thinking (verified
            # against qwen3:4b-instruct above), so this is safe for every route.
            "think": False,
        }
        if json_mode:
            payload["format"] = "json"

        intent = self.audit.append(
            "model_call",
            {
                "phase": "intent",
                "model_ref": ref,
                "num_ctx": max_ctx,
                "temperature": temperature,
                "json_mode": json_mode,
                "messages": len(messages),
            },
        )
        started = time.monotonic()
        try:
            body = self._post("/api/chat", payload)
        except BackendError as exc:
            self.audit.append(
                "error",
                {"phase": "result", "model_ref": ref, "error": str(exc)},
                ref=intent.seq,
            )
            raise

        msg = body.get("message", {})
        # qwen3-vl:4b's thinking parser leaves `content` empty and puts the
        # whole reply in `thinking` even with think:False (measured, not
        # documented) — fall back rather than silently returning "" to a loop
        # that then reports "model produced invalid output twice" for no
        # visible reason.
        text = msg.get("content") or msg.get("thinking", "")
        completion = Completion(
            text=text,
            model=ref,
            prompt_tokens=int(body.get("prompt_eval_count", 0)),
            eval_tokens=int(body.get("eval_count", 0)),
            ms=int((time.monotonic() - started) * 1000),
        )
        self.audit.append(
            "model_call",
            {
                "phase": "result",
                "model_ref": ref,
                "prompt_tokens": completion.prompt_tokens,
                "eval_tokens": completion.eval_tokens,
                "ms": completion.ms,
                "chars": len(completion.text),
            },
            ref=intent.seq,
        )
        return completion

    def _post(
        self, path: str, payload: dict[str, Any], method: str = "POST"
    ) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if method == "POST" else None
        req = urllib.request.Request(
            f"{self.host}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise BackendError(f"ollama HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise BackendError(
                f"ollama unreachable at {self.host} ({type(exc).__name__}: {exc}). "
                "Start it with `ollama serve`."
            ) from exc
        try:
            parsed: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BackendError(f"ollama returned non-JSON: {raw[:200]}") from exc
        return parsed
