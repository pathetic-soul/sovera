"""backends/ollama_backend.py — the think:False fix (measured against qwen3-vl:4b).

No real Ollama call: `_post` is the one HTTP seam, monkeypatched per test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from backends.base import Message
from backends.ollama_backend import OllamaBackend
from core.audit import AuditLog


@pytest.fixture
def backend(tmp_path: Path) -> OllamaBackend:
    return OllamaBackend(AuditLog(tmp_path / "audit.jsonl", "test"))


def test_chat_requests_thinking_off(
    backend: OllamaBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Measured: qwen3-vl:4b's thinking build took 104s/2173 eval tokens for
    one ReAct step with thinking on, 1s with it off. §8.4's budget assumes
    the latter."""
    sent: dict[str, Any] = {}

    def fake_post(path: str, payload: dict[str, Any], method: str = "POST") -> dict[str, Any]:
        sent.update(payload)
        return {"message": {"content": "{}"}, "prompt_eval_count": 1, "eval_count": 1}

    monkeypatch.setattr(backend, "_post", fake_post)
    backend.chat("qwen3-vl:4b", [Message(role="user", content="hi")], max_ctx=8192)
    assert sent["think"] is False


def test_chat_falls_back_to_thinking_field_when_content_is_empty(
    backend: OllamaBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """qwen3-vl:4b's parser leaves `content` empty and puts the reply in
    `thinking` even with think:False (measured, not documented)."""
    monkeypatch.setattr(
        backend,
        "_post",
        lambda path, payload, method="POST": {
            "message": {"content": "", "thinking": '{"thought": "x", "answer": "y"}'},
            "prompt_eval_count": 1,
            "eval_count": 1,
        },
    )
    completion = backend.chat("qwen3-vl:4b", [Message(role="user", content="hi")], max_ctx=8192)
    assert completion.text == '{"thought": "x", "answer": "y"}'


def test_chat_prefers_content_when_present(
    backend: OllamaBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        backend,
        "_post",
        lambda path, payload, method="POST": {
            "message": {"content": "real answer", "thinking": "ignored"},
            "prompt_eval_count": 1,
            "eval_count": 1,
        },
    )
    completion = backend.chat("driver", [Message(role="user", content="hi")], max_ctx=8192)
    assert completion.text == "real answer"
