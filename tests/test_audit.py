"""Chain integrity and guard behaviour. The only two things in leg 1 that can be
silently wrong: a chain that verifies when it shouldn't, and a guard that lets a
packet out."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from core.audit import AuditLog, ConcurrentAudit
from core.net_guard import EgressBlocked, _is_local, install_guard


def test_chain_verifies_and_resumes(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "a.jsonl", "s1")
    log.append("tool_call", {"tool": "fs_read"})
    log.append("file_read", {"path": "workspace/x.txt"})
    assert log.verify() == (True, None)

    reopened = AuditLog(tmp_path / "a.jsonl", "s1")
    rec = reopened.append("approval", {"by": "human"})
    assert rec.seq == 2
    assert reopened.verify() == (True, None)
    assert len(reopened.tail(10)) == 3


def test_a_second_writer_is_refused_instead_of_breaking_the_chain(tmp_path: Path) -> None:
    """The real incident this guards: a second orchestrator started while the
    first held port 8080, both resumed from the same seq, and the chain broke
    silently. Failing loudly beats a BROKEN verify discovered on stage."""
    path = tmp_path / "a.jsonl"
    first = AuditLog(path, "s1")
    first.append("tool_call", {"n": 1})

    second = AuditLog(path, "s2")      # a second process, resuming from the same point
    second.append("tool_call", {"n": 2})

    with pytest.raises(ConcurrentAudit):
        first.append("tool_call", {"n": 3})
    assert AuditLog(path, "s3").verify() == (True, None), "chain damaged despite the guard"


def test_concurrent_appends_from_real_threads_do_not_corrupt_the_chain(tmp_path: Path) -> None:
    """The real incident this guards: a tool call runs inside asyncio.to_thread
    (a genuine OS thread), and the operator kill switch can append from the
    main thread at the same instant. Without a lock, both read the same
    seq/prev_hash before either had written, and both wrote seq N — measured
    live: pressing Stop while a model call was in flight broke the chain on
    the very first try. This uses real `threading.Thread`s, not asyncio tasks,
    because that is what actually raced."""
    import threading

    path = tmp_path / "a.jsonl"
    log = AuditLog(path, "s1")
    barrier = threading.Barrier(8)

    def hammer(n: int) -> None:
        barrier.wait()  # line every thread up so they all append at once
        log.append("tool_call", {"n": n})

    threads = [threading.Thread(target=hammer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    seqs = [r["seq"] for r in rows]
    assert len(seqs) == len(set(seqs)), f"duplicate seq values: {seqs}"
    assert AuditLog(path, "s2").verify() == (True, None)


def test_verify_cli_fails_when_the_log_is_absent(tmp_path: Path) -> None:
    """§10.5 pre-demo assertion: a missing log must not print 'chain intact'.
    verify() itself still answers True — nothing is broken — but the CLI, whose
    whole job is to assert, has to treat 'nothing checked' as a failure."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "core.audit", "verify", str(tmp_path / "absent.jsonl")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1
    assert "no audit log" in (proc.stdout + proc.stderr)


def test_intent_and_result_are_linked(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "a.jsonl", "s1")
    intent = log.append("model_call", {"phase": "intent", "model": "driver"})
    result = log.append("model_call", {"phase": "result"}, ref=intent.seq)
    assert result.payload["ref"] == intent.seq


def test_tampered_payload_breaks_chain(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    log = AuditLog(path, "s1")
    log.append("file_write", {"path": "workspace/report.docx"})
    log.append("approval", {"by": "human"})

    lines = path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    rec["payload"]["path"] = "C:/Windows/System32/evil.dll"
    lines[0] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert AuditLog(path, "s1").verify() == (False, 0)


def test_tampered_metadata_breaks_chain(tmp_path: Path) -> None:
    """The reason we hash the whole record, not just prev_hash + payload."""
    path = tmp_path / "a.jsonl"
    log = AuditLog(path, "s1")
    log.append("file_write", {"path": "workspace/report.docx"})

    rec = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    rec["ts"] = "1999-01-01T00:00:00+00:00"
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    assert AuditLog(path, "s1").verify() == (False, 0)


@pytest.mark.parametrize(
    "addr,local",
    [
        (("127.0.0.1", 11434), True),
        (("192.168.1.10", 80), True),
        (("104.18.6.192", 443), False),
        (("api.openai.com", 443), False),  # unresolved hostname: deny by default
    ],
)
def test_is_local(addr: tuple[str, int], local: bool) -> None:
    assert _is_local(addr) is local


def test_guard_blocks_and_audits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log = AuditLog(tmp_path / "a.jsonl", "s1")
    monkeypatch.setattr(socket.socket, "connect", socket.socket.connect)  # restore after
    install_guard(log)

    with pytest.raises(EgressBlocked):
        socket.socket().connect(("104.18.6.192", 443))

    tail = log.tail(1)
    assert tail[0].kind == "egress_attempt"
    assert log.verify() == (True, None)
