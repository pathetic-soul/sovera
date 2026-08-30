"""Tool contract, filesystem jail, and the sandbox's refusal to fall back (§8.3, §2.3, §9.3)."""

from __future__ import annotations

from pathlib import Path

import pytest
from docx import Document

from core.audit import AuditLog
from tools.base import JailBreak, RunContext, resolve_in_jail, validate_args
from tools.doc_write import DocWrite
from tools.fs_read import FsRead
from tools.py_sandbox import PySandbox


@pytest.fixture
def ctx(tmp_path: Path) -> RunContext:
    ws = tmp_path / "workspace"
    (ws / "inbox").mkdir(parents=True)
    audit = AuditLog(ws / ".audit" / "audit.jsonl", "test")
    return RunContext(workspace=ws, audit=audit, session_id="test")


# --- the jail (§2.3) --------------------------------------------------------

def test_jail_allows_paths_inside(ctx: RunContext) -> None:
    got = resolve_in_jail(ctx.workspace, "inbox/report.md")
    assert got == (ctx.workspace / "inbox" / "report.md").resolve()


@pytest.mark.parametrize(
    "escape",
    [
        "../secrets.txt",
        "inbox/../../etc/passwd",
        "C:\\Windows\\System32\\config\\SAM",
        "/etc/shadow",
        "inbox/../../../..",
    ],
)
def test_jail_refuses_escapes(ctx: RunContext, escape: str) -> None:
    with pytest.raises(JailBreak):
        resolve_in_jail(ctx.workspace, escape)


def test_jail_refuses_symlink_out(ctx: RunContext, tmp_path: Path) -> None:
    """Resolving before comparing is what catches this; a string-prefix check
    on the argument would let it through."""
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = ctx.workspace / "inbox" / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks need Developer Mode or admin on Windows")
    with pytest.raises(JailBreak):
        resolve_in_jail(ctx.workspace, "inbox/link.txt")


# --- argument validation (§2.3, §8.4) ---------------------------------------

SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "e.g. inbox/x.md"},
        "lines": {"type": "integer"},
        "mode": {"type": "string", "enum": ["head", "tail"]},
    },
    "required": ["path"],
}


def test_valid_args_pass() -> None:
    assert validate_args(SCHEMA, {"path": "a.md", "lines": 10, "mode": "head"}) is None


def test_missing_required_names_the_parameter_and_gives_an_example() -> None:
    msg = validate_args(SCHEMA, {})
    assert msg is not None and "path" in msg and "inbox/x.md" in msg


def test_unknown_parameter_is_rejected_with_the_allowed_list() -> None:
    msg = validate_args(SCHEMA, {"path": "a.md", "recursive": True})
    assert msg is not None and "recursive" in msg and "lines" in msg


def test_wrong_type_is_rejected() -> None:
    assert validate_args(SCHEMA, {"path": 7}) is not None


def test_boolean_is_not_an_integer() -> None:
    """bool subclasses int in Python; a model sending true for a count is a
    real mistake, not a widening conversion."""
    assert validate_args(SCHEMA, {"path": "a.md", "lines": True}) is not None


def test_enum_is_enforced() -> None:
    assert validate_args(SCHEMA, {"path": "a.md", "mode": "sideways"}) is not None


# --- fs_read ----------------------------------------------------------------

def test_fs_read_returns_the_text_and_audits(ctx: RunContext) -> None:
    (ctx.workspace / "inbox" / "r.md").write_text("V-2301 min 8.9 mm", encoding="utf-8")
    result = FsRead().run({"path": "inbox/r.md"}, ctx)
    assert result.ok and "8.9 mm" in result.output
    assert [r.kind for r in ctx.audit.tail()] == ["file_read"]


def test_fs_read_truncates_to_the_context_budget(ctx: RunContext) -> None:
    (ctx.workspace / "inbox" / "big.md").write_text("x" * 50_000, encoding="utf-8")
    result = FsRead().run({"path": "inbox/big.md"}, ctx)
    assert result.ok and "truncated" in result.output
    assert len(result.output) < 50_000


def test_fs_read_missing_file_lists_what_exists(ctx: RunContext) -> None:
    """§12.6 — a 4B can only repair a guessed filename if told the real ones."""
    (ctx.workspace / "inbox" / "real.md").write_text("hi", encoding="utf-8")
    result = FsRead().run({"path": "inbox/guessed.md"}, ctx)
    assert not result.ok
    assert result.error is not None and "inbox/real.md" in result.error


def test_fs_read_refuses_to_escape(ctx: RunContext) -> None:
    result = FsRead().run({"path": "../../secrets.txt"}, ctx)
    assert not result.ok and result.error is not None and "outside" in result.error


# --- doc_write (§9.5) -------------------------------------------------------

def test_doc_write_produces_a_readable_docx(ctx: RunContext) -> None:
    body = "## Background\n- Minimum thickness 8.9 mm at S7\n\nRecommend coating."
    result = DocWrite().run(
        {"filename": "approval-note-V-2301", "title": "Approval Note - V-2301", "body": body},
        ctx,
    )
    assert result.ok and result.artifacts
    path = ctx.workspace / result.artifacts[0]
    assert path.is_file()

    text = "\n".join(p.text for p in Document(str(path)).paragraphs)
    assert "Approval Note - V-2301" in text
    assert "Minimum thickness 8.9 mm at S7" in text
    assert "Background" in text


def test_doc_write_is_gated_by_the_human(ctx: RunContext) -> None:
    """§2.4 is a compliance feature. If this flips to False, the gate is gone."""
    assert DocWrite().requires_approval is True


def test_doc_write_sanitises_a_hostile_filename(ctx: RunContext) -> None:
    """The filename comes from a 4B, so assume a full sentence or a path."""
    result = DocWrite().run(
        {"filename": "../../etc/pwn me.docx", "title": "T", "body": "b"}, ctx
    )
    assert result.ok
    written = ctx.workspace / result.artifacts[0]
    assert written.is_file()
    assert written.parent == (ctx.workspace / "out").resolve()


def test_doc_write_emits_intent_then_result(ctx: RunContext) -> None:
    """§2.2: write before the action, outcome after, as two linked records."""
    DocWrite().run({"filename": "n", "title": "T", "body": "b"}, ctx)
    records = [r for r in ctx.audit.tail() if r.kind == "file_write"]
    assert [r.payload["phase"] for r in records] == ["intent", "result"]
    assert records[1].payload["ref"] == records[0].seq


# --- py_sandbox (§9.3) ------------------------------------------------------

def test_sandbox_never_runs_on_the_host_when_docker_is_missing(
    ctx: RunContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The invariant of leg 5. A host fallback would void §9.3 entirely."""
    marker = ctx.workspace / "escaped.txt"
    monkeypatch.setattr("tools.py_sandbox.shutil.which", lambda _: None)
    result = PySandbox().run(
        {"code": f"open(r'{marker}', 'w').write('pwned')"}, ctx
    )
    assert not result.ok
    assert not marker.exists(), "sandbox executed code on the host"
    assert result.error is not None and "docker" in result.error.lower()


def test_sandbox_command_carries_the_hardening_flags() -> None:
    """§9.3 verbatim — --network none is the sovereignty artifact, and the UI
    renders these flags, so a silent drop would be a silent claim."""
    from tools.py_sandbox import HARDENING

    assert ("--network", "none") == HARDENING[1:3]
    for flag in ("--read-only", "--cap-drop", "--pids-limit", "--security-opt"):
        assert flag in HARDENING


def test_sandbox_is_gated_by_the_human() -> None:
    assert PySandbox().requires_approval is True
