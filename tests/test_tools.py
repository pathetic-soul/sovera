"""Tool contract, filesystem jail, and the sandbox's refusal to fall back (§8.3, §2.3, §9.3)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from docx import Document

from core.audit import AuditLog
from tools.base import JailBreak, RunContext, resolve_in_jail, validate_args
from tools.doc_write import DocWrite
from tools.fs_read import FsRead
from tools.fs_write import FsWrite
from tools.kb_search import KbSearch
from tools.ocr_read import OcrRead, _lines_from_result
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


# --- ocr_read (§5 `ocr` roster row) -----------------------------------------

def test_ocr_read_refuses_to_escape(ctx: RunContext) -> None:
    result = OcrRead().run({"path": "../../secrets.png"}, ctx)
    assert not result.ok and result.error is not None and "outside" in result.error


def test_ocr_read_missing_file_names_it(ctx: RunContext) -> None:
    result = OcrRead().run({"path": "inbox/missing.png"}, ctx)
    assert not result.ok
    assert result.error is not None and "no such file" in result.error


def test_ocr_read_rejects_non_image_extensions(ctx: RunContext) -> None:
    (ctx.workspace / "inbox" / "r.md").write_text("hi", encoding="utf-8")
    result = OcrRead().run({"path": "inbox/r.md"}, ctx)
    assert not result.ok
    assert result.error is not None and "not an image" in result.error


def test_ocr_read_names_the_install_command_when_paddleocr_is_absent(
    ctx: RunContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§12.6-style discipline: a missing optional dependency is a message a
    human can act on, not a traceback that kills the run."""
    (ctx.workspace / "inbox" / "scan.png").write_bytes(b"\x89PNG\r\n")

    def _raise() -> None:
        raise ImportError("no module named paddleocr")

    monkeypatch.setattr("tools.ocr_read._get_engine", _raise)
    result = OcrRead().run({"path": "inbox/scan.png"}, ctx)
    assert not result.ok
    assert result.error is not None and "pip install" in result.error


def test_ocr_read_audits_as_file_read_with_the_tool_name(
    ctx: RunContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """kind stays 'file_read' — §8.5's Kind literal is closed — and 'tool' in
    the payload is what tells ocr_read's reads apart from fs_read's."""
    (ctx.workspace / "inbox" / "scan.png").write_bytes(b"\x89PNG\r\n")
    monkeypatch.setattr(
        "tools.ocr_read._get_engine", lambda: pytest.fail("should not run without a real image")
    )
    OcrRead().run({"path": "inbox/missing2.png"}, ctx)
    records = ctx.audit.tail()
    assert records[-1].kind == "file_read"
    assert records[-1].payload["tool"] == "ocr_read"


def test_lines_from_result_reads_the_3x_dict_shape() -> None:
    page = {"rec_texts": ["V-2301", "8.9 mm"], "rec_scores": [0.99, 0.87]}
    assert _lines_from_result([page]) == [("V-2301", 0.99), ("8.9 mm", 0.87)]


def test_lines_from_result_reads_the_2x_box_tuple_shape() -> None:
    page = [[[[0, 0], [1, 0], [1, 1], [0, 1]], ("V-2301", 0.99)]]
    assert _lines_from_result([page]) == [("V-2301", 0.99)]


def test_ocr_read_extracts_real_text_from_a_generated_image(ctx: RunContext) -> None:
    """The one real, non-mocked check: render text into an image with PIL and
    confirm PaddleOCR actually reads it back — not just that the plumbing
    doesn't crash. Skips cleanly where the optional `ocr` extra is not
    installed (`pip install -e ".[ocr]"`)."""
    pytest.importorskip("paddleocr")
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (400, 80), "white")
    ImageDraw.Draw(img).text((10, 20), "V-2301 8.9 MM", fill="black")
    path = ctx.workspace / "inbox" / "scan.png"
    img.save(path)

    result = OcrRead().run({"path": "inbox/scan.png"}, ctx)
    assert result.ok
    assert "8.9" in result.output


# --- kb_search (§9.4, leg 7) -------------------------------------------------

def test_kb_search_rejects_empty_query(ctx: RunContext) -> None:
    result = KbSearch().run({"query": ""}, ctx)
    assert not result.ok and result.error == "empty query"


def test_kb_search_is_not_gated(ctx: RunContext) -> None:
    """§2.4 gates writes and execution; a read across the corpus is neither."""
    assert KbSearch().requires_approval is False


def test_kb_search_finds_a_real_equipment_tag_in_the_real_corpus(ctx: RunContext) -> None:
    """The one non-mocked check: search the actual data/corpus/ against a tag
    that is genuinely in it — V-2301, the vessel README's own demo runbook
    asks the agent about (data/corpus/inbox/UT-2024-114-V-2301.md)."""
    result = KbSearch().run({"query": "V-2301 thickness survey findings"}, ctx)
    assert result.ok
    assert "V-2301" in result.output


def test_kb_search_audits_the_query_and_source_docs(ctx: RunContext) -> None:
    KbSearch().run({"query": "corrosion allowance"}, ctx)
    records = [r for r in ctx.audit.tail() if r.kind == "tool_call" and r.payload.get("tool") == "kb_search"]
    assert records and records[-1].payload["ok"] is True
    assert records[-1].payload["hits"] > 0


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


# --- fs_write (§8.3, §2.3, §2.4) --------------------------------------------

def test_fs_write_creates_the_file_and_parent_dirs(ctx: RunContext) -> None:
    result = FsWrite().run({"path": "out/notes.txt", "content": "hello"}, ctx)
    assert result.ok and result.artifacts == ["out/notes.txt"]
    written = ctx.workspace / "out" / "notes.txt"
    assert written.read_text(encoding="utf-8") == "hello"


def test_fs_write_overwrites_an_existing_file(ctx: RunContext) -> None:
    (ctx.workspace / "inbox" / "r.md").write_text("old", encoding="utf-8")
    result = FsWrite().run({"path": "inbox/r.md", "content": "new"}, ctx)
    assert result.ok
    assert (ctx.workspace / "inbox" / "r.md").read_text(encoding="utf-8") == "new"


def test_fs_write_refuses_to_escape(ctx: RunContext) -> None:
    result = FsWrite().run({"path": "../../pwn.txt", "content": "x"}, ctx)
    assert not result.ok and result.error is not None and "outside" in result.error


def test_fs_write_is_gated_by_the_human(ctx: RunContext) -> None:
    """§2.4 is a compliance feature. If this flips to False, the gate is gone."""
    assert FsWrite().requires_approval is True


def test_fs_write_emits_intent_then_result(ctx: RunContext) -> None:
    """§2.2: write before the action, outcome after, as two linked records."""
    FsWrite().run({"path": "out/n.txt", "content": "b"}, ctx)
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


def test_sandbox_tmpfs_is_writable_by_the_non_root_user() -> None:
    """The image runs as uid 10001 and a tmpfs mounts root-owned 0755, so /work
    -- the sandbox's own cwd -- was not writable by the process using it. Every
    generated script that saved a scratch file died on PermissionError. The
    other sandbox tests are all mocked, so nothing caught it."""
    from tools.py_sandbox import HARDENING

    tmpfs = HARDENING[HARDENING.index("--tmpfs") + 1]
    assert "mode=1777" in tmpfs, f"/work would mount root-owned and unwritable: {tmpfs}"


@pytest.mark.skipif(shutil.which("docker") is None, reason="needs docker")
def test_sandbox_can_actually_create_and_read_files(ctx: RunContext) -> None:
    """The one non-mocked sandbox check: a script must be able to write a file
    in its working directory, read it back, and still be network-less."""
    if subprocess.run(["docker", "image", "inspect", "sandbox-py:local"],
                      capture_output=True).returncode != 0:
        pytest.skip("sandbox-py:local not built")
    result = PySandbox().run({"code": (
        "open('scratch.txt','w').write('working file')\n"
        "print('read back:', open('scratch.txt').read())\n"
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1',443), timeout=3); print('REACHABLE')\n"
        "except OSError as e: print('blocked:', e)\n"
    )}, ctx)
    assert result.ok, result.output
    assert "read back: working file" in result.output
    assert "Network is unreachable" in result.output, "containment regressed"
