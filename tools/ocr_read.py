"""Extract text from a scanned image via PaddleOCR (AGENTS.md §5, §8.3).

CPU only, ~1.5 GB RAM once loaded — this is the `ocr` row in the model roster
table (§5), sharing the machine's 64 GB RAM rather than the 5.2 GB VRAM budget
the LLMs fight over. It runs independently of the router: routing picks which
*chat* model answers the task, and this tool gives that model (or `vision`)
byte-exact text instead of asking it to read pixels itself.

Same jail and audit discipline as `fs_read.py`. No approval gate: like
`fs_read`, this only reads inside the jail and changes nothing (§2.4 gates
writes and execution).

PaddleOCR is a heavy, optional dependency (`pip install -e ".[ocr]"`) — it is
imported lazily so the app still starts without it, and the engine is built
once and cached, because constructing it loads the detection/recognition
weights from disk (seconds), which `run()` cannot afford to pay per call.
"""

from __future__ import annotations

from typing import Any

from tools.base import JailBreak, RunContext, Tool, ToolResult, resolve_in_jail

MAX_CHARS = 6000  # same context-budget cap as fs_read.py (§8.4)
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

_engine: Any = None  # lazy singleton; see _get_engine


def _get_engine() -> Any:
    global _engine
    if _engine is None:
        from paddleocr import PaddleOCR  # heavy import, kept off the app's startup path

        # enable_mkldnn=False works around a measured crash on this machine:
        # paddlepaddle 3.3.1's oneDNN CPU path raises NotImplementedError
        # ("ConvertPirAttribute2RuntimeAttribute ... pir::ArrayAttribute") on
        # the text-detection model at inference time. Plain CPU inference,
        # unaffected, is a bit slower — fine at demo scale (one page at a time).
        _engine = PaddleOCR(
            use_textline_orientation=True, lang="en", device="cpu", enable_mkldnn=False
        )
    return _engine


def _lines_from_result(result: Any) -> list[tuple[str, float]]:
    """Normalise PaddleOCR's output across the 2.x and 3.x result shapes.

    3.x `.predict()`/`.ocr()` returns a list of dict-like results carrying
    `rec_texts`/`rec_scores`. 2.x `.ocr()` returns `[[[box, (text, score)], ...]]`.
    Both are handled so this tool keeps working whichever line pip resolved.
    """
    lines: list[tuple[str, float]] = []
    for page in result:
        texts = getattr(page, "get", None) and page.get("rec_texts")
        if texts is not None:
            scores = page.get("rec_scores") or [1.0] * len(texts)
            lines.extend(zip(texts, scores))
            continue
        for item in page or []:
            try:
                _box, (text, score) = item
            except (TypeError, ValueError):
                continue
            lines.append((text, float(score)))
    return lines


class OcrRead(Tool):
    name = "ocr_read"
    description = (
        "Extract printed text from a scanned image or photo in the workspace, "
        "via PaddleOCR (not the vision model's guess). Use before answering "
        "questions about a scan's exact wording, tags, or numbers."
    )
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "workspace-relative image path, e.g. inbox/scan.png",
            }
        },
        "required": ["path"],
    }
    requires_approval = False

    def run(self, args: dict[str, Any], ctx: RunContext) -> ToolResult:
        rel = str(args["path"])

        def audit(fields: dict[str, Any]) -> None:
            # kind stays "file_read" — §8.5's Kind literal is closed, and this
            # tool is semantically a jailed read like fs_read. "tool" in the
            # payload is what tells the two apart in the log.
            ctx.audit.append("file_read", {"tool": "ocr_read", "path": rel, **fields})

        try:
            target = resolve_in_jail(ctx.workspace, rel)
        except JailBreak as exc:
            audit({"ok": False, "error": str(exc)})
            return ToolResult(ok=False, output="", error=str(exc))

        if not target.is_file():
            msg = f"no such file: {rel}"
            audit({"ok": False, "error": msg})
            return ToolResult(ok=False, output="", error=msg)

        if target.suffix.lower() not in IMAGE_EXT:
            msg = f"{rel} is not an image ({', '.join(sorted(IMAGE_EXT))})"
            audit({"ok": False, "error": msg})
            return ToolResult(ok=False, output="", error=msg)

        try:
            engine = _get_engine()
        except ImportError:
            msg = 'PaddleOCR is not installed. Run: pip install -e ".[ocr]"'
            audit({"ok": False, "error": msg})
            return ToolResult(ok=False, output="", error=msg)

        predict = getattr(engine, "predict", None) or engine.ocr
        raw = predict(str(target))
        lines = _lines_from_result(raw)

        if not lines:
            audit({"ok": True, "lines": 0})
            return ToolResult(ok=True, output="[no text detected in image]")

        text = "\n".join(t for t, _ in lines)
        avg_conf = sum(s for _, s in lines) / len(lines)
        truncated = len(text) > MAX_CHARS
        if truncated:
            text = text[:MAX_CHARS] + f"\n…[truncated, {len(text)} chars total]"

        audit({"ok": True, "lines": len(lines), "avg_confidence": round(avg_conf, 3)})
        return ToolResult(ok=True, output=f"[OCR, avg confidence {avg_conf:.2f}]\n{text}")
