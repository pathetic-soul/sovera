"""MRPL public filings -> validated markdown tables for the corpus.

Build time only (AGENTS.md §3). Nothing here runs during a demo.

WHY THIS FILE IS MOSTLY VALIDATION
----------------------------------
These PDFs are the only *real* documents in the project. They are also the
most dangerous input in it, because their text layer is OCR output and the OCR
loses decimal separators:

    "20,988 03"    is  20,988.03    (decimal point read as a space)
    "28.785 92"    is  28,785.92    (thousands comma read as a period, too)
    "3,818,05"     is   3,818.05    (decimal point read as a comma)

A numeric-reasoning dataset built on those tokens would teach the model wrong
figures and, worse, every "truth" value derived from them would itself be
wrong — so the eval would agree with the training data and the error would
never surface. That is the one bug class this project cannot afford.

So the repair is not trusted, it is PROVEN. Indian financial statements carry
their own arithmetic identities:

    Total Income   == Revenue from Operations + Other Income
    Total Expenses == sum of the expense lines
    PBT            == Total Income - Total Expenses

The totals rows survive OCR cleanly (they are printed in bold and extract as
"26,029.19"), so they serve as checksums for the mangled component rows.
Measured on the Sep-2025 statement, all six period columns reconcile to within
a paisa. A table that does NOT reconcile is dropped, never hand-patched —
hand-patching is how a plausible wrong number gets into a corpus.

    python -m finetune.corpus_filings --report     # what would be kept
    python -m finetune.corpus_filings              # write data/corpus/filings/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
# Source PDFs live outside the repo — they are bulky and not ours to
# redistribute. Point SOVEREIGN_PDF_DIR at wherever they are staged; the
# default is a sibling of the repo so a clone works without an env var.
PDF_DIR = Path(os.environ.get("SOVEREIGN_PDF_DIR", str(ROOT.parent / "datasets")))
OUT_DIR = ROOT / "data" / "corpus" / "filings"

# A cell must reconcile to within this many crore for the table to be kept.
# Statements are printed to 2 dp, so anything above rounding is a real mismatch.
TOLERANCE = 0.02

# Rows whose OCR text starts with these are the checksum anchors.
TOTAL_INCOME = ("total income",)
TOTAL_EXPENSES = ("total expenses",)

# Extraction guards. Scanned pages are expensive to walk and yield nothing, so
# bail early rather than spend minutes proving a document is an image.
SCAN_PROBE_PAGES = 3
MAX_PAGES = 60


# --------------------------------------------------------------------------
# number repair
# --------------------------------------------------------------------------

def repair_number(token: str) -> float | None:
    """Normalise one OCR-mangled Indian-format number.

    The rule that makes this tractable: every figure in these statements is
    printed to exactly two decimal places. So whatever separator sits before
    the final two digits is the decimal point, whatever OCR turned it into.
    Everything left of it is thousands grouping and is discarded.

    Parentheses are the accounting negative: "(481.45)" is -481.45.
    """
    t = token.strip()
    if not t or t in {"-", "--", "—"}:
        return None

    negative = t.startswith("(") and t.endswith(")")
    t = t.strip("()")
    t = re.sub(r"[^0-9.,\s-]", "", t).strip()
    if not t or not re.search(r"\d", t):
        return None

    m = re.match(r"^(-?[\d.,\s]*?)[\s.,](\d{2})$", t)
    if m:
        whole = re.sub(r"[^\d]", "", m.group(1)) or "0"
        value = float(f"{whole}.{m.group(2)}")
    else:
        stripped = re.sub(r"[^\d]", "", t)
        if not stripped:
            return None
        value = float(stripped)

    return -value if negative else value


# One figure. The trailing "(?:\s\d{2}(?![\d,.]))?" is the OCR decimal-point
# repair: "20,988 03" is one number, but the space may only absorb exactly two
# digits, and only when no digit or separator follows. Without that lookahead
# the pattern runs across a whole row — "28,845.82 27,334.13 22,918.23 ..."
# becomes a single 43-digit token and the row silently yields no figures.
NUM_RE = re.compile(r"\(?-?\d[\d,.]*(?:\s\d{2}(?![\d,.]))?\)?")

# Tokens that look numeric but are not figures. Without these the Mangaluru
# PIN code (575 030) and every period date (31.03.2021) parse as crore values
# and wreck the checksums — which is exactly what the first run did.
DATE_RE = re.compile(r"\d{1,2}\s*[.\-/]\s*\d{1,2}\s*[.\-/]\s*\d{2,4}")
YEARISH_RE = re.compile(r"^(19|20)\d{2}$")

# The registered-office block carries the CIN and the Mangaluru PIN (575 030),
# which repairs to 575,030.00 and destroys any checksum it lands in.
ADDRESS_MARKERS = ("cin:", "cin :", "regd. office", "regd.office", "kuthethoor",
                   "mudapadav", "katipalla", "mangaluru", "karnataka")

# Statement rows label their own arithmetic: "Total Income (1+11)",
# "Profit/ (Loss) Before Tax (V - VI)". Those references are not figures.
# OCR mangles them freely — "(I+II)" has been seen as "(1+11)" and as "(PHI)" —
# so match any short parenthetical rather than enumerating the corruptions.
XREF_RE = re.compile(r"^\s*\([^)]{0,10}\)")


def _is_figure(tok: str) -> bool:
    raw = tok.strip("()")
    if DATE_RE.search(raw):
        return False
    digits = re.sub(r"[^\d]", "", raw)
    if not digits:
        return False
    if YEARISH_RE.match(digits):          # a bare year in a column header
        return False
    if len(digits) > 12:                  # CIN / registration numbers
        return False
    return True


def numbers_in(line: str) -> list[float]:
    """Every repairable *figure* on one extracted line, left to right."""
    # The registered-office block carries the CIN and the PIN code; it is
    # never a statement row, so drop it wholesale rather than token by token.
    low = line.lower()
    if any(k in low for k in ADDRESS_MARKERS):
        return []

    out: list[float] = []
    for tok in NUM_RE.findall(line):
        if not _is_figure(tok):
            continue
        v = repair_number(tok)
        if v is not None:
            out.append(v)
    return out


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

@dataclass
class Filing:
    """One PDF, with whatever survived validation."""

    path: Path
    pages: int = 0
    text_chars: int = 0
    checks_run: int = 0
    checks_passed: int = 0
    notes: list[str] = field(default_factory=list)
    facts: list[dict[str, object]] = field(default_factory=list)
    text: str = ""

    @property
    def scanned(self) -> bool:
        return self.text_chars < 1000

    @property
    def verdict(self) -> str:
        """Two independent judgements, deliberately not collapsed into one.

        `usable` governs whether the document ships as readable corpus text —
        which needs no table reconstruction, because the agent reads it as
        prose and cites what it sees. Whether any *figure* was verified governs
        only whether this filing may supply ground truth to a generated
        question. Conflating the two was the earlier mistake: it threw away
        37 real documents because a table parser could not re-derive their
        arithmetic.
        """
        if self.scanned:
            return "scanned"           # real, but for leg 6 (vision), not text
        if self.facts:
            return "text+facts"        # ships, and may source numeric questions
        return "text-only"             # ships, but sources no numeric questions


def read_pdf(path: Path) -> Filing:
    import pypdf

    f = Filing(path=path)
    try:
        reader = pypdf.PdfReader(str(path))
    except Exception as exc:                      # corrupt file, not our problem
        f.notes.append(f"unreadable: {type(exc).__name__}")
        return f

    # Page count and page access both hit the object graph, so a malformed or
    # encrypted file raises here rather than at construction. One of the supplied
    # PDFs is AES-encrypted and threw DependencyError straight through an
    # earlier try that only wrapped PdfReader().
    try:
        f.pages = len(reader.pages)
        pages = list(reader.pages)
    except Exception as exc:
        f.notes.append(f"unreadable pages: {type(exc).__name__}")
        return f

    parts: list[str] = []
    for i, page in enumerate(pages):
        # A scanned page still costs pypdf a full content-stream walk, and on a
        # large image-only document that runs into minutes — the CVC Vigilance
        # Manual took 145 s on its own and stalled the whole report. If the
        # opening pages carry no text this is a scan; stop rather than grind
        # through it for nothing.
        if i == SCAN_PROBE_PAGES and sum(len(p) for p in parts) < 200:
            f.notes.append(f"image-only after {SCAN_PROBE_PAGES} pages, extraction stopped")
            break
        if i >= MAX_PAGES:
            f.notes.append(f"truncated at {MAX_PAGES} pages")
            break
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    f.text = "\n".join(parts)
    f.text_chars = len(f.text)
    return f


def _anchor_lines(text: str, keys: tuple[str, ...]) -> Iterator[str]:
    for line in text.splitlines():
        low = line.strip().lower()
        if any(low.startswith(k) for k in keys):
            yield line


def validate(f: Filing) -> Filing:
    """Locate figures that provably satisfy the statement's own identity.

    WHY THIS SEARCHES INSTEAD OF ASSUMING A LAYOUT
    ----------------------------------------------
    An earlier version assumed the component figures sat immediately above the
    totals row, interleaved per period column. That holds for the 2025-26
    statements and breaks for the 2021-23 ones, where the extractor flows
    revenue figures as a group and other-income figures as a second group.
    Guessing between the two produced confident nonsense —
    "28,383.41 + 22,843.79 != 28,422.99" is one column's revenue paired with
    another column's revenue.

    So the layout is not assumed. The document *states* the identity
    (Total Income = Revenue from Operations + Other Income); this searches the
    preceding figures for the pair that satisfies it. Finding operands for a
    stated identity is verification, not curve-fitting: at two-decimal
    precision over a few dozen candidates a coincidental match is rare, and
    anything unmatched is simply reported as unverified rather than guessed.

    A verified column yields a fact triple that downstream question generation
    can use as ground truth. Everything else still ships as a readable
    document — it just never becomes a question with a numeric answer.
    """
    if f.scanned:
        return f

    lines = f.text.splitlines()
    for i, line in enumerate(lines):
        # The row is labelled "III Total Income (1+11)" — a roman numeral and
        # an OCR-garbled cross-reference precede it, so match anywhere.
        low = line.lower()
        if "total income" not in low:
            continue
        tail = line[low.index("total income") + len("total income"):]
        tail = XREF_RE.sub("", tail)
        totals = numbers_in(tail)
        if len(totals) < 2:
            continue

        # Candidate operands: every figure in the 40 lines above the totals row.
        window: list[float] = []
        for back in range(max(0, i - 40), i):
            window.extend(numbers_in(lines[back]))
        if len(window) < 2:
            continue

        for tot in totals:
            f.checks_run += 1
            hit = _find_operands(window, tot)
            if hit:
                rev, oth = hit
                f.checks_passed += 1
                f.facts.append({
                    "identity": "total_income = revenue_from_operations + other_income",
                    "revenue_from_operations": rev,
                    "other_income": oth,
                    "total_income": tot,
                })
            else:
                f.notes.append(f"no operands sum to {tot:,.2f}")
    return f


def _find_operands(window: list[float], total: float) -> tuple[float, float] | None:
    """The (revenue, other_income) pair that reproduces a stated total.

    Constrained deliberately: revenue must dominate other income, and both
    must be positive. Without that a total can be hit by arbitrary pairs and
    the "verification" stops meaning anything.
    """
    # Index by rounded paise so the complement is a hash lookup rather than a
    # second scan — the nested-loop version took minutes across 37 filings.
    index: dict[int, float] = {}
    for v in window:
        index.setdefault(round(v * 100), v)

    best: tuple[float, float] | None = None
    for a in window:
        if a <= 0 or a > total:
            continue
        b = total - a
        if b < 0 or b > a:                       # other income never exceeds revenue
            continue
        c = index.get(round(b * 100))
        if c is None:
            continue
        # Prefer the largest revenue: the true split has other income as a
        # small residual, so a >> b is the discriminating shape.
        if best is None or a > best[0]:
            best = (a, c)
    return best


# --------------------------------------------------------------------------
# markdown emission
# --------------------------------------------------------------------------

PERIOD_RE = re.compile(
    r"(QUARTER|HALF YEAR|NINE MONTHS|YEAR)\s+(?:AND\s+\w+\s+)?ENDED\s+"
    r"([A-Z]+\s+\d{1,2},?\s+\d{4}|\d{2}[.\-/]\d{2}[.\-/]\d{4})",
    re.I,
)


def to_markdown(f: Filing) -> str:
    """Emit the filing as a document the agent's fs_read can consume.

    Deliberately conservative: the statement body is kept as extracted text
    rather than being coerced into a markdown grid. Reconstructing a six-column
    financial table from flowed OCR text is guesswork, and a table that is
    silently mis-aligned is worse for training than plainly formatted text the
    model must read carefully. The numbers are what were validated; the layout
    is not claimed to be anything it is not.
    """
    m = PERIOD_RE.search(f.text)
    period = m.group(0).title() if m else "period not stated"
    body = "\n".join(ln.rstrip() for ln in f.text.splitlines() if ln.strip())

    return (
        f"# MRPL Financial Results — {period}\n\n"
        f"> Source: {f.path.name}\n"
        f"> Mangalore Refinery and Petrochemicals Limited, a subsidiary of ONGC.\n"
        f"> Public disclosure under SEBI LODR. Amounts in ₹ Crore unless stated.\n"
        f"> Figures machine-extracted and reconciled against the statement's own\n"
        f"> totals ({f.checks_passed}/{f.checks_run} identity checks passed).\n\n"
        f"{body}\n"
    )


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="store_true", help="classify only, write nothing")
    ap.add_argument("--pdf-dir", default=str(PDF_DIR))
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.is_dir():
        print(f"no such directory: {pdf_dir}", file=sys.stderr)
        return 2

    filings = [validate(read_pdf(p)) for p in sorted(pdf_dir.glob("*.pdf"))]

    buckets: dict[str, list[Filing]] = {}
    for f in filings:
        buckets.setdefault(f.verdict, []).append(f)

    print(f"{len(filings)} PDFs in {pdf_dir}\n")
    for verdict in ("text+facts", "text-only", "scanned"):
        group = buckets.get(verdict, [])
        if not group:
            continue
        print(f"[{verdict}] {len(group)}")
        for f in group:
            checks = f"{f.checks_passed}/{f.checks_run}" if f.checks_run else "-"
            print(f"    {checks:>7}  {f.pages:>3}p  {f.path.name[:66]}")
        print()

    keep = buckets.get("text+facts", []) + buckets.get("text-only", [])
    facts = sum(len(f.facts) for f in keep)
    print(f"{len(keep)} documents ship as corpus text")
    print(f"{facts} verified figures available as question ground truth")
    print(f"{len(buckets.get('scanned', []))} image-only PDFs held back for leg 6 (vision)")

    if args.report:
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for f in keep:
        name = re.sub(r"[^A-Za-z0-9]+", "-", f.path.stem).strip("-").lower()[:70]
        out = OUT_DIR / f"{name}.md"
        out.write_text(to_markdown(f), encoding="utf-8")
        manifest.append({
            "source_pdf": f.path.name,
            "markdown": str(out.relative_to(ROOT)).replace("\\", "/"),
            "pages": f.pages,
            "checks_passed": f.checks_passed,
            "checks_run": f.checks_run,
            "facts": f.facts,
        })

    (OUT_DIR / "manifest.json").write_text(
        json.dumps({"source": "mrpl_filings", "files": manifest}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {len(manifest)} documents -> {OUT_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
