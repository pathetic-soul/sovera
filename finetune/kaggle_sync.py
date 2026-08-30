"""Publish datasets, models and training kernels to Kaggle. Build time only.

**Why this does not violate §2.1.** §2.1 governs *runtime*: no process in the
shipped workbench may open a socket. This module is build-time tooling in the
same category as `docker build` and `core.embed --stage` — it runs on a
developer machine, before the firewall is armed, and nothing in `core/`,
`tools/` or `backends/` imports it. The deployed system remains unable to
reach Kaggle or anything else. The honest framing for a reviewer who asks is
"train anywhere, deploy air-gapped" — where the model was fitted is not the
claim being defended; where it can send data at runtime is.

**Everything uploads private by default.** A Kaggle dataset, once public, is
indexable and effectively permanent, and this account is a real person's
portfolio. `--public` exists, but the default is private so that promoting a
dataset is a decision made in the Kaggle UI with eyes on it rather than a
side effect of running a script here.

**What must never go up.** `data/corpus/` may one day hold a real MRPL
document. The specs below name individual files rather than globbing
directories, and `guard_paths()` refuses anything under `data/corpus/`, so
adding a real document cannot silently push it.

    python -m finetune.kaggle_sync list                # what would be pushed
    python -m finetune.kaggle_sync push router         # one spec
    python -m finetune.kaggle_sync push all            # every dataset spec
    python -m finetune.kaggle_sync push-kernel         # the T4 training notebook
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "data" / "external" / "kaggle_staging"  # gitignored

# Never publishable, regardless of what a spec asks for.
FORBIDDEN = (ROOT / "data" / "corpus",)


@dataclass
class DatasetSpec:
    slug: str
    title: str
    subtitle: str
    description: str
    files: list[Path]
    licence: str = "CC0-1.0"
    extra: dict[str, str] = field(default_factory=dict)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def guard_paths(paths: list[Path]) -> None:
    """Refuse to stage anything under a forbidden root. Fails loudly, never silently."""
    for p in paths:
        resolved = p.resolve()
        for root in FORBIDDEN:
            if root.resolve() in resolved.parents or resolved == root.resolve():
                raise SystemExit(
                    f"REFUSED: {p} is under {root}, which may contain real MRPL "
                    f"documents and must never be uploaded. Edit the spec."
                )


def specs() -> list[DatasetSpec]:
    """The upload manifest. Files are named individually, never globbed."""
    router_readme = f"""# Refinery Task-Routing Exemplars

{_read(ROOT / 'config' / 'routing_exemplars.jsonl').count(chr(10))} hand-labelled
work requests in refinery / PSU English, each tagged with one of 11 task types:

`plan`, `qa`, `summarize`, `approval_note`, `code_write`, `code_fix`,
`scan_understanding`, `drawing_qa`, `handwriting`, `calc`, `spreadsheet`.

Built for a deterministic model router in an air-gapped agentic workbench
(Smart India Hackathon 2026, problem statement 26117 — MRPL). The router picks
which local open-weight model handles a request; no LLM makes that decision,
so the choice is reproducible and can be quoted as an accuracy number.

## Why this dataset exists

Public intent-classification corpora are customer-support or smalltalk
flavoured. None of them contain sentences like *"Draw up the MOC closeout for
the VDU vacuum ejector change"* or *"What's the retirement limit on this
vessel?"* — where the subject matter is shared across every class and only the
*verb* carries the intent. That property is what makes this corpus hard and
what makes it worth publishing.

## Benchmark

Stratified split, every 3rd exemplar per class held out (110 train / 66 held
out). Hyperparameters were chosen by leave-one-out CV on the training split
only; the held-out set was scored once, at the end.

| scorer | TRAIN LOO-CV | held-out |
|---|---|---|
| lexical TF-IDF (class-IDF weighted) | 74.5% | 75.8% |
| dense `bge-small-en-v1.5` centroids | 84.5% | - |
| **hybrid, w=0.8** | **85.5%** | **80.3%** |

The hybrid blends the two on *standardised* per-query scores, which matters:
lexical cosine spans ~0.1-0.5 with a wide spread, dense cosine ~0.6-0.9 with a
narrow one, so blending the raw numbers just lets the wider-spread scorer win
every time.

Remaining errors are dominated by subject matter overriding intent — *"write a
python script to compute corrosion rate"* routes to `calc` rather than
`code_write`.

## Format

JSON Lines, one object per row: `{{"prompt": "...", "task_type": "..."}}`
"""

    corpus_readme = """# Agent-Loop Traces for Tool-Calling Fine-Tunes

Multi-turn traces for supervised fine-tuning of a small tool-calling agent
(Qwen3-4B class) in an industrial-inspection setting. Every assistant turn is a
single JSON object — a tool call or a final answer — so one schema validates
both the training data and the runtime output.

Loss is intended on assistant turns only.

## Why this exists: three adapters that lost to their own base model

An earlier version of this corpus trained three adapters, and **all three scored
worse than the untouched base model** (0/8, 2/8 and 5/8 against base 5/8 on an
end-to-end inspection task). Profiling the corpus explained why, and every
design choice below is a direct answer to one of those defects:

| defect in the old corpus | consequence | fixed by |
|---|---|---|
| 600 of 612 traces opened with `fs_read` | learned "step one is always a read", not *when* to read | first action is now 69% read / 19% calc / 12% direct answer |
| the tool set was a frozen constant | offered a new tool it refused to use it — brittle to any tool change | **663 distinct tool subsets**, drawn from a pool of 11 |
| taught what was already at ceiling | base scored json_valid 100%, tool_match 97.3% — no headroom to buy | re-aimed at domain judgement and hard loop paths |
| no failed calls, ambiguity or refusals | the loop's difficult branches had zero supervision | dedicated `repair`, `clarify` and `unknown` families |

## Tool-schema generalisation

Traces are generated against a **random subset** of an 11-tool pool, and a tool
only ever appears in a trace when it is also printed in that trace's own system
prompt. The rules in the prompt are phrased against capabilities ("if a
calculation tool is listed") rather than tool names. What transfers is the habit
of reading the tool card, not a memorised list of four names.

## Composition

| family | traces | what it teaches |
|---|---|---|
| `inspection_derive` | 520 | read the report, cite the row, derive through the calc tool |
| `calc_only` | 286 | figures already given — reading a file is the wrong first move |
| `lookup` | 312 | the document states it; do not over-tool |
| `deliverable` | 260 | approval note with sign-off block via doc_write |
| `unknown` | 208 | genuinely absent fact — answer UNKNOWN, invent nothing |
| `repair` | 208 | a tool call fails; recover instead of reporting failure |
| `multi_doc` | 208 | two reports, one comparison, tags intact |
| `clarify` | 156 | ambiguous request — ask one question, never guess the tag |
| `direct` | 130 | general knowledge; no tool call at all |
| `finqa` | 181 | real human-written derivations over real filing tables |

## Ground truth is computed, never typed

No answer here is hand-written. Inspection figures are recomputed from the
readings by the same code path the agent's `calc` tool uses, and cross-check
against it exactly — so a training target cannot drift from the tool the model
will actually call. The `finqa` family carries real human-written derivation
programs over real filing tables; single-step programs only, because a
mis-converted multi-step program would put a wrong answer beside a real table.

A note on what is **not** here: two public "oil & gas pipeline" tabular datasets
were evaluated as sources and rejected. One pairs a non-ferrous material with a
carbon-steel specification in 37% of rows (Fiberglass / API 5L X52) and has 119
rows where thickness loss exceeds the original wall; the other has geometrically
impossible dimensions in 62% of rows. Both are numerically self-consistent,
which is what makes them dangerous. Real inspection codes deserve better than
plausible-looking noise.

## Files

- `train.jsonl` — 2102 traces
- `eval.jsonl` — 367 held-out traces, stratified so every family appears

Synthetic traces over an authored inspection corpus. No proprietary documents.
"""

    return [
        DatasetSpec(
            slug="refinery-task-routing-exemplars",
            title="Refinery Task-Routing Exemplars (11-class intent)",
            subtitle="176 hand-labelled refinery/PSU work requests for deterministic model routing",
            description=router_readme,
            files=[ROOT / "config" / "routing_exemplars.jsonl"],
            extra={"README.md": router_readme},
        ),
        DatasetSpec(
            slug="agent-loop-tool-calling-traces",
                        # Kaggle caps dataset titles at 50 characters.
            title="Agent-Loop Tool-Calling Traces",
            subtitle="2469 tool-call traces over 663 tool subsets, for small-model agent fine-tuning",
            description=corpus_readme,
            files=[ROOT / "finetune" / "data" / "train.jsonl",
                   ROOT / "finetune" / "data" / "eval.jsonl"],
            extra={"README.md": corpus_readme},
        ),
    ]


def stage(spec: DatasetSpec, username: str) -> Path:
    """Lay out one dataset folder with the metadata Kaggle requires."""
    guard_paths(spec.files)
    missing = [p for p in spec.files if not p.exists()]
    if missing:
        raise SystemExit(f"REFUSED: {spec.slug} is missing {missing}")

    folder = STAGING / spec.slug
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for path in spec.files:
        shutil.copy2(path, folder / path.name)
    for name, text in spec.extra.items():
        (folder / name).write_text(text, encoding="utf-8")

    (folder / "dataset-metadata.json").write_text(json.dumps({
        "title": spec.title,
        "id": f"{username}/{spec.slug}",
        "subtitle": spec.subtitle,
        "description": spec.description,
        "licenses": [{"name": spec.licence}],
    }, indent=2), encoding="utf-8")
    return folder


def api() -> Any:
    from kaggle.api.kaggle_api_extended import KaggleApi

    client = KaggleApi()
    client.authenticate()
    return client


def push_dataset(spec: DatasetSpec, client: Any, username: str, public: bool) -> None:
    folder = stage(spec, username)
    size = sum(f.stat().st_size for f in folder.iterdir()) / 1e6
    print(f"\n[{spec.slug}] staged {len(list(folder.iterdir()))} files, {size:.1f} MB")
    try:
        existing = client.dataset_list(user=username, search=spec.slug)
        refs = {str(d.ref).split("/")[-1] for d in existing}
    except Exception:
        refs = set()

    if spec.slug in refs:
        print(f"[{spec.slug}] exists -> pushing a new version")
        client.dataset_create_version(str(folder), version_notes="update", dir_mode="zip")
    else:
        print(f"[{spec.slug}] creating ({'PUBLIC' if public else 'private'})")
        client.dataset_create_new(str(folder), public=public, dir_mode="zip")
    print(f"[{spec.slug}] -> https://www.kaggle.com/datasets/{username}/{spec.slug}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["list", "push", "push-kernel"])
    parser.add_argument("which", nargs="?", default="all")
    parser.add_argument("--public", action="store_true",
                        help="publish publicly (default: private, promote by hand)")
    args = parser.parse_args()

    all_specs = specs()
    if args.action == "list":
        for spec in all_specs:
            guard_paths(spec.files)
            total = sum(p.stat().st_size for p in spec.files if p.exists()) / 1e6
            print(f"{spec.slug:<40} {total:7.2f} MB  {len(spec.files)} files")
            for p in spec.files:
                mark = "ok" if p.exists() else "MISSING"
                print(f"    [{mark}] {p.relative_to(ROOT)}")
        return 0

    client = api()
    username = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]

    if args.action == "push":
        chosen = all_specs if args.which == "all" else [
            s for s in all_specs if args.which in s.slug
        ]
        if not chosen:
            raise SystemExit(f"no spec matches {args.which!r}")
        for spec in chosen:
            push_dataset(spec, client, username, args.public)
        return 0

    from finetune.kaggle_kernel import push_kernel

    push_kernel(client, username, public=args.public)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
