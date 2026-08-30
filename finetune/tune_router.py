"""Select the hybrid router's blend weight by LOO-CV on the training split.

Build time only (§3): this never runs during a demo. It picks one float.

**The discipline this file exists to enforce.** tests/test_router.py holds out
every 3rd exemplar within each class and measures once, at the end. The moment
a hyperparameter is chosen by looking at that held-out set, the number stops
being an estimate of unseen performance and becomes a description of the tuning
run — and the headline claim in README.md ("router accuracy is N%") becomes a
sentence that cannot survive a reviewer asking how it was obtained.

So: the sweep below sees TRAIN only. Held-out scoring is behind an explicit
`--final` flag with a banner, so that touching it is a deliberate act that
shows up in the shell history rather than a default that happens quietly.

    python -m finetune.tune_router              # sweep on TRAIN (safe, repeatable)
    python -m finetune.tune_router --final 0.5  # score held-out ONCE at chosen w

Why leave-one-out rather than k-fold: at ~10 training exemplars per class,
5-fold removes a fifth of a class at a time and the centroid it leaves behind
is measurably worse than the one the shipped router will build from the full
set. LOO changes the training set by one row, so the model being validated is
the closest available stand-in for the model being shipped.
"""

from __future__ import annotations

import argparse

import numpy as np

from core.embed import get_encoder
from core.routing import ExemplarScorer, allowed_routes, load_exemplars
from core.routing.features import _standardise

# The split is imported, never reimplemented. This file used to carry its own
# copy, so the weight was selected on one partition and the headline accuracy
# reported against another that merely looked identical. See core/routing/split.py.
from core.routing.split import split

WEIGHTS = [round(w, 2) for w in np.arange(0.0, 1.01, 0.1)]


def _dense_loo_scores(
    vectors: np.ndarray, labels: list[str], i: int, allowed: tuple[str, ...] | None
) -> dict[str, float]:
    """Dense class scores for row `i`, from centroids built without row `i`.

    Reuses the embeddings computed once for the whole split rather than
    re-encoding each fold: encoding dominates the runtime, and recomputing a
    centroid over the remaining rows is arithmetically identical to rebuilding
    DenseScorer on the reduced set.
    """
    classes = sorted(set(labels))
    candidates = classes if allowed is None else [c for c in classes if c in allowed]
    query = vectors[i]
    out: dict[str, float] = {}
    for cls in candidates:
        idx = [j for j, lab in enumerate(labels) if lab == cls and j != i]
        if not idx:
            continue
        centroid = vectors[idx].mean(axis=0)
        norm = float(np.linalg.norm(centroid)) or 1.0
        out[cls] = float(query @ (centroid / norm))
    return out


def collect_fold_scores(
    rows: list[tuple[str, str]],
) -> list[tuple[str, dict[str, float], dict[str, float]]]:
    """One LOO pass. Returns (truth, lexical_scores, dense_scores) per row.

    Weight-independent by construction, so the sweep afterwards is free: the
    expensive half (encoding + 110 lexical rebuilds) happens exactly once.
    """
    prompts = [p for p, _ in rows]
    labels = [t for _, t in rows]
    vectors = get_encoder().encode(prompts)

    folds: list[tuple[str, dict[str, float], dict[str, float]]] = []
    for i, (prompt, truth) in enumerate(rows):
        allowed, _ = allowed_routes(prompt, "text")
        rest = [r for j, r in enumerate(rows) if j != i]
        lex = ExemplarScorer(rest).score(prompt, allowed)
        dense = _dense_loo_scores(vectors, labels, i, allowed)
        folds.append((truth, lex, dense))
    return folds


def accuracy_at(
    folds: list[tuple[str, dict[str, float], dict[str, float]]], weight: float
) -> float:
    hits = 0
    for truth, lex, dense in folds:
        zl, zd = _standardise(lex), _standardise(dense)
        blended = {
            label: (1.0 - weight) * zl.get(label, 0.0) + weight * zd.get(label, 0.0)
            for label in zl.keys() | zd.keys()
        }
        if blended and max(blended, key=lambda k: blended[k]) == truth:
            hits += 1
    return hits / len(folds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", type=float, default=None, metavar="W",
                        help="score the HELD-OUT set once at weight W (deliberate act)")
    args = parser.parse_args()

    rows = load_exemplars()
    train, held = split(rows)
    print(f"exemplars {len(rows)}  train {len(train)}  held-out {len(held)}")

    if args.final is not None:
        print("\n" + "=" * 66)
        print("  FINAL MEASUREMENT — scoring the HELD-OUT set.")
        print("  Every hyperparameter must already be frozen. Do not tune after this.")
        print("=" * 66)
        from core.routing import Classifier, HybridScorer

        clf = Classifier(HybridScorer(train, weight=args.final))
        wrong = [(p, t, clf.classify(p).task_type) for p, t in held
                 if clf.classify(p).task_type != t]
        acc = 1 - len(wrong) / len(held)
        print(f"\nHELD-OUT ACCURACY @ w={args.final}: {acc:.1%}  ({len(held) - len(wrong)}/{len(held)})")
        if wrong:
            print("\nmisses:")
            for prompt, want, got in wrong:
                print(f"  want {want:<19} got {got:<19} {prompt[:58]}")
        return 0

    print("collecting LOO folds on TRAIN (encoding once, then sweeping)...")
    folds = collect_fold_scores(train)

    print(f"\n{'weight':>8}  {'LOO acc':>8}   (0.0 = pure lexical, 1.0 = pure dense)")
    best_w, best_acc = 0.0, -1.0
    for w in WEIGHTS:
        acc = accuracy_at(folds, w)
        marker = ""
        if acc > best_acc:
            best_w, best_acc, marker = w, acc, "  <-- best so far"
        print(f"{w:>8.1f}  {acc:>7.1%}{marker}")

    print(f"\nBEST on TRAIN: weight={best_w}  LOO accuracy={best_acc:.1%}")
    print(f"lexical-only baseline: {accuracy_at(folds, 0.0):.1%}")
    print(f"dense-only  baseline: {accuracy_at(folds, 1.0):.1%}")
    print(f"\nNext: python -m finetune.tune_router --final {best_w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
