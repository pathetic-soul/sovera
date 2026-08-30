"""The router's train / held-out partition and its accuracy measurement.

AGENTS.md §13 sets a >=90% gate on held-out router accuracy, and §9.2 asks for
the number to be quoted on a slide. This module is the single definition of what
"held-out" means, because a measurement split that exists in more than one place
is a measurement that can silently stop meaning what it says.

CPU only, 0 GB VRAM. Importing this module does not load an encoder — it reads
the exemplar file and partitions strings. `held_out_accuracy` costs whatever the
classifier it is handed costs.

WHY THIS IS NOT IN tests/
-------------------------
It was, and `gates.py` imported it from there. Three things were wrong with that:

1. `gates.py` is production tooling — `docs/DEMO-DAY.md` tells you to run it on
   stage morning. It must not stop working because the test suite was
   reorganised, and `tests/` is not even an importable package (no `__init__`;
   it worked only because `gates.py` inserts ROOT on `sys.path`).
2. `finetune/tune_router.py` reimplemented the same split by hand, so the weight
   was selected on one partition and the accuracy reported against another
   partition that merely looked the same.
3. A test existed whose whole job was asserting (1) and (2) had not diverged.
   That test is deleted by this module existing.

THE DISCIPLINE THIS ENCODES
---------------------------
The split is deterministic and **stratified**: every 3rd exemplar *within each
class* is held out, so each task_type contributes proportionally and no class
can vanish from the measurement. The scorer never sees a held-out prompt.

Hyperparameters — the scorer variant and the hybrid blend weight — were chosen
by leave-one-out CV on TRAIN alone, and HELD_OUT was scored once, at the end.
That is the entire reason the number is worth quoting. Anything that tunes
against `HELD_OUT` converts an estimate of unseen performance into a description
of the tuning run.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from core.routing.classifier import load_exemplars
from core.routing.types import Row

# §13. The gate the held-out number is measured against, not the number itself.
ACCURACY_GATE = 0.90


class SupportsClassify(Protocol):
    """What `held_out_accuracy` needs. Deliberately narrower than `Classifier`.

    Typing the parameter structurally is what lets `gates.py` and the tuner
    measure any scorer arrangement — lexical, dense, hybrid — without this
    module importing the classifier's concrete construction path.
    """

    def classify(self, text: str, attachments: None = None) -> object: ...


def split(rows: list[Row]) -> tuple[list[Row], list[Row]]:
    """Partition exemplars into (train, held_out), stratified by task_type.

    Every 3rd exemplar within a class goes to held-out. Deterministic: the same
    exemplar file always produces the same partition, which is what makes an
    accuracy number reproducible by someone who did not run it.
    """
    by_class: defaultdict[str, list[str]] = defaultdict(list)
    for prompt, label in rows:
        by_class[label].append(prompt)
    held = [(p, t) for t, ps in by_class.items() for i, p in enumerate(ps) if i % 3 == 0]
    train = [(p, t) for t, ps in by_class.items() for i, p in enumerate(ps) if i % 3 != 0]
    return train, held


ROWS: list[Row] = load_exemplars()
TRAIN, HELD_OUT = split(ROWS)


def held_out_accuracy(clf: SupportsClassify) -> tuple[float, list[tuple[str, str, str]]]:
    """(accuracy, misses) over HELD_OUT. Each miss is (prompt, wanted, got).

    The misses are returned rather than counted because the failure *pattern* is
    the actionable part — the current 13 are dominated by subject matter
    overriding intent, which points at corpus size rather than at tuning.
    """
    wrong: list[tuple[str, str, str]] = []
    for prompt, want in HELD_OUT:
        got = str(getattr(clf.classify(prompt), "task_type"))
        if got != want:
            wrong.append((prompt, want, got))
    return 1 - len(wrong) / len(HELD_OUT), wrong
