"""The benchmark must be trustworthy before any number it produces is.

It replaces a gate that scored one task over eight trials and, on this repo's
own evidence, measured the same base model at 7/8 and then 5/8. These tests
guard the three properties that make the replacement worth having: the task set
is genuinely held out, the statistics are right, and the scoring cannot be
gamed by a confident wrong answer.

No model and no GPU.
"""

from __future__ import annotations

import json

import pytest

from finetune.benchmark import (BENCH_DOCS, BENCH_SEED, TOLERANCE, VERDICTS,
                                Result, Task, build_tasks, numbers_in, report,
                                run_benchmark, score, separated, wilson)
from finetune.corpus_inspection import SEED as TRAIN_SEED
from finetune.corpus_inspection import generate

TASKS = build_tasks()


# --- held out, and provably so ---------------------------------------------

def test_no_benchmark_document_appears_in_the_training_corpus() -> None:
    """The whole point. If a benchmark document were also a training document,
    the score would measure recall rather than generalisation, and every
    comparison against base would be meaningless."""
    assert BENCH_SEED != TRAIN_SEED, "benchmark must not reuse the training seed"
    train = {f"{r.doc_id}-{r.tag}" for r in generate(60, seed=TRAIN_SEED)}
    bench = {f"{r.doc_id}-{r.tag}" for r in generate(BENCH_DOCS, seed=BENCH_SEED)}
    assert not (train & bench), f"leaked documents: {sorted(train & bench)[:5]}"


def test_the_task_set_is_large_enough_to_resolve_a_real_difference() -> None:
    """8 samples gives a 95% interval so wide it spans most of the range. The
    old gate's verdicts were inside its own noise."""
    assert len(TASKS) >= 200, f"only {len(TASKS)} tasks"
    lo, hi = wilson(int(len(TASKS) * 0.7), len(TASKS))
    assert (hi - lo) < 0.15, "interval still too wide to act on"


def test_every_category_is_represented() -> None:
    cats = {t.category for t in TASKS}
    # derive_interval was split into interval_ceiling / interval_halflife once a
    # constant "10 years" answerer was measured at 15/15 on the ceiling half.
    for expected in ("derive_rate", "derive_life", "interval_ceiling",
                     "interval_halflife", "lookup", "unknown"):
        assert expected in cats, f"missing category {expected}"


def test_truth_values_are_computed_not_typed() -> None:
    """Every numeric task's truth must reconcile against the document's own
    readings through the same formulas tools/calc.py uses."""
    from finetune.corpus_inspection import generate as gen
    facts = {f"inbox/{r.doc_id}-{r.tag}.md": r.facts for r in gen(BENCH_DOCS, seed=BENCH_SEED)}
    checked = 0
    for t in TASKS:
        if t.truth is None or t.category != "derive_life":
            continue
        f = facts[t.doc]
        assert abs(t.truth - f["remaining_life_years"]) < 1e-6
        checked += 1
    assert checked > 10


# --- statistics -------------------------------------------------------------

def test_wilson_stays_inside_zero_and_one() -> None:
    """The normal approximation reports intervals below 0% at the extremes;
    several categories here sit near a boundary."""
    for successes, total in ((0, 30), (30, 30), (1, 240), (239, 240)):
        lo, hi = wilson(successes, total)
        assert 0.0 <= lo <= hi <= 1.0, (successes, total, lo, hi)


def test_wilson_narrows_as_samples_grow() -> None:
    wide = wilson(6, 8)
    narrow = wilson(180, 240)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0]) * 2


def test_separation_requires_non_overlapping_intervals() -> None:
    def res(correct: int, n: int) -> Result:
        r = Result(ref="x", runs=1)
        from finetune.benchmark import Outcome
        r.outcomes = [Outcome("t", "c", "correct" if i < correct else "wrong", "")
                      for i in range(n)]
        return r

    # An 8-sample "5 vs 7" — the old gate's exact verdict — must NOT separate.
    assert not separated(res(5, 8), res(7, 8))
    # A large, real gap must.
    assert separated(res(120, 240), res(200, 240))


def test_empty_result_does_not_divide_by_zero() -> None:
    assert wilson(0, 0) == (0.0, 0.0)
    assert Result(ref="x", runs=1).accuracy == 0.0


# --- scoring ----------------------------------------------------------------

def _task(**kw: object) -> Task:
    base = dict(id="t", category="derive_life", prompt="p", truth=5.0,
                doc="inbox/x.md", context="c", unit="years")
    base.update(kw)
    return Task(**base)  # type: ignore[arg-type]


def test_a_correct_number_within_tolerance_scores_correct() -> None:
    assert score(_task(), "Remaining life is 5.03 years.").verdict == "correct"
    assert score(_task(), f"{5.0 * (1 + TOLERANCE / 2):.4f} years").verdict == "correct"


def test_a_confident_wrong_number_scores_wrong_not_correct() -> None:
    out = score(_task(), "The remaining life is clearly 19.11 years.")
    assert out.verdict == "wrong"


def test_an_empty_or_numberless_answer_scores_failed_not_wrong() -> None:
    """'I could not do this' and 'here is the wrong number' are different
    failures and must not be averaged together."""
    assert score(_task(), "").verdict == "failed"
    assert score(_task(), "I am unable to determine this.").verdict == "failed"


def test_an_unsafe_claim_outranks_merely_being_wrong() -> None:
    """The only outcome that would end a demo — or a shift — badly."""
    t = _task(unsafe_if=("already exceeded",))
    out = score(t, "The wall is 2.1 mm and the limit is already exceeded; withdraw it.")
    assert out.verdict == "unsafe"


def test_unknown_tasks_require_the_word_and_reject_a_fabricated_answer() -> None:
    t = _task(truth=None, category="unknown", must_say=("UNKNOWN",))
    assert score(t, "UNKNOWN — the report does not state it.").verdict == "correct"
    assert score(t, "The hydrotest pressure was 15.75 barg.").verdict == "wrong"


def test_a_required_substring_must_be_present() -> None:
    """The grid citation. Answering about the right value from the wrong row is
    the §16 failure that started all of this."""
    t = _task(must_say=("S7",))
    assert score(t, "5.03 years at grid S7.").verdict == "correct"
    assert score(t, "5.03 years at grid S5.").verdict == "wrong"


def test_numbers_are_extracted_with_thousands_separators() -> None:
    assert 21026.06 in numbers_in("Total income is 21,026.06 crore")


# --- harness ----------------------------------------------------------------

def test_run_benchmark_is_backend_agnostic_and_counts_every_sample() -> None:
    tasks = TASKS[:5]
    calls: list[int] = []

    def fake(messages: list[dict[str, str]]) -> str:
        calls.append(1)
        return "5.0 years"

    res = run_benchmark("fake", fake, tasks, runs=3, progress=False)
    assert res.n == len(tasks) * 3 == len(calls)
    assert sum(res.tally().values()) == res.n


def test_a_generate_that_raises_is_recorded_as_failed_not_lost() -> None:
    def boom(messages: list[dict[str, str]]) -> str:
        raise RuntimeError("backend down")

    res = run_benchmark("boom", boom, TASKS[:4], runs=1, progress=False)
    assert res.n == 4
    assert res.tally()["failed"] == 4


def test_report_names_overlap_honestly() -> None:
    from finetune.benchmark import Outcome

    def res(correct: int, n: int, ref: str) -> Result:
        r = Result(ref=ref, runs=1)
        r.outcomes = [Outcome("t", "derive_life", "correct" if i < correct else "wrong", "")
                      for i in range(n)]
        return r

    text = report(res(122, 240, "cand"), res(120, 240, "base"))
    assert "NOT SEPARATED" in text, "a 2-sample gap must not be reported as a win"


# --- the degenerate-strategy floor -----------------------------------------

def test_a_constant_answer_cannot_score_well_overall() -> None:
    """A model that reads nothing and always says '10 years' must be visibly
    bad. Measured base failure on interval tasks IS that constant answer, so
    without this floor the failure reads as partial competence."""
    from finetune.benchmark import constant_generate

    res = run_benchmark("constant", constant_generate("10 years"), TASKS, 1, progress=False)
    assert res.accuracy < 0.15, (
        f"a model that understood nothing scored {res.accuracy:.1%}; the task "
        f"set is gameable")


def test_interval_tasks_are_split_by_which_branch_binds() -> None:
    """Merged, the ceiling half is free for a model that always answers 10 —
    measured at 15/15 for a constant answerer, and 93% for the base model
    against 48% on the other half. Collapsing them would report that as ~64%
    'accuracy' and hide the fact that nothing was understood."""
    cats = {t.category for t in TASKS}
    assert "interval_ceiling" in cats and "interval_halflife" in cats, \
        "interval tasks must stay split by binding branch"
    assert "derive_interval" not in cats, "the merged category is gameable"

    from finetune.benchmark import constant_generate
    res = run_benchmark("constant", constant_generate("10 years"), TASKS, 1, progress=False)
    by_cat = res.by_category()
    ceil_c, ceil_n = by_cat["interval_ceiling"]
    half_c, half_n = by_cat["interval_halflife"]
    # The split is only useful if it actually separates the two behaviours.
    assert ceil_c / ceil_n > 0.9, "ceiling tasks should be free for a constant 10"
    assert half_c / half_n < 0.1, "half-life tasks must not be"


def test_both_interval_branches_are_represented() -> None:
    """If every interval task had the same binding branch, the category would
    teach and measure only half the rule."""
    by_cat: dict[str, int] = {}
    for t in TASKS:
        by_cat[t.category] = by_cat.get(t.category, 0) + 1
    ceil, half = by_cat["interval_ceiling"], by_cat["interval_halflife"]
    assert ceil >= 5 and half >= 5, f"lopsided: {ceil} ceiling, {half} half-life"
