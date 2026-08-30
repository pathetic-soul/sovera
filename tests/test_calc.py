"""calc: fixed formulas, general maths, and the sandbox escape it must refuse.

The formula tests are the point of the tool. Over 8 trials on the leg-5 task
both the base and fine-tuned driver produced the wrong remaining-life formula
(1/8 and 0/8 correct) and py_sandbox executed it faithfully. These assertions
are what make that failure impossible: the algebra now lives in code, and if
someone edits it, the suite fails.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.audit import AuditLog
from tools.base import RunContext
from tools.calc import FORMULAS, Calc


@pytest.fixture
def ctx(tmp_path: Path) -> RunContext:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return RunContext(workspace=ws, audit=AuditLog(ws / ".audit" / "a.jsonl", "t"), session_id="t")


def run(ctx: RunContext, expression: str) -> tuple[bool, str]:
    result = Calc().run({"expression": expression}, ctx)
    return result.ok, (result.output if result.ok else (result.error or ""))


# --- the formulas that caused leg 5 -----------------------------------------

def test_remaining_life_is_the_v2301_answer(ctx: RunContext) -> None:
    """The exact number both models failed to produce. Truth is 5.0302 years;
    the fine-tuned model answered 19.11 (interval / rate) in 7 of 8 trials."""
    ok, out = run(ctx, "remaining_life(t_current=8.9, t_min=7.4, corrosion_rate=0.2982)")
    assert ok
    assert "5.030" in out
    assert "19.1" not in out


def test_corrosion_rate_matches_the_report(ctx: RunContext) -> None:
    ok, out = run(ctx, "corrosion_rate(t_previous=10.6, t_current=8.9, years=5.70)")
    assert ok and "0.2982" in out


def test_next_interval_is_half_remaining_life_capped_at_ten(ctx: RunContext) -> None:
    ok, out = run(ctx, "next_interval(remaining_life_years=5.03)")
    assert ok and "2.5150" in out
    ok, out = run(ctx, "next_interval(remaining_life_years=40)")
    assert ok and "10.0000" in out, "API 510 caps the interval at 10 years"


def test_every_result_shows_its_substitution(ctx: RunContext) -> None:
    """An approval note that says '5.03 years' is worthless to a reviewer who
    cannot see where it came from."""
    ok, out = run(ctx, "remaining_life(t_current=8.9, t_min=7.4, corrosion_rate=0.2982)")
    assert ok
    assert "formula:" in out and "substituted:" in out
    assert "(8.9 - 7.4) / 0.2982" in out
    assert "API 510" in out


def test_positional_and_keyword_arguments_agree(ctx: RunContext) -> None:
    _, kw = run(ctx, "remaining_life(t_current=8.9, t_min=7.4, corrosion_rate=0.2982)")
    _, pos = run(ctx, "remaining_life(8.9, 7.4, 0.2982)")
    assert "5.030" in kw and "5.030" in pos


def test_negative_remaining_life_is_flagged_not_silently_returned(ctx: RunContext) -> None:
    """8.9 above a 7.4 limit is healthy; below it is a retirement condition and
    must not slide past as just another number."""
    ok, out = run(ctx, "remaining_life(t_current=7.0, t_min=7.4, corrosion_rate=0.3)")
    assert ok and "NOTE" in out and "retirement" in out


def test_zero_corrosion_rate_is_refused_not_infinite(ctx: RunContext) -> None:
    ok, err = run(ctx, "remaining_life(t_current=8.9, t_min=7.4, corrosion_rate=0)")
    assert not ok and "positive" in err


def test_missing_parameter_names_what_is_missing(ctx: RunContext) -> None:
    ok, err = run(ctx, "remaining_life(t_current=8.9)")
    assert not ok and "t_min" in err and "corrosion_rate" in err


def test_unknown_parameter_is_rejected(ctx: RunContext) -> None:
    ok, err = run(ctx, "remaining_life(thickness=8.9, t_min=7.4, corrosion_rate=0.3)")
    assert not ok and "thickness" in err


@pytest.mark.parametrize("name", sorted(FORMULAS))
def test_every_formula_cites_a_code_reference(name: str) -> None:
    """A number an engineer cannot trace to a clause is not usable in a PSU
    approval note."""
    assert FORMULAS[name].source.strip()
    assert FORMULAS[name].unit.strip()


# --- general maths ----------------------------------------------------------

@pytest.mark.parametrize(
    "expression,expected",
    [
        ("diff(sin(x)*x**2, x)", "cos(x)"),
        ("integrate(x**2, (x, 0, 3))", "9"),
        ("limit(sin(x)/x, x, 0)", "1"),
        ("solve(Eq(x**2 - 4, 0), x)", "-2"),
        ("(3 + 4*I)*(1 - 2*I)", "11.0 - 2.0*I"),
        ("Abs(3 + 4*I)", "5"),
        ("Matrix([[1,2],[3,4]]).det()", "-2"),
        ("series(exp(x), x, 0, 4)", "O(x**4)"),
        ("sqrt(2)", "1.414"),
        ("(8.9 - 7.4) / 0.2982", "5.030"),
    ],
)
def test_general_maths(ctx: RunContext, expression: str, expected: str) -> None:
    ok, out = run(ctx, expression)
    assert ok, out
    assert expected in out


def test_exact_before_approximate(ctx: RunContext) -> None:
    """sqrt(2) is sqrt(2), not 1.414. The exact form is shown first because
    rounding early is how thickness maths goes wrong."""
    ok, out = run(ctx, "sqrt(2)")
    assert ok and "sqrt(2)" in out and "1.414" in out


# --- security ---------------------------------------------------------------

@pytest.mark.parametrize(
    "attack",
    [
        "().__class__.__bases__[0].__subclasses__()",   # the classic escape
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "eval('1+1')",
        "exec('x=1')",
        "globals()",
        "getattr(sqrt, 'func')",
        "(lambda: 1)()",
    ],
)
def test_sandbox_escapes_are_refused(ctx: RunContext, attack: str) -> None:
    """sympy's parser calls eval() internally, so the namespace is a whitelist
    with an empty global_dict. That alone leaves `().__class__` reachable, and
    from a class you can walk back to os — hence dunders are rejected outright."""
    ok, err = run(ctx, attack)
    assert not ok, f"escape was ALLOWED: {attack}"
    assert err


def test_absurdly_long_input_is_refused(ctx: RunContext) -> None:
    ok, err = run(ctx, "1+" * 400 + "1")
    assert not ok and "too long" in err


def test_empty_expression_is_refused(ctx: RunContext) -> None:
    ok, err = run(ctx, "   ")
    assert not ok


def test_calc_needs_no_approval_and_writes_nothing(ctx: RunContext) -> None:
    """§2.4 gates writes and execution. calc returns a number and touches no
    file, so gating it would add a click per arithmetic step for no safety."""
    assert Calc().requires_approval is False
    before = set(ctx.workspace.rglob("*"))
    run(ctx, "remaining_life(8.9, 7.4, 0.2982)")
    assert {p for p in ctx.workspace.rglob("*") if ".audit" not in p.parts} <= before


def test_calls_are_audited(ctx: RunContext) -> None:
    run(ctx, "corrosion_rate(10.6, 8.9, 5.7)")
    records = [r for r in ctx.audit.tail() if r.payload.get("tool") == "calc"]
    assert records and records[-1].payload["ok"] is True
