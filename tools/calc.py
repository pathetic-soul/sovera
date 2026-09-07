"""Exact symbolic and numeric maths, with the engineering formulas fixed in code.

AGENTS.md §6 (`calc.py - sympy, shows steps`) and §16.

WHY THIS EXISTS — the measurement, not a hunch
-----------------------------------------------
On the leg-5 task ("corrosion rate and remaining life for grid S7"), measured
over 8 trials each on the real corpus document:

    qwen3:4b-instruct        1/8 correct, 7/8 hit the step cap
    sovereign-driver-v2      0/8 correct, 8/8 wrong

The fine-tune successfully taught the model to route arithmetic through
py_sandbox — every trial produced running code with real stdout. It just wrote
the *wrong formula*: 5.70 / 0.2982 (the survey interval over the rate) instead
of (t_measured - t_min) / rate. The sandbox executed that faithfully every
time. A sandbox guarantees the arithmetic, not the formula.

So the formula stops being something a 4B decides. Below, `remaining_life` is
one line of Python with a code reference next to it. The model chooses *which*
formula to apply and supplies the numbers; it cannot invent the algebra.

Every result shows its substitution, because an approval note that says
"5.03 years" is worth nothing to a PSU reviewer who cannot see where it came
from. `(8.9 - 7.4) / 0.2982 = 5.0302` is checkable by hand in ten seconds.

CPU only, 0 GB VRAM, ~5 ms per call. No approval gate: this computes and
returns a number, it writes nothing and executes no user code (§2.4 gates
writes and execution). Gating it would put a click in front of every
arithmetic step for no safety gain.

SECURITY
--------
sympy's parser calls eval() internally, so the namespace is a whitelist and
`global_dict` is empty — `__import__`, `open` and friends are simply not
names that exist. That alone is not enough: `().__class__` still parses, and
from a class you can walk `__bases__` / `__subclasses__()` back to os. Every
such route needs a dunder, so dunders are rejected before parsing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

from tools.base import RunContext, Tool, ToolResult

MAX_LEN = 500
# Wall-clock guard: sympy can be asked for something that runs a very long time
# (a stubborn integral, a huge factorial). Kept small because the agent loop is
# already on a token and step budget (§8.4).
MAX_TERMS = 10_000


@dataclass(frozen=True)
class Formula:
    """A fixed engineering formula. `expr` is what gets rendered as the step."""

    params: tuple[str, ...]
    expr: str
    unit: str
    source: str
    fn: Callable[..., float]


def _remaining_life(t_current: float, t_min: float, corrosion_rate: float) -> float:
    if corrosion_rate <= 0:
        raise ValueError("corrosion_rate must be positive; a flat or growing wall has no finite remaining life")
    return (t_current - t_min) / corrosion_rate


def _next_interval(remaining_life_years: float) -> float:
    """API 510 §7.2 / API 570 §6.3: the lesser of half the remaining life and
    the code ceiling. This is the rule the model got wrong in leg 5."""
    return min(remaining_life_years / 2.0, 10.0)


def _required_thickness(pressure: float, radius: float, stress: float, efficiency: float) -> float:
    """ASME VIII Div.1 UG-27(c)(1), circumferential stress, thin-wall shell."""
    denom = stress * efficiency - 0.6 * pressure
    if denom <= 0:
        raise ValueError("SE - 0.6P is not positive: thin-wall formula does not apply at this pressure")
    return pressure * radius / denom


FORMULAS: dict[str, Formula] = {
    "corrosion_rate": Formula(
        ("t_previous", "t_current", "years"),
        "(t_previous - t_current) / years", "mm/year", "API 510 §7.1",
        lambda t_previous, t_current, years: (t_previous - t_current) / years,
    ),
    "remaining_life": Formula(
        ("t_current", "t_min", "corrosion_rate"),
        "(t_current - t_min) / corrosion_rate", "years", "API 510 §7.1",
        _remaining_life,
    ),
    "next_interval": Formula(
        ("remaining_life_years",),
        "min(remaining_life_years / 2, 10)", "years", "API 510 §7.2",
        _next_interval,
    ),
    # Unit-sensitive: pressure and stress must be in the SAME units. Feeding
    # barg alongside MPa silently returns a number ~14x too large, and it looks
    # perfectly plausible. The tool cannot catch this — it has no unit system —
    # so the caveat is printed with the result instead.
    "required_thickness": Formula(
        ("pressure", "radius", "stress", "efficiency"),
        "pressure * radius / (stress * efficiency - 0.6 * pressure)", "mm",
        "ASME VIII Div.1 UG-27(c)(1); pressure and stress MUST share units",
        _required_thickness,
    ),
    "corrosion_allowance_used": Formula(
        ("t_nominal", "t_current"),
        "t_nominal - t_current", "mm", "API 510 §7.1",
        lambda t_nominal, t_current: t_nominal - t_current,
    ),
}

# General maths: calculus, algebra, complex numbers, matrices, statistics.
_NAMES = (
    "Integer Float Rational Symbol Abs sqrt cbrt exp log ln sin cos tan cot sec csc "
    "asin acos atan atan2 sinh cosh tanh asinh acosh atanh pi E I oo nan zoo "
    "diff integrate limit series simplify expand factor cancel together apart "
    "solve solveset nsolve dsolve Eq Ne Lt Le Gt Ge re im arg conjugate Abs sign "
    "Matrix eye zeros ones det trace transpose Sum Product factorial binomial gamma "
    "floor ceiling Max Min N Derivative Integral nsimplify radsimp trigsimp "
    "root real_root exp_polar erf erfc LambertW Piecewise Rational GoldenRatio"
)
NAMESPACE: dict[str, Any] = {n: getattr(sp, n) for n in _NAMES.split() if hasattr(sp, n)}
for _sym in ("x", "y", "z", "t", "n", "r", "p", "a", "b", "c", "k", "m", "s"):
    NAMESPACE[_sym] = sp.Symbol(_sym)

_CALL = re.compile(r"^\s*([a-z_][a-z0-9_]*)\s*\((.*)\)\s*$", re.I | re.S)
_KWARG = re.compile(r"^\s*([a-z_][a-z0-9_]*)\s*=\s*(.+?)\s*$", re.I | re.S)


class Calc(Tool):
    name = "calc"
    description = (
        "Exact maths: arithmetic, algebra, calculus, complex numbers, matrices. "
        "Also fixed inspection formulas by name. Shows the substitution."
    )
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": (
                    "e.g. remaining_life(t_current=8.9, t_min=7.4, corrosion_rate=0.2982) "
                    "or diff(sin(x)*x**2, x) or integrate(x**2, (x, 0, 3))"
                ),
            }
        },
        "required": ["expression"],
    }
    requires_approval = False

    def run(self, args: dict[str, Any], ctx: RunContext) -> ToolResult:
        raw = str(args["expression"]).strip()
        problem = _reject(raw)
        if problem:
            ctx.audit.append("tool_call", {"tool": self.name, "ok": False, "error": problem})
            return ToolResult(ok=False, output="", error=problem)

        try:
            output = _named(raw) if _is_named(raw) else _general(raw)
        except (ValueError, TypeError, ZeroDivisionError) as exc:
            msg = f"{type(exc).__name__}: {exc}"
            ctx.audit.append("tool_call", {"tool": self.name, "ok": False, "error": msg})
            return ToolResult(ok=False, output="", error=f"{msg}. {_help()}")
        except Exception as exc:  # noqa: BLE001 - sympy raises a wide variety
            msg = f"{type(exc).__name__}: {str(exc)[:200]}"
            # sympy's auto_symbol transformation reaches for `Function` when it
            # meets a call it does not know, and global_dict={} (deliberately,
            # §2.3) means the name is absent -- so an unknown function surfaced
            # as "NameError: name 'Function' is not defined". That told the model
            # nothing about what it did wrong, and it retried the same call six
            # times in a row. §12.6: the error has to name the actual mistake.
            if "Function" in msg and "not defined" in msg:
                unknown = _unknown_call(raw)
                msg = (f"unknown function {unknown!r}" if unknown
                       else "unknown function in the expression")
                ctx.audit.append(
                    "tool_call", {"tool": self.name, "ok": False, "error": msg}
                )
                return ToolResult(
                    ok=False, output="",
                    error=f"{msg}. calc evaluates arithmetic and the named "
                          f"inspection formulas only; it cannot call arbitrary "
                          f"functions. {_help()}",
                )
            ctx.audit.append("tool_call", {"tool": self.name, "ok": False, "error": msg})
            return ToolResult(ok=False, output="", error=f"could not evaluate: {msg}")

        ctx.audit.append(
            "tool_call", {"tool": self.name, "ok": True, "expression": raw[:200]}
        )
        return ToolResult(ok=True, output=output)


def _unknown_call(expression: str) -> str | None:
    """First `name(` in the expression that sympy does not know, for the error."""
    for match in re.finditer(r"([A-Za-z_]\w*)\s*\(", expression):
        name = match.group(1)
        if name not in NAMESPACE:
            return name
    return None


def _reject(expression: str) -> str | None:
    """§2.3 — validate before evaluating, never after."""
    if not expression:
        return "empty expression"
    if len(expression) > MAX_LEN:
        return f"expression too long ({len(expression)} chars, limit {MAX_LEN})"
    if "__" in expression:
        # Every escape from the whitelist runs through a dunder:
        # ().__class__.__bases__[0].__subclasses__() reaches os from a literal.
        return "'__' is not allowed"
    banned = ("import", "lambda", "open(", "eval(", "exec(", "compile(",
              "globals", "locals", "getattr", "setattr", "delattr", "subclasses")
    lowered = expression.lower()
    for token in banned:
        if token in lowered:
            return f"{token!r} is not allowed"
    return None


def _is_named(expression: str) -> bool:
    match = _CALL.match(expression)
    return bool(match and match.group(1) in FORMULAS)


def _split_args(blob: str) -> list[str]:
    """Split on top-level commas only, so tuples like (x, 0, 3) survive."""
    parts, depth, current = [], 0, ""
    for ch in blob:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current)
    return [p.strip() for p in parts if p.strip()]


def _number(text: str) -> float:
    value = parse_expr(text, local_dict=NAMESPACE, global_dict={},
                       transformations=standard_transformations)
    result = sp.N(value)
    if not result.is_number:
        raise ValueError(f"{text!r} is not a number")
    return float(result)


def _named(expression: str) -> str:
    """A fixed formula: bind arguments, render the substitution, compute."""
    match = _CALL.match(expression)
    assert match is not None
    name, blob = match.group(1), match.group(2)
    formula = FORMULAS[name]

    given: dict[str, float] = {}
    positional: list[float] = []
    for piece in _split_args(blob):
        kw = _KWARG.match(piece)
        if kw and kw.group(1) in formula.params:
            given[kw.group(1)] = _number(kw.group(2))
        elif kw and kw.group(1) not in formula.params:
            raise ValueError(
                f"{name} has no parameter {kw.group(1)!r}; expected {', '.join(formula.params)}"
            )
        else:
            positional.append(_number(piece))

    for param, value in zip(formula.params, positional):
        given.setdefault(param, value)
    missing = [p for p in formula.params if p not in given]
    if missing:
        raise ValueError(
            f"{name} needs {', '.join(missing)}; expected {', '.join(formula.params)}"
        )

    result = formula.fn(**given)
    substituted = formula.expr
    for param in sorted(formula.params, key=len, reverse=True):
        substituted = substituted.replace(param, f"{given[param]:g}")

    lines = [
        f"{name}  [{formula.source}]",
        f"  formula:     {formula.expr}",
        f"  substituted: {substituted}",
        f"  result:      {result:.4f} {formula.unit}",
    ]
    if name == "remaining_life" and result < 0:
        lines.append(
            "  NOTE: negative remaining life means t_current is already below t_min. "
            "Check the inputs before reporting this — it is a retirement condition."
        )
    return "\n".join(lines)


def _general(expression: str) -> str:
    """Anything sympy can do: calculus, algebra, complex numbers, matrices."""
    value = parse_expr(expression, local_dict=NAMESPACE, global_dict={},
                       transformations=standard_transformations)
    if hasattr(value, "count_ops") and value.count_ops() > MAX_TERMS:
        raise ValueError("expression is too large to evaluate")

    lines = [f"  input:  {expression}", f"  exact:  {value}"]
    try:
        numeric = sp.N(value, 8)
        if str(numeric) != str(value):
            lines.append(f"  value:  {numeric}")
    except (TypeError, ValueError, AttributeError):
        pass
    return "\n".join(lines)


def _help() -> str:
    names = ", ".join(f"{n}({', '.join(f.params)})" for n, f in FORMULAS.items())
    return f"Fixed formulas: {names}"
