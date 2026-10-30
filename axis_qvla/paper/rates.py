"""The arithmetic of every number the paper prints, from the pooled cells.

Why: the paper reports success rates (SR, mean task success, %), differences in percentage points,
relative improvements 100(SR/SR_BC - 1) computed from UNROUNDED rates, BC-normalized differences
100(SR_A - SR_B)/SR_BC (Table I), and marginal SEs scaled by 100/SR_BC (Appendix A).

SR, every difference, ratio and Table I mean is computed EXACTLY from the per-task tallies (as
fractions) and rounded to a float once: a tie is exactly 0, never a last-bit residue of two float
sums, and the paper's "one tie" stays a tie even when recomputed from the printed SRs. (The pooled
cell's `agg` is a float sum, kept bit-for-bit in pooled_r1.json, and may differ from SR in the
last bit.) SE is the pooled cell's `se`, the task-stratified binomial SE
100 sqrt(sum_t p_t (1 - p_t) / n_t) / n_tasks; the SE of a difference of two cells is
sqrt(SE_a^2 + SE_b^2). Only square roots are taken in floating point (math.sqrt is correctly
rounded), so every output is byte-identical on Python 3.11-3.13. A missing cell gives None, which
every caller renders as an empty field.

Core tier: stdlib only.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from fractions import Fraction

from axis_qvla.core.ltr import ltr_sum

DASH = "—"


def sr(P: Mapping[str, dict], cell: str) -> float | None:
    """SR in %, the exact mean of the per-task rates rounded once."""
    x = exact(P, cell)
    return None if x is None else float(100 * x)


def se(P: Mapping[str, dict], cell: str) -> float | None:
    return P[cell]["se"] if cell in P else None


def se_diff(P: Mapping[str, dict], a: str, b: str) -> float | None:
    """The marginal SE of SR(a) - SR(b), two independent evaluations: sqrt(SE_a^2 + SE_b^2)."""
    x, y = se(P, a), se(P, b)
    return None if x is None or y is None else math.sqrt(x * x + y * y)


def exact(P: Mapping[str, dict], cell: str) -> Fraction | None:
    """SR / 100 as an exact fraction (the mean of the per-task rates); None when absent."""
    if cell not in P:
        return None
    pt = P[cell]["per_task"]
    return ltr_sum((Fraction(s, n) for s, n in pt.values()), Fraction(0)) / len(pt)


def sign_of(d: Fraction | None) -> int | None:
    return None if d is None else (d > 0) - (d < 0)


def sign(P: Mapping[str, dict], a: str, b: str) -> int | None:
    """+1, 0 or -1 for SR(a) - SR(b), decided exactly; None when a cell is missing."""
    x, y = exact(P, a), exact(P, b)
    return None if x is None or y is None else sign_of(x - y)


def diff(P: Mapping[str, dict], a: str, b: str) -> float | None:
    """SR(a) - SR(b) in percentage points."""
    x, y = exact(P, a), exact(P, b)
    return None if x is None or y is None else float(100 * (x - y))


def rel(P: Mapping[str, dict], a: str, ref: str) -> float | None:
    """100 (SR(a) / SR(ref) - 1): the relative improvement over a reference cell."""
    x, y = exact(P, a), exact(P, ref)
    return None if x is None or not y else float(100 * (x / y - 1))


def norm_diff(P: Mapping[str, dict], a: str, b: str, bc: str) -> Fraction | None:
    """100 (SR(a) - SR(b)) / SR(bc), exact (a Table I value)."""
    x, y, z = exact(P, a), exact(P, b), exact(P, bc)
    return None if x is None or y is None or not z else 100 * (x - y) / z


def scaled(x: float | None, ref: float | None) -> float | None:
    """x on the reference's relative scale, 100 x / ref (an SE)."""
    return None if x is None or not ref else 100 * x / ref


def mean(xs: Sequence[Fraction | None]) -> Fraction | None:
    if not xs or any(x is None for x in xs):
        return None
    return ltr_sum(xs, Fraction(0)) / len(xs)


def sample_sd(xs: Sequence[Fraction | None]) -> float | None:
    m = mean(xs)
    if m is None or len(xs) < 2:
        return None
    return math.sqrt(float(ltr_sum(((x - m) ** 2 for x in xs), Fraction(0)) / (len(xs) - 1)))


def as_float(x: Fraction | None) -> float | None:
    return None if x is None else float(x)


def shown(text: str) -> str:
    """A formatted number as the paper typesets it, with the minus sign U+2212. Only the human
    renderings (Markdown, HTML) use it; claims.json and the CSVs keep ASCII so float() parses them."""
    return "−" + text[1:] if text.startswith("-") else text


def fmt(x: float | None, dp: int, *, signed: bool = False) -> str:
    """The paper's way: fixed decimals, a sign on differences, and an unsigned zero."""
    if x is None:
        return DASH
    text = f"{x:.{dp}f}"
    if float(text) == 0:
        return f"{0:.{dp}f}"
    return f"{x:+.{dp}f}" if signed else text
