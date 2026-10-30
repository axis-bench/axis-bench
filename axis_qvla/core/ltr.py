"""Left-to-right summation: the one float sum T0 and the scorer use, identical on every Python >= 3.11.

Python 3.12 made the builtin sum() over floats compensated (Neumaier), so the same data summed on
3.11 and on 3.12 can differ in the last bit -- enough to change the pooled cells' bytes and fail
`paper tables --check` against goldens frozen on 3.11. ltr_sum is a plain `acc = acc + x` loop
from `start`, which is exactly what 3.11's sum did (its float fast path adds C doubles left to
right, and an int item is converted to float first, as `float + int` does here). The published
goldens are 3.11 bytes, so this keeps them, on every version.
tests/golden/paper/test_t0_version_independence.py keeps every float sum on the T0 path here.

Core tier: stdlib only, Python >= 3.11.
"""
from __future__ import annotations

from collections.abc import Iterable


def ltr_sum(xs: Iterable[float], start: float = 0) -> float:
    """sum(xs, start) as Python 3.11 computed it: strictly left to right, no compensation."""
    acc = start
    for x in xs:
        acc = acc + x
    return acc
