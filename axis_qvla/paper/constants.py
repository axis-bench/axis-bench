"""The numbers of reproducing the paper's tables, by name: the registry pattern of
axis_qvla.labels.constants; a constants page is generated from it (tests/docs/constants_page.py).

Core tier: stdlib only.
"""
from __future__ import annotations

from axis_qvla.labels.constants import Constant

CONSTANTS: tuple[Constant, ...] = (
    Constant("PAPER_RESULTS", 123, "implementation (not in the paper)",
             "evaluated results behind Table I and Figs. 1, 4(c) and 5-7; qvla paper tables recomputes each "
             "from its per-seed logs",
             "experiments/paper/cells.json", "len:cells"),
)
