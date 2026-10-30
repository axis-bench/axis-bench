"""Reproducing the paper's tables: the paper's simulation results -- Figs. 1, 4(c), 5-7, Table I and
the in-text numbers -- rebuilt from published per-seed rollout files.

Why this package exists: every simulation number of the paper must be recomputed byte-for-byte from
the released seed files by anyone, on any CPU, with a bare interpreter -- no GPU, no ML framework. The layout
of what the paper shows is data (experiments/paper/paper.json); the code is a date-free pipeline.

Core tier: stdlib only at module level (plus first-party modules under the import root), Python >= 3.11
-- enforced by tests/ratchets/test_venv_tiers.py.
"""
from __future__ import annotations
