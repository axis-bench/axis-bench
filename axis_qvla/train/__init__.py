"""Training: run specs, recipe presets and `qvla train`, for pi0.5 (openpi), SmolVLA and X-VLA (LeRobot).

A run spec (axis_qvla.train.armspec) names one run by content; a recipe holds the hyper-parameters it
names by preset id; `qvla train <run_spec>` renders the pair into the backbone's own config and
refuses, before any GPU time, a run whose compiled labels do not fit it.

Core tier: stdlib only at import; a numpy user in this package imports it inside the function
that needs it, never at module scope.
"""
from __future__ import annotations
