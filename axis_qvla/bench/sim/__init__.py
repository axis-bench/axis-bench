"""Closed-loop rollout evaluation: one runner over the sim domains and policy sources.

Import the submodules directly; this package deliberately re-exports nothing, so that importing
`axis_qvla.bench.sim.arms` (stdlib only) never drags in a rollout backend that needs mujoco, torch or
jax.
"""
from __future__ import annotations
