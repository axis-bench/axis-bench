"""Every global the dual-sim rollout body references must actually resolve.

A NameError here is not a normal bug: the module IMPORTS fine, so nothing catches it until a
rollout is already running against a served policy -- i.e. after adaptation has been paid for and
a GPU is held open. Exactly that happened: commit 8027c1c added a call to
`observation_for_obs_key` without its import, and the dual-sim path raised on its first
observation for every seed from then on.

The check is static so it needs no MuJoCo, no GPU and no robosuite -- it runs in any venv.
"""
from __future__ import annotations

import ast
import builtins
import pathlib


def _module_path() -> pathlib.Path:
    import axis_qvla.bench.sim as pkg
    return pathlib.Path(pkg.__file__).parent / "rollout_dualsim.py"


def test_no_undefined_global_is_called():
    src = _module_path().read_text()
    tree = ast.parse(src)

    bound: set[str] = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bound |= {(a.asname or a.name.split(".")[0]) for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            bound |= {(a.asname or a.name) for a in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.comprehension,)):
            pass
        elif isinstance(node, ast.Global):
            bound |= set(node.names)

    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    missing = sorted(called - bound)
    assert not missing, f"rollout_dualsim calls undefined name(s): {missing}"


def test_observation_for_obs_key_is_imported():
    """The specific regression: the geometry helper the harness applies per camera."""
    src = _module_path().read_text()
    assert "from axis_qvla.bench.sim.obs_geometry import observation_for_obs_key" in src
