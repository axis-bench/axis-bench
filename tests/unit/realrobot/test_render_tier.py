"""The render tier's import boundary and its lock.

The render package runs in its own venv (mujoco + mink + PyAV, axis_qvla/realrobot/render/render.lock.txt).
The offline venv has no mink, so an offline real-robot module that imports mink, av or the render
package at module scope breaks the first time offline tooling touches it. Module-level imports only,
by AST, as tests/ratchets/test_venv_tiers.py checks the other tiers.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

import axis_qvla.realrobot.render as _render_pkg

# The first-party package name, read from an import so it holds whatever the package is called
# in this tree. Derived, never a literal, so it survives a rename unchanged.
FIRST_PARTY = _render_pkg.__name__.split(".")[0]

REPO = Path(__file__).resolve().parents[3]
RENDER = REPO / "axis_qvla" / "realrobot" / "render"
LOCK = RENDER / "render.lock.txt"
RENDER_PACKAGES = {"numpy", "scipy", "mujoco", "mink", "av", "pandas", "pyarrow"}
STDLIB_ONLY = {"rig.py", "profile.py", "assets.py", "data.py", "__init__.py", "__main__.py", "cli.py"}
OFFLINE_ROOTS = [REPO / "axis_qvla" / "realrobot" / p for p in ("realign", "realdata", "fr3", "cli.py")]


def _modules(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _top_imports(path: Path) -> set[str]:
    out: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module)
    return out


def _third_party(names: set[str]) -> set[str]:
    tops = {n.split(".")[0] for n in names}
    return {n for n in tops if n not in sys.stdlib_module_names and n not in (FIRST_PARTY, "__future__")}


def _locked() -> set[str]:
    names = set()
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        m = re.fullmatch(r"([A-Za-z0-9_.-]+)==([0-9][A-Za-z0-9.]*)", line)
        assert m, f"not an exact pin: {line!r}"
        names.add(m.group(1).lower().replace("-", "_"))
    return names


def test_there_are_render_modules():
    assert len(_modules(RENDER)) >= 3


@pytest.mark.parametrize("path", _modules(RENDER), ids=lambda p: p.name)
def test_render_modules_import_only_the_render_tier(path):
    extra = _third_party(_top_imports(path)) - RENDER_PACKAGES
    assert not extra, f"{path.name} imports {sorted(extra)} at module scope"
    if path.name in STDLIB_ONLY:
        assert not _third_party(_top_imports(path)), f"{path.name} must stay stdlib only"


def test_the_lock_pins_every_render_import():
    used = set().union(*(_third_party(_top_imports(p)) for p in _modules(RENDER)))
    assert used <= _locked(), f"unpinned: {sorted(used - _locked())}"


@pytest.mark.parametrize("path", [p for r in OFFLINE_ROOTS for p in _modules(r)], ids=lambda p: p.name)
def test_offline_real_robot_modules_never_reach_the_render_tier(path):
    names = _top_imports(path)
    assert not {n.split(".")[0] for n in names} & {"mink", "av"}, path
    assert not any(n.startswith("axis_qvla.realrobot.render") for n in names), path


# --- what the render venv runs: the render package and every first-party module it reaches ----------

def _all_imports(path: Path, module: str) -> set[str]:
    """Every module `path` imports, at any scope (inside functions too), relative imports resolved
    against `module`; `from p import m` also names `p.m`, which may be a module."""
    pkg = module if path.name == "__init__.py" else module.rpartition(".")[0]
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = pkg.rsplit(".", node.level - 1)[0] if node.level > 1 else pkg
            mod = (f"{base}.{node.module}" if node.module else base) if node.level else (node.module or "")
            out |= {mod} | {f"{mod}.{a.name}" for a in node.names}
    return out


def _source(module: str) -> Path | None:
    parts = module.split(".")
    if parts[0] != FIRST_PARTY:
        return None
    p = REPO.joinpath(*parts)
    return p.with_suffix(".py") if p.with_suffix(".py").is_file() else (
        p / "__init__.py" if (p / "__init__.py").is_file() else None)


def _render_closure() -> dict[str, Path]:
    """{module: source} of the render package's modules and every first-party module they import,
    transitively, at any scope."""
    todo = [_render_pkg.__name__ + "".join(f".{q}" for q in p.relative_to(RENDER).with_suffix("").parts)
            for p in _modules(RENDER)]
    todo = [m.removesuffix(".__init__") for m in todo]
    seen: dict[str, Path] = {}
    while todo:
        m = todo.pop()
        src = _source(m)
        if src is None or m in seen:
            continue
        seen[m] = src
        for name in _all_imports(src, m):
            parts = name.split(".")
            todo += [".".join(parts[:i]) for i in range(1, len(parts) + 1)]
    return seen


def test_the_import_walker_resolves_relative_and_function_scope_imports(tmp_path):
    p = tmp_path / "m.py"
    p.write_text("from . import sib\nfrom ..up import x\n\n\ndef f():\n    import pandas\n    from .pkg import y\n")
    assert _all_imports(p, "a.b.c.m") == {"a.b.c", "a.b.c.sib", "a.b.up", "a.b.up.x", "pandas", "a.b.c.pkg",
                                          "a.b.c.pkg.y"}


def test_everything_the_render_venv_runs_imports_only_locked_render_packages():
    """Module scope in the render package is not the whole reach: the first-party modules it imports
    (the signal compiler, the REALALIGN tools, the DROID converter, ...) run in the render venv too, and
    so does every import inside their functions. A dependency one of them grows breaks the render job
    only where a test happens to reach it; this follows them all."""
    closure = _render_closure()
    assert len(closure) > len(_modules(RENDER)), "the walk reaches no first-party module outside the package"
    used: dict[str, str] = {}
    for mod, path in sorted(closure.items()):
        for name in sorted(_third_party(_all_imports(path, mod))):
            used.setdefault(name, mod)
    extra = {n: m for n, m in used.items() if n not in RENDER_PACKAGES}
    assert not extra, f"reached from the render package, imported by: {extra}"
    assert set(used) <= _locked(), f"unpinned: {sorted(set(used) - _locked())}"
