"""The core and offline tiers stay within their venv's package set, checked by AST.

The tiers follow the benchmark layout: `axis_qvla.core`, `axis_qvla.data`, `axis_qvla.paper` and `axis_qvla.bench`
(less its simulator and its policy servers) are core, as are the `config`, `run`, `release` and
`local_policies` packages where a checkout has them; `axis_qvla.labels`, `axis_qvla.signals`, `axis_qvla.realrobot`
and the simulator `axis_qvla.bench.sim` are offline, as are `episode`, `dataset` and `eval` where present.
Those optional packages are DEV_ONLY, and each guard below holds with or without them.

Core is loaded from BOTH venvs -- a SLURM job script and a rollout client both need path logic,
and T0 runs under a bare `python -S`. A third-party import here breaks one of the environments,
and the failure surfaces minutes into a job rather than at import.

The offline tier runs under the offline venv (numpy/pyarrow/duckdb/pandas, no
jax/flax/torch/lerobot), so a module-scope import of any online-only framework in it breaks
import the moment offline tooling touches it.

Generalises axis_data/tests/test_venv_purity.py, which checks the same invariant for the
pre-refactor layout.

`_module_level_imports` walks `tree.body`: the imports at module scope, as in
axis_data/tests/test_venv_purity.py. An import nested inside a top-level
`try/except ImportError:` or `if TYPE_CHECKING:` block is optional by construction and is
not counted.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CORE_DIRS = [
    REPO / "axis_qvla" / "config",
    REPO / "axis_qvla" / "run",
    # The core: paths, locks, canonical hashing and the `qvla` CLI (axis_qvla/cli.py, which T0 runs
    # under a bare `python -S`; installed, it is cli.py at the package root).
    REPO / "axis_qvla" / "core",
    # `qvla data verify` and the corpus index readers run under a bare python3.11 -S too.
    REPO / "axis_qvla" / "data",
    # The v1.0 publication packages: T0 and the release tooling run under a bare `python3.11 -S`,
    # so they are core tier from their first commit.
    REPO / "axis_qvla" / "paper",
    REPO / "axis_qvla" / "release",
    # bench is core except its simulator and its policy servers (eval tier, below): `bench check`,
    # `bench eval --dry-run` and `bench score` (axis_qvla.bench.results) run on any interpreter.
    REPO / "axis_qvla" / "bench",
    # The public training package: ArmSpec, recipes, config generation (`qvla train`), parity.
    # Its numpy users (bind, smoke) import numpy inside functions.
    REPO / "axis_qvla" / "train",
    # Torch policy classes live here where present (exempt below); swept as a directory,
    # so any other module added here is core unless it is named in CORE_EXEMPT.
    REPO / "axis_qvla" / "local_policies",
]
# Optional packages: a guard that a package contributes modules applies to these only where they
# exist.
DEV_ONLY = {REPO / "axis_qvla" / p for p in ("config", "run", "release", "local_policies", "episode", "dataset", "eval")}
# Directories inside a CORE_DIRS package that are not core. The simulator is the eval tier (MuJoCo,
# numpy, openpi_client inside functions), swept by the offline check below; the reference policy
# servers are the policy-server tier (torch, lerobot, jax, websockets) and are swept by neither.
CORE_EXEMPT_DIRS = [REPO / "axis_qvla" / "bench" / "sim", REPO / "axis_qvla" / "bench" / "servers"]
# Modules inside a CORE_DIRS package that are NOT core: torch policy classes, which only the
# local-policy eval backend imports (function-locally), where present. Each must really import an online framework (test below), so an exemption
# cannot outlive the reason for it.
CORE_EXEMPT = {
    REPO / "axis_qvla" / "local_policies" / "cfg_policy.py",
    REPO / "axis_qvla" / "local_policies" / "rnn_policy.py",
}
# Modules inside a CORE_DIRS package that are offline tier: axis_qvla.data's two corpus readers, which
# read LeRobot parquet with numpy, pandas and pyarrow at module scope. The offline check sweeps
# them instead, and each must really import an offline framework (test below).
CORE_OFFLINE = {
    REPO / "axis_qvla" / "data" / "convert_droid_actions.py",
    REPO / "axis_qvla" / "data" / "lerobot_local.py",
}
OFFLINE_FRAMEWORKS = {"numpy", "pandas", "pyarrow", "scipy", "duckdb", "mujoco"}
# cli.py is a single file, not a package directory, so it can't join CORE_DIRS' glob --
# but the governing constraint names it explicitly (it's loaded from both venvs too), so
# it must be scanned just the same. So is every cli_*.py beside it.
CORE_FILES = [REPO / "axis_qvla" / "cli.py", *sorted((REPO / "axis_qvla").glob("cli_*.py"))]

# The offline tier: numpy/pyarrow/duckdb/pandas are fine (that's what the offline venv
# ships), but jax/flax/torch/lerobot must never leak in, because these packages are also read
# by tooling running under the offline venv, where none of those packages are installed.
# Each root is swept recursively -- see `_py_modules` -- so a future offline subpackage is added
# here once and its own nested modules are still covered automatically.
#
# The simulator (bench/sim) is here despite driving GPU rollouts: its core (arms/record/runner) is
# deliberately stdlib-only so an arm name can be computed and a finished sweep scored without
# a simulator or a policy server. The rollout BACKENDS it dispatches to do need torch and
# openpi_client, and they are imported inside the functions that use them -- the physics and
# render backends need different venvs and cannot coexist in one process anyway. This ratchet
# checks module-level imports, so it pins exactly the property that matters: scoring and
# naming never acquire a heavyweight dependency.
OFFLINE_ROOTS = [REPO / "axis_qvla" / "episode", REPO / "axis_qvla" / "dataset", REPO / "axis_qvla" / "eval",
                 REPO / "axis_qvla" / "labels", REPO / "axis_qvla" / "signals",
                 REPO / "axis_qvla" / "bench" / "sim",       # eval tier: mujoco, numpy, openpi_client inside functions
                 REPO / "axis_qvla" / "realrobot"]   # realdata (FR3 data tools) and realign (REALALIGN signals)
# axis_qvla/bench/servers is the policy-server tier (it may import torch, lerobot, jax, websockets): in neither list.
ONLINE_ONLY = {"jax", "jaxlib", "flax", "torch", "lerobot", "tensorflow", "openpi"}
# Directories under OFFLINE_ROOTS that install from an environment of their own, not from the
# `offline` extra, each mapped to the file declaring that environment: the simulator is the eval tier
# (MuJoCo, openpi-client and the renderer's package), the real-robot renderer the render tier (its
# own exact lock, which the `render` extra repeats). The sweep above covers them like any offline
# module; tests/ratchets/test_offline_imports_declared.py checks the REST of the offline tier's
# imports against the `offline` extra.
OWN_ENVIRONMENT = {
    REPO / "axis_qvla" / "bench" / "sim": REPO / "conf" / "envs" / "eval.toml",
    REPO / "axis_qvla" / "realrobot" / "render": REPO / "axis_qvla" / "realrobot" / "render" / "render.lock.txt",
}

# The OTHER direction, and it is not symmetric with the one above. These modules are imported
# BY the online training path (openpi venv), which has no scipy and no mujoco -- so an
# accidental import here does not fail at review, it fails minutes into a training job, or
# quietly drags a simulator into a dataloader worker.
#
# Ported from `axis_data/tests/test_venv_purity.py`, which enforced this over `axis_data/*.py`
# and became vacuous once that package was drained by the migration. Deleting it outright
# would have dropped this direction entirely: the offline sweep above only checks that offline
# code avoids online frameworks, never the reverse.
#
# The SLB sidecar reader was on this list until the SLB sidecar pipeline it belonged to
# was archived; nothing online-facing replaced it.
ONLINE_FACING = [
    REPO / "axis_qvla" / "labels" / "idle_filter.py",
    REPO / "axis_qvla" / "core" / "paths.py",
]
OFFLINE_ONLY = {"scipy", "mujoco", "robosuite"}

STDLIB = set(sys.stdlib_module_names)
FIRST_PARTY = {"axis_qvla"}


def _py_modules(root: Path) -> list[Path]:
    """Every `.py` under `root`, at ANY depth.

    Recursive on purpose: the previous sweep listed each package directory by hand and
    globbed one level, so adding a new subpackage to an offline package would leave it
    silently unchecked while the count guard below still passed -- a test that keeps
    reporting green about code it no longer looks at.
    """
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _core_modules() -> list[Path]:
    out: list[Path] = []
    for d in CORE_DIRS:
        out.extend(p for p in _py_modules(d) if p not in CORE_EXEMPT and p not in CORE_OFFLINE
                   and not any(e in p.parents for e in CORE_EXEMPT_DIRS))
    out.extend(CORE_FILES)
    return out


def _module_level_imports(path: Path) -> set[str]:
    """Top-level package names imported at MODULE scope; function-local imports allowed."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_there_are_core_modules_to_check():
    """Guards the rest: a bad glob would make every assertion vacuous.

    Threshold is above the CLI files' count (1-2), so a CORE_DIRS sweep that silently
    stops matching anything still fails this guard instead of hiding behind the CLI files alone;
    and every core package (bench included) must contribute at least one module, a
    development-only one wherever it exists.
    """
    found = _core_modules()
    assert len(found) >= 6
    assert all(any(p.is_relative_to(d) for p in found) for d in CORE_DIRS if d not in DEV_ONLY or d.is_dir())
    assert all(p in found and p.is_file() for p in CORE_FILES)


def test_local_policies_are_swept_as_core():
    """Torch policies are swept as core with two named exemptions, where the package exists."""
    found = set(_core_modules())
    pkg = REPO / "axis_qvla" / "local_policies"
    if pkg.is_dir():
        modules = set(_py_modules(pkg))
        assert pkg / "__init__.py" in modules and modules - CORE_EXEMPT <= found
        assert CORE_EXEMPT <= modules                    # each exemption names a module of the package


def test_axis_train_holds_no_torch_policy():
    """axis_qvla/train is the public training package; the torch policies must not come back into it."""
    train = REPO / "axis_qvla" / "train"
    assert not (train / "cfg_policy.py").exists() and not (train / "rnn_policy.py").exists()


def test_axis_train_is_public_core():
    """The public training package is core tier and contributes modules in both trees."""
    found = _core_modules()
    assert REPO / "axis_qvla" / "train" / "armspec.py" in found
    assert REPO / "axis_qvla" / "train" not in DEV_ONLY


def test_bench_sim_and_servers_are_not_swept_as_core():
    """The simulator and the reference policy servers sit inside the core bench package but are
    not core: the simulator is checked as the eval tier, the servers by neither sweep."""
    found = set(_core_modules())
    for d in CORE_EXEMPT_DIRS:
        mods = _py_modules(d)
        assert mods, f"{d.relative_to(REPO)} has no modules: drop it from CORE_EXEMPT_DIRS"
        assert not found & set(mods)
    assert set(_py_modules(REPO / "axis_qvla" / "bench" / "sim")) <= set(_offline_modules())
    assert not set(_py_modules(REPO / "axis_qvla" / "bench" / "servers")) & set(_offline_modules())


def test_every_core_exemption_is_really_online():
    """An exempt module that no longer imports an online framework at module scope must rejoin
    the core sweep. Absent ones have nothing to check."""
    assert all(any(p.is_relative_to(d) for d in CORE_DIRS) for p in CORE_EXEMPT)
    stale = [p for p in sorted(CORE_EXEMPT) if p.is_file() and not _module_level_imports(p) & ONLINE_ONLY]
    assert not stale, f"{[str(p.relative_to(REPO)) for p in stale]} import no online framework: remove the exemption"


def test_every_offline_module_in_a_core_package_really_is_offline():
    """Likewise for CORE_OFFLINE: a listed module that stops importing an offline framework at
    module scope must rejoin the core sweep, and every listed one is swept by the offline check."""
    assert all(p.is_file() and any(p.is_relative_to(d) for d in CORE_DIRS) for p in CORE_OFFLINE)
    stale = [p for p in sorted(CORE_OFFLINE) if not _module_level_imports(p) & OFFLINE_FRAMEWORKS]
    assert not stale, f"{[str(p.relative_to(REPO)) for p in stale]} import no offline framework: remove them from CORE_OFFLINE"
    assert CORE_OFFLINE <= set(_offline_modules())


@pytest.mark.parametrize("path", _core_modules(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_core_modules_import_stdlib_only(path: Path):
    leaked = _module_level_imports(path) - STDLIB - FIRST_PARTY
    assert not leaked, (
        f"{path.relative_to(REPO)} imports {sorted(leaked)} at module scope. "
        f"Core modules are loaded from both venvs and must stay stdlib-only. "
        f"Move it inside the function that needs it, or put the module in another package."
    )


def _offline_modules() -> list[Path]:
    """Every `.py` under EITHER offline root, at any depth -- sorted so both roots interleave
    deterministically rather than one root's modules all sorting before the other's.
    """
    out: list[Path] = []
    for root in OFFLINE_ROOTS:
        out.extend(_py_modules(root))
    out.extend(CORE_OFFLINE)
    return sorted(out)


def test_the_module_sweep_reaches_a_nested_subpackage(tmp_path):
    """The sweep is recursive, so a new subpackage of an offline package is covered by the tier's
    isolation invariant without being listed by hand."""
    (tmp_path / "deep" / "nested").mkdir(parents=True)
    for rel in ("a.py", "deep/b.py", "deep/nested/c.py"):
        (tmp_path / rel).write_text("")
    assert [p.name for p in _py_modules(tmp_path)] == ["a.py", "b.py", "c.py"]


def test_there_are_offline_modules_to_check():
    """Guards the parametrised test below: a dead sweep would make it vacuous.

    Threshold 16 was derived from the real recursive count when there were only two offline
    roots; the signals, labels, realrobot and simulator packages alone hold far
    more, so a sweep that stops matching most of the tier still fails it.
    """
    found = _offline_modules()
    assert len(found) >= 16
    # Not just a count: the sweep must actually be everything under EITHER root, so a future
    # subpackage of either tier cannot slip past by keeping the total above the threshold.
    assert set(found) == {
        p
        for root in OFFLINE_ROOTS
        for p in root.rglob("*.py")
        if "__pycache__" not in p.parts
    } | CORE_OFFLINE


def test_each_own_environment_directory_is_an_offline_package_with_its_declaration():
    """An OWN_ENVIRONMENT entry that stops naming real offline modules, or whose environment
    declaration is gone, would hide those modules from the `offline` extra check for no reason."""
    for d, declaration in OWN_ENVIRONMENT.items():
        assert any(d.is_relative_to(r) for r in OFFLINE_ROOTS), d
        assert _py_modules(d) and declaration.is_file(), (d, declaration)


@pytest.mark.parametrize("path", _offline_modules(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_offline_modules_do_not_import_online_frameworks(path: Path):
    leaked = _module_level_imports(path) & ONLINE_ONLY
    assert not leaked, (
        f"{path.relative_to(REPO)} imports {sorted(leaked)} at module scope. The offline tier "
        f"runs under the offline venv, which has none of them."
    )


# --- the online-facing direction (ported from axis_data/tests/test_venv_purity.py) ---------

def test_the_online_facing_modules_all_exist():
    """Guards the parametrised test below. These are paths, not a sweep, so a module that
    MOVES silently stops being checked -- which is exactly what happened to the legacy guard
    when the migration drained axis_data/."""
    missing = [p for p in ONLINE_FACING if not p.is_file()]
    assert not missing, (
        f"{[str(p.relative_to(REPO)) for p in missing]} are listed as online-facing but do not "
        f"exist. Update the list to where they moved -- an unchecked module is worse than a "
        f"failing check."
    )


@pytest.mark.parametrize("path", ONLINE_FACING, ids=lambda p: p.name)
def test_online_facing_modules_do_not_import_offline_frameworks(path: Path):
    leaked = _module_level_imports(path) & OFFLINE_ONLY
    assert not leaked, (
        f"{path.relative_to(REPO)} imports {sorted(leaked)} at module scope, but it is imported "
        f"by the online training path, whose venv has neither. This does not fail at review -- "
        f"it fails minutes into a training job."
    )
