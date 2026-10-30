"""Every third-party package the shipped offline tier imports is declared, by the `offline` extra of
the public pyproject AND by the offline lock (conf/envs/offline.lock.txt), at any depth: imports
inside functions count. The shipped core tier imports nothing third-party when it is imported, and
what its functions import is declared the same way.

WHY: axis_qvla.labels.resample imports scipy inside the function that slerps the poses onto the render
rows. Neither the `offline` extra, the CI's constraints nor the offline lock carried scipy, so in
the public CI's own environment (`pip install -c .github/ci-constraints.txt -e ".[offline,test]"`)
3 label tests failed with ModuleNotFoundError, while the development venv, which has scipy, stayed
green. test_venv_tiers reads module-scope imports only, and checks only that a tier keeps OUT what
it must not have; nothing checked that what a tier does import is something its install provides.

The modules are test_venv_tiers' tier map: its offline and core sweeps, less the optional
packages (DEV_ONLY) and, for the offline tier,
less the directories that install from an environment of their own (OWN_ENVIRONMENT: the simulator
and the real-robot renderer). An import under `if TYPE_CHECKING:` never runs and is not counted.
"""
from __future__ import annotations

import ast
import dataclasses
import re
import tomllib
from collections.abc import Iterable, Iterator
from pathlib import Path

from tests.ratchets import test_venv_tiers as tiers

REPO = tiers.REPO
PKG = REPO / "axis_qvla"

# Import name -> distribution name, where the two differ; any other name is its own distribution.
DISTRIBUTION = {"PIL": "pillow", "cv2": "opencv-python-headless", "yaml": "pyyaml",
                "dateutil": "python-dateutil", "sklearn": "scikit-learn"}


@dataclasses.dataclass(frozen=True)
class Exemption:
    """An import the shipped offline tier makes although the `offline` extra does not declare it.
    Allowed only at `sites` (paths under the package root) and only inside a function, so the module
    itself imports without the package. The reason is exactly one of: another extra that declares
    it (`extra`), the submodule the CI installs it from (`submodule`, with the `job` that checks it
    out, installs it and runs a test importing the site), or a `try` that turns its absence into an
    install hint (`guarded`). The tests below check the reason, not just the name."""
    sites: tuple[str, ...]
    extra: str | None = None
    submodule: str | None = None
    job: str | None = None
    guarded: bool = False


NOT_IN_OFFLINE = {
    # Contact labels only. They read the training scenes' MuJoCo models (the CI data job
    # deselects the contact golden). The `test` extra pins MuJoCo exactly: the benchmark's physics.
    "mujoco": Exemption(("labels/contacts.py",), extra="test"),
    # The task checkers that extract the goal records: the axis-render submodule's package, which is
    # not on PyPI, so no extra can declare it. The goal-record golden is the test that reaches the
    # import, under `-m slow`, so the CI data job is the one that must install it from the submodule.
    "axis_replay_native": Exemption(("labels/goals.py",), submodule="third_party/axis-render", job="data"),
    # The FR3 data tools: OpenCV is the serve-time resize of the crop (the CI installs no
    # OpenCV). The import sits in a `try` whose handler names the
    # package to install.
    "cv2": Exemption(("realrobot/realdata/crop224.py",), guarded=True),
    # The real-robot line's robot side: openpi's client reads a policy server's handshake
    # (realrobot/fr3/probe.py, which only the openpi environment's interpreter runs, as `python -m`; the
    # serve step imports the port poll from realrobot/fr3/ports.py, never the probe) and drives the robot
    # with DROID (realrobot/fr3/droid_drive.py, run on the robot's control computer from the drive kit).
    # Neither runs in the offline environment; each import sits in a `try` whose handler names what to
    # install.
    "openpi_client": Exemption(("realrobot/fr3/probe.py", "realrobot/fr3/droid_drive.py"), guarded=True),
    "droid": Exemption(("realrobot/fr3/droid_drive.py",), guarded=True),
}


@dataclasses.dataclass(frozen=True)
class Import:
    name: str           # the top-level module
    line: int
    in_function: bool   # runs when a function is called, not when the module is imported
    guarded: bool       # in the body of a `try` that catches ImportError


_IMPORT_ERRORS = frozenset({"ImportError", "ModuleNotFoundError"})


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(isinstance(t, ast.Name) and t.id in _IMPORT_ERRORS for t in types)


def _type_checking(node: ast.If) -> bool:
    t = node.test
    return (isinstance(t, ast.Name) and t.id == "TYPE_CHECKING") or (isinstance(t, ast.Attribute) and t.attr == "TYPE_CHECKING")


def _walk(node: ast.AST, in_function: bool, guarded: bool) -> Iterator[Import]:
    if isinstance(node, ast.Import):
        for alias in node.names:
            yield Import(alias.name.split(".")[0], node.lineno, in_function, guarded)
        return
    if isinstance(node, ast.ImportFrom):
        if node.level == 0 and node.module:       # a relative import is this package's own
            yield Import(node.module.split(".")[0], node.lineno, in_function, guarded)
        return
    if isinstance(node, ast.If) and _type_checking(node):
        for stmt in node.orelse:
            yield from _walk(stmt, in_function, guarded)
        return
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        in_function, guarded = True, False        # a `try` around a `def` guards defining it, not running it
    if isinstance(node, (ast.Try, ast.TryStar)):
        catches = any(_catches_import_error(h) for h in node.handlers)
        for stmt in node.body:
            yield from _walk(stmt, in_function, guarded or catches)
        for stmt in (*node.handlers, *node.orelse, *node.finalbody):
            yield from _walk(stmt, in_function, guarded)
        return
    for child in ast.iter_child_nodes(node):
        yield from _walk(child, in_function, guarded)


def third_party_imports(path: Path) -> list[Import]:
    """Every import in `path` of a package that is neither the standard library nor this one."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [i for i in _walk(tree, False, False) if i.name not in tiers.STDLIB | tiers.FIRST_PARTY]


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def distribution(import_name: str) -> str:
    return _norm(DISTRIBUTION.get(import_name, import_name))


def requirement_names(requirements: Iterable[str]) -> set[str]:
    """The normalised distribution names of PEP 508 requirement strings."""
    return {_norm(re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", r.strip()).group()) for r in requirements}


def lock_names(text: str) -> set[str]:
    """The normalised distribution names a `name==version` lock pins."""
    return {_norm(ln.split("==")[0].strip()) for ln in text.splitlines() if "==" in ln and not ln.lstrip().startswith("#")}


def shipped(modules: Iterable[Path], *, without: Iterable[Path]) -> list[Path]:
    """`modules` less every one under a directory of `without`."""
    dirs = tuple(without)
    return [p for p in modules if not any(p.is_relative_to(d) for d in dirs)]


def undeclared(modules: Iterable[Path], package: Path, extra: set[str], lock: set[str],
               exemptions: dict[str, Exemption]) -> list[str]:
    """One line for each import in `modules` whose distribution the extra and the lock do not BOTH
    declare, unless an exemption allows it at that site, inside a function."""
    out: list[str] = []
    for path in modules:
        site = path.relative_to(package).as_posix()
        for i in third_party_imports(path):
            dist = distribution(i.name)
            missing = [what for what, names in (("the extra", extra), ("the lock", lock)) if dist not in names]
            ex = exemptions.get(i.name)
            if missing and not (ex is not None and site in ex.sites and i.in_function):
                out.append(f"{site}:{i.line} imports {i.name} ({dist}), not declared by {' or '.join(missing)}")
    return out


def at_import_time(modules: Iterable[Path], package: Path) -> list[str]:
    """One line for each third-party import in `modules` that runs when the module is imported."""
    return [f"{p.relative_to(package).as_posix()}:{i.line} imports {i.name}"
            for p in modules for i in third_party_imports(p) if not i.in_function]


def _public_file(name: str) -> Path:
    """`name` at the checkout root, or among the release overrides or templates where a checkout
    keeps it there."""
    public = REPO / "conf" / "release" / "public"
    for candidate in (REPO / name, public / "overrides" / name, public / f"{name}.tmpl"):
        if candidate.is_file():
            return candidate
    raise AssertionError(f"{name} is neither at the root of {REPO} nor among its release's public files")


def ci_jobs(ci: str) -> dict[str, str]:
    """Each job of a workflow by name: its text, from its `  <name>:` line under `jobs:` to the next."""
    body = ci.split("\njobs:\n", 1)[1]
    heads = list(re.finditer(r"^  ([A-Za-z0-9_-]+):[ \t]*$", body, re.M))
    return {m.group(1): body[m.start():(heads[i + 1].start() if i + 1 < len(heads) else len(body))]
            for i, m in enumerate(heads)}


def _pytest_paths(job: str) -> list[str]:
    """The test paths of a job's pytest commands, with the job's `env:` values expanded."""
    env = dict(re.findall(r'^\s+([A-Z][A-Z0-9_]*): "(.*)"$', job, re.M))
    text = job.replace("\\\n", " ")
    paths = []
    for cmd in re.findall(r"python -m pytest ([^\n|]*)", text):
        cmd = re.sub(r"\$(\w+)", lambda m: env.get(m.group(1), m.group(0)), cmd)
        toks = cmd.split()
        paths += [t.rstrip("/") for prev, t in zip(["", *toks], toks)
                  if re.match(r"tests(/|$)", t) and prev != "--deselect"]
    return paths


def _imports_module(source: str, dotted: str) -> bool:
    parent, _, leaf = dotted.rpartition(".")
    for n in ast.walk(ast.parse(source)):
        if isinstance(n, ast.Import) and any(a.name == dotted for a in n.names):
            return True
        if isinstance(n, ast.ImportFrom) and n.level == 0 and (
                n.module == dotted or (n.module == parent and any(a.name == leaf for a in n.names))):
            return True
    return False


def submodule_job_problems(ex: Exemption, ci: str, root: Path, package: str) -> list[str]:
    """What the job a submodule exemption names lacks: the job itself, the submodule's checkout, its
    install, and a test it runs that imports one of the sites. Any other job's install does not count:
    the job that runs the importer is the one that needs the package."""
    jobs = ci_jobs(ci)
    if ex.job not in jobs:
        return [f"no CI job {ex.job!r}"]
    job, out = jobs[ex.job], []
    checkouts = [ln.split() for ln in job.splitlines()]
    if not any(ex.submodule in w for w in checkouts if {"submodule", "update", "--init"} <= set(w)):
        out.append(f"the {ex.job} job does not check out {ex.submodule}")
    if not any(f"./{ex.submodule}" in ln.split() for ln in job.splitlines() if "pip install" in ln):
        out.append(f"the {ex.job} job does not install ./{ex.submodule}")
    modules = [f"{package}.{site.removesuffix('.py').replace('/', '.')}" for site in ex.sites]
    tests = [p for t in _pytest_paths(job) for p in ([root / t] if (root / t).is_file() else sorted((root / t).rglob("test_*.py")))]
    if not any(_imports_module(p.read_text(encoding="utf-8"), m) for p in tests for m in modules):
        out.append(f"the {ex.job} job runs no test that imports {', '.join(modules)}")
    return out


def _extras() -> dict[str, list[str]]:
    return tomllib.loads(_public_file("pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]


def _offline_declared() -> tuple[set[str], set[str]]:
    """The `offline` extra's distributions and those of the lock conf/envs/offline.toml builds from."""
    spec = tomllib.loads((REPO / "conf" / "envs" / "offline.toml").read_text(encoding="utf-8"))
    return requirement_names(_extras()["offline"]), lock_names((REPO / spec["lock"]).read_text(encoding="utf-8"))


def shipped_offline() -> list[Path]:
    return shipped(tiers._offline_modules(), without=(*tiers.DEV_ONLY, *tiers.OWN_ENVIRONMENT))


def shipped_core() -> list[Path]:
    return shipped(tiers._core_modules(), without=tiers.DEV_ONLY)


# --- the derivation, on a synthetic tree ----------------------------------------------------------

def _tree(root: Path, files: dict[str, str]) -> list[Path]:
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return sorted(root.rglob("*.py"))


def test_an_undeclared_import_inside_a_function_fails_until_the_extra_and_the_lock_declare_it(tmp_path):
    pkg = tmp_path / "pkg"
    modules = _tree(pkg, {
        "labels/resample.py": "import numpy as np\n\n\ndef slerp():\n    from scipy.spatial.transform import Slerp\n    return Slerp\n",
        "labels/frames.py": "from . import resample\n\n\ndef read(p):\n    import PIL.Image\n    return PIL.Image.open(p)\n",
        "dev/tool.py": "import duckdb\n",                         # a development-only package
        "sim/rollout.py": "def run():\n    import mujoco\n",       # a directory with its own environment
    })
    found = shipped(modules, without=(pkg / "dev", pkg / "sim"))
    assert found == [pkg / "labels" / "frames.py", pkg / "labels" / "resample.py"]

    extra, lock = requirement_names(["numpy>=1.26.4", "Pillow>=12.2.0"]), {"numpy", "pillow"}
    assert undeclared(found, pkg, extra, lock, {}) == [
        "labels/resample.py:5 imports scipy (scipy), not declared by the extra or the lock"]
    extra |= requirement_names(["scipy>=1.17.1"])
    assert undeclared(found, pkg, extra, lock, {}) == [
        "labels/resample.py:5 imports scipy (scipy), not declared by the lock"]
    assert undeclared(found, pkg, extra, lock | {"scipy"}, {}) == []
    # PIL is Pillow's import name: a requirement on a distribution called PIL declares nothing.
    assert undeclared(found, pkg, (extra - {"pillow"}) | {"pil"}, lock | {"scipy"}, {}) == [
        "labels/frames.py:5 imports PIL (pillow), not declared by the extra"]


def test_an_exemption_holds_only_at_its_sites_and_only_inside_a_function(tmp_path):
    pkg = tmp_path / "pkg"
    modules = _tree(pkg, {
        "labels/contacts.py": "def model(xml):\n    import mujoco\n    return mujoco.MjModel.from_xml_path(xml)\n",
        "labels/eager.py": "import mujoco\n",
        "signals/other.py": "def f():\n    import mujoco\n",
    })
    ex = {"mujoco": Exemption(("labels/contacts.py", "labels/eager.py"), extra="test")}
    assert undeclared(modules, pkg, {"numpy"}, {"numpy"}, ex) == [
        "labels/eager.py:1 imports mujoco (mujoco), not declared by the extra or the lock",
        "signals/other.py:2 imports mujoco (mujoco), not declared by the extra or the lock"]


def test_import_time_reaches_nested_blocks_and_skips_type_checking_and_functions(tmp_path):
    """test_venv_tiers reads `tree.body` only; an import inside a top-level `try` or `if` runs at
    import time too, and this sweep sees it. `if TYPE_CHECKING:` never runs; a relative import and
    the standard library are not third-party."""
    pkg = tmp_path / "pkg"
    modules = _tree(pkg, {
        "core/a.py": "import json\nfrom . import b\ntry:\n    import yaml\nexcept ImportError:\n    yaml = None\n",
        "core/b.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import numpy\n\n\ndef f():\n    import pandas\n",
        "core/c.py": "class C:\n    import numpy as np\n",
    })
    assert at_import_time(modules, pkg) == ["core/a.py:4 imports yaml", "core/c.py:2 imports numpy"]
    assert [(i.name, i.in_function) for i in third_party_imports(pkg / "core" / "b.py")] == [("pandas", True)]


def test_a_guard_is_a_try_catching_import_error_inside_the_same_function(tmp_path):
    pkg = tmp_path / "pkg"
    _tree(pkg, {"m.py": (
        "def a():\n    try:\n        import cv2\n    except ImportError as exc:\n        raise ImportError('pip install x') from exc\n"
        "def b():\n    import av\n"
        "def c():\n    try:\n        import av\n    except ValueError:\n        pass\n"
        "try:\n    def d():\n        import cv2\nexcept ImportError:\n    pass\n")})
    assert [(i.name, i.line, i.guarded) for i in third_party_imports(pkg / "m.py")] == [
        ("cv2", 3, True), ("av", 7, False), ("av", 10, False), ("cv2", 15, False)]


# The synthetic tree's names, assembled so that nothing reads them as paths of this tree.
_T, _SUB = "tests", "/".join(("third_party", "render"))
_SYNTHETIC_CI = f"""\
name: ci
jobs:
  test:
    steps:
      - run: git submodule update --init --depth 1 -- {_SUB}
      - run: pip install --no-deps ./{_SUB}
      - run: python -m pytest {_T} -m "not slow" -q
  data:
    env:
      DATA_TESTS: "{_T}/golden"
    steps:
      - run: |
          python -m pytest $DATA_TESTS -m slow \\
            --deselect {_T}/golden/test_goals.py::test_other | tee log
"""


def test_a_submodule_exemption_holds_only_in_the_job_that_runs_its_importer(tmp_path):
    """The install in another job does not count: the job that runs a test importing the site must
    check the submodule out and install it."""
    _tree(tmp_path, {f"{_T}/golden/test_goals.py": "from pkg.labels.goals import extract\n",
                     f"{_T}/unit/test_other.py": "import pkg.labels.other\n"})
    ex = Exemption(("labels/goals.py",), submodule=_SUB, job="data")
    assert submodule_job_problems(ex, _SYNTHETIC_CI, tmp_path, "pkg") == [
        f"the data job does not check out {_SUB}", f"the data job does not install ./{_SUB}"]
    steps = "    steps:\n      - run: |\n"
    fixed = _SYNTHETIC_CI.replace(steps, f"    steps:\n      - run: git submodule update --init --depth 1 -- {_SUB}\n"
                                  f"      - run: pip install --no-deps ./{_SUB}\n      - run: |\n")
    assert fixed.count(f"pip install --no-deps ./{_SUB}") == 2
    assert submodule_job_problems(ex, fixed, tmp_path, "pkg") == []
    elsewhere = Exemption(("labels/other.py",), submodule=_SUB, job="data")
    assert submodule_job_problems(elsewhere, fixed, tmp_path, "pkg") == [
        "the data job runs no test that imports pkg.labels.other"]
    assert submodule_job_problems(Exemption(("x.py",), submodule="s", job="nightly"), fixed, tmp_path, "pkg") == [
        "no CI job 'nightly'"]


# --- the real trees -------------------------------------------------------------------------------

def test_the_shipped_offline_tier_is_the_tier_map_less_dev_packages_and_own_environments():
    """Guards the checks below: a sweep that silently stops matching would make them vacuous. Every
    offline root the tree ships contributes, the module this file exists for is in, and nothing under
    a development-only package or an own-environment directory is."""
    found = set(shipped_offline())
    assert PKG / "labels" / "resample.py" in found and len(found) >= 60
    for root in tiers.OFFLINE_ROOTS:
        if not any(root.is_relative_to(d) for d in (*tiers.DEV_ONLY, *tiers.OWN_ENVIRONMENT)):
            assert any(p.is_relative_to(root) for p in found), root
    for d in (*tiers.DEV_ONLY, *tiers.OWN_ENVIRONMENT):
        assert not {p for p in found if p.is_relative_to(d)}, d
    assert set(tiers.CORE_OFFLINE) <= found


def test_every_third_party_import_of_the_shipped_offline_tier_is_declared():
    extra, lock = _offline_declared()
    problems = undeclared(shipped_offline(), PKG, extra, lock, NOT_IN_OFFLINE)
    assert not problems, (
        "the shipped offline tier imports packages a `pip install .[offline]` does not provide. Add each "
        "to the public pyproject's `offline` extra, conf/envs/offline.in and the CI constraints, and "
        "recompile conf/envs/offline.lock.txt; or, if it is optional, give it an Exemption here:\n  "
        + "\n  ".join(problems))


def test_each_exemption_is_still_needed_and_its_reason_holds():
    extras = _extras()
    offline = requirement_names(extras["offline"])
    ci = _public_file(".github/workflows/ci.yml").read_text(encoding="utf-8")
    found = set(shipped_offline())
    for name, ex in NOT_IN_OFFLINE.items():
        assert distribution(name) not in offline, f"the `offline` extra declares {name}: drop its exemption"
        assert [bool(ex.extra), bool(ex.submodule), ex.guarded].count(True) == 1, name
        assert bool(ex.job) == bool(ex.submodule), f"{name}: a submodule exemption, and only one, names its job"
        for site in ex.sites:
            assert PKG / site in found, f"{site} is not a shipped offline module"
            imports = [i for i in third_party_imports(PKG / site) if i.name == name]
            assert imports, f"{site} no longer imports {name}: drop it from NOT_IN_OFFLINE"
            assert all(i.in_function for i in imports), f"{site} imports {name} when imported, not only in a function"
            assert not ex.guarded or all(i.guarded for i in imports), f"{site} imports {name} outside a try catching ImportError"
        if ex.extra:
            assert distribution(name) in requirement_names(extras[ex.extra]), f"the `{ex.extra}` extra does not declare {name}"
        if ex.submodule:
            problems = submodule_job_problems(ex, ci, REPO, PKG.name)
            assert not problems, f"{name}: " + "; ".join(problems)


def test_the_shipped_core_tier_imports_nothing_third_party_when_imported():
    """Stricter than test_venv_tiers' module-scope sweep: an import in a top-level `try` or `if`, or
    in a class body, runs at import time as well."""
    found = shipped_core()
    assert PKG / "core" / "paths.py" in found and len(found) >= 40
    leaked = at_import_time(found, PKG)
    assert not leaked, "core modules are loaded by every venv and by `python -S`:\n  " + "\n  ".join(leaked)


def test_what_the_shipped_core_tiers_functions_import_is_declared():
    """Core functions import numpy, pandas or pyarrow where they need them (bind, smoke, the corpus
    index readers, Fig. 4); a caller has them from the `offline` extra, so they are declared there."""
    extra, lock = _offline_declared()
    problems = undeclared(shipped_core(), PKG, extra, lock, {})
    assert not problems, "\n  ".join(["undeclared imports in the core tier's functions:", *problems])
