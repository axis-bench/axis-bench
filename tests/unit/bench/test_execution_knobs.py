"""Every environment variable the benchmark package reads is a declared execution knob or named
here as outcome-neutral, and each knob is read the way its reader reads it."""
from __future__ import annotations

import ast
import dataclasses
import os
from pathlib import Path

import pytest

from axis_qvla import bench as bench_pkg
from axis_qvla.bench import execution

BENCH = Path(bench_pkg.__file__).resolve().parent
# Read by the benchmark package and changing nothing a rollout executes, each with the reason.
OUTCOME_NEUTRAL = {
    "AXIS_EVAL_PROVENANCE": "prints digests of the first four requests and replies; no action, "
                            "observation or seed file changes",
    "AXIS_ROLLOUT_VIDEO_EVERY": "the frame stride of a diagnostics video written beside the results, "
                                "never into them",
    "QVLA_EVAL_REEXEC": "set by `bench eval` on the env it execve's the render venv with, so the "
                        "re-executed process can tell it is already the second try; read before any "
                        "server starts, never by a rollout",
}
_ENVIRON = ("environ", "environb")
_VALUE_METHODS = ("get", "setdefault", "pop")
_GETENV = ("getenv", "getenvb")
REQUIRED = "<required: os.environ[...] has no default>"
PRESENCE = "<presence check: no value>"


@dataclasses.dataclass(frozen=True)
class Read:
    var: str | None      # None: the name is not a string constant of the module
    form: str
    default: object      # the literal default, None when absent, else REQUIRED / PRESENCE
    where: str


def _bindings(tree: ast.AST):
    """(name, value) for every way the module binds a name to an expression: assignment
    (tuple targets paired element-wise), annotated assignment, `:=`, and parameter defaults."""
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    yield t.id, n.value
                elif (isinstance(t, (ast.Tuple, ast.List)) and isinstance(n.value, (ast.Tuple, ast.List))
                      and len(t.elts) == len(n.value.elts)):
                    yield from ((e.id, v) for e, v in zip(t.elts, n.value.elts) if isinstance(e, ast.Name))
        elif isinstance(n, ast.AnnAssign) and n.value is not None and isinstance(n.target, ast.Name):
            yield n.target.id, n.value
        elif isinstance(n, ast.NamedExpr):
            yield n.target.id, n.value
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = n.args
            with_default = (a.posonlyargs + a.args)[len(a.posonlyargs) + len(a.args) - len(a.defaults):]
            yield from ((p.arg, d) for p, d in zip(with_default, a.defaults))
            yield from ((p.arg, d) for p, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None)


def env_reads(source: str, where: str) -> list[Read]:
    """Every environment access in `source`: `<x>.environ.get/setdefault/pop(...)`,
    `os.getenv/getenvb(...)`, `<x>.environ[...]`, `"X" in / not in <x>.environ`. The environment
    may be reached as `os.environ`, as `getattr(os, "environ")`, as `environ` imported from os, or
    through any local name bound to one of these (an assignment, a parameter default, a chain of
    such names); `getenv` and a bound `environ.get` likewise. A name argument is resolved through
    the module's string constants. A copy (`dict(os.environ)`) is a new mapping, not a read."""
    tree = ast.parse(source)
    consts = {t.id: n.value.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
              and isinstance(n.value, ast.Constant) and isinstance(n.value.value, (str, bytes))
              for t in n.targets if isinstance(t, ast.Name)}
    environ_names, getenv_names, method_names = set(_ENVIRON), set(), {}
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module == "os":
            for a in n.names:
                if a.name in _ENVIRON:
                    environ_names.add(a.asname or a.name)
                if a.name in _GETENV:
                    getenv_names.add(a.asname or a.name)

    def name_of(node: ast.AST) -> str | None:
        v = node.value if isinstance(node, ast.Constant) else consts.get(getattr(node, "id", None))
        return v.decode() if isinstance(v, bytes) else v if isinstance(v, str) else None

    def getattr_of(node: ast.AST, names: tuple[str, ...]) -> ast.expr | None:
        """`getattr(<obj>, "<one of names>")` -> <obj>, else None."""
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"
                and len(node.args) >= 2 and name_of(node.args[1]) in names):
            return node.args[0]
        return None

    def is_environ(node: ast.AST) -> bool:
        return ((isinstance(node, ast.Attribute) and node.attr in _ENVIRON)
                or (isinstance(node, ast.Name) and node.id in environ_names)
                or getattr_of(node, _ENVIRON) is not None)

    def is_getenv(node: ast.AST) -> bool:
        return ((isinstance(node, ast.Attribute) and node.attr in _GETENV)
                or (isinstance(node, ast.Name) and node.id in getenv_names)
                or getattr_of(node, _GETENV) is not None)

    def method_of(node: ast.AST) -> str | None:
        """`<environ>.get` (or getattr(<environ>, "get")) -> "get": a value method of the environment."""
        if isinstance(node, ast.Attribute) and node.attr in _VALUE_METHODS and is_environ(node.value):
            return node.attr
        obj = getattr_of(node, _VALUE_METHODS)
        if obj is not None and is_environ(obj):
            return name_of(node.args[1])
        return method_names.get(node.id) if isinstance(node, ast.Name) else None

    # Names bound to the environment, to getenv or to a bound value method, to a fixpoint so an
    # alias of an alias resolves. Module-wide, not per scope: the ratchet over-reports, never under.
    bindings = list(_bindings(tree))
    changed = True
    while changed:
        changed = False
        for name, value in bindings:
            if is_environ(value) and name not in environ_names:
                environ_names.add(name)
                changed = True
            elif is_getenv(value) and name not in getenv_names:
                getenv_names.add(name)
                changed = True
            elif (m := method_of(value)) is not None and name not in method_names:
                method_names[name] = m
                changed = True

    def default_of(args: list[ast.expr]) -> object:
        if len(args) < 2:
            return None
        try:
            return ast.literal_eval(args[1])
        except ValueError:
            return "<expression>"

    out = []
    for node in ast.walk(tree):
        at = f"{where}:{getattr(node, 'lineno', '?')}"
        if isinstance(node, ast.Call) and node.args:
            if (m := method_of(node.func)) is not None:
                out.append(Read(name_of(node.args[0]), f"environ.{m}", default_of(node.args), at))
            elif is_getenv(node.func):
                out.append(Read(name_of(node.args[0]), "getenv", default_of(node.args), at))
        elif isinstance(node, ast.Subscript) and is_environ(node.value):
            out.append(Read(name_of(node.slice), "environ[...]", REQUIRED, at))
        elif (isinstance(node, ast.Compare) and any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops)
              and any(is_environ(c) for c in node.comparators)):
            out.append(Read(name_of(node.left), "in environ", PRESENCE, at))
    return out


def package_reads() -> list[Read]:
    """Every read in axis_qvla/bench, recursively, except the declaration module itself."""
    out = []
    for path in sorted(BENCH.rglob("*.py")):
        if path == BENCH / "execution.py":
            continue
        out += env_reads(path.read_text(encoding="utf-8"), path.relative_to(BENCH.parent.parent).as_posix())
    return out


SPELLINGS = {
    "get": 'import os\nos.environ.get("K1", "d")\n',
    "get-via-constant": 'import os\nNAME = "K1"\nos.environ.get(NAME)\n',
    "aliased-module": 'import os as _os\n_os.environ.get("K1")\n',
    "getenv": 'import os\nos.getenv("K1", "d")\n',
    "getenvb": 'import os\nos.getenvb(b"K1")\n',
    "bare-getenv": 'from os import getenv\ngetenv("K1")\n',
    "subscript": 'import os\nx = os.environ["K1"]\n',
    "in": 'import os\nif "K1" in os.environ:\n    pass\n',
    "not-in": 'import os\nif "K1" not in os.environ:\n    pass\n',
    "setdefault": 'import os\nos.environ.setdefault("K1", "d")\n',
    "pop": 'import os\nos.environ.pop("K1", None)\n',
    "bare-environ": 'from os import environ\nenviron.get("K1")\n',
    "aliased-environ": 'from os import environ as E\nE["K1"]\n',
    # A local name bound to the environment, then read through.
    "local-alias": 'import os\nenv = os.environ\nenv.get("K1")\n',
    "local-alias-subscript": 'import os\ndef f():\n    env = os.environ\n    return env["K1"]\n',
    "local-alias-in": 'import os\nenv = os.environ\nok = "K1" in env\n',
    "alias-of-alias": 'import os\na = os.environ\nb = a\nb.get("K1")\n',
    "annotated-alias": 'import os\nenv: dict = os.environ\nenv.get("K1")\n',
    "walrus-alias": 'import os\nif (env := os.environ):\n    env.get("K1")\n',
    "tuple-alias": 'import os\nenv, n = os.environ, 3\nenv.get("K1")\n',
    "parameter-default": 'import os\ndef f(env=os.environ):\n    return env.get("K1")\n',
    "keyword-default": 'import os\ndef f(*, env=os.environ):\n    return env.get("K1")\n',
    "bound-method": 'import os\nget = os.environ.get\nget("K1")\n',
    "aliased-getenv": 'import os\ng = os.getenv\ng("K1")\n',
    # The environment reached through getattr.
    "getattr": 'import os\ngetattr(os, "environ").get("K1")\n',
    "getattr-subscript": 'import os\ngetattr(os, "environ")["K1"]\n',
    "getattr-alias": 'import os\nE = getattr(os, "environ")\nE.get("K1")\n',
    "getattr-via-constant": 'import os\nATTR = "environ"\ngetattr(os, ATTR).get("K1")\n',
    "getattr-getenv": 'import os\ngetattr(os, "getenv")("K1")\n',
    "getattr-method": 'import os\ngetattr(os.environ, "get")("K1")\n',
}


@pytest.mark.parametrize("name", sorted(SPELLINGS))
def test_each_spelling_of_an_environment_read_is_caught(name):
    assert [r.var for r in env_reads(SPELLINGS[name], "x.py")] == ["K1"]


def test_a_name_that_is_not_a_constant_is_caught_unresolved_and_a_copy_is_not_a_read():
    src = "import os\ndef f(n):\n    env = dict(os.environ)\n    return os.environ.get(n), env.get('K2')\n"
    assert [r.var for r in env_reads(src, "x.py")] == [None]


def test_a_local_alias_of_the_environment_is_read_through_and_its_default_checked():
    src = 'import os\ndef f():\n    env = os.environ\n    return env.get("PI05_REPLAN", "4")\n'
    assert env_reads(src, "x.py") == [Read("PI05_REPLAN", "environ.get", "4", "x.py:4")]
    assert "4" != execution.ENV["PI05_REPLAN"][1], "the default rule below would flag this read"


def test_the_environment_reached_through_getattr_is_read_through():
    src = 'import os\nv = getattr(os, "environ").get("MAX_ROLLOUT_STEPS", "900")\n'
    assert env_reads(src, "x.py") == [Read("MAX_ROLLOUT_STEPS", "environ.get", "900", "x.py:2")]
    assert env_reads('import os\ngetattr(os, "path").get("K1")\n', "x.py") == []   # not the environment


def test_a_copy_bound_to_a_name_stays_a_copy():
    src = 'import os\nenv = dict(os.environ)\nenv2 = {**os.environ}\nenv.get("K1"), env2.get("K2")\n'
    assert env_reads(src, "x.py") == []


def test_every_read_of_the_package_names_a_constant():
    unresolved = [r.where for r in package_reads() if r.var is None]
    assert unresolved == [], "name the variable with a string constant so it can be checked"


def test_every_variable_the_package_reads_is_a_declared_knob_or_outcome_neutral():
    names = {r.var for r in package_reads()}
    undeclared = sorted(names - set(execution.ENV) - set(OUTCOME_NEUTRAL))
    assert undeclared == [], "declare the knob in axis_qvla.bench.execution.ENV, or name it OUTCOME_NEUTRAL with why"
    assert set(OUTCOME_NEUTRAL) == set(execution.DIAGNOSTIC_ENV)
    assert set(execution.ENV) | set(OUTCOME_NEUTRAL) <= names, "a declared variable nothing reads"


def test_every_knob_is_read_with_its_declared_default():
    wrong = [(r.where, r.var, r.default) for r in package_reads()
             if r.var in execution.ENV and r.default is not PRESENCE and r.default != execution.ENV[r.var][1]]
    assert wrong == []


# --- each knob is read the way its reader reads it -----------------------------------------------

@pytest.mark.parametrize("var, raw, key, value", [
    ("PI05_FPS", "15.0", "fps", 15),                                # float()
    ("PI05_REPLAN", " 8 ", "replan", 8),                            # int() strips whitespace
    ("AXIS_EVAL_GRIP_STATE", " Fingers", "grip_state", "fingers"),  # .strip().lower()
    ("AXIS_EVAL_EEF_IK", "ITERATE", "eef_ik", "iterate"),
    ("AXIS_EVAL_IMAGE_GEOMETRY", " Raw ", "image_geometry", "raw"),
    ("AXIS_EVAL_IMAGE_GEOMETRY", "", "image_geometry", "profile"),  # empty: the reader's default
    ("AXIS_EVAL_TABLE_TEXTURE", "", "table_texture", None),         # falsy: no texture
    ("AXIS_EVAL_TABLE_TEXTURE", "  ", "table_texture", "  "),       # truthy: used as given
])
def test_a_value_is_read_the_way_its_reader_reads_it(var, raw, key, value):
    assert execution.executed({var: raw})[key] == value


@pytest.mark.parametrize("var, raw", [
    ("PI05_REPLAN", "8.0"),                # int() raises
    ("PI05_FPS", ""),                      # float("") raises: set to empty is not unset
    ("MAX_ROLLOUT_STEPS", "1e3"),
    ("N_STABLE_FRAMES", ""),
    ("AXIS_EVAL_ACTION_SPACE", "DROID8"),  # compared as given: the dual-sim loop raises
    ("AXIS_EVAL_ACTION_SPACE", ""),
    ("AXIS_EVAL_GRIP_STATE", "finger"),    # silently closedness at the reader
    ("AXIS_EVAL_EEF_IK", "iter"),          # silently single
    ("AXIS_EVAL_IMAGE_GEOMETRY", "rwa"),   # silently letterboxed
    ("AXIS_EVAL_REPLAN", "0"),             # effective_replan refuses < 1
    ("AXIS_EVAL_REPLAN", "two"),
])
def test_a_spelling_the_rollout_would_refuse_or_ignore_is_refused_by_name(var, raw):
    with pytest.raises(ValueError, match=var):
        execution.executed({var: raw})


@pytest.mark.parametrize("raw", ["fingers", " FINGERS ", "closedness", ""])
def test_the_grip_state_verdict_is_the_controllers(monkeypatch, raw):
    from axis_qvla.bench.sim import action_mapping

    monkeypatch.setenv("AXIS_EVAL_GRIP_STATE", raw)
    assert (execution.executed(os.environ)["grip_state"] == "fingers") == action_mapping._raw_finger_state_requested()


@pytest.mark.parametrize("raw", ["iterate", "Iterate ", "single", ""])
def test_the_eef_ik_verdict_is_the_controllers(monkeypatch, raw):
    from axis_qvla.bench.sim import action_mapping

    monkeypatch.setenv("AXIS_EVAL_EEF_IK", raw)
    assert (execution.executed(os.environ)["eef_ik"] == "iterate") == action_mapping._iterated_ik_requested()


@pytest.mark.parametrize("raw", ["raw", " RAW", "profile", ""])
def test_the_image_geometry_verdict_is_the_observation_builders(monkeypatch, raw):
    from axis_qvla.bench.sim import obs_geometry

    monkeypatch.setenv("AXIS_EVAL_IMAGE_GEOMETRY", raw)
    assert (execution.executed(os.environ)["image_geometry"] == "raw") == obs_geometry._passthrough_requested()


@pytest.mark.parametrize("dom", execution.DOMAINS)
@pytest.mark.parametrize("override", [None, "2", " 3 "])
def test_the_replan_override_counts_in_the_dual_sim_loop_only(monkeypatch, dom, override):
    from axis_qvla.bench.sim import domain

    if override is None:
        monkeypatch.delenv("AXIS_EVAL_REPLAN", raising=False)
    else:
        monkeypatch.setenv("AXIS_EVAL_REPLAN", override)
    assert execution.executed(os.environ, domain=dom)["replan"] == domain.executed_replan(dom, 8)


def test_a_malformed_override_outside_the_dual_sim_loop_is_not_read():
    assert execution.executed({"AXIS_EVAL_REPLAN": "two"}, domain="render")["replan"] == 8
    with pytest.raises(ValueError, match="domain"):
        execution.executed({}, domain="teleport")
