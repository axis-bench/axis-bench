"""Generated training configs never read an environment accessor or call an existence predicate.

A config that reads os.environ/getenv (under any name the `os` module is imported or aliased as)
or calls an `exists()`/`is_file()`/`is_dir()`-style existence predicate trains something different
on every machine (a config that falls back to another budget wherever a path is absent trains
another run there). Whether inputs exist is checked when a run is bound, never while its config is
generated. This does not ban every filesystem call -- `os.scandir`, `os.stat`,
`os.access` and `glob` are fine; only the specific environment accessors and existence predicates
named below are checked, by AST, over the generator modules.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GENERATOR_MODULES = [REPO / "axis_qvla" / "train" / f"{m}.py"
                     for m in ("armspec", "recipes", "ids", "generate", "lerobot_cfg", "parity")]
# Checked by attribute name alone, regardless of the name or alias the module is imported or
# bound under (`import os as _os; _os.environ`, `from os import unsetenv`): an attribute access
# of any of these names, or an import of one by name, is a hit.
_ENV_ATTRS = {"environ", "environb", "getenv", "getenvb", "putenv", "unsetenv"}
_PATH_TESTS = {"exists", "lexists", "isfile", "isdir", "is_file", "is_dir"}
_BANNED = _ENV_ATTRS | _PATH_TESTS


def violations(source: str) -> list[str]:
    out = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr in _BANNED:
            out.append(f"line {node.lineno}: .{node.attr}")
        elif isinstance(node, ast.ImportFrom) and node.module in ("os", "os.path"):
            bad = [a.name for a in node.names if a.name in _BANNED]
            if bad:
                out.append(f"line {node.lineno}: from {node.module} import {', '.join(bad)}")
    return out


def test_the_generator_modules_exist():
    missing = [str(p.relative_to(REPO)) for p in GENERATOR_MODULES if not p.is_file()]
    assert not missing


@pytest.mark.parametrize("path", GENERATOR_MODULES, ids=lambda p: p.name)
def test_generator_modules_read_no_env_and_test_no_path(path):
    found = violations(path.read_text())
    assert not found, f"{path.relative_to(REPO)}: {found}"


@pytest.mark.parametrize("snippet", [
    "import os\nx = os.environ['A']", "import os\nx = os.getenv('A')", "from os import environ",
    "from os import unsetenv", "import os\nos.path.exists('x')", "from os.path import isfile",
    "p.exists()", "p.is_dir()",
    # Aliased and dotted-attribute forms, which a checker matching only the base name would miss.
    "import os as _os\n_os.environ['A']", "import os as _os\nx = _os.getenv('A')",
    "import os\nos.environb[b'A']", "import os\nos.unsetenv('A')",
])
def test_the_checker_catches_each_form(snippet):
    assert violations(snippet)


@pytest.mark.parametrize("snippet", [
    "import os\nwith os.scandir('.') as it:\n    names = [e.name for e in it]\n",
    "import os\nst = os.stat('x')",
    "import os\nos.access('x', os.R_OK)",
    "import glob\nglob.glob('*.json')",
])
def test_the_checker_passes_clean_code(snippet):
    assert not violations(snippet)
