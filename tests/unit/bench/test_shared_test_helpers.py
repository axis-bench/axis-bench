"""Bench tests share helpers through helper modules, never through another test module's privates.

A private name is free to change with the module that owns it, so a test that imports one from
another test module breaks when that module is edited for its own reasons. The shared helpers are
plain modules: tests/unit/bench/eval_stubs.py, tests/unit/bench/sim/driver_stubs.py and
tests/unit/bench/results/fixtures.py.
"""
from __future__ import annotations

import ast
from pathlib import Path

BENCH_TESTS = Path(__file__).resolve().parent


def private_test_imports(src: str) -> list[str]:
    """`module.name` for every underscore name imported from a `tests.` module."""
    out = []
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module and n.module.startswith("tests."):
            out += [f"{n.module}.{a.name}" for a in n.names if a.name.startswith("_")]
    return out


def test_no_bench_test_imports_another_test_modules_privates():
    found = {p.relative_to(BENCH_TESTS).as_posix(): bad for p in sorted(BENCH_TESTS.rglob("*.py"))
             if (bad := private_test_imports(p.read_text(encoding="utf-8")))}
    assert found == {}


def test_the_scan_sees_a_private_import_and_ignores_public_and_package_ones():
    src = ("from tests.unit.bench.sim.test_x import _stack, helper\n"
           "from tests.unit.bench.results.fixtures import make_results\n"
           "from axis_qvla.bench.results.score import _seed_key\n"
           "def f():\n    from tests.unit.bench.test_y import _argv\n")
    assert private_test_imports(src) == ["tests.unit.bench.sim.test_x._stack", "tests.unit.bench.test_y._argv"]
