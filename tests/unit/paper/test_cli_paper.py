"""`paper tables` is registered, documents its exit codes, and runs with no site-packages."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from axis_qvla.cli import build_parser
from axis_qvla.paper.expected import write_expected
from axis_qvla.paper.fetch import LocalSource
from tests.golden.paper.test_t0_fixture import _rebuild
from tests.golden.paper.t0_fixture import build_pack

REPO = Path(__file__).resolve().parents[3]


def test_parser_registration():
    args = build_parser().parse_args(["paper", "tables", "--results", "x", "--check"])
    assert args.check and args.cells.endswith("experiments/paper/cells.json") and args.func.__name__ == "_cmd_tables"
    assert build_parser().parse_args(["paper", "tables"]).results == "hf://axisrobotics/qvla-v1-results@v1.0"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["paper"])


def test_stdlib_only_under_python_dash_S(tmp_path: Path):
    conf, pack, ref = build_pack(tmp_path)
    write_expected(_rebuild(conf, LocalSource(pack), ref).outputs, pack / "expected")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["PYTHONPATH"] = str(REPO)
    # -B: the PYTHON* variables were dropped above (PYTHONDONTWRITEBYTECODE with them), and a
    # __pycache__ written into the tree under test is a release-check PROBLEM.
    cmd = [sys.executable, "-S", "-B", "-c", "import sys; assert not any('site-packages' in p for p in sys.path); "
           "from axis_qvla.cli import main; sys.exit(main(sys.argv[1:]))",
           "paper", "tables", "--results", str(pack), "--results-map", str(ref), "--check",
           "--benchmark", "fixture_bench", "--bench-conf-dir", str(conf)]
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "IDENTICAL" in out.stdout
