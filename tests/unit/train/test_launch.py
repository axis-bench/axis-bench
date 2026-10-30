"""One composed command per launch: argv, working directory and the variables it adds, printed as one
runnable line, run with the output streamed and the exit code passed through."""
from __future__ import annotations

import sys

import pytest

from axis_qvla.train.launch import Launch, LaunchError, from_rendered, run_streaming


def test_a_rendered_pi05_command_runs_with_the_environments_python(tmp_path):
    py = tmp_path / "venv" / "bin" / "python"
    got = from_rendered(["env", "OPENPI_DATA_HOME=/x/openpi", "python", "train.py", "--config-json", "c.json"],
                        python=py, cwd=tmp_path, extra_env={"A": "1"})
    assert got.argv == (str(py), "train.py", "--config-json", "c.json")
    assert dict(got.env) == {"OPENPI_DATA_HOME": "/x/openpi", "A": "1"} and got.cwd == tmp_path


def test_torchrun_is_the_one_beside_the_environments_python(tmp_path):
    py = tmp_path / "venv" / "bin" / "python"
    got = from_rendered(["env", "HF_HOME=h", "HF_HUB_OFFLINE=1", "torchrun", "--standalone", "-m", "x"],
                        python=py, cwd=tmp_path)
    assert got.argv[0] == str(tmp_path / "venv" / "bin" / "torchrun") and dict(got.env)["HF_HUB_OFFLINE"] == "1"


def test_an_unknown_program_is_refused(tmp_path):
    with pytest.raises(LaunchError, match="bash"):
        from_rendered(["bash", "-c", "x"], python=tmp_path / "python", cwd=tmp_path)


def test_the_printed_command_is_one_runnable_line(tmp_path):
    assert Launch(("p", "a b"), tmp_path, {"K": "v w"}).shell() == f"cd {tmp_path} && env K='v w' p 'a b'"
    assert Launch(("p",), tmp_path).shell() == f"cd {tmp_path} && p"


def test_run_streaming_passes_the_exit_code_and_the_added_variables_through(tmp_path):
    code = ("import os, pathlib, sys; pathlib.Path('seen').write_text(os.environ['QVLA_PROBE']); "
            "print('hi'); sys.exit(7)")
    assert run_streaming(Launch((sys.executable, "-c", code), tmp_path, {"QVLA_PROBE": "yes"})) == 7
    assert (tmp_path / "seen").read_text() == "yes"
