"""`bench eval` ends with the run's results file and its all-24 and per-task table; `--print` prints
the one command; outside the render venv it re-executes there, and a missing environment is named
with the one command that builds it."""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

import pytest

from axis_qvla.bench import cli as bench_cli
from axis_qvla.bench import finish, launch
from axis_qvla.bench.results.format import Results, dumps_results, load_results
from axis_qvla.cli import main
from axis_qvla.core.hosts import REPO_ROOT
from tests.unit.bench.eval_stubs import eval_argv, fake_driver
from tests.unit.bench.results.fixtures import make_run
from tests.unit.bench.test_eval_roots import host, printed_eval  # noqa: F401  (host is a fixture)


def _given(d):
    return eval_argv(d, "--render-root", str(d / "render"), "--task-data-root", str(d / "tasks"))


def test_finish_writes_the_results_file_and_prints_the_table(tmp_path):
    _, bench = make_run(tmp_path)
    lines: list[str] = []
    assert finish.finish(tmp_path / "r1", "mine", bench, ft_seed=None, pretrain_seed=None, report=lines.append) == 0
    path = finish.results_path(tmp_path / "r1", "mine")
    assert path == tmp_path / "r1" / "out_mine" / "results.json"
    r = load_results(path.read_bytes())
    assert r.header.training_seed is None and len(r.rows) == 6
    text = "\n".join(lines)
    assert f"results: {path}" in text and "| mine_dualsim | 50.00 |" in text
    assert "| 806 | 2/3 | 66.7 |" in text and "| 1050 | 1/3 | 33.3 |" in text


def test_finish_records_the_seeds_given(tmp_path):
    _, bench = make_run(tmp_path)
    assert finish.finish(tmp_path / "r1", "mine", bench, ft_seed=3, pretrain_seed=42, report=lambda _: None) == 0
    r = load_results(finish.results_path(tmp_path / "r1", "mine").read_bytes())
    assert r.header.training_seed == {"pretrain": 42, "ft": 3}


def test_a_run_that_cannot_be_built_is_reported_not_raised(tmp_path):
    _, bench = make_run(tmp_path, errored=2)
    lines: list[str] = []
    assert finish.finish(tmp_path / "r1", "mine", bench, ft_seed=None, pretrain_seed=None, report=lines.append) == 2
    assert lines[0].startswith("FAIL  the run finished, but") and "2 rollouts ended with an `error:` reason" in lines[0]


def test_eval_finishes_the_run_it_filed_with_the_seeds_given(host, monkeypatch):
    seen = fake_driver(monkeypatch, host)
    calls = []
    monkeypatch.setattr(finish, "finish", lambda root, arm, b, **kw: calls.append((root, arm, kw)) or 0)
    assert main([*_given(host), "--ft-seed", "3"]) == 0
    assert seen["argv"] and calls == [(host / "r1", "probe", {"ft_seed": 3, "pretrain_seed": None})]


def test_a_failed_driver_is_not_finished(host, monkeypatch):
    from axis_qvla.bench.sim import campaign

    fake_driver(monkeypatch, host)
    monkeypatch.setattr(campaign, "main", lambda argv: 5)
    monkeypatch.setattr(finish, "finish", lambda *a, **k: pytest.fail("finished a failed run"))
    assert main(_given(host)) == 5


def test_a_pretrain_seed_needs_a_finetune_seed(capsys):
    assert main(["bench", "eval", "--policy", "127.0.0.1:9", "--run", "x", "--pretrain-seed", "42"]) == 2
    assert "--pretrain-seed needs --ft-seed" in capsys.readouterr().out


def test_print_prints_the_plan_and_one_command_and_starts_nothing(host, monkeypatch, capsys):
    """No GPU, no network, no data: only the profile. Comment lines first, the one command last."""
    import shutil

    from axis_qvla.bench.sim import campaign

    monkeypatch.setattr(campaign, "main", lambda argv: pytest.fail("driver must not run"))
    for d in ("render", "tasks", "assets", "ck"):
        shutil.rmtree(host / d)
    assert main(["bench", "eval", "--host", "demo", "--checkpoint", str(host / "ck"), "--config", "cfg",
                 "--run", "probe", "--print"]) == 0
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    python, args = printed_eval(out)
    assert python == str(host / "venvs" / "render" / "bin" / "python") and "--print" not in args
    assert all(ln.startswith("# ") for ln in lines[:-1]) and " bench eval " in lines[-1]
    assert f"# results file: {host / 'r1' / 'out_probe' / 'results.json'}" in lines
    assert any(ln.startswith("# inputs: ") and "qvla data get bench" in ln for ln in lines)
    assert not (host / "r1").exists()


def test_outside_the_render_venv_eval_reexecutes_there(host, monkeypatch):
    py = host / "venvs" / "render" / "bin" / "python"
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text("")
    calls = []
    monkeypatch.setattr(sys, "executable", str(host / "elsewhere" / "python"))
    monkeypatch.setattr(os, "execve", lambda path, argv, env: calls.append((path, argv, env)))
    assert main(_given(host)) == 0
    (path, argv, env), = calls
    assert path == str(py) and argv[:5] == [str(py), "-m", "axis_qvla.cli", "bench", "eval"]
    assert "--checkpoint" in argv and env["PYTHONPATH"] == str(REPO_ROOT)
    assert env[bench_cli.REEXEC_MARKER] == "1"


def test_reexec_flushes_stdout_and_stderr_first(host, monkeypatch):
    """Flushed before execve: once it replaces this process, anything still buffered is gone."""
    py = host / "venvs" / "render" / "bin" / "python"
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text("")
    monkeypatch.setattr(sys, "executable", str(host / "elsewhere" / "python"))
    monkeypatch.setattr(os, "execve", lambda *a: None)
    flushed = []
    monkeypatch.setattr(sys.stdout, "flush", lambda: flushed.append("stdout"))
    monkeypatch.setattr(sys.stderr, "flush", lambda: flushed.append("stderr"))
    assert main(_given(host)) == 0
    assert flushed == ["stdout", "stderr"]


def test_a_wrapper_python_that_keeps_reexecing_is_refused_not_looped(host, monkeypatch, capsys):
    """A venv's bin/python that execs some other interpreter under the hood would otherwise re-exec
    forever; the marker execve hands the child tells it this is already the second try."""
    py = host / "venvs" / "render" / "bin" / "python"
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text("")
    monkeypatch.setattr(sys, "executable", str(host / "elsewhere" / "python"))
    monkeypatch.setenv(bench_cli.REEXEC_MARKER, "1")
    monkeypatch.setattr(os, "execve", lambda *a: pytest.fail("must not re-exec a second time"))
    assert main(_given(host)) == 2
    out = capsys.readouterr().out
    assert "re-executed under" in out and str(py) in out
    assert "sys.executable is" in out and str(host / "elsewhere" / "python") in out


def test_execve_raising_oserror_is_reported_not_raised(host, monkeypatch, capsys):
    py = host / "venvs" / "render" / "bin" / "python"
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text("")
    monkeypatch.setattr(sys, "executable", str(host / "elsewhere" / "python"))

    def boom(*a):
        raise OSError(13, "Permission denied")
    monkeypatch.setattr(os, "execve", boom)
    assert main(_given(host)) == 2
    out = capsys.readouterr().out
    assert "FAIL" in out and str(py) in out and "Permission denied" in out


def test_the_reexec_marker_does_not_leak_to_the_driver(host, monkeypatch):
    """A process that arrives already matching launch.python (started there, or re-exec'd there)
    must not hand QVLA_EVAL_REEXEC on to the driver, its servers or its workers."""
    seen = fake_driver(monkeypatch, host)
    monkeypatch.setenv(bench_cli.REEXEC_MARKER, "1")
    assert main(_given(host)) == 0
    assert bench_cli.REEXEC_MARKER not in seen["env"]
    assert bench_cli.REEXEC_MARKER not in os.environ


def test_a_missing_render_venv_names_the_build_command(host, monkeypatch, capsys):
    monkeypatch.setattr(sys, "executable", str(host / "elsewhere" / "python"))
    monkeypatch.setattr(os, "execve", lambda *a: pytest.fail("no environment to execute"))
    assert main(_given(host)) == 2
    out = capsys.readouterr().out
    assert "must run under the evaluation environment" in out and "qvla env build eval" in out


def test_a_missing_server_environment_names_the_build_command(host, monkeypatch, capsys):
    fake_driver(monkeypatch, host)
    (host / "venvs" / "online" / "bin" / "python").unlink()
    assert main(_given(host)) == 2
    assert "qvla env build pi05" in capsys.readouterr().out


def test_finish_refuses_to_replace_a_different_results_file(tmp_path):
    """A seeded results file, then a tally with no seeds given: the file is left byte-identical and
    the second call fails loudly instead of silently overwriting the first run's record."""
    _, bench = make_run(tmp_path)
    assert finish.finish(tmp_path / "r1", "mine", bench, ft_seed=3, pretrain_seed=None, report=lambda _: None) == 0
    path = finish.results_path(tmp_path / "r1", "mine")
    before = path.read_bytes()
    lines: list[str] = []
    rc = finish.finish(tmp_path / "r1", "mine", bench, ft_seed=None, pretrain_seed=None, report=lines.append)
    assert rc != 0
    assert path.read_bytes() == before
    text = "\n".join(lines)
    assert "an earlier results file records training_seed" in text
    assert "pass the same --ft-seed/--pretrain-seed" in text and str(path) in text


def test_finish_is_a_no_op_pass_when_rerun_identically(tmp_path):
    _, bench = make_run(tmp_path)
    assert finish.finish(tmp_path / "r1", "mine", bench, ft_seed=3, pretrain_seed=None, report=lambda _: None) == 0
    path = finish.results_path(tmp_path / "r1", "mine")
    before = path.read_bytes()
    assert finish.finish(tmp_path / "r1", "mine", bench, ft_seed=3, pretrain_seed=None, report=lambda _: None) == 0
    assert path.read_bytes() == before


def test_a_rejected_score_writes_no_file(tmp_path, monkeypatch):
    """score() runs before the write, so a run it rejects leaves no results file at all."""
    _, bench = make_run(tmp_path)

    def boom(r, b, **kw):
        raise ValueError("score refuses this run")
    monkeypatch.setattr(finish, "score", boom)
    lines: list[str] = []
    assert finish.finish(tmp_path / "r1", "mine", bench, ft_seed=None, pretrain_seed=None, report=lines.append) == 2
    path = finish.results_path(tmp_path / "r1", "mine")
    assert not path.exists()
    assert "score refuses this run" in lines[0] and "No results file exists yet" in lines[0] and str(path) in lines[0]


def test_a_build_failure_reports_whether_an_earlier_file_survives(tmp_path):
    _, bench = make_run(tmp_path)
    assert finish.finish(tmp_path / "r1", "mine", bench, ft_seed=None, pretrain_seed=None, report=lambda _: None) == 0
    path = finish.results_path(tmp_path / "r1", "mine")
    before = path.read_bytes()
    # A later build under the same arm that cannot be built (errored rollouts): the FAIL line must
    # say the earlier file is still there, since this failure wrote nothing new and did not touch it.
    out_dir = tmp_path / "r1" / "out_mine"
    summary = json.loads((out_dir / "arm_summary.json").read_text())
    summary["errored_rollouts"] = 2
    (out_dir / "arm_summary.json").write_text(json.dumps(summary))
    lines: list[str] = []
    assert finish.finish(tmp_path / "r1", "mine", bench, ft_seed=None, pretrain_seed=None, report=lines.append) == 2
    assert path.exists() and path.read_bytes() == before
    assert "An earlier results file is still at" in lines[0] and str(path) in lines[0]


def test_a_build_failure_names_no_file_when_none_exists_yet(tmp_path):
    _, bench = make_run(tmp_path, errored=2)
    lines: list[str] = []
    assert finish.finish(tmp_path / "r1", "mine", bench, ft_seed=None, pretrain_seed=None, report=lines.append) == 2
    path = finish.results_path(tmp_path / "r1", "mine")
    assert not path.exists()
    assert "No results file exists yet at" in lines[0] and str(path) in lines[0]


def test_a_chmod0_run_directory_is_reported_not_raised(tmp_path):
    """A non-searchable out_<arm> directory used to let Path.exists()'s PermissionError escape as
    an uncaught traceback (3.11-3.13) or get silently read as "no results file" (3.14). (When the
    tests run as root, chmod 0 denies nothing, so the run just succeeds; rc 0.)"""
    _, bench = make_run(tmp_path)
    out_dir = tmp_path / "r1" / "out_mine"
    out_dir.chmod(0)
    try:
        lines: list[str] = []
        rc = finish.finish(tmp_path / "r1", "mine", bench, ft_seed=None, pretrain_seed=None, report=lines.append)
    finally:
        out_dir.chmod(0o755)
    if rc == 0:
        return   # ran as root: chmod 0 denied nothing, nothing left to assert
    assert rc == 3
    assert lines and lines[0].startswith("FAIL  the run finished, but its results file cannot be read or written")
    assert any(s in lines[0] for s in
              ("An earlier results file is still at", "No results file exists yet at", "cannot be checked"))


def test_a_rows_only_difference_names_counts_and_the_first_differing_seed(tmp_path):
    """Header fields agree but the rollout outcomes differ: name row counts and the first
    differing (task, seed), not a repr of the whole rows tuple."""
    _, bench = make_run(tmp_path)
    assert finish.finish(tmp_path / "r1", "mine", bench, ft_seed=None, pretrain_seed=None, report=lambda _: None) == 0
    base = load_results(finish.results_path(tmp_path / "r1", "mine").read_bytes())
    flipped = dataclasses.replace(base.rows[0], success=not base.rows[0].success)
    other = Results(base.header, (flipped, *base.rows[1:]))

    diff = finish._first_difference(base, other)
    assert diff is not None
    field, old_value, new_value = diff
    assert field == "rows"
    assert str(len(base.rows)) in str(old_value) and str(flipped.task) in str(old_value) and str(flipped.seed) in str(old_value)
    assert "ResultsRow(" not in str(old_value) and "ResultsRow(" not in str(new_value)

    msg = finish._conflict_refusal(dumps_results(base), other, Path("x"))
    assert "ResultsRow(" not in msg and repr(base.rows) not in msg
    assert f"records rows {old_value};" in msg   # the summary as prose, not quoted


def test_interpreter_problem_does_not_read_an_unreadable_parent_as_absent(tmp_path):
    locked = tmp_path / "locked"
    (locked / "bin").mkdir(parents=True)
    locked.chmod(0)
    try:
        assert launch.interpreter_problem(locked / "bin" / "python") in ("cannot be read (PermissionError)", "is not built")
    finally:
        locked.chmod(0o755)
    assert launch.interpreter_problem(tmp_path / "absent" / "python") == "is not built"
