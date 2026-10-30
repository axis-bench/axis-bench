"""`qvla paper reeval` as a command: --list, --print (every command in order, the last building the
tables), the run itself (fetches in this process, each `bench eval` a child waited on by pid,
finished results skipped, a failure stopping the run), the tables from the user's results; and
`paper tables` reading the v1.0 Hub results by default."""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest

from axis_qvla.bench import finish
from axis_qvla.bench.results.format import dumps_results
from axis_qvla.cli import build_parser
from axis_qvla.cli import main
from axis_qvla.core.hosts import REPO_ROOT
from axis_qvla.paper import reeval, reeval_run
from axis_qvla.paper.cells import load_cells
from axis_qvla.paper.cli import DEFAULT_CELLS
from axis_qvla.paper.spec import load_spec
from tests.unit.bench.results.fixtures import make_bench, make_results

CELLS, SPEC = load_cells(DEFAULT_CELLS.read_bytes()), load_spec()
PROFILE = """launcher = "local"
gpus = 1
requires_secrets = []

[roots]
corpus = "{d}/corpus"
cache = "{d}/cache"
scratch = "{d}/scratch"
runs = "{d}/runs"
r1 = "{d}/r1"
bench = "{d}/bench"
openpi = "{d}/openpi"
eval = "{d}/eval"
smolvla_bench = "{d}/smolvla_bench"
smolvla_venv = "{d}/smolvla_venv"

[venv]
render = "{d}/venvs/render"
online = "{d}/venvs/online"
"""


def _cell(name):
    return next(c for c in CELLS.cells if c.cell == name)


def test_paper_tables_reads_the_v1_hub_results_by_default():
    args = build_parser().parse_args(["paper", "tables", "--check"])
    assert args.results == "hf://axisrobotics/qvla-v1-results@v1.0"


def test_list_prints_every_result_with_its_uses(capsys):
    assert main(["paper", "reeval", "--list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(CELLS.cells)
    bc = next(ln for ln in lines if ln.startswith("bc|bare\t"))
    assert bc.split("\t")[1:3] == ["pi05", "r1bc_r2"] and "main-results" in bc.split("\t")[3]
    assert not any(k in ln for ln in lines for k in ("table1", "fig1", "fig4c", "fig5", "fig6", "fig7"))


def test_print_prints_every_command_in_order_ending_with_the_collect_line(tmp_path, capsys):
    """No GPU, no network, no data: the profile names the output paths and nothing else is read."""
    h = str(_host(tmp_path))
    assert main(["paper", "reeval", "--result", "bc|bare", "--host", h, "--print"]) == 0
    lines = capsys.readouterr().out.splitlines()
    cid = _cell("bc|bare").checkpoint["id"]
    q = shlex.quote(h)
    assert lines == [f"qvla data get bench --host {q}", f"qvla data get base-weights --serving pi05 --host {q}",
                     f"qvla data get checkpoints/{cid} --host {q}",
                     f"qvla bench eval --checkpoint-id {cid} --serving bare --run r1bc_r2 --ft-seed 42 "
                     f"--pretrain-seed 42 --host {q}",
                     f"# results files: {tmp_path / 'r1'}/out_<run>/results.json, one per result (1)",
                     f"# tables: {tmp_path / 'runs' / 'paper-reeval' / 'bc-bare'}",
                     f"qvla paper reeval --result 'bc|bare' --collect --host {q}"]
    assert shlex.split(lines[-1])[3:5] == ["--result", "bc|bare"]


def test_print_needs_no_profile_at_all(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AXIS_HOST", raising=False)
    assert main(["paper", "reeval", "--table", "main-results", "--backbone", "xvla", "--print"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "# tables: <runs>/paper-reeval/main-results-xvla" in lines and lines[-1].endswith("--collect")
    assert lines[-1] == "qvla paper reeval --table main-results --backbone xvla --collect"
    assert sum(ln.startswith("qvla bench eval ") for ln in lines) == 24


def test_the_old_unit_name_runs_the_same_reevaluation(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AXIS_HOST", raising=False)
    assert main(["paper", "reeval", "--figure", "scaling", "--print"]) == 0
    new = capsys.readouterr().out
    assert main(["paper", "reeval", "--figure", "fig6", "--print"]) == 0
    assert capsys.readouterr().out == new and "--figure scaling --collect" in new


def test_an_unknown_result_points_at_list(capsys):
    assert main(["paper", "reeval", "--result", "nope", "--print"]) == 2
    assert "--list" in capsys.readouterr().out


class _Proc:
    def __init__(self, rc):
        self.rc, self.killed, self.terminated = rc, False, False

    def wait(self, timeout=None):
        return self.rc

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def test_run_steps_fetches_here_evaluates_as_children_and_skips_what_is_done(tmp_path):
    sel = reeval.select(CELLS, SPEC, reeval.Unit("figure", "fig4c", "xvla"))
    done = sel[0]
    p = finish.results_path(tmp_path / "r1", done.run)
    p.parent.mkdir(parents=True)
    p.write_text("{}")
    fetched, started, lines = [], [], []

    def popen(argv, env):
        started.append((argv, env))
        return _Proc(0)

    rc = reeval_run.run_steps(reeval.plan(sel), run_root=tmp_path / "r1", render_python=tmp_path / "py",
                              in_process=lambda a: fetched.append(a) or 0, popen=popen, report=lines.append,
                              clock=iter(range(0, 10_000, 10)).__next__)
    assert rc == 0
    assert fetched[:2] == [["data", "get", "bench"], ["data", "get", "base-weights", "--serving", "xvla"]]
    assert ["data", "get", f"checkpoints/{done.checkpoint['id']}"] not in fetched
    assert [a[6] for a, _ in started] == [c.checkpoint["id"] for c in sel[1:]]
    assert all(a[:5] == [str(tmp_path / "py"), "-m", "axis_qvla.cli", "bench", "eval"] for a, _ in started)
    assert started[0][1]["PYTHONPATH"] == str(REPO_ROOT)
    assert any(ln.startswith("done already: qvla bench eval") and done.run in ln for ln in lines)
    assert any(ln.endswith("in 10 s (exit 0)") for ln in lines)


def test_a_failing_step_stops_the_run_and_says_it_resumes(tmp_path):
    sel = reeval.select(CELLS, SPEC, reeval.Unit("figure", "fig4c", "xvla"))
    rcs, started, lines = iter([0, 4]), [], []

    def popen(argv, env):
        started.append(argv)
        return _Proc(next(rcs))

    rc = reeval_run.run_steps(reeval.plan(sel), run_root=tmp_path / "r1", render_python=tmp_path / "py",
                              in_process=lambda a: 0, popen=popen, report=lines.append)
    assert rc == 4 and len(started) == 2 and "resumes" in lines[-1]


def test_ctrl_c_waits_for_the_child_and_kills_only_it_when_it_does_not_stop():
    class Stuck(_Proc):
        def wait(self, timeout=None):
            if self.killed:
                return -9
            if timeout is None:
                raise KeyboardInterrupt
            raise subprocess.TimeoutExpired("bench eval", timeout)

    p = Stuck(0)
    with pytest.raises(KeyboardInterrupt):
        reeval_run.wait_child(["x"], {}, popen=lambda argv, env: p)
    assert p.terminated and p.killed


def test_a_child_that_stops_on_sigterm_is_not_killed():
    class StopsOnTerm(_Proc):
        def wait(self, timeout=None):
            if self.terminated:
                return -15
            if timeout is None:
                raise KeyboardInterrupt
            raise subprocess.TimeoutExpired("bench eval", timeout)

    p = StopsOnTerm(0)
    with pytest.raises(KeyboardInterrupt):
        reeval_run.wait_child(["x"], {}, popen=lambda argv, env: p)
    assert p.terminated and not p.killed


def test_a_second_ctrl_c_during_the_stop_wait_kills_the_child_at_once():
    waits = []

    class Slow(_Proc):
        def wait(self, timeout=None):
            waits.append(timeout)
            if self.killed:
                return -9
            raise KeyboardInterrupt          # the first Ctrl-C, then a second one during the 120 s wait

    p = Slow(0)
    with pytest.raises(KeyboardInterrupt):
        reeval_run.wait_child(["x"], {}, popen=lambda argv, env: p)
    assert p.killed and not p.terminated and waits == [None, reeval_run.STOP_TIMEOUT_S, None]


@pytest.mark.parametrize("rc, want", [(0, 0), (3, 3), (-9, 137), (-15, 143), (-2, 130)])
def test_a_child_ended_by_a_signal_exits_128_plus_its_number(rc, want):
    assert reeval_run.wait_child(["x"], {}, popen=lambda argv, env: _Proc(rc)) == want


@pytest.mark.parametrize("code, want", [(None, 0), (0, 0), (2, 2), ("refused", 2)])
def test_a_fetchs_system_exit_is_its_exit_code_and_none_is_success(code, want):
    def fetch(argv):
        raise SystemExit(code)

    assert reeval_run._in_process(fetch, ["data", "get", "bench"]) == want


def test_paper_tables_never_imports_the_evaluation_runner(tmp_path):
    """The runner reads a clock and the environment: `paper tables` (T0) must not load it, neither on
    importing axis_qvla.paper.cli nor on a whole fixture run. A fresh interpreter, so no other test's
    import counts."""
    import os
    import sys

    from axis_qvla.paper.expected import write_expected
    from axis_qvla.paper.fetch import LocalSource
    from tests.golden.paper.t0_fixture import build_pack
    from tests.golden.paper.test_t0_fixture import _rebuild

    conf, pack, ref = build_pack(tmp_path)
    write_expected(_rebuild(conf, LocalSource(pack), ref).outputs, pack / "expected")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update(PYTHONPATH=str(REPO_ROOT), PYTHONDONTWRITEBYTECODE="1")
    code = ("import sys\n"
            "import axis_qvla.paper.cli\n"
            "assert 'axis_qvla.paper.reeval_run' not in sys.modules, 'on import'\n"
            "from axis_qvla.cli import main\n"
            "rc = main(sys.argv[1:])\n"
            "assert 'axis_qvla.paper.reeval_run' not in sys.modules, 'after paper tables'\n"
            "sys.exit(rc)\n")
    out = subprocess.run([sys.executable, "-c", code, "paper", "tables", "--results", str(pack), "--results-map",
                          str(ref), "--check", "--benchmark", "fixture_bench", "--bench-conf-dir", str(conf)],
                         capture_output=True, text=True, env=env, cwd=tmp_path, check=False)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "IDENTICAL" in out.stdout


def _host(d: Path) -> Path:
    f = d / "qvla.toml"
    f.write_text(PROFILE.format(d=d))
    return f


def test_without_a_render_venv_nothing_is_fetched(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(reeval_run, "run_steps", lambda *a, **k: pytest.fail("ran"))
    assert main(["paper", "reeval", "--result", "bc|bare", "--host", str(_host(tmp_path))]) == 2
    out = capsys.readouterr().out
    assert "qvla env build eval" in out and "Nothing was fetched" in out


def test_collect_writes_the_paper_layout_from_the_users_results(tmp_path, capsys):
    conf = make_bench(tmp_path / "conf", [806, 1050], 3)
    from axis_qvla.bench.spec import load_benchmark

    bench = load_benchmark("fixture_bench", conf_dir=conf)
    for name in ("xvla_bc|bare", "xvla_rand_ep|bare"):
        c = _cell(name)
        r = make_results({806: [True, False, True], 1050: [False, False, True]}, arm=c.served, bench=bench,
                         seed=c.training_seed["ft"], pretrain=c.training_seed["pretrain"])
        p = finish.results_path(tmp_path / "r1", c.run)
        p.parent.mkdir(parents=True)
        p.write_bytes(dumps_results(r))
    out = tmp_path / "out"
    assert main(["paper", "reeval", "--figure", "random-vs-filtered", "--backbone", "xvla", "--collect", "--host",
                 str(_host(tmp_path)), "--benchmark", "fixture_bench", "--bench-conf-dir", str(conf),
                 "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("vla,group,label") and "2 of 5 results re-evaluated" in printed
    # The unit is named by what it shows; the files keep the published layout's names.
    assert (out / "figures" / "fig4c.csv").is_file() and (out / "tables" / "table1.md").is_file()


def test_ctrl_c_stops_the_run_and_says_it_resumes(tmp_path):
    sel = reeval.select(CELLS, SPEC, reeval.Unit("result", "bc|bare"))
    lines = []

    def popen(argv, env):
        class P(_Proc):
            def wait(self, timeout=None):
                if timeout is None and not self.killed:
                    raise KeyboardInterrupt
                return -2
        return P(0)

    rc = reeval_run.run_steps(reeval.plan(sel), run_root=tmp_path / "r1", render_python=tmp_path / "py",
                              in_process=lambda a: 0, popen=popen, report=lines.append)
    assert rc == reeval_run.INTERRUPTED and lines[-1].startswith("INTERRUPTED") and "resumes" in lines[-1]


def test_collect_with_nothing_evaluated_names_the_missing_results(tmp_path, capsys):
    assert main(["paper", "reeval", "--figure", "random-vs-filtered", "--backbone", "xvla", "--collect", "--host",
                 str(_host(tmp_path))]) == 2
    out = capsys.readouterr().out
    assert all(c.cell in out for c in reeval.select(CELLS, SPEC, reeval.Unit("figure", "fig4c", "xvla")))


@pytest.mark.parametrize("given", [None, "hf://axisrobotics/qvla-v1-results@v1.0", "LOCAL"])
def test_paper_tables_opens_the_default_or_the_named_results(tmp_path, monkeypatch, capsys, given):
    """No network: the opener (and the checksum manifest an hf:// source reads) is swapped. Omitted and spelled out, the v1.0 Hub results are opened; a
    local pack named with --results is opened instead."""
    from axis_qvla.paper import cli as paper_cli

    opened = []

    def fake_open(results, **kw):
        opened.append(results)
        raise paper_cli.FetchError("stop here")

    monkeypatch.setattr(paper_cli, "open_source", fake_open)
    monkeypatch.setattr(paper_cli, "load_lock", lambda path: {})
    local = str(tmp_path / "pack")
    lock = tmp_path / "lock.json"          # there, so the missing-manifest refusal does not stop it first
    lock.write_text("{}")
    argv = ["paper", "tables", "--check", "--cache", str(tmp_path / "cache"), "--checksums", str(lock)]
    if given is not None:
        argv += ["--results", local if given == "LOCAL" else given]
    assert main(argv) == 3
    assert opened == [local if given == "LOCAL" else paper_cli.DEFAULT_RESULTS]
