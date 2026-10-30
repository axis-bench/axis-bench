"""`qvla bench eval` on a public checkout: the render and task-data roots are explicit inputs.

The rollout CLI requires the task-data root and defaults the render root to third_party/axis-render;
`bench eval` passes --render-root/--task-data-root through to every worker, refuses a run whose root
is missing before any server starts, and puts <render-root>/src on the workers' PYTHONPATH
(axis_replay_native lives there). Runs against a tmp host profile ($AXIS_HOSTS_DIR), so it needs no
machine profile.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from axis_qvla.bench import inputs, launch
from axis_qvla.cli import main
from axis_qvla.core.hosts import REPO_ROOT
from tests.unit.bench.eval_stubs import eval_argv, fake_driver

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

[venv]
render = "{d}/venvs/render"
online = "{d}/venvs/online"
"""
BANK = {"polyhaven_0000_a.jpg": b"aaa", "polyhaven_0001_b.jpg": b"bbbb"}


def make_bank(render_root: Path, lock_path: Path, monkeypatch, bank: dict | None = None) -> Path:
    """A lock with `bank`'s rows (plus one non-bank row) at lock_path, used by bench eval, and the
    same files under render_root; returns the bank directory."""
    bank = BANK if bank is None else bank
    rows = [{"path": inputs.BANK_LOCK_PREFIX + n, "sha256": hashlib.sha256(b).hexdigest(), "size": len(b)}
            for n, b in sorted(bank.items())]
    rows.append({"path": "assets/robots/a.stl", "sha256": "0" * 64, "size": 1})
    lock_path.write_text(json.dumps({"schema": "qvla.assets_lock/1", "files": rows}))
    monkeypatch.setattr(inputs, "ASSETS_LOCK", lock_path)
    d = render_root / inputs.BANK_RENDER_REL
    d.mkdir(parents=True, exist_ok=True)
    for n, b in bank.items():
        (d / n).write_bytes(b)
    return d


@pytest.fixture
def host(tmp_path, monkeypatch):
    hosts = tmp_path / "hosts"
    hosts.mkdir()
    (hosts / "demo.toml").write_text(PROFILE.format(d=tmp_path))
    monkeypatch.setenv("AXIS_HOSTS_DIR", str(hosts))
    monkeypatch.setattr(os, "environ", dict(os.environ))  # restored after the test
    for d in ("render", "tasks", "assets", "ck"):
        (tmp_path / d).mkdir()
    make_bank(tmp_path / "render", tmp_path / "lock.json", monkeypatch)
    return tmp_path


def test_the_roots_reach_the_driver_absolute_and_name_the_render_package(host, monkeypatch):
    seen = fake_driver(monkeypatch, host)
    monkeypatch.chdir(host)
    assert main(eval_argv(host, "--render-root", "render", "--task-data-root", "tasks")) == 0
    assert f"--render-root={host / 'render'}" in seen["argv"]
    assert f"--task-data-root={host / 'tasks'}" in seen["argv"]
    assert seen["env"]["PYTHONPATH"] == f"{REPO_ROOT}:{host / 'render' / 'src'}"


def test_without_the_flags_the_driver_argv_and_pythonpath_keep_their_defaults(host, monkeypatch):
    seen = fake_driver(monkeypatch, host)
    assert main(eval_argv(host, "--tally-only")) == 0
    assert not [a for a in seen["argv"] if a.startswith(("--render-root", "--task-data-root"))]
    assert seen["env"]["PYTHONPATH"] == f"{REPO_ROOT}:{REPO_ROOT / 'third_party' / 'axis-render' / 'src'}"


@pytest.mark.parametrize("flag", ["--render-root", "--task-data-root"])
def test_a_missing_root_is_refused_by_name_before_any_server(host, monkeypatch, capsys, flag):
    seen = fake_driver(monkeypatch, host)
    roots = {"--render-root": str(host / "render"), "--task-data-root": str(host / "tasks"),
             flag: str(host / "absent")}
    assert main(eval_argv(host, *[t for kv in roots.items() for t in kv])) == 2
    out = capsys.readouterr().out
    assert flag in out and "does not exist" in out and "before any server" in out
    assert seen == {}


def test_a_run_without_a_task_data_root_is_refused_by_name_before_any_server(host, monkeypatch, capsys):
    """No --task-data-root: the data-get location is used, and its absence is named with the command
    that fills it."""
    seen = fake_driver(monkeypatch, host)
    assert main(eval_argv(host, "--render-root", str(host / "render"))) == 2
    out = capsys.readouterr().out
    assert f"--task-data-root {host / 'corpus' / 'qvla-v1-bench' / 'tasks'} does not exist" in out
    assert "qvla data get bench" in out and "before any server" in out
    assert seen == {}


def test_compose_derives_the_render_package_path_from_the_render_root(host):
    from axis_qvla.bench.spec import load_benchmark
    from axis_qvla.core.hosts import load_host

    h, b = load_host("demo"), load_benchmark()
    given = launch.compose(launch.EvalRequest(checkpoint="x", config="c", render_root=str(host / "r")), h, b)
    default = launch.compose(launch.EvalRequest(checkpoint="x", config="c"), h, b)
    assert given.env["PYTHONPATH"] == f"{REPO_ROOT}:{host / 'r' / 'src'}"
    assert default.env["PYTHONPATH"] == f"{REPO_ROOT}:{REPO_ROOT / 'third_party' / 'axis-render' / 'src'}"


def printed_eval(text: str) -> tuple[str, list[str]]:
    """The interpreter and the `bench eval` arguments of the one command a run printed."""
    import shlex

    lines = [ln for ln in text.splitlines() if " -m axis_qvla.cli bench eval" in ln]
    assert len(lines) == 1, text
    toks = shlex.split(lines[0])
    assert toks[0] == f"PYTHONPATH={REPO_ROOT}", toks[0]
    i = toks.index("-m")
    assert toks[i + 1:i + 4] == ["axis_qvla.cli", "bench", "eval"], toks
    return toks[i - 1], toks[i + 4:]


def _listed_argv(text: str) -> list[str]:
    """The driver argv a dry run lists (`argv: -m axis_qvla.bench.sim.campaign ...`)."""
    import shlex

    line = next(ln for ln in text.splitlines() if ln.startswith("argv: -m axis_qvla.bench.sim.campaign "))
    return shlex.split(line)[3:]


def test_the_printed_command_is_bench_eval_itself_and_runs_what_the_dry_run_lists(host, monkeypatch, capsys):
    """The dry run and the venv refusal print one command: `bench eval` with the same arguments
    (never the driver it composes, which would skip every check `eval` makes), under the render
    venv. Run, it hands the driver exactly the argv the dry run listed -- --asset-dir,
    --render-root, --task-data-root and --asset-checksums included."""
    flags = ["--render-root", str(host / "render"), "--task-data-root", str(host / "tasks")]
    assert main(eval_argv(host, *flags, "--dry-run")) == 0
    out = capsys.readouterr().out
    listed = _listed_argv(out)
    for k in ("--asset-dir", "--render-root", "--task-data-root", "--asset-checksums"):
        assert any(t.startswith(k + "=") for t in listed), k
    python, args = printed_eval(out)
    assert python == str(host / "venvs" / "render" / "bin" / "python")
    pairs = lambda toks: sorted(zip(toks[::2], toks[1::2]))          # every option given here takes a value
    assert pairs(args) == pairs(eval_argv(host, *flags)[2:])          # the same arguments, less --dry-run
    assert main(eval_argv(host, *flags)) == 2                         # not under the render venv: refused
    assert printed_eval(capsys.readouterr().out) == (python, args)
    seen = fake_driver(monkeypatch, host)
    assert main(["bench", "eval", *args]) == 0
    assert seen["argv"] == listed


def test_a_printed_command_makes_the_texture_bank_check_it_was_printed_without(host, monkeypatch, capsys):
    """Off the render venv, a bank that is not the locked one is refused before any command is
    printed; and the command printed while the bank was fine refuses once it is not."""
    flags = ["--render-root", str(host / "render"), "--task-data-root", str(host / "tasks")]
    bank = host / "render" / inputs.BANK_RENDER_REL
    assert main(eval_argv(host, *flags)) == 2                         # not under the render venv
    _, args = printed_eval(capsys.readouterr().out)
    (bank / "polyhaven_0000_a.jpg").write_bytes(b"not the locked bytes")
    assert main(eval_argv(host, *flags)) == 2
    out = capsys.readouterr().out
    assert "sha256 differs from the checksum manifest" in out and " bench eval" not in out
    seen = fake_driver(monkeypatch, host)                             # now under the render venv
    assert main(["bench", "eval", *args]) == 2
    assert "sha256 differs from the checksum manifest" in capsys.readouterr().out and seen == {}


def test_off_the_render_venv_a_missing_asset_dir_or_root_prints_no_command(host, capsys):
    """With a corpus root declared, --asset-dir and --task-data-root default to the qvla-v1-bench
    copy `qvla data get bench` would have made there; this host's fixture never makes it, so each
    default is itself a missing root, named like any other."""
    assert main(["bench", "eval", "--host", "demo", "--checkpoint", str(host / "ck"), "--config", "cfg",
                 "--run", "probe", "--render-root", str(host / "render")]) == 2
    out = capsys.readouterr().out
    assert "--asset-dir" in out and "does not exist" in out and " bench eval" not in out
    assert main(eval_argv(host, "--render-root", str(host / "absent"), "--task-data-root", str(host / "tasks"))) == 2
    out = capsys.readouterr().out
    assert "--render-root" in out and "does not exist" in out and " bench eval" not in out
    assert main(eval_argv(host, "--render-root", str(host / "render"))) == 2
    out = capsys.readouterr().out
    assert "--task-data-root" in out and "does not exist" in out and " bench eval" not in out


def test_messages_name_the_program_the_user_ran(host, monkeypatch, capsys):
    """Every message that names a command names it after the parser the group was registered
    under (`qvla bench ...` when installed, `python -m axis_qvla.cli bench ...` from a checkout)."""
    with pytest.raises(SystemExit):
        main(["bench", "eval", "--help"])
    assert "qvla bench score --run-root" in capsys.readouterr().out
    roots = ["--render-root", str(host / "render"), "--task-data-root", str(host / "tasks")]
    assert main(eval_argv(host, *roots)) == 2                         # not under the render venv
    assert "`qvla bench eval` must run under the evaluation environment" in capsys.readouterr().out

    from axis_qvla.bench import spec

    conf = host / "conf_benchmarks"
    conf.mkdir()
    for name in ("franka_heldout24.json", "franka_heldout24.tasks.json", "franka_heldout24.layouts.json"):
        (conf / name).write_bytes((spec.CONF_DIR / name).read_bytes())
    t = json.loads((conf / "franka_heldout24.tasks.json").read_text())
    t["tasks"]["807"]["test_seeds"] = t["tasks"]["807"]["test_seeds"][:20]
    (conf / "franka_heldout24.tasks.json").write_text(json.dumps(t))
    monkeypatch.setattr(spec, "CONF_DIR", conf)
    assert main(eval_argv(host, "--dry-run")) == 2
    assert "(`qvla bench check`)" in capsys.readouterr().out


def test_the_eval_help_names_the_record_a_tally_writes(capsys):
    """--tally-only records itself in tally_manifest.json, beside the run's own
    run_manifest.json, and this command's --dry-run starts nothing and writes nothing but a paper
    checkpoint's rendered server config; the help described only run_manifest.json and
    arm_summary.json."""
    with pytest.raises(SystemExit):
        main(["bench", "eval", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    for needle in ("run_manifest.json", "arm_summary.json", "tally_manifest.json", "writes nothing"):
        assert needle in out, needle


def test_the_external_branch_shares_the_reference_branchs_eval_convention_env(host):
    """Both branches build their env with one `_worker_env` helper (AXIS_EVAL_IMAGE_GEOMETRY,
    AXIS_EVAL_ACTION_SPACE, the render tree on PYTHONPATH), so the same benchmark gives the same
    env: a pin on only one branch would silently miss --policy runs, a train/serve convention
    mismatch."""
    from axis_qvla.bench.spec import load_benchmark
    from axis_qvla.core.hosts import load_host

    h, b = load_host("demo"), load_benchmark()
    reference = launch.compose(launch.EvalRequest(checkpoint="x", config="c"), h, b)
    external = launch.compose(
        launch.EvalRequest(checkpoint=None, config=None, arm="ext", policy=("10.0.0.5:9000",)), h, b)
    for key in ("MUJOCO_GL", "AXIS_EVAL_IMAGE_GEOMETRY", "AXIS_EVAL_ACTION_SPACE", "PYTHONPATH",
                "AXIS_REPLAY_ASSET_CACHE_DIR"):
        assert reference.env[key] == external.env[key], key


def test_a_relative_checkpoint_reaches_the_driver_and_the_printed_command_absolute(host, monkeypatch, capsys):
    monkeypatch.chdir(host)
    flags = ["--render-root", str(host / "render"), "--task-data-root", str(host / "tasks")]
    given = ["bench", "eval", "--host", "demo", "--checkpoint", "ck", "--config", "cfg", "--run", "probe",
             "--asset-dir", str(host / "assets"), *flags]
    assert main([*given, "--dry-run"]) == 0
    _, args = printed_eval(capsys.readouterr().out)
    assert args[args.index("--checkpoint") + 1] == str(host / "ck")
    seen = fake_driver(monkeypatch, host)
    assert main(given) == 0
    assert f"--checkpoint={host / 'ck'}" in seen["argv"]
