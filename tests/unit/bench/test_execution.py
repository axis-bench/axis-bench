"""The execution constants a run executes, against the ones its benchmark declares."""
from __future__ import annotations

import argparse
import json
import os

import pytest

from axis_qvla.bench import execution, launch
from axis_qvla.bench.spec import load_benchmark
from axis_qvla.cli import main
from axis_qvla.core.hosts import load_host
from tests.unit.bench.eval_stubs import eval_argv, fake_driver
from tests.unit.bench.test_eval_roots import PROFILE, host  # noqa: F401 - fixture reuse

LAUNCH_ENV = {"AXIS_EVAL_IMAGE_GEOMETRY": "raw", "AXIS_EVAL_ACTION_SPACE": "droid8"}


def test_the_benchmark_under_its_launch_environment_has_no_deviation():
    assert execution.deviations(execution.executed(LAUNCH_ENV), execution.declared(load_benchmark())) == {}


@pytest.mark.parametrize("env, key, value", [
    ({"PI05_REPLAN": "4"}, "replan", 4),
    ({"AXIS_EVAL_REPLAN": "2"}, "replan", 2),
    ({"MAX_ROLLOUT_STEPS": "900"}, "max_steps", 900),
    ({"N_STABLE_FRAMES": "30"}, "n_stable_frames", 30),
    ({"AXIS_EVAL_GRIP_STATE": "fingers"}, "grip_state", "fingers"),
    ({"AXIS_EVAL_EEF_IK": "iterate"}, "eef_ik", "iterate"),
    ({"AXIS_EVAL_TABLE_TEXTURE": "carpet.png"}, "table_texture", "carpet.png"),
    ({"AXIS_EVAL_IMAGE_GEOMETRY": ""}, "image_geometry", "profile"),   # unset: the letterbox default
])
def test_each_knob_is_a_named_deviation(env, key, value):
    dev = execution.deviations(execution.executed({**LAUNCH_ENV, **env}), execution.declared(load_benchmark()))
    assert set(dev) == {key} and dev[key]["executed"] == value


def test_a_malformed_number_is_refused_and_no_declaration_has_nothing_to_deviate_from():
    with pytest.raises(ValueError, match="replan"):
        execution.executed({"PI05_REPLAN": "four"})
    assert execution.deviations(execution.executed({"PI05_REPLAN": "4"}), None) == {}


def test_bench_eval_refuses_an_exported_override_and_records_one_it_is_told_to(host, monkeypatch, capsys):
    seen = fake_driver(monkeypatch, host)
    roots = ["--render-root", str(host / "render"), "--task-data-root", str(host / "tasks")]
    monkeypatch.setitem(os.environ, "PI05_REPLAN", "4")
    assert main(eval_argv(host, *roots)) == 2
    out = capsys.readouterr().out
    assert "replan: this run would execute 4" in out and "PI05_REPLAN" in out and seen == {}
    assert main(eval_argv(host, *roots, "--off-protocol")) == 0
    assert "--off-protocol" in seen["argv"]


def test_an_external_policy_run_is_held_to_the_same_declaration(host, monkeypatch, capsys):
    """--policy composes through the other branch; the refusal and the flag are the same."""
    seen = fake_driver(monkeypatch, host)
    ext = ["bench", "eval", "--host", "demo", "--policy", "127.0.0.1:9000", "--run", "ext",
           "--asset-dir", str(host / "assets"), "--render-root", str(host / "render"),
           "--task-data-root", str(host / "tasks")]
    monkeypatch.setitem(os.environ, "MAX_ROLLOUT_STEPS", "900")
    assert main(ext) == 2
    assert "max_steps: this run would execute 900" in capsys.readouterr().out and seen == {}
    assert main([*ext, "--off-protocol"]) == 0
    assert "--off-protocol" in seen["argv"]


def test_both_compose_branches_export_the_declared_conventions_from_one_place(tmp_path):
    """A reference run and a --policy run get the declaration's conventions from the same env
    builder, so neither can execute the code default where the other executes the declared one."""
    (tmp_path / "hosts").mkdir()
    (tmp_path / "hosts" / "demo.toml").write_text(PROFILE.format(d=tmp_path))
    h = load_host("demo", hosts_dir=tmp_path / "hosts")
    b = load_benchmark()
    for req in (launch.EvalRequest(checkpoint="c", config="cfg", arm="probe"),
                launch.EvalRequest(checkpoint=None, config=None, arm="ext", policy=("127.0.0.1:9000",))):
        env = launch.compose(req, h, b).env
        assert {k: env[k] for k in execution.CONVENTION_ENV} == execution.convention_env(b)
        assert {k: env[k] for k in execution.GL_ENV} == execution.GL_ENV
        assert execution.run_record(env, b, off_protocol=False)["deviations"] == {}


# --- what the driver's workers execute -------------------------------------------------------------

def test_the_driver_domain_is_the_domain_its_workers_hand_every_rollout(tmp_path):
    """`run_record` computes a driver run's verdict for DRIVER_DOMAIN; the worker builds every
    rollout's argv with a hard-coded --dual-sim. Read that argv with the rollout CLI's own parser
    and domain rule, so a change on either side fails here instead of misrecording a run."""
    from axis_qvla.bench.sim import arms
    from axis_qvla.bench.sim import cli as rollout_cli
    from axis_qvla.bench.sim import queue_worker as qw

    cfg = qw.WorkerConfig(queue_root=tmp_path, worker_id="w", port=1, variant="x", checkpoint_dir="c",
                          work_dir=tmp_path / "work", log_root=tmp_path / "logs",
                          extra_args=("--task-data-root", str(tmp_path / "tasks")))
    cmd = qw._command(cfg, 806, [806000])
    assert cmd[1:3] == ["-m", qw.ROLLOUT_MODULE] and qw.ROLLOUT_MODULE == rollout_cli.__name__
    assert rollout_cli._domain_of(rollout_cli.build_parser().parse_args(cmd[3:])) == execution.DRIVER_DOMAIN
    assert execution.DOMAINS is arms.DOMAINS and execution.DRIVER_DOMAIN in arms.DOMAINS


def test_the_gl_platform_is_pinned_for_every_worker_and_recorded(tmp_path, monkeypatch):
    """A caller's shell that exported another GL platform reaches no worker (mujoco's EGL context
    refuses a PYOPENGL_PLATFORM other than egl), and the run record names the platform the workers
    rendered through, beside MUJOCO_GL."""
    from axis_qvla.bench.sim import campaign
    from tests.unit.bench.sim.driver_stubs import SubprocessShim, driver_stack, make_checkpoint

    driver_stack(monkeypatch, tmp_path)
    started, popen = [], campaign.subprocess.Popen

    def recording_popen(cmd, **kw):
        started.append(kw["env"])
        return popen(cmd, **kw)

    monkeypatch.setattr(campaign, "subprocess", SubprocessShim(Popen=recording_popen))
    for var in execution.ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "osmesa")
    rc = campaign.main(["--run", "probe", "--config", "cfg", "--checkpoint", str(make_checkpoint(tmp_path)), "--tasks", "806",
                        "--run-root", str(tmp_path / "r1"), "--log-root", str(tmp_path / "logs"),
                        "--openpi-dir", str(tmp_path), "--n-seeds", "3", "--gpus", "0,1", "--workers-per-gpu", "1",
                        "--task-data-root", str(tmp_path / "tasks")])
    assert rc == 0 and len(started) == 2
    assert all({k: env[k] for k in execution.GL_ENV} == execution.GL_ENV for env in started)
    pins = json.loads((tmp_path / "r1" / "out_probe" / "run_manifest.json").read_text())["env_pins"]
    assert pins["MUJOCO_GL"] == pins["PYOPENGL_PLATFORM"] == "egl"
    assert campaign.PINNED_ENV.index("PYOPENGL_PLATFORM") == campaign.PINNED_ENV.index("MUJOCO_GL") + 1


def test_the_driver_refuses_and_records(tmp_path, monkeypatch):
    from axis_qvla.bench.sim import campaign

    ck = tmp_path / "ck"
    ck.mkdir()
    (ck / "w.bin").write_bytes(b"x")
    base = ["--run", "probe", "--config", "cfg", "--checkpoint", str(ck), "--tasks", "806",
            "--run-root", str(tmp_path / "r1"), "--log-root", str(tmp_path / "logs"),
            "--openpi-dir", str(tmp_path), "--n-seeds", "1", "--gpus", "0", "--dry-run"]
    monkeypatch.setenv("N_STABLE_FRAMES", "30")
    with pytest.raises(SystemExit, match="n_stable_frames: this run would execute 30"):
        campaign.main(base)
    assert not (tmp_path / "r1").exists()     # refused before any directory, queue or record
    assert campaign.main([*base, "--off-protocol"]) == 0
    m = json.loads((tmp_path / "r1" / "out_probe" / "dry_run_manifest.json").read_text())
    assert m["execution"]["off_protocol"] is True and m["execution"]["env"]["N_STABLE_FRAMES"] == "30"
    assert m["execution"]["deviations"] == {"n_stable_frames": {"executed": 30, "declared": 45}}
    assert m["env_pins"]["N_STABLE_FRAMES"] == "30"


def test_a_malformed_variable_is_a_refusal_not_a_traceback(host, monkeypatch, capsys):
    from axis_qvla.bench.sim import campaign

    seen = fake_driver(monkeypatch, host)
    monkeypatch.setitem(os.environ, "MAX_ROLLOUT_STEPS", "lots")
    roots = ["--render-root", str(host / "render"), "--task-data-root", str(host / "tasks")]
    assert main(eval_argv(host, *roots, "--off-protocol")) == 2
    assert "max_steps: MAX_ROLLOUT_STEPS='lots' is not an integer" in capsys.readouterr().out and seen == {}
    with pytest.raises(SystemExit, match="max_steps: MAX_ROLLOUT_STEPS='lots' is not an integer"):
        campaign.execution_guard(argparse.Namespace(tally_only=False, benchmark=None, off_protocol=True),
                                 dict(os.environ))


def test_a_tally_executes_nothing_so_the_shell_cannot_refuse_it(tmp_path, monkeypatch):
    """--tally-only starts no rollout: an exported knob is not a deviation of anything it does."""
    from axis_qvla.bench.sim import campaign

    for name, stub in (("log", lambda msg: None), ("python_versions", lambda python: {}),
                       ("gpu_info", lambda: {}), ("git_state", lambda d: {"path": str(d)})):
        monkeypatch.setattr(campaign, name, stub)
    monkeypatch.setenv("PI05_REPLAN", "4")
    rc = campaign.main(["--run", "probe", "--config", "cfg", "--checkpoint", str(tmp_path), "--tasks", "806",
                        "--run-root", str(tmp_path / "r1"), "--log-root", str(tmp_path / "logs"),
                        "--openpi-dir", str(tmp_path), "--n-seeds", "1", "--gpus", "0", "--tally-only"])
    assert rc == 3       # nothing on disk to tally: refused as an empty arm, not as off-protocol
    m = json.loads((tmp_path / "r1" / "out_probe" / "tally_manifest.json").read_text())
    assert m["execution"] is None


def test_a_resumed_run_must_execute_what_the_earlier_run_did(tmp_path, monkeypatch):
    from axis_qvla.bench.sim import campaign

    out = tmp_path / "out"
    out.mkdir()
    b = load_benchmark()
    before = execution.run_record(LAUNCH_ENV, b, off_protocol=False)
    (out / "run_manifest.json").write_text(json.dumps({"execution": before}))
    campaign.refuse_execution_change(out, execution.run_record(LAUNCH_ENV, b, off_protocol=False))
    with pytest.raises(SystemExit, match="replan: 8 -> 4"):
        campaign.refuse_execution_change(out, execution.run_record({**LAUNCH_ENV, "PI05_REPLAN": "4"}, b,
                                                                   off_protocol=True))
    (out / "run_manifest.json").write_text("{")
    with pytest.raises(SystemExit, match="cannot verify"):
        campaign.refuse_execution_change(out, before)
