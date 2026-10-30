"""Development seeds: declared, disjoint from the test seeds, usable end to end, never filed
under a test arm."""
from __future__ import annotations

import json

import pytest

from axis_qvla.bench import execution, launch
from axis_qvla.bench.results.fingerprint import units_sha256
from axis_qvla.bench.sim import arms, campaign, cli, runner
from axis_qvla.bench.sim import queue_worker as qw
from axis_qvla.bench.spec import BenchmarkError, load_benchmark
from axis_qvla.core.hosts import load_host
from tests.unit.bench.sim.test_cli_end_to_end import dualsim_stubs, fixture_paths, stub_stack  # noqa: F401
from tests.unit.bench.sim.driver_stubs import driver_stack, make_checkpoint
from tests.unit.bench.test_eval_roots import PROFILE


def test_dev_seeds_are_declared_and_disjoint():
    b = load_benchmark()
    assert b.seeds_for(806, "dev") == tuple(806500 + i for i in range(50))
    assert not set(b.seeds_for(806, "dev")) & set(b.seeds_for(806, "test"))
    with pytest.raises(BenchmarkError, match="seed set"):
        b.seeds_for(806, "train")


def test_the_test_units_digest_is_unchanged_and_a_dev_one_differs():
    b = load_benchmark()
    assert units_sha256(b.task_ids, b.n_seeds).startswith("b3f14d19")   # the published results' digest
    assert units_sha256([806], 2, offset=500) != units_sha256([806], 2)


ROOTS = ["--log-root", "/logs", "--work-dir", "/work", "--task-data-root", "/tasks"]


def _ns(*extra):
    return cli.build_parser().parse_args(["--task-id", "806", "--variant", "x", "--checkpoint-dir", "c",
                                          *ROOTS, *extra])


def test_the_rollout_cli_resolves_dev_seeds_only_when_asked():
    src = cli.resolve_task_source(_ns("--seed-set", "dev"))
    assert src.seeds[0] == 806500 and src.seed_set == "dev"
    assert cli.resolve_task_source(_ns()).seeds[0] == 806000


@pytest.mark.parametrize("extra, why", [
    (["--variant", "x", "--seed-set", "dev"], "_dev"),
    (["--variant", "x_dev"], "development reads"),
    (["--variant", "x", "--run-name", "hx_806_x", "--seed-set", "dev"], "_dev"),
])
def test_the_rollout_cli_never_writes_one_seed_set_under_the_others_name(extra, why):
    """Refused before the server is contacted: the directory a rollout writes is named after the
    variant (or --run-name), and only a `_dev` name may hold development seeds."""
    with pytest.raises(SystemExit, match=why):
        cli.main(["--task-id", "806", "--checkpoint-dir", "c", *ROOTS, *extra])


def test_the_arm_name_says_which_seed_set_it_holds():
    assert arms.seed_set_arm("probe", "dev") == arms.seed_set_arm("probe_dev", "dev") == "probe_dev"
    assert arms.seed_set_arm("probe", "test") == "probe"
    with pytest.raises(arms.ArmError, match="development reads"):
        arms.seed_set_arm("probe_dev", "test")
    with pytest.raises(arms.ArmError, match="seed set"):
        arms.seed_set_arm("probe", "train")


def _driver_args(tmp_path, *extra):
    ck = tmp_path / "ck"
    ck.mkdir(exist_ok=True)
    (ck / "w.bin").write_bytes(b"x")
    return ["--config", "cfg", "--checkpoint", str(ck), "--tasks", "806", "--run-root", str(tmp_path / "r1"),
            "--log-root", str(tmp_path / "logs"), "--openpi-dir", str(tmp_path), "--n-seeds", "2",
            "--gpus", "0", "--dry-run", *extra]


def test_a_dev_run_must_be_filed_under_a_dev_arm(tmp_path):
    rest = _driver_args(tmp_path, "--seed-set", "dev")
    with pytest.raises(SystemExit, match="_dev"):
        campaign.main(["--run", "probe", *rest])
    assert campaign.main(["--run", "probe_dev", *rest]) == 0
    m = json.loads((tmp_path / "r1" / "out_probe_dev" / "dry_run_manifest.json").read_text())
    assert m["seed_set"] == "dev" and m["n_units"] == 2
    assert m["units_sha256"] == units_sha256([806], 2, offset=load_benchmark().dev["offset"])


def test_a_test_run_is_never_filed_under_a_dev_arm(tmp_path):
    with pytest.raises(SystemExit, match="development reads"):
        campaign.main(["--run", "probe_dev", *_driver_args(tmp_path)])


def test_the_worker_hands_dev_on_and_refuses_a_contradiction(tmp_path):
    dev = qw.WorkerConfig(queue_root=tmp_path, worker_id="w", port=1, variant="x_dev", checkpoint_dir="c",
                          seed_set="dev")
    cmd = qw._command(dev, 806, [806500])
    assert cmd[cmd.index("--seed-set") + 1] == "dev"
    test = qw.WorkerConfig(queue_root=tmp_path, worker_id="w", port=1, variant="x", checkpoint_dir="c")
    assert "--seed-set" not in qw._command(test, 806, [806000])
    with pytest.raises(SystemExit, match="--seed-set"):
        qw.refuse_second_copy(["--seed-set", "dev"], "--rollout-args", seed_set="test")


def test_compose_names_a_dev_arm_and_passes_the_set(tmp_path):
    (tmp_path / "hosts").mkdir()
    (tmp_path / "hosts" / "demo.toml").write_text(PROFILE.format(d=tmp_path))
    host = load_host("demo", hosts_dir=tmp_path / "hosts")
    out = launch.compose(launch.EvalRequest(checkpoint="c", config="cfg", arm="probe", seed_set="dev"),
                         host, load_benchmark())
    assert out.arm == "probe_dev" and "--run=probe_dev" in out.argv and "--seed-set=dev" in out.argv
    out = launch.compose(launch.EvalRequest(checkpoint="c", config="cfg", arm="probe"), host, load_benchmark())
    assert out.arm == "probe" and not [a for a in out.argv if a.startswith("--seed-set")]
    ext = launch.compose(launch.EvalRequest(checkpoint=None, config=None, arm="ext", seed_set="dev",
                                            policy=("127.0.0.1:9000",)), host, load_benchmark())
    assert ext.arm == "ext_dev" and "--run=ext_dev" in ext.argv and "--seed-set=dev" in ext.argv
    with pytest.raises(launch.LaunchError, match="development reads"):
        launch.compose(launch.EvalRequest(checkpoint="c", config="cfg", arm="probe_dev"), host, load_benchmark())


def test_one_check_refuses_a_misfiled_name_in_both_directions():
    arms.check_seed_set_arm("probe_dev", "dev")
    arms.check_seed_set_arm("probe", "test")
    with pytest.raises(arms.ArmError, match="must end in '_dev'"):
        arms.check_seed_set_arm("probe", "dev")
    with pytest.raises(arms.ArmError, match="development reads"):
        arms.check_seed_set_arm("probe_dev", "test")


def test_the_seed_refusals_name_the_seed_set_in_use():
    with pytest.raises(SystemExit, match="declares 2 dev seeds, need 3"):
        campaign.build_units([806], 3, lambda tid: [806500, 806501], "dev")
    with pytest.raises(runner.RunnerError, match="declared dev seeds"):
        runner.resolve_seeds([806500, 806501], None, only=[806000], seed_set="dev")
    seeds_help = next(a.help for a in cli.build_parser()._actions if "--seeds" in a.option_strings)
    assert "--seed-set" in seeds_help


# --- what the results builder reads: meta.json and arm_summary.json of a development run ----------

def _clean_execution_env(monkeypatch):
    for var in execution.ENV:
        monkeypatch.delenv(var, raising=False)


def test_a_dev_rollouts_meta_names_its_seed_set_and_what_it_executed(fixture_paths, dualsim_stubs, monkeypatch):
    _clean_execution_env(monkeypatch)
    for var, value in {**execution.convention_env(load_benchmark()), "AXIS_EVAL_REPLAN": "4"}.items():
        monkeypatch.setenv(var, value)
    data = str(fixture_paths["data"])
    assert cli.main(["--task-id", "806", "--variant", "probe_dev", "--checkpoint-dir", "/ckpt/c",
                     "--seed-set", "dev", "--seeds", "806500,806501", "--dual-sim",
                     "--log-root", str(fixture_paths["logs"]), "--work-dir", str(fixture_paths["work"]),
                     "--task-data-root", data, "--asset-dir", data, "--render-root", data]) == 0
    meta = json.loads((fixture_paths["logs"] / "probe_dev_dualsim" / "task_806" / "meta.json").read_text())
    assert meta["seed_set"] == "dev"
    assert meta["execution_env"] == {"AXIS_EVAL_REPLAN": "4", "AXIS_EVAL_ACTION_SPACE": "droid8",
                                     "AXIS_EVAL_IMAGE_GEOMETRY": "raw"}
    assert meta["execution_deviations"] == {"replan": {"executed": 4, "declared": 8}}
    assert meta["replan"] == 4
    assert [r["seed"] for r in dualsim_stubs["rollouts"]] == [806500, 806501]


def test_a_dev_runs_summary_and_manifest_name_the_seed_set(tmp_path, monkeypatch):
    seen = driver_stack(monkeypatch, tmp_path)
    _clean_execution_env(monkeypatch)
    rc = campaign.main(["--run", "probe_dev", "--config", "cfg", "--checkpoint", str(make_checkpoint(tmp_path)),
                        "--tasks", "806", "--run-root", str(tmp_path / "r1"), "--log-root", str(tmp_path / "logs"),
                        "--openpi-dir", str(tmp_path), "--n-seeds", "3", "--gpus", "0", "--workers-per-gpu", "1",
                        "--task-data-root", str(tmp_path / "tasks"), "--seed-set", "dev"])
    out = tmp_path / "r1" / "out_probe_dev"
    summary = json.loads((out / "arm_summary.json").read_text())
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert rc == 0 and summary["seed_set"] == manifest["seed_set"] == "dev"
    assert summary["rollouts"] == 3 and {ns.seed_set for ns in seen["rollouts"]} == {"dev"}
    assert manifest["execution"]["deviations"] == {} and manifest["execution"]["off_protocol"] is False
    assert manifest["units_sha256"] == units_sha256([806], 3, offset=500)
