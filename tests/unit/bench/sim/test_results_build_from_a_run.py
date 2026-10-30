"""A driver run, through its workers and the real rollout CLI (server, layout planner and simulator
stubbed), leaves exactly the records `qvla results build` reads: the file it builds names the
run's seed set, execution, checkpoint digest and driver digest, and the scorer accepts it."""
from __future__ import annotations

import json
import subprocess
import sys
import types

import pytest

from axis_qvla.bench import execution, spec
from axis_qvla.bench.results.build import BuildError, build_results
from axis_qvla.bench.results.fingerprint import units_sha256
from axis_qvla.bench.results.score import score
from axis_qvla.bench.sim import campaign, cli
from axis_qvla.bench.sim import queue_worker as qw
from axis_qvla.bench.spec import load_benchmark
from tests.unit.bench.results.fixtures import make_bench
from tests.unit.bench.sim.test_cli_end_to_end import dualsim_stubs, fixture_paths, stub_stack  # noqa: F401
from tests.unit.bench.sim.driver_stubs import SubprocessShim, driver_stack, make_checkpoint


def _bench(tmp_path, monkeypatch):
    """A two-task, three-seed benchmark with development seeds and the release's execution.json, in
    the directory every benchmark name resolves against (the driver, the worker and the rollout
    CLI each load it by name)."""
    conf = make_bench(tmp_path / "bench", [806, 1050], 3, extra={"dev_seeds": {"offset": 500, "n": 3}})
    (conf / "execution.json").write_bytes((spec.CONF_DIR / "execution.json").read_bytes())
    monkeypatch.setattr(spec, "CONF_DIR", conf)
    b = load_benchmark("fixture_bench")
    # The driver's identity stamp names the declaration relative to the repository; this one lives
    # in tmp_path, so the stamp is the same identity with the path left absolute.
    monkeypatch.setattr(campaign, "benchmark_identity", lambda ids, name=None: {
        **b.identity(), "manifest": str(b.declaration_path), "task_ids": sorted(ids),
        "matches_manifest": sorted(ids) == sorted(b.task_ids)})
    return b


def _drive(tmp_path, monkeypatch, paths, arm: str, *flags: str, env: dict | None = None):
    """campaign.main over the whole benchmark; every worker batch runs the rollout CLI in process."""
    b = _bench(tmp_path, monkeypatch)
    driver_stack(monkeypatch, tmp_path)
    data = str(paths["data"])

    def rollout(cmd, **kw):
        assert cmd[1:3] == ["-m", qw.ROLLOUT_MODULE]
        rc = cli.main([*cmd[3:], "--task-data-root", data, "--asset-dir", data, "--render-root", data])
        return subprocess.CompletedProcess(cmd, rc, "", "")

    class _Res:
        def __init__(self, seed: int):
            self.success = seed % 2 == 0
            self.n_steps, self.sim_time = 100, 6.5
            self.reason = "success" if self.success else "timeout"

    backend = types.ModuleType("axis_qvla.bench.sim.rollout_dualsim")
    backend.rollout_pi05_dualsim = lambda task_dir, client, *, init_state_seed, **kw: _Res(init_state_seed)
    monkeypatch.setitem(sys.modules, "axis_qvla.bench.sim.rollout_dualsim", backend)
    monkeypatch.setattr(qw, "subprocess", SubprocessShim(run=rollout))
    for var in execution.ENV:
        monkeypatch.delenv(var, raising=False)
    for var, value in {**execution.convention_env(b), **(env or {})}.items():
        monkeypatch.setenv(var, value)
    rc = campaign.main(["--run", arm, "--config", "cfg", "--checkpoint", str(make_checkpoint(tmp_path)),
                        "--benchmark", b.name, "--tasks", str(b.declaration_path), "--run-root", str(tmp_path / "r1"),
                        "--log-root", str(tmp_path / "logs"), "--openpi-dir", str(tmp_path),
                        "--gpus", "0", "--workers-per-gpu", "1", *flags])
    return rc, b


def _manifest(tmp_path, arm: str) -> dict:
    return json.loads((tmp_path / "r1" / f"out_{arm}" / "run_manifest.json").read_text())


def test_a_test_run_builds_from_its_own_records_and_scores(tmp_path, monkeypatch, fixture_paths, dualsim_stubs):
    rc, b = _drive(tmp_path, monkeypatch, fixture_paths, "probe")
    assert rc == 0
    r = build_results(tmp_path / "r1", "probe", b, ft_seed=42, pretrain_seed=None)
    h, m = r.header, _manifest(tmp_path, "probe")
    assert h.bench == {"name": b.name, "version": b.version, "version_sha256": b.version_sha256,
                       "declaration_sha256": b.sha256, "seed_set": "test", "identity_source": "run_record",
                       "units_sha256": units_sha256(b.task_ids, b.n_seeds)}
    assert h.bench["units_sha256"] == m["units_sha256"]                  # the units the run planned
    assert h.checkpoint["eval_digest"] == m["checkpoint"]["sha256"]
    driver = m["sources"]["files"]["axis_qvla/bench/sim/campaign.py"]
    assert driver is not None and h.harness["driver_sha256"] == driver
    assert h.execution == {"fps": 15, "replan": 8, "max_steps": 1500, "n_stable_frames": 45, "off_protocol": False}
    assert (h.served, h.serve.fields, h.image_geometry, h.action_space) == ("probe_dualsim", {}, "raw", "droid8")
    s = score(r, b)
    assert (s.rollouts, s.missing, round(s.pct, 2)) == (6, 0, 66.67)      # even seeds succeed: 2/3 a task


def test_a_dev_run_builds_as_a_dev_run_only(tmp_path, monkeypatch, fixture_paths, dualsim_stubs):
    rc, b = _drive(tmp_path, monkeypatch, fixture_paths, "probe_dev", "--seed-set", "dev")
    assert rc == 0
    r = build_results(tmp_path / "r1", "probe_dev", b, ft_seed=42, pretrain_seed=None)
    h = r.header
    assert h.bench["seed_set"] == "dev" and h.served == "probe_dev_dualsim"
    assert {x.seed for x in r.rows} == {t * 1000 + 500 + i for t in b.task_ids for i in range(3)}
    assert h.bench["units_sha256"] == _manifest(tmp_path, "probe_dev")["units_sha256"] == units_sha256(
        b.task_ids, b.n_seeds, offset=b.dev["offset"])
    with pytest.raises(BuildError, match="no arm_summary.json"):
        build_results(tmp_path / "r1", "probe", b, ft_seed=42, pretrain_seed=None)


def test_an_off_protocol_run_says_so_in_its_header(tmp_path, monkeypatch, fixture_paths, dualsim_stubs):
    rc, b = _drive(tmp_path, monkeypatch, fixture_paths, "probe", "--off-protocol", env={"AXIS_EVAL_REPLAN": "4"})
    assert rc == 0
    ex = build_results(tmp_path / "r1", "probe", b, ft_seed=42, pretrain_seed=None).header.execution
    assert ex["off_protocol"] is True and ex["replan"] == 4
    assert ex["deviations"] == {"replan": {"executed": 4, "declared": 8}}
    assert _manifest(tmp_path, "probe")["execution"]["deviations"] == ex["deviations"]
