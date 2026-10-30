"""Evaluating a server that is already running: `bench eval --policy` composes no server, the
driver checks and records every server's handshake and names the result directory from it."""
from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from axis_qvla.bench import launch
from axis_qvla.bench.sim import campaign
from axis_qvla.bench.spec import load_benchmark
from axis_qvla.cli import main
from axis_qvla.core.hosts import load_host
from tests.unit.bench import mock_server

_REAL_POPEN = subprocess.Popen

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
eval = "{d}/eval"

[venv]
render = "{d}/venvs/render"
"""


@pytest.fixture
def host(tmp_path):
    (tmp_path / "hosts").mkdir()
    (tmp_path / "hosts" / "demo.toml").write_text(PROFILE.format(d=tmp_path))
    return load_host("demo", hosts_dir=tmp_path / "hosts")


def test_compose_starts_no_server_and_needs_no_server_roots(host, tmp_path):
    req = launch.EvalRequest(checkpoint=None, config=None, arm="mine",
                             policy=("10.0.0.5:9000", "10.0.0.6:9000"))
    out = launch.compose(req, host, load_benchmark())
    assert "--policy=10.0.0.5:9000" in out.argv and "--policy=10.0.0.6:9000" in out.argv
    assert not [x for x in out.argv if x.startswith(("--server-python", "--openpi-dir", "--backend",
                                                       "--serving", "--config", "--checkpoint"))]
    assert f"--log-root={tmp_path / 'bench' / launch.ROLLOUT_LOGS_REL}" in out.argv
    assert "--xla-pin=-" in out.argv and out.python == tmp_path / "venvs" / "render" / "bin" / "python"


@pytest.mark.parametrize("extra, needle", [
    (["--backend", "smolvla", "--run", "x"], "--backend"),
    (["--serving", "conditioned", "--run", "x"], "--serving"),
    (["--config", "c", "--run", "x"], "--config"),
    ([], "--run"),
])
def test_policy_does_not_mix_with_reference_server_options(extra, needle, capsys):
    assert main(["bench", "eval", "--policy", "127.0.0.1:9", *extra]) == 2
    assert needle in capsys.readouterr().out


def test_without_policy_checkpoint_and_config_stay_required(capsys):
    assert main(["bench", "eval", "--run", "x"]) == 2
    assert "required unless --policy" in capsys.readouterr().out


def _argv(tmp_path, *extra):
    return ["--run", "ext", "--tasks", str(load_benchmark().declaration_path),
            "--log-root", str(tmp_path / "logs"), "--run-root", str(tmp_path / "r1"), "--gpus", "0", *extra]


def test_a_dry_run_records_the_servers_and_starts_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "start_server", lambda *a, **k: pytest.fail("a server was started"))
    assert campaign.main(_argv(tmp_path, "--policy", "127.0.0.1:9", "--dry-run")) == 0
    m = json.loads((tmp_path / "r1" / "out_ext" / "dry_run_manifest.json").read_text())
    assert m["serving"]["external"] == ["127.0.0.1:9"] and m["serving"]["backend"] == "external"
    assert m["checkpoint"] == {"path": None, "external": True}
    assert m["sources"]["server"] == {"external": ["127.0.0.1:9"]}
    # No openpi checkout is used under --policy; the default --openpi-dir must not be probed and
    # recorded as though it had run.
    assert m["sources"]["openpi"] == {"external": True}


def test_policy_refuses_options_that_describe_a_started_server(tmp_path):
    with pytest.raises(SystemExit, match="--serving"):
        campaign.main(_argv(tmp_path, "--policy", "127.0.0.1:9", "--serving", "bare", "--dry-run"))


def test_the_driver_refuses_config_alongside_policy_too(tmp_path):
    """`_policy_refusal` (cli.py) already refuses --policy + --config; the
    driver's own `external_protocol` must refuse the same combination instead of silently
    accepting and recording a --config that names no server this run actually talks to."""
    with pytest.raises(SystemExit, match="--config"):
        campaign.main(_argv(tmp_path, "--policy", "127.0.0.1:9", "--config", "c", "--dry-run"))


@pytest.fixture
def plain(tmp_path):
    proc, port = mock_server.start(tmp_path / "plain.log")
    yield port
    mock_server.stop(proc)


@pytest.fixture
def tagged(tmp_path):
    proc, port = mock_server.start(tmp_path / "tagged.log", "--quality-tag", "5")
    yield port
    mock_server.stop(proc)


def test_the_handshake_names_the_result_directory(plain, tagged):
    md = campaign.external_handshakes([("127.0.0.1", plain)], timeout_s=60)
    assert campaign.external_served("ext", md) == "ext_dualsim"
    md = campaign.external_handshakes([("127.0.0.1", tagged)], timeout_s=60)
    assert campaign.external_served("ext", md) == "ext_w0_dualsim"


def test_servers_that_disagree_on_conditioning_are_refused(plain, tagged):
    with pytest.raises(SystemExit, match="disagree"):
        campaign.external_handshakes([("127.0.0.1", plain), ("127.0.0.1", tagged)], timeout_s=60)


def test_two_models_under_the_same_conditioning_in_one_run_are_refused(tmp_path):
    """Two servers that agree on conditioning but serve different models are refused: workers are
    spread round-robin over the endpoints, so one arm's seed files would come from two models."""
    proc_a, port_a = mock_server.start(tmp_path / "a.log", "--ckpt", "model_A")
    proc_b, port_b = mock_server.start(tmp_path / "b.log", "--ckpt", "model_B")
    try:
        with pytest.raises(SystemExit, match=r"checkpoint_dir: 'model_A' vs 'model_B'"):
            campaign.external_handshakes([("127.0.0.1", port_a), ("127.0.0.1", port_b)], timeout_s=30)
    finally:
        mock_server.stop(proc_a)
        mock_server.stop(proc_b)


def test_two_identical_servers_are_accepted(tmp_path):
    proc_a, port_a = mock_server.start(tmp_path / "a.log")
    proc_b, port_b = mock_server.start(tmp_path / "b.log")
    try:
        mds = campaign.external_handshakes([("127.0.0.1", port_a), ("127.0.0.1", port_b)], timeout_s=30)
        assert len(mds) == 2
    finally:
        mock_server.stop(proc_a)
        mock_server.stop(proc_b)


def test_a_nonconformant_server_is_refused_before_any_rollout():
    srv, port = mock_server.bad_server({"supports_client_noise": True, "action_horizon": 0, "action_dim": 8},
                                       np.zeros((10, 8), np.float32))
    try:
        with pytest.raises(SystemExit, match="action_horizon"):
            campaign.external_handshakes([("127.0.0.1", port)], timeout_s=30)
    finally:
        srv.shutdown()


def test_a_tally_reads_the_recorded_handshake(tmp_path):
    out = tmp_path / "out_ext"
    out.mkdir()
    (out / "run_manifest.json").write_text(json.dumps({"serving": {"handshakes": [
        {"supports_client_noise": True, "action_horizon": 10, "action_dim": 32,
         "cfg_conditioning": True, "guidance_scale": 0.0}]}}))
    assert campaign.external_served("ext", campaign.recorded_handshakes(out)) == "ext_w0_dualsim"
    with pytest.raises(SystemExit, match="records no server handshake"):
        campaign.recorded_handshakes(tmp_path / "nothing")


def test_worker_argv_is_all_strings_without_a_checkpoint(tmp_path, monkeypatch):
    """`qvla bench eval --policy HOST:PORT --run <name>` with no --checkpoint -- what the CLI's own
    help text invites -- hands every worker a provenance STRING for --checkpoint-dir (provenance
    only on the rollout side, axis_qvla.bench.sim.cli), never `None`, which Popen cannot take, and the
    run reaches worker launch (handshake checked, manifest written, queue initialized)."""
    proc, port = mock_server.start(tmp_path / "srv.log")
    seen: list[list[str]] = []

    def popen(cmd, *a, **k):
        if isinstance(cmd, list) and any("queue_worker" in str(c) for c in cmd):
            seen.append(list(cmd))
            raise RuntimeError("intercepted worker")
        return _REAL_POPEN(cmd, *a, **k)

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(campaign, "start_server", lambda *a, **k: pytest.fail("a server was started"))
    try:
        with pytest.raises(RuntimeError, match="intercepted"):
            campaign.main(_argv(tmp_path, "--policy", f"127.0.0.1:{port}", "--ready-timeout-s", "60"))
    finally:
        mock_server.stop(proc)
    assert seen, "the run never reached worker launch"
    cmd = seen[0]
    assert all(isinstance(x, str) for x in cmd), cmd   # every element a str: Popen can take this argv
    i = cmd.index("--checkpoint-dir")
    assert cmd[i + 1] == f"external:127.0.0.1:{port}"
    # A real argv with a None element does raise TypeError at Popen: the property above matters.
    with pytest.raises(TypeError):
        _REAL_POPEN(["true", None])


def test_a_resumed_run_refuses_a_changed_conditioning(tmp_path, monkeypatch):
    """A resumed --policy run whose server now conditions differently from the handshake recorded
    in run_manifest.json is refused before the queue is reused or the manifest overwritten."""
    monkeypatch.setattr(campaign, "start_server", lambda *a, **k: pytest.fail("a server was started"))
    out_dir = tmp_path / "r1" / "out_ext"
    out_dir.mkdir(parents=True)
    recorded = {"serving": {"external": ["127.0.0.1:1"], "handshakes": [
        {"supports_client_noise": True, "action_horizon": 50, "action_dim": 32,
         "cfg_conditioning": False, "guidance_scale": 0.0}]}}
    (out_dir / "run_manifest.json").write_text(json.dumps(recorded))

    proc, port = mock_server.start(tmp_path / "tagged.log", "--quality-tag", "5")
    try:
        with pytest.raises(SystemExit, match="recorded a different server"):
            campaign.main(_argv(tmp_path, "--policy", f"127.0.0.1:{port}", "--ready-timeout-s", "60"))
    finally:
        mock_server.stop(proc)
    # Refused before the manifest is touched: a resumed run that disagrees must not overwrite it.
    assert json.loads((out_dir / "run_manifest.json").read_text()) == recorded


def test_a_different_model_under_the_same_conditioning_is_refused(tmp_path, monkeypatch):
    """A record of server A (checkpoint_dir "model_A", config_name "cfg_A", bare) followed by a
    mock server started with --ckpt model_B under the SAME conditioning must not reach worker
    launch, naming the identity key that changed and its old and new values."""
    monkeypatch.setattr(campaign, "start_server", lambda *a, **k: pytest.fail("a server was started"))
    out_dir = tmp_path / "r1" / "out_ext"
    out_dir.mkdir(parents=True)
    recorded = {"serving": {"external": ["127.0.0.1:1"], "handshakes": [
        {"supports_client_noise": True, "action_horizon": 50, "action_dim": 32,
         "cfg_conditioning": False, "guidance_scale": 0.0,
         "checkpoint_dir": "model_A", "config_name": "cfg_A"}]}}
    (out_dir / "run_manifest.json").write_text(json.dumps(recorded))

    proc, port = mock_server.start(tmp_path / "model_b.log", "--ckpt", "model_B", "--config-name", "cfg_A")
    try:
        with pytest.raises(SystemExit, match=r"checkpoint_dir: 'model_A' -> 'model_B'"):
            campaign.main(_argv(tmp_path, "--policy", f"127.0.0.1:{port}", "--ready-timeout-s", "60"))
    finally:
        mock_server.stop(proc)
    assert json.loads((out_dir / "run_manifest.json").read_text()) == recorded


def test_a_reference_server_run_record_is_refused(tmp_path, monkeypatch):
    """A bare `return` when no handshake was recorded let a resumed --policy run silently
    skip a record left by a REFERENCE-server run of the same arm (serving.external is None or
    absent, so it recorded no handshake either) -- a bare reference arm and a plain external arm
    both resolve to the same served directory (`<arm>_dualsim`), so this must be refused outright,
    not treated as nothing to compare."""
    monkeypatch.setattr(campaign, "start_server", lambda *a, **k: pytest.fail("a server was started"))
    out_dir = tmp_path / "r1" / "out_ext"
    out_dir.mkdir(parents=True)
    recorded = {"serving": {"external": None, "serving": "bare", "backend": "openpi"}}
    (out_dir / "run_manifest.json").write_text(json.dumps(recorded))

    proc, port = mock_server.start(tmp_path / "srv.log")
    try:
        with pytest.raises(SystemExit, match="reference-server run"):
            campaign.main(_argv(tmp_path, "--policy", f"127.0.0.1:{port}", "--ready-timeout-s", "60"))
    finally:
        mock_server.stop(proc)
    assert json.loads((out_dir / "run_manifest.json").read_text()) == recorded


def test_an_external_record_with_no_handshake_refuses(tmp_path, monkeypatch):
    """A --policy run's own record with serving.external set but no recorded handshake returned
    silently ("should not happen"). It is real information about the earlier run, same as a torn
    or unreadable record, so it must refuse the same way."""
    monkeypatch.setattr(campaign, "start_server", lambda *a, **k: pytest.fail("a server was started"))
    out_dir = tmp_path / "r1" / "out_ext"
    out_dir.mkdir(parents=True)
    recorded = {"serving": {"external": ["127.0.0.1:1"], "handshakes": None}}
    (out_dir / "run_manifest.json").write_text(json.dumps(recorded))

    proc, port = mock_server.start(tmp_path / "srv.log")
    try:
        with pytest.raises(SystemExit, match="cannot verify"):
            campaign.main(_argv(tmp_path, "--policy", f"127.0.0.1:{port}", "--ready-timeout-s", "60"))
    finally:
        mock_server.stop(proc)
    assert json.loads((out_dir / "run_manifest.json").read_text()) == recorded


def test_a_moved_endpoint_with_identical_declared_keys_is_accepted(tmp_path):
    """The endpoint itself (serving.external) is never compared: a restarted server may move host
    or port, and a resume must not refuse for that reason alone."""
    out_dir = tmp_path / "r1" / "out_ext"
    out_dir.mkdir(parents=True)
    proc, port = mock_server.start(tmp_path / "srv.log")
    try:
        md = campaign.external_handshakes([("127.0.0.1", port)], timeout_s=30)
        # A different (fake, long-gone) endpoint recorded; the same declared handshake either way.
        recorded = {"serving": {"external": ["127.0.0.1:1"], "handshakes": md}}
        (out_dir / "run_manifest.json").write_text(json.dumps(recorded))
        campaign.refuse_handshake_mismatch(out_dir, md, None)   # does not raise
    finally:
        mock_server.stop(proc)


def test_an_undeclared_extra_key_that_changed_is_accepted(tmp_path):
    """Keys the protocol permits but does not declare (a start time, a pid, a build stamp) are
    never part of the comparison, so changing one between runs must not refuse a resume."""
    out_dir = tmp_path / "r1" / "out_ext"
    out_dir.mkdir(parents=True)
    prev_hs = {"supports_client_noise": True, "action_horizon": 50, "action_dim": 32,
              "cfg_conditioning": False, "guidance_scale": 0.0, "pid": 111, "start_time": "t0"}
    now_hs = {**prev_hs, "pid": 222, "start_time": "t1"}
    recorded = {"serving": {"external": ["127.0.0.1:1"], "handshakes": [prev_hs]}}
    (out_dir / "run_manifest.json").write_text(json.dumps(recorded))
    campaign.refuse_handshake_mismatch(out_dir, [now_hs], None)   # does not raise


def test_the_no_identity_case_prints_a_notice(tmp_path, capsys):
    """A bare π0.5 server states no checkpoint_dir/config_name; when NEITHER the recorded nor the
    fresh handshake states an identity key, the resume is allowed but a notice says so -- the
    checked set is conditioning and shape only, not identity."""
    out_dir = tmp_path / "r1" / "out_ext"
    out_dir.mkdir(parents=True)
    bare_md = {"supports_client_noise": True, "action_horizon": 50, "action_dim": 32,
              "cfg_conditioning": False, "guidance_scale": 0.0}
    recorded = {"serving": {"external": ["127.0.0.1:1"], "handshakes": [bare_md]}}
    (out_dir / "run_manifest.json").write_text(json.dumps(recorded))
    campaign.refuse_handshake_mismatch(out_dir, [bare_md], None)
    assert "resume protection covers conditioning and shape only" in capsys.readouterr().out


def test_a_torn_record_refuses_instead_of_silently_resuming(tmp_path):
    """recorded_run_manifest turns a JSON parse error into an {"error": ...} record; the guard
    must refuse on that, never read it as "nothing to resume" -- a filesystem or parse problem is
    real information about the earlier run, not its absence."""
    out_dir = tmp_path / "r1" / "out_ext"
    out_dir.mkdir(parents=True)
    (out_dir / "run_manifest.json").write_text("{not valid json")
    with pytest.raises(SystemExit, match="cannot verify"):
        campaign.refuse_handshake_mismatch(out_dir, [{"cfg_conditioning": False}], None)


def test_an_unreadable_record_refuses_instead_of_silently_resuming(tmp_path):
    out_dir = tmp_path / "r1" / "out_ext"
    out_dir.mkdir(parents=True)
    p = out_dir / "run_manifest.json"
    p.write_text('{"serving": {"external": ["x"], "handshakes": [{}]}}')
    p.chmod(0)
    try:
        with pytest.raises(SystemExit, match="cannot verify"):
            campaign.refuse_handshake_mismatch(out_dir, [{"cfg_conditioning": False}], None)
    finally:
        p.chmod(0o644)   # tmp_path cleanup needs to read/delete it
