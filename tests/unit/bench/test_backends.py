"""The three reference servers are declared once; every reader takes them from that table."""
from __future__ import annotations

import pytest

from axis_qvla.bench import backends, launch
from axis_qvla.bench.sim import campaign
from axis_qvla.cli import main


def test_one_table_feeds_every_reader():
    assert launch.BACKENDS == campaign.BACKENDS == backends.BACKENDS == ("openpi", "smolvla", "xvla")
    assert launch._BACKEND_ROOTS == {
        "openpi": {"logs": "bench", "openpi_dir": "openpi", "base_port": 34000},
        "smolvla": {"logs": "smolvla_bench", "openpi_dir": "smolvla_bench", "base_port": 35000},
        "xvla": {"logs": "smolvla_bench", "openpi_dir": "smolvla_bench", "base_port": 38000}}
    assert campaign.SMOLVLA_SERVER_CMD == backends.SMOLVLA_SERVER_CMD
    assert campaign.XVLA_SERVER_CMD == backends.XVLA_SERVER_CMD and campaign.XVLA_DOMAIN_ID == 20
    assert campaign._LEROBOT_SERVER_CMDS == {"smolvla": backends.SMOLVLA_SERVER_CMD,
                                             "xvla": backends.XVLA_SERVER_CMD}
    assert "--domain-id 20" in backends.XVLA_SERVER_CMD


def test_the_cli_offers_exactly_the_reference_backends():
    with pytest.raises(SystemExit) as e:
        main(["bench", "eval", "--backend", "other", "--checkpoint", "c", "--config", "x"])
    assert e.value.code == 2


@pytest.fixture
def table_host(tmp_path):
    """A host declaring every root/venv the table's fields can name, including an extra ROOT
    (`other_venv_root`) no reference server uses today, so a `server_venv_root` edit has
    somewhere new to point at."""
    (tmp_path / "hosts").mkdir()
    (tmp_path / "hosts" / "demo.toml").write_text(f"""launcher = "local"
gpus = 1
requires_secrets = []

[roots]
corpus = "{tmp_path}/corpus"
cache = "{tmp_path}/cache"
scratch = "{tmp_path}/scratch"
runs = "{tmp_path}/runs"
r1 = "{tmp_path}/r1"
bench = "{tmp_path}/bench"
openpi = "{tmp_path}/openpi"
eval = "{tmp_path}/eval"
smolvla_bench = "{tmp_path}/smolvla_bench"
smolvla_venv = "{tmp_path}/smolvla_venv"
other_venv_root = "{tmp_path}/other_venv_root"
hf_cache = "{tmp_path}/hf_cache"

[venv]
render = "{tmp_path}/venvs/render"
online = "{tmp_path}/venvs/online"
""")
    from axis_qvla.core.hosts import load_host

    return load_host("demo", hosts_dir=tmp_path / "hosts")


def test_the_three_reference_servers_compose_from_their_table_rows(table_host):
    """openpi takes the online venv and the benchmark's XLA pin and no HF env; SmolVLA and X-VLA
    take the torch server's venv root, no XLA pin and the offline HF env; DEFAULT_BACKEND is
    openpi."""
    from axis_qvla.bench.spec import load_benchmark

    h, b = table_host, load_benchmark()
    openpi = launch.compose(launch.EvalRequest(checkpoint="x", config="c", backend="openpi"), h, b)
    assert f"--server-python={h.venvs['online'] / 'bin' / 'python'}" in openpi.argv
    assert f"--xla-pin={b.xla_pin}" in openpi.argv and "HF_HUB_OFFLINE" not in openpi.env
    for name in ("smolvla", "xvla"):
        out = launch.compose(launch.EvalRequest(checkpoint="x", config="c", backend=name), h, b)
        assert f"--server-python={h.roots['smolvla_venv'] / 'bin' / 'python'}" in out.argv
        assert "--xla-pin=-" in out.argv and out.env["HF_HUB_OFFLINE"] == "1"
    # The default backend is the table's, wherever a default is needed.
    assert launch.EvalRequest(checkpoint="x", config="c").backend == backends.DEFAULT_BACKEND == "openpi"


def test_a_table_edit_changes_the_composed_env_and_the_guidance_refusal(table_host, monkeypatch):
    """Every table field decides something on its own: `server_venv_root` the root key of the
    server's interpreter, `hf_env` the LeRobot/HF env, `xla` the online venv and the XLA pin, and
    `can_guide` the guidance refusal. Editing one field changes exactly its behaviour. Each check
    replaces the UNPATCHED smolvla row with exactly one field changed, so earlier patches in this
    test never leak into a later one."""
    import dataclasses

    from axis_qvla.bench.spec import load_benchmark

    h, b = table_host, load_benchmark()
    baseline = backends.REFERENCE_SERVERS["smolvla"]
    req = launch.EvalRequest(checkpoint="x", config="c", backend="smolvla")

    # server_venv_root alone: a different ROOT KEY is looked up; xla/hf_env untouched.
    monkeypatch.setitem(backends.REFERENCE_SERVERS, "smolvla",
                        dataclasses.replace(baseline, server_venv_root="other_venv_root"))
    out = launch.compose(req, h, b)
    assert f"--server-python={h.roots['other_venv_root'] / 'bin' / 'python'}" in out.argv
    assert out.env["HF_HUB_OFFLINE"] == "1"

    # hf_env alone: the LeRobot/HF block is gone; the venv choice is untouched (still smolvla_venv).
    monkeypatch.setitem(backends.REFERENCE_SERVERS, "smolvla", dataclasses.replace(baseline, hf_env=False))
    out = launch.compose(req, h, b)
    assert f"--server-python={h.roots['smolvla_venv'] / 'bin' / 'python'}" in out.argv
    assert "HF_HUB_OFFLINE" not in out.env

    # xla alone: the online venv and the benchmark's own XLA pin; hf_env (still True here, since
    # only xla was changed from the baseline) keeps the HF env.
    monkeypatch.setitem(backends.REFERENCE_SERVERS, "smolvla", dataclasses.replace(baseline, xla=True))
    out = launch.compose(req, h, b)
    assert f"--server-python={h.venvs['online'] / 'bin' / 'python'}" in out.argv
    assert f"--xla-pin={b.xla_pin}" in out.argv and out.env["HF_HUB_OFFLINE"] == "1"

    # can_guide: unpatched smolvla cannot guide, and the refusal names it by the table's label.
    with pytest.raises(ValueError, match="SmolVLA"):
        campaign.serving_protocol("conditioned", "smolvla", 1.5)
    # Flip can_guide: the IDENTICAL call succeeds -- the refusal reads the table, not a backend list.
    monkeypatch.setitem(backends.REFERENCE_SERVERS, "smolvla", dataclasses.replace(baseline, can_guide=True))
    campaign.serving_protocol("conditioned", "smolvla", 1.5)   # succeeds


def test_the_guidance_refusal_names_the_label_alone(monkeypatch):
    """The refusal names the server by its declared short_label, never the whole label, whose
    ", <script path>" suffix is for the --backend help text."""
    with pytest.raises(ValueError, match=r"^the SmolVLA server cannot apply guidance") as exc:
        campaign.serving_protocol("conditioned", "smolvla", 1.5)
    assert "axis_qvla/bench/servers/lerobot_server.py" not in str(exc.value)

    # The declared field, not label.split(","): editing short_label alone must change the message,
    # with label (and its script path) untouched.
    import dataclasses

    monkeypatch.setitem(backends.REFERENCE_SERVERS, "smolvla",
                        dataclasses.replace(backends.REFERENCE_SERVERS["smolvla"], short_label="SmolVLA2"))
    with pytest.raises(ValueError, match=r"^the SmolVLA2 server cannot apply guidance"):
        campaign.serving_protocol("conditioned", "smolvla", 1.5)
    assert backends.REFERENCE_SERVERS["smolvla"].label == "SmolVLA, axis_qvla/bench/servers/lerobot_server.py"


def test_the_driver_backend_help_is_built_from_the_table():
    """The campaign driver's --backend help names every label from the table, as the CLI's does."""
    import io
    from contextlib import redirect_stdout

    out = io.StringIO()
    with redirect_stdout(out), pytest.raises(SystemExit):
        campaign.main(["--help"])
    text = " ".join(out.getvalue().split())   # argparse wraps long help lines
    for server in backends.REFERENCE_SERVERS.values():
        assert " ".join(server.label.split()) in text
