"""`bench eval --checkpoint-id`: the pi0.5 reference server gets the config of the paper arm that
trained the checkpoint, rendered for serving, as a file (openpi's clean line registers no config by
name). The training package is replaced by a stub with its declared interface, so this runs
without it; tests/golden/bench/test_openpi_serve_config.py runs the real one over the paper arms."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from axis_qvla.bench import openpi_serve
from axis_qvla.core.canon import dumps_canonical
from axis_qvla.cli import main

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
ARMS = {"pi05/ft/x": "pi05/finetune/x.toml", "smolvla/ft/y": "smolvla/finetune/y.toml"}
SPECS = {"pi05/finetune/x.toml": SimpleNamespace(id="pi05.finetune.x", backbone="pi05-openpi",
                                                 preset="pi05.finetune.lora.paper"),
         "smolvla/finetune/y.toml": SimpleNamespace(id="smolvla.finetune.y", backbone="smolvla-lerobot",
                                                    preset="smolvla.finetune.paper")}


@pytest.fixture
def train(monkeypatch):
    calls: list = []

    def find_arm_by_checkpoint(cid):
        if cid not in ARMS:
            raise LookupError(f"0 paper runs name checkpoint {cid!r}")
        return ARMS[cid]

    def openpi_config(spec, recipe, paths, *, smoke=False, bundle=None, serve=False):
        calls.append((spec.id, recipe, serve))
        return {"identity": {"arm": spec.id}}

    for name, mod in {"axis_qvla.train.paper": SimpleNamespace(find_arm_by_checkpoint=find_arm_by_checkpoint),
                      "axis_qvla.train.armspec": SimpleNamespace(load_armspec=SPECS.__getitem__),
                      "axis_qvla.train.recipes": SimpleNamespace(load_recipe=lambda preset: f"recipe {preset}"),
                      "axis_qvla.train.generate": SimpleNamespace(openpi_config=openpi_config)}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


def test_the_config_is_the_paper_arms_rendered_for_serving(train):
    data = openpi_serve.render_serve_config("pi05/ft/x", paths=None)
    assert data == dumps_canonical({"identity": {"arm": "pi05.finetune.x"}})
    assert train == [("pi05.finetune.x", "recipe pi05.finetune.lora.paper", True)]


def test_an_unknown_checkpoint_or_a_lerobot_arm_is_refused(train):
    with pytest.raises(openpi_serve.ServeConfigError, match="0 paper runs"):
        openpi_serve.render_serve_config("pi05/ft/nope", paths=None)
    with pytest.raises(openpi_serve.ServeConfigError, match="only pi0.5 runs"):
        openpi_serve.render_serve_config("smolvla/ft/y", paths=None)


def test_the_config_sits_beside_the_run_manifest():
    assert openpi_serve.config_path(Path("r1"), "a") == Path("r1") / "out_a" / "serve" / "openpi_config.json"


@pytest.fixture
def host(tmp_path, monkeypatch):
    (tmp_path / "hosts").mkdir()
    (tmp_path / "hosts" / "demo.toml").write_text(PROFILE.format(d=tmp_path))
    monkeypatch.setenv("AXIS_HOSTS_DIR", str(tmp_path / "hosts"))
    monkeypatch.setattr(os, "environ", dict(os.environ))  # restored after the test
    (tmp_path / "ck").mkdir()
    return tmp_path


STUB = dumps_canonical({"identity": {"arm": "pi05.finetune.x"}})


def _driver(monkeypatch, d: Path) -> dict:
    """The campaign driver replaced by a recorder of its argv and of the config file it was handed,
    as the file stood when the driver started."""
    from axis_qvla.bench.sim import campaign

    seen: dict = {}

    def fake_main(argv):
        seen["argv"] = list(argv)
        cfg = Path(next(x for x in argv if x.startswith("--config="))[len("--config="):])
        seen["config_at_start"] = cfg.read_bytes() if cfg.is_file() else None
        return 0
    monkeypatch.setattr(campaign, "main", fake_main)
    monkeypatch.setattr(sys, "executable", str(d / "venvs" / "render" / "bin" / "python"))
    from tests.unit.bench.eval_stubs import ready_host

    ready_host(monkeypatch, d)
    return seen


def _eval(d: Path, *extra: str) -> list[str]:
    return ["bench", "eval", "--host", "demo", "--checkpoint", str(d / "ck"), "--run", "a", *extra]


def _inputs(d: Path, monkeypatch) -> list[str]:
    """The on-disk inputs a real run checks before the driver: a locked Polyhaven bank, tasks, assets."""
    from tests.unit.bench.test_eval_roots import make_bank

    make_bank(d / "render", d / "lock.json", monkeypatch)
    for sub in ("tasks", "assets"):
        (d / sub).mkdir()
    return ["--asset-dir", str(d / "assets"), "--render-root", str(d / "render"), "--task-data-root", str(d / "tasks")]


def _configs_named(out: str) -> list[Path]:
    """Every config path the printed commands carry: `--config=` tokens and the `serve config:` line."""
    return ([Path(m) for m in re.findall(r"--config=(\S+)", out)]
            + [Path(m) for m in re.findall(r"^serve config: (\S+)", out, re.M)])


def test_the_server_is_handed_the_rendered_file(train, host, monkeypatch):
    seen = _driver(monkeypatch, host)
    assert main(_eval(host, "--checkpoint-id", "pi05/ft/x", *_inputs(host, monkeypatch))) == 0
    cfg = host / "r1" / "out_a" / "serve" / "openpi_config.json"
    assert f"--config={cfg}" in seen["argv"]
    assert seen["config_at_start"] == STUB and json.loads(cfg.read_bytes()) == {"identity": {"arm": "pi05.finetune.x"}}


def test_a_dry_run_writes_the_config_its_commands_name(train, host, capsys):
    """The dry run lists the driver argv and `serve config:`; both name the rendered file, and the
    file is there. The command it prints is `bench eval` again, which renders the same bytes."""
    from tests.unit.bench.test_eval_roots import printed_eval

    assert main(_eval(host, "--checkpoint-id", "pi05/ft/x", "--dry-run")) == 0
    out = capsys.readouterr().out
    named = _configs_named(out)
    assert len(named) == 2 and len(set(named)) == 1, named
    assert named[0] == host / "r1" / "out_a" / "serve" / "openpi_config.json" and named[0].read_bytes() == STUB
    assert "written; rendered for pi05/ft/x" in out
    _, args = printed_eval(out)
    assert args[args.index("--checkpoint-id") + 1] == "pi05/ft/x" and "--dry-run" not in args


def test_the_venv_refusal_prints_bench_eval_again_with_the_config_written(train, host, monkeypatch, capsys):
    """Run from any interpreter but the render venv, `bench eval` refuses with the command to run
    instead -- itself, under the render venv -- and the config that run will find is on disk."""
    from axis_qvla.bench.sim import campaign
    from tests.unit.bench.test_eval_roots import printed_eval

    monkeypatch.setattr(campaign, "main", lambda argv: pytest.fail("the driver was started"))
    given = _eval(host, "--checkpoint-id", "pi05/ft/x", *_inputs(host, monkeypatch))
    assert main(given) == 2
    out = capsys.readouterr().out
    assert "must run under the evaluation environment" in out
    python, args = printed_eval(out)
    assert python == str(host / "venvs" / "render" / "bin" / "python")
    assert sorted(zip(args[::2], args[1::2])) == sorted(zip(given[2::2], given[3::2]))
    assert (host / "r1" / "out_a" / "serve" / "openpi_config.json").read_bytes() == STUB


def test_a_tally_neither_writes_nor_compares_the_config(train, host, monkeypatch):
    """A tally starts no server: it writes no config, and one recorded earlier, whatever it holds,
    neither refuses it nor is touched."""
    seen = _driver(monkeypatch, host)
    cfg = host / "r1" / "out_a" / "serve" / "openpi_config.json"
    assert main(_eval(host, "--checkpoint-id", "pi05/ft/x", "--tally-only")) == 0
    assert f"--config={cfg}" in seen["argv"] and not cfg.exists()
    cfg.parent.mkdir(parents=True)
    cfg.write_bytes(b'{"identity": {"arm": "rendered before a generator change"}}')
    assert main(_eval(host, "--checkpoint-id", "pi05/ft/x", "--tally-only")) == 0
    assert cfg.read_bytes() == b'{"identity": {"arm": "rendered before a generator change"}}'


@pytest.mark.parametrize("extra, needle", [
    (["--checkpoint-id", "pi05/ft/x", "--config", "c"], "not also pass --config"),
    (["--checkpoint-id", "pi05/ft/x", "--backend", "smolvla"], "--backend openpi"),
    (["--checkpoint-id", "pi05/ft/nope"], "0 paper runs"),
])
def test_what_cannot_be_served_is_refused_before_anything_starts(train, host, monkeypatch, capsys, extra, needle):
    seen = _driver(monkeypatch, host)
    assert main(_eval(host, *extra, "--tally-only")) == 2
    assert needle in capsys.readouterr().out and "argv" not in seen


def test_a_refusal_of_the_generator_is_a_serve_config_error(train, monkeypatch):
    def refuses(spec, recipe, paths, *, serve=False):
        raise ValueError(f"{spec.id}: a declaration, not a run qvla train renders")
    monkeypatch.setattr(sys.modules["axis_qvla.train.generate"], "openpi_config", refuses)
    with pytest.raises(openpi_serve.ServeConfigError, match=r"pi05/ft/x \(pi05\.finetune\.x\): .*a declaration"):
        openpi_serve.render_serve_config("pi05/ft/x", paths=None)


def test_a_dev_run_keeps_its_config_beside_its_own_run_manifest(train, host, capsys):
    assert main(_eval(host, "--checkpoint-id", "pi05/ft/x", "--seed-set", "dev", "--dry-run")) == 0
    out = capsys.readouterr().out
    cfg = host / "r1" / "out_a_dev" / "serve" / "openpi_config.json"
    assert "--run=a_dev" in out and set(_configs_named(out)) == {cfg}
    assert cfg.read_bytes() == STUB and not (host / "r1" / "out_a").exists()


def test_a_resume_keeps_the_config_and_another_config_is_refused(train, host, capsys):
    cfg = host / "r1" / "out_a" / "serve" / "openpi_config.json"
    assert main(_eval(host, "--checkpoint-id", "pi05/ft/x", "--dry-run")) == 0
    assert main(_eval(host, "--checkpoint-id", "pi05/ft/x", "--dry-run")) == 0
    assert cfg.read_bytes() == STUB
    capsys.readouterr()
    cfg.write_bytes(b'{"identity": {"arm": "another"}}')
    (host / "r1" / "out_a" / "run_manifest.json").write_text("{}")     # a run of this arm started
    assert main(_eval(host, "--checkpoint-id", "pi05/ft/x", "--dry-run")) == 2
    out = capsys.readouterr().out
    assert "already records another config for this run" in out and "differ in identity.arm" in out
    assert f"A run under this name recorded it ({host / 'r1' / 'out_a' / 'run_manifest.json'})" in out
    assert "new --run" in out and "moved [roots] entry" in out and "argv:" not in out
    assert cfg.read_bytes() == b'{"identity": {"arm": "another"}}'


def test_a_config_left_by_a_dry_run_names_the_file_to_delete(train, host, capsys):
    """A dry run with the wrong id writes its config; the corrected run is refused (rollouts must
    never be filed under two configs), and since no run of the arm started, the refusal names
    exactly the file to delete. Deleting it lets the corrected run through."""
    cfg = host / "r1" / "out_a" / "serve" / "openpi_config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_bytes(b'{"identity": {"arm": "a wrong id"}}')          # the wrong id's dry run
    assert main(_eval(host, "--checkpoint-id", "pi05/ft/x", "--dry-run")) == 2
    out = capsys.readouterr().out
    assert f"No run under this name started ({host / 'r1' / 'out_a' / 'run_manifest.json'} does not exist)" in out
    assert f"delete {cfg} and run again." in out and "new --run" not in out
    cfg.unlink()
    assert main(_eval(host, "--checkpoint-id", "pi05/ft/x", "--dry-run")) == 0
    assert cfg.read_bytes() == STUB


@pytest.mark.parametrize("arm", ["x/../../elsewhere", "a/b", "a\\b", "..", "a..b"])
def test_an_arm_that_is_not_one_path_component_is_refused_before_anything_is_written(train, host, capsys, arm):
    argv = ["bench", "eval", "--host", "demo", "--checkpoint", str(host / "ck"), "--run", arm,
            "--checkpoint-id", "pi05/ft/x", "--dry-run"]
    assert main(argv) == 2
    out = capsys.readouterr().out
    assert out.startswith("FAIL  ") and "must be one path component" in out
    assert not (host / "r1").exists() and not (host / "elsewhere").exists()


def test_a_config_that_cannot_be_written_is_a_refusal_line(train, host, capsys):
    """The dry run and every refusal print commands that name the file, so a run root it cannot be
    written under is refused, and says why, rather than printing a command that cannot run. The run
    file's temporary name is taken by a directory, so the write fails whoever runs the test (a mode
    bit would not stop root)."""
    (host / "r1" / "out_a" / "serve" / "openpi_config.json.tmp").mkdir(parents=True)
    assert main(_eval(host, "--checkpoint-id", "pi05/ft/x", "--dry-run")) == 2
    out = capsys.readouterr().out
    assert out.startswith("FAIL  the server's config cannot be written to ") and "IsADirectoryError" in out
    assert "argv:" not in out and not (host / "r1" / "out_a" / "serve" / "openpi_config.json").exists()


@pytest.mark.parametrize("argv, needle", [
    (["--policy", "127.0.0.1:9", "--run", "a", "--checkpoint-id", "pi05/ft/x"], "--checkpoint-id describe"),
])
def test_checkpoint_id_needs_a_checkpoint_and_no_running_server(argv, needle, capsys):
    assert main(["bench", "eval", *argv]) == 2
    assert needle in capsys.readouterr().out


def test_a_checkpoint_id_outside_the_results_map_needs_a_checkpoint(train, host, capsys):
    assert main(["bench", "eval", "--host", "demo", "--checkpoint-id", "pi05/ft/x", "--run", "a", "--tally-only"]) == 2
    assert "needs --checkpoint" in capsys.readouterr().out


def test_the_server_reads_the_file_with_its_own_flags_before_the_subcommand(train, host, monkeypatch):
    """openpi's serve_policy.py takes its own flags (--port, --quality-tag, --guidance-scale) before
    `policy:checkpoint` and refuses one after it; the rendered file is `--policy.config`."""
    from axis_qvla.bench.sim import campaign

    seen = _driver(monkeypatch, host)
    assert main(_eval(host, "--checkpoint-id", "pi05/ft/x", "--serving", "conditioned", "--tally-only")) == 0
    given = dict(x[2:].split("=", 1) for x in seen["argv"] if x.startswith("--") and "=" in x)
    flags = campaign.serving_protocol(given["serving"], given["backend"], float(given["guidance-scale"])).serve_flags
    argv = campaign.server_command(0, 34000, given["config"], given["checkpoint"], flags.split(), {"SERVER_PYTHON": "py"})
    cfg = host / "r1" / "out_a" / "serve" / "openpi_config.json"
    assert argv == ["py", "scripts/serve_policy.py", "--port=34000", "--quality-tag", "5", "--guidance-scale", "0",
                    "policy:checkpoint", f"--policy.config={cfg}", f"--policy.dir={host / 'ck'}"]


# --- a paper config name as --config ---------------------------------------------------------

def test_a_paper_config_name_is_served_as_its_checkpoint_id(train, host, monkeypatch, capsys):
    """--config <paper config name> takes the --checkpoint-id path: the name resolves (with the
    arm) to a checkpoint id, whose rendered config the driver is handed in place of the name."""
    asked = []
    monkeypatch.setattr(openpi_serve, "checkpoint_for_config",
                        lambda config, arm: asked.append((config, arm)) or ("pi05/ft/x" if config == "paper_cfg" else None))
    assert main(_eval(host, "--config", "paper_cfg", "--dry-run")) == 0
    out = capsys.readouterr().out
    cfg = host / "r1" / "out_a" / "serve" / "openpi_config.json"
    assert asked == [("paper_cfg", "a")] and set(_configs_named(out)) == {cfg} and cfg.read_bytes() == STUB
    assert "--config=paper_cfg" not in out and "rendered for pi05/ft/x" in out


@pytest.mark.parametrize("extra", [["--config", "cfg"], ["--config", "paper_cfg", "--backend", "smolvla"]])
def test_any_other_config_reaches_the_server_as_given(train, host, monkeypatch, capsys, extra):
    """A name no paper run carries, or any config of a LeRobot server, is handed on unchanged."""
    monkeypatch.setattr(openpi_serve, "checkpoint_for_config",
                        lambda config, arm: "pi05/ft/x" if config == "paper_cfg" else None)
    lerobot_roots = "".join(f'{k} = "{host}/{k}"\n' for k in ("smolvla_venv", "smolvla_bench", "hf_cache"))
    (host / "hosts" / "demo.toml").write_text(PROFILE.format(d=host).replace("[venv]", lerobot_roots + "\n[venv]"))
    assert main(_eval(host, *extra, "--dry-run")) == 0
    out = capsys.readouterr().out
    assert f"--config={extra[1]}" in out and "serve config:" not in out and not (host / "r1").exists()


def test_an_ambiguous_paper_config_name_is_refused_before_anything_starts(host, monkeypatch, capsys):
    seen = _driver(monkeypatch, host)
    assert main(_eval(host, "--config", "pi05_axis_heldout_qual_v2_bc", "--tally-only")) == 2
    out = capsys.readouterr().out
    assert "paper config of 2 checkpoints" in out and "--checkpoint-id" in out and "argv" not in seen


# --- the driver: a config file, its digest, and the tokenizer openpi downloads ---------------------

def _driver_argv(tmp_path: Path, config: str, *extra: str) -> list[str]:
    from tests.unit.bench.sim.driver_stubs import make_checkpoint

    return ["--run", "probe", "--config", config, "--checkpoint", str(make_checkpoint(tmp_path)), "--tasks", "806",
            "--run-root", str(tmp_path / "r1"), "--log-root", str(tmp_path / "logs"),
            "--openpi-dir", str(tmp_path), "--n-seeds", "1", "--gpus", "0", *extra]


def test_the_driver_refuses_a_missing_config_file_before_any_record(tmp_path):
    from axis_qvla.bench.sim import campaign

    with pytest.raises(SystemExit, match="x.json does not exist; refused before any record, queue or server"):
        campaign.main(_driver_argv(tmp_path, str(tmp_path / "x.json"), "--dry-run"))
    assert not (tmp_path / "r1").exists()


def test_the_run_record_carries_the_config_files_sha256(tmp_path, monkeypatch):
    from axis_qvla.bench.sim import campaign
    from tests.unit.bench.sim.driver_stubs import stub_probes

    stub_probes(monkeypatch)
    cfg = tmp_path / "c.json"
    cfg.write_bytes(STUB)
    assert campaign.main(_driver_argv(tmp_path, str(cfg), "--dry-run")) == 0
    m = json.loads((tmp_path / "r1" / "out_probe" / "dry_run_manifest.json").read_text())
    assert m["config"] == str(cfg) and m["config_sha256"] == hashlib.sha256(STUB).hexdigest()
    assert campaign.config_sha256("pi05_axis_heldout_qual_v2_bc") is None


@pytest.mark.parametrize("seeded", [False, True])
def test_a_server_start_without_the_tokenizer_names_the_file_and_the_root(tmp_path, monkeypatch, seeded):
    """Offline, openpi's server cannot download its tokenizer and exits at start: the failure line
    names the file and the download cache it was looked for in. With the file there, the failure is
    reported as it was."""
    from axis_qvla.bench.sim import campaign
    from tests.unit.bench.sim.driver_stubs import FakeProc, stub_probes

    stub_probes(monkeypatch)
    root = tmp_path / "cache" / "openpi"
    if seeded:
        (root / campaign.OPENPI_TOKENIZER).parent.mkdir(parents=True)
        (root / campaign.OPENPI_TOKENIZER).write_bytes(b"spm")
    monkeypatch.setenv("OPENPI_DATA_HOME", str(root))
    monkeypatch.setattr(campaign, "start_server", lambda *a, **k: FakeProc(rc=1))

    def exited(*a, **k):
        raise SystemExit("server on port 34000 exited rc=1 before becoming ready")
    monkeypatch.setattr(campaign, "wait_ready", exited)
    with pytest.raises(SystemExit) as e:
        campaign.main(_driver_argv(tmp_path, "cfg"))
    msg = str(e.value)
    assert msg.startswith("server on port 34000 exited rc=1")
    if seeded:
        assert "tokenizer" not in msg
    else:
        assert f"{root / 'big_vision' / 'paligemma_tokenizer.model'} is not there" in msg
        assert f"OPENPI_DATA_HOME={root}" in msg


def _line_source(rel: str) -> str:
    """A file of the openpi line: the checked-out submodule third_party/openpi, at its pin, else the
    nested openpi repository at the commit a submodule table pins for the line, never at a branch.
    A line found in neither place is a failure, never a skip."""
    repo = Path(__file__).resolve().parents[3]
    sub = repo / "third_party" / "openpi" / rel
    if sub.is_file():
        return sub.read_text()
    pins = repo / "conf" / "release" / "submodules.toml"      # only where a checkout keeps a submodule table
    if not pins.is_file():
        pytest.fail(f"no openpi line to read {rel} from: third_party/openpi is not checked out", pytrace=False)
    commit = tomllib.loads(pins.read_text())["openpi"]["commit"]
    r = subprocess.run(["git", "-C", str(repo / "openpi"), "show", f"{commit}:{rel}"],
                       capture_output=True, text=True)
    if r.returncode:
        pytest.fail(f"no openpi line to read {rel} from: neither third_party/openpi nor the pinned "
                    f"{commit[:12]} in ./openpi ({r.stderr.strip()[:200]})", pytrace=False)
    return r.stdout


def test_the_tokenizer_is_where_the_openpi_line_caches_it():
    """The file the driver names is the one the line downloads: the tokenizer's URL, cached at
    <OPENPI_DATA_HOME>/<host>/<path> by the line's download.maybe_download."""
    from axis_qvla.bench.sim import campaign

    assert f'"{campaign.OPENPI_TOKENIZER_URL}"' in _line_source("src/openpi/models/tokenizer.py")
    download = ast.unparse(ast.parse(_line_source("src/openpi/shared/download.py")))
    assert "local_path = cache_dir / parsed.netloc / parsed.path.strip('/')" in download
    assert "os.getenv(_OPENPI_DATA_HOME, DEFAULT_CACHE_DIR)" in download and "_OPENPI_DATA_HOME = 'OPENPI_DATA_HOME'" in download


# --- --verify-checkpoint ------------------------------------------------------------------------

def _cell_map(tmp_path: Path, monkeypatch, digest: dict | None) -> None:
    """A results map that `bench eval` can read (a results map it cannot read is refused, never
    taken for one without the id), holding one pi0.5 cell renamed to pi05/ft/x with `digest`."""
    from axis_qvla.paper import cli as paper_cli

    doc = json.loads(paper_cli.DEFAULT_CELLS.read_bytes())
    cell = next(c for c in doc["cells"] if str(c["checkpoint"].get("id", "")).startswith("pi05/"))
    cell["checkpoint"].update(id="pi05/ft/x", pub_digest=digest)
    doc["cells"] = [cell]
    p = tmp_path / "cells.json"
    p.write_text(json.dumps(doc))
    monkeypatch.setattr(paper_cli, "DEFAULT_CELLS", p)


def test_verify_checkpoint_passes_the_published_copy_and_refuses_another(train, host, monkeypatch, capsys):
    from axis_qvla.bench import checkpoint_digest

    (host / "ck" / "params").mkdir()
    (host / "ck" / "params" / "w").write_bytes(b"weights")
    _cell_map(host, monkeypatch, checkpoint_digest.pub_digest(host / "ck", "pi05"))
    cfg = host / "r1" / "out_a" / "serve" / "openpi_config.json"
    assert main(_eval(host, "--checkpoint-id", "pi05/ft/x", "--verify-checkpoint", "--dry-run")) == 0
    assert cfg.read_bytes() == STUB
    cfg.unlink()
    capsys.readouterr()
    (host / "ck" / "params" / "w").write_bytes(b"other weights")
    assert main(_eval(host, "--checkpoint-id", "pi05/ft/x", "--verify-checkpoint", "--dry-run")) == 2
    out = capsys.readouterr().out
    assert out.startswith(f"FAIL  --checkpoint {host / 'ck'} is not the published pi05/ft/x") and "argv:" not in out
    assert not cfg.exists()                                   # refused before the config is written
    # Off by default: the same copy is not hashed at all.
    monkeypatch.setattr(checkpoint_digest, "mismatch", lambda *a: pytest.fail("hashed without --verify-checkpoint"))
    assert main(_eval(host, "--checkpoint-id", "pi05/ft/x", "--dry-run")) == 0


def test_verify_checkpoint_needs_an_id_and_a_recorded_digest(train, host, monkeypatch, capsys):
    _cell_map(host, monkeypatch, None)
    assert main(_eval(host, "--checkpoint-id", "pi05/ft/x", "--verify-checkpoint", "--dry-run")) == 2
    assert "--verify-checkpoint: the results map records no pub_digest for checkpoint pi05/ft/x" in capsys.readouterr().out
    assert main(_eval(host, "--config", "cfg.json", "--verify-checkpoint", "--dry-run")) == 2
    out = capsys.readouterr().out
    assert "--verify-checkpoint compares a paper checkpoint with the pub_digest cells.json records for it: give --checkpoint-id" in out


def test_whether_a_run_started_is_read_from_the_out_directory_given(tmp_path):
    """The refusal reads run_manifest.json in the arm's out directory the caller names, not in a
    directory guessed from where the config file sits."""
    from axis_qvla.bench.cli import _write_serve_config

    cfg = tmp_path / "configs" / "openpi_config.json"
    cfg.parent.mkdir()
    cfg.write_bytes(b'{"identity": {"arm": "old"}}')
    out_dir = tmp_path / "elsewhere" / "out_a"
    out_dir.mkdir(parents=True)
    assert "No run under this name started" in _write_serve_config(cfg, STUB, out_dir)
    (out_dir / "run_manifest.json").write_text("{}")
    assert f"A run under this name recorded it ({out_dir / 'run_manifest.json'})" in _write_serve_config(cfg, STUB, out_dir)
