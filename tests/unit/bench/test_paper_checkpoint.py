"""`bench eval --checkpoint-id` for every backbone: the results map names the backend, the server's
config and the local copy `qvla data get checkpoints/<id>` made, so a paper checkpoint is named one
way only; and the reference servers read the base-weights copy that `qvla data get
base-weights` fills (OPENPI_DATA_HOME, HF_HOME)."""
from __future__ import annotations

import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

from axis_qvla.bench import launch, openpi_serve, paper_checkpoint as pc
from axis_qvla.bench.spec import load_benchmark
from axis_qvla.cli import main
from axis_qvla.core.hosts import load_host
from tests.unit.bench.test_openpi_serve import PROFILE, host, train  # noqa: F401  (fixtures)

REPO = Path(__file__).resolve().parents[3]
CELLS = json.loads((REPO / "experiments" / "paper" / "cells.json").read_bytes())["cells"]
LEROBOT_ROOTS = "".join(f'{k} = "{{d}}/{k}"\n' for k in ("smolvla_venv", "smolvla_bench"))


def _cell(name: str) -> dict:
    return next(c for c in CELLS if c["cell"] == name)


def _copy(root: Path, ref: str) -> Path:
    repo, _, rest = ref.partition("@")
    return root / "cache" / "checkpoints" / repo.split("/")[1] / rest.partition(":")[2]


@pytest.mark.parametrize("name, backend", [("droidinit|bare", "openpi"), ("svla_bc|bare", "smolvla"),
                                           ("xvla_base|bare", "xvla")])
def test_every_backbone_resolves_to_its_backend_config_and_local_copy(tmp_path, name, backend):
    c = _cell(name)
    p = pc.lookup(c["checkpoint"]["id"])
    assert p.backend == backend == pc.backend_of(c["checkpoint"]["id"])
    toks = shlex.split(c["eval_command"])
    assert p.config == (None if backend == "openpi" else toks[toks.index("--config") + 1])
    paths = SimpleNamespace(corpus=tmp_path / "corpus", cache=tmp_path / "cache")
    want = _copy(tmp_path, c["checkpoint"]["ref"])
    assert pc.local_dir(p, paths) == (want if backend == "openpi" else want / "pretrained_model")


def test_an_id_outside_the_results_map_or_of_no_backbone_is_refused():
    with pytest.raises(pc.PaperCheckpointError, match="needs --checkpoint"):
        pc.lookup("pi05/ft/not_published")
    with pytest.raises(pc.PaperCheckpointError, match="begins with"):
        pc.backend_of("octo/ft/x")


def _lerobot_host(d: Path) -> None:
    (d / "hosts" / "demo.toml").write_text(PROFILE.format(d=d).replace("[venv]", LEROBOT_ROOTS.format(d=d) + "\n[venv]"))


def test_a_lerobot_checkpoint_id_is_served_by_its_backend_from_the_data_get_copy(host, capsys):
    _lerobot_host(host)
    c = _cell("svla_bc|bare")
    assert main(["bench", "eval", "--host", "demo", "--checkpoint-id", c["checkpoint"]["id"], "--run", "s",
                 "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "--backend=smolvla" in out and "--config=axis_smolvla" in out and "serve config:" not in out
    assert f"--checkpoint={_copy(host, c['checkpoint']['ref']) / 'pretrained_model'}" in out
    assert f"env  HF_HOME={host / 'cache' / 'checkpoints' / 'qvla-v1-base-weights' / 'hf'}" in out


def test_a_pi05_checkpoint_id_defaults_to_its_data_get_copy(host, monkeypatch, capsys):
    monkeypatch.setattr(openpi_serve, "render_serve_config", lambda cid, paths: b'{"serve": 1}')
    c = _cell("droidinit|bare")
    assert main(["bench", "eval", "--host", "demo", "--checkpoint-id", c["checkpoint"]["id"], "--run", "a",
                 "--dry-run"]) == 0
    assert f"--checkpoint={_copy(host, c['checkpoint']['ref'])}" in capsys.readouterr().out


def test_a_defaulted_copy_that_is_not_there_names_its_data_get(host, monkeypatch, capsys):
    from tests.unit.bench.eval_stubs import fake_driver
    from tests.unit.bench.test_eval_roots import make_bank

    seen = fake_driver(monkeypatch, host)
    monkeypatch.setattr(openpi_serve, "render_serve_config", lambda cid, paths: b'{"serve": 1}')
    make_bank(host / "render", host / "lock.json", monkeypatch)
    for d in ("assets", "tasks"):
        (host / d).mkdir(exist_ok=True)
    cid = _cell("droidinit|bare")["checkpoint"]["id"]
    assert main(["bench", "eval", "--host", "demo", "--checkpoint-id", cid, "--run", "a", "--asset-dir",
                 str(host / "assets"), "--task-data-root", str(host / "tasks"), "--render-root", str(host / "render")]) == 2
    assert f"qvla data get checkpoints/{cid}" in capsys.readouterr().out and "argv" not in seen


def test_a_checkpoint_id_with_another_backend_is_refused(capsys):
    assert main(["bench", "eval", "--checkpoint-id", "smolvla/ft/y", "--backend", "openpi", "--run", "a"]) == 2
    assert "--backend smolvla" in capsys.readouterr().out


def _demo(tmp_path: Path, extra: str = ""):
    (tmp_path / "hosts").mkdir(exist_ok=True)
    (tmp_path / "hosts" / "demo.toml").write_text(PROFILE.format(d=tmp_path).replace("[venv]", extra + "\n[venv]"))
    return load_host("demo", hosts_dir=tmp_path / "hosts")


def test_the_pi05_server_reads_the_tokenizer_from_the_base_weights_copy_once_it_is_there(tmp_path):
    from axis_qvla.bench.sim.campaign import OPENPI_TOKENIZER

    h, b = _demo(tmp_path), load_benchmark()
    req = launch.EvalRequest(checkpoint="x", config="c")
    assert launch.compose(req, h, b).env["OPENPI_DATA_HOME"] == str(tmp_path / "cache" / "openpi")
    tok = tmp_path / "cache" / "checkpoints" / "qvla-v1-base-weights" / "openpi" / OPENPI_TOKENIZER
    tok.parent.mkdir(parents=True)
    tok.write_bytes(b"t")
    assert launch.compose(req, h, b).env["OPENPI_DATA_HOME"] == str(tmp_path / "cache" / "checkpoints" /
                                                                    "qvla-v1-base-weights" / "openpi")


def test_a_host_without_hf_cache_serves_lerobot_from_the_base_weights_copy(tmp_path):
    h = _demo(tmp_path, LEROBOT_ROOTS.format(d=tmp_path))
    env = launch.compose(launch.EvalRequest(checkpoint="x", config="c", backend="xvla"), h, load_benchmark()).env
    assert env["HF_HOME"] == str(tmp_path / "cache" / "checkpoints" / "qvla-v1-base-weights" / "hf")


def test_print_names_the_fetch_the_server_and_the_results_file_with_no_data(host, capsys):
    _lerobot_host(host)
    c = _cell("svla_bc|bare")
    cid = c["checkpoint"]["id"]
    assert main(["bench", "eval", "--host", "demo", "--checkpoint-id", cid, "--run", "s", "--print"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert f"# fetch first: qvla data get checkpoints/{cid}" in lines
    assert any(ln.startswith("# serves: --backend smolvla ") and "pretrained_model" in ln for ln in lines)
    assert f"# results file: {host / 'r1' / 'out_s' / 'results.json'}" in lines
    assert " bench eval --checkpoint-id " in lines[-1] and "--print" not in lines[-1]


# --- the defaulted copy is complete, and --verify-checkpoint for every backbone -------------------

LEROBOT_STEP = {"pretrained_model/config.json": b"{}", "pretrained_model/model.safetensors": b"weights",
                "training_state/optimizer_state.safetensors": b"not published"}
PI05_STEP = {"_CHECKPOINT_METADATA": b"m", "params/w": b"weights", "assets/norm_stats.json": b"{}"}


def _fill(root: Path, files: dict) -> Path:
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    return root


def _record(d: Path, monkeypatch, cid: str, step: Path, fam: str) -> None:
    """The committed results map with `cid`'s pub_digest replaced by the digest of `step`."""
    from axis_qvla.bench import checkpoint_digest
    from axis_qvla.paper import cli as paper_cli

    doc = json.loads((REPO / "experiments" / "paper" / "cells.json").read_bytes())
    for c in doc["cells"]:
        if c["checkpoint"].get("id") == cid:
            c["checkpoint"]["pub_digest"] = checkpoint_digest.pub_digest(step, fam)
    (d / "cells.json").write_text(json.dumps(doc))
    monkeypatch.setattr(paper_cli, "DEFAULT_CELLS", d / "cells.json")


def test_verify_checkpoint_compares_a_lerobot_step_directory(host, monkeypatch, capsys):
    _lerobot_host(host)
    c = _cell("svla_bc|bare")
    cid = c["checkpoint"]["id"]
    step = _fill(host / "step", LEROBOT_STEP)
    _record(host, monkeypatch, cid, step, "smolvla")
    named = ["bench", "eval", "--host", "demo", "--checkpoint-id", cid, "--run", "s", "--verify-checkpoint", "--dry-run"]
    assert main([*named, "--checkpoint", str(step / "pretrained_model")]) == 0
    capsys.readouterr()
    (step / "pretrained_model" / "model.safetensors").write_bytes(b"other weights")
    assert main([*named, "--checkpoint", str(step / "pretrained_model")]) == 2
    out = capsys.readouterr().out
    assert out.startswith(f"FAIL  --checkpoint {step} is not the published {cid}") and "argv:" not in out
    # A --checkpoint that is not a pretrained_model/ names no step directory to compare.
    assert main([*named, "--checkpoint", str(step)]) == 2
    out = capsys.readouterr().out
    assert "is not a pretrained_model/ directory" in out and "give --checkpoint-id" not in out
    # The defaulted copy is compared at its step directory, and a mismatch names its data get.
    copy = _fill(_copy(host, c["checkpoint"]["ref"]), LEROBOT_STEP)
    assert main(named) == 0
    capsys.readouterr()
    (copy / "pretrained_model" / "model.safetensors").write_bytes(b"wEIGHTS")     # same size, other bytes
    assert main(named) == 2
    out = capsys.readouterr().out
    assert f"is not the published {cid}" in out and f"qvla data get checkpoints/{cid}" in out
    # A defaulted copy is named as the local copy (no --checkpoint was given), with no other-run advice.
    assert out.startswith(f"FAIL  the local copy {copy} is not the published {cid}") and "--config" not in out


def test_verify_checkpoint_on_an_absent_defaulted_copy_names_its_data_get(host, monkeypatch, capsys):
    monkeypatch.setattr(openpi_serve, "render_serve_config", lambda cid, paths: b'{"serve": 1}')
    cid = _cell("droidinit|bare")["checkpoint"]["id"]
    assert main(["bench", "eval", "--host", "demo", "--checkpoint-id", cid, "--run", "a", "--verify-checkpoint",
                 "--dry-run"]) == 2
    out = capsys.readouterr().out
    assert "is not there" in out and f"qvla data get checkpoints/{cid}" in out


def test_a_partial_defaulted_copy_is_refused_naming_its_data_get(host, monkeypatch, capsys):
    from tests.unit.bench.eval_stubs import fake_driver
    from tests.unit.bench.test_eval_roots import make_bank

    seen = fake_driver(monkeypatch, host)
    monkeypatch.setattr(openpi_serve, "render_serve_config", lambda cid, paths: b'{"serve": 1}')
    make_bank(host / "render", host / "lock.json", monkeypatch)
    for d in ("assets", "tasks"):
        (host / d).mkdir(exist_ok=True)
    c = _cell("droidinit|bare")
    cid = c["checkpoint"]["id"]
    copy = _fill(_copy(host, c["checkpoint"]["ref"]), PI05_STEP)
    _record(host, monkeypatch, cid, copy, "pi05")
    (copy / "params" / "w").unlink()                     # an interrupted fetch
    argv = ["bench", "eval", "--host", "demo", "--checkpoint-id", cid, "--run", "a", "--asset-dir",
            str(host / "assets"), "--task-data-root", str(host / "tasks"), "--render-root", str(host / "render")]
    assert main(argv) == 2
    out = capsys.readouterr().out
    assert f"the local copy {copy} is incomplete: its published files are 2 files" in out
    assert f"qvla data get checkpoints/{cid}" in out and "argv" not in seen
    (copy / "params" / "w").write_bytes(b"weigh")        # all files there, one short
    assert main(argv) == 2
    assert "is incomplete" in capsys.readouterr().out and "argv" not in seen


def test_a_complete_defaulted_copy_passes_and_an_unreadable_one_is_reported(host, monkeypatch):
    import os

    from axis_qvla.bench import cli

    _lerobot_host(host)
    c = _cell("svla_bc|bare")
    cid = c["checkpoint"]["id"]
    copy = _fill(_copy(host, c["checkpoint"]["ref"]), LEROBOT_STEP)
    _record(host, monkeypatch, cid, copy, "smolvla")
    target = cli._Target(str(copy / "pretrained_model"), "axis_smolvla", "smolvla", "s", cid,
                         fetch=f"checkpoints/{cid}")
    assert cli._local_copy_refusal("qvla", target) is None
    (copy / "training_state" / "more").write_bytes(b"x")   # outside the published files: still complete
    assert cli._local_copy_refusal("qvla", target) is None
    if os.geteuid() == 0:
        pytest.skip("root reads a mode-000 directory")
    (copy / "pretrained_model").chmod(0)
    try:
        refused = cli._local_copy_refusal("qvla", target)
    finally:
        (copy / "pretrained_model").chmod(0o755)
    assert refused and "cannot be checked complete" in refused and "PermissionError" in refused


def test_a_lerobot_checkpoint_id_with_a_checkpoint_keeps_the_results_map_config(host, capsys):
    _lerobot_host(host)
    c = _cell("xvla_base|bare")
    assert main(["bench", "eval", "--host", "demo", "--checkpoint-id", c["checkpoint"]["id"], "--checkpoint",
                 str(host / "ck"), "--run", "x", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "--config=axis_xvla" in out and f"--checkpoint={host / 'ck'}" in out
