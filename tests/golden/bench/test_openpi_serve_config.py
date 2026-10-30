"""Every pi0.5 cell's checkpoint names exactly one paper arm, and that arm renders a serving config:
the seam between `bench eval --checkpoint-id` and the training package's paper arms.

The rendered config is compared with the run's row of conf/release/paper_configs.lock.json in the
lock's dotted openpi names -- the fields a TrainConfig built from it carries (seed, batch, steps and
schedule, every model field, the data fields the server's transforms read) -- so the config a
published checkpoint is served with is the configuration its run trained under. And every pi0.5
cell's own eval_command, run as a dry run, serves the bytes its checkpoint id renders, from a file
that every command it prints names and that is on disk. CPU only and model-free: nothing is
served, nothing is downloaded."""
from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path

import pytest

from axis_qvla.bench import openpi_serve
from axis_qvla.cli import main
from axis_qvla.core.hosts import load_host
from axis_qvla.core.paths import Paths
from axis_qvla.train.parity import DROPPED_INERT, load_lock, normalized

REPO = Path(__file__).resolve().parents[3]
CELLS = json.loads((REPO / "experiments" / "paper" / "cells.json").read_bytes())["cells"]
PI05 = sorted({c["checkpoint"]["id"] for c in CELLS if c["checkpoint"]["id"].startswith("pi05/")})
PI05_CELLS = [c for c in CELLS if c["backbone"] == "pi05"]
LOCK = load_lock()["pi05"]
GROUPS = ("model", "lr_schedule", "optimizer")
PROFILE = """launcher = "local"
gpus = 0
requires_secrets = []

[roots]
corpus = "{d}/corpus"
cache = "{d}/cache"
scratch = "{d}/scratch"
runs = "{d}/runs"
"""


def test_the_paper_evaluates_32_pi05_checkpoints():
    assert len(PI05) == 32


def test_the_paper_has_40_pi05_cells():
    """The eval_command golden is parametrized over these, and an empty parameter set is one skip,
    not a failure: the count is pinned so a dropped or relabelled cell fails here."""
    assert len(PI05_CELLS) == 40 and len({c["run"] for c in PI05_CELLS}) == 40


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    d = tmp_path_factory.mktemp("host")
    (d / "demo.toml").write_text(PROFILE.format(d=d))
    return Paths(load_host("demo", hosts_dir=d))


def _projected(cfg: dict) -> dict[str, object]:
    """The config in the lock's dotted names. The init is the content id the config records: the
    lock names it that way, and `weight_loader.params_path` is this machine's copy of it."""
    out: dict[str, object] = {k: cfg[k] for k in ("seed", "batch_size", "num_train_steps", "ema_decay")}
    out["weight_loader.params_path"] = cfg["identity"]["init"]
    for group in GROUPS:
        out.update({f"{group}.{k}": v for k, v in cfg[group].items()})
    out.update({f"data.{k}": cfg["data"][k] for k in ("repo_id", "image_center_crop", "default_prompt")})
    return out


@pytest.mark.parametrize("cid", PI05)
def test_the_arm_of_every_pi05_checkpoint_renders_for_serving(cid, paths):
    from axis_qvla.train.armspec import load_armspec
    from axis_qvla.train.paper import find_arm_by_checkpoint

    spec = load_armspec(find_arm_by_checkpoint(cid))
    cfg = json.loads(openpi_serve.render_serve_config(cid, paths))
    assert spec.stage == "finetune" and spec.backbone == openpi_serve.OPENPI_BACKBONE
    assert cfg["schema"] == "qvla.openpi_train_config/1" and cfg["identity"]["arm"] == spec.id
    assert cfg["exp_name"] == "serve" and cfg["data"]["bundle"] is None and cfg["data"]["bundle_id"] is None

    row = LOCK[spec.id]
    assert row["checkpoint"] == cid
    fields, got = row["fields"], _projected(cfg)
    assert set(got) <= set(fields), sorted(set(got) - set(fields))
    # The fork-only fields the openpi line does not have -- knowledge insulation and the EEF /
    # joint-9 action-space switches -- are locked at the inert value the line projects them as, and
    # every other locked model/schedule/optimizer field is in the config.
    inert = set(DROPPED_INERT)
    assert inert <= set(fields), sorted(inert - set(fields))
    assert {k: normalized(fields[k]) for k in inert} == {k: normalized(DROPPED_INERT[k]) for k in inert}
    unrendered = {k for k in fields if k.split(".")[0] in GROUPS} - set(got)
    assert unrendered <= inert, sorted(unrendered - inert)
    assert {k: v for k, v in got.items() if normalized(v) != normalized(fields[k])} == {}
    # The server normalises with <checkpoint>/assets/<asset id>/norm_stats.json: the run's.
    assert cfg["data"]["asset_id"] == fields["data.repo_id"]
    assert cfg["identity"]["norm"].rpartition("#sha256:")[2] == row["norm"]


# --- a cell's eval_command, as published --------------------------------------------------------

BENCH_PROFILE = PROFILE + """r1 = "{d}/r1"
bench = "{d}/bench"
openpi = "{d}/openpi"
eval = "{d}/eval"

[venv]
render = "{d}/venvs/render"
online = "{d}/venvs/online"
"""


@pytest.fixture(scope="module")
def bench_host(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("bench_host")
    (d / "demo.toml").write_text(BENCH_PROFILE.format(d=d))
    (d / "ck").mkdir()
    return d


def _filled(command: str, d: Path) -> list[str]:
    """The cell's eval_command with its placeholders filled for a scratch machine."""
    argv = shlex.split(command)
    assert argv[:3] == ["qvla", "bench", "eval"], argv[:3]
    fill = {"--host": str(d / "demo.toml"), "--checkpoint": str(d / "ck"), "--asset-dir": str(d / "assets"),
            "--task-data-root": str(d / "tasks")}
    return [fill.get(argv[i - 1], a) if a.startswith("<") else a for i, a in enumerate(argv)][1:]


@pytest.mark.parametrize("cell", PI05_CELLS, ids=lambda c: c["cell"])
def test_every_pi05_eval_command_serves_its_checkpoints_rendering(cell, bench_host, capsys, monkeypatch):
    """The eval_command names an openpi config, not the checkpoint id; dry-run as published, it
    serves exactly what `--checkpoint-id <the cell's checkpoint.id>` renders, from a file each
    printed command names and that is on disk."""
    monkeypatch.setattr(os, "environ", dict(os.environ))  # restored after the test
    argv = _filled(cell["eval_command"], bench_host)
    assert "--config" in argv and "--checkpoint-id" not in argv
    assert main([*argv, "--dry-run"]) == 0
    out = capsys.readouterr().out
    named = {Path(p) for p in re.findall(r"--config=(\S+)", out)} | \
        {Path(p) for p in re.findall(r"^serve config: (\S+)", out, re.M)}
    want = openpi_serve.render_serve_config(cell["checkpoint"]["id"], Paths(load_host("demo", hosts_dir=bench_host)))
    assert len(named) == 1 and next(iter(named)).read_bytes() == want
    assert f"rendered for {cell['checkpoint']['id']}" in out


def test_a_config_name_resolves_to_the_checkpoint_of_the_cell_its_command_belongs_to():
    """Seed replicates share a config name; the cell's run (its eval_command's --run) tells them
    apart. A unique name needs no arm, and a name no paper run carries is not a paper config."""
    bc = "pi05_axis_heldout_qual_v2_bc"
    by_run = {c["run"]: c["checkpoint"]["id"] for c in PI05_CELLS if f"--config {bc} " in c["eval_command"]}
    assert len(set(by_run.values())) == 2
    for run, cid in by_run.items():
        assert openpi_serve.checkpoint_for_config(bc, run) == cid
    with pytest.raises(openpi_serve.ServeConfigError, match="paper config of 2 checkpoints"):
        openpi_serve.checkpoint_for_config(bc, "my_rerun")
    assert openpi_serve.checkpoint_for_config("pi05_axis_heldout_qual_v2_droidctl", None) == \
        "pi05/ft/pi05_axis_heldout_qual_v2_droidctl__droidctl_ft"
    assert openpi_serve.checkpoint_for_config("pi05_libero", None) is None
