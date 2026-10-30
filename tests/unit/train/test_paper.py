"""axis_qvla.train.paper: finding a paper arm by checkpoint or by the config it evaluates under."""
from __future__ import annotations

import os
import pathlib
import stat

import pytest

from axis_qvla.train.paper import arms_for_paper_config, find_arm_by_checkpoint, paper_arm_files

H = "c" * 64


def _arm(path: pathlib.Path, *, id: str, checkpoint: str, paper_ref: str | None = "cfg_a",
         stage: str = "finetune", backbone: str = "pi05-openpi") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ref_line = f'paper_ref = "{paper_ref}"\n' if paper_ref is not None else ""
    path.write_text(f'''\
[arm]
id = "{id}"
backbone = "{backbone}"
stage = "{stage}"
track = "paper"
{ref_line}
[init]
ckpt = "base:pi05_droid"

[data]
rowspace = "heldout24_ft#sha256:{H}"

[signal]
bundle = "sha256:{H}"

[norm]
artifact = "heldout24_qual_v2#sha256:{H}"

[recipe]
preset = "pi05.finetune.lora.paper"
seed = 42

[paper]
checkpoint = "{checkpoint}"
cells = []
signals = ["ft:plain"]
''')


def test_paper_arm_files_finds_every_toml_recursively(tmp_path):
    _arm(tmp_path / "pi05" / "finetune" / "a.toml", id="pi05.finetune.a", checkpoint="pi05/ft/a")
    _arm(tmp_path / "pi05" / "pretrain" / "b.toml", id="pi05.pretrain.b", checkpoint="pi05/pretrain/b",
         stage="pretrain", paper_ref="cfg_b")
    (tmp_path / "index.json").write_text("{}")   # a non-.toml file must not be picked up
    assert [p.name for p in paper_arm_files(tmp_path)] == ["a.toml", "b.toml"]


def test_find_arm_by_checkpoint_with_zero_one_and_two_matches(tmp_path):
    _arm(tmp_path / "a.toml", id="pi05.finetune.a", checkpoint="pi05/ft/a")
    with pytest.raises(LookupError, match="0 paper runs"):
        find_arm_by_checkpoint("pi05/ft/does-not-exist", tmp_path)
    assert find_arm_by_checkpoint("pi05/ft/a", tmp_path) == tmp_path / "a.toml"
    _arm(tmp_path / "b.toml", id="pi05.finetune.b", checkpoint="pi05/ft/a")   # a second arm names it too
    with pytest.raises(LookupError, match="2 paper runs"):
        find_arm_by_checkpoint("pi05/ft/a", tmp_path)


def test_an_unreadable_arms_dir_raises_rather_than_reading_as_empty(tmp_path):
    """A filesystem probe must not swallow EACCES: an unreadable directory under `arms_dir` must
    fail loud, not silently read as "no arms"."""
    if os.geteuid() == 0:
        pytest.skip("root bypasses directory permissions")
    locked = tmp_path / "locked"
    _arm(locked / "a.toml", id="pi05.finetune.a", checkpoint="pi05/ft/a")
    locked.chmod(0o000)
    try:
        with pytest.raises(OSError):
            paper_arm_files(tmp_path)
        with pytest.raises(OSError):
            find_arm_by_checkpoint("pi05/ft/a", tmp_path)
    finally:
        locked.chmod(stat.S_IRWXU)


def test_arms_for_paper_config_matches_and_raises_for_a_typo(tmp_path):
    """B's serving lookup: a typo'd config name must raise, not read as "no arms serve this cell"."""
    _arm(tmp_path / "a.toml", id="pi05.finetune.a", checkpoint="pi05/ft/a", paper_ref="cfg_a")
    _arm(tmp_path / "b.toml", id="pi05.finetune.b", checkpoint="pi05/ft/b", paper_ref="cfg_a")
    _arm(tmp_path / "c.toml", id="pi05.pretrain.c", checkpoint="pi05/pretrain/c", paper_ref="cfg_a",
         stage="pretrain")   # not a finetune: must not match even though paper_ref agrees
    assert sorted(p.name for p in arms_for_paper_config("cfg_a", tmp_path)) == ["a.toml", "b.toml"]
    with pytest.raises(LookupError, match="cfg_typo"):
        arms_for_paper_config("cfg_typo", tmp_path)


def test_arms_for_paper_config_never_returns_a_lerobot_arm(tmp_path):
    """`config_name` names an openpi config: only a pi0.5 arm can carry it, checked by backbone
    explicitly rather than relying on a LeRobot arm's paper_ref happening to be unset."""
    _arm(tmp_path / "pi.toml", id="pi05.finetune.a", checkpoint="pi05/ft/a", paper_ref="cfg_a")
    # A LeRobot arm's paper_ref is None by construction (the writer never sets it), but even one
    # that somehow carried the same string as a pi0.5 config must not be returned.
    _arm(tmp_path / "lr.toml", id="smolvla.finetune.a", checkpoint="smolvla/ft/a", paper_ref="cfg_a",
         backbone="smolvla-lerobot")
    assert [p.name for p in arms_for_paper_config("cfg_a", tmp_path)] == ["pi.toml"]


def test_the_paper_run_specs_live_under_experiments_paper_runs():
    from axis_qvla.train.paper import RUNS_DIR

    repo = RUNS_DIR.parents[2]
    assert RUNS_DIR.relative_to(repo).as_posix() == "experiments/paper/runs"
    assert (RUNS_DIR / "index.json").is_file()
    assert not (repo / "experiments" / "paper" / "arms").exists()
    assert not (repo / "experiments" / "examples" / "arms").exists()
    assert (repo / "experiments" / "examples" / "runs" / "pi05.pretrain.keep_top.toml").is_file()
