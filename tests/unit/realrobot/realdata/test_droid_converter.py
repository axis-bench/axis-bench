"""openpi's DROID converter, in the environment `qvla env build droid_convert` builds for it, on a raw DROID
recording, and `realdata training-set` on what it writes: the tutorial's step 6 end to end.

Slow: it builds that environment with uv from the network, from the `droid_convert` environment's build lines
(conf/envs/droid_convert.toml, the ones `qvla env build droid_convert` runs; held equal here), then converts
two episodes of 1280 x 720 video.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from axis_qvla.realrobot.realdata.training_set import write_training_set
from axis_qvla.train import envs
from tests.docs.blocks import blocks

REPO = envs.REPO
PAGE = REPO / "docs" / "tutorials" / "real-world-your-workspace.md"
CONVERTER_REL = "examples/droid/convert_droid_data_to_lerobot.py"
CONVERTER = REPO / envs.load_env("openpi").checkout / CONVERTER_REL
VENV = ".venv-droid-convert"            # where the droid_convert environment's build puts it
OUTPUT = "your_hf_username/my_droid_dataset"     # the converter's REPO_NAME, under HF_LEROBOT_HOME


def _environment_commands() -> list[str]:
    """The `droid_convert` environment's build lines, the ones `qvla env build droid_convert` runs."""
    return list(envs.load_env("droid_convert").build)


def test_the_tutorial_converts_with_the_command_that_builds_the_droid_convert_environment(capsys):
    """Step 6 of the tutorial is one `qvla realrobot demos` command; a missing converter environment is named
    by `qvla env build droid_convert` (axis_qvla.realrobot.steps.demos), which prints exactly that environment's
    build lines."""
    from axis_qvla.cli import main

    text = PAGE.read_text(encoding="utf-8")
    step = text[text.index("## 6. "):text.index("## 7. ")]
    assert [b.text for b in blocks(step) if b.tag == "gpu"] == ["qvla realrobot demos <your DROID recordings>\n"]
    assert main(["env", "build", "droid_convert", "--print"]) == 0
    printed = capsys.readouterr().out
    assert all(f"sh -c {shlex.quote(line)}" in printed for line in _environment_commands())


@pytest.mark.slow
def test_the_droid_converter_runs_in_its_environment_and_training_set_reads_what_it_writes(tmp_path):
    if shutil.which("uv") is None:
        pytest.skip("uv is not on PATH: the tutorial builds the converter's environment with it")
    if not CONVERTER.is_file():
        pytest.skip(f"{CONVERTER} is not checked out (the openpi submodule)")
    python = str(tmp_path / VENV / "bin" / "python")
    for line in _environment_commands():           # run in the checkout, the venv placed in tmp_path
        cmd = [w.replace(VENV, str(tmp_path / VENV), 1) if w.startswith(VENV) else w for w in shlex.split(line)]
        subprocess.run(cmd, cwd=REPO, check=True)
    assert envs.main(["check", "droid_convert", "--python", python]) == 0

    env = {**os.environ, "PYTHONPATH": str(REPO), "HF_LEROBOT_HOME": str(tmp_path / "hf")}
    subprocess.run([python, "-m", "tests.unit.realrobot.realdata.droid_raw", str(tmp_path / "raw")],
                   cwd=tmp_path, env=env, check=True)
    subprocess.run([python, str(CONVERTER), "--data_dir", str(tmp_path / "raw")], cwd=tmp_path, env=env, check=True)

    from tests.unit.realrobot.realdata.droid_raw import INSTRUCTIONS, episode_arrays

    got = write_training_set(tmp_path / "hf" / OUTPUT, tmp_path / "demos")
    assert got == {"tasks": 2, "episodes": 2, "frames": 36}
    # Tasks in sorted instruction order: task_1 "open the drawer" (episode 1, 16 steps), task_2 "put the cup
    # on the plate" (episode 0, 20 steps). State and action are the recorded ones, as float32.
    for task, episode, n in (("task_1", 1, 16), ("task_2", 0, 20)):
        prompts = pq.read_table(tmp_path / "demos" / task / "meta" / "tasks.parquet").to_pandas()
        assert list(prompts.index) == [INSTRUCTIONS[episode]]
        rows = pq.read_table(tmp_path / "demos" / task / "data" / "chunk-000" / "file-000.parquet").to_pandas()
        a = episode_arrays(episode, n)
        state = np.concatenate([a["joint_positions"], a["gripper_position"][:, None]], axis=1).astype(np.float32)
        action = np.concatenate([a["joint_velocity"], a["action_gripper"][:, None]], axis=1).astype(np.float32)
        np.testing.assert_array_equal(np.stack(rows["observation.state"]), state)
        np.testing.assert_array_equal(np.stack(rows["action"]), action)
