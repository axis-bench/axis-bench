"""Step 6: openpi's DROID converter run in its own environment, then the training set and its norm statistics."""
from __future__ import annotations

import json
import subprocess
import sys

from axis_qvla.cli import main
from axis_qvla.realrobot.layout import CONVERTED, REPO, layout
from axis_qvla.realrobot.steps import demos as step
from tests.unit.realrobot.realdata.test_training_set import write_droid_recording
from tests.unit.realrobot.steps.hosts import qvla_toml


def test_a_converted_dataset_becomes_the_training_set_and_its_norm(tmp_path, capsys):
    toml = str(qvla_toml(tmp_path))
    lay = layout(toml)
    rec = write_droid_recording(tmp_path / "rec")
    assert main(["realrobot", "demos", "--converted", str(rec), "--host", toml]) == 0
    assert (lay.demos / "roots.json").is_file() and lay.norm_file("finetune").is_file()
    assert "3 episodes / 62 frames of 2 instructions ->" in capsys.readouterr().out
    assert main(["realrobot", "demos", "--converted", str(rec), "--host", toml]) == 2      # exists
    assert "--replace" in capsys.readouterr().err
    assert main(["realrobot", "demos", "--converted", str(rec), "--replace", "--host", toml]) == 0


def test_print_shows_the_converter_in_its_environment_with_its_output_in_the_layout(tmp_path, capsys):
    toml = str(qvla_toml(tmp_path))
    lay = layout(toml)
    assert main(["realrobot", "demos", str(tmp_path / "recordings"), "--host", toml, "--print"]) == 0
    out = capsys.readouterr().out
    assert f"HF_LEROBOT_HOME={lay.lerobot}" in out and CONVERTED in out
    assert str(REPO / ".venv-droid-convert" / "bin" / "python") in out
    assert str(tmp_path / "openpi" / "examples" / "droid" / "convert_droid_data_to_lerobot.py") in out


def test_an_unbuilt_converter_environment_names_its_build(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(step, "env_python", lambda tier: tmp_path / "nope" / "bin" / "python")
    assert main(["realrobot", "demos", str(tmp_path), "--host", str(qvla_toml(tmp_path))]) == 2
    assert "qvla env build droid_convert" in capsys.readouterr().err


def test_the_converter_environment_is_the_one_conf_envs_builds():
    assert step.env_python("droid_convert") == REPO / ".venv-droid-convert" / "bin" / "python"


def test_the_launcher_sets_repo_name_before_main_runs(tmp_path):
    conv = tmp_path / "convert.py"
    conv.write_text("import json, os\nREPO_NAME = 'your_hf_username/my_droid_dataset'\n"
                    "def main(data_dir, *, push_to_hub=False):\n"
                    "    print(json.dumps([REPO_NAME, data_dir, os.environ['HF_LEROBOT_HOME']]))\n")
    got = subprocess.run([sys.executable, "-c", step.LAUNCH, str(conv), CONVERTED, str(tmp_path / "rec")],
                         env={"HF_LEROBOT_HOME": str(tmp_path / "lr")}, capture_output=True, text=True, check=True)
    assert json.loads(got.stdout) == [CONVERTED, str(tmp_path / "rec"), str(tmp_path / "lr")]
