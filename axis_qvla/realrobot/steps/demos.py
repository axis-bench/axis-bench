"""Step 6 of the real-world tutorial: `qvla realrobot demos RECORDINGS`, your DROID recordings as a fine-tuning set.

One command for the step. The converter runs in its own environment
(`droid_convert`, conf/envs/droid_convert.toml). Its output goes under the layout's lerobot/, and its
REPO_NAME is set from outside its file before its main() runs. The training set is written to demos/,
and its norm statistics go where the fine-tuning runs read them. `--converted DIR` starts from a
dataset the converter already wrote; `--print` prints the converter's command.

Offline tier: the training set needs numpy, pyarrow, pandas, PyAV and Pillow. Module scope is stdlib only.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

from axis_qvla.realrobot.layout import CONVERTED, REPO
from axis_qvla.realrobot.steps.common import ensure_norm, fail

CONVERTER = Path("examples") / "droid" / "convert_droid_data_to_lerobot.py"     # under qvla.toml's [roots] openpi
# The converter keeps its output name in a module constant. It is set here, after the module loads and
# before its main() runs, so nobody edits the file.
LAUNCH = ("import importlib.util, sys\n"
          "spec = importlib.util.spec_from_file_location(\"convert_droid_data_to_lerobot\", sys.argv[1])\n"
          "mod = importlib.util.module_from_spec(spec)\n"
          "spec.loader.exec_module(mod)\n"
          "mod.REPO_NAME = sys.argv[2]\n"
          "mod.main(sys.argv[3])\n")


def env_python(tier: str) -> Path:
    """The interpreter of a conf/envs environment that is built inside its checkout (its `uv venv <dir>` line)."""
    from axis_qvla.train.envs import load_env

    spec = load_env(tier)
    venv = next(line.split()[-1] for line in spec.build if line.startswith("uv venv"))
    return REPO / spec.checkout / venv / "bin" / "python"


def convert_command(lay, recordings: Path) -> tuple[list[str], dict[str, str]]:
    openpi = lay.paths.host.roots.get("openpi")
    if openpi is None:
        raise ValueError("qvla.toml declares no [roots] openpi: the openpi checkout whose examples/droid converter runs")
    argv = [str(env_python("droid_convert")), "-c", LAUNCH, str(Path(openpi) / CONVERTER), CONVERTED,
            str(Path(recordings).resolve())]
    return argv, {"HF_LEROBOT_HOME": str(lay.lerobot)}


def demos(args) -> int:
    from axis_qvla.core.hosts import HostProfileError
    from axis_qvla.realrobot.layout import layout
    from axis_qvla.realrobot.realdata.lerobot_v21 import RealdataError
    from axis_qvla.realrobot.realdata.training_set import write_training_set

    try:
        lay = layout(args.host)
        converted = Path(args.converted) if args.converted is not None else lay.converted
        if args.converted is None:
            if args.recordings is None:
                return fail("give your DROID recordings directory, or --converted DIR (a dataset openpi's converter wrote)")
            argv, env = convert_command(lay, args.recordings)
            if args.print:
                print(shlex.join(["env", *(f"{k}={v}" for k, v in env.items()), *argv]))
                return 0
            if not Path(argv[0]).is_file():
                return fail(f"no droid_convert environment ({argv[0]} is missing): build it with "
                            f"`qvla env build droid_convert`")
            print(f"converting {args.recordings} with openpi's DROID converter -> {converted}")
            rc = subprocess.run(argv, env={**os.environ, **env}, cwd=REPO, check=False).returncode
            if rc != 0:
                return fail(f"openpi's DROID converter exited {rc}", rc)
        if lay.demos.exists():
            if not args.replace:
                return fail(f"{lay.demos} exists: pass --replace to write it again from {converted}")
            shutil.rmtree(lay.demos)
            # The old set's statistics would pass the norm check's tolerances on a set that differs by a
            # few episodes; a new training set gets its own every-frame statistics.
            lay.norm_file("finetune").unlink(missing_ok=True)
        got = write_training_set(converted, lay.demos)
        print(f"{got['episodes']} episodes / {got['frames']} frames of {got['tasks']} instructions -> {lay.demos}")
        ensure_norm(lay, "finetune")
    except (RealdataError, ValueError, OSError, HostProfileError) as exc:
        return fail(str(exc))
    return 0
