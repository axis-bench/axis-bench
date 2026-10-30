"""The paper's training runs as run specs: experiments/paper/runs/<family>/<stage>/<name>.toml.

Each file is an ArmSpec on track "paper" whose id is
`<family>.<pretrain|finetune>.<checkpoint name>`; experiments/paper/runs/index.json lists each one's
file, checkpoint, cells, seeds and init, and the replicate pairing. SIGNALS.lock.json and the signal
compiler's bundle manifest name the same runs `<pi05|svla|xvla>.<pt|ft>.<run name>` (`signals_key`).

Core tier: stdlib only.
"""
from __future__ import annotations

import os
import pathlib

from axis_qvla.train.armspec import load_armspec

RUNS_DIR = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "paper" / "runs"
_SIGNALS_FAMILY = {"pi05": "pi05", "smolvla": "svla", "xvla": "xvla"}
_SIGNALS_STAGE = {"pretrain": "pt", "finetune": "ft"}


def signals_key(arm_id: str) -> str:
    """The name SIGNALS.lock.json (`trained[].arm`) and the bundle manifest give this arm's run:
    `pi05.pretrain.<config>__<run>` -> `pi05.pt.<run>`, `smolvla.finetune.<run>` -> `svla.ft.<run>`."""
    family, stage, name = arm_id.split(".", 2)
    return f"{_SIGNALS_FAMILY[family]}.{_SIGNALS_STAGE[stage]}.{name.rsplit('__', 1)[-1]}"


def paper_arm_files(arms_dir: pathlib.Path = RUNS_DIR) -> list[pathlib.Path]:
    """Every ArmSpec under `arms_dir`. os.walk with onerror: an unreadable directory raises instead
    of reading as an empty one."""
    def fail(exc: OSError) -> None:
        raise exc

    out = []
    for root, _dirs, files in os.walk(arms_dir, onerror=fail):
        out += [pathlib.Path(root) / f for f in files if f.endswith(".toml")]
    return sorted(out)


def find_arm_by_checkpoint(checkpoint_id: str, arms_dir: pathlib.Path = RUNS_DIR) -> pathlib.Path:
    """The arm that trained `checkpoint_id` (a cells.json `checkpoint.id`)."""
    hits = [p for p in paper_arm_files(arms_dir) if load_armspec(p).paper.get("checkpoint") == checkpoint_id]
    if len(hits) != 1:
        raise LookupError(f"{len(hits)} paper runs name checkpoint {checkpoint_id!r}")
    return hits[0]


def arms_for_paper_config(config_name: str, arms_dir: pathlib.Path = RUNS_DIR) -> list[pathlib.Path]:
    """The pi0.5 finetune arms evaluated under openpi config `config_name` (cells.json
    eval_command --config); `config_name` names an openpi config, so only a pi0.5 arm can carry it
    -- checked explicitly (not just by `paper_ref` happening to be unset on every other backbone)
    since `check_arm` makes a LeRobot arm's `paper_ref is None` an enforced invariant.

    Raises `LookupError` when no arm carries it, rather than returning an empty list: a typo in a
    config name must not read as "no arms serve this cell"."""
    hits = [p for p in paper_arm_files(arms_dir)
            if (s := load_armspec(p)).backbone == "pi05-openpi" and s.stage == "finetune"
            and s.paper_ref == config_name]
    if not hits:
        raise LookupError(f"no paper run is evaluated under openpi config {config_name!r}")
    return hits
