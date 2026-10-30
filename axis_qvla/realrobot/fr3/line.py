"""The real-robot line record (experiments/realrobot/line.toml) and the ArmSpec each of its runs declares.

`run_spec(line, run_id)` is the one mapping from the record to an ArmSpec: the run files in
experiments/realrobot/runs/ are its canonical TOML (axis_qvla.train.armspec.dumps_armspec), and the checks
compare every file with it. Inputs that ship nowhere are `local:` names: the rendered corpus's DROID-8D
view and the idle-filtered FR3 copy with its tail-10 ranges (the row spaces), the compiled labels
`realign compile` writes (`realalign_<signal>`), and the norm statistics `fr3 norm` computes
(`norm_<embodiment>`). Offline tier (stdlib only).
"""
from __future__ import annotations

import dataclasses
import tomllib
import types
from pathlib import Path

from axis_qvla.train.armspec import ArmSpec

SCHEMA = "qvla.realrobot_line/1"
BACKBONE = "pi05-openpi"
TRACK = "realrobot"
SEED = 42                      # openpi TrainConfig's default seed
ACTION_SPACE = "droid8"
ROWSPACES = {"pretrain": "local:realalign_droid8d", "finetune": "local:fr3_392_tail10"}
PRESETS = {"pretrain": "pi05.pretrain.lora.realrobot", "finetune": "pi05.finetune.lora.realrobot",
           "finetune_only": "pi05.finetune.lora.realrobot_only"}
QUALITY_TAG = 5                # the tagged finetune's code: the top quality bin, on every FR3 prompt
# A stage's `image` -> the transform that made its training frames from the raw camera frames, as a
# handshake's `image_geometry` names it (axis_qvla.bench.protocol.IMAGE_TRANSFORMS).
IMAGES = {"center_crop": "center_crop224", "letterbox": "letterbox224"}


class LineError(ValueError):
    """The line record is unreadable or of another schema."""


@dataclasses.dataclass(frozen=True)
class Line:
    stage1: dict
    stage2: dict
    units: dict
    serve: dict
    runs: dict


def load_line(path: Path) -> Line:
    try:
        d = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise LineError(f"{path}: cannot read ({exc.__class__.__name__}: {exc})") from None
    if d.get("schema") != SCHEMA:
        raise LineError(f"{path}: schema {d.get('schema')!r}, expected {SCHEMA!r}")
    missing = [t for t in ("stage1", "stage2", "units", "serve", "runs") if t not in d]
    if missing:
        raise LineError(f"{path}: missing table(s) {missing}")
    return Line(d["stage1"], d["stage2"], d["units"], d["serve"], d["runs"])


def kind(spec: dict) -> str:
    """`pretrain`, `finetune` (from a stage-1 run) or `finetune_only` (from a base model)."""
    return "finetune_only" if spec["stage"] == "finetune" and spec["init"].startswith("base:") else spec["stage"]


def stage_of(line: Line, spec: dict) -> dict:
    return line.stage1 if spec["stage"] == "pretrain" else line.stage2


def run_spec(line: Line, run_id: str) -> ArmSpec:
    """The ArmSpec the record declares for `run_id`."""
    s = line.runs[run_id]
    st = stage_of(line, s)
    pretrain = s["stage"] == "pretrain"
    return ArmSpec(
        id=run_id, backbone=BACKBONE, stage=s["stage"], track=TRACK, paper_ref=None,
        init=s.get("init", line.stage1["init"]), rowspace=ROWSPACES[s["stage"]],
        bundle=f"local:realalign_{s['signal']}" if pretrain else None,
        norm=f"local:norm_{st['embodiment']}", preset=PRESETS[kind(s)], seed=SEED,
        overrides=types.MappingProxyType({"data.quality_tag": QUALITY_TAG} if s.get("quality_tag") else {}),
        data=types.MappingProxyType({"action_space": ACTION_SPACE, "image": st["image"]}))
