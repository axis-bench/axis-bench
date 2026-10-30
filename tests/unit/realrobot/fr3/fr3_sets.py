"""Synthetic FR3 finetune sets under a test's tmp_path, with 224 x 224 frames of a chosen geometry.

The layout is the realdata tests' FR3 world (tests/unit/realrobot/realdata/fr3_world.py); the frames are
what a real set's pixels look like at 224: a 1280 x 720 frame letterboxed to a square has 49 black rows
above and below its 126 rows of content, a centre crop has content in every row. Content is a column
ramp, because a scene has texture in every row and a pad has none. Short episodes, because writing a set
computes lerobot's image statistics over every frame; and each kind of set is written once per test
process, beside the first test's own directory, and copied after that.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from tests.unit.realrobot.realdata.fr3_world import write_fr3_dataset
from tests.unit.realrobot.realdata.test_export import _episode
from tests.unit.realrobot.realdata.test_idle_filter import _recording

SIZE = 224
BAR = 49            # (224 - 224 * 720 / 1280) / 2


def _shade(episode: int, camera: int) -> int:
    return 90 + 40 * camera + episode


def content(episode: int, camera: int) -> np.ndarray:
    """A full 224 x 224 frame of scene: a ramp across the columns, in every row."""
    ramp = (_shade(episode, camera) + np.arange(SIZE) % 24).astype(np.uint8)
    return np.ascontiguousarray(np.broadcast_to(ramp[None, :, None], (SIZE, SIZE, 3)))


def crop_frame(episode: int, row: int, camera: int) -> np.ndarray:
    """A centre-cropped frame: content in every row."""
    return content(episode, camera)


def letterbox_frame(episode: int, row: int, camera: int) -> np.ndarray:
    """A letterboxed frame: black bars of BAR rows above and below the content."""
    x = content(episode, camera)
    x[:BAR] = x[SIZE - BAR:] = 0
    return x


def episodes() -> list:
    """Three short episodes of moving joints, two tasks; the second commands the gripper."""
    step = np.zeros(12)
    step[6:] = 1.0
    return [_episode(_recording(12), np.zeros(12), np.zeros(12), 0),
            _episode(np.outer(np.arange(8) * 0.01, np.ones(7)), step[:8], np.zeros(8), 1),
            _episode(_recording(12), step, step, 1)]


_WRITTEN: dict[str, Path] = {}


def fr3_set(root: Path, frame=crop_frame) -> Path:
    """A conforming FR3 finetune set at `root` (which must not exist): LeRobot v2.1, 15 fps, 8-wide
    state and actions, `image` / `extra_view_image` at 224 x 224, frames from `frame(episode, row, camera)`."""
    root = Path(root)
    key = f"{frame.__module__}.{frame.__qualname__}"
    src = _WRITTEN.get(key)
    if src is None or not (src / "meta" / "info.json").is_file():
        src = write_fr3_dataset(root.parent / f"_written_{frame.__name__}", episodes(), image_shape=(SIZE, SIZE, 3),
                                image_of=frame)
        _WRITTEN[key] = src
    return Path(shutil.copytree(src, root))


def edit_info(root: Path, edit) -> Path:
    """Rewrite `root`/meta/info.json as `edit(info)` returns it."""
    p = Path(root) / "meta" / "info.json"
    p.write_text(json.dumps(edit(json.loads(p.read_text()))))
    return root
