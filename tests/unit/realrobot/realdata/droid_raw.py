"""A raw DROID recording in the layout openpi's examples/droid/convert_droid_data_to_lerobot.py reads.

Per episode, `<root>/<episode>/`: `trajectory.h5` (`observation/{camera_type,controller_info,robot_state,
timestamp/cameras}` and `action/{joint_velocity,gripper_position}`, one row per step),
`recordings/MP4/<serial>.mp4` for each camera (1280 x 720, 15 fps, BGR as OpenCV writes it) and
`metadata_<id>.json`; at `<root>/aggregated-annotations-030724.json` each episode's instruction under its
`<id>`. Run in the converter's environment (numpy, h5py, OpenCV): `python -m tests.unit.realrobot.realdata.droid_raw
<root>` writes two episodes of 20 and 16 steps.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

WRIST, EXTERIOR_1, EXTERIOR_2 = "13062452", "24259877", "24514023"
CAMERA_TYPE = {WRIST: 0, EXTERIOR_1: 1, EXTERIOR_2: 1}      # DROID's camera types: 0 hand, 1 varied
H, W = 720, 1280
INSTRUCTIONS = ("put the cup on the plate", "open the drawer")


def raw_frame(episode: int, k: int, camera: int) -> np.ndarray:
    """A smooth RGB frame with a square that moves each step; each camera and episode its own colour."""
    y, x = np.mgrid[0:H, 0:W]
    img = np.stack([x * 255 // W, y * 255 // H, np.full_like(x, 40 + 50 * episode + 60 * camera)], axis=-1)
    x0 = (96 * k) % (W - 160)
    img[160:320, x0:x0 + 160] = (240, 240 - 100 * camera, 30)
    return img.astype(np.uint8)


def episode_arrays(episode: int, n: int) -> dict[str, np.ndarray]:
    """The recorded joint positions, gripper position and actions of one episode."""
    rng = np.random.default_rng(100 + episode)
    return {"joint_positions": rng.normal(size=(n, 7)), "gripper_position": (np.arange(n) + 1) / (n + 1),
            "joint_velocity": rng.uniform(-1, 1, (n, 7)), "action_gripper": rng.uniform(0, 1, n)}


def write_droid_raw(root: Path, lengths=(20, 16)) -> dict[str, dict]:
    """Write the recording; {episode id: {"instruction", "n", **episode_arrays}} of what it holds."""
    import cv2
    import h5py

    root = Path(root)
    out, notes = {}, {}
    for e, n in enumerate(lengths):
        eid = f"EP{e}"
        ep = root / f"episode_{e}"
        (ep / "recordings" / "MP4").mkdir(parents=True)
        arrays = episode_arrays(e, n)
        with h5py.File(ep / "trajectory.h5", "w") as f:
            for serial, kind in CAMERA_TYPE.items():
                f[f"observation/camera_type/{serial}"] = np.full(n, kind, np.int64)
                f[f"observation/timestamp/cameras/{serial}_frame_received"] = np.arange(n, dtype=np.int64)
            f["observation/controller_info/movement_enabled"] = np.ones(n, bool)
            f["observation/robot_state/joint_positions"] = arrays["joint_positions"]
            f["observation/robot_state/gripper_position"] = arrays["gripper_position"]
            f["action/joint_velocity"] = arrays["joint_velocity"]
            f["action/gripper_position"] = arrays["action_gripper"]
        for c, serial in enumerate(CAMERA_TYPE):
            video = cv2.VideoWriter(str(ep / "recordings" / "MP4" / f"{serial}.mp4"),
                                    cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H))
            for k in range(n):
                video.write(raw_frame(e, k, c)[..., ::-1])
            video.release()
        (ep / f"metadata_{eid}.json").write_text(json.dumps({"uuid": eid}))
        notes[eid] = {"language_instruction1": INSTRUCTIONS[e % len(INSTRUCTIONS)]}
        out[eid] = {"instruction": notes[eid]["language_instruction1"], "n": n, **arrays}
    (root / "aggregated-annotations-030724.json").write_text(json.dumps(notes))
    return out


if __name__ == "__main__":
    write_droid_raw(Path(sys.argv[1]))
    print(f"OK    raw DROID recording -> {sys.argv[1]}")
