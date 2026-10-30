"""The request a policy server of the real-world runs reads, as openpi's DROID example client sends it.

openpi's examples/droid/main.py drives a DROID robot with `RobotEnv(action_space="joint_velocity",
gripper_action_space="position")` at 15 Hz: each action is DROID's normalised joint-velocity command and
the absolute gripper position, which is what a run fine-tuned on DROID-layout demonstrations predicts.
droid_drive.py, beside this module, is that client with the three edits made: it sends `request(...)`
instead of the DROID keys, holds every reply to CHUNK and keeps executing OPEN_LOOP actions per call.
`qvla realrobot serve` copies both files into a run's drive kit.

The frames a run trains on are the ones openpi's DROID converter stores: each RGB camera frame resized
from RECORDED to STORED by its `resize_image` (Pillow, bicubic, uint8). `request` resizes the client's
camera frames with that same function, so the server receives the pixels a training frame holds. From
there the server applies its config's input transform, identical at training and serving: the centre
square crop of each camera (`AxisFrankaInputs(center_crop=True)`, the runs' `image_center_crop`), then
the resize with padding to 224 x 224 (`ResizeImages(224, 224)`). The state is one float32 array in the
model's order, `[q0..q6, gripper]`, as training-set writes it.

This module imports numpy and, inside `resize_image`, Pillow, both of which the client's environment
has, so it can be copied next to the client. `python -m axis_qvla.realrobot.fr3.droid_client EXTERIOR WRIST`
prints the request built from two camera frames (image files). Offline tier.
"""
from __future__ import annotations

import sys

import numpy as np

OBS_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "state", "prompt")   # axis_qvla.bench.protocol.OBS_KEYS
CHUNK = (15, 8)          # the FR3 runs' action horizon, and 7 joint commands + the gripper
OPEN_LOOP = 8            # actions executed per call, as the client's open_loop_horizon
RECORDED = (1280, 720)   # (width, height) of the DROID cameras' frames
STORED = (320, 180)      # (width, height) of the frames the converter stores and the runs train on


def resize_image(image, size):
    """openpi examples/droid/convert_droid_data_to_lerobot.py's `resize_image`."""
    from PIL import Image
    image = Image.fromarray(image)
    return np.array(image.resize(size, resample=Image.BICUBIC))


def frame(image, camera: str) -> np.ndarray:
    """`camera`'s RGB frame at the stored size: a recorded frame resized as the converter resizes it, a
    stored one unchanged. Any other size raises ValueError."""
    image = np.asarray(image)
    size = (image.shape[1], image.shape[0]) if image.ndim == 3 and image.shape[2] == 3 else None
    if image.dtype != np.uint8 or size not in (RECORDED, STORED):
        got = f"{size[0]} x {size[1]} {image.dtype}" if size else f"{image.dtype} of shape {image.shape}"
        raise ValueError(f"the {camera} camera's frame is {got}; a frame is uint8 RGB at {RECORDED[0]} x "
                         f"{RECORDED[1]} (as recorded) or {STORED[0]} x {STORED[1]} (as stored): pass the "
                         f"client's camera frames as captured, after its BGR-to-RGB flip, not its "
                         f"resize_with_pad(..., 224, 224) output")
    return np.ascontiguousarray(image) if size == STORED else resize_image(image, STORED)


def request(exterior: np.ndarray, wrist: np.ndarray, joint_position, gripper_position, prompt: str) -> dict:
    """The observation a real-world run's server reads: both frames at the stored size, the state
    `[q0..q6, gripper]` as float32, and the instruction."""
    state = np.concatenate([np.asarray(joint_position, np.float32).reshape(7),
                            np.asarray(gripper_position, np.float32).reshape(1)])
    return dict(zip(OBS_KEYS, (frame(exterior, "exterior"), frame(wrist, "wrist"), state, str(prompt))))


def main(argv: list[str] | None = None) -> int:
    from PIL import Image

    paths = sys.argv[1:] if argv is None else argv
    if len(paths) != 2:
        print("usage: python -m axis_qvla.realrobot.fr3.droid_client EXTERIOR WRIST", file=sys.stderr)
        return 2
    frames = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
    try:
        got = request(*frames, np.zeros(7), np.zeros(1), "")
    except ValueError as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        return 2
    sizes = [f"{key} {got[key].shape[1]} x {got[key].shape[0]} (from {f.shape[1]} x {f.shape[0]})"
             for key, f in zip(OBS_KEYS, frames)]
    print(f"{', '.join(sizes)}, state {got['state'].size} {got['state'].dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
