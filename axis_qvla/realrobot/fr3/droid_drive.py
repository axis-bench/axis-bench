"""Drive a DROID robot with a real-world run's policy server: openpi's DROID example client with the
three edits this line needs already made.

openpi's examples/droid/main.py (Apache-2.0, Physical Intelligence) steps DROID's
RobotEnv(action_space="joint_velocity", gripper_action_space="position") at 15 Hz. It asks the server for
a chunk of actions every 8 steps. A server of a real-world run reads other keys and returns 15 actions,
which took three hand edits of that file. This client has them made:
- it sends `droid_client.request(...)`: both camera frames at the size openpi's DROID converter stored
  (320 x 180, with its bicubic resize), the state `[q0..q6, gripper]` as one float32 array, and the
  instruction;
- it holds every reply to droid_client.CHUNK, (15, 8), and runs OPEN_LOOP = 8 actions of each;
- it finds the exterior camera your demonstrations stored as `exterior_image_1_left`, by comparing each
  exterior camera's first frame with the drive kit's reference frame (`--external-camera` names it
  instead), so nobody picks the camera by eye.
Like openpi's client, it binarises the gripper at 0.5 and clips every action to [-1, 1]. It refuses a
server whose handshake is not an FR3 one (DROID's joint-velocity command at 15 Hz, 15 actions of 8), so
it never drives the robot with another policy. It also refuses a reply that is the right shape but holds
a non-finite value (nan or inf), the same way it refuses a wrongly shaped one: the rollout stops instead
of commanding the robot with it.

Run it on the robot's control computer, in DROID's environment with openpi's client installed, from the
drive kit `qvla realrobot serve` writes (this file, droid_client.py, reference_exterior.png, drive.json):

    python3 droid_drive.py --remote-host <server ip>

The cameras' serial numbers come from DROID's own parameters (droid.misc.parameters) unless given. Each
rollout is recorded in results/eval_<time>.csv in the working directory, one row per rollout with the
columns run, remote_host, instruction, success, steps, camera (the run from drive.json, the server this
command talked to, the instruction driven, whether it succeeded, how many steps it ran, and the exterior
camera used). Neither the per-rollout video nor the live robot_camera_views.png is recorded; only this
CSV is. Module scope needs only the standard library and numpy; droid and openpi_client are imported in
`main`, and Pillow when the reference frame is read.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime
import importlib.util
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np


def _sibling(name: str):
    """`name`.py beside this file, loaded by path. In a checkout that is this package's module, in a drive
    kit its copy; nothing has to be installed for either."""
    spec = importlib.util.spec_from_file_location(f"_droid_drive_{name}", Path(__file__).resolve().parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_client = _sibling("droid_client")
CHUNK, OPEN_LOOP, request, frame = _client.CHUNK, _client.OPEN_LOOP, _client.request, _client.frame
FPS = 15
GRIPPER_THRESHOLD = 0.5
FR3 = {"action_units": "droid_normalized_joint_velocity_command", "action_fps": FPS,
       "action_horizon": CHUNK[0], "action_dim_out": CHUNK[1]}
MARGIN = 1.2      # the other exterior camera must be this many times farther from the reference frame


def check_handshake(md) -> None:
    wrong = [f"{k} {md.get(k)!r} (an FR3 server declares {v!r})" for k, v in FR3.items() if md.get(k) != v]
    if wrong:
        raise ValueError("the server does not serve a real-world run: " + "; ".join(wrong))


def lens(obs: dict, serial: str) -> np.ndarray:
    """Camera `serial`'s left-lens frame in a RobotEnv observation, as openpi's client takes it: the image
    key naming the serial and `left`, alpha dropped, BGR flipped to RGB."""
    for key, image in obs["image"].items():
        if serial in key and "left" in key:
            return np.ascontiguousarray(np.asarray(image)[..., :3][..., ::-1])
    raise KeyError(f"no left-lens frame of camera {serial} in the observation (images: {sorted(obs['image'])})")


def pick_camera(frames: dict, reference: np.ndarray) -> tuple[str, dict]:
    """(serial, {serial: distance}): the exterior camera whose frame, at the stored size, is nearest the
    reference frame (mean absolute difference). ValueError when the next one is not MARGIN times farther."""
    ref = np.asarray(reference, np.float32)
    dist = {s: float(np.abs(frame(f, s).astype(np.float32) - ref).mean()) for s, f in frames.items()}
    ranked = sorted(dist, key=dist.get)
    if len(ranked) > 1 and dist[ranked[1]] <= MARGIN * dist[ranked[0]]:
        raise ValueError(f"cameras {ranked[0]} and {ranked[1]} are about as near the reference frame "
                         f"({dist[ranked[0]]:.1f} and {dist[ranked[1]]:.1f}): pass --external-camera <serial>")
    return ranked[0], dist


def to_command(action) -> np.ndarray:
    a = np.array(action, dtype=np.float64)
    a[-1] = 1.0 if a[-1] > GRIPPER_THRESHOLD else 0.0
    return np.clip(a, -1.0, 1.0)


@contextlib.contextmanager
def _deferred_interrupt():
    """Ctrl+C during a server call is held until the reply is in, as openpi's client does: an interrupted
    call drops the connection."""
    hit = []
    previous = signal.signal(signal.SIGINT, lambda *_: hit.append(True))
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
        if hit:
            raise KeyboardInterrupt


def rollout(env, policy, instruction: str, exterior: str, wrist: str, *, max_steps: int = 600,
            clock=time.monotonic, sleep=time.sleep, frames=None) -> int:
    """One rollout: a server call every OPEN_LOOP steps, one action per step at FPS. Returns the steps run
    (Ctrl+C ends it early)."""
    chunk, used = None, OPEN_LOOP
    for step in range(max_steps):
        start = clock()
        try:
            obs = env.get_observation()
            ext = lens(obs, exterior)
            if frames is not None:
                frames.append(ext)
            if used >= OPEN_LOOP:
                state = obs["robot_state"]
                req = request(ext, lens(obs, wrist), state["joint_positions"], [state["gripper_position"]], instruction)
                with _deferred_interrupt():
                    chunk = np.asarray(policy.infer(req)["actions"])
                if chunk.shape != CHUNK:
                    raise ValueError(f"the server returned actions of shape {chunk.shape}, not {CHUNK}")
                if not np.isfinite(chunk).all():
                    raise ValueError("the server returned a non-finite action (nan or inf): refusing to "
                                     "command the robot")
                used = 0
            env.step(to_command(chunk[used]))
            used += 1
            spare = 1.0 / FPS - (clock() - start)
            if spare > 0:
                sleep(spare)
        except KeyboardInterrupt:
            return step
    return max_steps


def _droid_serials():
    """(wrist, (exterior, exterior)) from DROID's parameters, or (None, ()) without them."""
    try:
        from droid.misc import parameters
    except ImportError:
        return None, ()
    ext = tuple(s for s in (getattr(parameters, "varied_camera_1_id", ""), getattr(parameters, "varied_camera_2_id", "")) if s)
    return getattr(parameters, "hand_camera_id", "") or None, ext


def _instruction(choices) -> str:
    """One of `choices` by number, or any other non-empty text typed instead. Asks again for an empty
    answer and for digits that are not 1..len(choices) -- a number out of range is a typo, not an
    instruction literally named "9"."""
    for i, text in enumerate(choices, 1):
        print(f"  {i}. {text}")
    while True:
        got = input("Instruction (a number above, or the words): ").strip()
        if not got:
            print("  give a number above, or the instruction's words")
            continue
        if got.isdigit():
            i = int(got)
            if 1 <= i <= len(choices):
                return choices[i - 1]
            print(f"  {i} is not 1..{len(choices)}: give a number in that range, or the words")
            continue
        return got


def _success() -> float:
    while True:
        got = input("Did the rollout succeed? (y = 100%, n = 0%, or 0-100): ").strip().lower()
        value = {"y": 100.0, "n": 0.0}.get(got)
        if value is None:
            try:
                value = float(got)
            except ValueError:
                continue
        if 0 <= value <= 100:
            return value / 100


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--remote-host", required=True, help="the policy server (`qvla realrobot serve` runs it)")
    ap.add_argument("--remote-port", type=int, default=8000)
    ap.add_argument("--external-camera", default=None, help="the exterior camera's serial (default: the one nearest the kit's reference frame)")
    ap.add_argument("--exterior-camera-ids", nargs=2, default=None, metavar="SERIAL", help="the two exterior cameras (default: DROID's parameters)")
    ap.add_argument("--wrist-camera-id", default=None, help="the wrist camera (default: DROID's parameters)")
    ap.add_argument("--max-timesteps", type=int, default=600)
    ap.add_argument("--kit", type=Path, default=Path(__file__).resolve().parent, help="the drive kit (default: this file's directory)")
    a = ap.parse_args(argv)
    try:
        from droid.robot_env import RobotEnv
        from openpi_client import websocket_client_policy
    except ImportError as exc:
        print(f"FAIL  {exc.name} is not installed: run this on the robot's control computer, in DROID's environment "
              f"with openpi's client (pip install -e <openpi checkout>/packages/openpi-client)", file=sys.stderr)
        return 2
    kit = json.loads((a.kit / "drive.json").read_text(encoding="utf-8"))
    wrist, exteriors = _droid_serials()
    wrist, exteriors = a.wrist_camera_id or wrist, tuple(a.exterior_camera_ids or exteriors)
    if wrist is None or (a.external_camera is None and len(exteriors) != 2):
        print("FAIL  name the cameras: --wrist-camera-id and --exterior-camera-ids (DROID's parameters name none)", file=sys.stderr)
        return 2
    policy = websocket_client_policy.WebsocketClientPolicy(a.remote_host, a.remote_port)
    try:
        check_handshake(policy.get_server_metadata())
    except ValueError as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        return 2
    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    exterior = a.external_camera
    if exterior is None:
        from PIL import Image
        reference = np.asarray(Image.open(a.kit / kit["reference"]).convert("RGB"))
        obs = env.get_observation()
        try:
            exterior, dist = pick_camera({s: lens(obs, s) for s in exteriors}, reference)
        except (KeyError, ValueError) as exc:
            print(f"FAIL  {exc}", file=sys.stderr)
            return 2
        print(f"exterior camera {exterior} (distance to the reference frame: "
              + ", ".join(f"{s} {d:.1f}" for s, d in dist.items()) + ")")
    out = Path("results") / f"eval_{datetime.datetime.now():%Y_%m_%d_%H%M%S}.csv"
    out.parent.mkdir(exist_ok=True)
    run, host = kit.get("run", ""), a.remote_host
    with out.open("w", newline="") as f:
        rows = csv.writer(f)
        rows.writerow(["run", "remote_host", "instruction", "success", "steps", "camera"])
        while True:
            instruction = _instruction(kit.get("instructions", []))
            print("Running the rollout; Ctrl+C stops it early.")
            steps = rollout(env, policy, instruction, exterior, wrist, max_steps=a.max_timesteps)
            rows.writerow([run, host, instruction, _success(), steps, exterior])
            f.flush()
            if input("Another rollout? (y/n): ").strip().lower() != "y":
                break
            env.reset()
    print(f"results -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
