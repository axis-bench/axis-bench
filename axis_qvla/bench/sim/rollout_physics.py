"""Close the loop: drive a served pi0.5 policy through a MuJoCo rollout.

Runs in the offline environment (mujoco + openpi_client). pi0.5 inference runs in a separate
openpi/.venv process behind a websocket (openpi's serve_policy.py), because jax and mujoco
cannot share a process. This is why the state-only MLP eval existed -- it sidestepped that
boundary; this crosses it.

Per control step (15 Hz):
  1. render pi05/base_cam + pi05/wrist_cam (224x224 uint8, letterboxed from 640x360)
  2. build the 8-D state obs, send {state, base_0_rgb, left_wrist_0_rgb, prompt}
  3. the server returns a (16, 8) action chunk; consume `replan` of it, then re-infer
  4. integrate each 8-D action into the sim (action_mapping_pi05)
  5. evaluate the task's REAL checker (task_checker.make_stable_checker)

Every non-trivial piece here is validated separately before use: cameras against the
dataset's own calibration (pi05_cameras), the action mapping by open-loop replay
(action_mapping_pi05), the checker across all 86 tasks (the seed screener). This module
only wires them together.
"""

from __future__ import annotations

import collections
import json
import os
import tempfile
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from axis_qvla.bench.sim.asset_guard import raise_if_refused
from axis_qvla.bench.sim.assets import make_asset_store
from axis_replay_native.asset_materializer import materialize_scene
from axis_replay_native.state_restore import restore_scene_state
from axis_replay_native.rollout import RolloutResult

from openpi_client import image_tools

from axis_qvla.bench.sim.obs_cameras import compile_with_cameras, BASE_CAM, WRIST_CAM
from axis_qvla.bench.sim.obs_geometry import observation_for_obs_key
from axis_qvla.bench.sim.action_mapping import Pi05ActionController, encode_state8

CheckerFactory = Callable[[dict | None], Callable[[mujoco.MjModel, mujoco.MjData], bool]]

RENDER_H, RENDER_W = 360, 640  # native training resolution; resize_with_pad -> 224 letterbox


def _load_task_payload(task_dir: Path) -> dict:
    return json.loads((task_dir / "task.json").read_text())


def _resolve_variant(payload: dict, seed: int) -> tuple[str | None, dict | None]:
    """Same seed % N selection rollout_policy uses."""
    variants = payload.get("scene_variants") or []
    if variants:
        v = variants[seed % len(variants)]
        return v.get("mjcf_xml") or payload.get("mjcf_xml"), (
            v.get("initial_state") or payload.get("initial_state")
        )
    return payload.get("mjcf_xml"), payload.get("initial_state")


def _build_obs(renderer, model, data, ctrl, *, open_w, closed_w, prompt) -> dict:
    """The exact payload the server expects: PRE-REPACKED keys, uint8 224 HWC images.

    The server does NOT apply repack/data transforms (create_trained_policy is built with
    an empty transform group), so the client must send base_0_rgb / left_wrist_0_rgb /
    state / prompt directly. left_wrist is mandatory: training always had it, so its mask
    was always True; omitting it flips the mask to a state the model never saw.
    """
    renderer.update_scene(data, camera=BASE_CAM)
    # Declared geometry, not resize_with_pad -- see axis_qvla/core/camera_geometry.py.
    base = observation_for_obs_key(renderer.render(), "base_0_rgb")
    renderer.update_scene(data, camera=WRIST_CAM)
    wrist = observation_for_obs_key(renderer.render(), "left_wrist_0_rgb")
    state8 = encode_state8(data, ctrl, open_w=open_w, closed_w=closed_w)
    return {
        "state": np.asarray(state8, dtype=np.float32),
        "base_0_rgb": np.asarray(base, dtype=np.uint8),
        "left_wrist_0_rgb": np.asarray(wrist, dtype=np.uint8),
        "prompt": prompt,
    }


def rollout_pi05_policy(
    task_dir: Path | str,
    client,
    *,
    init_state_seed: int,
    prompt: str,
    open_w: float,
    closed_w: float,
    checker_factory: CheckerFactory,
    fps: float = 15.0,
    replan: int = 8,
    max_steps: int = 1500,
    asset_dir: str | None = None,
    asset_fetch: str = "refuse",
    asset_base_url: str | None = None,
    asset_lock: str | None = None,
    video_path: Path | str | None = None,
    video_every: int = 5,
) -> RolloutResult:
    """One closed-loop pi0.5 rollout. ``client`` is an openpi_client WebsocketClientPolicy.

    When ``video_path`` is set, every ``video_every``-th base-camera frame (the SAME image
    the policy sees) is written to an mp4 -- a diagnostic to watch what the policy does,
    off by default so the eval path is unchanged.
    """
    task_dir = Path(task_dir)
    seed = int(init_state_seed)
    try:
        payload = _load_task_payload(task_dir)
    except Exception as exc:  # noqa: BLE001
        return RolloutResult(False, 0, 0.0, reason=f"error:load_task:{exc}")

    mjcf_xml, initial_state = _resolve_variant(payload, seed)
    if not mjcf_xml:
        return RolloutResult(False, 0, 0.0, reason="error:no_mjcf")

    checker_fn = checker_factory(initial_state)

    with tempfile.TemporaryDirectory(prefix="axis_pi05_") as work_dir:
        store = make_asset_store(asset_dir, fetch=asset_fetch, base_url=asset_base_url, lock=asset_lock)
        try:
            resolution = materialize_scene(mjcf_xml, work_dir, store)
        finally:
            close = getattr(store, "close", None)
            if callable(close):
                close()
        raise_if_refused(store)   # a refused include the materialiser swallowed (asset_guard)
        if resolution.missing_files:
            return RolloutResult(False, 0, 0.0, reason=f"error:missing_assets:{resolution.missing_files[:3]}")

        # MJCF asset paths are relative to the scene dir; compile from inside it.
        cwd = os.getcwd()
        os.chdir(work_dir)
        renderer = None
        try:
            try:
                model, data = compile_with_cameras(resolution.entry_xml_path)
            except Exception as exc:  # noqa: BLE001
                return RolloutResult(False, 0, 0.0, reason=f"error:compile:{exc}")
            try:
                restore_scene_state(model, data, initial_state, None, None)
            except Exception as exc:  # noqa: BLE001
                return RolloutResult(False, 0, 0.0, reason=f"error:restore:{exc}")

            ctrl = Pi05ActionController.from_model(model, fps=fps)
            ctrl.reset(data)
            renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
            client.reset()

            frames: list = [] if video_path is not None else None
            plan: collections.deque = collections.deque()
            success = False
            reason = "timeout"
            step = 0
            for step in range(max_steps):
                if not plan:
                    obs = _build_obs(
                        renderer, model, data, ctrl,
                        open_w=open_w, closed_w=closed_w, prompt=prompt,
                    )
                    chunk = np.asarray(client.infer(obs)["actions"])
                    if chunk.ndim != 2 or chunk.shape[1] < 8:
                        return RolloutResult(False, step, data.time, reason=f"error:bad_chunk:{chunk.shape}")
                    plan.extend(chunk[:replan])
                ctrl.apply(model, data, plan.popleft())
                if frames is not None and step % video_every == 0:
                    # base | wrist, side by side -- the two views the policy actually
                    # consumes, so the wrist framing can be checked over the whole rollout.
                    renderer.update_scene(data, camera=BASE_CAM)
                    base_f = np.asarray(renderer.render(), dtype=np.uint8)
                    renderer.update_scene(data, camera=WRIST_CAM)
                    wrist_f = np.asarray(renderer.render(), dtype=np.uint8)
                    frames.append(np.concatenate([base_f, wrist_f], axis=1))
                if checker_fn(model, data):
                    success, reason = True, "success"
                    break
            if frames:
                import imageio.v2 as imageio

                imageio.mimwrite(str(video_path), frames, fps=int(fps / video_every) or 1)
            return RolloutResult(success, step + 1, float(data.time), reason=reason)
        finally:
            if renderer is not None:
                renderer.close()
            os.chdir(cwd)
