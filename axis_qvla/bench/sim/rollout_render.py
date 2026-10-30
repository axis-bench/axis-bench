"""Closed-loop pi0.5 rollout in the AXIS RENDER domain (robosuite TableArena).

Same control loop as rollout_pi05.py, but the scene comes from render_scene_pi05
(table + room + calibrated D405/ZED rig) instead of the bare physics scene, so the
policy sees the visual domain it was fine-tuned on.

Runs in the eval environment. pi0.5 inference stays in openpi/.venv behind the websocket,
exactly as before -- only the sim/render side changes environment.

Camera name mapping (their rig -> pi0.5 policy input keys):
    frontview -> base_0_rgb          (exterior D405 proxy)
    wrist     -> left_wrist_0_rgb    (link8 ZED proxy)
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Callable

import dataclasses

import mujoco
import numpy as np
from openpi_client import image_tools

from axis_qvla.bench.sim.action_mapping import Pi05ActionController, encode_state8
from axis_qvla.bench.sim.assets import AssetFetchRefused
from axis_qvla.bench.sim.layouts import LayoutSpec
from axis_qvla.bench.sim.render_scene import build_render_scene

CheckerFactory = Callable[[dict | None], Callable[[mujoco.MjModel, mujoco.MjData], bool]]

RENDER_W, RENDER_H = 640, 360  # native training resolution; resize_with_pad -> 224 letterbox
from axis_qvla.bench.sim.obs_geometry import observation_for_obs_key  # noqa: E402

CAMERA_TO_OBS_KEY = {"frontview": "base_0_rgb", "wrist": "left_wrist_0_rgb"}


@dataclasses.dataclass
class RolloutResult:
    """Field-compatible with axis_replay_native.rollout.RolloutResult.

    Redefined here rather than imported: in the eval environment the `axis_replay_native`
    package resolves to the AXIS-Render clone, which has no `rollout` module (that is our
    AXIS-Replay addition). Keeping the field names identical means the driver writes the
    same per-seed JSON schema as every other eval path.
    """

    success: bool
    n_steps: int
    sim_time: float
    trajectory: list = dataclasses.field(default_factory=list)
    reason: str = ""


def _build_obs(scene, ctrl, *, open_w: float, closed_w: float, prompt: str) -> dict:
    """Pre-repacked payload; the server applies no repack transform (see rollout_pi05)."""
    obs: dict = {}
    for cam, key in CAMERA_TO_OBS_KEY.items():
        frame = scene.render(cam, width=RENDER_W, height=RENDER_H)
        # Geometry comes from the declared profile, NOT resize_with_pad: the corpus and the
        # harness must apply the identical transform, and for a non-droid profile they differ.
        obs[key] = np.asarray(observation_for_obs_key(frame, key), dtype=np.uint8)
    state8 = encode_state8(scene.data, ctrl, open_w=open_w, closed_w=closed_w)
    obs["state"] = np.asarray(state8, dtype=np.float32)
    obs["prompt"] = prompt
    return obs


def _tracked_body_names(task_id: int, task_data_root: Path) -> list[str]:
    """Object/reference body names the task's success checker tracks -- so the diagnostic
    can report directly whether each exists in the compiled scene (the 'absent target body'
    hypothesis for an all-timeout result). `task_data_root` holds task_<id>/ (the rollout's
    task_dir.parent), so the diagnostic reads the same checker config the scorer does."""
    from axis_qvla.bench.sim.checker import load_checker_config

    names: list[str] = []

    def _walk(node):
        if not isinstance(node, dict):
            return
        for k in ("objName", "obj_name", "refName", "ref_name"):
            v = node.get(k)
            if isinstance(v, str) and v and v not in names:
                names.append(v)
        for sub in node.get("checkers", []) or []:
            _walk(sub)

    cfg = load_checker_config(task_id, task_data_root=task_data_root)
    _walk(cfg.get("checker", cfg))
    return names


class _VideoCapture:
    """Opt-in per-step frame capture, so a failed rollout can be WATCHED rather than inferred.

    Off unless $AXIS_ROLLOUT_VIDEO_EVERY is set, and even then only when diagnostics are on --
    the scoring path must stay exactly what produced the numbers of record. Rendering costs
    real time per frame, which is why this is not simply always on. A `timeout` says only that
    the arm never finished; a video shows what it did instead.
    """

    def __init__(self, path: Path, every: int) -> None:
        self.path, self.every, self.frames = path, max(1, every), []

    @classmethod
    def maybe(cls, trace_path, video_path=None):
        """Explicit `video_path` wins; otherwise fall back to $AXIS_ROLLOUT_VIDEO_EVERY.

        Two entry points because callers differ: the driver passes `video_path` directly when
        --diagnostics is on, while the env var lets an existing driver be re-run for video
        without editing it. Either way the stride comes from the env var, defaulting to 5.
        """
        import os

        every = os.environ.get("AXIS_ROLLOUT_VIDEO_EVERY")
        try:
            n = int(every) if every else 5
        except ValueError:
            n = 5
        if video_path is not None:
            return cls(Path(video_path), n)
        if not every or trace_path is None:
            return None
        return cls(Path(trace_path).with_suffix(".mp4"), n)

    def maybe_grab(self, scene, step: int) -> None:
        if step % self.every:
            return
        try:
            frame = scene.render("frontview", width=RENDER_W, height=RENDER_H)
            self.frames.append(np.asarray(frame, dtype=np.uint8))
        except Exception:  # noqa: BLE001
            pass

    def write(self, fps: float = 10.0) -> None:
        if not self.frames:
            return
        try:
            import imageio.v2 as imageio

            self.path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(self.path, self.frames, fps=fps, macro_block_size=1)
        except Exception:  # noqa: BLE001
            pass


def _save_frame(scene, path: Path) -> None:
    """Best-effort single frontview PNG. A render/imageio hiccup must not fail the rollout."""
    try:
        import imageio.v2 as imageio

        frame = scene.render("frontview", width=RENDER_W, height=RENDER_H)
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(path, np.asarray(frame, dtype=np.uint8))
    except Exception:  # noqa: BLE001
        pass


def rollout_pi05_render(
    task_dir: Path | str,
    client,
    *,
    init_state_seed: int,
    task_id: int,
    task_name: str,
    prompt: str,
    open_w: float,
    closed_w: float,
    checker_factory: CheckerFactory,
    render_root: Path | str,
    work_dir: Path | str,
    fps: float = 15.0,
    replan: int = 8,
    max_steps: int = 1500,
    eval_layout: LayoutSpec | None = None,
    trace_path: Path | str | None = None,
    video_path: Path | str | None = None,
    asset_dir: str | None = None,
    asset_fetch: str = "refuse",
    asset_base_url: str | None = None,
    asset_lock: str | None = None,
) -> RolloutResult:
    """One closed-loop rollout against the render-domain scene.

    `eval_layout` replaces the shipped scene-variant object XY with a freshly sampled,
    novel-but-equally-hard arrangement (see eval_layouts). Passing the spec rather than a
    manifest keeps the sampling decision -- including "this task is unevaluable" -- with the
    driver, which is the only place that can record it.

    `trace_path` is an OPT-IN diagnostic sidecar. When None (the default) the loop is
    byte-for-byte the normal eval path -- no extra imports, no extra work. When set, a
    read-only per-step numeric trace (actuator types, command-tracking gap, qpos, action
    stats, checker delta, tracked-body presence) is written there; see rollout_diagnostics.
    """
    scene = None
    try:
        try:
            scene = build_render_scene(
                task_dir,
                init_state_seed=init_state_seed,
                task_id=task_id,
                task_name=task_name,
                render_root=render_root,
                work_dir=work_dir,
                global_seed=0,
                eval_layout=eval_layout,
                asset_dir=asset_dir,
                asset_fetch=asset_fetch,
                asset_base_url=asset_base_url,
                asset_lock=asset_lock,
            )
        except AssetFetchRefused:
            raise   # a broken eval input aborts the run; it is not a failed rollout
        except Exception as exc:  # noqa: BLE001
            return RolloutResult(False, 0, 0.0, reason=f"error:build_scene:{exc}")

        model, data = scene.model, scene.data
        # scene.reset() already restored the variant pose AND raised objects onto the table.
        checker_fn = checker_factory(scene.initial_state)
        ctrl = Pi05ActionController.from_model(model, fps=fps)
        ctrl.reset(data)
        client.reset()

        diag = None
        if trace_path is not None:
            from axis_qvla.bench.sim.diagnostics import RolloutDiagnostics
            from axis_qvla.bench.sim.checker import load_checker_config, make_detail_probe

            task_data_root = Path(task_dir).parent
            probe = make_detail_probe(task_id, scene.initial_state, task_data_root=task_data_root)
            diag = RolloutDiagnostics.build(trace_path, ctrl, probe)
            diag.capture_static(model, data,
                                tracked_bodies=_tracked_body_names(task_id, task_data_root))
            _save_frame(scene, Path(trace_path).with_name(f"{Path(trace_path).stem}_frame0.png"))

        video = _VideoCapture.maybe(trace_path, video_path)
        plan: collections.deque = collections.deque()
        last_chunk = None
        success, reason, step = False, "timeout", 0
        for step in range(max_steps):
            if not plan:
                obs = _build_obs(scene, ctrl, open_w=open_w, closed_w=closed_w, prompt=prompt)
                chunk = np.asarray(client.infer(obs)["actions"])
                if chunk.ndim != 2 or chunk.shape[1] < 8:
                    return RolloutResult(False, step, float(data.time), reason=f"error:bad_chunk:{chunk.shape}")
                last_chunk = chunk
                plan.extend(chunk[:replan])
            action = plan.popleft()
            ctrl.apply(model, data, action)
            if diag is not None:
                diag.record_step(step, model, data, ctrl.q_cmd, action, last_chunk)
            if video is not None:
                video.maybe_grab(scene, step)
            if checker_fn(model, data):
                success, reason = True, "success"
                break
        if diag is not None:
            diag.record_step(step, model, data, ctrl.q_cmd, action, last_chunk, force=True)
            _save_frame(scene, Path(trace_path).with_name(f"{Path(trace_path).stem}_frameN.png"))
            diag.write(n_steps=step + 1, reason=reason, success=success)
        if video is not None:
            video.maybe_grab(scene, 0)   # final state, regardless of stride
            video.write()
        return RolloutResult(success, step + 1, float(data.time), reason=reason)
    finally:
        if scene is not None:
            scene.close()
