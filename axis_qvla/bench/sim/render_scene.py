"""Build the AXIS render-domain scene (robosuite TableArena) for pi0.5 closed-loop eval.

WHY THIS EXISTS
    The task's physics scene (task.json) is a *physics twin*: an infinite procedural
    gray-checker plane, no table. The pi0.5 training frames were rendered by AXIS-Render,
    which swaps that shell for a robosuite `TableArena` (herringbone tabletop, composed
    room, calibrated D405/ZED camera rig) and raises the robot + objects onto the table.
    Evaluating on the bare physics scene put the fully-trainable SigLIP tower far
    out-of-distribution; this module reproduces the training visual domain instead.

    Runs in the eval environment (robosuite 1.5.2 + mujoco 3.3.7 + numpy<2), NOT the
    offline environment -- robosuite pins numpy<2 and would break the offline pipeline.

WHAT IT RETURNS
    A compiled scene whose `model`/`data` are ordinary MuJoCo handles with the SAME
    nq/nu as the physics scene, so `action_mapping_pi05.Pi05ActionController` and the
    task's real CheckerManager both apply unchanged.

KEY DETAIL -- the object raise
    The renderer lifts the robot root in the XML but leaves object free joints where the
    source put them; their z must be raised by `xml_metadata["free_joint_qpos_z_offset"]`
    AFTER restoring the scene-variant initial_state. Skipping it leaves every object
    buried under the tabletop (visually: an empty table). This is the same transform
    `_build_scene_state_transform` applies to replay trajectories.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from axis_qvla.bench.sim.surface_appearance import DEFAULT_TABLE_TEXTURE, surface_appearance_override
from axis_qvla.bench.sim.asset_guard import fresh_scene_dir, raise_if_refused
from axis_qvla.bench.sim.assets import make_asset_store
from axis_replay_native.asset_materializer import materialize_scene
from axis_replay_native.state_restore import restore_scene_state
from axis_replay_native.robosuite_render import (
    AxisReplayRobosuiteEnv,
    _build_scene_state_transform,
    _load_embodiment_camera_config,
)
from axis_replay_native.robosuite_visual_scene import resolve_visual_scene_profile

CAMERA_NAMES = ("frontview", "wrist")


@dataclasses.dataclass
class RenderScene:
    """A compiled render-domain scene plus the handles the rollout loop needs."""

    env: Any
    model: mujoco.MjModel
    data: mujoco.MjData
    initial_state: dict | None
    qpos_offset: np.ndarray
    camera_names: tuple[str, ...]
    variant_index: int = -1   # scene variant actually used
    layout_seed: int = -1     # >=0 when object XY came from a generated eval layout

    def reset(self) -> None:
        """Restore the scene-variant pose, then raise objects onto the table."""
        restore_scene_state(self.model, self.data, self.initial_state, None, None)
        # Objects only: qpos_offset is non-zero exactly at free-joint z indices.
        self.data.qpos[:] = np.asarray(self.data.qpos) + self.qpos_offset
        mujoco.mj_forward(self.model, self.data)

    def render(self, camera: str, width: int = 640, height: int = 360) -> np.ndarray:
        """RGB uint8 HWC. Robosuite offscreen output is upside down (README)."""
        return np.flipud(self.env.sim.render(camera_name=camera, width=width, height=height))

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()


def _resolve_variant(payload: dict, seed: int, task_id: int) -> tuple[str | None, dict | None, int]:
    """The scene of a task evaluated without an eval layout: its single shipped scene.

    A task with scene variants is evaluated on a generated layout (`eval_layout` or
    `eval_layout_manifest`), which names its base variant; picking a variant by seed instead
    would evaluate on arrangements the benchmark does not declare, so it is refused here.
    """
    if payload.get("scene_variants"):
        raise ValueError(
            f"task {task_id} has scene variants: pass eval_layout or eval_layout_manifest "
            f"(the benchmark's layouts) to choose the scene for seed {seed}")
    return payload.get("mjcf_xml"), payload.get("initial_state"), -1


def build_render_scene(
    task_dir: Path | str,
    *,
    init_state_seed: int,
    task_id: int,
    task_name: str,
    render_root: Path | str,
    work_dir: Path | str,
    attempt_id: int = 0,
    global_seed: int = 0,
    table_texture: str | None = DEFAULT_TABLE_TEXTURE,
    asset_dir: str | None = None,
    asset_fetch: str = "refuse",
    asset_base_url: str | None = None,
    asset_lock: str | None = None,
    eval_layout_manifest: Path | str | None = None,
    eval_layout: Any | None = None,
) -> RenderScene:
    """Materialise the task scene and wrap it in the AXIS render shell.

    `eval_layout_manifest` (the homogeneous manifest) switches object placement from the
    shipped scene-variant layout to a FRESHLY GENERATED one for this seed: same object set,
    same 50x50 cm randomization box, but an arrangement no training demo used, so success
    measures competence rather than recall. The variant is still used for geometry/assets;
    only the object XY is resampled. A task with scene variants requires one of the two.

    `eval_layout` (an `eval_layouts.LayoutSpec`) supplies that layout directly and takes
    precedence. A rollout driver has to RECORD which layout each seed used, and it must
    learn that a task is unevaluable before it starts rolling out rather than on some seed
    in the middle -- both need the LayoutSpec object, not a path to sample it from again.
    """
    task_dir = Path(task_dir)
    render_root = Path(render_root)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads((task_dir / "task.json").read_text())

    layout_seed = -1
    layout = eval_layout
    if layout is None and eval_layout_manifest is not None:
        from axis_qvla.bench.sim.layouts import generate_layout

        layout = generate_layout(
            int(task_id), int(init_state_seed), homogeneous_manifest=eval_layout_manifest,
            task_data_root=task_dir.parent,
        )
    if layout is not None:
        # The layout names its base variant, so the seed->variant map (_resolve_variant) is not
        # consulted. The selection lives in layouts.scene_for_layout so the offline-venv scene
        # golden hashes this exact code.
        from axis_qvla.bench.sim.layouts import scene_for_layout

        mjcf_xml, initial_state, variant_index = scene_for_layout(payload, layout, task_id)
        layout_seed = int(layout.seed)
    else:
        mjcf_xml, initial_state, variant_index = _resolve_variant(payload, int(init_state_seed), task_id)
    if not mjcf_xml:
        raise ValueError(f"no mjcf_xml for {task_dir}")

    # An emptied scene dir: the materialiser would otherwise reuse whatever a previous build left
    # in work_dir without asking the store, i.e. without the asset dir or the lock (asset_guard).
    scene_dir = fresh_scene_dir(work_dir)
    store = make_asset_store(asset_dir, fetch=asset_fetch, base_url=asset_base_url, lock=asset_lock)
    try:
        resolution = materialize_scene(mjcf_xml, str(scene_dir), store)
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()
    raise_if_refused(store)
    if resolution.missing_files:
        raise RuntimeError(f"missing assets: {resolution.missing_files[:3]}")
    entry = Path(resolution.entry_xml_path)
    if not entry.is_absolute():
        entry = (scene_dir / entry).resolve()

    cfg = _load_embodiment_camera_config("franka", CAMERA_NAMES)
    override = surface_appearance_override(cfg, table_texture)
    if override is not None:
        cfg.setdefault("render_profile", {})["surface_appearance"] = override
    cfg = resolve_visual_scene_profile(
        cfg,
        project_root=render_root,
        global_seed=global_seed,
        task_id=task_id,
        attempt_id=attempt_id,
        task_name=task_name,
        variant_id=variant_index,
    )

    env = AxisReplayRobosuiteEnv(
        entry_xml=entry,
        camera_names=CAMERA_NAMES,
        camera_config=cfg,
        layout_cache_dir=work_dir / "layouts",
        has_renderer=False,
        has_offscreen_renderer=True,
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=True,
        hard_reset=False,
        renderer="mujoco",
        seed=global_seed,
    )
    model = env.sim.model._model
    data = env.sim.data._data
    qpos_offset, _ = _build_scene_state_transform(model, env.xml_metadata)

    scene = RenderScene(
        env=env,
        model=model,
        data=data,
        initial_state=initial_state,
        qpos_offset=np.asarray(qpos_offset, dtype=np.float64),
        camera_names=CAMERA_NAMES,
        variant_index=variant_index,
        layout_seed=layout_seed,
    )
    scene.reset()
    return scene
