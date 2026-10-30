"""DUAL-SIM closed-loop pi0.5 rollout: physics for dynamics, render for pixels.

WHY THIS EXISTS
    Single-domain rollout fails both ways (measured):
      * RENDER domain (rollout_pi05_render): photorealistic, but AXIS-Render objects are
        visual-only (contype=0/conaffinity=0), so they free-fall at t=0 and no task can
        succeed.
      * PHYSICS domain (rollout_pi05): objects hold and the arm is controllable, but the
        scene renders as bare gray MuJoCo (checkerboard floor, untextured arm) -- far
        out-of-distribution vs the photorealistic frames pi0.5 was fine-tuned on.

    Dual-sim runs BOTH scenes in one process and splits their roles:
      * a PHYSICS scene (real collision, objects hold) is the one we STEP; and
      * a RENDER scene (robosuite TableArena, photorealistic) that we NEVER step -- we copy
        the physics joint state into it and mj_forward (kinematics only) so it renders the
        exact same configuration in the visual domain the policy expects.

    Feasibility is proven: both builds share an IDENTICAL joint layout (13 joints, nq=37,
    every joint name shared -- franka/panda_joint1-7, the 2 fingers, and the object free
    joints), so the physics->render state sync is a per-joint copy by name. The render
    shell lifts the whole scene onto a table, so object free-joint z gets the render
    scene's `qpos_offset` added on top of the copy (exactly what RenderScene.reset does).

    Runs in the eval environment (robosuite 1.5.2 + axis_replay_native core + mujoco 3.3.7).
    pi0.5 inference stays in openpi/.venv behind the websocket, as in every other path.

    NOTE: this module deliberately does NOT import axis_qvla.bench.sim.rollout_physics --
    that module imports `axis_replay_native.rollout` at module scope, which is absent in
    the eval environment. The physics build (materialize + compile + restore) is inlined here
    from the core axis_replay_native modules that ARE present.
"""

from __future__ import annotations

import collections
import json
import os
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np
from openpi_client import image_tools

import os as _os
from axis_qvla.bench.sim.inference_noise import noise_for
_PROV = _os.environ.get("AXIS_EVAL_PROVENANCE") == "1"
from axis_qvla.bench.sim.replan import effective_replan
from axis_qvla.bench.sim.asset_guard import fresh_scene_dir, raise_if_refused
from axis_qvla.bench.sim.assets import AssetFetchRefused, make_asset_store
from axis_replay_native.asset_materializer import materialize_scene
from axis_replay_native.state_restore import restore_scene_state

from axis_qvla.bench.sim.action_mapping import (Pi05ActionController, encode_state8, encode_state9,
                                          encode_state_eef)

# WHICH ACTION SPACE THE SERVED POLICY SPEAKS.
#   "droid8" -- 7 joint VELOCITIES + closedness (convert_droid_actions output). The default, and
#               what every paper checkpoint was trained on.
#   "joint9" -- 7 absolute joint POSITIONS + 2 finger widths, which is what the stage-1 pretraining
#               corpus and the held-out render actually store.
# This is NOT cosmetic: measured by open-loop replay of one demo, replaying its 9-D actions through
# the joint9 path tracks the recording to 0.0087 rad mean, and through the droid8 path to 1.78 rad
# -- the arm ends up ~102 degrees from where the demonstration was. The served action space must
# therefore be the one the model was trained on.
# The resolved value is logged at rollout start and recorded in the result, so a run's action space
# is recoverable afterwards rather than inferred from an env var that may since have changed.
EVAL_ACTION_SPACE = os.environ.get("AXIS_EVAL_ACTION_SPACE", "droid8")
_MIN_ACTION_WIDTH = {"eef7": 7, "droid8": 8, "joint9": 9}.get(EVAL_ACTION_SPACE, 8)
# Where the gripper channel sits in the action vector -- eef7 puts it at 6, not 7. The DIAG
# telemetry indexed [7] unconditionally and raised IndexError on every EEF rollout, so turning
# diagnostics on was enough to kill a run that was otherwise fine.
_GRIPPER_IDX = {"eef7": 6, "droid8": 7, "joint9": 7}.get(EVAL_ACTION_SPACE, 7)
if EVAL_ACTION_SPACE not in ("droid8", "joint9", "eef7"):
    raise ValueError(
        f"AXIS_EVAL_ACTION_SPACE must be droid8, joint9 or eef7, got {EVAL_ACTION_SPACE!r}")
from axis_qvla.bench.sim.layouts import LayoutSpec
from axis_qvla.bench.sim.obs_cameras import compile_with_cameras, set_camera_near_plane
# `_build_obs` has always CALLED this and never imported it, so every dualsim rollout died on its
# first observation with `NameError: name 'observation_for_obs_key' is not defined`. It is the
# declared-profile geometry transform -- NOT resize_with_pad, which differs for a non-droid
# profile -- and rollout_render/rollout_physics both import it from here.
from axis_qvla.bench.sim.obs_geometry import observation_for_obs_key
from axis_qvla.bench.sim.render_scene import build_render_scene
from axis_qvla.bench.sim.rollout_render import (
    CAMERA_TO_OBS_KEY,
    RENDER_H,
    RENDER_W,
    RolloutResult,
    _save_frame,
    _tracked_body_names,
)

CheckerFactory = Callable[[dict | None], Callable[[mujoco.MjModel, mujoco.MjData], bool]]

# qpos width per mujoco joint type: FREE=7, BALL=4, SLIDE=1, HINGE=1.
_JNT_QPOS_WIDTH = {
    mujoco.mjtJoint.mjJNT_FREE: 7,
    mujoco.mjtJoint.mjJNT_BALL: 4,
    mujoco.mjtJoint.mjJNT_SLIDE: 1,
    mujoco.mjtJoint.mjJNT_HINGE: 1,
}


def _joint_qpos_map(model: mujoco.MjModel) -> dict[str, tuple[int, int]]:
    """joint name -> (qpos start address, width) for every joint in the model."""
    out: dict[str, tuple[int, int]] = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        adr = int(model.jnt_qposadr[j])
        width = _JNT_QPOS_WIDTH[mujoco.mjtJoint(model.jnt_type[j])]
        out[name] = (adr, width)
    return out


def _build_state_sync(
    phys_model: mujoco.MjModel, render_model: mujoco.MjModel
) -> list[tuple[int, int, int]]:
    """Per-joint copy plan (phys_adr, render_adr, width) for joints shared by both models.

    Copying by name rather than assuming identical qpos ordering is defensive: the two
    builds are verified to share every joint name, but free-joint ordering between the two
    compilers is not guaranteed, and a silent misalignment would sync garbage.
    """
    phys = _joint_qpos_map(phys_model)
    ren = _joint_qpos_map(render_model)
    plan: list[tuple[int, int, int]] = []
    for name, (r_adr, r_w) in ren.items():
        if name not in phys:
            continue
        p_adr, p_w = phys[name]
        w = min(p_w, r_w)
        plan.append((p_adr, r_adr, w))
    return plan


def _free_joint_z(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    """world-Z of every FREE joint (i.e. every movable object), keyed by joint name.

    Reads qpos directly (pos is the first 3 of a free joint), so it needs no body lookup
    and works identically in the physics and render models. This is the number that proves
    'objects hold' vs 'objects free-fell to -49540'.
    """
    out: dict[str, float] = {}
    for j in range(model.njnt):
        if mujoco.mjtJoint(model.jnt_type[j]) != mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        adr = int(model.jnt_qposadr[j])
        out[name] = float(data.qpos[adr + 2])
    return out


def _camera_info(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> dict:
    """World pose (post-forward), fovy, and effective near plane of a named camera.

    The near plane is `model.vis.map.znear * model.stat.extent` -- the number that decides
    whether the wrist view culls everything within ~20 cm (black during a grasp) or renders
    the fingers framing the object. Reported so the dual-sim proves both obs cameras sit at
    the pi05-calibrated pose and the wrist view is not clipped.
    """
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if cid < 0:
        return {"present": False}
    return {
        "present": True,
        "world_pos": [float(x) for x in data.cam_xpos[cid]],
        "fovy": float(model.cam_fovy[cid]),
        "near_plane_m": float(model.vis.map.znear * model.stat.extent),
        "extent_m": float(model.stat.extent),
    }


def _sync_render_to_physics(
    phys_data: mujoco.MjData,
    render_model: mujoco.MjModel,
    render_data: mujoco.MjData,
    sync_plan: list[tuple[int, int, int]],
    qpos_offset: np.ndarray,
) -> None:
    """Copy physics joint state into the render model, then mj_forward (kinematics only).

    NEVER mj_step the render model: its objects have contype=0/conaffinity=0, so a step
    would free-fall them. mj_forward propagates the copied qpos into xpos/cameras without
    integrating dynamics. `qpos_offset` is the render scene's object-raise (non-zero only
    at free-joint z), which lifts objects onto the table exactly as RenderScene.reset does.
    """
    for p_adr, r_adr, w in sync_plan:
        render_data.qpos[r_adr : r_adr + w] = phys_data.qpos[p_adr : p_adr + w]
    render_data.qpos[:] = np.asarray(render_data.qpos) + qpos_offset
    mujoco.mj_forward(render_model, render_data)


def _dump_obs_images(obs: dict, trace_path: Path, tag: str) -> dict:
    """Save the EXACT obs images the policy sees (base + wrist, 224 letterboxed) as PNGs and
    return their mean pixel. A near-black wrist mean (< ~15) is the signature of near-plane
    culling; a photorealistic base view is bright and textured (mean well above the ~30-60 a
    gray-checkerboard MuJoCo frame produces)."""
    stats: dict = {}
    try:
        import imageio.v2 as imageio

        for key, short in (("base_0_rgb", "base"), ("left_wrist_0_rgb", "wrist")):
            img = np.asarray(obs[key], dtype=np.uint8)
            imageio.imwrite(trace_path.with_name(f"{trace_path.stem}_{tag}_{short}.png"), img)
            stats[short] = {"mean_px": float(img.mean()), "std_px": float(img.std())}
    except Exception:  # noqa: BLE001
        pass
    return stats


def _build_obs(render_scene, phys_data, phys_ctrl, *, open_w, closed_w, prompt) -> dict:
    """Images from the RENDER scene (photorealistic), state from the PHYSICS sim (the real
    joint positions). qpos is synced, so either model reports the same arm angles; reading
    state from the physics controller keeps the state consistent with what actually stepped.
    """
    obs: dict = {}
    for cam, key in CAMERA_TO_OBS_KEY.items():
        frame = render_scene.render(cam, width=RENDER_W, height=RENDER_H)
        # Geometry comes from the declared profile, NOT resize_with_pad: the corpus and the
        # harness must apply the identical transform, and for a non-droid profile they differ.
        obs[key] = np.asarray(observation_for_obs_key(frame, key), dtype=np.uint8)
    if EVAL_ACTION_SPACE == "eef7":
        state = encode_state_eef(phys_data, phys_ctrl)
    elif EVAL_ACTION_SPACE == "joint9":
        state = encode_state9(phys_data, phys_ctrl)
    else:
        state = encode_state8(phys_data, phys_ctrl, open_w=open_w, closed_w=closed_w)
    obs["state"] = np.asarray(state, dtype=np.float32)
    obs["prompt"] = prompt
    return obs


def _build_physics_scene(
    payload: dict,
    variant_index: int,
    initial_state: dict | None,
    work_dir: Path,
    *,
    asset_dir: str | None,
    asset_fetch: str,
    asset_base_url: str | None,
    asset_lock: str | None,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Materialise + compile the BARE physics twin of the render scene's variant.

    Uses the SAME variant index and the SAME (layout-applied) initial_state the render
    scene resolved, so the two scenes describe the same arrangement. No object z-raise
    here: this is the ground-plane physics scene where objects rest and hold.
    """
    variants = payload.get("scene_variants") or []
    if 0 <= variant_index < len(variants):
        mjcf_xml = variants[variant_index].get("mjcf_xml") or payload.get("mjcf_xml")
    else:
        mjcf_xml = payload.get("mjcf_xml")
    if not mjcf_xml:
        raise ValueError("no mjcf_xml for physics scene")

    scene_dir = fresh_scene_dir(work_dir)   # never a previous build's files (asset_guard)
    store = make_asset_store(asset_dir, fetch=asset_fetch, base_url=asset_base_url, lock=asset_lock)
    try:
        resolution = materialize_scene(mjcf_xml, str(scene_dir), store)
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()
    raise_if_refused(store)
    if resolution.missing_files:
        raise RuntimeError(f"missing physics assets: {resolution.missing_files[:3]}")

    cwd = os.getcwd()
    os.chdir(scene_dir)  # MJCF asset paths are relative to the scene dir
    try:
        model, data = compile_with_cameras(resolution.entry_xml_path)
        restore_scene_state(model, data, initial_state, None, None)
        mujoco.mj_forward(model, data)
    finally:
        os.chdir(cwd)
    return model, data


def rollout_pi05_dualsim(
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
    asset_dir: str | None = None,
    asset_fetch: str = "refuse",
    asset_base_url: str | None = None,
    asset_lock: str | None = None,
) -> RolloutResult:
    """One dual-sim rollout: STEP the physics scene, RENDER the render scene, sync each step.

    `eval_layout` is applied to BOTH scenes (the render builder applies it; the physics
    build mirrors the render scene's resolved variant + initial_state), so their initial
    arrangements are identical before the first step.

    `trace_path`, when set (DIAG=1), writes a photorealistic frame0/frameN PNG, an mp4 of
    the base-camera obs, and a JSON of object world-Z from BOTH sims at step 0 and end --
    the evidence that objects hold and the pixels are photorealistic.
    """
    scene = None
    work_dir = Path(work_dir)
    try:
        try:
            # BACKGROUND-PERTURBATION HOOK. Unset -> the training-default table texture,
            # byte-identical behavior. Set (a PNG path; MuJoCo rejects JPEG) -> the eval table is
            # repainted with a texture the policy never trained on. Same declared-not-inferred
            # pattern as AXIS_EVAL_IMAGE_GEOMETRY / AXIS_EVAL_ACTION_SPACE.
            import os as _os
            _tex = _os.environ.get("AXIS_EVAL_TABLE_TEXTURE")
            _tex_kw = {"table_texture": _tex} if _tex else {}
            scene = build_render_scene(
                task_dir,
                init_state_seed=init_state_seed,
                task_id=task_id,
                task_name=task_name,
                render_root=render_root,
                work_dir=work_dir / "render",
                global_seed=0,
                eval_layout=eval_layout,
                asset_dir=asset_dir,
                asset_fetch=asset_fetch,
                asset_base_url=asset_base_url,
                asset_lock=asset_lock,
                **_tex_kw,
            )
        except AssetFetchRefused:
            # A refused scene asset (missing, or off ASSETS.lock.json) is a broken eval input,
            # not a policy outcome: abort the run instead of writing an error seed.
            raise
        except Exception as exc:  # noqa: BLE001
            # Name the TYPE, not just the message. Several exceptions that reach here have an
            # empty str() -- StopIteration is the one that actually happened -- and the bare
            # `error:build_render:` it produced carried no information at all about a failure
            # that had already cost a full 20-seed allocation.
            return RolloutResult(
                False, 0, 0.0, reason=f"error:build_render:{type(exc).__name__}: {exc}"
            )

        render_model, render_data = scene.model, scene.data

        # WRIST NEAR-PLANE: render_scene_pi05 never pulls the near clipping plane in, so with
        # the scene's large stat.extent the wrist camera would cull everything within ~20 cm
        # and render BLACK during exactly the grasp phase that carries the most signal.
        # compile_with_cameras applies this to the physics model; the render model (the one we
        # actually take obs from) needs the identical treatment. Global (all cameras), so it
        # also fixes any close-range base-view clipping.
        set_camera_near_plane(render_model)

        # Build the physics twin from the render scene's resolved variant + initial_state,
        # so the two scenes are guaranteed to describe the same arrangement.
        payload = json.loads((Path(task_dir) / "task.json").read_text())
        try:
            phys_model, phys_data = _build_physics_scene(
                payload,
                scene.variant_index,
                scene.initial_state,
                work_dir / "physics",
                asset_dir=asset_dir,
                asset_fetch=asset_fetch, asset_lock=asset_lock,
                asset_base_url=asset_base_url,
            )
        except AssetFetchRefused:
            raise
        except Exception as exc:  # noqa: BLE001
            return RolloutResult(False, 0, 0.0, reason=f"error:build_physics:{exc}")

        sync_plan = _build_state_sync(phys_model, render_model)
        qpos_offset = scene.qpos_offset

        # The checker reads the PHYSICS state (real collision); it is built from the physics
        # scene's restored initial_state.
        checker_fn = checker_factory(scene.initial_state)
        ctrl = Pi05ActionController.from_model(phys_model, fps=fps)
        ctrl.action_space = EVAL_ACTION_SPACE
        ctrl.reset(phys_data)
        client.reset()

        # Per-rollout sampler noise. Without it the server draws from a per-process key stream
        # that advances once per request, so a rollout's actions depend on how many calls
        # preceded it -- which makes a reused/shared server silently order-dependent. Refuse
        # rather than fall back: a stale server would produce those numbers under this run's name.
        srv_meta = client.get_server_metadata() or {}
        if not srv_meta.get("supports_client_noise"):
            raise RuntimeError(
                "policy server does not advertise supports_client_noise; update serve_policy.py"
            )
        noise_h = int(srv_meta["action_horizon"])
        noise_d = int(srv_meta["action_dim"])
        infer_call_index = 0

        # --- diagnostics setup -------------------------------------------------
        diag_dir = None
        video_frames: list | None = None
        z_log: dict = {}
        if trace_path is not None:
            trace_path = Path(trace_path)
            diag_dir = trace_path.parent
            diag_dir.mkdir(parents=True, exist_ok=True)
            video_frames = []
            _sync_render_to_physics(phys_data, render_model, render_data, sync_plan, qpos_offset)
            _save_frame(scene, trace_path.with_name(f"{trace_path.stem}_frame0.png"))
            obs0 = _build_obs(scene, phys_data, ctrl, open_w=open_w, closed_w=closed_w, prompt=prompt)
            z_log["step0"] = {
                "physics": _free_joint_z(phys_model, phys_data),
                "render": _free_joint_z(render_model, render_data),
                "obs_images": _dump_obs_images(obs0, trace_path, "frame0"),
            }
            z_log["joint_layout_aligned"] = bool(
                phys_model.nq == render_model.nq
                and len(sync_plan) == render_model.njnt == phys_model.njnt
            )
            z_log["phys_nq"] = int(phys_model.nq)
            z_log["render_nq"] = int(render_model.nq)
            z_log["n_synced_joints"] = len(sync_plan)
            # Camera pose parity + wrist near-plane evidence. The render obs cameras must sit
            # at the pi05-calibrated pose (frontview ~[0.95,0,0.95] fovy 58; wrist link8 mount
            # fovy 52) and NOT cull the close-range wrist view.
            z_log["render_obs_cameras"] = {
                c: _camera_info(render_model, render_data, c) for c in ("frontview", "wrist")
            }
            z_log["physics_pi05_cameras"] = {
                c: _camera_info(phys_model, phys_data, c)
                for c in ("pi05/base_cam", "pi05/wrist_cam")
            }

        plan: collections.deque = collections.deque()
        # DIAG gripper telemetry: does the POLICY ever command a close (action[7]>=0.5),
        # do the fingers actually close, and does the target object move? Isolates
        # "never commands grasp" vs "commands but mis-positioned" vs physics.
        grip_log = {"a7": [], "fw": [], "objz": []} if trace_path is not None else None
        success, reason, step = False, "timeout", 0
        for step in range(max_steps):
            if not plan:
                _sync_render_to_physics(phys_data, render_model, render_data, sync_plan, qpos_offset)
                obs = _build_obs(scene, phys_data, ctrl, open_w=open_w, closed_w=closed_w, prompt=prompt)
                if video_frames is not None:
                    video_frames.append(obs["base_0_rgb"])
                chunk_noise = noise_for(task_id, init_state_seed, infer_call_index,
                                        noise_h, noise_d)
                chunk = np.asarray(client.infer(obs, noise=chunk_noise)["actions"])
                if _PROV and infer_call_index < 4:
                    import hashlib as _h
                    _sha = lambda x: _h.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]
                    print(
                        f"PROV call={infer_call_index} "
                        f"base={_sha(obs['base_0_rgb'])} wrist={_sha(obs['left_wrist_0_rgb'])} "
                        f"state={_sha(np.asarray(obs['state'], dtype=np.float32))} "
                        f"noise={_sha(chunk_noise)} chunk={_sha(chunk.astype(np.float32))}",
                        flush=True,
                    )
                infer_call_index += 1
                # The chunk width is the ACTION SPACE's width, not a constant: eef7 returns 7
                # (dpos, daxis-angle, gripper), droid8 8, joint9 9. Hardcoding >=8 rejected every
                # EEF chunk as (16, 7) with n_steps=0 -- the guard firing correctly on a stale
                # assumption rather than on bad data.
                if chunk.ndim != 2 or chunk.shape[1] < _MIN_ACTION_WIDTH:
                    return RolloutResult(False, step, float(phys_data.time), reason=f"error:bad_chunk:{chunk.shape}")
                plan.extend(chunk[:effective_replan(replan)])
            action = plan.popleft()
            ctrl.apply(phys_model, phys_data, action)  # THE real sim step
            if grip_log is not None and step % 5 == 0:
                grip_log["a7"].append(float(np.asarray(action).reshape(-1)[_GRIPPER_IDX]))
                grip_log["fw"].append(
                    float(0.5 * (phys_data.qpos[ctrl.finger_qadr[0]] + phys_data.qpos[ctrl.finger_qadr[1]]))
                )
                grip_log["objz"].append(float(_free_joint_z(phys_model, phys_data).get("hot_dog_2_free", float("nan"))))
            if checker_fn(phys_model, phys_data):
                success, reason = True, "success"
                break

        if trace_path is not None:
            _sync_render_to_physics(phys_data, render_model, render_data, sync_plan, qpos_offset)
            _save_frame(scene, trace_path.with_name(f"{trace_path.stem}_frameN.png"))
            obsN = _build_obs(scene, phys_data, ctrl, open_w=open_w, closed_w=closed_w, prompt=prompt)
            z_log["stepN"] = {
                "step": step,
                "physics": _free_joint_z(phys_model, phys_data),
                "render": _free_joint_z(render_model, render_data),
                "obs_images": _dump_obs_images(obsN, trace_path, "frameN"),
            }
            z_log["result"] = {"success": success, "reason": reason, "n_steps": step + 1}
            if grip_log is not None and grip_log["a7"]:
                a7 = np.asarray(grip_log["a7"]); fw = np.asarray(grip_log["fw"]); oz = np.asarray(grip_log["objz"])
                z_log["grip_summary"] = {
                    "a7_max": float(a7.max()), "a7_p90": float(np.percentile(a7, 90)),
                    "a7_frac_ge_0.5": float((a7 >= 0.5).mean()),
                    "finger_width_min": float(fw.min()),   # ~0.04 open, ~0.0 fully closed
                    "objz_start": float(oz[0]) if len(oz) else None, "objz_max": float(np.nanmax(oz)),
                }
            trace_path.with_name(f"{trace_path.stem}_objz.json").write_text(json.dumps(z_log, indent=2))
            if video_frames:
                try:
                    import imageio.v2 as imageio

                    imageio.mimwrite(
                        str(trace_path.with_name(f"{trace_path.stem}.mp4")),
                        video_frames,
                        fps=max(int(fps / replan), 1),
                    )
                except Exception:  # noqa: BLE001
                    pass

        return RolloutResult(success, step + 1, float(phys_data.time), reason=reason)
    finally:
        if scene is not None:
            scene.close()
