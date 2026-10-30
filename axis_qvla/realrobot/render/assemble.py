"""The render model: the recorded scene, its arm kept, its hand swapped for a Robotiq 2F-85, on a table.

Render tier (mujoco, numpy). Built with MjSpec on the pinned MuJoCo.

What changes from the recorded scene, and why:
- the source hand subtree and every actuator, tendon, equality, exclude, pair, keyframe, light and
  camera go, and so does the floor plane: the published render disables every unmanaged light and
  draws its own arena;
- every top-level body, geom and site is lifted by the profile's table height: the published world is
  the recorded one on the table, and free-joint object poses are lifted the same amount when copied;
- the 2F-85 is attached to panda_link7 through a frame chosen so its `base` body lands exactly at
  rig.GRIPPER_BASE_IN_EE (measured on the published render), whatever the gripper file's own offsets;
- the workspace profile's arena (table, its fixtures, the rear wall, the floor) with per-attempt
  colour and material jitter under the material caps, one directional light without shadow, the
  headlight, and the two cameras.
The IK model (objects=False) keeps only the robot.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import mujoco
import numpy as np

from axis_qvla.realrobot.render import rig
from axis_qvla.realrobot.render.profile import Surface, WorkspaceProfile, load_profile


class AssemblyError(ValueError):
    """A recorded scene or gripper the renderer cannot assemble."""


@dataclasses.dataclass(frozen=True)
class TargetScene:
    model: mujoco.MjModel
    arm_qpos: np.ndarray
    arm_dof: np.ndarray
    gripper_qpos: np.ndarray
    gripper_dof: np.ndarray
    gripper_actuator: int
    ee_body: int
    base_body: int
    pinch_site: int
    finger_bodies: tuple[int, int]
    shared: tuple[tuple[int, int, int, int, int, int, bool], ...]  # src_q, tgt_q, nq, src_d, tgt_d, nv, free
    cameras: dict[str, int]
    camera_specs: tuple[rig.Camera, ...]


def _id(model: mujoco.MjModel, kind, name: str) -> int:
    i = mujoco.mj_name2id(model, kind, name)
    if i < 0:
        raise AssemblyError(f"{name!r} is not in the model")
    return int(i)


def mount_frame(gripper_xml: Path) -> tuple[list[float], list[float]]:
    """The frame on panda_link7 that puts the gripper's `base` body at rig.GRIPPER_BASE_IN_EE."""
    gm = mujoco.MjModel.from_xml_path(str(gripper_xml))
    gd = mujoco.MjData(gm)
    mujoco.mj_kinematics(gm, gd)
    b = _id(gm, mujoco.mjtObj.mjOBJ_BODY, rig.GRIPPER_BASE_BODY)
    p_g, q_g = gd.xpos[b].copy(), gd.xquat[b].copy()
    inv, q_f, r = np.zeros(4), np.zeros(4), np.zeros(3)
    mujoco.mju_negQuat(inv, q_g)
    mujoco.mju_mulQuat(q_f, np.asarray(rig.GRIPPER_BASE_IN_EE.quat, np.float64), inv)
    mujoco.mju_rotVecQuat(r, p_g, q_f)
    return (np.asarray(rig.GRIPPER_BASE_IN_EE.pos, np.float64) - r).tolist(), q_f.tolist()


def _strip(spec: mujoco.MjSpec, *, objects: bool) -> None:
    hand = spec.body(rig.SOURCE_PREFIX + rig.SOURCE_HAND_BODY)
    if hand is None:
        raise AssemblyError(f"the recorded scene has no {rig.SOURCE_PREFIX + rig.SOURCE_HAND_BODY!r}")
    spec.delete(hand)
    for group in (spec.actuators, spec.tendons, spec.equalities, spec.excludes, spec.pairs, spec.keys,
                  spec.lights, spec.cameras):
        for element in list(group):
            spec.delete(element)
    for g in list(spec.worldbody.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_PLANE:
            spec.delete(g)
    if not objects:
        dropped: set[str] = set()
        for b in list(spec.worldbody.bodies):
            if not b.name.startswith(rig.SOURCE_PREFIX):
                dropped |= {g.meshname for g in b.find_all(mujoco.mjtObj.mjOBJ_GEOM) if g.meshname}
                spec.delete(b)
        # Their meshes go with them: MuJoCo 3.3.7's compiler can crash on a mesh no geom uses when its
        # inertia is "convex" (task 1338, variant 23). Nothing of the IK model reads them.
        used = {g.meshname for g in spec.geoms}
        for m in list(spec.meshes):
            if m.name in dropped and m.name not in used:
                spec.delete(m)


def _lift(spec: mujoco.MjSpec, height: float) -> None:
    for element in (*spec.worldbody.bodies, *spec.worldbody.geoms, *spec.worldbody.sites):
        p = list(element.pos)
        p[2] += height
        element.pos = p


def _material(spec: mujoco.MjSpec, name: str, s: Surface, rng: np.random.Generator) -> None:
    shift = float(rng.uniform(-s.rgb_jitter, s.rgb_jitter))
    jit = rng.uniform(-s.material_jitter, s.material_jitter, 3)
    refl, shin, spec_ = np.minimum(np.clip(np.asarray(s.material) + jit, 0.0, None), s.material_caps)
    m = spec.add_material()
    m.name = f"qvla_{name}"
    m.rgba = [*np.clip(np.asarray(s.rgb) + shift, 0.0, 1.0).tolist(), 1.0]
    m.reflectance, m.shininess, m.specular, m.emission = float(refl), float(shin), float(spec_), float(s.emission)
    if s.texture is not None:
        t = spec.add_texture()
        t.name, t.type, t.file = f"qvla_{name}", mujoco.mjtTexture.mjTEXTURE_2D, str(s.texture)
        m.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = t.name
        m.texrepeat = list(s.texture_repeat)


def _materials(spec: mujoco.MjSpec, profile: WorkspaceProfile, key: tuple[int, int]) -> None:
    rng = np.random.default_rng([int(key[0]), int(key[1]), 20261019])
    for name, s in profile.surfaces():
        _material(spec, name, s, rng)
    for name, rgba in profile.materials.items():
        m = spec.add_material()
        m.name, m.rgba = f"qvla_{name}", list(rgba)


def _box(spec: mujoco.MjSpec, name: str, pos, half, material: str) -> None:
    g = spec.worldbody.add_geom()
    g.name, g.type, g.pos, g.size, g.material = name, mujoco.mjtGeom.mjGEOM_BOX, list(pos), list(half), material
    g.contype, g.conaffinity, g.group = 0, 0, 1


def _arena(spec: mujoco.MjSpec, profile: WorkspaceProfile, key: tuple[int, int]) -> None:
    _materials(spec, profile, key)
    t, bg = profile.table, profile.background
    top = t.height
    cx, cy = t.center
    sx, sy, sz = t.size
    rear = cx - sx / 2
    _box(spec, "qvla_table", (cx, cy, top - sz / 2), (sx / 2, sy / 2, sz / 2), "qvla_table")
    for f in profile.fixtures:
        x, y, z = f.pos
        _box(spec, f"qvla_{f.name}", (x, y, top + z), f.half, f"qvla_{f.material}")
    wall_x = rear - bg.wall_distance
    _box(spec, "qvla_rear_wall", (wall_x - 0.01, cy, bg.wall_height / 2),
         (0.01, bg.wall_half_width, bg.wall_height / 2), "qvla_wall")
    f = spec.worldbody.add_geom()
    f.name, f.type, f.size, f.material = "qvla_floor", mujoco.mjtGeom.mjGEOM_PLANE, [10.0, 10.0, 0.05], "qvla_floor"
    f.contype, f.conaffinity, f.group = 0, 0, 1


def _light_and_cameras(spec: mujoco.MjSpec, cameras: tuple[rig.Camera, ...]) -> None:
    light = spec.worldbody.add_light()
    light.name, light.type, light.castshadow = "qvla_light", mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, False
    light.pos, light.dir = list(rig.LIGHT.pos), list(rig.LIGHT.dir)
    light.ambient, light.diffuse, light.specular = list(rig.LIGHT.ambient), list(rig.LIGHT.diffuse), list(rig.LIGHT.specular)
    head = spec.visual.headlight
    head.ambient, head.diffuse, head.specular = (list(rig.HEADLIGHT.ambient), list(rig.HEADLIGHT.diffuse),
                                                 list(rig.HEADLIGHT.specular))
    head.active = 1
    spec.visual.global_.offwidth, spec.visual.global_.offheight = rig.WIDTH, rig.HEIGHT
    for cam in cameras:
        parent = spec.worldbody if cam.body is None else spec.body(rig.GRIPPER_PREFIX + cam.body)
        c = parent.add_camera()
        c.name, c.pos, c.quat, c.fovy = cam.name, list(cam.pos), list(cam.quat), cam.fovy


def _width(model: mujoco.MjModel, j: int) -> tuple[int, int]:
    t = int(model.jnt_type[j])
    if t == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7, 6
    if t == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4, 3
    return 1, 1


def _index(model: mujoco.MjModel, source: mujoco.MjModel, cameras: tuple[rig.Camera, ...]) -> TargetScene:
    J, B, S, A = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_SITE, mujoco.mjtObj.mjOBJ_ACTUATOR
    arm = [_id(model, J, rig.SOURCE_PREFIX + n) for n in rig.ARM_JOINTS]
    grip = [j for j in range(model.njnt) if (mujoco.mj_id2name(model, J, j) or "").startswith(rig.GRIPPER_PREFIX)]
    shared = []
    for j in range(model.njnt):
        if j in arm or j in grip:
            continue
        name = mujoco.mj_id2name(model, J, j) or ""
        sj = mujoco.mj_name2id(source, J, name)
        if sj < 0:
            raise AssemblyError(f"joint {name!r} is not in the recorded scene")
        nq, nv = _width(model, j)
        shared.append((int(source.jnt_qposadr[sj]), int(model.jnt_qposadr[j]), nq, int(source.jnt_dofadr[sj]),
                       int(model.jnt_dofadr[j]), nv, int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)))
    pinch = _id(model, S, rig.GRIPPER_PREFIX + rig.GRIPPER_TCP_SITE)
    base = _id(model, B, rig.GRIPPER_PREFIX + rig.GRIPPER_BASE_BODY)
    if int(model.site_bodyid[pinch]) != base:
        raise AssemblyError("the gripper's pinch site must ride on its base body")
    return TargetScene(
        model=model,
        arm_qpos=np.array([model.jnt_qposadr[j] for j in arm]), arm_dof=np.array([model.jnt_dofadr[j] for j in arm]),
        gripper_qpos=np.array([model.jnt_qposadr[j] for j in grip]),
        gripper_dof=np.array([model.jnt_dofadr[j] for j in grip]),
        gripper_actuator=_id(model, A, rig.GRIPPER_PREFIX + rig.GRIPPER_ACTUATOR),
        ee_body=_id(model, B, rig.SOURCE_PREFIX + rig.EE_BODY), base_body=base, pinch_site=pinch,
        finger_bodies=tuple(_id(model, B, rig.GRIPPER_PREFIX + f) for f in rig.GRIPPER_FINGER_BODIES),
        shared=tuple(shared),
        cameras={c.key: _id(model, mujoco.mjtObj.mjOBJ_CAMERA, c.name) for c in cameras},
        camera_specs=tuple(cameras))


def build_target_scene(scene_xml: Path, gripper_xml: Path, *, objects: bool = True,
                       arena: bool | None = None, key: tuple[int, int] = (0, 0),
                       profile: WorkspaceProfile | None = None) -> TargetScene:
    """The render model (objects=True) or the IK model (objects=False) of `profile`'s workspace (the
    reference workspace when None). `key` = (task_id, attempt_id) draws the surface colours. `arena`
    builds the profile's table, fixtures, wall and floor; default is `objects`, so the IK model skips
    it, but the overlay check (robot only, no recorded objects, arena still wanted) passes it explicitly."""
    profile = profile or load_profile()
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    _strip(spec, objects=objects)
    _lift(spec, profile.table.height)
    link7 = spec.body(rig.SOURCE_PREFIX + rig.EE_BODY)
    if link7 is None:
        raise AssemblyError(f"the recorded scene has no {rig.SOURCE_PREFIX + rig.EE_BODY!r}")
    frame = link7.add_frame()
    frame.pos, frame.quat = mount_frame(gripper_xml)
    spec.copy_during_attach = True
    spec.attach(mujoco.MjSpec.from_file(str(gripper_xml)), prefix=rig.GRIPPER_PREFIX, frame=frame)
    if objects if arena is None else arena:
        _arena(spec, profile, key)
    _light_and_cameras(spec, profile.cameras)
    model = spec.compile()
    return _index(model, mujoco.MjModel.from_xml_path(str(scene_xml)), profile.cameras)
