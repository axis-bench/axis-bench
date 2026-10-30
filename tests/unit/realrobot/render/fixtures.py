"""A recorded attempt small enough for CI: a Panda-named 7-joint arm with a hand, one free box, and a
2F-85-shaped fixture gripper (same body, site and actuator names as Menagerie's; slide fingers).

The arm's body frames are the recorded scenes' Panda frames (scene.xml of every recorded task), so FK,
the hand frame and the TCP offset behave as on real data; the geoms are primitives.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from axis_qvla.realrobot.render import source

JOINT_RANGES = ((-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973), (-3.0718, -0.0698),
                (-2.8973, 2.8973), (-0.0175, 3.7525), (-2.8973, 2.8973))
_CHAIN = (("panda_link1", 'pos="0 0 0.333"'), ("panda_link2", 'quat="1 -1 0 0"'),
          ("panda_link3", 'pos="0 -0.316 0" quat="1 1 0 0"'), ("panda_link4", 'pos="0.0825 0 0" quat="1 1 0 0"'),
          ("panda_link5", 'pos="-0.0825 0.384 0" quat="1 -1 0 0"'), ("panda_link6", 'quat="1 1 0 0"'),
          ("panda_link7", 'pos="0.088 0 0" quat="1 1 0 0"'))


def _arm_xml() -> str:
    hand = ('<body name="franka/panda_hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">'
            '<geom type="box" size="0.02 0.08 0.03"/>'
            '<body name="franka/panda_leftfinger" pos="0 0 0.0584">'
            '<joint name="franka/panda_finger_joint1" type="slide" axis="0 1 0" range="0 0.04"/>'
            '<geom type="box" size="0.01 0.01 0.02"/></body>'
            '<body name="franka/panda_rightfinger" pos="0 0 0.0584" quat="0 0 0 1">'
            '<joint name="franka/panda_finger_joint2" type="slide" axis="0 1 0" range="0 0.04"/>'
            '<geom type="box" size="0.01 0.01 0.02"/></body></body>')
    inner = hand
    for i in range(7, 0, -1):
        name, attrs = _CHAIN[i - 1]
        lo, hi = JOINT_RANGES[i - 1]
        inner = (f'<body name="franka/{name}" {attrs}><joint name="franka/panda_joint{i}" range="{lo} {hi}"/>'
                 f'<geom/>{inner}</body>')
    return f'<body name="franka/"><body name="franka/panda_link0" childclass="franka/panda"><geom/>{inner}</body></body>'


SCENE_XML = f"""<mujoco model="fixture_source">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <default><default class="franka/panda">
    <joint axis="0 0 1" damping="1" armature="0.1"/>
    <geom type="capsule" size="0.04 0.05" contype="0" conaffinity="0" rgba="0.9 0.9 0.9 1"/>
  </default></default>
  <worldbody>
    <light name="franka/top" pos="0 0 2"/>
    <camera name="camera0" pos="1.5 -1.5 1.5" xyaxes="0.707 0.707 0 -0.408 0.408 0.816"/>
    <geom name="floor" type="plane" size="5 5 0.01"/>
    {_arm_xml()}
    <body name="box_1" pos="0.5 0 0.03"><freejoint name="box_1_free"/>
      <geom type="box" size="0.03 0.03 0.03" rgba="0.8 0.2 0.2 1"/></body>
  </worldbody>
  <tendon><fixed name="franka/split"><joint joint="franka/panda_finger_joint1" coef="0.5"/>
    <joint joint="franka/panda_finger_joint2" coef="0.5"/></fixed></tendon>
  <actuator>
    {''.join(f'<position name="franka/actuator{i}" joint="franka/panda_joint{i}" kp="100"/>' for i in range(1, 8))}
    <position name="franka/actuator8" tendon="franka/split" kp="100" ctrlrange="0 0.04"/>
  </actuator>
</mujoco>
"""

# Menagerie-shaped: base_mount -> base (-90 deg about z), pads closing along +-y of `base`, the pinch
# site 0.145 m along the approach axis, one actuator 0 (open) .. 255 (closed).
GRIPPER_XML = """<mujoco model="fixture_2f85">
  <worldbody>
    <body name="base_mount" pos="0 0 0.007">
      <body name="base" pos="0 0 0.0038" quat="1 0 0 -1">
        <geom type="cylinder" size="0.04 0.03" rgba="0.2 0.2 0.2 1" contype="0" conaffinity="0"/>
        <site name="pinch" pos="0 0 0.145" size="0.005"/>
        <body name="right_driver" pos="0 0 0.1">
          <joint name="right_driver_joint" type="slide" axis="0 -1 0" range="0 0.04" damping="5" armature="0.01"/>
          <body name="right_pad" pos="0 0.049 0.03"><geom type="box" size="0.005 0.005 0.01" contype="0" conaffinity="0"/></body>
        </body>
        <body name="left_driver" pos="0 0 0.1">
          <joint name="left_driver_joint" type="slide" axis="0 1 0" range="0 0.04" damping="5" armature="0.01"/>
          <body name="left_pad" pos="0 -0.049 0.03"><geom type="box" size="0.005 0.005 0.01" contype="0" conaffinity="0"/></body>
        </body>
      </body>
    </body>
  </worldbody>
  <equality><joint joint1="left_driver_joint" joint2="right_driver_joint"/></equality>
  <actuator>
    <general name="fingers_actuator" joint="right_driver_joint" ctrlrange="0 255"
             gainprm="0.015686274509803921" biasprm="0 -100 -10" biastype="affine"/>
  </actuator>
</mujoco>
"""
OPEN_SEP, CLOSED_SEP = 0.098, 0.018      # the fixture's pad separations at ctrl 0 and 255
SOURCE_HOME = (0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398)
REACH = (0.3, -0.35, 0.1, -2.05, 0.05, 1.75, 0.95)


def write_scene(d: Path) -> Path:
    Path(d).mkdir(parents=True, exist_ok=True)
    (Path(d) / "scene.xml").write_text(SCENE_XML)
    return Path(d) / "scene.xml"


def write_gripper(d: Path) -> Path:
    Path(d).mkdir(parents=True, exist_ok=True)
    (Path(d) / "2f85.xml").write_text(GRIPPER_XML)
    return Path(d) / "2f85.xml"


def fixture_packed(scene_xml: Path, n: int = 12) -> source.PackedState:
    """n source rows at 5 Hz: the arm eases from the source home to REACH, the fingers close at row n/2."""
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    adr = lambda name: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)])
    arm = [adr(f"franka/panda_joint{i}") for i in range(1, 8)]
    fingers = [adr("franka/panda_finger_joint1"), adr("franka/panda_finger_joint2")]
    s = np.linspace(0.0, 1.0, n)
    s = s * s * (3 - 2 * s)
    qpos = np.tile(m.qpos0, (n, 1))
    qpos[:, arm] = np.asarray(SOURCE_HOME) + s[:, None] * (np.asarray(REACH) - np.asarray(SOURCE_HOME))
    qpos[:, fingers] = np.where(np.arange(n)[:, None] < n // 2, 0.04, 0.02)
    times = np.arange(n) * 0.2
    qvel = np.zeros((n, m.nv))
    qvel[:, :7] = np.gradient(qpos[:, arm], times, axis=0)
    ctrl = np.zeros((n, m.nu))
    ctrl[:, :7] = qpos[:, arm]
    ctrl[:, 7] = qpos[:, fingers[0]]
    q0 = qpos[0].copy()
    q0[arm] = SOURCE_HOME
    return source.PackedState(0.001, 200, n, m.nq, m.nv, m.nu, 0.0, q0, np.zeros(m.nv), ctrl[0].copy(),
                              qpos, qvel, ctrl)


def write_episode(d: Path, n: int = 12, task_id: int = 9001, attempt_id: int = 7001) -> Path:
    scene = write_scene(Path(d))
    order = source.joint_names(mujoco.MjModel.from_xml_path(str(scene)))
    doc = source.source_episode_doc(task_id=task_id, attempt_id=attempt_id, prompt="push the red box",
                                    scene_xml="scene.xml",
                                    trajectory_data=source.encode_packed_state_v1(fixture_packed(scene, n),
                                                                                  task_id=task_id),
                                    joint_order=order)
    path = Path(d) / "source_episode.json"
    source.write_source_episode(path, doc)
    return path


def write_trace(path: Path, scene_xml: Path, n: int = 12, order=None, times=None) -> Path:
    """The survey's trace format: one `state` row per packed row, joint_qpos keyed by name (in `order`,
    default the scene's), then a `final` row. Row k's `sim_time` is `times[k]`, default (k + 1) * 0.2."""
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    p = fixture_packed(scene_xml, n)
    names = order or source.joint_names(m)
    sim = [float(t) for t in times] if times is not None else [(k + 1) * 0.2 for k in range(n)]
    rows = []
    for k in range(n):
        q = {}
        for name in names:
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            a, w = int(m.jnt_qposadr[j]), (7 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1)
            v = p.qpos[k, a:a + w].tolist()
            q[name] = v if w > 1 else v[0]
        rows.append({"event": "state", "source": "packed_state_v1_direct", "step_index": k,
                     "sim_time": sim[k], "joint_qpos": q})
    rows.append({"event": "final", "final_sim_time": sim[-1], "total_state_frames": n})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return Path(path)


def attempt_curves(n_frames: int, task_id: int = 9001, attempt_id: int = 7001) -> dict:
    """An attempt-keyed phase curve on the render frame base: rows before the first anchor invalid."""
    from axis_qvla.realrobot.realign.curves import PhaseCurve
    curve = np.clip(np.linspace(-0.1, 1.05, n_frames), 0.0, 1.0).astype(np.float32)
    valid = np.ones(n_frames, bool)
    valid[:2] = False
    return {(task_id, attempt_id): PhaseCurve(curve, valid, n_frames // 2)}
