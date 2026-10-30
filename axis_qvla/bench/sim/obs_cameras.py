"""Inject DROID-style base + wrist cameras into a Franka scene.

pi0.5 (pi05_droid) consumes an *exterior* third-person view and a
*wrist* view. AXIS-Replay scenes carry no named cameras (video render
uses a free camera), and the MJCF is materialised per task with an
``<include>`` of the shared Franka body. Rather than editing the shared
include (which the diffusion baseline also loads), we add the two
cameras at compile time via ``mujoco.MjSpec`` so the change is scoped to
the pi0.5 pipeline only.

Both cameras are *fixed* in their parent frame:
  * ``pi05/wrist_cam`` is parented to the hand body and looks down the
    gripper approach axis (hand +z, toward the fingers).
  * ``pi05/base_cam`` sits in the world and looks at the workspace in
    front of the robot base.

The base camera's pose and both fovys are NOT hand-tuned: they are derived
from the rendered dataset's own per-frame calibration
(``observation.cameras.<cam>.{intrinsics,camera_to_world}``) by a
one-off calibration script (``camera_calib``), which also holds the
convention proof.

WHY DERIVED RATHER THAN CHOSEN: training frames come from the
AXIS/robosuite render pipeline and eval frames from our MuJoCo rollout,
so camera identity has to be established, not assumed. pi0.5's SigLIP
tower is fully trainable under our freeze regex and adapts hard to the training
views, making it especially sensitive to a view mismatch: a wrong camera
would show up as a low success rate that looks like a bad policy.
"""

from __future__ import annotations

import numpy as np
import mujoco

WRIST_CAM = "pi05/wrist_cam"
BASE_CAM = "pi05/base_cam"

# --- Derived from the dataset, not chosen. ---------------------------------
# Derived against `render/out_eef/task_1889`, the pretraining data this benchmark evaluates.
# These constants reach the PHYSICS model (`compile_with_cameras`, live for
# `rollout_physics.py`); dual-sim renders its images from the render scene's own cameras.
# The world offset is zero: MuJoCo FK on franka/panda_hand against `observation.ee_state`
# over 48 frames gives mean and std [0, 0, 0].
_WORLD_OFFSET = np.array([0.0, 0.0, 0.0])

# third_person is static across the corpus (1 distinct camera_to_world), so its world
# pose transfers directly once the (now zero) offset is applied.
_BASE_POS = tuple((np.array([1.008599997, 0.0, 1.190400004]) + _WORLD_OFFSET).tolist())
_BASE_XYAXES = (0.0, 1.0, 0.0, -0.628267467, 0.0, 0.777997434)
_BASE_FOVY = 45.0  # 2*atan((360/2)/434.5584411621094) = 45.000000

# The wrist camera is arm-mounted, so its pose is stored as a mount transform in the
# hand frame, recovered by camera_calib.solve_wrist_mount().
#
# CORRECTNESS CHECK, not a diagnostic: a rigid mount must give the same transform at
# every arm pose. The derived value is constant to max std 3.17e-08 over 48 frames
# (the threshold for "meaningful" is 1e-4). It lands on [0.05, 0, 0] on the hand -- the
# mount the pretraining data was rendered with.
_WRIST_FOVY = 75.0  # 2*atan((360/2)/234.58056640625) = 75.000000
_WRIST_POS = (0.05, 0.0, 0.0)
_WRIST_XYAXES = (0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
_WRIST_MOUNT_IS_CALIBRATED = True

# Re-derive before trusting these on a corpus other than the held-out 20: a scene that
# places the robot base at a different height changes the offset.
_WORLD_OFFSET_CALIBRATED_ON_TASK = 1889


# MuJoCo's near clipping plane is vis.map.znear * model.stat.extent. These scenes
# have extent = 20 m and the default znear = 0.01, giving a 0.2 m near plane -- so a
# wrist camera culls EVERYTHING within 20 cm. During a grasp it sits ~5 cm from the
# object, so the wrist view renders black exactly when it carries the most signal,
# while the base view looks perfect throughout. Measured on task 1424 frame 300:
# mean pixel 5.11 (black) at the default vs 119.54 with a 1 cm near plane, where the
# gripper fingers frame the object just as they do in the training frames.
_NEAR_PLANE_M = 0.01


def set_camera_near_plane(model, near_m: float = _NEAR_PLANE_M):
    """Pull the near clipping plane in so close-range wrist views are not culled.

    Must be called on the COMPILED model: the stored value is a fraction of
    ``model.stat.extent``, which is only known after compilation. ``compile_with_cameras``
    does this for you; call it yourself if you compile the spec directly.
    """
    model.vis.map.znear = float(near_m) / float(model.stat.extent)
    return model


def _find_body(spec: mujoco.MjSpec, name: str):
    for b in spec.bodies:
        if b.name == name:
            return b
    return None


def add_pi05_cameras(
    spec: mujoco.MjSpec,
    *,
    hand_body: str = "franka/panda_hand",
) -> mujoco.MjSpec:
    """Add base + wrist cameras to ``spec`` in place; returns ``spec``."""
    world = spec.worldbody
    bcam = world.add_camera()
    bcam.name = BASE_CAM
    bcam.pos = list(_BASE_POS)
    bcam.fovy = _BASE_FOVY
    bcam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    bcam.alt.xyaxes = list(_BASE_XYAXES)

    hand = _find_body(spec, hand_body)
    if hand is None:
        raise ValueError(f"hand body not found for wrist camera: {hand_body}")
    wcam = hand.add_camera()
    wcam.name = WRIST_CAM
    wcam.pos = list(_WRIST_POS)
    wcam.fovy = _WRIST_FOVY
    wcam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    wcam.alt.xyaxes = list(_WRIST_XYAXES)
    return spec


def compile_with_cameras(
    entry_xml_path: str,
    *,
    hand_body: str = "franka/panda_hand",
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load a materialised scene, add cameras, and compile."""
    spec = mujoco.MjSpec.from_file(str(entry_xml_path))
    add_pi05_cameras(spec, hand_body=hand_body)
    model = spec.compile()
    set_camera_near_plane(model)  # else the wrist view is black during every grasp
    data = mujoco.MjData(model)
    return model, data
