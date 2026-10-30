"""The pinned parameters of the kinematic renderer that are not the workspace's: the robot, the gripper
mount, the wrist camera, the light, the IK, the gates, the frame size and rate.

Render-tier package, but this module is stdlib only: the gates, the writer schema and the FR3 checks
read it without MuJoCo. The workspace -- exterior camera, table, background -- is a profile
(axis_qvla.realrobot.render.profile). The robot, gripper mount, wrist camera, light and IK values reproduce
the published REALALIGN render (read from its metadata, taken from the renderer that made it, or measured
on its episodes); PROFILE_ID names this renderer, and the gate thresholds and the QP solver are this
repository's.
"""
from __future__ import annotations

import dataclasses
import math

PROFILE_ID = "qvla_realalign_kinematic_v1"
FPS = 15
WIDTH, HEIGHT = 640, 360
# The DROID-8D training view (droid8d.py): its state and action layouts, and the clip on the joint
# velocities (rad/s). The FR3 line's checks hold the realalign_sim embodiment to these.
DROID8D_STATE = ("q0", "q1", "q2", "q3", "q4", "q5", "q6", "closedness")
DROID8D_ACTION = ("dq0", "dq1", "dq2", "dq3", "dq4", "dq5", "dq6", "closedness")
VELOCITY_CLIP = 1.0
SOURCE_ROW_DT = 0.2        # the nominal source period (5 Hz rows); a step-count recording's opening
                           # copy of row 0 sits this far before it
KNOT_ROW_OFFSET = 2        # the frame of the first recorded row; post-step row k is frame 3k + 2
# The first knot of every rendered episode (the published render's observation.state row 0).
PIXELMATCH_HOME = (0.7687, -0.5876, -0.4589, -1.7823, -0.2579, 1.2274, 2.6505)
# The IK's posture target and seed centre: the Franka's default joint positions.
POSTURE_HOME = (0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398)

SOURCE_PREFIX = "franka/"
ARM_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))
SOURCE_FINGER_JOINTS = ("panda_finger_joint1", "panda_finger_joint2")
SOURCE_HAND_BODY = "panda_hand"
SOURCE_BASE_BODY = "panda_link0"
EE_BODY = "panda_link7"
SOURCE_TCP_IN_HAND = (0.0, 0.0, 0.1084)            # the source TCP: 0.1084 m along the hand's z

GRIPPER_PREFIX = "robotiq/"
GRIPPER_BASE_BODY = "base"
GRIPPER_FINGER_BODIES = ("right_pad", "left_pad")  # their separation is the gripper opening
GRIPPER_TCP_SITE = "pinch"                         # the gripper's TCP
GRIPPER_ACTUATOR = "fingers_actuator"              # ctrl 0 (open) .. 255 (closed)
GRIPPER_SETTLE_STEPS = 1500                        # steps the linkage settles at each end of the ctrl range
GRIPPER_OPEN_SEPARATION_M = 0.09832364010417671     # the pad separation at ctrl 0
GRIPPER_CLOSED_SEPARATION_M = 0.014909658471763475  # the pad separation at ctrl 255
GRIPPER_SEPARATION_TOL_M = 1e-4


@dataclasses.dataclass(frozen=True)
class Pose:
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]  # wxyz


# panda_link7 -> 2F-85 `base`: the published ee_state is FK of the published joints through it (residual < 1e-6).
GRIPPER_BASE_IN_EE = Pose((0.0, 0.0, 0.1178), (0.38268343236508984, 0.0, 0.0, -0.9238795325112867))


@dataclasses.dataclass(frozen=True)
class IkParams:
    """The IK's constants (ik.py)."""

    position_cost: float = 1.0
    orientation_cost: float = 0.6     # arms with >= 7 joints
    lm_damping: float = 5e-2
    posture_cost: float = 1e-2
    velocity_limit: float = 4.0       # rad/s, arm joints
    dt: float = 0.02
    iterations: int = 24
    converge_tol: float = 1.5e-3      # m
    solver_damping: float = 1e-3
    seeds_first: int = 8              # draws consumed at knot 0
    seeds_rescue: int = 4             # half of seeds_first
    rescue_tol: float = 5e-3          # m: a warm start worse than this is rescued
    reach_tol: float = 0.02           # m
    seed_std: float = 0.8
    rng_seed: int = 0
    collision_margin: float = 0.01
    min_reach_pct: float = 60.0       # fewer rows reached: IkError, which stops the render command
    solver: str = "quadprog"          # the QP solver render.lock.txt installs


IK = IkParams()
TCP_GATE_M = 0.003          # convergence threshold: the first knot this close to the source TCP converges
CONVERGE_WITHIN_KNOTS = 8   # the convergence knot must come no later than this


@dataclasses.dataclass(frozen=True)
class Camera:
    name: str                          # MuJoCo camera name
    key: str                           # LeRobot key: observation.images.<key>
    body: str | None                   # None = world-fixed; else the unprefixed gripper body it rides on
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]   # wxyz, MuJoCo's order
    fovy: float


WRIST = Camera("wrist", "wrist", GRIPPER_BASE_BODY, (-0.074, 0.031, 0.011),
               (0.1138238760221829, 0.7041526740254301, -0.6921340038864419, -0.11028897304012754),
               52.3933414386577)        # the DROID wrist camera's mount on the 2F-85 base
EXTERIOR_NAME, EXTERIOR_KEY = "frontview", "third_person"   # the MuJoCo name and LeRobot key of the profile's camera
CAMERA_KEYS = (EXTERIOR_KEY, WRIST.key)


def focal(cam: Camera, height: int = HEIGHT) -> float:
    return (height / 2.0) / math.tan(math.radians(cam.fovy) / 2.0)


def intrinsics(cam: Camera, width: int = WIDTH, height: int = HEIGHT) -> tuple[float, ...]:
    """Row-major 3x3 pinhole matrix: fx = fy from fovy, principal point at the image centre."""
    f = focal(cam, height)
    return (f, 0.0, (width - 1) / 2.0, 0.0, f, (height - 1) / 2.0, 0.0, 0.0, 1.0)


@dataclasses.dataclass(frozen=True)
class Light:
    pos: tuple[float, float, float]
    dir: tuple[float, float, float]
    ambient: tuple[float, float, float]
    diffuse: tuple[float, float, float]
    specular: tuple[float, float, float]


# One directional light, no shadow; every other light disabled.
LIGHT = Light((1.0, 1.0, 1.5), (-0.19245008972987523, -0.19245008972987523, -0.9622504486493761),
              (0.55, 0.55, 0.55), (0.12, 0.12, 0.12), (0.0, 0.0, 0.0))
HEADLIGHT = Light((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.18, 0.18, 0.18), (0.03, 0.03, 0.03), (0.0, 0.0, 0.0))


def quat_mul(a, b) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def quat_to_mat(q) -> tuple[tuple[float, float, float], ...]:
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))


def mat_to_quat(R) -> tuple[float, float, float, float]:
    """Shepperd's method; w >= 0 when the trace is positive."""
    (r00, r01, r02), (r10, r11, r12), (r20, r21, r22) = R
    tr = r00 + r11 + r22
    if tr > 0:
        s = 2.0 * math.sqrt(1.0 + tr)
        return (0.25 * s, (r21 - r12) / s, (r02 - r20) / s, (r10 - r01) / s)
    if r00 > r11 and r00 > r22:
        s = 2.0 * math.sqrt(1.0 + r00 - r11 - r22)
        return ((r21 - r12) / s, 0.25 * s, (r01 + r10) / s, (r02 + r20) / s)
    if r11 > r22:
        s = 2.0 * math.sqrt(1.0 + r11 - r00 - r22)
        return ((r02 - r20) / s, (r01 + r10) / s, 0.25 * s, (r12 + r21) / s)
    s = 2.0 * math.sqrt(1.0 + r22 - r00 - r11)
    return ((r10 - r01) / s, (r02 + r20) / s, (r12 + r21) / s, 0.25 * s)


def lookat_quat(eye, target, up=(0.0, 0.0, 1.0)) -> tuple[float, float, float, float]:
    """A MuJoCo/OpenGL camera (x right, y up, -z forward) at `eye` looking at `target`."""
    f = [t - e for t, e in zip(target, eye)]
    n = math.sqrt(sum(c * c for c in f))
    f = [c / n for c in f]
    r = [f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0]]
    n = math.sqrt(sum(c * c for c in r))
    r = [c / n for c in r]
    u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]]
    R = ((r[0], u[0], -f[0]), (r[1], u[1], -f[1]), (r[2], u[2], -f[2]))
    return mat_to_quat(R)
