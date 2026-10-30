"""Map between pi0.5's 8-D DROID action/state space and the MuJoCo sim.

Isolated from the rollout loop so the highest-risk conversion is unit-testable and
validatable by open-loop replay before any server is involved. Runs in the offline environment.

THE 8-D SPACE (produced by axis_qvla/data/convert_droid_actions.py):
  action[0:7] = arm joint VELOCITY, rad/s, for franka/panda_joint1..7
                (converter: (state[t+1] - state[t]) * fps)
  action[7]   = gripper closedness in [0,1], 1 = closed
                (converter: (open_w - width) / (open_w - closed_w), clipped)
  state[0:7]  = arm joint positions, rad
  state[7]    = gripper closedness in [0,1] (same encoding as the action's gripper)

WHY VELOCITY INTEGRATION WITH A PERSISTENT COMMAND (highest-risk line):
  The arm actuators are affine-bias position servos, so a velocity action must be
  integrated into a position target: q_cmd += v * dt. That running command MUST persist
  across control steps. Re-reading data.qpos each step (what the 5-D rollout does) instead
  of accumulating the command gives a ~90x larger tracking error (measured mean 0.25 vs
  0.003 rad) because the servo lags and each step would re-anchor to the lagged position.

  The integration is exact by construction: the converter set v = (s[t+1]-s[t])*fps, so
  q_cmd += v*(1/fps) = q_cmd + (s[t+1]-s[t]). Restoring to s[0] and replaying the recorded
  actions must therefore reproduce the recorded joint trajectory -- which is exactly what
  the open-loop replay validation checks.

GRIPPER IS BINARY, NOT CALIBRATED-LINEAR:
  The dataset's finger "width" (open_w ~ 0.047) is in robosuite units and does NOT match
  the MuJoCo finger actuator range ([0, 0.04]); there is no recovered mapping between them.
  So closedness is thresholded: closed -> finger ctrl 0.0, open -> 0.04. For pick-and-place
  the gripper is functionally binary anyway. The open-loop replay logs finger tracking so
  this stays visible, but the arm (units verified) is what the pass/fail threshold checks.
"""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

# The end-effector frame the render records as `observation.ee_state`. Identified by MEASUREMENT,
# not by name: setting the arm to a demo's recorded joints and searching every body/site, this frame
# reproduces the recorded position to 0.00000 m (panda_link8 is coincident and equivalent).
EEF_BODY = "franka/panda_hand"
# `eef_pose`/`state_eef` are expressed in the TABLE frame, which the scene shell lifts by this much.
# Measured identical (0.48) across all 20 held-out tasks in the render metadata, and confirmed as the
# exact mean offset between the corpus's ee_state and eef_pose columns.
EEF_TABLE_Z_OFFSET = 0.48

ARM_JOINTS = tuple(f"franka/panda_joint{i}" for i in range(1, 8))
FINGER_JOINTS = ("franka/panda_finger_joint1", "franka/panda_finger_joint2")
ARM_ACTUATORS = ARM_JOINTS  # actuator names equal joint names in this model
FINGER_ACTUATORS = FINGER_JOINTS

FINGER_OPEN_CTRL = 0.04   # ctrlrange max -> gripper open
FINGER_CLOSE_CTRL = 0.0   # ctrlrange min -> gripper closed
CLOSEDNESS_THRESHOLD = 0.5

# The rate `axis_qvla.data.convert_droid_actions` uses to differentiate joint positions into
# velocities, and therefore the ONLY rate at which this controller's integration inverts it.
# It is the pretraining side's stated contract too ("action = 7 joint velocities rad/s, fps 15,
# forward difference zeroed at each episode's last frame"), so it is written down on both sides
# rather than surviving as two independently-chosen defaults that happen to match today.
CONVERTER_FPS = 15.0

# Damped least-squares IK for the EEF action space. Damping keeps the step finite near singularities.
EEF_IK_DAMPING = 1e-4

# Iterating the IK on a SCRATCH MjData, opt-in via AXIS_EVAL_EEF_IK=iterate.
#
# It tracks the commanded pose more closely than the single damped step, which stays the default.
# Open-loop replay accuracy is a controller-correctness check, not a proxy for success: tracking the
# commanded delta more faithfully also tracks the policy's errors more faithfully. The iteration
# runs on a scratch MjData, so the live state the physics steps from is never written.
# robosuite `controllers/parts/arm/ik.py` constants -- the reference that, unlike OSC, also outputs
# JOINT POSITIONS, so its formulation is directly adoptable by our position-servo path.
# Selected with AXIS_EVAL_EEF_IK=robosuite.
IK_RS_DAMPING = 0.05 ** 2                 # ik.py:220 -- 25x tighter than our 1e-4
IK_RS_NULLSPACE_KP = np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])   # ik.py:219
IK_RS_MAX_ANGVEL = 1.0                    # ik.py:192, rad/s cap on the joint VELOCITY
IK_RS_RELAX = 0.95                        # ik.py:221-222, chase 95% of the residual
# ik.py:228. Load-bearing: dq is a VELOCITY there -- the residual is divided by this and the result
# multiplied back by it, so it cancels for the TASK term but NOT for the nullspace term. It is what
# sets their relative weight. Dropping it made the nullspace bias 10x too strong, saturated the
# clamp, and scaled the task motion down with it: replay tracking went 0.018 -> 0.243 m.
IK_RS_INTEGRATION_DT = 0.1

EEF_IK_ITERS = 8
EEF_IK_POS_TOL = 1e-4        # metres; stop early once the residual is below the recording's noise


def corpus_fps(corpus_root) -> float | None:
    """The fps a LeRobot corpus DECLARES, or None if it does not say.

    Stronger than comparing two code defaults: this asks the data what rate it was written at.
    Returns None rather than guessing -- a corpus with no declared fps is a fact the caller has to
    decide about, not one this function should paper over with 15.0.
    """
    import json
    from pathlib import Path

    info = Path(corpus_root) / "meta" / "info.json"
    try:
        d = json.loads(info.read_text())
    except OSError:
        # Not swallowed into None-means-absent: an unreadable directory is not an absent one.
        raise
    v = d.get("fps")
    return float(v) if v is not None else None


def assert_corpus_fps(corpus_root, fps: float = CONVERTER_FPS) -> float:
    """Refuse to proceed when the corpus was written at a different rate than we will replay it.

    The failure this prevents is silent: the joint velocities are simply scaled, the rollout runs,
    the arm moves at the wrong speed and times out, and every guard stays green.
    """
    declared = corpus_fps(corpus_root)
    if declared is None:
        raise ValueError(
            f"{corpus_root} declares no fps in meta/info.json, so the droid8 velocity contract "
            f"cannot be verified. Convert it with axis_qvla.data.convert_droid_actions, which writes "
            f"the rate it used."
        )
    if abs(declared - float(fps)) > 1e-9:
        raise ValueError(
            f"{corpus_root} was written at fps={declared} but the eval replays at {fps}; every "
            f"joint velocity would be off by {float(fps) / declared:.4g}x."
        )
    return declared


def _iterated_ik_requested() -> bool:
    """`AXIS_EVAL_EEF_IK=iterate` selects the scratch-MjData solve. Default keeps the single step,
    so no existing eval changes meaning until the A/B says it should."""
    import os

    return _ik_mode() == "iterate"


def _ik_mode() -> str:
    """`AXIS_EVAL_EEF_IK`: `single` (default), `iterate`, or `robosuite`.

    `robosuite` mirrors `controllers/parts/arm/ik.py` -- robosuite's OTHER arm controller, and
    the one that outputs joint positions like ours (OSC outputs torques, which our position
    actuators cannot consume). It keeps the single step but adds what ours omits:
    a NULLSPACE posture bias, 25x tighter damping, a per-step joint clamp, and 0.95 under-relaxation.

    WHY THE NULLSPACE TERM. A 7-DoF arm is redundant for a 6-D task; our minimum-norm solve leaves
    the null direction unconstrained, so the elbow random-walks. Replaying recorded actions for the
    eval's full 1500-step horizon, `q_cmd` hits a joint limit on 28-39% of steps and the Jacobian's
    smallest singular value collapses from 0.2247 to 1e-4..1e-3.
    """
    import os

    return os.environ.get("AXIS_EVAL_EEF_IK", "single").strip().lower()


def _axis_angle_to_mat(v: np.ndarray) -> np.ndarray:
    """Rodrigues: axis-angle (3,) -> rotation matrix."""
    v = np.asarray(v, dtype=np.float64)
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def _mat_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle (3,), shortest arc."""
    R = np.asarray(R, dtype=np.float64)
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    th = float(np.arccos(c))
    if th < 1e-12:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2.0 * np.sin(th)) * th


def _ids(model: mujoco.MjModel, names, obj) -> list[int]:
    out = []
    for n in names:
        i = mujoco.mj_name2id(model, obj, n)
        if i < 0:
            raise ValueError(f"missing {obj!r} {n!r} in model")
        out.append(i)
    return out


@dataclasses.dataclass
class Pi05ActionController:
    """Applies pi0.5 8-D actions to the sim; holds the persistent joint command.

    Build once per rollout with :meth:`from_model`, call :meth:`reset` after the scene
    is restored (to seed the command from the actual start pose), then :meth:`apply` per
    action.
    """

    arm_qadr: np.ndarray       # qpos addresses of the 7 arm joints
    finger_qadr: np.ndarray    # qpos addresses of the 2 finger joints
    arm_ctrl: np.ndarray       # ctrl indices of the 7 arm actuators
    finger_ctrl: np.ndarray    # ctrl indices of the 2 finger actuators
    jnt_lo: np.ndarray         # arm joint lower limits (rad)
    jnt_hi: np.ndarray         # arm joint upper limits (rad)
    fps: float
    n_substeps: int
    q_cmd: np.ndarray | None = None
    arm_dofadr: np.ndarray | None = None   # dof addresses of the 7 arm joints (for the Jacobian)
    eef_bid: int = -1                      # body id of EEF_BODY
    _ik_scratch: object = None             # private MjData for the iterated IK solve
    _ik_home: object = None                # posture target for the robosuite nullspace term
    # "eef7"    -- 7-D relative EEF: [dpos(3), daxis-angle(3), gripper], the space STAGE-1
    #              PRETRAINING uses (state_eef/action_eef). Resolved to joint targets by IK.
    # "droid8"  -- 7 joint VELOCITIES + closedness, integrated (the original DROID-8D space).
    # "joint9"  -- 7 absolute joint POSITIONS + 2 finger widths, commanded directly.
    #
    # WHY joint9 EXISTS. The stage-1 pretraining corpus stores 9-D joint POSITIONS plus two finger
    # widths, and so does the held-out benchmark's own render. Evaluating such a model through the
    # velocity path made it command ~-1.9 rad/s on joint 4 (its mean joint POSITION), which drives
    # the arm into its stops: the policy acts for the whole episode and never completes. Converting
    # the fine-tune data to 8-D instead would work, but it forces the model to re-learn the action
    # space and so discards part of the pretraining the benchmark exists to measure.
    #
    # Unlike the DROID case there is no unit mismatch here: these finger widths come from the same
    # robosuite/MuJoCo sim as the eval ([0, 0.04]), so they are commanded as widths directly rather
    # than thresholded through a closedness encoding.
    action_space: str = "droid8"

    @classmethod
    def from_model(
        cls, model: mujoco.MjModel, *, fps: float = 15.0, allow_fps_mismatch: bool = False
    ) -> "Pi05ActionController":
        # FPS IS HALF OF A CONTRACT, AND THE OTHER HALF LIVES IN THE CORPUS.
        # The converter wrote v = (s[t+1] - s[t]) * fps; this controller replays q_cmd += v * (1/fps).
        # Those cancel EXACTLY, and only exactly, when the two fps agree. Disagree by a factor k and
        # every joint velocity is silently scaled by k -- no error, no log line, a policy that
        # simply moves at the wrong speed and times out. Nothing else in the pipeline checks this,
        # which is why it is checked at construction rather than trusted to two matching defaults.
        if not allow_fps_mismatch and abs(float(fps) - CONVERTER_FPS) > 1e-9:
            raise ValueError(
                f"controller fps={fps} but the droid8 pretraining data was converted at {CONVERTER_FPS} "
                f"(axis_qvla.data.convert_droid_actions --fps). Every joint velocity would be "
                f"rescaled by {float(fps) / CONVERTER_FPS:.4g}x. Re-convert the pretraining data at this fps "
                f"or run the eval at {CONVERTER_FPS}; pass allow_fps_mismatch=True only if you "
                f"have verified the pretraining data really was written at this rate."
            )
        arm_jids = _ids(model, ARM_JOINTS, mujoco.mjtObj.mjOBJ_JOINT)
        finger_jids = _ids(model, FINGER_JOINTS, mujoco.mjtObj.mjOBJ_JOINT)
        arm_qadr = np.array([model.jnt_qposadr[j] for j in arm_jids])
        finger_qadr = np.array([model.jnt_qposadr[j] for j in finger_jids])
        arm_ctrl = np.array(_ids(model, ARM_ACTUATORS, mujoco.mjtObj.mjOBJ_ACTUATOR))
        finger_ctrl = np.array(_ids(model, FINGER_ACTUATORS, mujoco.mjtObj.mjOBJ_ACTUATOR))

        # Arm joints 1/3/5/7 have ctrllimited=0 (ctrlrange [0,0]), so the actuator
        # imposes NO clip -- we must clip the command against the JOINT range ourselves,
        # or an out-of-range velocity command drives the servo past the physical stop.
        jr = np.array([model.jnt_range[j] for j in arm_jids], dtype=np.float64)
        n_sub = int(round((1.0 / fps) / model.opt.timestep))
        arm_dofadr = np.array([model.jnt_dofadr[j] for j in arm_jids], dtype=int)
        eef_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EEF_BODY)
        if eef_bid < 0:
            raise ValueError(f"{EEF_BODY!r} not in this model; the EEF action space needs it")
        return cls(
            arm_dofadr=arm_dofadr, eef_bid=eef_bid,
            arm_qadr=arm_qadr, finger_qadr=finger_qadr,
            arm_ctrl=arm_ctrl, finger_ctrl=finger_ctrl,
            jnt_lo=jr[:, 0], jnt_hi=jr[:, 1], fps=fps, n_substeps=n_sub,
        )

    def reset(self, data: mujoco.MjData) -> None:
        """Seed the persistent command from the current (restored) joint positions."""
        self.q_cmd = data.qpos[self.arm_qadr].astype(np.float64).copy()
        data.ctrl[self.arm_ctrl] = self.q_cmd
        data.ctrl[self.finger_ctrl] = data.qpos[self.finger_qadr]

    def apply(self, model: mujoco.MjModel, data: mujoco.MjData, action: np.ndarray) -> None:
        """Integrate one 8-D action and step the sim ``n_substeps`` times."""
        if self.q_cmd is None:
            raise RuntimeError("call reset() after restoring the scene, before apply()")
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        need = {"joint9": 9, "eef7": 7}.get(self.action_space, 8)
        if a.shape[0] < need:
            raise ValueError(
                f"expected a {need}-D action for action_space={self.action_space!r}, "
                f"got shape {a.shape}. A 9-D joint-position action fed to the 8-D velocity path "
                f"commands joint POSITIONS as velocities and drives the arm into its stops."
            )
        if self.action_space == "eef7":
            # [dpos(3, world), daxis-angle(3, world), gripper_target]. Resolved to joint targets by
            # damped least-squares IK on the EEF body Jacobian -- the scene is a bare MjModel, not a
            # robosuite robot with a controller attached, so robosuite's OSC cannot be used directly.
            # Damping keeps the step finite near singularities instead of producing a huge dq.
            # ONE damped least-squares step per control step, against the commanded delta.
            #
            # Iterating needs a scratch MjData (AXIS_EVAL_EEF_IK=iterate above); writing data.qpos
            # between iterations would move the arm the physics steps from.
            if _ik_mode() == "robosuite":
                self.q_cmd = self._solve_eef_ik_robosuite(model, data, a[:6])
            elif _iterated_ik_requested():
                self.q_cmd = self._solve_eef_ik(model, data, a[:6])
            else:
                jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
                mujoco.mj_jacBody(model, data, jacp, jacr, self.eef_bid)
                J = np.vstack([jacp[:, self.arm_dofadr], jacr[:, self.arm_dofadr]])   # 6 x 7
                twist = np.concatenate([a[:3], a[3:6]])
                dq = J.T @ np.linalg.solve(J @ J.T + EEF_IK_DAMPING * np.eye(6), twist)
                self.q_cmd = np.clip(self.q_cmd + dq, self.jnt_lo, self.jnt_hi)
            data.ctrl[self.arm_ctrl] = self.q_cmd
            grip = FINGER_CLOSE_CTRL if float(a[6]) >= CLOSEDNESS_THRESHOLD else FINGER_OPEN_CTRL
            data.ctrl[self.finger_ctrl] = grip
        elif self.action_space == "joint9":
            # Absolute joint-position command; no integration, so there is no persistent-command
            # subtlety here. Still clipped to the joint range: arm joints 1/3/5/7 have
            # ctrllimited=0, so the actuator imposes no clip of its own.
            self.q_cmd = np.clip(a[:7], self.jnt_lo, self.jnt_hi)
            data.ctrl[self.arm_ctrl] = self.q_cmd
            # Finger widths in sim units, clipped to the actuator's own range.
            data.ctrl[self.finger_ctrl] = np.clip(a[7:9], FINGER_CLOSE_CTRL, FINGER_OPEN_CTRL)
        else:
            dt = 1.0 / self.fps
            # Integrate arm velocity into the persistent position command, clip to joint range.
            self.q_cmd = np.clip(self.q_cmd + a[:7] * dt, self.jnt_lo, self.jnt_hi)
            data.ctrl[self.arm_ctrl] = self.q_cmd
            # Binary gripper: closedness >= 0.5 -> close.
            grip = FINGER_CLOSE_CTRL if float(a[7]) >= CLOSEDNESS_THRESHOLD else FINGER_OPEN_CTRL
            data.ctrl[self.finger_ctrl] = grip
        for _ in range(self.n_substeps):
            mujoco.mj_step(model, data)

    def _solve_eef_ik_robosuite(self, model: mujoco.MjModel, data: mujoco.MjData,
                                delta: np.ndarray) -> np.ndarray:
        """One damped step with a nullspace posture bias, per robosuite `ik.py`.

        Differences from `_solve_eef_ik`'s minimum-norm step, all taken from the reference:
        damping 0.05^2 rather than 1e-4, a posture term projected through `I - pinv(J) @ J` that
        pulls the redundant DoF back toward the configuration captured at reset, a 0.1 rad per-step
        joint clamp, and 0.95 under-relaxation of the residual.
        """
        if self._ik_home is None:
            self._ik_home = np.asarray(data.qpos[self.arm_qadr], dtype=np.float64).copy()
        jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, self.eef_bid)
        J = np.vstack([jacp[:, self.arm_dofadr], jacr[:, self.arm_dofadr]])      # 6 x 7
        # dq is a VELOCITY here, exactly as in ik.py: the residual is divided by integration_dt and
        # the solution multiplied back by it below.
        twist = IK_RS_RELAX * np.concatenate([delta[:3], delta[3:6]]) / IK_RS_INTEGRATION_DT
        dq = J.T @ np.linalg.solve(J @ J.T + IK_RS_DAMPING * np.eye(6), twist)
        # nullspace posture bias: projected through I - pinv(J) @ J, so it costs the Cartesian task
        # nothing while restraining the redundant DoF
        q = np.asarray(self.q_cmd, dtype=np.float64)
        null = np.eye(len(q)) - np.linalg.pinv(J) @ J
        dq = dq + null @ (IK_RS_NULLSPACE_KP * (self._ik_home - q))
        peak = float(np.abs(dq).max())
        if peak > IK_RS_MAX_ANGVEL:
            dq *= IK_RS_MAX_ANGVEL / peak
        return np.clip(q + dq * IK_RS_INTEGRATION_DT, self.jnt_lo, self.jnt_hi)

    def _solve_eef_ik(self, model: mujoco.MjModel, data: mujoco.MjData,
                      delta: np.ndarray) -> np.ndarray:
        """Joint command reaching the CURRENT eef pose plus ``delta``, solved on a scratch MjData.

        The target is formed from the live pose once; the iteration then runs entirely on a private
        copy, so the live `data` this function is handed is never written and the physics still
        steps from exactly the state it was in. That is the difference from the attempt that
        measured worse.
        """
        if self._ik_scratch is None:
            self._ik_scratch = mujoco.MjData(model)
        sd = self._ik_scratch

        # ANCHOR THE TARGET IN COMMAND SPACE, not on the live pose. Reading the current pose off
        # `data` while solving from `q_cmd` mixes two frames: the solve then has to cover the delta
        # PLUS the standing gap between where the arm is and where it was told to be, and that gap
        # is re-added every step. Measured: 0.183 m mean replay error against 0.018 m for the single
        # step. Anchoring both on q_cmd keeps this a pure command-space integrator, which is what
        # the single-step version is and why it tracks.
        q = np.asarray(self.q_cmd, dtype=np.float64).copy()
        sd.qpos[:] = data.qpos
        sd.qpos[self.arm_qadr] = q
        mujoco.mj_kinematics(model, sd)
        mujoco.mj_comPos(model, sd)
        p_cur = np.asarray(sd.xpos[self.eef_bid], dtype=np.float64).copy()
        R_cur = np.asarray(sd.xmat[self.eef_bid], dtype=np.float64).reshape(3, 3).copy()
        p_tgt = p_cur + np.asarray(delta[:3], dtype=np.float64)
        R_tgt = _axis_angle_to_mat(np.asarray(delta[3:6], dtype=np.float64)) @ R_cur
        jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
        for _ in range(EEF_IK_ITERS):
            sd.qpos[:] = data.qpos
            sd.qpos[self.arm_qadr] = q
            mujoco.mj_kinematics(model, sd)
            mujoco.mj_comPos(model, sd)
            p = np.asarray(sd.xpos[self.eef_bid], dtype=np.float64)
            R = np.asarray(sd.xmat[self.eef_bid], dtype=np.float64).reshape(3, 3)
            e_pos = p_tgt - p
            e_rot = _mat_to_axis_angle(R_tgt @ R.T)
            if np.linalg.norm(e_pos) < EEF_IK_POS_TOL and np.linalg.norm(e_rot) < 1e-3:
                break
            mujoco.mj_jacBody(model, sd, jacp, jacr, self.eef_bid)
            J = np.vstack([jacp[:, self.arm_dofadr], jacr[:, self.arm_dofadr]])
            twist = np.concatenate([e_pos, e_rot])
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + EEF_IK_DAMPING * np.eye(6), twist),
                        self.jnt_lo, self.jnt_hi)
        return q


def encode_state8(
    data: mujoco.MjData,
    ctrl: Pi05ActionController,
    *,
    open_w: float | None = None,
    closed_w: float | None = None,
) -> np.ndarray:
    """Build the 8-D observation the policy expects, from live sim state.

    state[0:7] = arm joint angles (rad); state[7] = gripper closedness in [0,1] (0 = open,
    1 = closed), matching the training-data encoding (linear, per convert_droid_actions).

    CRITICAL -- normalise against the SIM's finger geometry, NOT the dataset calibration:
    the training closedness was `(open_w - width)/(open_w - closed_w)` on ROBOSUITE widths
    (open_w ~ 0.047, closed_w ~ 0.012 for task 1644), but here `width` is a MuJoCo finger
    width whose scale is [0, 0.04] -- a different unit system with no recovered mapping (see
    the module docstring). Applying the robosuite (open_w, closed_w) to a MuJoCo width read
    a fully-open gripper (width 0.04) as ~0.20 closed instead of 0.0 and compressed the whole
    range -- a constant skew on the highest-variance state dim at every grasp. So closedness
    is normalised by the sim's own open/closed finger widths (FINGER_OPEN_CTRL/FINGER_CLOSE_CTRL,
    the same constants the binary action mapping tracks to), giving open -> 0, closed -> 1 to
    match how the training encoder mapped robosuite open/closed to the same endpoints.

    `open_w`/`closed_w` are accepted for call-compatibility and provenance logging only; they
    are the TRAINING-data (robosuite) calibration and MUST NOT drive the sim-side closedness.

    Float32, shape (8,). The caller MUST assert this shape: a wrong width passes silently
    through the whole server transform chain.
    """
    q = data.qpos[ctrl.arm_qadr].astype(np.float32)
    width = 0.5 * (data.qpos[ctrl.finger_qadr[0]] + data.qpos[ctrl.finger_qadr[1]])
    denom = FINGER_OPEN_CTRL - FINGER_CLOSE_CTRL
    c = 0.0 if denom == 0 else float(np.clip((FINGER_OPEN_CTRL - width) / denom, 0.0, 1.0))
    state8 = np.concatenate([q, np.float32([c])]).astype(np.float32)
    assert state8.shape == (8,), state8.shape
    return state8


def _raw_finger_state_requested() -> bool:
    """`AXIS_EVAL_GRIP_STATE=fingers` puts the two RAW finger joint positions in state dims 6-7.

    WHAT IT REPLACES, measured over 5,949 frames of the same episodes:

        closedness the policy TRAINS on   mean 0.408
        closedness it RECEIVES at eval    mean 0.138
        correlation 0.971, mean |difference| 0.270

    The corpus normalises by the ROBOSUITE finger range (open 0.0468 m, closed 0.0170 m); this
    module normalised by the SIM ctrlrange (0.0400 / 0.0000). Same grasp, different number -- a
    0.270 offset on a [0,1] channel, 1.14 sigma of that dimension's own std.

    The duplicate was also lossy: the two fingers correlate only 0.281, so writing one scalar into
    both slots discarded a real second measurement.

    Requires a corpus whose `state_eef[6:8]` holds raw finger metres (`out_eef_lib2`). Default is
    unchanged, so every existing checkpoint keeps the state form it was trained on.
    """
    import os

    return os.environ.get("AXIS_EVAL_GRIP_STATE", "closedness").strip().lower() == "fingers"


def encode_state_eef(data: mujoco.MjData, ctrl: Pi05ActionController,
                     *, table_z_offset: float = EEF_TABLE_Z_OFFSET) -> np.ndarray:
    """The 8-D `state_eef` observation stage-1 pretraining uses.

    [pos(3) in the TABLE frame, canonical axis-angle(3, w>=0), gripper, gripper] -- the layout
    `openpi.policies.eef_math.episode_eef_columns` produces, duplicated gripper included.

    TWO THINGS THAT MUST MATCH THE DATA OR THE POLICY IS FED OFF-DISTRIBUTION, SILENTLY:
      * the FRAME. `eef_pose` is table-relative; the sim reports world coordinates, so the scene's
        z-offset is subtracted. Skipping it puts every observation half a metre high.
      * the SIGN. The axis-angle is canonicalised to w>=0 (shortest arc), the same convention
        eef_math uses. A second implementation drifting by a sign trains fine and serves wrong.

    Gripper closedness is normalised by the SIM's finger range, not the dataset's robosuite widths
    -- the same correction :func:`encode_state8` documents.
    """
    xpos = np.asarray(data.xpos[ctrl.eef_bid], dtype=np.float64).copy()
    xpos[2] -= table_z_offset
    w, x, y, z = (float(v) for v in data.xquat[ctrl.eef_bid])   # mujoco stores wxyz
    if w < 0.0:                                                  # canonical: shortest arc
        w, x, y, z = -w, -x, -y, -z
    n = float(np.linalg.norm([x, y, z]))
    angle = 2.0 * float(np.arctan2(n, w))
    axis_angle = np.zeros(3) if n < 1e-12 else (np.array([x, y, z]) / n) * angle

    if _raw_finger_state_requested():
        # robosuite's `robot0_gripper_qpos`: the two finger joint positions themselves, in metres.
        # Storing the RAW measurement removes the mismatch by construction -- there is no derived
        # constant left for the corpus and the eval to disagree about. See
        # `_raw_finger_state_requested` for the 1.14-sigma offset this replaces.
        f0 = float(data.qpos[ctrl.finger_qadr[0]])
        f1 = float(data.qpos[ctrl.finger_qadr[1]])
        state = np.concatenate([xpos, axis_angle, [f0, f1]]).astype(np.float32)
        assert state.shape == (8,), state.shape
        return state

    width = 0.5 * (data.qpos[ctrl.finger_qadr[0]] + data.qpos[ctrl.finger_qadr[1]])
    denom = FINGER_OPEN_CTRL - FINGER_CLOSE_CTRL
    c = 0.0 if denom == 0 else float(np.clip((FINGER_OPEN_CTRL - width) / denom, 0.0, 1.0))

    state = np.concatenate([xpos, axis_angle, [c, c]]).astype(np.float32)
    assert state.shape == (8,), state.shape
    return state


def encode_state9(data: mujoco.MjData, ctrl: Pi05ActionController) -> np.ndarray:
    """The 9-D observation for the joint-position action space: 7 joint angles + 2 finger widths.

    This is the state the stage-1 pretraining corpus and the held-out render both store, read
    straight off the sim. No closedness encoding is involved, and therefore none of the
    robosuite-vs-MuJoCo width mismatch that :func:`encode_state8` has to correct for: the widths
    here and the widths in the data come from the same simulator in the same units.

    Float32, shape (9,). The caller MUST assert this shape -- a wrong width passes silently through
    the whole server transform chain and shows up only as a policy that acts but never succeeds.
    """
    q = data.qpos[ctrl.arm_qadr].astype(np.float32)
    fingers = data.qpos[ctrl.finger_qadr].astype(np.float32)
    state9 = np.concatenate([q, fingers]).astype(np.float32)
    assert state9.shape == (9,), state9.shape
    return state9
