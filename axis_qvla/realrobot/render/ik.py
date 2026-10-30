"""Full-SE(3) TCP IK with mink at the recorded rows: the target's pinch follows the source TCP.

Render tier; the only module that imports mink. A FrameTask on panda_link7 whose target puts the gripper
frame -- approach axis to the pinch site, closing axis between the pads, both measured at the posture
home -- on the source hand frame and the pinch on the source TCP; a PostureTask toward the posture home;
configuration, velocity (4 rad/s) and self-collision limits; up to 24 iterations of dt 0.02 per knot,
warm-started from the previous knot, rescued from the posture home and four random seeds when the warm
start misses by more than 5 mm; among candidates within rig.IK.reach_tol (2 cm) of the source TCP the
smallest approach-axis error plus 0.1 x joint jump wins, and when none is, the smallest position error
plus 0.3 x joint jump (constants: rig.IK).

Knot 0 is the pixelmatch home (rig.PIXELMATCH_HOME), not a solve. The eight seed draws of knot 0 are
drawn and discarded, so later rescue seeds come from the same stream. Each knot iterates toward 1.5 mm
of the source TCP (rig.IK.converge_tol); the convergence knot is the first after knot 0 within
rig.TCP_GATE_M (3 mm). Some recordings move the hand where the Panda with the 2F-85 cannot follow; there
the render keeps the pose the IK selects, which can be several centimetres from the source TCP. An
episode is accepted when at least rig.IK.min_reach_pct (60%) of its knots come within rig.IK.reach_tol
(2 cm) of the source TCP, otherwise IkError. The redundant joint is free, so joint parity with another
render is a reported metric (report.py).
"""
from __future__ import annotations

import dataclasses

import mink
import mujoco
import numpy as np

from axis_qvla.realrobot.render import rig
from axis_qvla.realrobot.render.assemble import TargetScene
from axis_qvla.realrobot.render.source import SourceStream


class IkError(RuntimeError):
    """The IK cannot follow the source TCP well enough to render the attempt."""


@dataclasses.dataclass(frozen=True)
class IkResult:
    arm_q: np.ndarray
    perr: np.ndarray
    oerr: np.ndarray
    seeds: np.ndarray

    @property
    def reach_pct(self) -> float:
        return float(np.mean(self.perr < rig.IK.reach_tol) * 100.0)

    @property
    def convergence_knot(self) -> int | None:
        hits = np.flatnonzero(self.perr[1:] <= rig.TCP_GATE_M)
        return int(hits[0]) + 1 if hits.size else None


def _ortho(z: np.ndarray, y0: np.ndarray) -> np.ndarray:
    z = z / np.linalg.norm(z)
    y = y0 - (y0 @ z) * z
    y /= np.linalg.norm(y)
    return np.column_stack([np.cross(y, z), y, z])


def _collision_limits(m: mujoco.MjModel, scene: TargetScene, names: list[str], params: rig.IkParams) -> list:
    J = mujoco.mjtObj.mjOBJ_JOINT
    arm_b = [int(m.jnt_bodyid[mujoco.mj_name2id(m, J, n)]) for n in names]
    proximal_bodies = (int(m.body_parentid[arm_b[0]]), arm_b[0], arm_b[1])
    collides = lambda g: bool(m.geom_contype[g] or m.geom_conaffinity[g])
    distal = [g for g in mink.get_subtree_geom_ids(m, scene.ee_body) if collides(g)]
    proximal = [g for b in proximal_bodies for g in mink.get_body_geom_ids(m, b) if collides(g)]
    if not (distal and proximal):
        return []
    return [mink.CollisionAvoidanceLimit(m, [(distal, proximal)], minimum_distance_from_collisions=params.collision_margin)]


def solve_knots(scene: TargetScene, stream: SourceStream, *, params: rig.IkParams = rig.IK,
                home=rig.PIXELMATCH_HOME, posture=rig.POSTURE_HOME) -> IkResult:
    m = scene.model
    names = [rig.SOURCE_PREFIX + n for n in rig.ARM_JOINTS]
    qadr = scene.arm_qpos
    lims = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])

    def full(arm) -> np.ndarray:
        q = m.qpos0.copy()
        q[qadr] = np.clip(np.asarray(arm, np.float64), lims[:, 0], lims[:, 1])
        return q

    home_q, posture_q = full(home), full(posture)
    d0 = mujoco.MjData(m)
    d0.qpos[:] = posture_q
    mujoco.mj_forward(m, d0)
    ee = scene.ee_body
    eeR0 = d0.xmat[ee].reshape(3, 3)
    offa = eeR0.T @ (d0.site_xpos[scene.pinch_site] - d0.xpos[ee])
    ga = _ortho(offa / np.linalg.norm(offa),
                eeR0.T @ (d0.xpos[scene.finger_bodies[0]] - d0.xpos[scene.finger_bodies[1]]))
    solver = params.solver
    config = mink.Configuration(m)
    config.update(posture_q)
    task = mink.FrameTask(frame_name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, ee), frame_type="body",
                          position_cost=params.position_cost, orientation_cost=params.orientation_cost,
                          lm_damping=params.lm_damping)
    posture_task = mink.PostureTask(m, cost=params.posture_cost)
    posture_task.set_target(posture_q)
    limits = [mink.ConfigurationLimit(m), mink.VelocityLimit(m, {n: params.velocity_limit for n in names})]
    limits += _collision_limits(m, scene, names, params)
    rng = np.random.default_rng(params.rng_seed)
    tcp, hand_R = stream.tcp, stream.hand_R

    def seed() -> np.ndarray:
        s = posture_q.copy()
        for k, a in enumerate(qadr):
            s[a] = np.clip(posture_q[a] + rng.normal(0.0, params.seed_std), lims[k, 0], lims[k, 1])
        return s

    def tcp_error(t: int, data: mujoco.MjData) -> float:
        return float(np.linalg.norm(data.xpos[ee] + data.xmat[ee].reshape(3, 3) @ offa - tcp[t]))

    def approach_error(t: int, data: mujoco.MjData) -> float:
        rc = data.xmat[ee].reshape(3, 3)
        return float(np.arccos(np.clip(np.dot(rc @ ga[:, 2], hand_R[t][:, 2]), -1.0, 1.0)))

    def converge(t: int):
        e = 9.99
        for _ in range(params.iterations):
            vel, last = None, None
            for lim in (limits, limits[:2]):
                try:
                    vel = mink.solve_ik(config, [task, posture_task], params.dt, solver, limits=lim,
                                        damping=params.solver_damping)
                    break
                except Exception as exc:      # an infeasible QP is retried without the collision limit
                    last = exc
            if vel is None:
                raise IkError(f"mink IK failed with solver {solver!r}") from last
            config.integrate_inplace(vel, params.dt)
            mujoco.mj_forward(m, config.data)
            e = tcp_error(t, config.data)
            if e < params.converge_tol:
                break
        return config.q.copy(), e, approach_error(t, config.data)

    K = len(tcp)
    arm, perr, oerr, nseed = np.zeros((K, len(qadr))), np.zeros(K), np.zeros(K), np.zeros(K, np.int64)
    for _ in range(params.seeds_first):
        seed()
    config.update(home_q)
    mujoco.mj_forward(m, config.data)
    arm[0], perr[0], oerr[0] = home_q[qadr], tcp_error(0, config.data), approach_error(0, config.data)
    prev = home_q.copy()
    for t in range(1, K):
        r_ee = hand_R[t] @ ga.T
        task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(r_ee), tcp[t] - r_ee @ offa))
        config.update(prev)
        cands = [converge(t)]
        if cands[0][1] > params.rescue_tol:
            config.update(posture_q)
            cands.append(converge(t))
            for _ in range(params.seeds_rescue):
                config.update(seed())
                cands.append(converge(t))
        jump = lambda c: float(np.linalg.norm(c[0][qadr] - prev[qadr]))
        valid = [c for c in cands if c[1] < params.reach_tol]
        best = min(valid, key=lambda c: c[2] + 0.1 * jump(c)) if valid else min(cands, key=lambda c: c[1] + 0.3 * jump(c))
        config.update(best[0])
        prev = best[0]
        arm[t], perr[t], oerr[t], nseed[t] = best[0][qadr], best[1], best[2], len(cands)
    result = IkResult(arm, perr, oerr, nseed)
    if result.reach_pct < params.min_reach_pct:
        raise IkError(f"the retargeted arm reaches only {result.reach_pct:.1f}% of rows within "
                      f"{params.reach_tol * 100:.0f} cm; the minimum is {params.min_reach_pct:.0f}%")
    return result
