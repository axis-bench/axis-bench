"""Opt-in diagnostic capture for the render-domain pi0.5 rollout.

Exists to answer the three-way question a bare {success, n_steps, reason} record cannot,
when a smoke rollout scores 0/N with every seed `reason=timeout`:

  (a) POLICY  -- the arm moves but does not solve the task,
  (b) HARNESS -- the success checker tracks a body absent from the scene (or otherwise
                 never fires), so no policy could ever pass,
  (c) CONTROL -- the position-servo actuators do not track the integrated velocity
                 command, so the arm lags/freezes regardless of policy quality.

It captures, all READ-ONLY (it observes the sim, never steps or writes it, so enabling
it cannot change a rollout's outcome): the arm actuators' MuJoCo transmission/gain/bias
types and prms (position vs velocity vs general/OSC, directly), the per-step
command-tracking gap ||q_cmd - qpos||, the arm qpos trajectory, the commanded action /
chunk statistics, the presence of the checker's tracked bodies in the compiled model, and
the checker's own tracked quantity (the obj-ref relative delta and gripper joint values)
at every sampled step.

Runs on the CLIENT side (the eval environment), alongside rollout_pi05_render. Numeric only;
frame capture, if any, is the caller's concern.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

# probe(model, data) -> (passed, [sub_result_dict, ...]); see task_checker.make_detail_probe.
DetailProbe = Callable[[mujoco.MjModel, mujoco.MjData], "tuple[bool, list[dict]]"]


def _enum_name(enum_cls, val: int) -> str:
    try:
        return enum_cls(int(val)).name
    except Exception:  # noqa: BLE001
        return str(int(val))


def _actuator_report(model: mujoco.MjModel, ctrl_ids: np.ndarray) -> list[dict[str, Any]]:
    """Per-actuator transmission/gain/bias signature -- position vs velocity vs general.

    A Franka affine-bias POSITION servo reads gaintype=FIXED, biastype=AFFINE with
    biasprm=[0, -kp, -kv]; a true VELOCITY actuator has biasprm=[0, 0, -kv]; an OSC/general
    actuator uses a non-JOINT trntype or a dyntype. Reported raw so the reader classifies.
    """
    out: list[dict[str, Any]] = []
    for i in (int(x) for x in ctrl_ids):
        out.append({
            "index": i,
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "",
            "trntype": _enum_name(mujoco.mjtTrn, model.actuator_trntype[i]),
            "gaintype": _enum_name(mujoco.mjtGain, model.actuator_gaintype[i]),
            "biastype": _enum_name(mujoco.mjtBias, model.actuator_biastype[i]),
            "dyntype": _enum_name(mujoco.mjtDyn, model.actuator_dyntype[i]),
            "gainprm": [float(v) for v in model.actuator_gainprm[i][:3]],
            "biasprm": [float(v) for v in model.actuator_biasprm[i][:3]],
            "ctrlrange": [float(v) for v in model.actuator_ctrlrange[i]],
            "ctrllimited": bool(model.actuator_ctrllimited[i]),
            "trnid_joint": mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[i][0])
            ) or "",
        })
    return out


def _tracked_quantity(sub_results: list[dict]) -> dict[str, Any]:
    """Pull the checker's own numbers out of check_with_details: the relative delta the
    RelativePositionBoundsChecker tests, and the gripper joint values / threshold."""
    out: dict[str, Any] = {}
    for r in sub_results:
        t = r.get("checker_type")
        if t == "RelativePositionBoundsChecker":
            delta = r.get("delta")
            out["rel_delta"] = [float(x) for x in delta] if delta is not None else None
            out["rel_delta_norm"] = (
                float(np.linalg.norm(delta)) if delta is not None else None
            )
            out["rel_obj"] = r.get("obj_name")
            out["rel_ref"] = r.get("ref_name")
            out["rel_obj_pos"] = r.get("obj_position")
            out["rel_ref_pos"] = r.get("ref_position")
            out["rel_passed"] = bool(r.get("passed"))
        elif t == "GripperOpenChecker":
            out["gripper_joint_values"] = r.get("joint_values")
            out["gripper_threshold"] = r.get("threshold")
            out["gripper_passed"] = bool(r.get("passed"))
    return out


@dataclasses.dataclass
class RolloutDiagnostics:
    """Accumulates a per-step numeric trace and writes it to one sidecar json."""

    trace_path: Path
    arm_qadr: np.ndarray
    finger_qadr: np.ndarray
    arm_ctrl: np.ndarray
    finger_ctrl: np.ndarray
    probe: DetailProbe | None
    every: int = 10
    static: dict[str, Any] = dataclasses.field(default_factory=dict)
    rows: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    q0: np.ndarray | None = None

    @classmethod
    def build(cls, trace_path: Path | str, ctrl, probe: DetailProbe | None, *, every: int = 10):
        return cls(
            trace_path=Path(trace_path),
            arm_qadr=np.asarray(ctrl.arm_qadr),
            finger_qadr=np.asarray(ctrl.finger_qadr),
            arm_ctrl=np.asarray(ctrl.arm_ctrl),
            finger_ctrl=np.asarray(ctrl.finger_ctrl),
            probe=probe,
            every=int(every),
        )

    def capture_static(self, model: mujoco.MjModel, data: mujoco.MjData, *, tracked_bodies) -> None:
        """One-shot: actuator signatures, the full model body-name list, and whether each
        checker-tracked body actually resolves in the compiled model."""
        body_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            for i in range(model.nbody)
        ]
        body_set = set(body_names)

        def _present(name: str) -> dict[str, Any]:
            exact = name in body_set
            suffix = next((b for b in body_names if b.endswith("/" + name) or b.endswith(name)), None)
            return {"name": name, "exact": exact, "resolved_as": suffix, "present": exact or suffix is not None}

        self.q0 = data.qpos[self.arm_qadr].astype(float).copy()
        self.static = {
            "arm_actuators": _actuator_report(model, self.arm_ctrl),
            "finger_actuators": _actuator_report(model, self.finger_ctrl),
            "n_bodies": int(model.nbody),
            "body_names": body_names,
            "tracked_bodies": [_present(n) for n in tracked_bodies],
            "opt_timestep": float(model.opt.timestep),
        }

    def record_step(self, step: int, model, data, q_cmd, action, chunk, *, force: bool = False) -> None:
        if not force and step % self.every != 0:
            return
        qpos = data.qpos[self.arm_qadr].astype(float)
        q_cmd = np.asarray(q_cmd, dtype=float)
        a = np.asarray(action, dtype=float).reshape(-1)
        row: dict[str, Any] = {
            "step": int(step),
            "final": bool(force),
            "sim_time": float(data.time),
            "arm_qpos": qpos.tolist(),
            "q_cmd": q_cmd.tolist(),
            "cmd_gap_norm": float(np.linalg.norm(q_cmd - qpos)),
            "cmd_gap_per_joint": (q_cmd - qpos).tolist(),
            "qpos_delta_from_start_norm": (
                float(np.linalg.norm(qpos - self.q0)) if self.q0 is not None else None
            ),
            "finger_qpos": data.qpos[self.finger_qadr].astype(float).tolist(),
            "action_arm_mean": float(np.mean(a[:7])),
            "action_arm_absmean": float(np.mean(np.abs(a[:7]))),
            "action_arm_norm": float(np.linalg.norm(a[:7])),
            "action_gripper": float(a[7]) if a.shape[0] > 7 else None,
        }
        if chunk is not None:
            ch = np.asarray(chunk, dtype=float)
            row["chunk_shape"] = list(ch.shape)
            row["chunk_arm_absmean"] = float(np.mean(np.abs(ch[:, :7])))
            row["chunk_arm_norm"] = float(np.linalg.norm(ch[:, :7]))
        if self.probe is not None:
            try:
                passed, sub = self.probe(model, data)
                row["checker_passed"] = bool(passed)
                row["checker"] = _tracked_quantity(sub)
            except Exception as exc:  # noqa: BLE001
                row["checker_error"] = str(exc)
        self.rows.append(row)

    def write(self, *, n_steps: int, reason: str, success: bool) -> Path:
        sampled = [r for r in self.rows if not r["final"]]
        first = sampled[0] if sampled else (self.rows[0] if self.rows else None)
        last = self.rows[-1] if self.rows else None

        def _gap(r):
            return None if r is None else r.get("cmd_gap_norm")

        def _rel(r):
            return None if r is None else (r.get("checker") or {}).get("rel_delta_norm")

        def _qd(r):
            return None if r is None else r.get("qpos_delta_from_start_norm")

        mid = sampled[len(sampled) // 2] if sampled else None
        summary = {
            "n_steps": int(n_steps),
            "reason": reason,
            "success": bool(success),
            "n_rows": len(self.rows),
            "arm_qpos_delta_total_norm": _qd(last),
            "cmd_gap_norm_step0": _gap(first),
            "cmd_gap_norm_mid": _gap(mid),
            "cmd_gap_norm_final": _gap(last),
            "rel_delta_norm_step0": _rel(first),
            "rel_delta_norm_final": _rel(last),
        }
        payload = {"summary": summary, "static": self.static, "trace": self.rows}
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_path.write_text(json.dumps(payload, indent=2))
        return self.trace_path
