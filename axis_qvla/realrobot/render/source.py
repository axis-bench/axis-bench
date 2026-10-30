"""One recorded attempt as the renderer reads it: packed MuJoCo state, scene, task and attempt id.

Render tier (numpy, mujoco).

Input: a `source_episode.json` (schema qvla.realrobot.source_episode/2) holding the task id, the attempt
id, the instruction, the RELATIVE path of the attempt's materialised scene MJCF, the attempt's
`trajectory_data` (packed_state_v1), and `joint_order`: the joint NAMES in the order the trajectory's
columns were recorded. The recorded order is not always the scene file's (one task stores two free
objects in the opposite order to every one of its scene variants), and a positional read puts each
object where the other was, with no error. So `read_source_episode` aligns the columns into the scene's
own layout by name and refuses a name set that differs from the scene's.

Two writers. `from_trace` reads a survey trace (`event: state` rows with `joint_qpos` keyed by name, the
format of the published traces sample); a trace has no velocities, so qvel is NaN and the resampler
uses PCHIP where it would have trusted a recorded velocity. `from_attempt_record` reads an AXIS attempt
record, takes the joint order as a required argument, and copies nothing else: the record's user,
stage, nonce and storage fields never leave it, of the initial snapshot only its state, controls and
time do, and the packed payload is re-compressed under a constant gzip header (no time stamp, OS
unknown), so its bytes depend on the trajectory alone.

Two time bases: without `step_counts`, packed row k is the state at t0 + (k + 1) * dt after k + 1
control periods of dt = timestep * steps_per_ctrl (the initial snapshot is the state at t0); with
`step_counts` (length n_steps - 1), row 0 is the state at t0 and row k follows row k - 1 by
step_counts[k - 1] * timestep. `row_times` gives either.

The source stream: the hand pose plus 0.1084 m along the hand's z (arm joints only, every other joint
at qpos0) and the finger openness (each finger's position over its joint range, clipped, averaged), in
the render world.
"""
from __future__ import annotations

import base64
import dataclasses
import gzip
import json
import struct
import zlib
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

import mujoco
import numpy as np

from axis_qvla.realrobot.render import rig

SCHEMA = "qvla.realrobot.source_episode/2"
PRIVATE_FIELDS = ("user_id", "user_stage", "submit_nonce", "trajectory_s3_bucket", "trajectory_s3_key")
TRAJECTORY_KEYS = ("format", "task_id", "timestep", "steps_per_ctrl", "step_counts", "n_steps", "nq", "nv",
                   "nu", "initial_snapshot", "data_b64_gzip")
SNAPSHOT_KEYS = ("qpos", "qvel", "ctrl", "time")
GZIP_HEADER = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff"     # deflate, no name, MTIME 0, best, OS unknown


class SourceError(ValueError):
    """A recorded attempt the renderer cannot read without guessing."""


@dataclasses.dataclass(frozen=True)
class PackedState:
    timestep: float
    steps_per_ctrl: int
    n_steps: int
    nq: int
    nv: int
    nu: int
    initial_time: float
    initial_qpos: np.ndarray
    initial_qvel: np.ndarray
    initial_ctrl: np.ndarray | None
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    step_counts: tuple[int, ...] | None = None

    @property
    def row_dt(self) -> float:
        return self.timestep * self.steps_per_ctrl

    @property
    def row_times(self) -> np.ndarray:
        """The time of each row: t0 + (k + 1) * dt, or t0 advanced one timestep at a time through the step
        counts, summed in the simulator's order so a row on an exact half frame rounds as its trace's does."""
        if self.step_counts is None:
            return self.initial_time + (np.arange(self.n_steps) + 1) * self.row_dt
        steps = np.concatenate([[0], np.cumsum(self.step_counts, dtype=np.int64)])
        return np.cumsum(np.concatenate([[self.initial_time], np.full(int(steps[-1]), self.timestep)]))[steps]

    @property
    def lead_time(self) -> float:
        """Where the copy of row 0 that opens an episode sits: the initial snapshot's time for 5 Hz
        post-step rows, one nominal source period before row 0 otherwise."""
        if self.step_counts is None and abs(self.row_dt - rig.SOURCE_ROW_DT) <= 1e-9:
            return self.initial_time
        return float(self.row_times[0]) - rig.SOURCE_ROW_DT


def decode_packed_state_v1(doc: Mapping) -> PackedState:
    if doc.get("format") != "packed_state_v1":
        raise SourceError(f"format {doc.get('format')!r}; expected 'packed_state_v1'")
    try:
        n, nq, nv, nu = (int(doc[k]) for k in ("n_steps", "nq", "nv", "nu"))
        timestep = float(doc["timestep"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SourceError("packed_state_v1 needs numeric timestep, n_steps, nq, nv and nu") from exc
    try:
        spc = int(doc.get("steps_per_ctrl"))
    except (TypeError, ValueError) as exc:
        raise SourceError("packed_state_v1 needs an integer steps_per_ctrl") from exc
    if n < 1 or nq < 1 or nv < 1 or nu < 0 or spc < 1:
        raise SourceError("packed_state_v1 dimensions are invalid")
    counts = doc.get("step_counts")
    if counts is not None:
        if (not isinstance(counts, list) or len(counts) != n - 1
                or not all(isinstance(c, int) and not isinstance(c, bool) and c >= 1 for c in counts)):
            raise SourceError(f"step_counts must be {n - 1} positive integers (n_steps - 1)")
        counts = tuple(counts)
    snap = doc.get("initial_snapshot") or {}
    q0 = np.asarray(snap.get("qpos", ()), np.float64)
    v0 = np.asarray(snap.get("qvel", ()), np.float64)
    if q0.shape != (nq,) or v0.shape != (nv,):
        raise SourceError("initial_snapshot qpos/qvel do not match the header")
    c0 = None if snap.get("ctrl") is None else np.asarray(snap["ctrl"], np.float64)
    rows = np.frombuffer(_payload(doc), dtype=np.float32)
    if rows.size != n * (nq + nv + nu):
        raise SourceError(f"payload has {rows.size} float32 values; expected {n * (nq + nv + nu)}")
    rows = rows.reshape(n, nq + nv + nu).astype(np.float64)
    return PackedState(timestep, spc, n, nq, nv, nu, float(snap.get("time", 0.0) or 0.0), q0, v0, c0,
                       rows[:, :nq], rows[:, nq:nq + nv], rows[:, nq + nv:], step_counts=counts)


def _payload(doc: Mapping) -> bytes:
    try:
        return gzip.decompress(base64.b64decode(doc["data_b64_gzip"], validate=True))
    except Exception as exc:
        raise SourceError("data_b64_gzip is not base64 gzip") from exc


def _b64_gzip(raw: bytes) -> str:
    """`raw` as base64 gzip under GZIP_HEADER: the same bytes for the same data, whenever and wherever written."""
    tail = struct.pack("<LL", zlib.crc32(raw), len(raw) & 0xFFFFFFFF)
    return base64.b64encode(GZIP_HEADER + zlib.compress(raw, 9, wbits=-15) + tail).decode("ascii")


def encode_packed_state_v1(p: PackedState, *, task_id: int | None = None) -> dict:
    rows = np.concatenate([p.qpos, p.qvel, p.ctrl], axis=1).astype(np.float32)
    snap = {"qpos": p.initial_qpos.tolist(), "qvel": p.initial_qvel.tolist(), "time": p.initial_time}
    if p.initial_ctrl is not None:
        snap["ctrl"] = p.initial_ctrl.tolist()
    counts = None if p.step_counts is None else list(p.step_counts)
    return {"format": "packed_state_v1", "task_id": task_id, "timestep": p.timestep,
            "steps_per_ctrl": p.steps_per_ctrl, "step_counts": counts, "n_steps": p.n_steps, "nq": p.nq,
            "nv": p.nv, "nu": p.nu, "initial_snapshot": snap,
            "data_b64_gzip": _b64_gzip(rows.tobytes())}


def knot_times(p: PackedState) -> np.ndarray:
    """The time of each row (`PackedState.row_times`)."""
    return p.row_times


@dataclasses.dataclass(frozen=True)
class SourceEpisode:
    task_id: int
    attempt_id: int
    prompt: str
    scene_xml: Path
    packed: PackedState      # columns in the SCENE's layout (align_to_scene)


def joint_names(model: mujoco.MjModel) -> list[str]:
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "" for j in range(model.njnt)]


def _widths(model: mujoco.MjModel, j: int) -> tuple[int, int]:
    t = int(model.jnt_type[j])
    return {int(mujoco.mjtJoint.mjJNT_FREE): (7, 6), int(mujoco.mjtJoint.mjJNT_BALL): (4, 3)}.get(t, (1, 1))


def align_to_scene(p: PackedState, joint_order, model: mujoco.MjModel) -> PackedState:
    """`p` with its columns moved from the recorded joint order into the scene's own layout, by name.
    Controls are dropped: the renderer reads the arm from IK and the gripper from finger positions."""
    order = [str(n) for n in joint_order]
    have, want = set(order), set(joint_names(model))
    if len(order) != len(have):
        raise SourceError("joint_order names a joint twice")
    if want - have:
        raise SourceError(f"joint_order is missing scene joints {sorted(want - have)[:5]}")
    if have - want:
        raise SourceError(f"joint_order names joints not in the scene: {sorted(have - want)[:5]}")
    q_src, v_src = [], []
    qa = va = 0
    for name in order:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        nq, nv = _widths(model, j)
        q_src.append((qa, int(model.jnt_qposadr[j]), nq))
        v_src.append((va, int(model.jnt_dofadr[j]), nv))
        qa, va = qa + nq, va + nv
    if (qa, va) != (p.nq, p.nv) or (qa, va) != (model.nq, model.nv):
        raise SourceError(f"recorded widths {(p.nq, p.nv)} differ from the scene's {(model.nq, model.nv)}")
    qpos, qvel = np.empty_like(p.qpos), np.empty_like(p.qvel)
    q0, v0 = np.empty_like(p.initial_qpos), np.empty_like(p.initial_qvel)
    for (s, t, w) in q_src:
        qpos[:, t:t + w], q0[t:t + w] = p.qpos[:, s:s + w], p.initial_qpos[s:s + w]
    for (s, t, w) in v_src:
        qvel[:, t:t + w], v0[t:t + w] = p.qvel[:, s:s + w], p.initial_qvel[s:s + w]
    return PackedState(p.timestep, p.steps_per_ctrl, p.n_steps, p.nq, p.nv, 0, p.initial_time, q0, v0, None,
                       qpos, qvel, np.zeros((p.n_steps, 0)), step_counts=p.step_counts)


def source_episode_doc(*, task_id: int, attempt_id: int, prompt: str, scene_xml: str,
                       trajectory_data: Mapping, joint_order) -> dict:
    traj = {k: trajectory_data[k] for k in TRAJECTORY_KEYS if k in trajectory_data}
    if isinstance(traj.get("initial_snapshot"), Mapping):
        traj["initial_snapshot"] = {k: traj["initial_snapshot"][k] for k in SNAPSHOT_KEYS if k in traj["initial_snapshot"]}
    return {"schema": SCHEMA, "task_id": int(task_id), "attempt_id": int(attempt_id), "prompt": str(prompt),
            "scene_xml": str(scene_xml), "joint_order": [str(n) for n in joint_order], "trajectory_data": traj}


def from_attempt_record(record: Mapping, *, scene_xml: str, joint_order) -> dict:
    """A source_episode document from an AXIS attempt record and its RECORDED joint order (the order of the
    record's model, e.g. the key order of the attempt's trace); refuses one this module could not render."""
    attempt, task = record["attempt"], record["task"]
    doc = source_episode_doc(task_id=int(attempt["task_id"]), attempt_id=int(attempt["id"]),
                             prompt=str(task["name"]), scene_xml=scene_xml,
                             trajectory_data=record["task_trajectory"]["trajectory_data"], joint_order=joint_order)
    doc["trajectory_data"]["data_b64_gzip"] = _b64_gzip(_payload(doc["trajectory_data"]))
    decode_packed_state_v1(doc["trajectory_data"])
    return doc


def from_trace(trace: Path, *, task_id: int, attempt_id: int, prompt: str, scene_xml: str, scene_dir: Path,
               timestep: float = 0.001, steps_per_ctrl: int = 200) -> dict:
    """A source_episode document from a recorded trace, joints by name. Rows (k + 1) * dt apart from 0
    are the post-step base; any other `sim_time`s are kept as step counts of `timestep`."""
    model = mujoco.MjModel.from_xml_path(str(Path(scene_dir) / scene_xml))
    rows = [json.loads(line) for line in Path(trace).read_text(encoding="utf-8").splitlines() if line.strip()]
    states = [r for r in rows if r.get("event") == "state"]
    if not states:
        raise SourceError(f"{trace}: no state rows")
    dt = timestep * steps_per_ctrl
    order = list(states[0]["joint_qpos"])
    qpos = np.zeros((len(states), model.nq))
    flat_cols = []
    for name in order:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            raise SourceError(f"{trace}: joint {name!r} is not in the scene")
        flat_cols.append(_widths(model, j)[0])
    lacking = [n for n in joint_names(model) if n not in order]
    if lacking:
        raise SourceError(f"{trace}: the trace lacks the scene's joints {lacking[:5]}")
    sim = np.array([float(r["sim_time"]) for r in states])
    for k, r in enumerate(states):
        if int(r["step_index"]) != k:
            raise SourceError(f"{trace}: row {k} has step_index {r['step_index']}")
        if list(r["joint_qpos"]) != order:
            raise SourceError(f"{trace}: row {k} names its joints in another order")
        vals = [np.atleast_1d(np.asarray(r["joint_qpos"][n], np.float64)) for n in order]
        if [v.size for v in vals] != flat_cols:
            raise SourceError(f"{trace}: row {k} has joint widths {[v.size for v in vals]}, the scene {flat_cols}")
        qpos[k] = np.concatenate(vals)
    post_step = bool(np.all(np.abs(sim - (np.arange(len(sim)) + 1) * dt) <= 1e-6))
    counts = None
    if not post_step:
        steps = np.diff(sim) / timestep
        counts = np.rint(steps).astype(int)
        bad = np.flatnonzero((np.abs(steps - counts) > 1e-6) | (counts < 1))
        if bad.size:
            k = int(bad[0]) + 1
            raise SourceError(f"{trace}: row {k} is not a whole number of simulator steps of {timestep} s after "
                              f"row {k - 1} ({sim[k] - sim[k - 1]!r} s)")
    qvel = np.full((len(states), model.nv), np.nan)
    p = PackedState(timestep, steps_per_ctrl, len(states), model.nq, model.nv, 0, 0.0 if post_step else float(sim[0]),
                    qpos[0].copy(), np.zeros(model.nv), None, qpos, qvel, np.zeros((len(states), 0)),
                    step_counts=None if post_step else tuple(int(c) for c in counts))
    return source_episode_doc(task_id=task_id, attempt_id=attempt_id, prompt=prompt, scene_xml=scene_xml,
                              trajectory_data=encode_packed_state_v1(p, task_id=task_id), joint_order=order)


def write_source_episode(path: Path, doc: Mapping) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc), encoding="utf-8")


def episode_task_id(path: Path) -> int:
    """The task id a source_episode.json names, read without decoding its trajectory or compiling its scene."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise SourceError(f"{path}: schema {doc.get('schema')!r}; expected {SCHEMA!r}")
    return int(doc["task_id"])


def read_source_episode(path: Path) -> SourceEpisode:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise SourceError(f"{path}: schema {doc.get('schema')!r}; expected {SCHEMA!r}")
    rel = PurePosixPath(str(doc["scene_xml"]))
    if rel.is_absolute() or ".." in rel.parts:
        raise SourceError(f"{path}: scene_xml must be a relative path inside the episode's directory")
    scene = Path(path).parent / rel
    packed = align_to_scene(decode_packed_state_v1(doc["trajectory_data"]), doc["joint_order"],
                            mujoco.MjModel.from_xml_path(str(scene)))
    return SourceEpisode(int(doc["task_id"]), int(doc["attempt_id"]), str(doc["prompt"]), scene, packed)


@dataclasses.dataclass(frozen=True)
class SourceStream:
    tcp: np.ndarray        # (K, 3) source TCP per row, render world
    hand_R: np.ndarray     # (K, 3, 3) source hand orientation per row
    openness: np.ndarray   # (K,) in [0, 1]; 1 = open
    base_pos: np.ndarray   # (3,) source panda_link0 at qpos0, render world


def _id(model: mujoco.MjModel, kind, name: str) -> int:
    i = mujoco.mj_name2id(model, kind, name)
    if i < 0:
        raise SourceError(f"{name!r} is not in the recorded scene")
    return int(i)


def source_stream(model: mujoco.MjModel, p: PackedState, height: float) -> SourceStream:
    if (model.nq, model.nv) != (p.nq, p.nv):
        raise SourceError(f"scene nq/nv {(model.nq, model.nv)} differ from the trajectory's {(p.nq, p.nv)}")
    J, B = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_BODY
    arm = [int(model.jnt_qposadr[_id(model, J, rig.SOURCE_PREFIX + n)]) for n in rig.ARM_JOINTS]
    fingers = [_id(model, J, rig.SOURCE_PREFIX + n) for n in rig.SOURCE_FINGER_JOINTS]
    hand = _id(model, B, rig.SOURCE_PREFIX + rig.SOURCE_HAND_BODY)
    base = _id(model, B, rig.SOURCE_PREFIX + rig.SOURCE_BASE_BODY)
    lift = np.array([0.0, 0.0, height])
    off = np.asarray(rig.SOURCE_TCP_IN_HAND)
    d = mujoco.MjData(model)
    tcp, rot = np.zeros((p.n_steps, 3)), np.zeros((p.n_steps, 3, 3))
    for k in range(p.n_steps):
        d.qpos[:] = model.qpos0
        d.qpos[arm] = p.qpos[k, arm]
        mujoco.mj_kinematics(model, d)
        rot[k] = d.xmat[hand].reshape(3, 3)
        tcp[k] = d.xpos[hand] + rot[k] @ off + lift
    d.qpos[:] = model.qpos0
    mujoco.mj_kinematics(model, d)
    ranges = np.asarray(model.jnt_range[fingers], np.float64)
    if not (np.all(model.jnt_limited[fingers]) and np.all(ranges[:, 1] > ranges[:, 0])):
        raise SourceError("the source fingers need finite, increasing joint ranges")
    pos = p.qpos[:, model.jnt_qposadr[fingers]]
    u = np.clip((pos - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0]), 0.0, 1.0).mean(axis=1)
    return SourceStream(tcp, rot, u, d.xpos[base] + lift)
