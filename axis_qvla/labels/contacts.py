"""Per-frame geometric contact from the task's MuJoCo scene, and the body-name matching it depends on.

Two pairs of flags: (eef, finger) say the hand or a fingertip touches ANY object; (eef_target,
finger_target) say the contact is with the reward leaf's body -- only the second pair tells grasping
the task object from brushing a distractor. Every free-joint object is positioned before contacts
are read: an object left at its MJCF default phantom-collides with the hand.

Offline tier: numpy plus a lazily imported MuJoCo.
"""
from __future__ import annotations

import re

import numpy as np

ROBOT_PREFIX = "franka"
HAND_BODY = "franka/panda_hand"
FINGER_BODIES = ("franka/panda_leftfinger", "franka/panda_rightfinger")
FINGER_JOINTS = ("franka/panda_finger_joint1", "franka/panda_finger_joint2")
# Per-frame contact booleans, in the order the corpus writes them.
CONTACT_COLUMNS = ("contact_eef", "contact_finger", "contact_target_eef", "contact_target_finger")


def stem(name: str) -> str:
    """Body identity without a path prefix or a trailing `_<digits>` instance index."""
    return re.sub(r"_\d+$", "", name.split("/")[-1])


class ContactError(ValueError):
    """Contact could not be computed as asked, or was asked for in a way that hides a bug."""


class ContactModel:
    """One task's MJCF, held open for per-row contact queries.

    Load once per task, then call `contacts(joint_qpos, object_poses, target_bodies)` per render
    row: it writes the robot joints and every object free joint, runs `mj_forward`, and reads
    `data.contact`.
    """

    def __init__(self, mjcf_path: str):
        import mujoco

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)

        def jn(j):
            return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)

        self._adr = {jn(j): int(self.model.jnt_qposadr[j]) for j in range(self.model.njnt)}
        self._free = [
            jn(j) for j in range(self.model.njnt)
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        ]
        self._obj_bodies = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.jnt_bodyid[j]))
            for j in range(self.model.njnt)
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        }
        self._body_of_geom = lambda g: mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[g])
        )

    @property
    def free_joints(self) -> list[str]:
        return list(self._free)

    @property
    def object_bodies(self) -> set[str]:
        return set(self._obj_bodies)

    def contacts(
        self, joint_qpos: dict, object_poses: dict, target_bodies=None
    ) -> tuple[bool, bool, bool, bool]:
        """`(eef, finger, eef_target, finger_target)` for one frame.

        `joint_qpos` is `{joint_name: value}`; `object_poses` is
        `{free_joint_name: (7,) pos + quat wxyz}` and must cover EVERY free joint -- see the
        module docstring on phantom contact.
        """
        missing = [fj for fj in self._free if fj not in object_poses]
        if missing:
            raise ContactError(
                f"unpositioned free-joint objects {missing} would sit at their MJCF default "
                f"pose and produce false contacts. Provide poses for ALL scene objects "
                f"(stem-match trace bodies to free joints), not just the stored-schema union."
            )
        for name, v in joint_qpos.items():
            a = self._adr.get(name)
            if a is None:
                continue
            arr = np.atleast_1d(np.asarray(v, dtype=float))
            self.data.qpos[a : a + arr.size] = arr
        for fj, pose in object_poses.items():
            a = self._adr.get(fj)
            if a is None:
                continue
            self.data.qpos[a : a + 7] = np.asarray(pose, dtype=float)
        self._mj.mj_forward(self.model, self.data)

        eef = finger = eef_t = finger_t = False
        tset = set(target_bodies or ())
        for c in range(self.data.ncon):
            b1 = self._body_of_geom(self.data.contact[c].geom1)
            b2 = self._body_of_geom(self.data.contact[c].geom2)
            robot = [b for b in (b1, b2) if b in (*FINGER_BODIES, HAND_BODY)]
            obj = [b for b in (b1, b2) if b in self._obj_bodies]
            if not (robot and obj):
                continue
            on_target = any(o in tset for o in obj)
            if any(r in FINGER_BODIES for r in robot):
                finger = True
                finger_t |= on_target
            if any(r == HAND_BODY for r in robot):
                eef = True
                eef_t |= on_target
        return eef, finger, eef_t, finger_t


def resolve_target_bodies(target_name: str, candidate_bodies) -> set[str]:
    """Map the reward leaf's body onto the scene's actual body names, BY STEM.

    The leaf names a trace body (`scissors_0`) while the MJCF carries its own instance indices
    (`scissors_4`). Matching raw names
    returns nothing, and contact scoped to an empty set reads as "never touched" rather than
    failing, so callers must treat an empty result as an ERROR, not as absence of contact.

    Returns every instance sharing the stem: a scene can hold several and the checker's OR over
    them means any one may be the manipulated object.
    """
    s = stem(target_name)
    return {b for b in candidate_bodies if stem(b) == s}


def checker_body_stems(checker_config: dict) -> set[str]:
    """Stems of every `objName`/`refName` in the (possibly nested) checker tree."""
    stems: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            for key in ("objName", "refName"):
                v = node.get(key)
                if isinstance(v, str):
                    stems.add(stem(v))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(checker_config)
    return stems


def union_bodies(checker_config: dict, trace_body_names: list[str]) -> list[str]:
    """Trace bodies to keep: robot bodies union checker-referenced objects (stem-matched).

    `world` is dropped (fixed frame origin). Order follows `trace_body_names` for stability.

    NOT what the corpus stores -- the builder publishes EVERY scene body, because contact has to
    position all free-joint objects anyway and a consumer handed only the union cannot reproduce
    our contact signal. Kept for callers that genuinely want the narrow set.
    """
    wanted = checker_body_stems(checker_config)
    out: list[str] = []
    for name in trace_body_names:
        if name == "world":
            continue
        if name.startswith(ROBOT_PREFIX) or stem(name) in wanted:
            out.append(name)
    return out
