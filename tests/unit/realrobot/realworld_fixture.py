"""A two-episode copy of the published real-world data's layout (axis_qvla.realrobot.render.data), over the
render fixture's scene: what the real-world tools' tests and the tutorial's CI blocks read."""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import mujoco

from axis_qvla.realrobot.render import data, source
from tests.unit.labels.test_reward import LEAF
from tests.unit.realrobot.render import fixtures

PROMPT = "push the red box"
LAYOUT = "fixture_layout_1"


def _trace(path: Path, scene_xml: Path, n: int) -> None:
    """The recorded trace format: per row, every joint by name and every body's pose."""
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    d = mujoco.MjData(m)
    p = fixtures.fixture_packed(scene_xml, n)
    names = source.joint_names(m)
    rows = []
    for k in range(n):
        d.qpos[:] = p.qpos[k]
        mujoco.mj_kinematics(m, d)
        q = {}
        for name in names:
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            a, w = int(m.jnt_qposadr[j]), (7 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1)
            v = p.qpos[k, a:a + w].tolist()
            q[name] = v if w > 1 else v[0]
        bodies = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "world":
                  {"pos": d.xpos[b].tolist(), "quat": d.xquat[b].tolist()} for b in range(m.nbody)}
        rows.append({"event": "state", "source": "packed_state_v1_direct", "step_index": k,
                     "sim_time": (k + 1) * 0.2, "joint_qpos": q, "body_pose": bodies})
    rows.append({"event": "final", "final_sim_time": n * 0.2, "total_state_frames": n})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def write_realworld(root: Path, *, task_id: int = 9001, attempts=(7001, 7002), n: int = 12) -> dict:
    root = Path(root)
    scene_rel = data.scene_rel(task_id, 0)
    scene = root / scene_rel
    scene.parent.mkdir(parents=True, exist_ok=True)
    scene.write_text(fixtures.SCENE_XML)
    order = source.joint_names(mujoco.MjModel.from_xml_path(str(scene)))
    episodes = []
    for aid in attempts:
        doc = source.source_episode_doc(task_id=task_id, attempt_id=aid, prompt=PROMPT, scene_xml="variant_0.xml",
                                        trajectory_data=source.encode_packed_state_v1(fixtures.fixture_packed(scene, n),
                                                                                      task_id=task_id),
                                        joint_order=order)
        source.write_source_episode(root / data.episode_rel(task_id, aid), doc)
        _trace(root / data.trace_rel(task_id, aid), scene, n)
        leaf = {**LEAF, "tracked_body_name": "box_1", "goal_xyz": [0.5, 0.0, 0.1], "pass_frame": n - 2}
        goals = {"attempt_id": aid, "n_steps": n, "top_first_pass_frame": n - 2, "leaves": [leaf]}
        (root / data.goals_rel(task_id, aid)).parent.mkdir(parents=True, exist_ok=True)
        (root / data.goals_rel(task_id, aid)).write_text(json.dumps(goals))
        episodes.append({"attempt_id": aid, "variant": 0})
    (root / data.checker_rel(task_id)).parent.mkdir(parents=True, exist_ok=True)
    (root / data.checker_rel(task_id)).write_text(json.dumps({"checker": {"type": "RelativePositionBoundsChecker",
                                                                           "objName": "box_1"}}))
    (root / "robots" / "franka_emika_panda").mkdir(parents=True, exist_ok=True)
    (root / "robots" / "franka_emika_panda" / "LICENSE").write_text("Apache-2.0\n")
    tar = root / data.mesh_tar_rel(LAYOUT)
    tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar, "w", format=tarfile.PAX_FORMAT) as t:
        info = tarfile.TarInfo(f"{LAYOUT}/box_1/README")
        body = b"fixture mesh directory\n"
        info.size = len(body)
        t.addfile(info, io.BytesIO(body))
    manifest = {"schema": data.SCHEMA, "episodes": len(attempts), "tasks": {str(task_id): {
        "prompt": PROMPT, "variants": [0], "layouts": [LAYOUT], "episodes": episodes}},
        "robots": ["robots/franka_emika_panda/LICENSE"]}
    (root / data.MANIFEST).write_text(json.dumps(manifest, indent=1))
    return manifest
