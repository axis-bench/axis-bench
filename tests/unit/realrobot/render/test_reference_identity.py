"""The reference workspace's render model, camera poses, intrinsics and time base, number for number.

`reference_render.json` holds, for the fixture scene, the bodies, geoms, materials, cameras and lights
of the compiled render and IK models, both cameras' poses and the end-effector pose at the pixelmatch
home, both cameras' intrinsics, and the frames, knot rows, states and actions of one composed
12-knot episode. A render's frames are a function of the model and the states, so equal arrays render
equal frames on one GL stack. The model arrays compare exactly; the composed states, which pass through
scipy's interpolants, to 1e-12. Regenerate only for an intended change, in the render environment:
    python -c "from tests.unit.realrobot.render.test_reference_identity import write_expected; write_expected()"
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from axis_qvla.realrobot.render import assemble, draw, episode, gripper, rig, source
from tests.unit.realrobot.render import fixtures

EXPECTED = Path(__file__).with_name("reference_render.json")
KEY = (9001, 7001)


def _name(m: mujoco.MjModel, kind, i: int) -> str | None:
    return None if i < 0 else (mujoco.mj_id2name(m, kind, int(i)) or "")


def model_record(m: mujoco.MjModel) -> dict:
    G, M, C, B, L = (mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_MATERIAL, mujoco.mjtObj.mjOBJ_CAMERA,
                     mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_LIGHT)
    return {
        "sizes": [int(m.nq), int(m.nv), int(m.nu), int(m.nbody), int(m.ngeom), int(m.nmat), int(m.ncam), int(m.nlight)],
        "bodies": [[_name(m, B, i), m.body_pos[i].tolist(), m.body_quat[i].tolist()] for i in range(m.nbody)],
        "geoms": [[_name(m, G, i), int(m.geom_type[i]), int(m.geom_bodyid[i]), m.geom_pos[i].tolist(),
                   m.geom_quat[i].tolist(), m.geom_size[i].tolist(), m.geom_rgba[i].tolist(),
                   _name(m, M, int(m.geom_matid[i]))] for i in range(m.ngeom)],
        "materials": [[_name(m, M, i), m.mat_rgba[i].tolist(), float(m.mat_reflectance[i]),
                       float(m.mat_shininess[i]), float(m.mat_specular[i]), float(m.mat_emission[i])]
                      for i in range(m.nmat)],
        "cameras": [[_name(m, C, i), int(m.cam_bodyid[i]), m.cam_pos[i].tolist(), m.cam_quat[i].tolist(),
                     float(m.cam_fovy[i])] for i in range(m.ncam)],
        "lights": [[_name(m, L, i), int(m.light_type[i]), int(m.light_castshadow[i]), m.light_pos[i].tolist(),
                    m.light_dir[i].tolist(), m.light_ambient[i].tolist(), m.light_diffuse[i].tolist(),
                    m.light_specular[i].tolist()] for i in range(m.nlight)],
        "headlight": [m.vis.headlight.ambient.tolist(), m.vis.headlight.diffuse.tolist(),
                      m.vis.headlight.specular.tolist(), int(m.vis.headlight.active)],
        "offscreen": [int(m.vis.global_.offwidth), int(m.vis.global_.offheight)],
    }


def _composed(tmp: Path, render: assemble.TargetScene) -> dict:
    """One 12-knot fixture episode composed from a fixed arm path and fixed gripper extremes, so the
    numbers depend on the time base and the interpolants alone (no IK, no physics)."""
    ep = source.read_source_episode(fixtures.write_episode(tmp / "ep", n=12))
    arm_q = np.asarray(rig.PIXELMATCH_HOME) + np.linspace(0.0, 0.3, 12)[:, None] * np.ones(7)
    ng = len(render.gripper_qpos)
    ext = gripper.GripperExtremes(np.zeros(ng), np.full(ng, 0.04), 0.0, 255.0, 0.098, 0.018)
    openness = np.where(np.arange(12) < 6, 1.0, 0.5)
    states = episode.compose(render, ep.packed, arm_q, ext, openness, 0.48)
    return {"grid": states.grid_t.tolist(), "knot_rows": states.knot_rows.tolist(),
            "qpos": states.qpos.tolist(), "action": states.action.tolist()}


def snapshot(tmp: Path) -> dict:
    scene_xml, grip = fixtures.write_scene(tmp / "s"), fixtures.write_gripper(tmp / "g")
    render = assemble.build_target_scene(scene_xml, grip, key=KEY)
    ik = assemble.build_target_scene(scene_xml, grip, objects=False, key=KEY)
    q = np.tile(render.model.qpos0, (2, 1))
    q[:, render.arm_qpos] = rig.PIXELMATCH_HOME
    c2w, ee = draw.poses(render, q)
    return {"render": model_record(render.model), "ik": model_record(ik.model),
            "camera_to_world": {k: v[0].tolist() for k, v in sorted(c2w.items())}, "ee_state": ee[0].tolist(),
            "intrinsics": {c.key: list(rig.intrinsics(c)) for c in render.camera_specs},
            "composed": _composed(tmp, render)}


def write_expected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        EXPECTED.write_text(json.dumps(snapshot(Path(tmp)), indent=1, sort_keys=True) + "\n")


def test_the_reference_workspace_compiles_to_the_pinned_render_model(tmp_path):
    got, want = snapshot(tmp_path), json.loads(EXPECTED.read_text())
    composed, want_composed = got.pop("composed"), want.pop("composed")
    assert got == want
    assert composed["knot_rows"] == want_composed["knot_rows"]
    for key in ("grid", "qpos", "action"):
        np.testing.assert_allclose(composed[key], want_composed[key], rtol=0, atol=1e-12, err_msg=key)
