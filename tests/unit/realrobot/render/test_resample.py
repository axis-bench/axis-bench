from __future__ import annotations

import mujoco
import numpy as np
import pytest

from axis_qvla.realrobot.render import resample, rig


def _kt(n):
    return 0.0 + (np.arange(n) + 1) * (0.001 * 200)


def test_an_81_row_attempt_is_243_frames_end_aligned_with_knots_on_rows_3k_plus_2():
    kt = _kt(81)
    g = resample.frame_grid(kt)[0]
    assert len(g) == 243 and g[-1] == kt[-1]
    rows = resample.frame_grid(kt)[1]
    np.testing.assert_array_equal(rows, 3 * np.arange(81) + 2)
    assert np.all(np.abs(g[rows] - kt) < 1e-12)


def test_the_duration_rule_gives_12_frames_for_0_8_seconds():
    assert len(resample.frame_grid(_kt(4))[0]) == 12


def test_zoh_is_causal_and_holds_until_the_next_knot():
    kt = np.array([0.0, 0.2, 0.4])
    g = np.array([0.0, 0.1, 0.2, np.nextafter(0.4, 0.0), 0.4])
    np.testing.assert_array_equal(resample.zoh(kt, np.array([10.0, 20.0, 30.0]), g), [10, 10, 20, 20, 30])


def test_the_hermite_guard_falls_back_to_pchip_on_overshoot():
    kt = np.array([0.0, 0.2, 0.4, 0.6])
    v = np.array([0.0, 1.0, 1.0, 1.0])
    vel = np.array([0.0, 50.0, -50.0, 0.0])        # wildly wrong velocities
    out, method = resample.interpolate_coordinate(kt, v, vel, np.linspace(0, 0.6, 13), speed_floor=1e-3)
    assert method.startswith("pchip_") and out.min() >= -1e-12 and out.max() <= 1.0 + 1e-12


def test_reliable_velocities_use_the_cubic_hermite():
    kt = np.linspace(0.0, 1.0, 6)
    v = np.sin(kt)
    out, method = resample.interpolate_coordinate(kt, v, np.cos(kt), kt, speed_floor=1e-3)
    assert method == "cubic_hermite_reliable_qvel"
    np.testing.assert_allclose(out, v, atol=1e-12)


def test_slerp_aligns_hemispheres_and_stays_unit():
    kt = np.array([0.0, 1.0])
    q = np.array([[1.0, 0, 0, 0], [-np.cos(0.2), 0, 0, -np.sin(0.2)]])
    out = resample.slerp(kt, q, np.linspace(0, 1, 5))
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-12)
    assert out[2][0] > 0.99


def test_resample_qpos_covers_hinge_slide_and_free_joints():
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint name="h" range="-1 1"/><geom size=".1"/>'
        '<body><joint name="s" type="slide" range="0 1"/><geom size=".1"/></body></body>'
        '<body name="b"><freejoint/><geom size=".1"/></body></worldbody></mujoco>')
    kt = np.array([0.0, 0.2, 0.4, 0.6])
    q = np.zeros((4, m.nq))
    q[:, 0] = [0.0, 0.1, 0.2, 0.3]
    q[:, 1] = [0.0, 0.5, 0.5, 1.0]
    q[:, 2:5] = [[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0]]
    q[:, 5] = 1.0
    out, meta = resample.resample_qpos(m, kt, q, np.zeros((4, m.nv)), np.linspace(0, 0.6, 10))
    assert out.shape == (10, m.nq) and np.all(np.isfinite(out))
    np.testing.assert_allclose(np.linalg.norm(out[:, 5:9], axis=1), 1.0, atol=1e-12)
    assert set(meta) >= {"hinge", "slide", "free"}


def test_the_frames_never_start_before_the_opening_copy_of_row_0():
    rng = np.random.default_rng(0)
    for _ in range(200):
        kt = np.r_[0.0, np.cumsum(rng.integers(1, 120, 40))] * 0.002
        grid, rows = resample.frame_grid(kt)
        assert grid[0] > kt[0] - rig.SOURCE_ROW_DT and grid[-1] == kt[-1] and rows[0] == rig.KNOT_ROW_OFFSET
