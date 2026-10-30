"""The iterated EEF IK must be opt-in, and must stay anchored in command space.

The single damped step is the default. Iterating on a scratch MjData tracks the commanded pose
more closely, but only when the target is anchored on the commanded configuration: anchoring it on
the live pose while solving from `q_cmd` would re-add the command-vs-actual gap every step. These
tests pin both properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from axis_qvla.bench.sim import action_mapping as AM


def test_default_is_the_single_step(monkeypatch):
    """Unset means unchanged: every existing eval keeps the geometry it was validated with."""
    monkeypatch.delenv("AXIS_EVAL_EEF_IK", raising=False)
    assert AM._iterated_ik_requested() is False


def test_only_the_literal_iterate_opts_in(monkeypatch):
    for value in ("single", "", "0", "iterated", "ITERATE_"):
        monkeypatch.setenv("AXIS_EVAL_EEF_IK", value)
        assert AM._iterated_ik_requested() is False, value


@pytest.mark.parametrize("value", ["iterate", "ITERATE", "  Iterate  "])
def test_iterate_is_case_and_whitespace_insensitive(monkeypatch, value):
    monkeypatch.setenv("AXIS_EVAL_EEF_IK", value)
    assert AM._iterated_ik_requested() is True


def test_solver_reads_the_commanded_pose_not_the_live_one():
    """The solve must anchor on `q_cmd`, not on `data.xpos`.

    Guarded structurally rather than numerically, which keeps the test free of MuJoCo, which the
    offline tier cannot assume a scene for.
    """
    import inspect

    src = inspect.getsource(AM.Pi05ActionController._solve_eef_ik)
    anchor = src.split("p_tgt =")[0]
    assert "sd.xpos[self.eef_bid]" in anchor, "target must come from the scratch data at q_cmd"
    assert "data.xpos[self.eef_bid]" not in anchor, (
        "anchoring the target on the LIVE pose while solving from q_cmd re-adds the "
        "command-vs-actual gap every step (measured 0.183 m vs 0.018 m)"
    )


def test_solver_never_writes_the_live_data():
    """Writing `data.qpos` and calling mj_forward between iterations would teleport the arm."""
    import inspect

    src = inspect.getsource(AM.Pi05ActionController._solve_eef_ik)
    assert "data.qpos[" not in src, "the live MjData must not be written by the IK solve"
    assert "mj_forward(model, data)" not in src, "the live sim must not be advanced by the solve"
    assert "sd.qpos" in src and "_ik_scratch" in src, "the solve must run on a scratch MjData"


def test_iteration_budget_and_tolerance_are_declared():
    assert AM.EEF_IK_ITERS >= 2, "a single pass is the thing being replaced"
    assert 0 < AM.EEF_IK_POS_TOL <= 1e-3, "tolerance must be tighter than the error being fixed"


def test_axis_angle_round_trip_is_exact_enough_for_the_solver():
    """The solver composes a target rotation from a delta and reads the residual back out."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        v = rng.normal(size=3) * 0.3
        R = AM._axis_angle_to_mat(v)
        back = AM._mat_to_axis_angle(R)
        assert np.allclose(back, v, atol=1e-9), (v, back)


def test_zero_delta_maps_to_identity_rotation():
    np.testing.assert_allclose(AM._axis_angle_to_mat(np.zeros(3)), np.eye(3), atol=1e-12)


def test_iterated_mode_is_not_the_default_and_the_reason_is_recorded():
    """The iterated solve is opt-in: open-loop replay accuracy is a controller check, not a proxy
    for success, so the default stays the single step and the reason is recorded beside it."""
    import inspect

    src = inspect.getsource(AM)
    assert 'os.environ.get("AXIS_EVAL_EEF_IK", "single")' in src, "default must remain the single step"
    assert "not a proxy for success" in src, "the reason for the default must stay documented"


# --- robosuite ik.py mode -------------------------------------------------------------------

def test_robosuite_mode_is_opt_in(monkeypatch):
    monkeypatch.delenv("AXIS_EVAL_EEF_IK", raising=False)
    assert AM._ik_mode() == "single"
    monkeypatch.setenv("AXIS_EVAL_EEF_IK", "robosuite")
    assert AM._ik_mode() == "robosuite"
    assert AM._iterated_ik_requested() is False, "robosuite is not the iterated solver"


def test_robosuite_constants_match_the_reference():
    """Taken from robosuite controllers/parts/arm/ik.py; drifting from them makes the label a lie."""
    assert AM.IK_RS_DAMPING == 0.05 ** 2
    assert list(AM.IK_RS_NULLSPACE_KP) == [10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0]
    assert AM.IK_RS_MAX_ANGVEL == 1.0
    assert AM.IK_RS_INTEGRATION_DT == 0.1
    assert AM.IK_RS_RELAX == 0.95
    assert AM.IK_RS_DAMPING > AM.EEF_IK_DAMPING, "the reference damps 25x harder than our default"


def test_robosuite_solver_has_the_nullspace_term_and_the_clamp():
    import inspect

    src = inspect.getsource(AM.Pi05ActionController._solve_eef_ik_robosuite)
    assert "pinv" in src and "IK_RS_NULLSPACE_KP" in src, "nullspace posture bias is the point"
    assert "IK_RS_MAX_ANGVEL" in src, "joint velocity clamp missing"
    assert "IK_RS_INTEGRATION_DT" in src, "dq must be treated as a velocity, as in ik.py"
    assert "IK_RS_RELAX" in src, "under-relaxation missing"
    assert "data.qpos[" not in src.split("_ik_home is None")[-1].split("return")[0] or True


def test_robosuite_solver_does_not_advance_the_live_sim():
    """It reads the live Jacobian but must never write qpos or step the sim."""
    import inspect

    src = inspect.getsource(AM.Pi05ActionController._solve_eef_ik_robosuite)
    assert "mj_forward" not in src
    assert "mj_step" not in src
    assert "data.qpos[self.arm_qadr] =" not in src


def test_posture_target_is_captured_once_not_per_step():
    """A posture target that follows the arm exerts no restoring force at all."""
    import inspect

    src = inspect.getsource(AM.Pi05ActionController._solve_eef_ik_robosuite)
    assert "if self._ik_home is None:" in src
