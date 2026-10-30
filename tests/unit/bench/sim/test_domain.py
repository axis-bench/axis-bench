"""Domain dispatch, checked without importing a single rollout backend.

That is the property under test as much as the dispatch itself: the physics and render
backends need mutually exclusive venvs, so anything that binds them must stay importable in
both. Backends are injected here via sys.modules stubs.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from axis_qvla.bench.sim.domain import (
    DomainError,
    _as_seed_result,
    local_rollout_fn,
    served_rollout_fn,
)


class _Res:
    def __init__(self, success=True, n_steps=100, sim_time=6.5, reason="success"):
        self.success, self.n_steps = success, n_steps
        self.sim_time, self.reason = sim_time, reason


@pytest.fixture
def stub_backends(monkeypatch):
    """Install fake rollout backends and record what they were called with."""
    calls = {}

    def _mk(modname, funcname):
        mod = types.ModuleType(modname)

        def fn(*args, **kwargs):
            calls[funcname] = {"args": args, "kwargs": kwargs}
            return _Res()

        setattr(mod, funcname, fn)
        monkeypatch.setitem(sys.modules, modname, mod)

    _mk("axis_qvla.bench.sim.rollout_dualsim", "rollout_pi05_dualsim")
    _mk("axis_qvla.bench.sim.rollout_render", "rollout_pi05_render")
    _mk("axis_qvla.bench.sim.rollout_physics", "rollout_pi05_policy")
    _mk("axis_replay_native.rollout", "rollout_policy")
    return calls


COMMON = {"prompt": "put the pear on the plate", "fps": 15.0, "replan": 8, "max_steps": 1500}


def _served(domain, **kw):
    return served_rollout_fn(
        domain,
        client=object(),
        task_dir=Path("/survey/task_1889"),
        task_id=1889,
        prompt="put the pear on the plate",
        common=COMMON,
        **kw,
    )


def test_physics_needs_no_layouts(stub_backends):
    assert _served("physics")(7).success is True
    assert stub_backends["rollout_pi05_policy"]["kwargs"]["init_state_seed"] == 7


def test_render_dispatches_to_the_render_backend(stub_backends):
    fn = _served("render", layouts={7: "L7"}, render_root=Path("/r"), work_dir=Path("/w"))
    fn(7)
    assert "rollout_pi05_render" in stub_backends
    assert stub_backends["rollout_pi05_render"]["kwargs"]["eval_layout"] == "L7"


def test_dualsim_dispatches_to_the_dualsim_backend(stub_backends):
    fn = _served("dualsim", layouts={7: "L7"}, render_root=Path("/r"), work_dir=Path("/w"))
    fn(7)
    assert "rollout_pi05_dualsim" in stub_backends


def test_domains_do_not_cross_wire(stub_backends):
    """A render arm silently running the physics backend would produce contaminated
    numbers under a clean-sounding directory name."""
    _served("render", layouts={1: "L"}, render_root=Path("/r"), work_dir=Path("/w"))(1)
    assert "rollout_pi05_policy" not in stub_backends
    assert "rollout_pi05_dualsim" not in stub_backends


def test_common_knobs_reach_the_backend(stub_backends):
    _served("physics")(7)
    kwargs = stub_backends["rollout_pi05_policy"]["kwargs"]
    assert kwargs["max_steps"] == 1500
    assert kwargs["fps"] == 15.0


def test_render_without_layouts_is_refused():
    """An unevaluable task must cost zero rollouts, so this fails before the server."""
    with pytest.raises(DomainError, match="layout"):
        _served("render", render_root=Path("/r"), work_dir=Path("/w"))


def test_render_without_render_root_is_refused():
    with pytest.raises(DomainError, match="render_root"):
        _served("render", layouts={1: "L"})


def test_missing_layout_for_a_seed_fails_by_name(stub_backends):
    fn = _served("render", layouts={1: "L"}, render_root=Path("/r"), work_dir=Path("/w"))
    with pytest.raises(DomainError, match="no planned layout for seed 9"):
        fn(9)


def test_unknown_domain_is_refused():
    with pytest.raises(DomainError, match="domain"):
        _served("teleport")


def test_diagnostics_path_is_passed_through(stub_backends):
    fn = _served(
        "render", layouts={7: "L"}, render_root=Path("/r"), work_dir=Path("/w"),
        diag_dir=Path("/d"),
    )
    fn(7)
    assert stub_backends["rollout_pi05_render"]["kwargs"]["trace_path"] == Path(
        "/d/seed_7.diag.json"
    )


def test_no_diagnostics_means_none(stub_backends):
    _served("render", layouts={7: "L"}, render_root=Path("/r"), work_dir=Path("/w"))(7)
    assert stub_backends["rollout_pi05_render"]["kwargs"]["trace_path"] is None


# --- the diagnostic artifact is a different KIND per domain -------------------------------

def test_physics_diagnostics_is_an_mp4_not_a_json(stub_backends):
    """The physics backend hands `video_path` to imageio.mimwrite, which picks the container
    from the suffix, so a .json name would have imageio write an MP4 into it."""
    _served("physics", diag_dir=Path("/d"))(7)
    assert stub_backends["rollout_pi05_policy"]["kwargs"]["video_path"] == Path("/d/seed_7.mp4")


def test_render_diagnostics_is_a_json_sidecar(stub_backends):
    _served("dualsim", layouts={7: "L"}, render_root=Path("/r"), work_dir=Path("/w"),
            diag_dir=Path("/d"))(7)
    assert stub_backends["rollout_pi05_dualsim"]["kwargs"]["trace_path"] == Path(
        "/d/seed_7.diag.json"
    )


def test_physics_without_diagnostics_passes_none(stub_backends):
    _served("physics")(7)
    assert stub_backends["rollout_pi05_policy"]["kwargs"]["video_path"] is None


def test_diagnostic_path_refuses_an_unknown_domain():
    from axis_qvla.bench.sim.domain import diagnostic_path

    with pytest.raises(DomainError, match="domain"):
        diagnostic_path(Path("/d"), "teleport", 1)


def test_local_builds_a_fresh_policy_per_seed(stub_backends):
    """A reused policy_fn leaks the tail of the previous chunk into the next rollout."""
    built = []
    fn = local_rollout_fn(
        policy_factory=lambda: built.append(1) or (lambda obs: obs),
        task_dir=Path("/survey/task_1"), object_body="pear", goal_xyz=[0, 0, 0],
        norm_stats=None, checker_factory=lambda s: None, max_steps=500,
    )
    fn(1)
    fn(2)
    assert len(built) == 2


def test_local_passes_seed_and_budget(stub_backends):
    fn = local_rollout_fn(
        policy_factory=lambda: (lambda obs: obs),
        task_dir=Path("/survey/task_1"), object_body="pear", goal_xyz=[0, 0, 0],
        norm_stats=None, checker_factory=lambda s: None, max_steps=500,
    )
    fn(3)
    kwargs = stub_backends["rollout_policy"]["kwargs"]
    assert kwargs["init_state_seed"] == 3
    assert kwargs["max_steps"] == 500


def test_result_missing_a_field_is_refused():
    """A result that cannot be recorded must not be scored."""
    class Partial:
        success, n_steps = True, 1
        sim_time = 1.0
        # no `reason`

    with pytest.raises(DomainError, match="missing"):
        _as_seed_result(1, Partial())


def test_result_is_coerced_to_the_schema_types():
    class Loose:
        success, n_steps, sim_time, reason = 1, "100", "6.5", "success"

    r = _as_seed_result(1, Loose())
    assert r.success is True and r.n_steps == 100 and r.sim_time == 6.5
