"""The seed loop, exercised with a fake rollout_fn -- no simulator, no GPU, no policy server.

The behaviour under test: a sweep that dies partway leaves its completed seeds attributable to a
checkpoint.
"""
from __future__ import annotations

import json

import pytest

from axis_qvla.bench.sim.record import SeedResult
from axis_qvla.bench.sim.runner import RunnerError, RunSpec, resolve_seeds, run_seeds


def _spec(tmp_path, seeds=(1, 2, 3), **kw):
    return RunSpec(
        task_id=kw.get("task_id", 1889),
        arm=kw.get("arm", "vanilla_fresh"),
        seeds=tuple(seeds),
        log_dir=tmp_path,
        provenance=kw.get("provenance", {"checkpoint_dir": "/ckpt/2999"}),
    )


def _always(success, reason="success"):
    def fn(seed):
        return SeedResult(
            seed=seed, success=success, n_steps=100, sim_time=6.5, reason=reason
        )
    return fn


def _quiet(_msg):
    pass


def test_runs_every_seed_and_tallies(tmp_path):
    assert run_seeds(_spec(tmp_path), _always(True), report=_quiet) == (3, 3)


def test_writes_one_file_per_seed(tmp_path):
    run_seeds(_spec(tmp_path), _always(True), report=_quiet)
    assert sorted(p.name for p in tmp_path.glob("seed_*.json")) == [
        "seed_1.json", "seed_2.json", "seed_3.json"
    ]


def test_mixed_outcomes_tally_correctly(tmp_path):
    def fn(seed):
        return SeedResult(seed, seed != 2, 100, 6.5, "success" if seed != 2 else "timeout")
    assert run_seeds(_spec(tmp_path), fn, report=_quiet) == (2, 3)


def test_meta_is_written_before_the_first_rollout(tmp_path):
    """The point of the change: a crash on seed 1 still leaves attributable provenance."""
    seen = {}

    def fn(seed):
        if seed == 1:
            seen["meta"] = json.loads((tmp_path / "meta.json").read_text())
        return SeedResult(seed, True, 100, 6.5, "success")

    run_seeds(_spec(tmp_path), fn, report=_quiet)
    assert seen["meta"]["complete"] is False
    assert seen["meta"]["checkpoint_dir"] == "/ckpt/2999"


def test_crash_leaves_partial_results_attributable(tmp_path):
    """Legacy behaviour lost meta.json entirely, orphaning the seeds already paid for."""
    def fn(seed):
        if seed == 3:
            raise RuntimeError("policy server died")
        return SeedResult(seed, True, 100, 6.5, "success")

    with pytest.raises(RuntimeError, match="policy server died"):
        run_seeds(_spec(tmp_path), fn, report=_quiet)

    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["complete"] is False
    assert meta["checkpoint_dir"] == "/ckpt/2999"
    assert sorted(p.name for p in tmp_path.glob("seed_*.json")) == [
        "seed_1.json", "seed_2.json"
    ]


def test_backend_failure_is_not_swallowed_into_zeros(tmp_path):
    """A column of zeros from a broken sim reads as 'the policy cannot do this task'."""
    with pytest.raises(RuntimeError):
        run_seeds(_spec(tmp_path), lambda s: (_ for _ in ()).throw(RuntimeError("egl")),
                  report=_quiet)
    assert not list(tmp_path.glob("seed_*.json"))


def test_completed_meta_carries_the_tally(tmp_path):
    run_seeds(_spec(tmp_path, seeds=(1, 2, 3, 4)), _always(False, "timeout"), report=_quiet)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["complete"] is True
    assert (meta["n_success"], meta["n_seeds"]) == (0, 4)
    assert meta["success_rate"] == 0.0


def test_success_rate_is_recorded(tmp_path):
    def fn(seed):
        return SeedResult(seed, seed <= 2, 100, 6.5, "success")
    run_seeds(_spec(tmp_path, seeds=(1, 2, 3, 4)), fn, report=_quiet)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["success_rate"] == 0.5


def test_per_seed_extras_are_written(tmp_path):
    run_seeds(
        _spec(tmp_path), _always(True),
        extra_fn=lambda s: {"guidance_scale": 2.0, "base_variant_index": s},
        report=_quiet,
    )
    payload = json.loads((tmp_path / "seed_2.json").read_text())
    assert payload["guidance_scale"] == 2.0
    assert payload["base_variant_index"] == 2


def test_empty_seed_list_is_refused(tmp_path):
    """0/0 is indistinguishable from a run that failed."""
    with pytest.raises(RunnerError, match="no seeds"):
        _spec(tmp_path, seeds=())


def test_duplicate_seeds_are_refused(tmp_path):
    """A repeat overwrites its earlier result and silently shrinks the denominator."""
    with pytest.raises(RunnerError, match="duplicate seeds"):
        _spec(tmp_path, seeds=(1, 2, 2))


def test_arm_and_task_reach_the_meta(tmp_path):
    run_seeds(_spec(tmp_path, arm="cfg_w2_fresh", task_id=809), _always(True), report=_quiet)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["arm"] == "cfg_w2_fresh"
    assert meta["task_id"] == 809


def test_resolve_seeds_preserves_manifest_order(tmp_path):
    """Sorting would change which seeds a truncated --n-seeds run evaluates."""
    assert resolve_seeds([31, 4, 17, 92], 3) == (31, 4, 17)


def test_resolve_seeds_without_limit_takes_all():
    assert resolve_seeds([31, 4, 17], None) == (31, 4, 17)


def test_resolve_seeds_rejects_nonpositive():
    with pytest.raises(RunnerError, match="must be positive"):
        resolve_seeds([1, 2, 3], 0)


# --- explicit seed subsets: what lets a work queue split one task across workers -------------

def test_only_selects_a_subset_in_declared_order():
    """A subset run must agree seed-for-seed with the full run, so order stays the manifest's."""
    assert resolve_seeds([9, 3, 7, 1], None, only=[1, 9]) == (9, 1)


def test_only_refuses_an_undeclared_seed():
    """CATCHES a queue typo evaluating an unscreened initial condition that still lands in the
    denominator -- an off-benchmark rollout silently scored as if it were part of the suite."""
    with pytest.raises(RunnerError, match="not among this task's declared test seeds"):
        resolve_seeds([9, 3, 7, 1], None, only=[9, 42])


def test_only_is_filtered_against_the_truncated_list_not_the_full_one():
    """With --n-seeds 2 the benchmark is seeds [9, 3]; asking for 7 must fail, not silently widen."""
    with pytest.raises(RunnerError):
        resolve_seeds([9, 3, 7, 1], 2, only=[7])


def test_empty_only_is_refused():
    with pytest.raises(RunnerError, match="nothing would be evaluated"):
        resolve_seeds([9, 3, 7, 1], None, only=[])


def test_subsets_partition_the_full_seed_list():
    """The queue invariant: disjoint subsets must reassemble into exactly the full run."""
    declared = [100, 101, 102, 103, 104]
    a = resolve_seeds(declared, None, only=[100, 102])
    b = resolve_seeds(declared, None, only=[101, 103, 104])
    assert sorted(a + b) == sorted(resolve_seeds(declared, None))
    assert set(a) & set(b) == set()


def test_no_only_is_unchanged():
    assert resolve_seeds([9, 3, 7, 1], 3) == (9, 3, 7)
