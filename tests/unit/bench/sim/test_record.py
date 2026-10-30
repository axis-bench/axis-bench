"""Result writing must be atomic, must refuse inconsistent records, and must fail loudly on EACCES.

A result reader raises on an unreadable directory rather than reporting it as "no successes", so a
permissions problem is never read as a result.
"""
from __future__ import annotations

import json
import os
import stat

import pytest

from axis_qvla.bench.sim.record import (
    SCHEMA_VERSION,
    RecordError,
    SeedResult,
    git_sha,
    read_seeds,
    score,
    write_meta,
    write_seed,
)


def _result(seed=7, success=True, n_steps=120, sim_time=8.0, reason="success"):
    return SeedResult(
        seed=seed, success=success, n_steps=n_steps, sim_time=sim_time, reason=reason
    )


def test_seed_file_has_the_schema_report_tooling_parses(tmp_path):
    """report.py / compare_variants.py read exactly these five keys."""
    write_seed(tmp_path, _result())
    payload = json.loads((tmp_path / "seed_7.json").read_text())
    assert set(payload) == {"seed", "success", "n_steps", "sim_time", "reason"}
    assert payload["success"] is True
    assert payload["reason"] == "success"


def test_extra_provenance_travels_with_the_seed_file(tmp_path):
    """Seed files get copied; a guidance sweep's records are otherwise indistinguishable."""
    write_seed(tmp_path, _result(), extra={"guidance_scale": 2.0, "eval_layouts": True})
    payload = json.loads((tmp_path / "seed_7.json").read_text())
    assert payload["guidance_scale"] == 2.0
    assert payload["eval_layouts"] is True
    assert payload["success"] is True


def test_extra_cannot_overwrite_the_outcome(tmp_path):
    """Reversing `success` through a side channel would be invisible in the output."""
    with pytest.raises(RecordError, match="success"):
        write_seed(tmp_path, _result(success=False), extra={"success": True})


def test_meta_stamps_schema_version_and_sha(tmp_path):
    write_meta(tmp_path, {"task_id": 1889, "arm": "vanilla_fresh"})
    payload = json.loads((tmp_path / "meta.json").read_text())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["git_sha"]
    assert payload["task_id"] == 1889


def test_caller_supplied_provenance_is_not_overwritten(tmp_path):
    """A replayed run states its own provenance, not this process's."""
    write_meta(tmp_path, {"git_sha": "deadbee", "schema_version": 0})
    payload = json.loads((tmp_path / "meta.json").read_text())
    assert payload["git_sha"] == "deadbee"
    assert payload["schema_version"] == 0


def test_no_temp_files_survive_a_successful_write(tmp_path):
    """A stray .tmp beside the results would be globbed by tooling looking for seeds."""
    write_seed(tmp_path, _result())
    write_meta(tmp_path, {"task_id": 1})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["meta.json", "seed_7.json"]


def test_read_seeds_orders_by_seed_not_filename(tmp_path):
    """Lexicographic filename order puts seed_10 before seed_9; scoring must not care,
    but a reader presenting results in that order misleads."""
    for s in (9, 10, 2):
        write_seed(tmp_path, _result(seed=s))
    assert [r["seed"] for r in read_seeds(tmp_path)] == [2, 9, 10]


def test_read_seeds_ignores_diagnostic_sidecars(tmp_path):
    """seed_<n>.diag.json is a read-only diagnostic; counting it would double the denominator."""
    write_seed(tmp_path, _result(seed=1))
    (tmp_path / "seed_1.diag.json").write_text(json.dumps({"seed": 1, "qpos": []}))
    assert len(read_seeds(tmp_path)) == 1


def test_missing_directory_means_no_results(tmp_path):
    assert read_seeds(tmp_path / "never_ran") == []


def test_unreadable_directory_raises_rather_than_scoring_zero(tmp_path):
    """Path.glob swallows PermissionError, so the reader checks access itself: an unreadable arm
    directory raises instead of scoring 0/0."""
    if os.geteuid() == 0:
        pytest.skip("root bypasses directory permissions")
    d = tmp_path / "locked"
    d.mkdir()
    write_seed(d, _result())
    d.chmod(0o000)
    try:
        with pytest.raises(RecordError, match="cannot list"):
            read_seeds(d)
    finally:
        d.chmod(stat.S_IRWXU)


def test_corrupt_seed_file_raises_rather_than_counting_as_failure(tmp_path):
    """A truncated file is a broken run, not a rollout that failed."""
    write_seed(tmp_path, _result(seed=3))
    (tmp_path / "seed_3.json").write_text('{"seed": 3, "success"')
    with pytest.raises(RecordError, match="cannot read result"):
        read_seeds(tmp_path)


def test_unwritable_directory_raises(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root bypasses directory permissions")
    d = tmp_path / "ro"
    d.mkdir()
    d.chmod(0o500)
    try:
        with pytest.raises(RecordError, match="cannot write"):
            write_seed(d, _result())
    finally:
        d.chmod(stat.S_IRWXU)


def test_score_counts_successes(tmp_path):
    for s, ok in ((1, True), (2, False), (3, True)):
        write_seed(tmp_path, _result(seed=s, success=ok))
    assert score(read_seeds(tmp_path)) == (2, 3)


def test_score_of_nothing_is_zero_of_zero():
    """Distinguishable from 0/50 -- an arm that never ran is not an arm that failed."""
    assert score([]) == (0, 0)


def test_git_sha_never_raises(tmp_path):
    """Provenance must not abort a finished eval, even outside a repository."""
    assert isinstance(git_sha(tmp_path), str)
    assert git_sha(tmp_path)


def test_only_seed_int_json_is_a_result(tmp_path):
    """Under --diagnostics the dual-sim backend also writes
    seed_<n>.diag_objz.json, which `seed_*.json` minus `*.diag.json` read as a second record of
    seed <n>. A result is exactly the name write_seed gives it: seed_<int>.json."""
    write_seed(tmp_path, SeedResult(1, True, 10, 0.5, "success"))
    for name in ("seed_1.diag.json", "seed_1.diag_objz.json", "seed_x.json", "seed_.json",
                 "seed_1.json.bak", "seed_1.mp4", "seed_1.diag_frame0.png", "meta.json"):
        (tmp_path / name).write_text(json.dumps({"seed": 1, "success": True}))
    assert [r["seed"] for r in read_seeds(tmp_path)] == [1]


def test_is_seed_file_is_the_one_selection_rule():
    from axis_qvla.bench.sim.record import is_seed_file

    assert is_seed_file("seed_806000.json") and is_seed_file("seed_0.json")
    assert not any(is_seed_file(n) for n in ("seed_1.diag_objz.json", "seed_1.diag.json", "seed_-1.json",
                                             "seed_1.json.tmp", ".seed_1.json.tmp9", "xseed_1.json"))
    # ASCII digits only: Python's \d also matches other scripts' digits (Arabic-Indic, fullwidth).
    assert not any(is_seed_file(n) for n in ("seed_\u0661\u0662.json", "seed_\uff11.json"))
