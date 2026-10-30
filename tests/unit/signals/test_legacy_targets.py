"""Every locked file resolves to a legacy target, and the recorded config hashes recompute from the
recorded generator digests plus the inputs' digests -- no corpus needed (conf/ only)."""
from __future__ import annotations

import dataclasses
import json

import pytest

from axis_qvla.signals import legacy

LOCK = json.loads(legacy.SIGNALS_LOCK.read_text())
SPEC = json.loads(legacy.GENERATORS.read_text())
ROWS = {r["id"]: r for r in LOCK["files"]}


def _arg(argv, flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


def _lock_input(token: str) -> str:
    assert token.startswith("{lock:") and token.endswith("}"), token
    return token[len("{lock:"):-1]


@pytest.mark.parametrize("lock_id", sorted(ROWS))
def test_every_locked_file_is_a_target_in_its_consumers_format(lock_id):
    row = ROWS[lock_id]
    fmt = legacy.KIND_FORMATS[row["kind"]][0]
    t = legacy.legacy_target(lock_id, fmt)
    assert (t.path, t.kind, t.sha256, t.format) == (row["path"], row["kind"], row["sha256"], fmt)
    recorded = (row.get("generator") or {}).get("generator_sha256")
    if recorded:
        assert t.generator_sha256 == recorded


def test_a_target_refuses_the_wrong_format():
    with pytest.raises(legacy.LegacyError, match="lerobot_rows"):
        legacy.legacy_target("lerobot.rows", "openpi-legacy")
    with pytest.raises(legacy.LegacyError):
        legacy.legacy_target("no.such.file", "openpi-legacy")


def test_the_roots_index_each_generator_hashed_is_read_off_the_generator_map():
    form = {i: legacy.legacy_target(i, legacy.KIND_FORMATS[ROWS[i]["kind"]][0]).roots_form for i in ROWS}
    assert form["approach_aware.chunk.schedule_drop_top50"] == "v2"
    assert form["approach_aware.episode.schedule_drop_top50"] == "d8"
    assert form["object_centric.episode.schedule_drop_top50"] == "d8"
    assert form["approach_aware.chunk.tags"] == form["object_centric.chunk.tags"] == "d8"
    assert form["approach_aware.episode.tags"] == form["object_centric.episode.tags"] == "v2"
    assert legacy.legacy_target("lerobot.schedule_drop_top_rand50_ep_smolvla", "lerobot-legacy").template_name \
        == "awr_weights_v2.json"


def test_a_target_is_hashable_and_cannot_change_the_cached_generator_map():
    """The tolerance is stored as sorted (key, bound) pairs, never the generator map's own dict: a
    frozen target must hash, and writing through it must not reach the map every later target is
    read from."""
    t = legacy.legacy_target("approach_aware.chunk.tags", "openpi-legacy")
    assert t.tolerance == (("meta.q_given_task_entropy_ratio", 4e-16),)
    assert hash(t) == hash(legacy.legacy_target("approach_aware.chunk.tags", "openpi-legacy"))
    with pytest.raises(TypeError):
        t.tolerance["meta.other"] = 1.0
    assert SPEC["files"]["approach_aware.chunk.tags"]["tolerance"] == legacy._maps()[0]["files"][
        "approach_aware.chunk.tags"]["tolerance"] == {"meta.q_given_task_entropy_ratio": 4e-16}
    assert dataclasses.replace(t, tolerance={"b": 1, "a": 2}).tolerance == (("a", 2.0), ("b", 1.0))
    assert legacy.legacy_target("index.ranges", "openpi-legacy").tolerance is None
    oc = legacy.legacy_target("object_centric.chunk.tags", "lerobot-legacy")
    assert (oc.numerics, oc.tolerance) == ("log2-libm", None)


SCHEDULES = [i for i, e in SPEC["files"].items() if e["generator"].startswith("schedule@")]


@pytest.mark.parametrize("lock_id", SCHEDULES)
def test_every_schedules_config_hash_recomputes_from_the_recorded_digest(lock_id):
    argv = SPEC["files"][lock_id]["argv"]
    t = legacy.legacy_target(lock_id, legacy.KIND_FORMATS[ROWS[lock_id]["kind"]][0])
    roots = SPEC["inputs"]["roots_v2"]["sha256"] if t.roots_form == "v2" else ROWS["index.roots_d8"]["sha256"]
    got = legacy.schedule_config_hash(
        mode=_arg(argv, "--mode"), keep=float(_arg(argv, "--keep-top-frac", 0.5)),
        steps=int(_arg(argv, "--total-steps", 20605)), batch=int(_arg(argv, "--batch", 64)),
        seed=int(_arg(argv, "--seed", 0)), generator_sha256=t.generator_sha256,
        weights_sha256=ROWS[_lock_input(_arg(argv, "--weights"))]["sha256"], roots_sha256=roots,
        ranges_sha256=ROWS["index.ranges"]["sha256"])
    assert got == ROWS[lock_id]["generator"]["config_hash"]
    assert len(SCHEDULES) == 8
