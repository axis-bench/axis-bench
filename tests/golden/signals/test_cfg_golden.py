"""CFG regenerates the 4 locked tag files (--pretraining-data, --episodes), byte for byte.

Two chunk-tag files stamp an entropy ratio whose last bits follow how log2 rounds, and each is
evaluated with the log2 the generator map declares: the approach-aware one with a correctly rounded
log2, which needs an extended long double (x86-64; elsewhere that export is refused), and
the object-centric one with glibc 2.28's double log2, evaluated the same way on
every host (axis_qvla.signals.numerics). So every file must come out as its locked bytes; a
difference fails, naming the first member or meta key that differs.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from axis_qvla.signals.bundle import export_legacy
from axis_qvla.signals.imported import ImportedScores
from axis_qvla.signals.legacy import legacy_target
from axis_qvla.signals.methods.cfg import CfgTags
from axis_qvla.signals.diff import first_difference
from axis_qvla.signals.schema import Budget


def _scores(pack, lock, lock_id):
    row = next(r for r in lock["files"] if r["id"] == lock_id)
    return ImportedScores.from_json(pack / row["path"], name=lock_id, expect_sha256=row["sha256"])


def _assert_locked(got, pack, target) -> None:
    if hashlib.sha256(got.read_bytes()).hexdigest() != target.sha256:
        raise AssertionError(f"{target.lock_id} differs from its locked bytes at "
                             f"{first_difference(got, pack / target.path) or 'the file bytes'}")


@pytest.mark.slow
@pytest.mark.parametrize("lock_id,source,granularity,fmt,convention,steps,seed", [
    ("approach_aware.episode.tags", "approach_aware.chunk.awr_weights", "episode", "openpi-legacy", "openpi-rowsampler", 20605, 42),
    ("object_centric.episode.tags", "object_centric.chunk.awr_weights", "episode", "openpi-legacy", "openpi-rowsampler", 20605, 42),
    ("approach_aware.chunk.tags", "approach_aware.chunk.awr_weights", "row", "openpi-legacy", "openpi-rowsampler", 20605, 42),
    ("object_centric.chunk.tags", "object_centric.chunk.awr_weights", "row", "lerobot-legacy", "lerobot-axisrowsampler", 20621, 0),
])
def test_cfg_regenerates_the_locked_tags(paper_rowspaces, signals_pack, lock, tmp_path,
                                         lock_id, source, granularity, fmt, convention, steps, seed):
    target = legacy_target(lock_id, fmt)
    b = CfgTags(granularity, convention, exports=(target,)).compile(
        _scores(signals_pack, lock, source), paper_rowspaces["libero5k_v2_d8"], Budget(steps, 64), seed)
    [got] = export_legacy(b, fmt, tmp_path)
    _assert_locked(got, signals_pack, target)
    with np.load(got) as z:
        np.testing.assert_array_equal(b.cond[..., 0], z["tag"][b.plan])


@pytest.mark.slow
def test_a_scaling_arm_draws_its_subset_and_reuses_the_full_tags(paper_rowspaces, signals_pack, lock, tmp_path):
    target = legacy_target("approach_aware.episode.tags", "openpi-legacy")
    full, sub = paper_rowspaces["libero5k_v2_d8"], paper_rowspaces["libero5k_v2_d8.sub1000"]
    b = CfgTags("episode", "openpi-rowsampler", tag_rows=full, exports=(target,)).compile(
        _scores(signals_pack, lock, "approach_aware.chunk.awr_weights"), sub, Budget(4168, 64), 42)
    [got] = export_legacy(b, "openpi-legacy", tmp_path)
    assert hashlib.sha256(got.read_bytes()).hexdigest() == target.sha256
