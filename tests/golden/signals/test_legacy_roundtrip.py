"""Every locked signal file is a legacy record: decompose it, render it, get its locked bytes back.

Needs the published corpus pack (--pretraining-data). This proves the codecs, not the methods: the
method goldens prove that compiling from inputs produces these same records.
"""
from __future__ import annotations

import dataclasses
import hashlib

import pytest

from axis_qvla.signals import legacy
from axis_qvla.signals.bundle import export_legacy, load_bundle, write_bundle
from axis_qvla.signals.schema import FORMATS
from axis_qvla.signals.testing import toy_bundle


def lock_format(row: dict) -> str:
    return legacy.KIND_FORMATS[row["kind"]][0]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.mark.slow
def test_every_locked_file_decomposes_and_renders_back_to_its_locked_bytes(signals_pack, lock):
    assert len(lock["files"]) == 30
    for row in lock["files"]:
        data = (signals_pack / row["path"]).read_bytes()
        assert _sha(data) == row["sha256"], row["id"]
        lf, arrays = legacy.decompose(data, path=row["path"], lock_id=row["id"], fmt=lock_format(row),
                                      name=row["id"].replace(".", "_"))
        assert _sha(legacy.render(lf, {f"legacy.{k}": v for k, v in arrays.items()})) == row["sha256"], row["id"]


@pytest.mark.slow
def test_a_written_and_loaded_bundle_exports_the_same_bytes(signals_pack, lock, tmp_path):
    """The manifest is sorted-key JSON; ordered trees must carry every key order through it."""
    ids = ["approach_aware.chunk.awr_weights", "approach_aware.chunk.tags", "approach_aware.chunk.schedule_drop_top50",
           "index.ranges", "index.roots_d8", "index.scaling_1000", "lerobot.rows", "lerobot.awr_flat_phase"]
    rows = {r["id"]: r for r in lock["files"]}
    files, arrays = [], {}
    for i in ids:
        lf, a = legacy.decompose((signals_pack / rows[i]["path"]).read_bytes(), path=rows[i]["path"], lock_id=i,
                                 fmt=lock_format(rows[i]), name=i.replace(".", "_"))
        files.append(lf)
        arrays.update(a)
    b = dataclasses.replace(toy_bundle(), legacy=tuple(files), legacy_arrays=arrays)
    loaded = load_bundle(tmp_path / "b" / write_bundle(b, tmp_path / "b"))
    by_path = {r["path"]: r["sha256"] for r in rows.values()}
    seen = 0
    for fmt in FORMATS:
        for p in export_legacy(loaded, fmt, tmp_path / fmt):
            assert _sha(p.read_bytes()) == by_path[p.relative_to(tmp_path / fmt).as_posix()]
            seen += 1
    assert seen == len(ids)
