"""BC, drop-top-50 and random-50 regenerate their 18 locked files byte for byte from the published
inputs (--pretraining-data) over the row spaces built from corpus metadata (--episodes)."""
from __future__ import annotations

import hashlib

import pytest

from axis_qvla.signals.bundle import bundle_id, export_legacy
from axis_qvla.signals.compile import compile_program
from axis_qvla.signals.imported import ImportedScores
from axis_qvla.signals.legacy import legacy_target
from axis_qvla.signals.methods.bc import BC
from axis_qvla.signals.methods.filtering import DropTop50, Random50
from axis_qvla.signals.schema import Budget


def _exported(bundle, fmt, out) -> dict:
    return {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in export_legacy(bundle, fmt, out)}


def _want(lock, ids) -> dict:
    rows = {r["id"]: r for r in lock["files"]}
    return {rows[i]["path"]: rows[i]["sha256"] for i in ids}


def _scores(pack, lock, lock_id) -> ImportedScores:
    row = next(r for r in lock["files"] if r["id"] == lock_id)
    return ImportedScores.from_json(pack / row["path"], name=lock_id, expect_sha256=row["sha256"])


@pytest.mark.slow
@pytest.mark.parametrize("rowspace,convention,steps,seed,ids", [
    ("libero5k_v2_d8", "openpi-rowsampler", 20605, 42, ["index.roots_d8", "index.ranges"]),
    ("libero5k_v2_d8.sub1000", "openpi-rowsampler", 4168, 42, ["index.roots_d8", "index.scaling_1000"]),
    ("libero5k_v2_d8.sub2500", "openpi-rowsampler", 10472, 42, ["index.roots_d8", "index.scaling_2500"]),
    ("libero5k_v2_d8", "lerobot-axisrowsampler", 20621, 0, ["lerobot.rows"]),
    ("libero5k_v2_d8.sub1000", "lerobot-axisrowsampler", 4168, 0, ["lerobot.rows_s1000"]),
    ("libero5k_v2_d8.sub2500", "lerobot-axisrowsampler", 10472, 0, ["lerobot.rows_s2500"]),
    ("heldout24_ft", "lerobot-axisrowsampler", 6790, 0, ["lerobot.ft_rows_all"]),
])
def test_bc_exports_the_locked_index_files_and_row_plans(paper_rowspaces, lock, tmp_path, rowspace, convention,
                                                         steps, seed, ids):
    fmt = "openpi-legacy" if convention == "openpi-rowsampler" else "lerobot-legacy"
    b = BC(convention, exports=tuple(legacy_target(i, fmt) for i in ids)).compile(
        None, paper_rowspaces[rowspace], Budget(steps, 64), seed)
    assert _exported(b, fmt, tmp_path) == _want(lock, ids)


DROP = [
    ("approach_aware.chunk.schedule_drop_top50", "approach_aware.chunk.awr_weights", 20605, "openpi-legacy"),
    ("approach_aware.episode.schedule_drop_top50", "approach_aware.episode.awr_weights", 20605, "openpi-legacy"),
    ("object_centric.chunk.schedule_drop_top50", "object_centric.chunk.awr_weights", 20605, "openpi-legacy"),
    ("object_centric.episode.schedule_drop_top50", "object_centric.episode.awr_weights", 20605, "openpi-legacy"),
    ("lerobot.schedule_drop_top_phase_smolvla", "approach_aware.chunk.awr_weights", 20621, "lerobot-legacy"),
    ("lerobot.schedule_drop_top_phase_ep_smolvla", "approach_aware.episode.awr_weights", 20621, "lerobot-legacy"),
    ("lerobot.schedule_drop_top_v2_smolvla", "object_centric.chunk.awr_weights", 20621, "lerobot-legacy"),
    ("lerobot.schedule_drop_top_v2_ep_smolvla", "object_centric.episode.awr_weights", 20621, "lerobot-legacy"),
]


@pytest.mark.slow
@pytest.mark.parametrize("lock_id,weights_id,steps,fmt", DROP)
def test_drop_top50_regenerates_the_locked_schedules(paper_rowspaces, signals_pack, lock, tmp_path,
                                                     lock_id, weights_id, steps, fmt):
    b = DropTop50(exports=(legacy_target(lock_id, fmt),)).compile(
        _scores(signals_pack, lock, weights_id), paper_rowspaces["libero5k_v2_d8"], Budget(steps, 64), 0)
    assert _exported(b, fmt, tmp_path) == _want(lock, [lock_id])


@pytest.mark.slow
@pytest.mark.parametrize("lock_id,granularity,seed", [
    ("lerobot.schedule_drop_random_rand50_row_smolvla", "chunk", 42),
    ("lerobot.schedule_drop_top_rand50_ep_smolvla", "episode", 0),
])
def test_random50_regenerates_the_locked_controls(paper_rowspaces, signals_pack, lock, tmp_path, lock_id,
                                                   granularity, seed):
    t = legacy_target(lock_id, "lerobot-legacy")
    tpl = _scores(signals_pack, lock, "object_centric.chunk.awr_weights")
    b = Random50(granularity, tpl, template_name=t.template_name, exports=(t,)).compile(
        None, paper_rowspaces["libero5k_v2_d8"], Budget(20621, 64), seed)
    assert _exported(b, "lerobot-legacy", tmp_path) == _want(lock, [lock_id])


@pytest.mark.slow
def test_the_lerobot_chunk_drop_names_the_locked_random_control(paper_rowspaces, signals_pack, lock):
    t = legacy_target("lerobot.schedule_drop_random_rand50_row_smolvla", "lerobot-legacy")
    tpl = _scores(signals_pack, lock, "object_centric.chunk.awr_weights")
    ctl = Random50("chunk", tpl, template_name=t.template_name, exports=(t,))
    rows, budget = paper_rowspaces["libero5k_v2_d8"], Budget(20621, 64)
    b, controls = compile_program(DropTop50(control=ctl, control_seed=42),
                                  _scores(signals_pack, lock, "approach_aware.chunk.awr_weights"), rows, budget, 0)
    assert b.controls == (bundle_id(ctl.compile(None, rows, budget, 42)),) == (bundle_id(controls[0]),)
    assert controls[0].diagnostics["keep_fraction"] == b.diagnostics["keep_fraction"]
