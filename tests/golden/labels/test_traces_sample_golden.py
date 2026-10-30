"""On the traces sample (--traces-sample): its manifest verifies and nothing in it identifies anybody;
its corpus files are the published ones (--episodes, the published pretraining episodes); the label pipeline
reproduces the published per-frame labels from the sample alone, the position-delta bounds included;
with the scenes (--scene-cache) it reproduces the contacts; the task checkers reproduce the goal
records; and the sample's episodes reproduce their entries of the locked chunk weights and ranges
(--pretraining-data)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import shutil

import numpy as np
import pandas as pd
import pytest

from axis_qvla.data import episode_names as en
from axis_qvla.data.index.ranges import task_ranges
from axis_qvla.labels import reward
from axis_qvla.labels import traces_sample as ts
from axis_qvla.labels.contacts import CONTACT_COLUMNS, ContactModel
from axis_qvla.labels.goals import extract_for_attempt
from axis_qvla.labels.phase import phase_reward
from axis_qvla.labels.pipeline import label_episode, trace_states
from axis_qvla.labels.progress import pick_reward_leaf
from axis_qvla.signals.blocks.awr import awr_weight, build_episode_weights, episode_key
from axis_qvla.signals.methods.awr_chunk import _neutralised_mask_clamped, phase_delta_clamped

REPO = pathlib.Path(__file__).resolve().parents[3]
# The one goal-record field compared to a declared bound rather than exactly: d_init_goal, a placing
# leaf's float64 distance from its initial object position to its goal, which on an AVX-512 host can
# differ from the archived value in the last place. It also decides whether a leaf is a trivial
# placement; that decision is written to goal_source, which -- like every other field -- is exact.
GOAL_FIELD_REL_TOL = {"d_init_goal": 1e-12}
# Sample episodes whose reward leaf is a position-delta leaf without per-axis bounds, so that their
# reward takes the bounds from the sample's data/task_<id>/checker_config.json.
BOUNDS_FROM_CHECKER_CONFIG = 44


def _episodes(sample):
    man = json.loads((sample / "manifest.json").read_text())
    for t in man["tasks"]:
        tid = t["task_id"]
        files = sorted((sample / "corpus" / f"task_{tid}" / "data").rglob("*.parquet"))
        frames = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        for e in t["episodes"]:
            g = frames[frames["episode_index"] == e["episode_index"]].sort_values("frame_index").reset_index(drop=True)
            assert len(g) == e["length"]
            yield tid, e, g


def _inputs(sample, tid, e):
    trace = sample / "traces" / f"task_{tid}_attempt_{e['attempt_id']}.jsonl"
    goals = json.loads((sample / "goals_per_attempt" / f"task_{tid}_attempt_{e['attempt_id']}.json").read_text())
    return trace, goals


def _same_field(key: str, live, archived) -> bool:
    tol = GOAL_FIELD_REL_TOL.get(key)
    if tol is None or not (isinstance(live, float) and isinstance(archived, float)):
        return live == archived
    return math.isclose(live, archived, rel_tol=tol, abs_tol=0.0)


@pytest.mark.slow
def test_the_manifest_verifies_covers_every_family_and_identifies_nobody(traces_sample):
    man = json.loads((traces_sample / "manifest.json").read_text())
    entries = [pathlib.Path(d, n) for d, dirs, files in os.walk(traces_sample) for n in dirs + files]
    assert [p for p in entries if p.is_symlink()] == []
    on_disk = sorted(p.relative_to(traces_sample).as_posix() for p in entries if p.is_file())
    assert on_disk == sorted([*man["files"], "manifest.json"])
    for rel, sha in man["files"].items():
        assert hashlib.sha256((traces_sample / rel).read_bytes()).hexdigest() == sha, rel
    assert len(man["families"]) == 20 and {t["family"] for t in man["tasks"]} == set(man["families"])
    ids = [t["task_id"] for t in man["tasks"]]
    assert len(ids) == len(set(ids)) <= len(man["families"]) + len(man["required"])
    assert set(man["required"]) <= set(ids)
    for t in man["tasks"]:
        tid = t["task_id"]
        assert f"data/task_{tid}/checker_config.json" in man["files"], tid
        assert any(rel.startswith(f"corpus/task_{tid}/meta/episodes/") for rel in man["files"]), tid
        for e in t["episodes"]:
            for rel in (f"traces/task_{tid}_attempt_{e['attempt_id']}.jsonl",
                        f"goals_per_attempt/task_{tid}_attempt_{e['attempt_id']}.json"):
                assert rel in man["files"], rel
    assert ts.scan_tree(traces_sample) == []


@pytest.mark.slow
def test_the_staged_corpus_files_are_the_published_ones(traces_sample, corpus):
    """--episodes here is the PUBLISHED per-task corpus (qvla-v1-corpus episodes/), not a local build."""
    man = json.loads((traces_sample / "manifest.json").read_text())
    staged = [rel for rel in man["files"] if rel.startswith("corpus/")]
    assert staged
    for rel in staged:
        assert (traces_sample / rel).read_bytes() == (corpus / rel[len("corpus/"):]).read_bytes(), rel


@pytest.mark.slow
def test_rewards_validity_success_anchors_and_poses_reproduce_the_published_labels(traces_sample):
    n = 0
    for tid, e, g in _episodes(traces_sample):
        trace, goals = _inputs(traces_sample, tid, e)
        cols = label_episode(trace_states(trace), goals, n_rows=len(g), trace_path=trace)
        np.testing.assert_allclose(cols["reward"], g["reward"].to_numpy(float), rtol=0, atol=1e-12)
        for c in ("reward_valid", "success", "is_anchor"):
            assert (cols[c] == g[c].to_numpy(bool)).all(), (tid, e, c)
        # The corpus stores poses and joint positions as float32: the pipeline's columns, cast to
        # float32, are those bytes.
        for c in ("observation.body_pose", "observation.eef_pose", "observation.joint_qpos"):
            published = np.stack(g[c].to_numpy())
            assert published.dtype == np.float32, (tid, e, c, published.dtype)
            np.testing.assert_array_equal(np.asarray(cols[c]).reshape(len(g), -1).astype(np.float32),
                                          published.reshape(len(g), -1), err_msg=f"{tid} {e} {c}")
        n += 1
    man = json.loads((traces_sample / "manifest.json").read_text())
    assert n == sum(len(t["episodes"]) for t in man["tasks"]) > 0


@pytest.mark.slow
def test_position_delta_rewards_take_their_bounds_from_the_sample_alone(traces_sample, tmp_path):
    """The episodes whose reward leaf has no per-axis bounds (their labels are checked above) find
    them in the sample: the lookup resolves to the sample's own checker configuration, a file the
    manifest lists, and the same trace with no checker configuration beside it refuses to label."""
    man = json.loads((traces_sample / "manifest.json").read_text())
    n = 0
    for tid, e, g in _episodes(traces_sample):
        trace, goals = _inputs(traces_sample, tid, e)
        leaf = pick_reward_leaf(goals)
        if leaf is None or leaf.get("goal_type") != "position_delta" or leaf.get("axis_bounds"):
            continue
        cfg = reward._checker_config_path(trace)
        assert cfg == traces_sample / "data" / f"task_{tid}" / "checker_config.json", cfg
        assert hashlib.sha256(cfg.read_bytes()).hexdigest() == man["files"][f"data/task_{tid}/checker_config.json"]
        alone = tmp_path / f"task_{tid}_{e['episode_index']}" / "traces" / trace.name
        alone.parent.mkdir(parents=True)
        shutil.copyfile(trace, alone)
        with pytest.raises(reward.AxisBoundsError, match="checker_config.json: does not exist"):
            label_episode(trace_states(alone), goals, n_rows=len(g), trace_path=alone)
        n += 1
    assert n == BOUNDS_FROM_CHECKER_CONFIG


@pytest.mark.slow
def test_contacts_reproduce_the_published_labels(traces_sample, scene_cache):
    models = {}
    for tid, e, g in _episodes(traces_sample):
        model = models.setdefault(tid, ContactModel(str(scene_cache / f"_scene_task_{tid}" / "scene.xml")))
        trace, goals = _inputs(traces_sample, tid, e)
        cols = label_episode(trace_states(trace), goals, n_rows=len(g), trace_path=trace, contact_model=model)
        for c in CONTACT_COLUMNS:
            assert (cols[c] == g[c].to_numpy(bool)).all(), (tid, e, c)


@pytest.mark.slow
def test_the_task_checkers_reproduce_the_goal_records(traces_sample):
    """Every field of every archived goal record is reproduced, exactly but for GOAL_FIELD_REL_TOL.
    The archive predates the extractor's per-axis bounds on PositionDeltaChecker leaves
    (`axis_bounds`); when a leaf lacks them the reward recovers the same bounds from the task's
    checker configuration, so the new field must equal those."""
    n_records = n_leaves = n_bounds = n_tolerated = 0
    worst = 0.0
    for tid, e, _g in _episodes(traces_sample):
        trace, goals = _inputs(traces_sample, tid, e)
        got = extract_for_attempt(traces_sample / "data" / f"task_{tid}", trace)
        assert {k: v for k, v in got.items() if k != "leaves"} == {k: goals[k] for k in got if k != "leaves"}, (tid, e)
        assert len(got["leaves"]) == len(goals["leaves"]), (tid, e)
        for live, archived in zip(got["leaves"], goals["leaves"]):
            where = (tid, e["episode_index"], archived.get("path"))
            assert set(archived) <= set(live), (*where, set(archived) - set(live))
            assert live.get("goal_source") == archived.get("goal_source"), where
            for k, v in archived.items():
                assert _same_field(k, live[k], v), (*where, k, live[k], v)
                if k in GOAL_FIELD_REL_TOL and live[k] != v:
                    n_tolerated += 1
                    worst = max(worst, abs(live[k] - v) / abs(v))
            extra = set(live) - set(archived)
            assert extra <= {"axis_bounds"}, (*where, extra)
            if extra:
                assert live["axis_bounds"] == reward._maybe_recover_axis_bounds(archived, trace), where
                n_bounds += 1
            n_leaves += 1
        n_records += 1
    print(f"goal records: {n_records} records, {n_leaves} leaves; every field exact but {n_tolerated} "
          f"d_init_goal value(s) within {GOAL_FIELD_REL_TOL['d_init_goal']:g} relative (largest {worst:.2e}); "
          f"axis_bounds recovered on {n_bounds} leaves")
    assert n_bounds > 0


@pytest.mark.slow
def test_the_sample_episodes_reproduce_the_locked_chunk_weights_and_ranges(traces_sample, signals_pack, lock):
    rows = {r["id"]: r for r in lock["files"]}
    aa = json.loads((signals_pack / rows["approach_aware.chunk.awr_weights"]["path"]).read_text())
    oc = json.loads((signals_pack / rows["object_centric.chunk.awr_weights"]["path"]).read_text())
    ranges = json.loads((signals_pack / rows["index.ranges"]["path"]).read_text())
    scales = aa["provenance"]["phase_balance"]
    names = en.load_table(REPO / "conf" / "signals" / "episode_body_names.json")
    for tid, e, g in _episodes(traces_sample):
        key = episode_key(tid, e["episode_index"])
        valid = g["reward_valid"].to_numpy(bool)
        w = build_episode_weights(tid, e["episode_index"], g["reward"].to_numpy(float), valid).weights
        assert w.astype(np.float64).tolist() == oc["weights"][key], key
        body_names, obj = names[key]
        res = phase_reward(g, {"body_names": body_names, "object_body": obj})
        if res is None:
            assert set(aa["weights"][key]) == {1.0}, key
            continue
        curve, t0 = res
        d, _ok = phase_delta_clamped(np.asarray(curve, float), int(t0), 10)
        d[:t0] *= scales["scale1"]
        d[t0:] *= scales["scale2"]
        w = np.where(_neutralised_mask_clamped(valid, int(t0), 10), 1.0, awr_weight(d, 10.0, 2.0)).astype(np.float32)
        assert w.astype(np.float64).tolist() == aa["weights"][key], key
    man = json.loads((traces_sample / "manifest.json").read_text())
    for t in man["tasks"]:
        for key, spans in task_ranges(traces_sample / "corpus" / f"task_{t['task_id']}").items():
            assert spans == ranges[key], key
