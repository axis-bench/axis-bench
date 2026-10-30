"""The traces sample: one task per verb family plus the required ones, staged from the published copy
with nothing identifying."""
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from axis_qvla.labels import traces_sample as ts
from axis_qvla.signals.staging import StagingError

# Synthetic identifying values, assembled so that no file of this repository spells one whole.
ROOTED = "/".join(["", "machine", "x"])                 # an absolute machine path
MAIL = "@".join(["someone", "example.org"])             # an e-mail address
STORE_URL = "://".join(["gs", "bucket/key"])            # an object-store URL
PROMPT = "Put the cup on the plate"
BODIES = ["franka/panda_hand", "cup_0", "plate_0"]
POSE = {"pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]}
TRACE = [{"event": "state", "source": "packed_state_v1_direct", "step_index": 0, "sim_time": 0.2,
          "joint_qpos": {"franka/panda_joint1": 0.0, "cup_0_free": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]},
          "body_pose": {"world": POSE, "franka/panda_hand": POSE, "cup_0": POSE}},
         {"event": "final", "source": "packed_state_v1_direct", "final_sim_time": 0.4, "total_state_frames": 2,
          "state_time_mode": "steps_per_ctrl_post_step", "position_errors": {}, "rotation_errors_deg": {},
          "error_report": []}]
LEAF = {"path": "0", "checker_type": "PositionDeltaChecker", "tracked_body_name": "cup_0",
        "resolved_body_name": "cup_0", "pass_frame": 1, "n_pass_frames": 1, "goal_source": "first_pass",
        "goal_type": "position_delta", "goal_xyz": [0.1, 0.0, 0.0], "delta_axes": ["x"]}


def test_one_task_per_verb_family_fewest_episodes_first_and_required_tasks_are_added():
    tasks = {800: ("Put the cup on the plate", [(0, 1, 9)] * 3), 801: ("Put the bowl away", [(0, 2, 9)]),
             802: ("Rotate the knob", [(0, 3, 9)] * 2), 803: ("Rotate the cap", [(0, 4, 9)] * 2),
             804: ("Squeegee/wipe push", [(0, 5, 9)]), 805: ("Put the pen down", [(0, 6, 9)] * 5)}
    assert ts.select(tasks) == [("put", 801), ("rotate", 802), ("squeegee/wipe", 804)]
    assert ts.select(tasks, require=[805, 802]) == [("put", 801), ("rotate", 802), ("squeegee/wipe", 804),
                                                     ("put", 805)]
    with pytest.raises(ValueError, match="not pretraining tasks"):
        ts.select(tasks, require=[999])


def test_a_task_without_episodes_names_no_family_and_cannot_be_required():
    tasks = {800: ("Put the cup on the plate", [(0, 1, 9)]), 801: ("Fold the towel", [])}
    assert ts.select(tasks) == [("put", 800)]
    assert ts.families(tasks) == ["put"]
    with pytest.raises(ValueError, match=r"required tasks \[801\] have no pretraining episodes"):
        ts.select(tasks, require=[801])


def _manifest(root) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _toy_survey(root):
    """A one-task, one-episode corpus, survey and trace directory, shaped like the published files."""
    corpus = root / "corpus" / "task_800"
    (corpus / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (corpus / "data" / "chunk-000").mkdir(parents=True)
    pd.DataFrame([{"episode_index": 0, "tasks": [PROMPT], "length": 2, "dataset_from_index": 0, "attempt_id": 11,
                   "body_names": BODIES, "object_body": "cup_0", "target_body": "cup_0"}]).to_parquet(
        corpus / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)
    pd.DataFrame({"task_index": [0]}, index=pd.Index([PROMPT])).to_parquet(corpus / "meta" / "tasks.parquet")
    (corpus / "meta" / "info.json").write_text(json.dumps({"fps": 15}))
    (corpus / "meta" / "axis_task_dataset.json").write_text(json.dumps({"dataset_path": ROOTED}))
    pq.write_table(pa.table({"episode_index": [0, 0], "frame_index": [0, 1]}),
                   corpus / "data" / "chunk-000" / "file-000.parquet")
    published = root / "published"
    shutil.copytree(root / "corpus", published)
    roots = root / "roots.json"
    roots.write_text(json.dumps({"version": 2, "base": "corpus", "roots": {"800": "task_800"}}))
    survey = root / "survey"
    (survey / "goals_per_attempt").mkdir(parents=True)
    (survey / "data" / "task_800").mkdir(parents=True)
    (survey / "goals_per_attempt" / "task_800_attempt_11.json").write_text(json.dumps(
        {"n_steps": 2, "top_first_pass_frame": 1, "leaves": [LEAF], "task_dir": ROOTED, "trace": ROOTED,
         "attempt_id": "11"}))
    (survey / "data" / "task_800" / "checker_config.json").write_text(json.dumps(
        {"checker": {"type": "PositionDeltaChecker", "sampleBodyName": "cup_0", "axes": ["x"], "minDeltaX": 0.1}}))
    traces = root / "traces"
    traces.mkdir()
    (traces / "task_800_attempt_11.jsonl").write_text("".join(json.dumps(e) + "\n" for e in TRACE))
    return dict(corpus=root / "corpus", roots_file=roots, survey=survey, traces=traces, published=published,
                expect=_manifest(published))


def test_stage_copies_the_published_files_traces_goals_and_configs_and_drops_machine_paths(tmp_path):
    kw = _toy_survey(tmp_path)
    out = tmp_path / "out"
    man = ts.stage(out=out, require=[800], **kw)
    assert man["tasks"] == [{"task_id": 800, "family": "put", "prompt": PROMPT,
                             "episodes": [{"episode_index": 0, "attempt_id": 11, "length": 2}]}]
    assert man["required"] == [800] and man["families"] == ["put"]
    assert sorted(p.name for p in (out / "traces").iterdir()) == ["task_800_attempt_11.jsonl"]
    assert json.loads((out / "goals_per_attempt" / "task_800_attempt_11.json").read_text()) == {
        "attempt_id": "11", "n_steps": 2, "top_first_pass_frame": 1, "leaves": [LEAF]}
    assert not (out / "corpus" / "task_800" / "meta" / "axis_task_dataset.json").exists()
    for rel, sha in man["files"].items():
        assert hashlib.sha256((out / rel).read_bytes()).hexdigest() == sha
    assert json.loads((out / "manifest.json").read_text()) == man
    assert sorted(_manifest(out)) == sorted([*man["files"], "manifest.json"])
    assert ts.scan_tree(out) == []


def test_stage_refuses_a_corpus_file_that_is_not_the_published_one(tmp_path):
    kw = _toy_survey(tmp_path)
    (kw["published"] / "task_800" / "meta" / "info.json").write_text(json.dumps({"fps": 30}))
    with pytest.raises(ValueError, match="is not the published"):
        ts.stage(out=tmp_path / "out", **kw)
    kw = _toy_survey(tmp_path / "again")
    del kw["expect"]["task_800/data/chunk-000/file-000.parquet"]
    with pytest.raises(ValueError, match="is not the published"):
        ts.stage(out=tmp_path / "again" / "out", **kw)


def test_stage_refuses_an_identifying_field_in_a_trace(tmp_path):
    kw = _toy_survey(tmp_path)
    (kw["traces"] / "task_800_attempt_11.jsonl").write_text(json.dumps({"event": "state", "user_id": 7}) + "\n")
    with pytest.raises(ValueError, match="user_id"):
        ts.stage(out=tmp_path / "out", **kw)
    assert not (tmp_path / "out").exists()


def _plant_goal_leaf(kw):
    p = kw["survey"] / "goals_per_attempt" / "task_800_attempt_11.json"
    p.write_text(json.dumps({**json.loads(p.read_text()), "leaves": [{**LEAF, "note": MAIL}]}))


def _plant_config(kw):
    (kw["survey"] / "data" / "task_800" / "checker_config.json").write_text(
        json.dumps({"checker": {"type": "PositionDeltaChecker", "user_stage": "pro"}}))


def _plant_info(kw):
    for root in (kw["corpus"], kw["published"]):
        (root / "task_800" / "meta" / "info.json").write_text(json.dumps({"fps": 15, "source": STORE_URL}))
    kw["expect"].update(_manifest(kw["published"]))


def _plant_parquet(kw):
    for root in (kw["corpus"], kw["published"]):
        pq.write_table(pa.table({"episode_index": [0, 0], "frame_index": [0, 1], "submit_nonce": [1, 1]}),
                       root / "task_800" / "data" / "chunk-000" / "file-000.parquet")
    kw["expect"].update(_manifest(kw["published"]))


@pytest.mark.parametrize("plant,match", [(_plant_goal_leaf, "note"), (_plant_config, "user_stage"),
                                         (_plant_info, "source"), (_plant_parquet, "submit_nonce")])
def test_stage_refuses_a_planted_identifier_in_any_staged_file_and_writes_nothing(tmp_path, plant, match):
    kw = _toy_survey(tmp_path)
    plant(kw)
    with pytest.raises(ValueError, match=match):
        ts.stage(out=tmp_path / "out", **kw)
    assert not (tmp_path / "out").exists()


def test_scan_tree_finds_a_planted_identifier_in_a_staged_tree(tmp_path):
    kw = _toy_survey(tmp_path)
    out = tmp_path / "out"
    ts.stage(out=out, **kw)
    trace = out / "traces" / "task_800_attempt_11.jsonl"
    trace.write_text(trace.read_text() + json.dumps({"event": "final", "user_id": 7}) + "\n")
    assert [f for f in ts.scan_tree(out) if "user_id" in f]
    assert ts.main(["scan", str(out)]) == 1


def test_a_missing_trace_says_to_restore_it(tmp_path):
    kw = _toy_survey(tmp_path)
    (kw["traces"] / "task_800_attempt_11.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="restore"):
        ts.stage(out=tmp_path / "out", **kw)


def test_a_directory_that_is_not_empty_is_refused(tmp_path):
    kw = _toy_survey(tmp_path)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "stale.json").write_text("{}")
    with pytest.raises(ValueError, match="not empty"):
        ts.stage(out=tmp_path / "out", **kw)
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["stale.json"]


def test_a_failed_write_removes_out_and_a_failed_removal_is_part_of_the_error(tmp_path, monkeypatch):
    kw = _toy_survey(tmp_path)
    writes = []

    def write_bytes(self, data):
        if writes:
            raise OSError(28, "No space left on device")
        writes.append(self)
        return pathlib.Path.open(self, "wb").write(data)

    monkeypatch.setattr(pathlib.Path, "write_bytes", write_bytes)
    with pytest.raises(OSError, match="No space left") as exc:
        ts.stage(out=tmp_path / "out", **kw)
    assert not (tmp_path / "out").exists() and not getattr(exc.value, "__notes__", None)

    writes.clear()

    def rmtree(path, *a, **k):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(ts.shutil, "rmtree", rmtree)
    with pytest.raises(OSError, match="No space left") as exc:
        ts.stage(out=tmp_path / "again", **kw)
    assert any("failed too" in n and "remove it before staging again" in n for n in exc.value.__notes__)


def test_inside_a_published_tree_only_traces_sample_is_written(tmp_path):
    kw = _toy_survey(tmp_path)
    pack = tmp_path / "qvla-v1-corpus"
    (pack / "index").mkdir(parents=True)
    (pack / "SIGNALS.lock.json").write_text(json.dumps({"files": []}))
    with pytest.raises(StagingError):
        ts.stage(out=pack / "index", **kw)
    assert ts.stage(out=pack / "traces_sample", **kw)["tasks"][0]["task_id"] == 800
    assert sorted(p.name for p in (pack / "index").iterdir()) == []
