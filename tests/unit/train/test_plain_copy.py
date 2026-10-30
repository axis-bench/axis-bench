"""A plain merged LeRobot copy's inputs, checked the way the LeRobot line's merge compares them."""
from __future__ import annotations

import copy
import json
import pathlib
import shutil

from axis_qvla.cli import main
from axis_qvla.train import bind, ids, launch, plain_copy, smoke
from axis_qvla.train.armspec import dumps_armspec
from tests.unit.train._arms import H, host

SCENES = {2013: ["plate_1", "bowl_2", "plum_2", "lychee_4"], 2232: ["plate_3", "spaghetti_4", "fork_3", "tomato_0"]}
VIDEO = {"dtype": "video", "shape": [360, 640, 3], "names": ["height", "width", "channel"],
         "info": {"video.fps": 15, "video.codec": "h264", "video.is_depth_map": False}}


def _features(tid: int) -> dict:
    return {"observation.state": {"dtype": "float32", "shape": [8], "names": ["state"]},
            "action": {"dtype": "float32", "shape": [8], "names": ["action"]},
            "observation.images.wrist": copy.deepcopy(VIDEO),
            "observation.body_pose": {"dtype": "float32", "shape": [56], "names": ["franka/", *SCENES[tid]]}}


def _task(root: pathlib.Path, tid: int, *, features: dict | None = None, fps: int = 15) -> pathlib.Path:
    """A task directory's meta/info.json, as the corpus writes it: the body poses name its own scene."""
    d = root / f"task_{tid}"
    (d / "meta").mkdir(parents=True)
    (d / "meta" / "info.json").write_text(json.dumps(
        {"fps": fps, "robot_type": None, "features": features or _features(tid)}, indent=2))
    return d


def _pair(root: pathlib.Path) -> list[pathlib.Path]:
    return [_task(root, tid) for tid in SCENES]


def test_tasks_of_different_scenes_are_refused_with_the_feature_the_tasks_and_the_remedy(tmp_path, capsys):
    tasks = _pair(tmp_path)
    (problem,) = plain_copy.merge_problems(tasks)
    assert problem.startswith("feature 'observation.body_pose': its names differ between the tasks")
    assert "task_2013: plate_1, bowl_2, plum_2, lychee_4; task_2232: plate_3, spaghetti_4, fork_3, tomato_0" in problem
    assert "axis_qvla.train.plain_copy.clear_differing_names" in problem and "never edit the published" in problem
    assert main(["train", "merge-check", *map(str, tasks)]) == 2
    out = capsys.readouterr().out
    assert out.startswith("FAIL  feature 'observation.body_pose'") and out.count("FAIL  ") == 1


def test_copies_whose_differing_names_are_cleared_merge(tmp_path, capsys):
    copies = [shutil.copytree(t, tmp_path / "copies" / t.name) for t in _pair(tmp_path / "corpus")]
    assert plain_copy.clear_differing_names(copies) == ["observation.body_pose"]
    infos = [json.loads((d / "meta" / "info.json").read_text()) for d in copies]
    assert infos[0] == infos[1] and infos[0]["features"]["observation.body_pose"]["names"] is None
    assert infos[0]["features"]["observation.state"]["names"] == ["state"]                 # equal: kept
    assert plain_copy.merge_problems(copies) == [] and plain_copy.clear_differing_names(copies) == []
    assert main(["train", "merge-check", *map(str, copies)]) == 0
    assert capsys.readouterr().out == "OK    2 task directories have equal features; a plain merge accepts them\n"
    # The originals are untouched.
    assert len(plain_copy.merge_problems(sorted((tmp_path / "corpus").iterdir()))) == 1


def test_a_difference_names_cannot_fix_is_refused_as_not_one_dataset(tmp_path):
    shape = _features(2232)
    shape["observation.state"]["shape"] = [9]
    del shape["action"]
    tasks = [_task(tmp_path, 2013), _task(tmp_path, 2232, features=shape, fps=30)]
    problems = plain_copy.merge_problems(tasks)
    assert problems[0].startswith("fps differs between the tasks ({'task_2013': 15, 'task_2232': 30})")
    assert any(p.startswith("feature 'action': task_2232 has none and task_2013 has it") for p in problems)
    state = next(p for p in problems if p.startswith("feature 'observation.state'"))
    assert "dtype or shape differs between the tasks (task_2013 | task_2232)" in state
    assert all("not one dataset" in p for p in problems if "body_pose" not in p)


def test_what_the_merge_ignores_is_ignored(tmp_path):
    """The video encoder settings and the depth flag's two spellings do not stop the merge."""
    other = _features(2013)
    other["observation.images.wrist"]["info"] = {"video.fps": 15, "video.codec": "h264", "video.crf": 30,
                                                 "video.preset": "slow", "is_depth_map": False}
    tasks = [_task(tmp_path / "a", 2013), _task(tmp_path / "b", 2013, features=other)]
    assert plain_copy.merge_problems(tasks) == []
    other["observation.images.wrist"]["info"]["video.codec"] = "av1"
    assert len(plain_copy.merge_problems([tasks[0], _task(tmp_path / "c", 2013, features=other)])) == 1


def test_binding_names_the_merge_refusal_while_a_plain_copy_is_not_built(tmp_path, capsys):
    hosts, paths = host(tmp_path)
    rowspace = f"{smoke.SMOKE_ROWSPACE}#sha256:{H}"
    arms = smoke.fixture_arms(rowspace, {"pi05": f"sha256:{H}", "lerobot": f"sha256:{H}"}, rowspace, 609)
    dirs = ids.rowspace_dirs(paths, rowspace)
    tasks = _pair(dirs["corpus_base"])
    dirs["roots"].parent.mkdir(parents=True)
    dirs["roots"].write_text(json.dumps({"version": 2, "base": "../corpus_d8",
                                         "roots": {"2232": "task_2232", "2013": "task_2013"}}))
    missing, refused = bind._check_rowspace(arms["smolvla"], paths, None)      # noqa: SLF001
    assert missing == f"rowspace: {dirs['lerobot_dataset']} is not there"
    assert refused.startswith(f"rowspace: the pinned LeRobot fork refuses a plain merge of the tasks in "
                              f"{dirs['roots']}: feature 'observation.body_pose': its names differ")
    arm = tmp_path / "smolvla.toml"
    arm.write_text(dumps_armspec(arms["smolvla"]))
    assert main(["train", "--render", str(arm), "--host", str(hosts), "--smoke"]) == 2      # listed with the other refusals
    assert "refuses a plain merge" in capsys.readouterr().out
    plain_copy.clear_differing_names(tasks)                                       # the fixture's copies
    assert bind._check_rowspace(arms["smolvla"], paths, None) == [missing]     # noqa: SLF001
    dirs["lerobot_dataset"].mkdir()
    assert bind._check_rowspace(arms["xvla"], paths, None) == []                # noqa: SLF001


def test_a_plain_copy_qvla_train_merged_is_refused_once_its_roots_index_changes(tmp_path):
    """The copy's frames do not say which task set they were merged from; the record qvla train writes
    after its merge does, and binding compares it with the roots index on disk."""
    _, paths = host(tmp_path)
    rowspace = f"{smoke.SMOKE_ROWSPACE}#sha256:{H}"
    arms = smoke.fixture_arms(rowspace, {"pi05": f"sha256:{H}", "lerobot": f"sha256:{H}"}, rowspace, 609)
    dirs = ids.rowspace_dirs(paths, rowspace)
    plain_copy.clear_differing_names(_pair(dirs["corpus_base"]))
    dirs["roots"].parent.mkdir(parents=True)
    dirs["roots"].write_text(json.dumps({"version": 2, "base": "../corpus_d8",
                                         "roots": {"2232": "task_2232", "2013": "task_2013"}}))
    dirs["lerobot_dataset"].mkdir(parents=True)
    assert bind._check_rowspace(arms["smolvla"], paths, None) == []            # noqa: SLF001  built by hand
    launch.record_merge(arms["smolvla"], paths)                                # what qvla train does after its merge
    assert json.loads((dirs["lerobot_dataset"] / plain_copy.PLAIN_RECORD).read_text())["roots_sha256"]
    assert bind._check_rowspace(arms["smolvla"], paths, None) == []            # noqa: SLF001
    dirs["roots"].write_text(json.dumps({"version": 2, "base": "../corpus_d8", "roots": {"2013": "task_2013"}}))
    (problem,) = bind._check_rowspace(arms["xvla"], paths, None)               # noqa: SLF001
    assert problem.startswith(f"rowspace: {dirs['lerobot_dataset']} was merged from a roots index with sha256 ")
    assert f"remove {dirs['lerobot_dataset']} yourself" in problem
    (dirs["lerobot_dataset"] / plain_copy.PLAIN_RECORD).write_text("{}")
    assert bind._check_rowspace(arms["xvla"], paths, None) == [               # noqa: SLF001
        f"rowspace: {dirs['lerobot_dataset'] / plain_copy.PLAIN_RECORD} is not a qvla.plain_copy/1 record"]


def test_a_roots_index_is_walked_in_task_id_order(tmp_path):
    roots = tmp_path / "roots.json"
    roots.write_text(json.dumps({"version": 2, "base": ".", "roots": {"1000": "task_1000", "800": "task_800"}}))
    assert plain_copy.roots_task_dirs(roots, tmp_path / "c") == [tmp_path / "c/task_800", tmp_path / "c/task_1000"]
    roots.write_text("[]")
    assert plain_copy.rowspace_problems({"roots": roots, "corpus_base": tmp_path})[0].startswith(
        "rowspace: the plain copy's inputs cannot be checked")
