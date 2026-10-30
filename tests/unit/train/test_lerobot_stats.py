"""A LeRobot dataset's statistics against its features' declared shapes, at both levels."""
from __future__ import annotations

import json
import pathlib
import shutil

from axis_qvla.cli import main
from axis_qvla.train import bind, ids, lerobot_stats, plain_copy, smoke
from axis_qvla.train.recipes import load_recipe, load_template
from tests.unit.train._arms import H, host

STATS = ("min", "max", "mean", "std", "q01", "q99")


def _dataset(root: pathlib.Path, *, state: int = 8, episode_state: int | None = None, name: str = "task_2013",
             body: str = "plate_1") -> pathlib.Path:
    """A converted dataset's meta/: info.json declares an 8-D state and action; its stats.json and
    its meta/episodes columns give the state `state` (and `episode_state`) values per statistic."""
    import pandas as pd

    d = root / name
    (d / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    features = {"observation.state": {"dtype": "float32", "shape": [8], "names": ["state"]},
                "action": {"dtype": "float32", "shape": [8], "names": ["action"]},
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "observation.body_pose": {"dtype": "float32", "shape": [14], "names": ["franka/", body]},
                "observation.images.wrist": {"dtype": "video", "shape": [360, 640, 3],
                                             "names": ["height", "width", "channel"], "info": {"video.fps": 15}}}
    (d / "meta" / "info.json").write_text(json.dumps({"fps": 15, "robot_type": None, "features": features}))

    def stats(width: int) -> dict:
        return {**{s: [0.5] * width for s in STATS}, "count": [10]}

    (d / "meta" / "stats.json").write_text(json.dumps({
        "observation.state": stats(state), "action": stats(8), "timestamp": stats(1),
        "observation.images.wrist": {s: [[[0.5]], [[0.5]], [[0.5]]] for s in STATS}}))
    ep = episode_state if episode_state is not None else state
    rows = {"episode_index": [0, 1], "length": [5, 5], "dataset_from_index": [0, 5]}
    for f, width in (("observation.state", ep), ("action", 8)):
        rows |= {f"stats/{f}/{s}": [[0.5] * width] * 2 for s in STATS}
        rows[f"stats/{f}/count"] = [[5], [5]]
    pd.DataFrame(rows).to_parquet(d / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    return d


def test_9d_statistics_of_an_8d_feature_are_refused_at_both_levels(tmp_path):
    d = _dataset(tmp_path, state=9)
    dataset_level, episode_level = lerobot_stats.stats_problems(d)
    assert dataset_level.startswith(f"{d}/meta/stats.json: 'observation.state' max has [9] values but meta/info.json "
                                    f"declares the shape [8]: the statistics were not recomputed when the feature "
                                    f"was converted")
    assert episode_level.startswith(f"{d}/meta/episodes (episode 0): 'observation.state' ")
    assert "has [9] values but meta/info.json declares the shape [8]" in episode_level
    assert "convert_droid_actions" in episode_level                     # what rebuilds it
    # The normaliser reads only meta/stats.json; a merge carries the per-episode columns forward.
    assert "normaliser reads these" in dataset_level and "a merge copies these per-episode columns" in episode_level


def test_a_correct_copy_passes_and_only_the_named_features_are_checked(tmp_path):
    assert lerobot_stats.stats_problems(_dataset(tmp_path / "ok")) == []    # video stats, count: not shapes
    stale = _dataset(tmp_path / "stale", state=9)
    assert lerobot_stats.stats_problems(stale, ["action"]) == []
    assert len(lerobot_stats.stats_problems(stale, ["observation.state", "not.a.feature"])) == 2
    assert lerobot_stats.stats_problems(tmp_path / "absent") == []          # nothing to compare


def test_stale_episode_statistics_alone_are_refused(tmp_path):
    """A merge copies the per-episode columns unchanged, so a source whose stats.json is correct but
    whose episode table is stale still hands its stale statistics to every copy merged from it."""
    (problem,) = lerobot_stats.stats_problems(_dataset(tmp_path, state=8, episode_state=9))
    assert "meta/episodes (episode 0): 'observation.state'" in problem and "has [9] values" in problem


def test_merge_check_refuses_an_input_with_stale_statistics(tmp_path, capsys):
    good = _dataset(tmp_path / "a", name="task_2013")
    stale = _dataset(tmp_path / "b", state=9, name="task_2232")
    problems = plain_copy.merge_problems([good, stale])
    assert len(problems) == 2 and all("'observation.state'" in p and str(stale) in p for p in problems)
    assert main(["train", "merge-check", str(good), str(stale)]) == 2
    assert capsys.readouterr().out.count("FAIL  ") == 2
    fixed = _dataset(tmp_path / "c", name="task_2232")
    assert plain_copy.merge_problems([good, fixed]) == []


def test_binding_refuses_a_merged_copy_whose_statistics_do_not_fit_its_features(tmp_path):
    _, paths = host(tmp_path)
    rowspace = f"{smoke.SMOKE_ROWSPACE}#sha256:{H}"
    arms = smoke.fixture_arms(rowspace, {"pi05": f"sha256:{H}", "lerobot": f"sha256:{H}"}, rowspace, 609)
    copy_dir = ids.rowspace_dirs(paths, rowspace)["lerobot_dataset"]
    _dataset(copy_dir.parent, state=9, name=copy_dir.name)
    for arm in ("smolvla", "xvla"):
        problems = bind._check_rowspace(arms[arm], paths, None)              # noqa: SLF001
        assert len(problems) == 2 and all(p.startswith("rowspace: ") and "'observation.state'" in p for p in problems)
    shutil.rmtree(copy_dir)                                                  # rebuilt from a fixed conversion
    _dataset(copy_dir.parent, name=copy_dir.name)
    assert bind._check_rowspace(arms["smolvla"], paths, None) == []          # noqa: SLF001


def test_the_features_a_lerobot_policy_normalises_are_its_state_and_action():
    for preset in ("smolvla.pretrain.smoke", "xvla.pretrain.smoke", "smolvla.finetune.paper", "xvla.pretrain.paper"):
        assert lerobot_stats.trained_features(load_template(load_recipe(preset))) == ["action", "observation.state"]
