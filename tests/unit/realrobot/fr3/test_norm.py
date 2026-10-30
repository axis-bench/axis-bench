"""Norm statistics on a run's own training set, in the model's state order, and the check that they are."""
from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from axis_qvla.realrobot.fr3 import checks, norm
from axis_qvla.realrobot.realdata.lerobot_v21 import Dataset
from tests.unit.realrobot.fr3.fr3_sets import fr3_set


def test_fr3_norm_stats_are_in_the_models_state_order(tmp_path):
    ds = Dataset.open(fr3_set(tmp_path / "fr3_392"))
    stats = norm.compute_norm(ds.root, "fr3")["norm_stats"]
    st = np.concatenate([np.asarray(ds.read_episode(e, ["state"]).column("state").combine_chunks().flatten()).reshape(-1, 8)
                         for e in range(len(ds.episodes))])
    np.testing.assert_allclose(stats["state"]["mean"], st[:, [1, 2, 3, 4, 5, 6, 7, 0]].mean(0), rtol=0, atol=1e-6)
    assert set(stats["state"]) == {"mean", "std", "q01", "q99"} and set(stats) == {"state", "actions"}


def test_fr3_actions_keep_their_stored_order(tmp_path):
    ds = Dataset.open(fr3_set(tmp_path / "fr3_392"))
    stats = norm.compute_norm(ds.root, "fr3")["norm_stats"]
    ac = np.concatenate([np.asarray(ds.read_episode(e, ["actions"]).column("actions").combine_chunks().flatten()).reshape(-1, 8)
                         for e in range(len(ds.episodes))])
    np.testing.assert_allclose(stats["actions"]["q99"], np.quantile(ac, 0.99, axis=0), rtol=0, atol=1e-6)


def test_check_norm_accepts_its_own_output_and_refuses_stored_order(tmp_path):
    ds = Dataset.open(fr3_set(tmp_path / "fr3_392"))
    norm.write_norm(tmp_path / "good.json", norm.compute_norm(ds.root, "fr3"))
    assert checks.check_norm(tmp_path / "good.json", ds.root, "fr3") == []
    doc = json.loads((tmp_path / "good.json").read_text())
    s = doc["norm_stats"]["state"]
    for k in s:
        s[k] = s[k][-1:] + s[k][:-1]          # gripper first again: the stored order
    (tmp_path / "bad.json").write_text(json.dumps(doc))
    assert any("state" in p for p in checks.check_norm(tmp_path / "bad.json", ds.root, "fr3"))


def test_check_norm_names_a_file_it_cannot_use(tmp_path):
    ds = Dataset.open(fr3_set(tmp_path / "fr3_392"))
    assert checks.check_norm(tmp_path / "absent.json", ds.root, "fr3")[0].startswith(f"{tmp_path / 'absent.json'}: cannot read")
    (tmp_path / "narrow.json").write_text(json.dumps({"norm_stats": {"state": {"mean": [0.0] * 7, "std": [1.0] * 7},
                                                                      "actions": {}}}))
    problems = checks.check_norm(tmp_path / "narrow.json", ds.root, "fr3")
    assert "state: statistics of width 7, the dataset has 8" in problems
    assert any(p.startswith("actions:") for p in problems)


def _droid8d_corpus(root, x):
    d = root / "task_9001" / "data" / "chunk-000"
    d.mkdir(parents=True)
    col = lambda a: pa.FixedSizeListArray.from_arrays(pa.array(a.reshape(-1)), 8)   # noqa: E731
    pq.write_table(pa.table({"observation.state": col(x), "action": col(x / 10)}), d / "file-000.parquet")
    (root / "roots.json").write_text(json.dumps({"version": 2, "base": ".", "roots": {"9001": "task_9001"}}))
    return root


def test_the_rendered_corpus_norm_reads_the_droid8d_view(tmp_path):
    x = np.arange(48, dtype=np.float32).reshape(6, 8)
    root = _droid8d_corpus(tmp_path / "d8", x)
    stats = norm.compute_norm(root, "realalign_sim")["norm_stats"]
    np.testing.assert_allclose(stats["actions"]["mean"], (x / 10).mean(0), atol=1e-6)
    np.testing.assert_allclose(stats["state"]["mean"], x.mean(0), atol=1e-6)
    norm.write_norm(tmp_path / "n" / "norm_stats.json", norm.compute_norm(root, "realalign_sim"))
    assert checks.check_norm(tmp_path / "n" / "norm_stats.json", root, "realalign_sim") == []


def test_an_embodiment_without_a_reader_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no norm reader"):
        norm.compute_norm(tmp_path, "ur5")


def test_a_task_of_the_rendered_view_without_data_is_named(tmp_path):
    root = _droid8d_corpus(tmp_path / "d8", np.zeros((3, 8), np.float32))
    (root / "roots.json").write_text(json.dumps({"version": 2, "base": ".", "roots": {"9001": "task_9001",
                                                                                     "9002": "task_9002"}}))
    with pytest.raises(ValueError, match=r"cannot read the realalign_sim training set \(FileNotFoundError.*task_9002.data"):
        norm.compute_norm(root, "realalign_sim")
    (root / "task_9002" / "data").mkdir(parents=True)
    with pytest.raises(ValueError, match=r"task_9002.data holds no parquet files"):
        norm.compute_norm(root, "realalign_sim")


def _good(tmp_path):
    ds = Dataset.open(fr3_set(tmp_path / "fr3_392"))
    norm.write_norm(tmp_path / "good.json", norm.compute_norm(ds.root, "fr3"))
    return ds, json.loads((tmp_path / "good.json").read_text())


def _write(tmp_path, doc, name="planted.json"):
    (tmp_path / name).write_text(json.dumps(doc))
    return tmp_path / name


def _stored_order(s):
    s["q01"], s["q99"] = s["q01"][-1:] + s["q01"][:-1], s["q99"][-1:] + s["q99"][:-1]


def _wide(s):
    s["q01"], s["q99"] = [-100.0] * len(s["q01"]), [100.0] * len(s["q99"])


def _swapped(s):
    s["q01"], s["q99"] = s["q99"], s["q01"]


@pytest.mark.parametrize("plant, named", [
    (_stored_order, "state: q99 is not this dataset's in dims [0, 7]"),   # the gripper's land on q0, q6's on the gripper
    (_wide, "state: q01 is not this dataset's in dims [0, 1, 2, 3, 4, 5, 6, 7]"),
    (_swapped, "state: q01 above q99 in dims [0, 1, 2, 3, 4, 5, 6, 7]"),
])
def test_the_quantiles_pi05_normalises_with_are_checked(tmp_path, plant, named):
    """Mean and std correct, quantiles wrong: pi0.5 scales by (x - q01) / (q99 - q01), so the file is wrong."""
    ds, doc = _good(tmp_path)
    plant(doc["norm_stats"]["state"])
    problems = checks.check_norm(_write(tmp_path, doc), ds.root, "fr3")
    assert any(p.startswith(named) for p in problems) and not any(p.startswith("actions") for p in problems), problems


def _histogram_quantile(x, q, bins=5000):
    """openpi's estimate: the bin edge where the cumulative count reaches q of a histogram over [min, max]."""
    edges = np.linspace(x.min(), x.max(), bins + 1)
    counts = np.cumsum(np.histogram(x, edges)[0])
    return float(edges[np.searchsorted(counts, q * len(x))])


def test_a_histogram_estimate_of_the_quantiles_passes(tmp_path):
    ds, doc = _good(tmp_path)
    state, actions = norm.columns(ds.root, "fr3")
    for name, x in (("state", state), ("actions", actions)):
        doc["norm_stats"][name]["q01"] = [_histogram_quantile(x[:, d], 0.01) for d in range(x.shape[1])]
        doc["norm_stats"][name]["q99"] = [_histogram_quantile(x[:, d], 0.99) for d in range(x.shape[1])]
    assert checks.check_norm(_write(tmp_path, doc), ds.root, "fr3") == []


@pytest.mark.parametrize("edit, named", [
    (lambda d: d["norm_stats"]["state"].update(mean=["a"] * 8), "state: mean is not a list of 8 numbers"),
    (lambda d: d["norm_stats"]["state"].update(std=[True] * 8), "state: std is not a list of 8 numbers"),
    (lambda d: d["norm_stats"].update(state=[0.0] * 8), "state: the statistics are a list, not a table"),
    (lambda d: d.update(norm_stats=[0.0]), "norm_stats is a list, not a table"),
    (lambda d: d["norm_stats"]["actions"].pop("q99"), "actions: q99 is not a list of 8 numbers"),
    (lambda d: d["norm_stats"].pop("actions"), "actions: no statistics"),
])
def test_malformed_norm_content_is_a_named_problem_not_a_traceback(tmp_path, edit, named):
    ds, doc = _good(tmp_path)
    edit(doc)
    problems = checks.check_norm(_write(tmp_path, doc), ds.root, "fr3")
    assert any(named in p for p in problems), problems


def test_a_file_of_another_width_is_one_problem_per_table(tmp_path):
    ds, doc = _good(tmp_path)
    for name in ("state", "actions"):
        doc["norm_stats"][name] = {k: [0.0] * 32 for k in ("mean", "std", "q01", "q99")}
    assert checks.check_norm(_write(tmp_path, doc), ds.root, "fr3") == [
        "state: statistics of width 32, the dataset has 8", "actions: statistics of width 32, the dataset has 8"]


def test_a_norm_file_that_is_not_a_table_says_so(tmp_path):
    ds, _doc = _good(tmp_path)
    path = _write(tmp_path, [{"norm_stats": {}}])
    assert checks.check_norm(path, ds.root, "fr3") == [
        f"{path}: cannot read norm statistics (the document is a list, not a table)"]


@pytest.mark.parametrize("roots", [[1], {"version": 2, "base": ".", "roots": ["task_9001"]}, {"version": 2},
                                   {"version": 2, "base": ".", "roots": {"9001": 5}}])
def test_a_rendered_views_roots_index_of_another_shape_is_named(tmp_path, roots):
    root = tmp_path / "v"
    _droid8d_corpus(root, np.zeros((4, 8)))
    (root / "roots.json").write_text(json.dumps(roots))
    with pytest.raises(ValueError, match="roots.json is not a roots index"):
        norm.compute_norm(root, "realalign_sim")
