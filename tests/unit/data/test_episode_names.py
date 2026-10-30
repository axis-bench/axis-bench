"""The approach-aware reward's object binding is a pinned input: shipped, digested, enforced.

Eq. (3)'s d_t is measured to the body `build_awr_weights --binding names` finds through the
per-episode `body_names`/`object_body` meta, which the corpus fingerprint does not cover. A corpus
copy whose meta carries other names would rebuild different weights under the same fingerprint,
so the table pins them. These tests pin the table format, the shipped table's digest,
and the builder's refusal; the corpus here is synthetic.

Offline tier: numpy/pandas only. No private data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from axis_qvla.signals.methods import awr_chunk as baw
from axis_qvla.data import episode_names as en

REPO = Path(__file__).resolve().parents[3]
GENERATORS = REPO / "conf" / "signals" / "generators.json"

TRUE = ["franka/hand", "plate_0", "cup_0"]        # the order the pose columns were stacked in
WRONG = ["franka/hand", "cup_0", "plate_0"]       # same length, wrong order: binds the plate


def _episode(T=80, tc=40):
    """cup_0 (column 2) is approached, grasped at tc and lifted; plate_0 (column 1) never moves."""
    d = np.r_[np.linspace(0.40, 0.05, tc), np.full(T - tc, 0.05)]
    p = np.r_[np.zeros(tc), np.linspace(0.1, 0.9, T - tc)]
    r = 0.8 * p + 0.2 * (1 - np.tanh(5 * d))
    hand = np.zeros((T, 3)); hand[:, 0] = d
    cup = np.zeros((T, 3)); cup[tc:, 2] = np.linspace(0, 0.2, T - tc); hand[tc:, 2] = cup[tc:, 2]
    q = np.tile([1.0, 0, 0, 0], (T, 1))
    link = np.c_[hand, q]
    plate = np.c_[np.tile([0.3, 0.3, 0.0], (T, 1)), q]
    bp = np.concatenate([link, plate, np.c_[cup, q]], axis=1)
    tgt = np.zeros(T, bool); tgt[tc:] = True
    return pd.DataFrame({
        "reward": r, "reward_valid": np.ones(T, bool),
        "contact_target_finger": tgt, "contact_target_eef": np.zeros(T, bool),
        "observation.body_pose": list(bp),
        "observation.eef_pose": [np.r_[hand[t], 1.0, 0, 0, 0] for t in range(T)],
        "frame_index": np.arange(T),
    })


def _corpus(root, meta_names: dict[int, list[str]], n_ep=3):
    """One task, `n_ep` episodes; episode e's meta carries `meta_names[e]`, object cup_0."""
    task = root / "corpus" / "task_900"
    (task / "data" / "chunk-000").mkdir(parents=True)
    (task / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    frames = []
    for e in range(n_ep):
        g = _episode(tc=30 + 5 * e); g["episode_index"] = e; frames.append(g)
    pd.concat(frames, ignore_index=True).to_parquet(task / "data" / "chunk-000" / "file-000.parquet")
    pd.DataFrame([{"episode_index": e, "length": 80, "attempt_id": 100 + e,
                   "body_names": meta_names.get(e, TRUE), "object_body": "cup_0"}
                  for e in range(n_ep)]).to_parquet(task / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    (root / "artifacts").mkdir()
    idx = root / "artifacts" / "roots.json"
    idx.write_text(json.dumps({"version": 2, "base": "../corpus", "roots": {"900": "task_900"}}))
    return idx


def _true_table(n_ep=3):
    return {f"task_900--{e}": (list(TRUE), "cup_0") for e in range(n_ep)}


def _build(monkeypatch, idx, out, *extra):
    monkeypatch.setattr(sys, "argv", ["x", "--reward", "phase", "--binding", "names", "--index", str(idx),
                                      "--out", str(out), *extra])
    return baw.main()


def test_a_table_round_trips_and_digests_by_content(tmp_path):
    t = {"task_801--1": (["franka/a", "franka/b", "bowl_0"], "bowl_0"),
         "task_801--0": (["franka/a", "franka/b", "cup_1", "bowl_0"], "cup_1"),
         "task_1000--0": (["franka/a", "franka/b", "cup_1", "bowl_0"], "bowl_0")}
    f = tmp_path / "t.json"
    digest = en.write_table(t, f, about="toy")
    assert en.load_table(f) == t
    assert digest == en.table_digest(dict(reversed(list(t.items()))))     # order-free
    doc = json.loads(f.read_text())
    assert doc["robot_prefix"] == ["franka/a", "franka/b"] and len(doc["scenes"]) == 2
    assert list(doc["episodes"]) == ["task_801--0", "task_801--1", "task_1000--0"]
    swapped = {**t, "task_801--0": (["franka/a", "franka/b", "bowl_0", "cup_1"], "cup_1")}
    assert en.table_digest(swapped) != digest                          # ORDER is the content


def test_the_shipped_table_is_the_one_the_generator_map_pins():
    spec = json.loads(GENERATORS.read_text())
    pin = spec["inputs"]["episode_names"]
    table = en.load_table(REPO / pin["file"])
    assert en.table_digest(table) == pin["reward_input_sha256"]
    assert len(table) == pin["episodes"] == 5000
    assert all(obj in names for names, obj in table.values())
    argv = spec["files"]["approach_aware.chunk.awr_weights"]["argv"]
    assert argv[argv.index("--names-map") + 1] == "{names_map}"
    assert argv[argv.index("--names-sha256") + 1] == "{names_sha256}"


def test_corpus_table_reads_the_meta_join(tmp_path):
    idx = _corpus(tmp_path, {1: WRONG})
    t = en.corpus_table(idx)
    assert t == {**_true_table(), "task_900--1": (WRONG, "cup_0")}


def test_the_names_binding_refuses_a_corpus_whose_names_are_not_the_pinned_table(tmp_path, monkeypatch):
    idx = _corpus(tmp_path, {1: WRONG})
    out = tmp_path / "w.json"
    with pytest.raises(SystemExit, match="names table"):
        _build(monkeypatch, idx, out, "--names-sha256", en.table_digest(_true_table()))
    assert not out.exists()


def test_the_names_map_replaces_the_corpus_meta_and_reproduces_the_fixed_corpus(tmp_path, monkeypatch):
    fixed = _corpus(tmp_path / "fixed", {})
    want = tmp_path / "fixed.json"
    assert _build(monkeypatch, fixed, want) == 0
    stale = _corpus(tmp_path / "stale", {0: WRONG, 2: WRONG})
    plain = tmp_path / "stale_plain.json"
    assert _build(monkeypatch, stale, plain) == 0
    table = tmp_path / "names.json"
    digest = en.write_table(_true_table(), table, about="toy")
    got = tmp_path / "stale_mapped.json"
    assert _build(monkeypatch, stale, got, "--names-map", str(table), "--names-sha256", digest) == 0
    w = {p: json.loads(p.read_text()) for p in (want, plain, got)}
    assert w[got]["weights"] == w[want]["weights"] != w[plain]["weights"]
    assert w[got]["provenance"] == w[want]["provenance"]


def test_the_names_map_must_cover_every_corpus_episode(tmp_path, monkeypatch):
    idx = _corpus(tmp_path, {})
    table = tmp_path / "names.json"
    en.write_table({k: v for k, v in _true_table().items() if k != "task_900--2"}, table, about="toy")
    with pytest.raises(SystemExit, match="task_900--2"):
        _build(monkeypatch, idx, tmp_path / "w.json", "--names-map", str(table))


def test_the_names_flags_are_refused_where_no_name_is_read(tmp_path, monkeypatch):
    idx = _corpus(tmp_path, {})
    monkeypatch.setattr(sys, "argv", ["x", "--reward", "v2", "--index", str(idx), "--out", str(tmp_path / "w.json"),
                                      "--names-sha256", "0" * 64])
    with pytest.raises(SystemExit) as exc:
        baw.main()
    assert exc.value.code == 2
