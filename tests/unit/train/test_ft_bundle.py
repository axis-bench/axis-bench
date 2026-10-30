"""The pi0.5 finetune signal: the fork's loader stream and dropout rule, reproduced offline."""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest

from axis_qvla.signals.planners import torch_randperm
from axis_qvla.signals.rowspace import flat_rowspace
from axis_qvla.train import ft_bundle

# LiberoQualityConditioning(q_ep=5).dropped(e, f, p) in the paper's openpi fork, for p in (0, 1, 4),
# e in (0, 3, 17), f in (0, 1, 250), in that nesting order.
PIN_DROPPED = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

# What the fork's loader yields for a 10-row finetune at batch 4, seed 42: its PresentationSampler
# under torch's DataLoader (drop_last=True, a fresh iterator per pass), each index decoded
# `divmod(index, 10)` into (presentation, row). A 10-row pass makes two whole batches; its last two
# rows are dropped, so pass e fills batches 2e and 2e + 1. Measured with the fork's own code and
# torch 2.7.1.
FORK_ROWS = [[2, 6, 1, 8], [4, 5, 0, 9], [8, 4, 9, 0], [5, 1, 6, 7], [2, 6, 3, 8],
             [7, 9, 1, 4], [7, 6, 5, 3], [4, 9, 0, 8], [3, 2, 1, 5], [6, 4, 8, 9]]
FORK_PRESENTATION = [[k // 2] * 4 for k in range(10)]

# sha256 of torch.randperm(86912, generator=torch.Generator().manual_seed(s)) as int64 bytes, torch
# 2.7.1 (the openpi line's): the five passes of a seed-42 or seed-43 finetune use seeds 42..47.
TORCH_FT_PERMS = {
    42: "79e13f4001dc4e3aa9fbc4e677230609a8d17c35b81891414f46de2eb95e49cd",
    43: "e2f7341907d320903b384d6c5709b31f70ef27ede0367783a84eae16a5d55daa",
    44: "4c96db1e0fcd6749c98f610a44f1977fa4ee3cb461fcac1dd35175cb03ccc21a",
    45: "04b571c6ee0d07a6d31f85c0f55a9816f2c4e98f4084ff353486cb465a57a309",
    46: "c791e9ffc84d04e1b99d291f1116a81297dfd131f500b5408e06ee8d6117aaaa",
    47: "f4a39401b3d34fdab27d2c6a61c05cb51c1cb760a129650b1d7f107a24ad42fc",
}
FT_ROWS = 86_912

# The fork's ranges walk (pretrain_sampler.plan_pretrain_rows) over task 800 -- episodes 0, 1, 2
# of 5, 3 and 4 frames -- and task 1000 -- one episode of 6 -- with FORK_RANGES: the concat rows it
# keeps, in order. Measured with the fork's own code.
FORK_RANGES = {"task_1000--0": [[4, 99]], "task_800--2": [[2, 4], [-3, 1]], "task_800--0": [[1, 3]]}
FORK_RANGE_ROWS = [1, 2, 10, 11, 8, 16, 17]


def _compile(n, *, steps, batch, seed=42, quality_tag=None, episode_index=None, frame_index=None,
             prompts=("pick up the bowl",)):
    rs = flat_rowspace(name="ft", n_rows=n, source="test")
    return ft_bundle.compile_finetune(
        rs, episode_index=np.zeros(n, np.int64) if episode_index is None else episode_index,
        frame_index=np.arange(n) if frame_index is None else frame_index, prompts=prompts, steps=steps,
        batch=batch, seed=seed, quality_tag=quality_tag)


def test_dropped_is_the_forks_rule():
    got = [int(ft_bundle.dropped(e, f, p)) for p in (0, 1, 4) for e in (0, 3, 17) for f in (0, 1, 250)]
    assert got == PIN_DROPPED


def test_the_plan_is_the_forks_loader_stream():
    assert torch_randperm(10, 42).tolist() == [2, 6, 1, 8, 4, 5, 0, 9, 3, 7]
    assert torch_randperm(10, 43).tolist() == [8, 4, 9, 0, 5, 1, 6, 7, 2, 3]
    b = _compile(10, steps=5, batch=4)
    assert b.plan.tolist() == FORK_ROWS[:5]
    assert b.cond is None and b.vocab is None and b.reducer == "mean" and b.requires == {"plan"}
    assert b.program["params"]["quality_tag"] is None


def test_presentations_count_whole_batches_per_pass():
    assert ft_bundle.presentations(10, 10, 4).reshape(10, 4).tolist() == FORK_PRESENTATION
    assert ft_bundle.presentations(8, 4, 4).tolist() == [0] * 8 + [1] * 8     # a pass is two batches
    assert ft_bundle.presentations(10, 2, 4).tolist() == [0] * 8              # one pass, nothing dropped


def test_the_finetune_passes_are_torchs():
    for seed, want in TORCH_FT_PERMS.items():
        assert hashlib.sha256(torch_randperm(FT_ROWS, seed).tobytes()).hexdigest() == want, seed


@pytest.mark.parametrize("seed", [42, 43])
def test_a_paper_finetune_plan_is_five_torch_passes(seed):
    """13,580 x 32 over the 86,912-row finetune set: pass e is torch's permutation for seed + e."""
    b = _compile(FT_ROWS, steps=13580, batch=32, seed=seed)
    passes = b.plan.reshape(5, FT_ROWS)
    for e in range(5):
        assert hashlib.sha256(passes[e].tobytes()).hexdigest() == TORCH_FT_PERMS[seed + e], (seed, e)


def test_tagged_cond_follows_the_presentation():
    ep, fr = np.array([0] * 5 + [3] * 5), np.array([0, 1, 2, 3, 4] * 2)
    b = _compile(10, steps=10, batch=4, quality_tag=5, episode_index=ep, frame_index=fr)
    assert b.plan.tolist() == FORK_ROWS
    rows, pres = sum(FORK_ROWS, []), sum(FORK_PRESENTATION, [])
    want = [0 if ft_bundle.dropped(ep[r], fr[r], p) else 5 for r, p in zip(rows, pres)]
    assert b.cond.reshape(-1).tolist() == want and b.cond.shape == (10, 4, 1)
    assert 0 in want and 5 in want
    assert b.vocab.render("pick up the bowl", [5]) == "pick up the bowl\nQuality: 5"
    assert b.requires == {"plan", "cond_prompt"} and not any(s.guided for s in b.serve)
    assert b.diagnostics["tagged_fraction"] == want.count(5) / len(want)


@pytest.mark.parametrize("tag", [0, 6, True, "5"])
def test_a_tag_outside_the_bins_is_refused(tag):
    with pytest.raises(ft_bundle.FinetuneBundleError, match="not a bin"):
        _compile(4, steps=1, batch=4, quality_tag=tag)


def test_a_corpus_that_is_not_the_row_space_is_refused():
    rs = flat_rowspace(name="ft", n_rows=10, source="test")
    with pytest.raises(ft_bundle.FinetuneBundleError, match="has 9 rows"):
        ft_bundle.compile_finetune(rs, episode_index=np.zeros(9), frame_index=np.arange(9), prompts=("p",),
                                   steps=1, batch=4, seed=0, quality_tag=5)


def test_a_tagged_finetune_needs_its_prompts():
    with pytest.raises(ft_bundle.FinetuneBundleError, match="prompts"):
        _compile(4, steps=1, batch=4, quality_tag=5, prompts=())


def _task(base, tid, lengths, *, starts=None):
    import pandas as pd

    meta = base / f"task_{tid}" / "meta"
    (meta / "episodes").mkdir(parents=True)
    if starts is None:
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    pd.DataFrame({"episode_index": range(len(lengths)), "dataset_from_index": starts,
                  "length": lengths}).to_parquet(meta / "episodes" / "file-000.parquet")
    pd.DataFrame({"task_index": [0]}, index=pd.Index([f"task {tid}"], name="task")).to_parquet(meta / "tasks.parquet")


def test_episode_table_walks_tasks_in_id_order(tmp_path):
    for tid, lengths in ((1000, [2]), (800, [3, 1])):
        _task(tmp_path, tid, lengths)
    roots = tmp_path / "roots.json"
    roots.write_text(json.dumps({"version": 2, "base": ".", "roots": {"1000": "task_1000", "800": "task_800"}}))
    ep, fr, prompts = ft_bundle.episode_table(roots, tmp_path)
    assert ep.tolist() == [0, 0, 0, 1, 0, 0] and fr.tolist() == [0, 1, 2, 0, 0, 1]
    assert prompts == ("task 1000", "task 800")


def test_episode_table_keeps_the_frames_a_ranges_file_names(tmp_path):
    """The rows are the frames inside each episode's spans, in file order, clamped to the episode;
    an episode the ranges file does not name contributes nothing: row r is the frame at the fork's
    r-th planned concat row."""
    _task(tmp_path, 800, [5, 3, 4])
    _task(tmp_path, 1000, [6])
    roots = tmp_path / "roots.json"
    roots.write_text(json.dumps({"1000": "task_1000", "800": "task_800"}))      # a bare mapping reads too
    ranges = tmp_path / "ranges.json"
    ranges.write_text(json.dumps(FORK_RANGES))
    ep, fr, _ = ft_bundle.episode_table(roots, tmp_path, ranges)
    # concat row -> (episode, frame): task 800 holds rows 0..11 (episodes from 0, 5, 8), task 1000 rows 12..17
    concat = [(0, f) for f in range(5)] + [(1, f) for f in range(3)] + [(2, f) for f in range(4)] + \
        [(0, f) for f in range(6)]
    assert list(zip(ep.tolist(), fr.tolist())) == [concat[r] for r in FORK_RANGE_ROWS]
    assert ep.tolist() == [0, 0, 2, 2, 2, 0, 0] and fr.tolist() == [1, 2, 2, 3, 0, 4, 5]
    ranges.write_text(json.dumps({"task_800--0": [[1, 3]], "task_900--0": [[0, 1]]}))
    with pytest.raises(ft_bundle.FinetuneBundleError, match="1 of its 2 episodes"):
        ft_bundle.episode_table(roots, tmp_path, ranges)


def test_an_episode_whose_offset_is_not_its_rows_is_refused_only_if_it_has_rows(tmp_path):
    """Episode 2 records offset 9 but its rows follow episode 1's (offset 5), so the frame the loader
    reads there is not its own. Some finetune tasks carry such episodes outside their ranges."""
    _task(tmp_path, 800, [3, 2, 4], starts=[0, 3, 9])
    roots = tmp_path / "roots.json"
    roots.write_text(json.dumps({"800": "task_800"}))
    ranges = tmp_path / "ranges.json"
    ranges.write_text(json.dumps({"task_800--0": [[0, 3]], "task_800--1": [[1, 2]]}))
    ep, fr, _ = ft_bundle.episode_table(roots, tmp_path, ranges)
    assert ep.tolist() == [0, 0, 0, 1] and fr.tolist() == [0, 1, 2, 1]
    with pytest.raises(ft_bundle.FinetuneBundleError, match="episode 2's dataset_from_index"):
        ft_bundle.episode_table(roots, tmp_path)
    ranges.write_text(json.dumps({"task_800--2": [[0, 1]]}))
    with pytest.raises(ft_bundle.FinetuneBundleError, match="episode 2's dataset_from_index"):
        ft_bundle.episode_table(roots, tmp_path, ranges)


def test_a_roots_index_must_record_the_corpus_it_is_read_against(tmp_path):
    _task(tmp_path / "corpus", 800, [3])
    roots = tmp_path / "index" / "roots.json"
    roots.parent.mkdir()
    roots.write_text(json.dumps({"version": 2, "base": "../corpus", "roots": {"800": "task_800"}}))
    assert ft_bundle.episode_table(roots, tmp_path / "corpus")[0].tolist() == [0, 0, 0]
    roots.write_text(json.dumps({"version": 2, "base": "../other", "roots": {"800": "task_800"}}))
    with pytest.raises(ft_bundle.FinetuneBundleError, match="records base '../other'"):
        ft_bundle.episode_table(roots, tmp_path / "corpus")


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads through a 000 directory")
def test_unreadable_episode_metadata_is_not_read_as_missing(tmp_path):
    _task(tmp_path, 800, [3])
    roots = tmp_path / "roots.json"
    roots.write_text(json.dumps({"800": "task_800"}))
    episodes = tmp_path / "task_800" / "meta" / "episodes"
    episodes.chmod(0)
    try:
        with pytest.raises(ft_bundle.FinetuneBundleError) as e:
            ft_bundle.episode_table(roots, tmp_path)
    finally:
        episodes.chmod(0o755)
    assert str(e.value) == f"cannot read {episodes} (PermissionError)"
    (episodes / "file-000.parquet").unlink()
    with pytest.raises(ft_bundle.FinetuneBundleError, match="no meta/episodes parquet"):
        ft_bundle.episode_table(roots, tmp_path)


def test_a_tagged_plan_whose_passes_are_not_the_planners_is_refused(monkeypatch):
    """The dropout's presentation is checked against the planner's own passes, so a pass split
    that drifted from the planner's drop rule refuses instead of keying the wrong pass."""
    monkeypatch.setattr(ft_bundle, "presentations", lambda n, steps, batch: np.arange(steps * batch) // n)
    with pytest.raises(ft_bundle.FinetuneBundleError, match="pass 0 of the plan is not the planner's pass 0"):
        _compile(10, steps=10, batch=4, quality_tag=5)


def test_a_tagged_paper_finetune_has_the_forks_tagged_marginal():
    """The full tagged compile (434,560 dropout draws; about 10 s on one core, numpy only and no
    published data, so it runs in the fast suite): the plan is still torch's five passes, and the
    realized tagged fraction is the rule's 0.85 x 0.95 = 0.8075 (its binomial s.d. at this size is
    0.0006)."""
    rows = np.arange(FT_ROWS)
    b = _compile(FT_ROWS, steps=13580, batch=32, quality_tag=5, episode_index=rows // 300, frame_index=rows % 300)
    for e in range(5):
        assert hashlib.sha256(b.plan.reshape(5, FT_ROWS)[e].tobytes()).hexdigest() == TORCH_FT_PERMS[42 + e]
    assert set(np.unique(b.cond).tolist()) == {0, 5}
    assert abs(b.diagnostics["tagged_fraction"] - 0.8075) < 0.003
    flat, cond = b.plan.reshape(-1), b.cond.reshape(-1)
    for k in range(0, flat.size, 9973):
        r = int(flat[k])
        assert cond[k] == (0 if ft_bundle.dropped(r // 300, r % 300, k // FT_ROWS) else 5), k
