"""The artifact must answer 'what was the model conditioned on?' without rerunning anything, and
must refuse every way of being the WRONG arm's artifact under the right arm's name.

The filename check is the sharpest of those. `quality_v2.npz` and `quality_phase.npz` are built
by the same command with one path changed; nothing inside the file would look wrong if the two
were swapped, and the config name (`pi05_axis_cfg`) is shared. This is the identical hole
`_check_schedule_reward_id` was added to close on the schedule side.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import axis_qvla.signals.methods.cfg_tags as bql
from axis_qvla.signals.methods import schedule as bis
from axis_qvla.signals.blocks import schedule as index_schedule
from axis_qvla.signals.blocks import tags as quality_labels
from axis_qvla.signals.methods.cfg_tags import (
    check_artifact_filename,
    corpus_prompts,
    episode_task_ids,
    main,
    write_quality,
)
from axis_qvla.signals.blocks.tags import NOT_TRAINABLE


def _fixture(n_ep: int = 20, rows_per_ep: int = 10, stride: int = 25):
    blocks = [np.arange(i * stride, i * stride + rows_per_ep, dtype=np.int64) for i in range(n_ep)]
    q_ep = np.array([(i % 5) + 1 for i in range(n_ep)], dtype=np.int64)
    task_ids = np.array([i // 4 for i in range(n_ep)], dtype=np.int64)
    n_rows = n_ep * stride
    tag = np.full(n_rows, NOT_TRAINABLE, dtype=np.uint8)
    for block, q in zip(blocks, q_ep, strict=True):
        tag[block] = np.uint8(q)
    return blocks, q_ep, task_ids, n_rows, tag


def _write(tmp_path, name="quality_v2.npz", **overrides):
    blocks, q_ep, task_ids, n_rows, tag = _fixture()
    kw = {
        "reward_id": "v2",
        "seed": 0,
        "horizon": 10,
        "n_rows": n_rows,
        "config_hash": "abc123",
        "episode_rows": blocks,
        "q_ep": q_ep,
        "task_ids": task_ids,
        "dropout_meta": {
            "realized_drop_whole": 0.15,
            "realized_drop_component": 0.05,
            "realized_tagged_fraction": 0.8075,
            "bin_row_counts": {str(b): 40 for b in range(1, 6)},
            "n_trainable": int(sum(len(b) for b in blocks)),
        },
    }
    kw.update(overrides)
    p = tmp_path / name
    meta = write_quality(p, tag, ["pick up the bowl", "open the drawer"], **kw)
    return p, meta, tag


def test_artifact_roundtrips_the_tag_array_exactly(tmp_path):
    p, _, tag = _write(tmp_path)
    got = np.load(p, allow_pickle=False)
    assert np.array_equal(got["tag"], tag)
    assert got["tag"].dtype == np.uint8


def test_prompts_travel_with_the_artifact(tmp_path):
    """The token-budget guard tokenizes these; if they do not reach the artifact the
    guard has nothing to check and silently passes."""
    p, _, _ = _write(tmp_path)
    got = np.load(p, allow_pickle=False)
    assert [str(s) for s in got["prompts"]] == ["pick up the bowl", "open the drawer"]


def test_meta_on_disk_is_identical_to_the_meta_returned(tmp_path):
    """Catches ANY divergence between the block a caller logs and the block a consumer reads --
    including the int/str JSON key drift the schedule artifact already had to fix."""
    p, meta, _ = _write(tmp_path)
    on_disk = json.loads(str(np.load(p, allow_pickle=False)["meta"]))
    assert on_disk == meta


def test_meta_records_every_field_the_spec_requires(tmp_path):
    _, meta, _ = _write(tmp_path)
    for key in (
        "reward_id", "seed", "horizon", "n_rows", "n_trainable", "config_hash",
        "generator_sha256", "bin_row_counts", "realized_drop_whole", "realized_drop_component",
        "realized_tagged_fraction", "q_given_task_entropy_ratio", "n_prompts", "n_episodes",
    ):
        assert key in meta, f"missing meta field {key}"
    assert meta["reward_id"] == "v2"
    assert 0.0 < meta["q_given_task_entropy_ratio"] <= 1.0


def test_a_filename_that_disagrees_with_the_reward_id_raises(tmp_path):
    """Catches: quality_v2.npz built from the phase weights. Same config name, same everything
    inside except one string -- the filename is the ONLY thing separating the two arms."""
    with pytest.raises(ValueError, match="reward_id"):
        _write(tmp_path, name="quality_phase.npz", reward_id="v2")
    # ...and the same check must accept the matching pair
    _write(tmp_path, name="quality_phase.npz", reward_id="phase")


def test_filename_check_is_a_noop_on_a_name_that_makes_no_claim(tmp_path):
    """A staging path or a dated file claims nothing, so there is nothing to contradict. The
    check must not turn a naming convention into a requirement."""
    check_artifact_filename(tmp_path / "staging-2026-08-14.npz", "v2")


def test_a_tag_array_shorter_than_the_corpus_raises(tmp_path):
    blocks, q_ep, task_ids, n_rows, tag = _fixture()
    with pytest.raises(ValueError, match="n_rows"):
        write_quality(
            tmp_path / "quality_v2.npz", tag[:-1], ["p"], reward_id="v2", seed=0, horizon=10,
            n_rows=n_rows, config_hash="x", episode_rows=blocks, q_ep=q_ep, task_ids=task_ids,
            dropout_meta={"realized_drop_whole": 0.15, "realized_drop_component": 0.05,
                          "realized_tagged_fraction": 0.8075, "bin_row_counts": {},
                          "n_trainable": 200},
        )


def test_empty_prompts_raise(tmp_path):
    """A prompt list that came back empty makes the token-budget guard vacuous -- the exact
    'probe answers confidently when it should fail' class this repo keeps paying for."""
    blocks, q_ep, task_ids, n_rows, tag = _fixture()
    with pytest.raises(ValueError, match="no prompts"):
        write_quality(
            tmp_path / "quality_v2.npz", tag, [], reward_id="v2", seed=0, horizon=10,
            n_rows=n_rows, config_hash="x", episode_rows=blocks, q_ep=q_ep, task_ids=task_ids,
            dropout_meta={"realized_drop_whole": 0.15, "realized_drop_component": 0.05,
                          "realized_tagged_fraction": 0.8075, "bin_row_counts": {},
                          "n_trainable": 200},
        )


def test_task_ids_must_agree_with_the_authoritative_enumeration(tmp_path, monkeypatch):
    """`episode_task_ids` walks the corpus a SECOND time to retain the task id that
    `episode_rows` discards. That duplication is the whole risk of this file -- a walk that
    disagrees by one episode attributes every quality bin to the wrong task and the entropy
    ratio becomes meaningless while looking fine.

    This only exercises `_check_task_id_walk`, the FIRST guard `episode_task_ids` runs -- the
    episode-COUNT mismatch (an extra or missing episode in the second walk). It does NOT touch the
    second guard, `if lengths != list(block_lengths)` (a per-episode ROW-COUNT disagreement when
    the episode counts happen to match), nor the real second-walk logic
    (`_ordered_roots` -> `_episode_offsets` -> `_frames_in_spans`) at all -- both of those are
    covered on a real toy corpus by `test_episode_task_ids_returns_the_task_id_of_each_episode`
    and `test_episode_task_ids_raises_on_a_per_episode_row_count_mismatch` below.
    """
    with pytest.raises(ValueError, match="disagrees with"):
        bql._check_task_id_walk([10, 10, 20], [5, 5])


# --- the corpus driver -------------------------------------------------------------------------
#
# A real toy corpus, same shape as `build_index_schedule`'s test fixture (task 800: 3 episodes /
# 15 rows with ep2 having no ranges entry, task 1000: 1 episode / 8 rows -> corpus n_rows == 23,
# drop=False walk gives blocks [[1,2],[6,7],[15,16,17,21,22]]), so results here can be checked
# against numbers already pinned on the schedule side. Each root also carries a
# `meta/tasks.parquet`, which `episode_task_ids`'s corpus does not need but `main` does (it calls
# `corpus_prompts` on the same roots index).


def _write_meta(root, episodes: list[tuple[int, int, int]]) -> None:
    """episodes = [(episode_index, dataset_from_index, length), ...] -- the loader's own meta."""
    import pandas as pd
    from pathlib import Path

    meta = Path(root) / "meta" / "episodes" / "chunk-000"
    meta.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"episode_index": e, "dataset_from_index": f, "dataset_to_index": f + n, "length": n}
         for e, f, n in episodes]
    ).to_parquet(meta / "file-000.parquet")


def _write_tasks(root, prompts: list[str]) -> None:
    """A flat `meta/tasks.parquet` with a 'task' column -- the primary LeRobot shape."""
    import pandas as pd
    from pathlib import Path

    d = Path(root) / "meta"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"task": prompts}).to_parquet(d / "tasks.parquet")


def _weights_file(tmp_path, weights: dict, **header):
    from pathlib import Path

    p = Path(tmp_path) / f"w{len(list(Path(tmp_path).glob('w*.json')))}.json"
    p.write_text(json.dumps({"weights": weights, **header}))
    return p


def _corpus(tmp_path, *, reward_id: str = "v2"):
    from pathlib import Path

    a, b = Path(tmp_path) / "task_800", Path(tmp_path) / "task_1000"
    _write_meta(a, [(0, 0, 6), (1, 6, 4), (2, 10, 5)])
    _write_meta(b, [(0, 0, 8)])
    _write_tasks(a, ["pick up the bowl"])
    _write_tasks(b, ["open the drawer"])
    roots = Path(tmp_path) / "roots.json"
    roots.write_text(json.dumps(
        {"version": 2, "base": ".", "roots": {"1000": "task_1000", "800": "task_800"}}))
    ranges = Path(tmp_path) / "ranges.json"
    ranges.write_text(json.dumps({
        "task_800--0": [[1, 3]],
        "task_800--1": [[0, 2]],
        "task_1000--0": [[0, 3], [6, 100]],   # clamped to the episode's 8 rows
    }))
    ep_weights = {
        "task_800--0": [2.0, 1.0, 0.5, 2.0, 2.0, 2.0],
        "task_800--1": [1.5, 1.5, 0.5, 0.5],
        "task_800--2": [1.0, 1.0, 1.0, 1.0, 1.0],
        "task_1000--0": [1.0] * 8,            # neutralised -> unlabelled
    }
    weights = _weights_file(tmp_path, ep_weights, reward_id=reward_id, tau=10.0, cap=2.0,
                            horizon=10, n_rows=sum(len(w) for w in ep_weights.values()))
    return roots, ranges, weights


def _stats(weights):
    return bis.episode_stats_from_weights(weights, tau=10.0, cap=2.0)


def _roots_file(tmp_path, mapping: dict[int, str]):
    """A version-2 roots index pointing at task dirs already created under `tmp_path`."""
    from pathlib import Path

    p = Path(tmp_path) / "roots.json"
    p.write_text(json.dumps({"version": 2, "base": ".",
                             "roots": {str(k): v for k, v in mapping.items()}}))
    return p


# --- gap 2: episode_task_ids is never called by any test ---------------------------------------


def test_episode_task_ids_returns_the_task_id_of_each_episode(tmp_path):
    """Exercises the REAL second walk (`_ordered_roots` -> `_episode_offsets` ->
    `_frames_in_spans`), not just the `_check_task_id_walk` length guard. Catches: a walk that
    emits task ids in the wrong order (e.g. sorted as strings, so task 1000 would sort before 800),
    or one that skips the no-ranges episode (task_800--2) differently than
    `build_index_schedule.episode_rows` does -- either would attribute every quality bin in the
    corpus to the wrong task while raising nothing.
    """
    roots, ranges, weights = _corpus(tmp_path)
    stats = _stats(weights)
    blocks, *_ = bis.episode_rows(roots, ranges, stats, drop=False)
    task_ids = episode_task_ids(roots, ranges, stats, block_lengths=[len(b) for b in blocks])
    assert task_ids.tolist() == [800, 800, 1000]
    assert task_ids.dtype == np.int64


def test_episode_task_ids_raises_on_a_per_episode_row_count_mismatch(tmp_path):
    """The guard this file's docstring calls "never exercised": `episode_rows`' per-episode ROW
    COUNTS disagree with the second walk's while the episode COUNT matches, so
    `_check_task_id_walk` (which only compares `len()`s) passes it through and
    `if lengths != list(block_lengths)` has to catch it alone. Without this guard, a walk that
    agreed on episode count but silently miscounted one episode's rows (e.g. a ranges-clamp bug)
    would attribute every quality bin in that episode to the wrong span with no error -- exactly
    the failure mode the module docstring says the entropy ratio would then hide.
    """
    roots, ranges, weights = _corpus(tmp_path)
    stats = _stats(weights)
    blocks, *_ = bis.episode_rows(roots, ranges, stats, drop=False)
    real_lengths = [len(b) for b in blocks]
    assert len(real_lengths) == 3  # same episode COUNT as the corrupted lengths below
    wrong_lengths = [real_lengths[0] + 1, real_lengths[1], real_lengths[2]]
    with pytest.raises(ValueError, match="per-episode"):
        episode_task_ids(roots, ranges, stats, block_lengths=wrong_lengths)


# --- gap 3: corpus_prompts is never imported or tested ------------------------------------------


def test_corpus_prompts_reads_the_task_column(tmp_path):
    """The primary, most common `meta/tasks.parquet` shape. Catches: a build that only checks the
    index and never looks at a 'task' column at all."""
    import pandas as pd

    root = tmp_path / "task_1"
    (root / "meta").mkdir(parents=True)
    pd.DataFrame({"task": ["pick up the bowl", "pick up the bowl", "open the drawer"]}).to_parquet(
        root / "meta" / "tasks.parquet")
    roots = _roots_file(tmp_path, {1: "task_1"})
    assert corpus_prompts(roots) == ["open the drawer", "pick up the bowl"]


def test_corpus_prompts_falls_back_to_the_nested_glob(tmp_path):
    """Catches: a build that only tries the flat `meta/tasks*.parquet` glob and either raises or
    silently returns nothing for a corpus that only ships the nested
    `meta/tasks/**/*.parquet` layout: the two-glob fallback, exercised."""
    import pandas as pd

    root = tmp_path / "task_1"
    nested = root / "meta" / "tasks" / "chunk-000"
    nested.mkdir(parents=True)
    pd.DataFrame({"task": ["close the drawer"]}).to_parquet(nested / "file-000.parquet")
    roots = _roots_file(tmp_path, {1: "task_1"})
    assert corpus_prompts(roots) == ["close the drawer"]


def test_corpus_prompts_reads_a_string_index_when_there_is_no_task_column(tmp_path):
    """The alternate LeRobot layout, where the instruction is the row index rather than a column.
    Catches: a build that only checks for the 'task' column and raises (or drops every prompt) on
    this shape."""
    import pandas as pd

    root = tmp_path / "task_1"
    (root / "meta").mkdir(parents=True)
    df = pd.DataFrame({"count": [3]}, index=pd.Index(["stack the cups"]))
    df.to_parquet(root / "meta" / "tasks.parquet")
    roots = _roots_file(tmp_path, {1: "task_1"})
    assert corpus_prompts(roots) == ["stack the cups"]


def test_corpus_prompts_raises_when_neither_a_task_column_nor_a_string_index_exists(tmp_path):
    """Catches: a build that silently returns an empty or garbage prompt list instead of raising
    when the parquet has neither shape -- exactly the "probe answers confidently when it should
    fail" class `write_quality`'s empty-prompts guard exists to catch one layer downstream, but
    only if this function fails loudly first instead of handing it nonsense."""
    import pandas as pd

    root = tmp_path / "task_1"
    (root / "meta").mkdir(parents=True)
    pd.DataFrame({"count": [3, 4]}).to_parquet(root / "meta" / "tasks.parquet")  # default int index
    roots = _roots_file(tmp_path, {1: "task_1"})
    with pytest.raises(ValueError, match="neither"):
        corpus_prompts(roots)


def test_corpus_prompts_raises_filenotfound_when_no_tasks_parquet_exists_at_all(tmp_path):
    """Catches: a build that returns `[]` instead of raising when a root is missing its tasks
    table entirely -- which `write_quality`'s empty-prompts guard cannot distinguish from a
    legitimately empty corpus."""
    root = tmp_path / "task_1"
    (root / "meta").mkdir(parents=True)
    roots = _roots_file(tmp_path, {1: "task_1"})
    with pytest.raises(FileNotFoundError, match="task_1"):
        corpus_prompts(roots)


# --- gap 1: main() -- the CLI -- has zero coverage ----------------------------------------------


def test_main_writes_a_quality_artifact_with_real_provenance(tmp_path):
    """No prior test calls `main()` at all. Catches: an argv/argparse wiring bug, a header field
    silently mis-plumbed into `write_quality`, or `dense_tags`/`corpus_prompts` being called
    against the wrong roots/ranges argument."""
    roots, ranges, weights = _corpus(tmp_path)
    out = tmp_path / "quality_v2.npz"
    argv = ["--weights", str(weights), "--roots", str(roots), "--ranges", str(ranges),
            "--out", str(out), "--seed", "0"]
    assert main(argv) == 0
    with np.load(out, allow_pickle=False) as z:
        tag = z["tag"]
        meta = json.loads(str(z["meta"]))
    assert tag.shape == (23,)
    assert tag.dtype == np.uint8
    assert meta["reward_id"] == "v2"
    assert meta["n_trainable"] == 9


def test_reward_id_propagates_into_meta(tmp_path):
    """Catches: `main` reading the header but never threading `reward_id` through to
    `write_quality`, which the filename check and every downstream consumer key on. Two arms
    built from otherwise-identical corpora and headers, differing only in `reward_id` -- exactly
    the case `check_artifact_filename` exists to police."""
    for reward_id in ("v2", "phase"):
        roots, ranges, weights = _corpus(tmp_path, reward_id=reward_id)
        out = tmp_path / f"quality_{reward_id}.npz"
        argv = ["--weights", str(weights), "--roots", str(roots), "--ranges", str(ranges),
                "--out", str(out), "--seed", "0"]
        assert main(argv) == 0
        with np.load(out, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
        assert meta["reward_id"] == reward_id
        assert meta["horizon"] == 10


def test_n_rows_comes_from_enumeration_not_the_weights_header(tmp_path):
    """`meta["n_rows"]` must describe the corpus the tag's INDICES were computed in -- the roots
    index -- not the weights file's own (possibly stale) claim. Catches: `main` recording
    `header.get("n_rows")` directly instead of the `episode_rows` enumeration's `n_rows`."""
    roots, ranges, weights = _corpus(tmp_path)
    payload = json.loads(weights.read_text())
    del payload["n_rows"]
    weights.write_text(json.dumps(payload))
    out = tmp_path / "quality_v2.npz"
    argv = ["--weights", str(weights), "--roots", str(roots), "--ranges", str(ranges),
            "--out", str(out), "--seed", "0"]
    assert main(argv) == 0
    with np.load(out, allow_pickle=False) as z:
        assert json.loads(str(z["meta"]))["n_rows"] == 23


def test_refuses_when_the_weights_header_n_rows_disagrees_with_the_enumerated_corpus(tmp_path):
    """The `--roots roots_OLD --weights weights_NEW` build: episodes absent from `ranges` never
    reach the weights lookup, so every per-episode length check waves this through and the
    header/enumeration disagreement is the only signal left. Catches: `main` skipping this check
    (or checking the wrong two numbers) and silently binding a `quality_<reward_id>.npz` tag to an
    index space that is not the one it was built against."""
    roots, ranges, weights = _corpus(tmp_path)
    payload = json.loads(weights.read_text())
    payload["n_rows"] = 99  # as if the corpus grew after the weights were built
    weights.write_text(json.dumps(payload))
    argv = ["--weights", str(weights), "--roots", str(roots), "--ranges", str(ranges),
            "--out", str(tmp_path / "quality_v2.npz"), "--seed", "0"]
    with pytest.raises(ValueError, match="different pretraining data"):
        main(argv)


def test_main_is_deterministic_for_a_fixed_seed(tmp_path):
    """The whole reason the dropout is baked into an artifact rather than drawn per-batch: two
    runs at the same seed must be byte-identical, and a different seed must actually differ (a
    build that ignores `--seed` entirely would also pass the first half alone). Catches: `main`
    failing to thread `--seed` into `dense_tags`, or threading it inconsistently between runs."""
    roots, ranges, weights = _corpus(tmp_path)
    drawn = []
    for name in ("a.npz", "b.npz"):
        argv = ["--weights", str(weights), "--roots", str(roots), "--ranges", str(ranges),
                "--out", str(tmp_path / name), "--seed", "7"]
        assert main(argv) == 0
        with np.load(tmp_path / name, allow_pickle=False) as z:
            drawn.append(z["tag"])
    assert np.array_equal(drawn[0], drawn[1])

    argv = ["--weights", str(weights), "--roots", str(roots), "--ranges", str(ranges),
            "--out", str(tmp_path / "c.npz"), "--seed", "8"]
    assert main(argv) == 0
    with np.load(tmp_path / "c.npz", allow_pickle=False) as z:
        drawn_other_seed = z["tag"]
    assert not np.array_equal(drawn[0], drawn_other_seed)


def test_config_hash_changes_when_the_weights_change_under_the_same_filename(tmp_path):
    """The corpus artifacts are regenerated in place, so a hash over filenames alone would call
    the old and the new weights artifact the same thing. Catches: a `config_hash` that omits the
    weights digest (or hashes `args.weights` the string, not its contents)."""
    roots, ranges, weights = _corpus(tmp_path)
    out = tmp_path / "quality_v2.npz"
    argv = ["--weights", str(weights), "--roots", str(roots), "--ranges", str(ranges),
            "--out", str(out), "--seed", "0"]
    main(argv)
    with np.load(out, allow_pickle=False) as z:
        before = json.loads(str(z["meta"]))["config_hash"]
    payload = json.loads(weights.read_text())
    payload["weights"]["task_800--0"] = [0.5] * 6
    weights.write_text(json.dumps(payload))
    main(argv)
    with np.load(out, allow_pickle=False) as z:
        after = json.loads(str(z["meta"]))["config_hash"]
    assert before != after


# --- _config_hash / _generator_digest ------------------------------------------------------------


def test_generator_digest_moves_when_any_of_the_four_sources_changes(monkeypatch):
    """The load-bearing property this artifact's whole staleness detection rests on: the digest
    covers FOUR source files (`index_schedule.py` decides q, `build_index_schedule.py` decides the
    corpus walk and therefore which dropout draw each row gets, `quality_labels.py` decides the
    dropout, and `build_quality_labels.py` itself decides the rest), so a generator change to ANY
    of them must move the digest -- or a schedule/tag artifact could claim "identical" while every
    value moved.

    Each source is perturbed by intercepting `Path.read_bytes` for that file's resolved path only,
    one module at a time, and restored before the next -- the real modules on disk are never
    edited.
    """
    import pathlib

    modules = {
        "index_schedule.py": index_schedule,
        "build_index_schedule.py": bis,
        "quality_labels.py": quality_labels,
        "build_quality_labels.py": bql,
    }
    baseline = bql._generator_digest()
    original = pathlib.Path.read_bytes
    for name, module in modules.items():
        target = pathlib.Path(module.__file__).resolve()

        def _perturbed(self, _target=target, _orig=original):
            data = _orig(self)
            return data + b"# perturbed for test" if self.resolve() == _target else data

        monkeypatch.setattr(pathlib.Path, "read_bytes", _perturbed)
        try:
            changed = bql._generator_digest()
        finally:
            monkeypatch.setattr(pathlib.Path, "read_bytes", original)
        assert changed != baseline, f"{name} did not move the generator digest"


def test_config_hash_changes_when_the_generator_digest_changes(tmp_path, monkeypatch):
    """`config_hash` mixes `_generator_digest()` into an opaque sha256; if a future edit forgot to
    thread that digest into the hashed payload, the test above (which moves
    `_generator_digest()` itself) would still pass while `config_hash` -- the value every other
    check and every artifact comparison actually looks at -- silently stayed put."""
    roots, ranges, weights = _corpus(tmp_path)

    class _Args:
        seed = 0

    a = _Args()
    a.weights, a.roots, a.ranges = str(weights), str(roots), str(ranges)
    before = bql._config_hash(a)
    monkeypatch.setattr(bql, "_generator_digest", lambda: "a-different-digest")
    after = bql._config_hash(a)
    assert before != after


def test_config_hash_changes_when_an_input_changes(tmp_path):
    """The complement of the generator-source tests: a hash that is real over its declared inputs,
    not just the generator digest. Same weights-content-under-the-same-filename change as the
    `main()`-level test above, exercised directly on `_config_hash` so the two layers are each
    pinned independently."""
    roots, ranges, weights = _corpus(tmp_path)

    class _Args:
        seed = 0

    a = _Args()
    a.weights, a.roots, a.ranges = str(weights), str(roots), str(ranges)
    before = bql._config_hash(a)
    payload = json.loads(weights.read_text())
    payload["weights"]["task_800--0"] = [0.5] * 6
    weights.write_text(json.dumps(payload))
    after = bql._config_hash(a)
    assert before != after


# --- write_quality's atomic-write path -----------------------------------------------------------


def test_successful_write_leaves_no_stray_tmp_files(tmp_path):
    """`write_quality`'s atomic-write path is a thin reuse of the already-tested
    `build_index_schedule._unique_tmp_path` -- this only pins that the reuse actually cleans up
    after itself here too, without duplicating the sibling's full concurrency suite."""
    for _ in range(3):
        _write(tmp_path)
    stray = list(tmp_path.glob("*.tmp-*"))
    assert stray == [], f"stray temp files left behind: {stray}"


# ---------------------------------------------------------------------------
# Row (segment) granularity -- filename claims and the end-to-end builder path.


def test_seg_filename_binds_granularity_in_both_directions(tmp_path):
    """Catches: a row-built artifact recorded under the episode arm's name, and vice versa. The
    `_seg` suffix is the only thing separating the two granularities, exactly as the reward id is
    the only thing separating v2 from phase."""
    check_artifact_filename(tmp_path / "quality_phase_seg.npz", "phase", "row")
    check_artifact_filename(tmp_path / "quality_phase.npz", "phase", "episode")
    with pytest.raises(ValueError, match="granularity"):
        check_artifact_filename(tmp_path / "quality_phase.npz", "phase", "row")
    with pytest.raises(ValueError, match="granularity"):
        check_artifact_filename(tmp_path / "quality_phase_seg.npz", "phase", "episode")
    with pytest.raises(ValueError, match="reward_id"):
        check_artifact_filename(tmp_path / "quality_v2_seg.npz", "phase", "row")
    check_artifact_filename(tmp_path / "staging.npz", "phase", "row")  # no claim, no-op


def test_main_granularity_row_writes_rowwise_tags_with_diagnostics(tmp_path):
    """End-to-end `--granularity row` at the DEFAULT plateau="rank". Hand-solvable:
    trainable flat w = [1.0, 0.5, 1.5, 1.5, 1.0 x5] -> delta_quality_bins gives episode blocks
    [1,1], [4,5], [2,2,3,3,4], so episodes 1 and 2 are multibin (fraction 2/3) BEFORE dropout --
    which is what the meta diagnostic must report, since it describes the bins, not the draw."""
    roots, ranges, weights = _corpus(tmp_path)
    out = tmp_path / "quality_v2_seg.npz"
    argv = ["--weights", str(weights), "--roots", str(roots), "--ranges", str(ranges),
            "--out", str(out), "--seed", "0", "--granularity", "row"]
    assert main(argv) == 0
    with np.load(out, allow_pickle=False) as z:
        tag = z["tag"]
        meta = json.loads(str(z["meta"]))
    assert meta["granularity"] == "row"
    assert meta["plateau"] == "rank"
    assert meta["n_trainable"] == 9
    assert meta["episodes_multibin_fraction"] == pytest.approx(2.0 / 3.0)
    assert meta["within_episode_bin_std_mean"] > 0.0
    assert tag.shape == (23,)
    trainable = tag[tag != quality_labels.NOT_TRAINABLE]
    tagged = trainable[trainable != quality_labels.NO_TAG]
    assert set(np.unique(tagged).tolist()) <= {1, 2, 3, 4, 5}


def test_main_default_granularity_is_episode_and_unsuffixed_name_passes(tmp_path):
    """Catches: the new flag changing the default path's behaviour or its filename contract."""
    roots, ranges, weights = _corpus(tmp_path)
    out = tmp_path / "quality_v2.npz"
    argv = ["--weights", str(weights), "--roots", str(roots), "--ranges", str(ranges),
            "--out", str(out), "--seed", "0"]
    assert main(argv) == 0
    with np.load(out, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
    assert meta["granularity"] == "episode"
    assert "episodes_multibin_fraction" not in meta


def test_row_quality_bins_default_is_plain_delta_ranking():
    """Catches: the plateau policy drifting off the shipped artifact's behaviour. The default
    must reproduce `delta_quality_bins` verbatim -- this keeps the phase reward's own
    semantics (a Delta == 0 row ranks with everything else)."""
    from axis_qvla.signals.methods.cfg_tags import row_quality_bins

    row_w = [np.array([1.0, 0.5]), np.array([1.5, 1.5]), np.array([1.0] * 5)]
    expect = index_schedule.delta_quality_bins(row_w)
    got = row_quality_bins(row_w)
    for g, e in zip(got, expect):
        assert g.tolist() == e.tolist()

