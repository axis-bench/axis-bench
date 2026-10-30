"""A composed method trains on SmolVLA and X-VLA what it trains on π0.5: its LeRobot files, read the way
the pinned LeRobot fork reads them, draw the compiled labels' plan step by step, with their loss weight
and their prompt on every drawn frame.

Two readers: `FormatReaders` reads the fork's documented file formats with numpy (its row sampler's
draw through axis_qvla.signals.planners, which replays torch's randperm bit for bit), and `ForkReaders` runs
the fork's own readers (AxisScheduleSampler, AxisRowSampler, AxisFlatWeighter, AxisQualityTags with
apply_quality_tags). The fork's readers need the LeRobot environment (torch, the fork), which the
offline suite does not have, so their cases are marked slow, deselected by default and run in
that environment:

    PYTHONPATH=. third_party/lerobot/.venv/bin/python -m pytest -m slow -q -p no:cacheprovider \
        tests/unit/signals/test_lerobot_files.py

Each mutation of an exported file (wrong order, a dropped row, a scaled weight, an off-by-one tag, another
run seed) must make the comparison fail.
"""
from __future__ import annotations

import importlib
import json

import numpy as np
import pytest

from axis_qvla.signals.bundle import export_legacy
from axis_qvla.signals.compile import Compose, compile_program
from axis_qvla.signals.imported import ImportedScores
from axis_qvla.signals import lerobot_files as lf
from axis_qvla.signals.lerobot_files import ROW_SAMPLER
from axis_qvla.signals.methods import schedule as bis
from axis_qvla.signals.methods.filtering import TopFraction
from axis_qvla.signals.planners import PermutationPlanner, SchedulePlanner
from axis_qvla.signals.program import Conditioning, EpisodeValues
from axis_qvla.signals.rowspace import episode_keys, trainable_rows
from axis_qvla.signals.schema import BARE, FORBIDDEN, Budget, CondVocab, ServeCondition
from axis_qvla.train.bind import legacy_field
from axis_qvla.train.parity import load_signals_lock
from experiments.examples.keep_top import keep_top
from tests.unit.signals.toy_corpus import write_toy_corpus

PROMPT = "pick up the bowl"


class FrameIndex:
    """Scorer: every frame scores its own index."""

    def score(self, labels, rows):
        return EpisodeValues({k: np.arange(n, dtype=np.float64)
                              for k, n in zip(episode_keys(rows), rows.arrays["episode_length"].tolist())})


class PlusOne:
    """Weighter: w = 1 + value."""

    def weigh(self, values, rows):
        return EpisodeValues({k: 1.0 + v for k, v in values.values.items()})


class FrameCodes:
    """Conditioner: code 1 + (row % 5) on every trainable row, every third one dropped to 0; `text`
    renders code q (the fork's own text by default)."""

    def __init__(self, text="\nQuality: {}"):
        self.text = text

    def condition(self, values, rows):
        dense = np.full((rows.n_rows, 1), FORBIDDEN, dtype=np.uint8)
        t = trainable_rows(rows)
        dense[t, 0] = 1 + t % 5
        dense[t[::3], 0] = 0
        vocab = CondVocab(("quality",), {"quality": {q: self.text.format(q) for q in range(1, 6)}}, rows.prompts)
        return Conditioning(dense, vocab, (BARE, ServeCondition({"quality": "5"}, None, 1.0)))


@pytest.fixture
def toy(tmp_path, monkeypatch):
    monkeypatch.delenv(bis.CORPUS_ROOT_ENV, raising=False)
    return write_toy_corpus(tmp_path)


def _compose(method, **kw):
    return Compose(method=method, version=1, params={}, planner=PermutationPlanner(ROW_SAMPLER), **kw)


def _methods(toy):
    """name -> (compiled labels, the run seed): keep_top (a selection drawn by the schedule planner),
    a weighted and a tagged method on the fork's row sampler, and the row sampler's plain draw under
    openpi's convention, which only a schedule replays."""
    scores = ImportedScores.from_json(toy.weights)
    weighted = _compose("toy.weighted", scorer=FrameIndex(), weighter=PlusOne(), loss_reducer="per_rank_ratio")
    tagged = _compose("toy.tagged", scorer=FrameIndex(), conditioner=FrameCodes())
    openpi = Compose(method="toy.openpi", version=1, params={}, planner=PermutationPlanner("openpi-rowsampler"))
    return {"keep_top": (compile_program(keep_top(0.5), scores, toy.rs, Budget(6, 2), 0)[0], 0),
            # 18 draws over the 9 trainable rows: two epochs of the row sampler.
            "weighted": (compile_program(weighted, None, toy.rs, Budget(6, 3), 7)[0], 7),
            "tagged": (compile_program(tagged, None, toy.rs, Budget(6, 3), 11)[0], 11),
            "openpi": (compile_program(openpi, None, toy.rs, Budget(4, 2), 3)[0], 3)}


def _export(bundle, out):
    """train_config field -> exported file, mapped as binding maps a non-paper file."""
    lock = load_signals_lock()
    paths = export_legacy(bundle, "lerobot-legacy", out)
    records = [f for f in bundle.legacy if f.format == "lerobot-legacy"]
    return {legacy_field(rec, lock): p for rec, p in zip(records, paths, strict=True)}


class FormatReaders:
    """The fork's file formats, read with numpy as its readers document them."""

    def schedule(self, path, n_rows):
        with np.load(path, allow_pickle=False) as z:
            rows, meta = z["rows"], json.loads(str(z["meta"]))
        assert rows.ndim == 2 and np.issubdtype(rows.dtype, np.integer) and meta["n_rows"] == n_rows
        return rows.reshape(-1)

    def row_stream(self, path, seed, total):
        return PermutationPlanner(ROW_SAMPLER).stream(np.load(path)["rows"], total, seed)

    def weights(self, path, n_rows, index):
        flat = np.load(path)["weights"]
        assert flat.dtype == np.float32 and flat.shape == (n_rows,)
        return flat[index]

    def tagged(self, path, n_rows, index, tasks):
        with np.load(path, allow_pickle=False) as z:
            tag, meta = z["tag"], json.loads(str(z["meta"]))
        assert tag.dtype == np.uint8 and tag.shape == (n_rows,) == (meta["n_rows"],)
        assert max(int(k) for k in meta["bin_row_counts"]) == 5 and meta["reward_id"]
        return [t if int(q) == 0 else f"{t}\nQuality: {int(q)}" for t, q in zip(tasks, tag[index])]


class ForkReaders:
    """The pinned LeRobot fork's own readers; outside the LeRobot environment the import fails the test."""

    def __init__(self):
        self.torch = importlib.import_module("torch")
        self.sampler = importlib.import_module("lerobot.datasets.axis_sampler")
        self.quality = importlib.import_module("lerobot.utils.axis_quality")
        self.weighting = importlib.import_module("lerobot.utils.axis_sample_weighting")
        self.config = importlib.import_module("lerobot.utils.sample_weighting")

    def schedule(self, path, n_rows):
        return np.asarray(list(self.sampler.AxisScheduleSampler(str(path), expected_frames=n_rows)), dtype=np.int64)

    def row_stream(self, path, seed, total):
        sampler, out = self.sampler.AxisRowSampler(rows=np.load(path)["rows"], seed=seed), []
        while len(out) < total:
            out += list(iter(sampler))
        return np.asarray(out[:total], dtype=np.int64)

    def weights(self, path, n_rows, index):
        cfg = self.config.SampleWeightingConfig(type="axis_flat",
                                                extra_params={"weights_path": str(path), "expected_frames": n_rows})
        weighter = self.weighting.AxisFlatWeighter(cfg, self.torch.device("cpu"))
        return weighter.compute_batch_weights({"index": self.torch.as_tensor(index)})[0].numpy()

    def tagged(self, path, n_rows, index, tasks):
        tags = self.quality.AxisQualityTags(str(path), expected_frames=n_rows)
        batch = {"index": self.torch.as_tensor(index), "task": list(tasks)}
        self.quality.apply_quality_tags(batch, tags)
        return batch["task"]


def assert_trains_the_bundle(bundle, files, readers, seed):
    """What the fork trains from `files` under run seed `seed` is the bundle's plan, step by step, and
    on every drawn frame the bundle's loss weight and the prompt the bundle renders (what π0.5 reads)."""
    n_rows, plan = bundle.rowspace.n_rows, bundle.plan
    if "axis_schedule_path" in files:
        drawn = readers.schedule(files["axis_schedule_path"], n_rows)
    else:
        drawn = readers.row_stream(files["axis_rows_path"], seed, plan.size)
    np.testing.assert_array_equal(drawn, plan.reshape(-1))
    drawn = drawn.reshape(plan.shape)
    weights = files.get("sample_weighting.extra_params.weights_path")
    assert (weights is None) == (bundle.weights is None)
    quality = files.get("axis_quality_path")
    assert (quality is None) == (bundle.cond is None)
    for s in range(plan.shape[0]):
        if weights is not None:
            np.testing.assert_array_equal(readers.weights(weights, n_rows, drawn[s]), bundle.weights[plan[s]])
        if quality is not None:
            got = readers.tagged(quality, n_rows, drawn[s], [PROMPT] * plan.shape[1])
            assert got == [bundle.vocab.render(PROMPT, c) for c in bundle.cond[s]]


EXPORTS = [("keep_top", ["axis_schedule_path"]),
           ("weighted", ["axis_rows_path", "sample_weighting.extra_params.weights_path"]),
           ("tagged", ["axis_quality_path", "axis_rows_path"]),
           ("openpi", ["axis_schedule_path"])]


def _trains(toy, tmp_path, readers, name, fields):
    bundle, seed = _methods(toy)[name]
    files = _export(bundle, tmp_path / "out")
    assert sorted(files) == fields
    assert_trains_the_bundle(bundle, files, readers, seed)


@pytest.mark.parametrize("name, fields", EXPORTS)
def test_the_lerobot_files_train_the_compiled_labels(toy, tmp_path, name, fields):
    _trains(toy, tmp_path, FormatReaders(), name, fields)


@pytest.mark.slow
@pytest.mark.parametrize("name, fields", EXPORTS)
def test_the_forks_own_readers_train_the_compiled_labels(toy, tmp_path, name, fields):
    _trains(toy, tmp_path, ForkReaders(), name, fields)


def _rewrite(path, out, **members):
    """`path`'s npz members with `members` replaced, written at `out`."""
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    arrays.update(members)
    np.savez(out, **arrays)
    return out


def _swap_steps(rows):
    rows = rows.copy()
    a = next(i for i in range(1, len(rows)) if not np.array_equal(rows[i], rows[0]))
    rows[[0, a]] = rows[[a, 0]]
    return rows


def _bump_drawn_tag(tag, plan):
    tag = tag.copy()
    r = next(int(r) for r in plan.reshape(-1) if tag[r] < 5)
    tag[r] += 1
    return tag


MUTATIONS = {
    # name: (method, field, mutate(npz, bundle) -> members, or None for "the run's seed is another")
    "schedule in another order": ("keep_top", "axis_schedule_path", lambda z, b: {"rows": _swap_steps(z["rows"])}),
    "schedule with a dropped step": ("keep_top", "axis_schedule_path", lambda z, b: {"rows": z["rows"][:-1]}),
    "row set with a dropped row": ("weighted", "axis_rows_path", lambda z, b: {"rows": z["rows"][:-1]}),
    "row set in another order": ("weighted", "axis_rows_path", lambda z, b: {"rows": z["rows"][::-1].copy()}),
    "a scaled weight": ("weighted", "sample_weighting.extra_params.weights_path",
                        lambda z, b: {"weights": np.where(np.arange(z["weights"].size) == b.plan[0, 0],
                                                          z["weights"] * 2, z["weights"]).astype(np.float32)}),
    "an off-by-one tag": ("tagged", "axis_quality_path", lambda z, b: {"tag": _bump_drawn_tag(z["tag"], b.plan)}),
    "another run seed": ("tagged", None, None),
}


def _mutated_fails(toy, tmp_path, readers, mutation):
    name, field, mutate = MUTATIONS[mutation]
    bundle, seed = _methods(toy)[name]
    files = _export(bundle, tmp_path / "out")
    if field is None:
        seed += 1
    else:
        with np.load(files[field], allow_pickle=False) as z:
            members = mutate(z, bundle)
        files[field] = _rewrite(files[field], tmp_path / "mutated.npz", **members)
    with pytest.raises(AssertionError):
        assert_trains_the_bundle(bundle, files, readers, seed)


@pytest.mark.parametrize("mutation", sorted(MUTATIONS))
def test_a_mutated_file_does_not_train_the_compiled_labels(toy, tmp_path, mutation):
    _mutated_fails(toy, tmp_path, FormatReaders(), mutation)


@pytest.mark.slow
@pytest.mark.parametrize("mutation", sorted(MUTATIONS))
def test_a_mutated_file_does_not_train_the_compiled_labels_with_the_forks_readers(toy, tmp_path, mutation):
    _mutated_fails(toy, tmp_path, ForkReaders(), mutation)


def test_the_schedule_and_tags_name_the_method_for_the_run_to_bind(toy, tmp_path):
    """The fork binds a schedule to the run's axis_expected_mode / axis_expected_reward and a tag file
    to its axis_expected_reward: the schedule planner's mode or the planner's convention, and the method."""
    methods = _methods(toy)
    files = _export(methods["keep_top"][0], tmp_path / "k")
    with np.load(files["axis_schedule_path"]) as z:
        meta = json.loads(str(z["meta"]))
    assert (meta["mode"], meta["reward_id"], meta["n_rows"]) == ("drop_top", "example.keep_top", 23)
    files = _export(methods["openpi"][0], tmp_path / "o")
    with np.load(files["axis_schedule_path"]) as z:
        assert json.loads(str(z["meta"]))["mode"] == "openpi-rowsampler"
    files = _export(methods["tagged"][0], tmp_path / "t")
    with np.load(files["axis_quality_path"]) as z:
        meta, prompts = json.loads(str(z["meta"])), z["prompts"].tolist()
    assert (meta["reward_id"], prompts) == ("toy.tagged", list(toy.rs.prompts))


def test_the_control_trains_on_lerobot_too(toy, tmp_path):
    bundle, controls = compile_program(keep_top(0.5), ImportedScores.from_json(toy.weights), toy.rs, Budget(6, 2), 0)
    files = _export(controls[0], tmp_path / "c")
    assert list(files) == ["axis_schedule_path"]
    assert_trains_the_bundle(controls[0], files, FormatReaders(), 0)


@pytest.mark.parametrize("program, why", [
    # The fork trains loss weights or tags, never both (its train config's validate()).
    (_compose("toy.both", scorer=FrameIndex(), weighter=PlusOne(), conditioner=FrameCodes(),
              loss_reducer="per_rank_ratio"), lf.BOTH),
    # A selection is replayed as a schedule, which the fork trains without weights or tags.
    (Compose(method="toy.keep_weighted", version=1, params={"keep": 0.5}, scorer=FrameIndex(),
             selector=TopFraction(0.5), planner=SchedulePlanner("drop_top"), weighter=PlusOne(),
             loss_reducer="per_rank_ratio"), lf.NOT_DRAWN),
    # Openpi's draw is not the fork's row sampler's, so its weights could ride only on a schedule.
    (Compose(method="toy.openpi_weighted", version=1, params={}, planner=PermutationPlanner("openpi-rowsampler"),
             scorer=FrameIndex(), weighter=PlusOne(), loss_reducer="global_ratio"), lf.OPENPI_ORDER),
    # The fork appends "\nQuality: q"; a vocabulary rendering other text cannot be trained there.
    (_compose("toy.other_text", scorer=FrameIndex(), conditioner=FrameCodes("\nScore: {}")), lf.TAG_TEXT),
], ids=["weights-and-tags", "selection-and-weights", "openpi-draw-and-weights", "other-tag-text"])
def test_a_method_the_fork_cannot_train_carries_no_lerobot_files_and_says_why(toy, program, why):
    bundle, _ = compile_program(program, None, toy.rs, Budget(6, 3), 1)
    assert not [f for f in bundle.legacy if f.format == "lerobot-legacy"]
    assert lf.refusal(bundle) == why


def test_openpis_draw_at_seed_0_is_the_row_samplers_and_exports_the_row_set(toy):
    """Both conventions seed epoch e with e at seed 0, so that plan is the fork's draw and trains there."""
    weighted = Compose(method="toy.openpi_weighted", version=1, params={},
                       planner=PermutationPlanner("openpi-rowsampler"), scorer=FrameIndex(), weighter=PlusOne(),
                       loss_reducer="per_rank_ratio")
    bundle, _ = compile_program(weighted, None, toy.rs, Budget(6, 3), 0)
    assert [f.path for f in bundle.legacy] == ["signals/lerobot_rows_toy.openpi_weighted.npz",
                                               "signals/lerobot_awr_flat_toy.openpi_weighted.npz"]
    assert lf.refusal(bundle) is None


@pytest.mark.parametrize("name", ["keep_top", "weighted", "tagged", "openpi"])
def test_a_method_the_fork_trains_has_no_refusal(toy, name):
    assert lf.refusal(_methods(toy)[name][0]) is None
