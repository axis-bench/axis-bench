"""`qvla train` binds a SmolVLA or X-VLA run to a composed method's compiled labels: it exports the
LeRobot files they carry (axis_qvla.signals.lerobot_files) under their compiled-labels id, maps each to its
train_config field, and refuses, before launch, what the pinned LeRobot fork would refuse at start."""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from axis_qvla.cli import main
from axis_qvla.data.layout import CORPUS, default_dest
from axis_qvla.signals.bundle import write_bundle
from axis_qvla.signals.compile import Compose, compile_program
from axis_qvla.signals.imported import ImportedScores
from axis_qvla.signals.methods import schedule as bis
from axis_qvla.signals.planners import PermutationPlanner
from axis_qvla.signals.schema import Budget
from axis_qvla.train.armspec import loads_armspec
from axis_qvla.train.bind import BindError, bind
from axis_qvla.train.recipes import load_recipe
from experiments.examples.keep_top import keep_top
from tests.unit.signals.test_lerobot_files import FrameCodes, FrameIndex, PlusOne
from tests.unit.signals.toy_corpus import write_toy_corpus
from tests.unit.train._arms import arm_toml, host
from tests.unit.train._bundles import lerobot_rows_legacy, rowspace_ref, write_fixture_bundle

EXPECTED = 'axis_expected_mode = "drop_top"\naxis_expected_reward = "example.keep_top"\naxis_expected_frames = 23'


@pytest.fixture
def toy(tmp_path, monkeypatch):
    """The toy corpus's frame index under the paper's frame-index name, so it binds to that layout."""
    monkeypatch.delenv(bis.CORPUS_ROOT_ENV, raising=False)
    return dataclasses.replace(write_toy_corpus(tmp_path / "toy").rs, name="libero5k_v2_d8")


def _published(paths, bundle) -> str:
    default_dest(CORPUS, paths, "lerobot_merged").mkdir(parents=True, exist_ok=True)
    return "sha256:" + write_bundle(bundle, default_dest(CORPUS, paths, "bundles"))


def _arm(rs, bid, backbone="smolvla-lerobot", overrides=EXPECTED, seed=0):
    name, init = ("smolvla", "base:smolvlm2_500m") if backbone == "smolvla-lerobot" else ("xvla", "base:xvla_base")
    return arm_toml(id=f"{name}.pretrain.keep_top", backbone=backbone, stage="pretrain", track="smoke",
                    preset=f"{name}.pretrain.smoke", rowspace=rowspace_ref(rs), init=init, norm=False, bundle=bid,
                    seed=seed, overrides=overrides)


def _keep_top(toy, tmp_path):
    scores = ImportedScores.from_json(tmp_path / "toy" / "w.json")
    return compile_program(keep_top(0.5), scores, toy, Budget(20, 8), 42)[0]


@pytest.mark.parametrize("backbone", ["smolvla-lerobot", "xvla-lerobot"])
def test_qvla_train_renders_a_run_on_a_composed_methods_labels(toy, tmp_path, capsys, backbone):
    hosts, paths = host(tmp_path)
    bundle = _keep_top(toy, tmp_path)
    bid = _published(paths, bundle)
    arm = tmp_path / "arm.toml"
    arm.write_text(_arm(toy, bid, backbone))
    assert main(["train", "--render", str(arm), "--host", str(hosts), "--smoke"]) == 0
    spec = loads_armspec(arm.read_text())
    smoke = paths.runs / spec.id / "rendered-smoke"
    cfg = json.loads((smoke / "train_config.json").read_text())
    schedule = smoke / "signals" / f"{bid[7:19]}-lerobot_schedule_example.keep_top.npz"
    assert cfg["axis_schedule_path"] == str(schedule) and cfg["axis_rows_path"] is None
    assert (cfg["axis_expected_mode"], cfg["axis_expected_reward"], cfg["seed"]) == ("drop_top", "example.keep_top", 0)
    with np.load(schedule) as z:
        np.testing.assert_array_equal(z["rows"], bundle.plan)
        assert json.loads(str(z["meta"]))["n_rows"] == cfg["axis_expected_frames"] == 23
    assert json.loads((smoke / "binding.json").read_text())["bundle"] == bid


def test_a_schedule_the_run_does_not_name_is_refused(toy, tmp_path):
    _, paths = host(tmp_path)
    bid = _published(paths, _keep_top(toy, tmp_path))
    spec = loads_armspec(_arm(toy, bid, overrides='axis_expected_mode = "anneal"'))
    with pytest.raises(BindError) as e:
        bind(spec, load_recipe(spec.preset), paths, smoke=True)
    assert str(e.value).splitlines()[1:] == [
        "  recipe: signals/lerobot_schedule_example.keep_top.npz records mode 'drop_top', which the pinned LeRobot "
        "fork requires as axis_expected_mode (the config has 'anneal'); declare axis_expected_mode = 'drop_top' in "
        "[recipe.overrides]",
        "  recipe: signals/lerobot_schedule_example.keep_top.npz records reward_id 'example.keep_top', which the "
        "pinned LeRobot fork requires as axis_expected_reward (the config has None); declare axis_expected_reward "
        "= 'example.keep_top' in [recipe.overrides]"]
    assert not (paths.runs / spec.id).exists()


def test_a_row_set_bound_under_another_seed_is_refused(toy, tmp_path):
    _, paths = host(tmp_path)
    plan = np.resize(np.array([1, 2, 6, 7, 15, 16, 17, 21]), (20, 8)).astype(np.int64)
    bid = write_fixture_bundle(default_dest(CORPUS, paths, "bundles"), plan=plan, rowspace=toy,
                               **lerobot_rows_legacy([1, 2, 6, 7, 15, 16, 17, 21, 22]))
    default_dest(CORPUS, paths, "lerobot_merged").mkdir(parents=True)
    spec = loads_armspec(_arm(toy, bid, overrides="axis_expected_frames = 23", seed=5))
    with pytest.raises(BindError) as e:
        bind(spec, load_recipe(spec.preset), paths, smoke=True)
    assert str(e.value).splitlines()[1:] == [
        "  seed: the pinned LeRobot fork draws signals/test/lerobot_rows.npz in the order its row sampler gives "
        "the run's seed 5; the compiled labels' plan was drawn with seed 0"]


def test_a_method_weighing_and_tagging_frames_is_refused_on_lerobot(toy, tmp_path):
    _, paths = host(tmp_path)
    both = Compose(method="toy.both", version=1, params={}, planner=PermutationPlanner("lerobot-axisrowsampler"),
                   scorer=FrameIndex(), weighter=PlusOne(), conditioner=FrameCodes(), loss_reducer="per_rank_ratio")
    bid = _published(paths, compile_program(both, None, toy, Budget(1, 8), 0)[0])
    spec = loads_armspec(_arm(toy, bid, overrides="axis_expected_frames = 23"))
    with pytest.raises(BindError) as e:
        bind(spec, load_recipe(spec.preset), paths, smoke=True)
    assert ("  export: the compiled labels carry no lerobot-legacy files, the only label files the pinned LeRobot "
            "fork reads: they hold loss weights and quality tags together: the pinned LeRobot fork trains one or "
            "the other (its configs/train.py refuses both in one run), so this method trains on pi0.5 only"
            ) in str(e.value).splitlines()


def test_the_refusal_names_why_the_fork_cannot_train_the_labels(toy, tmp_path):
    _, paths = host(tmp_path)
    openpi = Compose(method="toy.openpi_weighted", version=1, params={},
                     planner=PermutationPlanner("openpi-rowsampler"), scorer=FrameIndex(), weighter=PlusOne(),
                     loss_reducer="per_rank_ratio")
    bid = _published(paths, compile_program(openpi, None, toy, Budget(1, 8), 3)[0])
    spec = loads_armspec(_arm(toy, bid, overrides="axis_expected_frames = 23"))
    with pytest.raises(BindError) as e:
        bind(spec, load_recipe(spec.preset), paths, smoke=True)
    assert ("  export: the compiled labels carry no lerobot-legacy files, the only label files the pinned LeRobot "
            "fork reads: they hold loss weights or quality tags on openpi's draw order, which the LeRobot fork's "
            "row sampler does not draw: compile the method with PermutationPlanner('lerobot-axisrowsampler') and "
            "the run's seed") in str(e.value).splitlines()


class EveryFrameTagged(FrameCodes):
    """Pure conditioning: every trainable frame tagged, none dropped."""

    def condition(self, values, rows):
        c = super().condition(values, rows)
        dense = c.dense.copy()
        dense[dense == 0] = 1
        return dataclasses.replace(c, dense=dense)


@pytest.mark.parametrize("unconditional, refused", [(None, True), (False, False)])
def test_labels_tagging_every_frame_bind_only_as_pure_conditioning(toy, tmp_path, unconditional, refused):
    _, paths = host(tmp_path)
    tagged = Compose(method="toy.tagged", version=1, params={}, planner=PermutationPlanner("lerobot-axisrowsampler"),
                     scorer=FrameIndex(), conditioner=EveryFrameTagged())
    bid = _published(paths, compile_program(tagged, None, toy, Budget(1, 8), 0)[0])
    over = 'axis_expected_frames = 23\naxis_expected_reward = "toy.tagged"'
    if unconditional is not None:
        over += f"\naxis_quality_unconditional = {str(unconditional).lower()}"
    spec = loads_armspec(_arm(toy, bid, overrides=over))
    with pytest.raises(BindError) as e:    # a one-step plan (the toy's 9 frames are no epoch of 8): refused anyway
        bind(spec, load_recipe(spec.preset), paths, smoke=True)
    problem = ("  recipe: signals/quality_toy.tagged.npz tags every frame (no code 0), which the pinned LeRobot fork "
               "trains only as pure conditioning: declare axis_quality_unconditional = false in [recipe.overrides]")
    assert (problem in str(e.value).splitlines()) == refused
