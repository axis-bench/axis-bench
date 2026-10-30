"""The compiler assembles a bundle from a program's outputs and refuses any that break a composition rule."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from axis_qvla.signals import planners as pl
from axis_qvla.signals.bundle import bundle_id
from axis_qvla.signals.compile import Compose, CompositionError, compile_program, flat_weights
from axis_qvla.signals.imported import ImportedScores
from axis_qvla.signals.methods.filtering import DropTop50, Random50, RandomFraction, TopFraction
from axis_qvla.signals.methods.random50 import SEED_CONTROL
from axis_qvla.signals.program import Compiled, Conditioning, EpisodeValues, Selection
from axis_qvla.signals.rowspace import episode_keys, flat_rowspace, trainable_blocks
from axis_qvla.signals.schema import BARE, FORBIDDEN, Budget, CondVocab, ServeCondition
from axis_qvla.signals.testing import TOY_TRAINABLE, toy_rowspace

RS = toy_rowspace()
B = Budget(4, 2)
PLANNER = pl.PermutationPlanner("openpi-rowsampler")
VOCAB = CondVocab(("quality",), {"quality": {q: f"\nQuality: {q}" for q in range(1, 6)}}, RS.prompts)


class FrameIndex:
    """Scorer: every frame scores its own index."""

    def score(self, labels, rows):
        return EpisodeValues({k: np.arange(n, dtype=np.float64)
                              for k, n in zip(episode_keys(rows), rows.arrays["episode_length"].tolist())})


class PlusOne:
    """Weighter: w = 1 + value."""

    def weigh(self, values, rows):
        return EpisodeValues({k: 1.0 + v for k, v in values.values.items()})


def _keep(rows, values, pick):
    blocks = trainable_blocks(rows)
    return Selection(tuple(pick(b[1]) for b in blocks), tuple(pick(values.values[b[0]][b[2]]) for b in blocks),
                     rows.n_trainable)


class EverySecond:
    """Selector: every second trainable row of each episode (5 rows); its size-matched control keeps
    the same number of rows from each episode's end."""

    def select(self, values, rows):
        return _keep(rows, values, lambda a: a[::2])

    def control(self):
        return LastHalf()


class LastHalf:
    """Keeps the last ceil(n/2) trainable rows of each episode: {2, 7, 17, 21, 22}."""

    def select(self, values, rows):
        return _keep(rows, values, lambda a: a[len(a) // 2:])

    def control(self):
        return None


class FirstOnly:
    """Keeps the first trainable row of each episode: 3 rows, not size-matched to EverySecond."""

    def select(self, values, rows):
        return _keep(rows, values, lambda a: a[:1])

    def control(self):
        return None


@dataclasses.dataclass(frozen=True, eq=False)
class Fixed:
    """A program that returns a prepared Compiled, to test one rule at a time."""

    compiled: Compiled

    def spec(self):
        return {"method": "fixed", "version": 1, "params": {}}

    def run(self, labels, rows, budget, seed):
        return self.compiled


def _plan(seed=42):
    return PLANNER.plan(RS, None, None, B, seed)


def _cond(guided: bool, drop: bool) -> Conditioning:
    dense = np.full((RS.n_rows, 1), FORBIDDEN, dtype=np.uint8)
    dense[list(TOY_TRAINABLE), 0] = 5
    if drop:
        dense[list(TOY_TRAINABLE)[::2], 0] = 0
    serve = (BARE, ServeCondition({"quality": "5"}, {} if guided else None, 2.0 if guided else 1.0))
    return Conditioning(dense, VOCAB, serve)


def _cond_program(guided, drop, dense=None):
    c = _cond(guided, drop)
    if dense is not None:
        c = dataclasses.replace(c, dense=dense)
    return Fixed(Compiled(plan=_plan(), planner=PLANNER, produced={"plan": ("P",), "cond": ("C",)}, conditioning=c))


def test_a_bc_program_compiles_to_the_planner_stream():
    prog = Compose(method="toy.bc", version=1, params={}, planner=PLANNER)
    b, controls = compile_program(prog, None, RS, B, 42)
    np.testing.assert_array_equal(b.plan.ravel(), np.array(TOY_TRAINABLE)[pl.torch_randperm(9, 42)][:8])
    assert (b.reducer, b.requires, b.weights, b.cond, controls) == ("mean", frozenset({"plan"}), None, None, ())
    assert b.program == {"method": "toy.bc", "version": 1, "params": {}}


def test_weights_are_float32_per_corpus_row_with_nan_off_the_trainable_rows():
    prog = Compose(method="toy.w", version=1, params={}, planner=PLANNER, scorer=FrameIndex(), weighter=PlusOne(),
                   loss_reducer="global_ratio")
    b = prog.compile(None, RS, B, 42)
    assert b.weights.dtype == np.float32 and b.reducer == "global_ratio" and "loss_weights" in b.requires
    assert np.isnan(b.weights[0]) and b.weights[1] == 2.0 and b.weights[22] == 8.0
    np.testing.assert_array_equal(b.weights, flat_weights(RS, PlusOne().weigh(FrameIndex().score(None, RS), RS)))


def test_a_coverage_neutral_plan_must_equal_bcs():
    with pytest.raises(CompositionError, match="BC's plan"):
        compile_program(Fixed(Compiled(plan=_plan(43), planner=PLANNER, produced={"plan": ("P",)})), None, RS, B, 42)


def test_each_output_has_exactly_one_producing_block():
    c = Compiled(plan=_plan(), planner=PLANNER, produced={"plan": ("P",), "weights": ("W1", "W2")},
                 weights=PlusOne().weigh(FrameIndex().score(None, RS), RS), reducer="global_ratio")
    with pytest.raises(CompositionError, match="exactly one producing block"):
        compile_program(Fixed(c), None, RS, B, 42)


def test_a_plan_outside_the_trainable_rows_is_refused():
    plan = _plan().copy()
    plan[0, 0] = 0                       # frame 0 of task_800 episode 0 is not trainable
    c = Compiled(plan=plan, planner=PLANNER, produced={"plan": ("P",)}, selection=Selection((), (), 9))
    with pytest.raises(CompositionError, match="outside the trainable set"):
        compile_program(Fixed(c), None, RS, B, 42)


def test_pure_conditioning_compiles_and_guidance_needs_the_unconditional_branch():
    pure = compile_program(_cond_program(guided=False, drop=False), None, RS, B, 42)[0]
    assert pure.cond.shape == (4, 2, 1) and set(np.unique(pure.cond).tolist()) == {5}
    assert pure.requires == frozenset({"plan", "cond_prompt"})
    with pytest.raises(CompositionError, match="unconditional branch"):
        compile_program(_cond_program(guided=True, drop=False), None, RS, B, 42)
    guided = compile_program(_cond_program(guided=True, drop=True), None, RS, B, 42)[0]
    assert {"cond_prompt", "guidance"} <= guided.requires


def test_a_forbidden_code_on_a_drawn_row_is_refused():
    dense = _cond(False, False).dense.copy()
    dense[_plan()[0, 0], 0] = FORBIDDEN
    with pytest.raises(CompositionError, match="forbidden"):
        compile_program(_cond_program(False, False, dense=dense), None, RS, B, 42)


def test_a_selector_emits_its_control_and_the_bundle_records_it():
    prog = Compose(method="toy.every_second", version=1, params={}, planner=PLANNER, scorer=FrameIndex(),
                   selector=EverySecond())
    b, controls = compile_program(prog, None, RS, B, 42)
    assert len(controls) == 1 and b.controls == (bundle_id(controls[0]),)
    assert controls[0].program["method"] == "toy.every_second.control"
    assert set(controls[0].plan.ravel().tolist()) <= {2, 7, 17, 21, 22}
    assert set(b.plan.ravel().tolist()) <= {1, 6, 15, 17, 22}


def test_codes_are_gathered_only_for_a_plan_inside_the_row_space():
    for bad in (RS.n_rows, -1):
        plan = _plan().copy()
        plan[0, 0] = bad
        c = Compiled(plan=plan, planner=PLANNER, produced={"plan": ("P",), "cond": ("C",)},
                     selection=Selection((), (), 9), conditioning=_cond(False, False))
        with pytest.raises(CompositionError, match="outside the trainable set"):
            compile_program(Fixed(c), None, RS, B, 42)



# --- the coverage-neutral rule: a Selector-less plan is the canonical BC permutation, or declared otherwise


class OneRow:
    """A "BC" planner that claims openpi's convention but draws one row every time."""

    convention = "openpi-rowsampler"

    def plan(self, rows, selection, values, budget, seed):
        return np.full((budget.steps, budget.global_batch), TOY_TRAINABLE[0], dtype=np.int64)


@dataclasses.dataclass(frozen=True)
class TopScored:
    """A planner that draws the highest-scoring trainable rows: it reads the scores."""

    convention: str = "toy-top-score"

    def plan(self, rows, selection, values, budget, seed):
        flat = np.concatenate([values.values[k][f] for k, _, f, _ in trainable_blocks(rows)])
        best = np.concatenate([g for _, g, _, _ in trainable_blocks(rows)])[np.argsort(-flat, kind="stable")]
        return np.resize(best[:budget.global_batch], budget.samples).reshape(budget.steps, budget.global_batch)


def test_a_single_row_bc_planner_is_refused():
    with pytest.raises(CompositionError, match="BC's plan"):
        compile_program(Compose(method="toy.one_row", version=1, params={}, planner=OneRow()), None, RS, B, 42)


@pytest.mark.parametrize("convention,match", [
    ("toy-top-score", "has no canonical BC permutation: .* declares coverage_neutral = False"),
    ("openpi-rowsampler", "must draw exactly BC's plan: PermutationPlanner\\('openpi-rowsampler'\\)"),
])
def test_a_score_using_planner_must_declare_itself_non_neutral(convention, match):
    prog = Compose(method="toy.top", version=1, params={}, planner=TopScored(convention=convention),
                   scorer=FrameIndex())
    with pytest.raises(CompositionError, match=match):
        compile_program(prog, None, RS, B, 42)


def test_a_declared_non_neutral_planner_compiles():
    prog = Compose(method="toy.top", version=1, params={}, planner=TopScored(), scorer=FrameIndex(),
                   coverage_neutral=False)
    b, controls = compile_program(prog, None, RS, B, 42)
    np.testing.assert_array_equal(b.plan, TopScored().plan(RS, None, FrameIndex().score(None, RS), B, 42))
    assert b.program == {"method": "toy.top", "version": 1, "params": {}, "coverage_neutral": False}
    assert controls == () and set(b.plan.ravel().tolist()) <= set(TOY_TRAINABLE)


def test_the_built_in_planners_compile():
    for convention in pl.CONVENTIONS:
        planner = pl.PermutationPlanner(convention)
        b = Compose(method="toy.bc", version=1, params={}, planner=planner).compile(None, RS, B, 7)
        np.testing.assert_array_equal(b.plan, planner.plan(RS, None, None, B, 7))
    prog = Compose(method="toy.drop", version=1, params={}, planner=pl.SchedulePlanner("drop_top"),
                   scorer=FrameIndex(), selector=EverySecond())
    b, controls = compile_program(prog, None, RS, B, 0)
    assert set(b.plan.ravel().tolist()) <= {1, 6, 15, 17, 22} and len(controls) == 1
    assert set(controls[0].plan.ravel().tolist()) <= {2, 7, 17, 21, 22}


# --- guidance: a drawn row must carry exactly the unconditional pass's codes

VOCAB2 = CondVocab(("quality", "speed"), {"quality": {q: f"\nQuality: {q}" for q in range(1, 6)},
                                          "speed": {1: "\nSpeed: slow", 2: "\nSpeed: fast"}}, RS.prompts)


def _two_field_program(uncond):
    dense = np.full((RS.n_rows, 2), FORBIDDEN, dtype=np.uint8)
    dense[list(TOY_TRAINABLE)] = (5, 2)
    dense[list(TOY_TRAINABLE)[::2], 1] = 0          # speed dropped on 1, 6, 15, 17, 22
    dense[list(TOY_TRAINABLE)[1::2], 0] = 0         # quality dropped on 2, 7, 16, 21; never both
    serve = (BARE, ServeCondition({"quality": "5", "speed": "2"}, uncond, 2.0))
    return Fixed(Compiled(plan=_plan(), planner=PLANNER, produced={"plan": ("P",), "cond": ("C",)},
                          conditioning=Conditioning(dense, VOCAB2, serve)))


def test_guidance_needs_a_drawn_row_coded_as_the_unconditional_pass():
    with pytest.raises(CompositionError, match="unconditional branch"):
        compile_program(_two_field_program({}), None, RS, B, 42)        # needs (0, 0): never drawn
    for uncond in ({"quality": "5"}, {"speed": "2"}):                    # needs (5, 0) / (0, 2): drawn
        b = compile_program(_two_field_program(uncond), None, RS, B, 42)[0]
        assert "guidance" in b.requires


# --- controls: required of a program's Selector, size-matched, kept, bounded


class Unmatched(EverySecond):
    def control(self):
        return FirstOnly()


class NoControl(EverySecond):
    def control(self):
        return None


class ChainedLastHalf(LastHalf):
    def control(self):
        return LastHalf()


class Chained(EverySecond):
    def control(self):
        return ChainedLastHalf()


class SelfControlled(EverySecond):
    def control(self):
        return self


def _selecting(selector):
    return Compose(method="toy.sel", version=1, params={}, planner=PLANNER, scorer=FrameIndex(), selector=selector)


def test_a_control_must_be_size_matched():
    with pytest.raises(CompositionError, match="size-matched"):
        compile_program(_selecting(Unmatched()), None, RS, B, 42)


def test_a_selector_without_a_control_compiles():
    """A random control is itself a Selector program whose control() is None: it compiles standalone."""
    b, controls = compile_program(_selecting(NoControl()), None, RS, B, 42)
    assert controls == () and b.controls == () and set(b.plan.ravel().tolist()) <= {1, 6, 15, 17, 22}


# --- the built-in selectors and filtering programs under the control rules

# Scores shaped like an AWR weights file over the toy row space: every frame scores its index.
_FRAME_INDEX = FrameIndex().score(None, RS).values
TPL = ImportedScores({"version": 1, "reward_id": "toy", "tau": 10.0, "cap": 2.0, "horizon": 10, "n_episodes": 4,
                      "n_rows": RS.n_rows, "weights": {k: v.tolist() for k, v in _FRAME_INDEX.items()}},
                     "0" * 64, "toy_weights.json")


def test_compose_auto_emits_the_selectors_size_matched_random_control():
    prog = Compose(method="toy.top_half", version=1, params={}, planner=pl.SchedulePlanner("drop_top"),
                   scorer=FrameIndex(), selector=TopFraction(0.5))
    b, controls = compile_program(prog, None, RS, B, 0)
    assert len(controls) == 1 and b.controls == (bundle_id(controls[0]),)
    assert controls[0].program["method"] == "toy.top_half.control"
    assert TopFraction(0.5).control() == RandomFraction(0.5, SEED_CONTROL) and RandomFraction().control() is None
    values = FrameIndex().score(None, RS)
    want = RandomFraction(0.5, SEED_CONTROL).select(values, RS)
    assert want.kept == TopFraction(0.5).select(values, RS).kept == 4      # round(0.5 * 9)
    assert set(controls[0].plan.ravel().tolist()) <= set(np.concatenate(want.blocks).tolist())


def test_a_standalone_random_control_and_a_drop_top_without_a_control_compile():
    random50 = Random50("chunk", TPL)
    assert compile_program(random50, None, RS, B, 42)[1] == ()
    b, controls = compile_program(DropTop50(), TPL, RS, B, 0)
    assert controls == () and b.controls == ()
    b, controls = compile_program(DropTop50(control=random50, control_seed=42), TPL, RS, B, 0)
    assert b.controls == (bundle_id(controls[0]),) and controls[0].seed == 42
    with pytest.raises(CompositionError, match="size-matched"):
        compile_program(DropTop50(control=Random50("chunk", TPL, keep=0.25), control_seed=42), TPL, RS, B, 0)


def test_a_plan_outside_the_selections_kept_rows_is_refused():
    sel = EverySecond().select(FrameIndex().score(None, RS), RS)
    c = Compiled(plan=_plan(), planner=PLANNER, produced={"plan": ("P",)}, selection=sel)
    with pytest.raises(CompositionError, match="outside the selection's kept rows"):
        compile_program(Fixed(c), None, RS, B, 42)


def test_a_controls_own_controls_are_kept():
    b, controls = compile_program(_selecting(Chained()), None, RS, B, 42)
    assert [c.program["method"] for c in controls] == ["toy.sel.control", "toy.sel.control.control"]
    assert b.controls == (bundle_id(controls[0]),) and controls[0].controls == (bundle_id(controls[1]),)
    assert controls[1].controls == ()


def test_control_recursion_is_bounded():
    with pytest.raises(CompositionError, match="nest deeper"):
        compile_program(_selecting(SelfControlled()), None, RS, B, 42)


# --- serve conditions follow the schema's rules


def _serve_program(*serve):
    c = _cond(guided=False, drop=True)
    return Fixed(Compiled(plan=_plan(), planner=PLANNER, produced={"plan": ("P",), "cond": ("C",)},
                          conditioning=dataclasses.replace(c, serve=(BARE,) + serve)))


def test_serve_conditions_follow_the_schema_rules():
    b = compile_program(_serve_program(ServeCondition({"quality": "0"})), None, RS, B, 42)[0]
    assert b.serve[1].fields == {"quality": "0"}                         # code 0 is the dropped field
    for bad in (ServeCondition({"speed": "1"}), ServeCondition({"quality": "9"}),
                ServeCondition({"quality": "high"}), ServeCondition({"quality": "5"}, {"speed": "0"}, 2.0),
                ServeCondition({"quality": "5"}, {"quality": "9"}, 2.0)):
        with pytest.raises(CompositionError, match="serve condition"):
            compile_program(_serve_program(bad), None, RS, B, 42)


# --- flat row spaces carry plans only

FLAT = flat_rowspace(name="flat12", n_rows=12, source="tests")


class NoScores:
    def score(self, labels, rows):
        return EpisodeValues({})


def test_episode_signals_over_a_flat_row_space_are_refused():
    b = Compose(method="toy.bc", version=1, params={}, planner=PLANNER).compile(None, FLAT, B, 42)
    np.testing.assert_array_equal(b.plan, PLANNER.plan(FLAT, None, None, B, 42))
    weighted = Compose(method="toy.w", version=1, params={}, planner=PLANNER, scorer=NoScores(), weighter=PlusOne(),
                       loss_reducer="global_ratio")
    selecting = Compose(method="toy.s", version=1, params={}, planner=PLANNER, scorer=NoScores(),
                        selector=EverySecond())
    for prog in (weighted, selecting):
        with pytest.raises(CompositionError, match="flat frame index"):
            compile_program(prog, None, FLAT, B, 42)


# --- the remaining refusal branches, one each


def _fixed(**kw):
    kw.setdefault("plan", _plan())
    kw.setdefault("produced", {"plan": ("P",)})
    return Fixed(Compiled(planner=PLANNER, **kw))


WEIGHTS = PlusOne().weigh(FrameIndex().score(None, RS), RS)


@pytest.mark.parametrize("kw,match", [
    (dict(produced={}), "no producing block"),
    (dict(produced={"plan": ("P",), "weights": ("W",)}), "does not match"),
    (dict(weights=WEIGHTS, reducer="global_ratio"), "does not match"),
    (dict(produced={"plan": ("P",), "cond": ("C",)}), "does not match"),
    (dict(plan=_plan().astype(np.int32)), "not int64"),
    (dict(plan=_plan().ravel()), "not int64"),
    (dict(plan=_plan().astype(np.float64), produced={"plan": ("P",), "cond": ("C",)},
          conditioning=_cond(False, False)), "not int64"),
    (dict(reducer="global_ratio"), "without weights"),
    (dict(produced={"plan": ("P",), "weights": ("W",)}, weights=WEIGHTS), "global_ratio or per_rank_ratio"),
    (dict(produced={"plan": ("P",), "weights": ("W",)}, weights=WEIGHTS, reducer="sum"),
     "global_ratio or per_rank_ratio"),
    (dict(diagnostics={"keep_frac": float("nan")}), "not a finite number"),
    (dict(diagnostics={"keep_frac": float("inf")}), "not a finite number"),
    (dict(diagnostics={"keep_frac": "half"}), "not a finite number"),
])
def test_each_refusal_branch(kw, match):
    with pytest.raises(CompositionError, match=match):
        compile_program(_fixed(**kw), None, RS, B, 42)


@pytest.mark.parametrize("dense,match", [
    (np.full((RS.n_rows, 1), 5, dtype=np.int64), "codes must be uint8"),
    (np.full((RS.n_rows - 1, 1), 5, dtype=np.uint8), "codes must be uint8"),
    (np.full(RS.n_rows, 5, dtype=np.uint8), "codes must be uint8"),
    (np.full((RS.n_rows, 2), 5, dtype=np.uint8), "do not match the plan and 1 fields"),
    (np.full((RS.n_rows, 1), 7, dtype=np.uint8), "have no text"),
])
def test_codes_of_the_wrong_shape_dtype_or_text_are_refused(dense, match):
    with pytest.raises(CompositionError, match=match):
        compile_program(_cond_program(False, False, dense=dense), None, RS, B, 42)


def test_flat_weights_refusals():
    good = dict(FrameIndex().score(None, RS).values)
    np.testing.assert_array_equal(flat_weights(RS, EpisodeValues(dict(good, **{"task_800--0": np.r_[np.nan, good[
        "task_800--0"][1:]]})))[1:3], [1.0, 2.0])                       # NaN off the trainable frames is fine
    missing = {k: v for k, v in good.items() if k != "task_800--1"}
    short = dict(good, **{"task_800--1": np.arange(3.0)})
    nonfinite = dict(good, **{"task_1000--0": np.r_[good["task_1000--0"][:6], np.inf, good["task_1000--0"][7:]]})
    for values, match in ((missing, "no weight"), (short, "3 weights for a 4-frame episode"),
                          (nonfinite, "non-finite weight")):
        with pytest.raises(CompositionError, match=match):
            flat_weights(RS, EpisodeValues(values))
