"""What the scorer refuses: another benchmark version, an edited frozen version, off-protocol
execution (unless asked), mixed seed sets or execution in one contrast, a quality tag that is not a
condition, and a re-read posing as a replicate. And what a contrast reports: the paired difference
at every training replicate, with that pair's own eval SE."""
from __future__ import annotations

import dataclasses
import math
import shutil
from pathlib import Path

import pytest

from axis_qvla.bench import spec
from axis_qvla.bench.results.format import Results, ServeCondition
from axis_qvla.bench.results.score import contrast, score
from axis_qvla.bench.spec import load_benchmark
from tests.unit.bench.results.fixtures import make_bench, make_dev_results, make_results

GOOD = {806: [True, True, False], 1050: [False, True, True]}
BAD = {806: [True, False, False], 1050: [False, False, False]}


@pytest.fixture()
def bench(tmp_path: Path):
    return load_benchmark("fixture_bench", conf_dir=make_bench(tmp_path / "conf", [806, 1050], 3,
                                                               extra={"dev_seeds": {"offset": 500, "n": 3}}))


@pytest.fixture()
def declared(tmp_path: Path):
    """The fixture benchmark with the release's execution.json: its constants are declared."""
    conf = make_bench(tmp_path / "declared", [806, 1050], 3)
    shutil.copyfile(spec.CONF_DIR / "execution.json", conf / "execution.json")
    return load_benchmark("fixture_bench", conf_dir=conf)


def _with(r: Results, **bench_keys) -> Results:
    return dataclasses.replace(r, header=dataclasses.replace(r.header, bench={**r.header.bench, **bench_keys}))


def _executing(r: Results, **constants) -> Results:
    return dataclasses.replace(r, header=dataclasses.replace(r.header, execution={**r.header.execution, **constants}))


def test_a_result_of_another_version_is_refused_and_a_legacy_one_is_not(bench):
    r = make_results(GOOD, bench=bench)
    assert score(r, bench).pct > 0                                    # no version keys: lineage only
    assert score(_with(r, version="fixture-v1", version_sha256=bench.version_sha256), bench).pct > 0
    with pytest.raises(ValueError, match="never pooled"):
        score(_with(r, version="other", version_sha256="e" * 64), bench)


def test_an_edited_frozen_version_scores_nothing(bench):
    with pytest.raises(ValueError, match="edited after it was frozen"):
        score(make_results(GOOD, bench=bench), dataclasses.replace(bench, frozen_sha256="0" * 64))


def test_off_protocol_results_need_to_be_asked_for_and_are_labelled(bench):
    r = make_results(GOOD, bench=bench)
    off = _executing(r, off_protocol=True)
    with pytest.raises(ValueError, match="off-protocol .*--allow-off-protocol"):
        score(off, bench)
    assert score(off, bench, allow_off_protocol=True).off_protocol is True
    assert score(r, bench).off_protocol is False
    recorded = _executing(r, off_protocol=False, deviations={"replan": {"executed": 4, "declared": 8}})
    with pytest.raises(ValueError, match="replan 4, declared 8"):       # a recorded deviation alone is enough
        score(recorded, bench)


def test_a_constant_that_differs_from_the_declared_one_is_off_protocol(declared):
    r = make_results(GOOD, bench=declared)
    assert score(r, declared).off_protocol is False
    with pytest.raises(ValueError, match="off-protocol"):
        score(_executing(r, replan=4), declared)


def test_a_convention_that_differs_from_the_declared_one_is_off_protocol(declared):
    r = make_results(GOOD, bench=declared)
    letterboxed = dataclasses.replace(r, header=dataclasses.replace(r.header, image_geometry="profile"))
    with pytest.raises(ValueError, match="image_geometry 'profile', declared 'raw'"):
        score(letterboxed, declared)
    assert score(letterboxed, declared, allow_off_protocol=True).off_protocol is True


def test_dev_results_score_on_dev_seeds_and_never_mix_with_test_results(bench):
    d = make_dev_results(bench, GOOD)
    s = score(d, bench)
    assert s.seed_set == "dev" and s.rollouts == 6 and s.missing == 0
    with pytest.raises(ValueError, match="undeclared"):               # dev rows under a test header
        score(Results(make_results(GOOD, bench=bench).header, d.rows), bench)
    with pytest.raises(ValueError, match="units"):                    # a dev header over the test units
        score(_with(make_results(GOOD, bench=bench), seed_set="dev"), bench)
    with pytest.raises(ValueError, match="seed set"):
        contrast([d], [make_results(GOOD, bench=bench, digest="e" * 64)], bench)


def test_an_unknown_seed_set_or_an_undeclared_dev_set_is_refused(bench, declared):
    with pytest.raises(ValueError, match="seed set 'train'"):
        score(_with(make_results(GOOD, bench=bench), seed_set="train"), bench)
    with pytest.raises(ValueError, match="declares no development seeds"):
        score(_with(make_results(GOOD, bench=declared), seed_set="dev"), declared)


def test_an_eval_digest_identifies_a_re_read(bench):
    a = make_results(GOOD, bench=bench, digest=None)
    a = dataclasses.replace(a, header=dataclasses.replace(a.header, checkpoint={**a.header.checkpoint, "eval_digest": "c" * 64}))
    b = dataclasses.replace(a, header=dataclasses.replace(a.header, training_seed={"pretrain": None, "ft": 43}))
    with pytest.raises(ValueError, match="re-reads"):
        contrast([a, b], [a, b], bench)


def test_scores_carry_their_tallies_in_pooled_order(bench):
    s = score(make_results(GOOD, bench=bench), bench)
    assert s.tallies == ((1050, 2, 3), (806, 2, 3))    # task-id STRING order, as pool.py averages


def test_quality_tag_0_is_not_a_serving_condition(bench):
    """Tag 0 is the trainer's untagged sentinel: a file served "conditioned on 0" is neither the bare
    cell nor a conditioned one, so it scores as neither."""
    r = make_results(GOOD, bench=bench)
    for tag, match in (("0", "untagged sentinel"), ("", "not a quality bin"), ("x", "not a quality bin")):
        tagged = dataclasses.replace(r, header=dataclasses.replace(
            r.header, serve=ServeCondition(fields={"quality": tag}, uncond_fields=None, beta=1.0)))
        with pytest.raises(ValueError, match=match):
            score(tagged, bench)
    five = dataclasses.replace(r, header=dataclasses.replace(
        r.header, serve=ServeCondition(fields={"quality": "5"}, uncond_fields=None, beta=1.0)))
    assert score(five, bench).rollouts == 6


def _side(name: str, outcomes: dict, seeds=(42, 43, 44), bench=None, **constants) -> list[Results]:
    return [_executing(make_results(outcomes, pretrain=s, seed=42, digest=f"{name}{s}", arm=f"{name}{s}_dualsim",
                                    bench=bench), **constants) for s in seeds]


# Three replicates whose differences all differ (+66.67, +16.67, +33.33 pt), so pairing the wrong
# replicates, or the population SD, gives another answer.
A_OUT = {42: {806: [True] * 3, 1050: [True, False, False]},          # 66.67
         43: {806: [True, True, False], 1050: [False] * 3},          # 33.33
         44: {806: [True] * 3, 1050: [True] * 3}}                    # 100.00
B_OUT = {42: {806: [False] * 3, 1050: [False] * 3},                  # 0.00
         43: {806: [True, False, False], 1050: [False] * 3},         # 16.67
         44: {806: [True, True, False], 1050: [True, True, False]}}  # 66.67


def _rep(side: str, outs: dict, s: int, bench) -> Results:
    return make_results(outs[s], pretrain=s, seed=42, digest=f"{side}{s}", arm=f"{side}{s}_dualsim", bench=bench)


def test_a_contrast_reports_the_difference_at_every_replicate_with_that_pairs_eval_se(bench):
    a = [_rep("a", A_OUT, s, bench) for s in (42, 43, 44)]
    b = [_rep("b", B_OUT, s, bench) for s in (44, 42, 43)]          # given in another order: paired by key
    c = contrast(a, b, bench)
    assert c.replicates == (42, 43, 44)
    assert c.diffs == pytest.approx([200 / 3, 50 / 3, 100 / 3])
    assert c.mean == pytest.approx(350 / 9)
    # sample SD (n - 1): the deviations from the mean are 250/9, -200/9, -50/9
    assert c.sd == pytest.approx(math.sqrt((250 ** 2 + 200 ** 2 + 50 ** 2) / 81 / 2))
    assert c.sd != pytest.approx(math.sqrt((250 ** 2 + 200 ** 2 + 50 ** 2) / 81 / 3))   # not the population SD
    by_key = {x.header.training_seed["pretrain"]: x for x in b}
    assert c.eval_se == pytest.approx([math.sqrt(score(x, bench).se ** 2 + score(by_key[k], bench).se ** 2)
                                       for k, x in zip((42, 43, 44), a)])


def _pct_se(r: Results, bench) -> tuple[float, float]:
    s = score(r, bench)
    return s.pct, s.se


def test_the_relative_eval_se_is_the_delta_method_one_with_the_bc_reads_own_error(bench):
    a = [_rep("a", A_OUT, s, bench) for s in (42, 43, 44)]
    b = [_rep("b", B_OUT, s, bench) for s in (42, 43, 44)]
    bc_out = {42: A_OUT[43], 43: A_OUT[42],                          # rates 33.33, 66.67, 50.00: no BC
              44: {806: [True, False, False], 1050: [True, True, False]}}   # read is also an A or B read
    bc = [_rep("bc", bc_out, s, bench) for s in (42, 43, 44)]
    independent = contrast(a, b, bench, bc=bc)
    as_b = contrast(a, bc, bench, bc=bc)                              # B IS the plain-BC read
    old_rule = []
    for i in range(3):
        (pa, sea), (pb, seb), (pc, sec) = _pct_se(a[i], bench), _pct_se(b[i], bench), _pct_se(bc[i], bench)
        g, h = 100 / pc, -100 * (pa - pb) / pc ** 2
        assert independent.eval_se[i] == pytest.approx(math.sqrt(g * g * (sea ** 2 + seb ** 2) + h * h * sec ** 2))
        # 100 (A - C) / C = 100 (A / C - 1): var = (100/C)^2 var_A + (100 A / C^2)^2 var_C
        assert as_b.eval_se[i] == pytest.approx(math.sqrt((100 / pc) ** 2 * sea ** 2
                                                          + (100 * pa / pc ** 2) ** 2 * sec ** 2))
        old_rule.append(100 * math.sqrt(sea ** 2 + sec ** 2) / pc)   # sqrt(se_a^2 + se_b^2) scaled by 100/C
    assert as_b.eval_se != pytest.approx(old_rule)                    # the BC read's own error is in it
    with pytest.raises(ValueError, match="no success"):             # replicate 42's B, as the BC, is 0 %
        contrast(a[:1], b[:1], bench, bc=b[:1])


def test_a_contrast_pairs_only_reads_executed_alike(bench):
    a4, b4 = _side("a", GOOD, bench=bench, replan=4, off_protocol=True), _side("b", BAD, bench=bench, replan=4,
                                                                                off_protocol=True)
    with pytest.raises(ValueError, match="off-protocol"):
        contrast(a4, b4, bench)
    c = contrast(a4, b4, bench, allow_off_protocol=True)            # all at replan 4: alike, and labelled
    assert c.n_pairs == 3 and "OFF-PROTOCOL" in c.label
    with pytest.raises(ValueError, match="executed differently .*replan 4 vs 8"):
        contrast(a4, _side("b", BAD, bench=bench), bench, allow_off_protocol=True)
    grip = _side("b", BAD, bench=bench, replan=4, off_protocol=True,
                 deviations={"grip_state": {"executed": "fingers", "declared": "closedness"}})
    with pytest.raises(ValueError, match="grip_state"):             # a deviation only one side executed
        contrast(a4, grip, bench, allow_off_protocol=True)


def test_an_errored_rollout_is_a_broken_harness_not_a_failure(bench):
    r = make_results(GOOD, bench=bench)
    broken = dataclasses.replace(r, rows=(dataclasses.replace(r.rows[0], success=False, reason="error:build_scene: x"),
                                          *r.rows[1:]))
    with pytest.raises(ValueError, match="1 rollouts ended with an `error:` reason .*broken harness"):
        score(broken, bench)
