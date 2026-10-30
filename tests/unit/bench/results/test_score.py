"""The reporting rules: all-24 headline, missing rollouts fail, paired seeds, >= 3 for a claim."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from axis_qvla.bench.spec import Benchmark, load_benchmark
from axis_qvla.paper.pool import pool_cell
from axis_qvla.bench.results.fingerprint import units_sha256
from axis_qvla.bench.results.format import Results, ResultsRow
from axis_qvla.bench.results.score import contrast, score
from tests.unit.bench.results.fixtures import header, make_bench, make_results

GOLD = Path(__file__).resolve().parents[3] / "golden" / "paper"


@pytest.fixture()
def bench(tmp_path: Path):
    return load_benchmark("fixture_bench", conf_dir=make_bench(tmp_path / "conf", [806, 1050], 3))


def test_all24_is_the_headline_and_other_views_are_appendix(bench):
    r = make_results({806: [True, True, False], 1050: [False, False, False]}, bench=bench)
    s = score(r, bench)
    assert (s.n_tasks, s.rollouts, s.missing, s.appendix) == (2, 6, 0, False)
    assert s.pct == pytest.approx(100 * (2 / 3) / 2)
    assert score(r, bench, view="usable_12").appendix is True
    assert score(r, bench, view="usable_12").pct == pytest.approx(100 * 2 / 3)


def test_a_missing_rollout_counts_as_a_failure(bench):
    full = make_results({806: [True, True, True], 1050: [True, True, True]}, bench=bench)
    part = make_results({806: [True, True], 1050: [True, True, True]}, bench=bench)
    assert score(part, bench).missing == 1 and score(part, bench).pct < score(full, bench).pct


def test_results_for_another_benchmark_are_refused(bench):
    r = make_results({806: [True] * 3, 1050: [True] * 3})
    r = dataclasses.replace(r, header=dataclasses.replace(r.header, bench={"name": "other", "units_sha256": "0"}))
    with pytest.raises(ValueError, match="benchmark"):
        score(r, bench)


def test_wrong_declaration_sha_is_refused(bench):
    """Benchmark identity was fail-open on declaration_sha256 (name + units only)."""
    r = make_results({806: [True] * 3, 1050: [True] * 3}, bench=bench)
    r = dataclasses.replace(r, header=dataclasses.replace(
        r.header, bench={**r.header.bench, "declaration_sha256": "WRONG" * 16}))
    with pytest.raises(ValueError, match="benchmark"):
        score(r, bench)


def test_missing_declaration_key_fails_closed(bench):
    r = make_results({806: [True] * 3, 1050: [True] * 3}, bench=bench)
    stripped = {k: v for k, v in r.header.bench.items() if k != "declaration_sha256"}
    r = dataclasses.replace(r, header=dataclasses.replace(r.header, bench=stripped))
    with pytest.raises(ValueError, match="benchmark"):
        score(r, bench)


def test_a_row_for_an_undeclared_seed_is_refused(bench):
    r = make_results({806: [True] * 3, 1050: [True] * 3}, bench=bench)
    stray = dataclasses.replace(r, rows=r.rows + (ResultsRow(806, 806999, True, 100, 6.6, "success"),))
    with pytest.raises(ValueError, match="undeclared"):
        score(stray, bench)


def test_score_uses_the_benchmarks_declared_seeds_not_a_formula(tmp_path: Path):
    """Declared seeds must come from TaskEntry.seeds, not a t*1000+i formula."""
    conf = tmp_path / "conf"
    conf.mkdir()
    decl = {"name": "oddseeds", "version": "v1", "n_seeds": 2, "dual_sim": True,
            "action_space": "droid8", "image_geometry": "raw", "xla_pin": "",
            "tasks_file": "oddseeds.tasks.json", "layouts_file": "oddseeds.layouts.json",
            "task_ids": [9], "subsets": {"informative_20": [9], "usable_12": [9]}}
    tasks = {"tasks": {"9": {"task_name_canonical": "task nine", "task_name": None,
                             "test_seeds": [111, 222],
                             "gripper": {"open_w": 0.08, "closed_w": 0.0}}}}
    layouts = {"tasks": {"9": {"train": [0]}}}
    (conf / "oddseeds.json").write_text(json.dumps(decl))
    (conf / "oddseeds.tasks.json").write_text(json.dumps(tasks))
    (conf / "oddseeds.layouts.json").write_text(json.dumps(layouts))
    odd = load_benchmark("oddseeds", conf_dir=conf)
    h = dataclasses.replace(header("f" * 64), bench={"name": odd.name, "declaration_sha256": odd.sha256,
                                                      "units_sha256": units_sha256(odd.task_ids, odd.n_seeds)})
    r = Results(h, (ResultsRow(9, 111, True, 100, 6.6, "success"), ResultsRow(9, 222, False, 100, 6.6, "timeout")))
    s = score(r, odd)
    assert (s.n_tasks, s.rollouts, s.missing) == (1, 2, 0)
    assert s.pct == pytest.approx(50.0)


def test_full_arm_equals_the_pooled_paper_cell():
    real = load_benchmark()
    cell = json.loads((GOLD / "pooled_r1.json").read_text())["bc|bare"]
    outcomes = {int(t): [i < s for i in range(n)] for t, (s, n) in cell["per_task"].items()}
    r = Results(dataclasses.replace(header("f" * 64),
                                    bench={"name": real.name, "declaration_sha256": real.sha256,
                                           "units_sha256": units_sha256(real.task_ids, real.n_seeds)}),
                tuple(ResultsRow(t, t * 1000 + i, ok, 1, 0.1, "success") for t in sorted(outcomes)
                      for i, ok in enumerate(outcomes[t])))
    s = score(r, real)
    tallies = {int(t): tuple(v) for t, v in cell["per_task"].items()}
    assert (s.pct, s.se) == (pool_cell("r1bc_r2", tallies)["agg"], pool_cell("r1bc_r2", tallies)["se"])
    assert (s.pct, s.se) == (cell["agg"], cell["se"])


def test_two_seeds_are_descriptive_three_allow_a_claim(bench):
    good = {806: [True] * 3, 1050: [True, False, False]}
    bad = {806: [True, False, False], 1050: [False] * 3}
    a = [make_results(good, seed=s, digest=f"a{s}", arm=f"a{s}_dualsim", bench=bench) for s in (42, 43)]
    b = [make_results(bad, seed=s, digest=f"b{s}", arm=f"b{s}_dualsim", bench=bench) for s in (42, 43)]
    c2 = contrast(a, b, bench)
    assert (c2.n_pairs, c2.claim_allowed, c2.label) == (2, False, "descriptive, n_seeds=2")
    assert c2.sd == 0.0 and len(c2.eval_se) == 2
    a.append(make_results(good, seed=44, digest="a44", arm="a44_dualsim", bench=bench))
    b.append(make_results(bad, seed=44, digest="b44", arm="b44_dualsim", bench=bench))
    c3 = contrast(a, b, bench)
    assert (c3.claim_allowed, c3.label) == (True, "paired, n_seeds=3")
    assert c3.mean == pytest.approx(100 * ((1 + 1 / 3) - (1 / 3)) / 2)


def test_replicates_pair_on_the_pretrain_seed_and_disclose_ft_mismatch(bench):
    """A replicate pairs on the pretrain seed: seed43_bc ran FT seed 43 while seed43_cfgpt
    ran FT seed 42 -- pairing must use the pretrain seed, not the full (pretrain, ft) tuple."""
    good = {806: [True] * 3, 1050: [True, False, False]}
    bad = {806: [True, False, False], 1050: [False] * 3}
    a = [make_results(good, pretrain=42, seed=42, digest="bc42", arm="bc42_dualsim", bench=bench),
         make_results(good, pretrain=43, seed=43, digest="bc43", arm="bc43_dualsim", bench=bench),
         make_results(good, pretrain=44, seed=42, digest="bc44", arm="bc44_dualsim", bench=bench)]
    b = [make_results(bad, pretrain=42, seed=42, digest="cfgpt42", arm="cfgpt42_dualsim", bench=bench),
         make_results(bad, pretrain=43, seed=42, digest="cfgpt43", arm="cfgpt43_dualsim", bench=bench),
         make_results(bad, pretrain=44, seed=42, digest="cfgpt44", arm="cfgpt44_dualsim", bench=bench)]
    c = contrast(a, b, bench)
    assert (c.n_pairs, c.claim_allowed) == (3, True)
    assert c.ft_seed_differs == (43,)
    assert "paired, n_seeds=3" in c.label and "replicate" in c.label and "43" in c.label


def test_an_arm_with_no_pretrain_pairs_on_its_ft_seed(bench):
    """An arm with no pretraining has no pretrain seed to key on, so its replicate is its FT
    seed -- which may coincide with another arm's pretrain seed (both are "42")."""
    good = {806: [True] * 3, 1050: [True, False, False]}
    bad = {806: [True, False, False], 1050: [False] * 3}
    a = [make_results(good, pretrain=None, seed=42, digest="bare42", arm="bare42_dualsim", bench=bench)]
    b = [make_results(bad, pretrain=42, seed=42, digest="bc42", arm="bc42_dualsim", bench=bench)]
    c = contrast(a, b, bench)
    assert (c.n_pairs, c.ft_seed_differs) == (1, ())


def test_two_entries_at_the_same_replicate_key_are_refused(bench):
    good = {806: [True] * 3, 1050: [True, False, False]}
    a = [make_results(good, pretrain=43, seed=43, digest="x1", arm="a1_dualsim", bench=bench),
         make_results(good, pretrain=43, seed=42, digest="x2", arm="a2_dualsim", bench=bench)]
    b = [make_results(good, pretrain=43, seed=42, digest="y1", arm="b1_dualsim", bench=bench)]
    with pytest.raises(ValueError, match="missing or repeated"):
        contrast(a, b, bench)


def test_re_reads_and_unpaired_seeds_are_refused(bench):
    x = make_results({806: [True] * 3, 1050: [True] * 3}, seed=42, digest="same")
    y = make_results({806: [True] * 3, 1050: [True] * 3}, seed=43, digest="same")
    with pytest.raises(ValueError, match="re-reads"):
        contrast([x, y], [x], bench)
    z = make_results({806: [True] * 3, 1050: [True] * 3}, seed=44, digest="other")
    with pytest.raises(ValueError, match="do not pair"):
        contrast([x], [z], bench)


# --- declared lineage: an older declaration the benchmark names as compatible -------------------

OLD_DECL = "a" * 64


@pytest.fixture()
def lineage_bench(tmp_path: Path):
    extra = {"compatible_declarations": [{"sha256": OLD_DECL, "reason": "fixture: an older edition"}]}
    return load_benchmark("fixture_bench", conf_dir=make_bench(tmp_path / "conf", [806, 1050], 3, extra=extra))


def _stamped(bench: Benchmark, decl: str, units: str | None = None) -> Results:
    r = make_results({806: [True] * 3, 1050: [True, False, False]}, bench=bench)
    return dataclasses.replace(r, header=dataclasses.replace(r.header, bench={
        "name": bench.name, "declaration_sha256": decl,
        "units_sha256": units or units_sha256(bench.task_ids, bench.n_seeds)}))


def test_a_header_stamped_with_a_declared_compatible_sha_is_accepted(lineage_bench):
    assert score(_stamped(lineage_bench, OLD_DECL), lineage_bench).rollouts == 6


def test_an_unknown_declaration_sha_is_refused_even_with_a_lineage(lineage_bench):
    with pytest.raises(ValueError, match="declaration"):
        score(_stamped(lineage_bench, "b" * 64), lineage_bench)


def test_a_compatible_sha_with_different_units_is_refused(lineage_bench):
    other_units = units_sha256(lineage_bench.task_ids, lineage_bench.n_seeds + 1)
    with pytest.raises(ValueError, match="units"):
        score(_stamped(lineage_bench, OLD_DECL, other_units), lineage_bench)


def test_a_compatible_sha_is_refused_by_a_benchmark_that_does_not_declare_it(bench):
    with pytest.raises(ValueError, match="declaration"):
        score(_stamped(bench, OLD_DECL), bench)


def test_an_r1_stamped_full_arm_scores_against_the_shipped_benchmark():
    """The paper's results carry the declaration sha they were run under; the shipped declaration
    has changed since (loader keys, informative_20, v1 input pins, dev_seeds), and its declared
    lineage is what lets those results be scored."""
    real = load_benchmark()
    r1 = "fcff87cbf738fb15849f15adf91e06e8d3e88434c6aa0b0f8d19d6fe209eb037"
    assert real.sha256 != r1
    cell = json.loads((GOLD / "pooled_r1.json").read_text())["bc|bare"]
    outcomes = {int(t): [i < s for i in range(n)] for t, (s, n) in cell["per_task"].items()}
    r = Results(dataclasses.replace(header("f" * 64), bench={
        "name": real.name, "declaration_sha256": r1,
        "units_sha256": units_sha256(real.task_ids, real.n_seeds)}),
        tuple(ResultsRow(t, t * 1000 + i, ok, 1, 0.1, "success") for t in sorted(outcomes)
              for i, ok in enumerate(outcomes[t])))
    assert (score(r, real).pct, score(r, real).se) == (cell["agg"], cell["se"])


def test_relative_contrast_is_table_is_bc_normalised_difference(bench):
    """`--relative`: per replicate 100 (SR_A - SR_B) / SR_BC with that replicate's BC, then mean and
    sample SD -- the paper's Table I statistic -- instead of the raw percentage points."""
    from fractions import Fraction
    import statistics

    a_out = [{806: [True] * 3, 1050: [True, False, False]}, {806: [True] * 3, 1050: [False] * 3},
             {806: [True, True, False], 1050: [True] * 3}]
    b_out = [{806: [True, False, False], 1050: [False] * 3}] * 3
    bc_out = [{806: [True, True, False], 1050: [False] * 3}, {806: [True] * 3, 1050: [True, False, False]},
              {806: [True, False, False], 1050: [True, False, False]}]
    mk = lambda outs, side: [make_results(o, pretrain=s, seed=42, digest=f"{side}{s}", arm=f"{side}{s}_dualsim", bench=bench)  # noqa: E731
                             for s, o in zip((42, 43, 44), outs)]
    c = contrast(mk(a_out, "a"), mk(b_out, "b"), bench, bc=mk(bc_out, "bc"))
    rate = lambda o: (Fraction(sum(o[806]), 3) + Fraction(sum(o[1050]), 3)) / 2   # noqa: E731
    want = [float(100 * (rate(x) - rate(y)) / rate(z)) for x, y, z in zip(a_out, b_out, bc_out)]
    assert c.relative and c.n_pairs == 3 and c.claim_allowed
    assert c.mean == pytest.approx(sum(want) / 3, rel=1e-12) and c.sd == pytest.approx(statistics.stdev(want), rel=1e-12)
    assert "BC-normalized" in c.label            # the paper's spelling (Table I caption)
    raw = contrast(mk(a_out, "a"), mk(b_out, "b"), bench)
    assert not raw.relative and raw.mean == pytest.approx(100 * sum(rate(x) - rate(y) for x, y in zip(a_out, b_out)) / 3)


def test_a_relative_contrast_discloses_an_ft_seed_that_differs_on_the_bc_side(bench):
    """An FT seed that differs inside a replicate is disclosed, the BC normaliser's
    included -- the paper's pi0.5 seed-43 BC was fine-tuned with seed 43, the cells it normalises
    with 42."""
    good = {806: [True] * 3, 1050: [True, False, False]}

    def side(name: str, ft43: int) -> list:
        return [make_results(good, pretrain=s, seed=ft43 if s == 43 else 42, digest=f"{name}{s}",
                         arm=f"{name}{s}_dualsim", bench=bench) for s in (42, 43, 44)]

    c = contrast(side("a", 42), side("b", 42), bench, bc=side("bc", 43))
    assert c.ft_seed_differs == (43,) and "ft seeds differ in replicate(s) 43" in c.label
    assert contrast(side("a", 42), side("b", 42), bench, bc=side("bc", 42)).ft_seed_differs == ()
    assert contrast(side("a", 43), side("b", 42), bench, bc=side("bc", 42)).ft_seed_differs == (43,)


def test_a_relative_contrast_needs_bc_at_every_replicate(bench):
    good = {806: [True] * 3, 1050: [True, False, False]}
    a = [make_results(good, pretrain=s, seed=42, digest=f"a{s}", arm=f"a{s}_dualsim", bench=bench) for s in (42, 43)]
    b = [make_results(good, pretrain=s, seed=42, digest=f"b{s}", arm=f"b{s}_dualsim", bench=bench) for s in (42, 43)]
    with pytest.raises(ValueError, match="do not pair"):
        contrast(a, b, bench, bc=b[:1])
    zero = [make_results({806: [False] * 3, 1050: [False] * 3}, pretrain=s, seed=42, digest=f"z{s}", arm=f"z{s}_dualsim",
                     bench=bench) for s in (42, 43)]
    with pytest.raises(ValueError, match="no success"):
        contrast(a, b, bench, bc=zero)
