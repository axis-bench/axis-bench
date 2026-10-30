"""`paper reeval`: which results a result, Table I or a figure re-evaluates, the `qvla` commands that
measure them, and the paper-layout outputs built from the user's own results files, blank where a
result was not re-evaluated."""
from __future__ import annotations

import dataclasses
import json
import re

import pytest

from axis_qvla.bench import finish
from axis_qvla.bench.results.format import dumps_results
from axis_qvla.bench.spec import load_benchmark
from axis_qvla.paper import reeval, reeval_tables
from axis_qvla.paper.cells import load_cells
from axis_qvla.paper.cli import DEFAULT_CELLS
from axis_qvla.paper.spec import load_spec
from tests.unit.bench.results.fixtures import make_bench, make_results

CELLS, SPEC = load_cells(DEFAULT_CELLS.read_bytes()), load_spec()


def _sel(kind, name, backbone=None):
    return reeval.select(CELLS, SPEC, reeval.Unit(kind, name, backbone))


def test_table1_is_every_replicate_of_every_role():
    assert len(_sel("table", "table1")) == 72
    pi05 = _sel("table", "table1", "pi05")
    assert len(pi05) == 24 and {c.backbone for c in pi05} == {"pi05"}


def test_fig1_is_the_first_comparison_and_its_bc_per_replicate():
    assert len(_sel("figure", "fig1")) == 18


def test_a_figure_brings_each_of_its_vlas_plain_bc():
    sel = {c.cell for c in _sel("figure", "fig4c")}
    assert len(sel) == 10 and {"svla_bc|bare", "xvla_bc|bare"} <= sel
    assert {c.backbone for c in _sel("figure", "fig4c", "xvla")} == {"xvla"}


def test_a_result_is_named_by_its_cell_or_its_run():
    assert _sel("result", "bc|bare") == _sel("result", "r1bc_r2") and len(_sel("result", "bc|bare")) == 1


@pytest.mark.parametrize("unit, needle", [
    (("result", "nope"), "--list"), (("table", "table2"), "main-results"), (("figure", "fig9"), "random-vs-filtered"),
    (("figure", "main-results"), "the figures are headline, random-vs-filtered"), (("table", "scaling"), "the tables are main-results"),
    (("result", "svla_bc|bare", "pi05"), "is a smolvla result"), (("figure", "fig4c", "pi05"), "reads no pi05 result"),
])
def test_what_names_nothing_is_refused(unit, needle):
    with pytest.raises(reeval.ReevalError, match=re.escape(needle)):
        reeval.select(CELLS, SPEC, reeval.Unit(*unit))


def test_the_eval_command_carries_the_checkpoint_serving_run_and_seeds():
    (c,) = _sel("result", "cfg_ph_ep|cond")
    assert reeval.eval_argv(c) == ("bench", "eval", "--checkpoint-id", c.checkpoint["id"], "--serving", "conditioned",
                                   "--run", "r1cfgphep_c", "--ft-seed", "42", "--pretrain-seed", "42")
    (d,) = _sel("result", "droidinit|bare")
    assert "--pretrain-seed" not in reeval.eval_argv(d) and reeval.eval_argv(d)[-2:] == ("--ft-seed", "42")


def test_the_plan_fetches_once_then_evaluates_each_result():
    sel = _sel("table", "table1", "pi05")
    steps = reeval.plan(sel, host="my.toml")
    argvs = [s.argv for s in steps]
    assert argvs[0] == ("data", "get", "bench", "--host", "my.toml")
    assert argvs[1] == ("data", "get", "base-weights", "--serving", "pi05", "--host", "my.toml")
    gets = [a for a in argvs if a[2].startswith("checkpoints/")]
    assert len(gets) == len({c.checkpoint["id"] for c in sel}) == 18
    assert [s.runs for s in steps if s.argv[:2] == ("bench", "eval")] == [(c.run,) for c in sel]
    assert all(a[-2:] == ("--host", "my.toml") for a in argvs)


def test_the_unit_spells_its_own_collect_line_and_directory():
    u = reeval.Unit("result", "bc|bare")
    assert u.flags() == ["--result", "bc|bare"] and u.slug() == "bc-bare"
    assert reeval.Unit("table", "main-results", "xvla").flags() == ["--table", "main-results", "--backbone", "xvla"]
    assert reeval.Unit("figure", "random-vs-filtered", "xvla").slug() == "random-vs-filtered-xvla"


@pytest.mark.parametrize("old, new", [("table1", "main-results"), ("fig1", "headline"), ("fig4c", "random-vs-filtered"),
                                      ("fig5", "methods"), ("fig6", "scaling"), ("fig7", "conditioning")])
def test_the_layout_keys_still_name_their_unit(old, new):
    """The names before v1.0's are hidden aliases: the same unit, flags and directory, the same results,
    and the published file it is laid out in keeps the layout's key."""
    kind = "table" if old == "table1" else "figure"
    u = reeval.Unit(kind, old)
    assert u == reeval.Unit(kind, new) and u.name == new and u.key == old and u.flags() == [f"--{kind}", new]
    assert _sel(kind, old) == _sel(kind, new)


def test_list_names_where_each_result_is_used():
    used = reeval.uses(SPEC)
    assert "main-results" in used["bc|bare"] and "headline" in used["bc|bare"]
    assert used["xvla_rand_ep|bare"] == ["random-vs-filtered"]


def _write(root, cell, bench, outcomes):
    c = next(x for x in CELLS.cells if x.cell == cell)
    seed = c.training_seed
    r = make_results(outcomes, arm=c.served, bench=bench, seed=seed["ft"], pretrain=seed["pretrain"])
    p = finish.results_path(root, c.run)
    p.parent.mkdir(parents=True)
    p.write_bytes(dumps_results(r))


def test_tables_hold_what_was_re_evaluated_and_leave_the_rest_blank(tmp_path):
    bench = load_benchmark("fixture_bench", conf_dir=make_bench(tmp_path / "conf", [806, 1050], 3))
    sel = _sel("figure", "fig4c", "xvla")
    half = {806: [True, False, True], 1050: [False, False, True]}
    for cell in ("xvla_bc|bare", "xvla_rand_ep|bare"):
        _write(tmp_path / "r1", cell, bench, half)
    P, missing = reeval_tables.collect(sel, tmp_path / "r1", bench)
    assert set(P) == {"xvla_bc|bare", "xvla_rand_ep|bare"}
    assert set(missing) == {c.cell for c in sel} - set(P)
    out = reeval_tables.tables(P, sel, SPEC)
    rows = {r.split(",")[3]: r.split(",")[4:7] for r in out["figures/fig4c.csv"].decode().splitlines()[1:]}
    assert rows["xvla_rand_ep|bare"] == ["50.0", "50.0", "0.0"]
    assert rows["xvla_rand_row|bare"] == ["", "50.0", ""]
    assert rows["svla_rand_ep|bare"] == ["", "", ""]
    assert json.loads(out["pooled_reeval.json"])["xvla_bc|bare"]["per_task"] == {"806": [2, 3], "1050": [1, 3]}
    assert reeval_tables.summary(reeval.Unit("figure", "fig4c", "xvla"), out, P).startswith("vla,group,label")


@pytest.mark.parametrize("outcomes, kw, needle", [
    ({806: [True, False], 1050: [False, False, True]}, {}, "1 of 6 rollouts are missing"),
    ({806: [True, False, True], 1050: [False, False, True]}, {"ft": 7}, "records training seeds"),
])
def test_a_results_file_that_is_not_the_planned_evaluation_is_refused(tmp_path, outcomes, kw, needle):
    bench = load_benchmark("fixture_bench", conf_dir=make_bench(tmp_path / "conf", [806, 1050], 3))
    (c,) = _sel("result", "xvla_bc|bare")
    seed = {**c.training_seed, **kw}
    r = make_results(outcomes, arm=c.served, bench=bench, seed=seed["ft"], pretrain=seed["pretrain"])
    p = finish.results_path(tmp_path / "r1", c.run)
    p.parent.mkdir(parents=True)
    p.write_bytes(dumps_results(r))
    with pytest.raises(ValueError, match=re.escape(needle)):
        reeval_tables.collect((c,), tmp_path / "r1", bench)


def test_every_fetch_precedes_every_evaluation():
    for unit in (reeval.Unit("table", "table1"), reeval.Unit("figure", "fig1"), reeval.Unit("figure", "fig4c")):
        kinds = [s.argv[:2] for s in reeval.plan(reeval.select(CELLS, SPEC, unit))]
        last_get = max(i for i, k in enumerate(kinds) if k == ("data", "get"))
        assert all(k == ("bench", "eval") for k in kinds[last_get + 1:]) and kinds.count(("bench", "eval")) > 0
        assert ("bench", "eval") not in kinds[:last_get]


def test_a_recorded_command_without_serving_is_refused_not_read_as_bare():
    (c,) = _sel("result", "cfg_ph_ep|cond")
    bare = dataclasses.replace(c, eval_command=c.eval_command.replace("--serving conditioned", ""))
    with pytest.raises(reeval.ReevalError, match="names no --serving"):
        reeval.eval_argv(bare)


def test_a_unit_kind_without_a_layout_is_refused():
    with pytest.raises(reeval.ReevalError, match="only a table or a figure"):
        reeval.layout_cells(SPEC, reeval.Unit("result", "bc|bare"))


def test_collect_names_every_bad_results_file_in_one_error(tmp_path):
    bench = load_benchmark("fixture_bench", conf_dir=make_bench(tmp_path / "conf", [806, 1050], 3))
    sel = _sel("figure", "fig4c", "xvla")
    for c in sel[:2]:
        p = finish.results_path(tmp_path / "r1", c.run)
        p.parent.mkdir(parents=True)
        p.write_text("{}")
    with pytest.raises(ValueError) as exc:
        reeval_tables.collect(sel, tmp_path / "r1", bench)
    assert all(c.run in str(exc.value) for c in sel[:2])
