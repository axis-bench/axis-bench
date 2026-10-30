"""A re-read of a checkpoint is a reproducibility check against its published read: the success
rates agree when |diff| <= 1.96 x sqrt(se_pub^2 + se_new^2), that pair's own evaluation error.
Per-rollout outcomes are counted and reported, not compared."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from axis_qvla.bench.results.format import dumps_results
from axis_qvla.bench.results.reeval import Z95, reeval
from axis_qvla.bench.spec import load_benchmark
from axis_qvla.cli import main
from tests.unit.bench.results.fixtures import make_bench, make_dev_results, make_results


@pytest.fixture()
def bench(tmp_path: Path):
    return load_benchmark("fixture_bench", conf_dir=make_bench(tmp_path / "conf", [806, 1050], 50,
                                                               extra={"dev_seeds": {"offset": 500, "n": 50}}))


def _outcomes(k806: int, k1050: int) -> dict[int, list[bool]]:
    return {806: [i < k806 for i in range(50)], 1050: [i < k1050 for i in range(50)]}


def test_a_close_read_agrees_and_a_far_one_does_not(bench):
    pub = make_results(_outcomes(25, 25), bench=bench)
    near = reeval(pub, make_results(_outcomes(27, 24), bench=bench), bench)       # the same checkpoint
    assert near.agrees and abs(near.z) <= Z95 and near.tolerance == pytest.approx(Z95 * near.se)
    far = reeval(pub, make_results(_outcomes(45, 45), bench=bench), bench)
    assert not far.agrees and far.diff == pytest.approx(40.0)


def test_identical_certain_reads_agree_with_zero_error(bench):
    r = make_results(_outcomes(50, 0), bench=bench)
    g = reeval(r, r, bench)
    assert g.agrees and g.se == 0.0 and g.z is None and (g.flipped, g.compared) == (0, 100)


def test_rollout_flips_are_counted_not_checked(bench):
    """The same success count on different seeds: every changed seed flips, yet the rates agree."""
    pub = make_results(_outcomes(25, 25), bench=bench)
    shifted = make_results({806: [i >= 25 for i in range(50)], 1050: [i < 25 for i in range(50)]}, bench=bench)
    g = reeval(pub, shifted, bench)
    assert g.agrees and g.diff == 0.0 and (g.flipped, g.compared) == (50, 100)


def test_the_checkpoint_is_compared_by_content_digest(bench):
    pub = make_results(_outcomes(25, 25), bench=bench, digest="d" * 64)
    ck = {**pub.header.checkpoint, "pub_digest": None, "eval_digest": "d" * 64}
    mine = dataclasses.replace(pub, header=dataclasses.replace(pub.header, checkpoint=ck))
    assert reeval(pub, mine, bench).same_checkpoint is True          # eval_digest of the published file set
    assert reeval(pub, make_results(_outcomes(25, 25), bench=bench, digest=None), bench).same_checkpoint is None


def test_two_different_checkpoints_are_refused_unless_asked_for(bench):
    """The check applies only to one checkpoint read twice: two files that both name content
    digests and share none are not a re-read."""
    pub = make_results(_outcomes(25, 25), bench=bench, digest="d" * 64)
    other = make_results(_outcomes(25, 25), bench=bench, digest="e" * 64)
    with pytest.raises(ValueError, match="not a re-read: .*only to one checkpoint read twice"):
        reeval(pub, other, bench)
    g = reeval(pub, other, bench, allow_different_checkpoints=True)
    assert g.same_checkpoint is False and g.agrees


def test_a_re_read_reads_the_seeds_and_the_execution_of_the_read_it_checks(bench):
    pub = make_results(_outcomes(25, 25), bench=bench)
    with pytest.raises(ValueError, match="seeds of the read it checks"):
        reeval(pub, make_dev_results(bench, _outcomes(25, 25)), bench)
    off = dataclasses.replace(pub, header=dataclasses.replace(
        pub.header, execution={**pub.header.execution, "max_steps": 900, "off_protocol": True}))
    with pytest.raises(ValueError, match="off-protocol"):
        reeval(pub, off, bench)


def test_the_cli_reports_both_reads_and_exits_1_on_disagreement(tmp_path: Path, bench, capsys):
    conf = str(bench.conf_dir)
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_bytes(dumps_results(make_results(_outcomes(25, 25), bench=bench)))
    b.write_bytes(dumps_results(make_results(_outcomes(45, 45), bench=bench)))
    other = tmp_path / "other.json"
    other.write_bytes(dumps_results(make_results(_outcomes(25, 25), bench=bench, digest="e" * 64)))
    common = ["--benchmark", "fixture_bench", "--bench-conf-dir", conf]
    assert main(["results", "reeval", "--published", str(a), "--new", str(a), *common]) == 0
    out = capsys.readouterr().out
    assert "published  armx_dualsim\tall24\t50.00 ±" in out and "re-read    armx_dualsim\tall24\t50.00 ±" in out
    assert "checkpoint: the same" in out and "0 of 100 (per-rollout outcomes are reported" in out
    assert "reproducibility check: +0.00 pt" in out and ": consistent with the published read" in out
    assert main(["results", "reeval", "--published", str(a), "--new", str(b), *common]) == 1
    out = capsys.readouterr().out
    assert "NOT consistent" in out and "checkpoint: the same" in out
    assert main(["results", "reeval", "--published", str(a), "--new", str(other), *common]) == 2
    assert "not a re-read" in capsys.readouterr().out
    assert main(["results", "reeval", "--published", str(a), "--new", str(other), "--allow-different-checkpoints",
                 *common]) == 0
    assert "checkpoint: DIFFERENT checkpoints, compared anyway (--allow-different-checkpoints)" in capsys.readouterr().out
    assert main(["results", "reeval", "--published", str(a), "--new", str(tmp_path / "absent.json"), *common]) == 3
