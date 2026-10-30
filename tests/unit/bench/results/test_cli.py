"""`results score|contrast` end to end on fixture files."""
from __future__ import annotations

from pathlib import Path

from axis_qvla.bench.spec import load_benchmark
from axis_qvla.cli import main
from axis_qvla.bench.results.format import dumps_results
from tests.unit.bench.results.fixtures import make_bench, make_results


def _write(tmp_path: Path, name: str, r) -> str:
    p = tmp_path / name
    p.write_bytes(dumps_results(r))
    return str(p)


def test_score_and_contrast(tmp_path: Path, capsys):
    conf = make_bench(tmp_path / "conf", [806, 1050], 3)
    bench = load_benchmark("fixture_bench", conf_dir=conf)
    a = _write(tmp_path, "a.json", make_results({806: [True] * 3, 1050: [True, False, False]}, bench=bench))
    b = _write(tmp_path, "b.json",
              make_results({806: [True, False, False], 1050: [False] * 3}, digest="e" * 64, bench=bench))
    common = ["--benchmark", "fixture_bench", "--bench-conf-dir", str(conf)]
    assert main(["results", "score", a, *common]) == 0
    assert "armx_dualsim\tall24\t66.67 ±" in capsys.readouterr().out
    assert main(["results", "score", a, "--view", "usable_12", *common]) == 0
    assert "(appendix view)" in capsys.readouterr().out
    assert main(["results", "contrast", "--a", a, "--b", b, *common]) == 0
    out = capsys.readouterr().out
    assert "descriptive, n_seeds=1" in out and "no claim" in out
    # the paired difference at each replicate first, with that pair's own eval SE
    assert out.startswith("replicate 42: +50.00 pt (eval SE of this pair ")
    assert "mean +50.00 pt, sample SD n/a, pairs 1" in out


def test_contrast_discloses_a_differing_ft_seed(tmp_path: Path, capsys):
    """A replicate pairs on the pretrain seed; a differing FT seed on
    one side of that pair must be disclosed in the contrast output, not silently absorbed."""
    conf = make_bench(tmp_path / "conf", [806, 1050], 3)
    bench = load_benchmark("fixture_bench", conf_dir=conf)
    good = {806: [True] * 3, 1050: [True, False, False]}
    a = _write(tmp_path, "a.json",
              make_results(good, pretrain=43, seed=43, digest="a43", arm="bc43_dualsim", bench=bench))
    b = _write(tmp_path, "b.json",
              make_results(good, pretrain=43, seed=42, digest="b43", arm="cfgpt43_dualsim", bench=bench))
    common = ["--benchmark", "fixture_bench", "--bench-conf-dir", str(conf)]
    assert main(["results", "contrast", "--a", a, "--b", b, *common]) == 0
    out = capsys.readouterr().out
    assert "ft seeds differ in replicate(s) 43" in out


def test_bad_file_exits_2_and_missing_file_exits_3(tmp_path: Path, capsys):
    conf = make_bench(tmp_path / "conf", [806, 1050], 3)
    (tmp_path / "x.json").write_text('{"schema": "other"}')
    common = ["--benchmark", "fixture_bench", "--bench-conf-dir", str(conf)]
    assert main(["results", "score", str(tmp_path / "x.json"), *common]) == 2
    assert main(["results", "score", str(tmp_path / "absent.json"), *common]) == 3


def test_contrast_relative_prints_the_bc_normalised_statistic(tmp_path: Path, capsys):
    conf = make_bench(tmp_path / "conf", [806, 1050], 3)
    bench = load_benchmark("fixture_bench", conf_dir=conf)
    a = _write(tmp_path, "a.json", make_results({806: [True] * 3, 1050: [True, False, False]}, bench=bench))
    b = _write(tmp_path, "b.json",
              make_results({806: [True, False, False], 1050: [False] * 3}, digest="e" * 64, bench=bench))
    common = ["--benchmark", "fixture_bench", "--bench-conf-dir", str(conf)]
    assert main(["results", "contrast", "--a", a, "--b", b, "--bc", b, "--relative", *common]) == 0
    out = capsys.readouterr().out
    assert "mean +300.00 % of plain BC" in out and "BC-normalized" in out
    assert main(["results", "contrast", "--a", a, "--b", b, "--relative", *common]) == 2
    assert "--relative needs --bc" in capsys.readouterr().out
    assert main(["results", "contrast", "--a", a, "--b", b, "--bc", b, *common]) == 2
