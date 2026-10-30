"""`qvla results build`: an evaluation on disk becomes a results file the scorer accepts, and
anything that cannot be tied together is refused."""
from __future__ import annotations

import dataclasses
import json

import pytest

from axis_qvla.bench import execution
from axis_qvla.bench.results import build
from axis_qvla.bench.results.build import BuildError, build_results, executed_constants, serve_condition
from axis_qvla.bench.results.fingerprint import eval_fingerprint, units_sha256
from axis_qvla.bench.results.format import load_results
from axis_qvla.bench.results.score import score
from axis_qvla.bench.results.seedfiles import read_served_dir
from axis_qvla.bench.sim import arms
from axis_qvla.bench.spec import load_benchmark
from axis_qvla.cli import main
from tests.unit.bench.results.fixtures import MD, make_bench, make_run

TAGGED = {**MD, "cfg_conditioning": True, "guidance_scale": 0.0, "quality_tag": 5}


def _build(tmp_path, bench, arm="mine", **kw):
    return build_results(tmp_path / "r1", arm, bench, ft_seed=0, pretrain_seed=None, **kw)


def test_a_run_becomes_a_scorable_results_file(tmp_path):
    _, bench = make_run(tmp_path)
    r = _build(tmp_path, bench)
    h = r.header
    assert h.bench == {"name": "fixture_bench", "version": "fixture-v1", "version_sha256": bench.version_sha256,
                       "declaration_sha256": bench.sha256, "units_sha256": units_sha256([806, 1050], 3),
                       "seed_set": "test", "identity_source": "summary"}
    assert h.checkpoint == {"ref": "my-checkpoint", "pub_digest": None, "eval_digest": "c" * 64}
    assert h.training_seed == {"pretrain": None, "ft": 0}
    assert h.execution == {"fps": 15, "replan": 8, "max_steps": 1500, "n_stable_frames": 45, "off_protocol": False}
    assert h.serve.fields == {} and h.harness["driver_sha256"] == "d" * 64 and h.noise_salt == 1096304979
    assert (h.image_geometry, h.action_space, h.xla_pin) == ("raw", "droid8", "--xla_gpu_autotune_level=0")
    s = score(r, bench)
    assert s.rollouts == 6 and round(s.pct, 2) == 50.0


def test_a_tagged_server_becomes_serve_fields(tmp_path):
    _, bench = make_run(tmp_path, md=TAGGED, served="mine_w0_dualsim")
    r = build_results(tmp_path / "r1", "mine", bench, ft_seed=0, pretrain_seed=0)
    assert r.header.serve.fields == {"quality": "5"} and r.header.serve.uncond_fields is None
    assert r.header.serve.beta == 1.0 and r.header.served == "mine_w0_dualsim"


def test_a_guided_server_adds_the_unconditional_pass_at_its_beta():
    sc = serve_condition([{"server_metadata": {**TAGGED, "guidance_scale": 0.5, "beta": 1.5}}] * 2)
    assert (sc.fields, sc.uncond_fields, sc.beta) == ({"quality": "5"}, {}, 1.5)
    with pytest.raises(BuildError, match="beta is 1 \\+ w"):
        serve_condition([{"server_metadata": {**TAGGED, "guidance_scale": 0.5, "beta": 1.0}}])
    with pytest.raises(BuildError, match="no quality_tag"):
        serve_condition([{"server_metadata": {**MD, "cfg_conditioning": True}}])


@pytest.mark.parametrize("tag, match", [(0, "untagged sentinel"), ("", "not a quality bin"),
                                        (True, "not a quality bin"), (-1, "not a quality bin")])
def test_a_conditioned_server_whose_tag_is_not_a_bin_is_refused(tag, match):
    """Tag 0 is the trainer's untagged sentinel: a server conditioned on it appended a suffix no
    model was trained on, so it is neither the bare cell nor a conditioned one."""
    with pytest.raises(BuildError, match=match):
        serve_condition([{"server_metadata": {**TAGGED, "quality_tag": tag}}] * 2)
    assert serve_condition([{"server_metadata": {**TAGGED, "quality_tag": 3}}]).fields == {"quality": "3"}
    assert serve_condition([{"server_metadata": {**MD, "quality_tag": 0, "cfg_conditioning": False}}]).fields == {}


def test_a_recorded_deviation_marks_the_results_off_protocol(tmp_path):
    dev = {"replan": {"executed": 4, "declared": 8}}
    extra = {t: {"replan": 4, "execution_deviations": dev} for t in (806, 1050)}
    _, bench = make_run(tmp_path, meta_extra=extra)
    ex = _build(tmp_path, bench).header.execution
    assert ex["off_protocol"] is True and ex["replan"] == 4 and ex["deviations"] == dev


def test_a_run_allowed_to_deviate_is_off_protocol_even_without_a_deviation(tmp_path):
    record = {"execution": {"declared": None, "executed": {}, "deviations": {}, "off_protocol": True, "env": {}}}
    _, bench = make_run(tmp_path, provenance_extra=record)
    ex = _build(tmp_path, bench).header.execution
    assert ex["off_protocol"] is True and "deviations" not in ex


def test_the_run_records_deviation_reaches_the_header_and_a_contradiction_is_refused(tmp_path):
    dev = {"n_stable_frames": {"executed": 30, "declared": 45}}
    record = {"execution": {"deviations": dev, "off_protocol": True}}
    extra = {t: {"n_stable_frames": 30} for t in (806, 1050)}
    _, bench = make_run(tmp_path, provenance_extra=record, meta_extra=extra, execution_json=True)
    ex = _build(tmp_path, bench).header.execution
    assert ex["deviations"] == dev and ex["n_stable_frames"] == 30 and ex["off_protocol"] is True
    with pytest.raises(BuildError, match="disagree on what executed"):
        executed_constants([{"n_stable_frames": 20}], {"execution": {"deviations": dev}}, bench)


def test_a_constant_that_is_not_the_declared_one_is_a_deviation_though_nothing_recorded_it(tmp_path):
    _, bench = make_run(tmp_path, meta_extra={t: {"max_steps": 900} for t in (806, 1050)}, execution_json=True)
    ex = _build(tmp_path, bench).header.execution
    assert ex["deviations"] == {"max_steps": {"executed": 900, "declared": 1500}} and ex["off_protocol"] is True


def test_a_rollout_outside_the_declared_domain_is_off_protocol(tmp_path):
    _, bench = make_run(tmp_path, meta_extra={t: {"domain": "render"} for t in (806, 1050)})
    ex = _build(tmp_path, bench).header.execution
    assert ex["deviations"] == {"domain": {"executed": "render", "declared": "dualsim"}}
    assert arms.DUAL_SIM in arms.DOMAINS and execution.DRIVER_DOMAIN == arms.DUAL_SIM


@pytest.mark.parametrize("ft, pre", [(True, None), ("0", None), (0, 1.0)])
def test_training_seeds_are_ints(tmp_path, ft, pre):
    _, bench = make_run(tmp_path)
    with pytest.raises(BuildError, match="training seeds are ints"):
        build_results(tmp_path / "r1", "mine", bench, ft_seed=ft, pretrain_seed=pre)


@pytest.mark.parametrize("extra, match", [
    ({806: {"replan": 4}}, "constants differ between task directories"),
    ({806: {"execution_deviations": {"replan": {"executed": 4, "declared": 8}}},
      1050: {"execution_deviations": {}}}, "different execution deviations"),
    ({806: {"domain": "render"}}, "different domains"),
    ({1050: {"server_metadata": TAGGED}}, "served two ways"),
])
def test_tasks_that_disagree_are_refused(tmp_path, extra, match):
    _, bench = make_run(tmp_path, meta_extra=extra)
    with pytest.raises(BuildError, match=match):
        _build(tmp_path, bench)


def test_errored_rollouts_are_refused(tmp_path):
    _, bench = make_run(tmp_path, errored=2)
    with pytest.raises(BuildError, match="broken harness"):
        _build(tmp_path, bench)


def test_seed_files_that_do_not_reproduce_the_fingerprint_are_refused(tmp_path):
    _, bench = make_run(tmp_path)
    (tmp_path / "logs" / "mine_dualsim" / "task_806" / "seed_806001.json").write_text(
        json.dumps({"seed": 806001, "success": True, "n_steps": 1, "sim_time": 0.1, "reason": "success"}))
    with pytest.raises(BuildError, match="fingerprint"):
        _build(tmp_path, bench)


def test_another_benchmark_is_refused(tmp_path):
    make_run(tmp_path)
    other = load_benchmark("fixture_bench", conf_dir=make_bench(tmp_path / "other", [806, 1050], 2))
    with pytest.raises(BuildError, match="evaluated on benchmark"):
        _build(tmp_path, other)


@pytest.mark.parametrize("change", ["fewer_seeds", "other_tasks"])
def test_a_run_of_fewer_tasks_or_seeds_is_not_a_benchmark_result(tmp_path, change):
    _, bench = make_run(tmp_path)
    p = tmp_path / "r1" / "out_mine" / "arm_summary.json"
    doc = json.loads(p.read_text())
    if change == "fewer_seeds":
        doc["n_seeds"] = 2
    else:
        doc["benchmark"] = {**doc["benchmark"], "matches_manifest": False, "task_ids": [806]}
    p.write_text(json.dumps(doc))
    with pytest.raises(BuildError, match="not a benchmark result"):
        _build(tmp_path, bench)


def test_an_unreadable_run_record_is_refused_not_read_as_absent(tmp_path):
    _, bench = make_run(tmp_path, summary_extra={"provenance": {"error": "cannot read run_manifest.json"}})
    with pytest.raises(BuildError, match="run record cannot be read"):
        _build(tmp_path, bench)


def test_a_run_with_no_record_builds_with_what_its_rollouts_recorded(tmp_path):
    _, bench = make_run(tmp_path, summary_extra={"provenance": None})
    h = _build(tmp_path, bench).header
    assert h.harness is None and h.checkpoint["eval_digest"] is None and h.noise_salt is None
    assert h.execution["off_protocol"] is False and h.image_geometry == bench.image_geometry


def test_a_dev_run_says_so_and_never_builds_as_a_test_run(tmp_path):
    _, bench = make_run(tmp_path, seed_set="dev", arm="mine_dev", served="mine_dev_dualsim",
                    meta_extra={t: {"seed_set": "dev"} for t in (806, 1050)})
    r = _build(tmp_path, bench, arm="mine_dev")
    assert r.header.bench["seed_set"] == "dev"
    assert r.header.bench["units_sha256"] == units_sha256([806, 1050], 3, offset=500)
    assert {x.seed for x in r.rows} == {806500, 806501, 806502, 1050500, 1050501, 1050502}
    s = score(r, bench)     # scored on the development seeds, and says so
    assert (s.seed_set, s.rollouts, s.missing) == ("dev", 6, 0)
    as_test = dataclasses.replace(r, header=dataclasses.replace(r.header, bench={**r.header.bench, "seed_set": "test"}))
    with pytest.raises(ValueError, match="results are for benchmark"):
        score(as_test, bench)     # its units are the development ones: it never scores as a test read


@pytest.mark.parametrize("summary, record, meta, match", [
    ("test", "dev", "dev", "the summary says seed set 'test', the run record says 'dev'"),
    ("dev", "test", "dev", "the summary says seed set 'dev', the run record says 'test'"),
    ("dev", "dev", "test", "the summary says seed set 'dev', task 806's meta.json says 'test'"),
    ("test", "test", "test", "must end in|ends in '_dev'"),
])
def test_a_dev_run_whose_records_claim_test_is_refused(tmp_path, summary, record, meta, match):
    _, bench = make_run(tmp_path, seed_set="dev", arm="mine_dev", served="mine_dev_dualsim",
                    summary_extra={"seed_set": summary}, provenance_extra={"seed_set": record},
                    meta_extra={t: {"seed_set": meta} for t in (806, 1050)})
    with pytest.raises(BuildError, match=match):
        _build(tmp_path, bench, arm="mine_dev")


def test_development_seeds_under_a_test_arm_are_refused(tmp_path):
    """Records that all say test, an arm name without _dev, but the seed files are dev seeds."""
    _, bench = make_run(tmp_path, seed_set="dev", summary_extra={"seed_set": "test"})
    with pytest.raises(BuildError, match=r"task 806: 3 missing \[806000, 806001, 806002\]; 3 not test seeds"):
        _build(tmp_path, bench)


def test_a_failed_run_and_a_mismatched_summary_are_named(tmp_path):
    _, bench = make_run(tmp_path)
    out = tmp_path / "r1" / "out_mine"
    (out / "arm_summary.json").rename(out / "arm_summary.FAILED.json")
    with pytest.raises(BuildError, match="no rollouts"):
        _build(tmp_path, bench)
    (out / "arm_summary.FAILED.json").rename(out / "arm_summary.json")
    (tmp_path / "r1" / "out_mine").rename(tmp_path / "r1" / "out_other")
    with pytest.raises(BuildError, match="summary of run 'mine', not 'other'"):
        _build(tmp_path, bench, arm="other")


def test_moved_seed_files_are_found_with_log_root(tmp_path):
    _, bench = make_run(tmp_path)
    (tmp_path / "logs").rename(tmp_path / "moved")
    with pytest.raises(BuildError, match="cannot read"):
        _build(tmp_path, bench)
    assert len(_build(tmp_path, bench, log_root=tmp_path / "moved").rows) == 6


def test_the_cli_writes_a_file_the_scorer_reads(tmp_path, capsys):
    conf, _ = make_run(tmp_path)
    out = tmp_path / "mine.json"
    common = ["--benchmark", "fixture_bench", "--bench-conf-dir", str(conf)]
    assert main(["results", "build", "--run-root", str(tmp_path / "r1"), "--run", "mine", "--ft-seed", "0",
                 "--pretrain-seed", "none", "--out", str(out), *common]) == 0
    assert load_results(out.read_bytes()).header.served == "mine_dualsim"
    assert main(["results", "score", str(out), *common]) == 0
    assert main(["results", "build", "--run-root", str(tmp_path / "r1"), "--run", "absent", "--ft-seed", "0",
                 "--pretrain-seed", "none", "--out", str(out), *common]) == 2
    assert "no arm_summary.json" in capsys.readouterr().out


def test_the_cli_names_an_off_protocol_build(tmp_path, capsys):
    extra = {t: {"replan": 4, "execution_deviations": {"replan": {"executed": 4, "declared": 8}}}
             for t in (806, 1050)}
    conf, _ = make_run(tmp_path, meta_extra=extra)
    assert main(["results", "build", "--run-root", str(tmp_path / "r1"), "--run", "mine", "--ft-seed", "1",
                 "--pretrain-seed", "7", "--out", str(tmp_path / "m.json"), "--benchmark", "fixture_bench",
                 "--bench-conf-dir", str(conf)]) == 0
    assert "OFF-PROTOCOL" in capsys.readouterr().out
    r = load_results((tmp_path / "m.json").read_bytes())
    assert r.header.training_seed == {"pretrain": 7, "ft": 1} and r.header.execution["off_protocol"] is True


# --- exact seeds, the run record's benchmark, executed conventions, exit codes ----------------------

def _summary(tmp_path, arm="mine"):
    return tmp_path / "r1" / f"out_{arm}" / "arm_summary.json"


def _edit_summary(tmp_path, **changes):
    doc = json.loads(_summary(tmp_path).read_text())
    doc.update(changes)
    _summary(tmp_path).write_text(json.dumps(doc))
    return doc


def test_a_missing_seed_file_is_refused_naming_the_task_and_the_seed(tmp_path):
    """The seed set is exact: a missing seed would score as a failure (33.33 instead of 50)."""
    _, bench = make_run(tmp_path)
    (tmp_path / "logs" / "mine_dualsim" / "task_806" / "seed_806001.json").unlink()
    _edit_summary(tmp_path, eval_fingerprint=eval_fingerprint(read_served_dir(tmp_path / "logs", "mine_dualsim"),
                                                              bench.task_ids))
    with pytest.raises(BuildError, match=r"task 806: 1 missing \[806001\]"):
        _build(tmp_path, bench)


def test_a_run_with_incomplete_tasks_is_refused(tmp_path):
    _, bench = make_run(tmp_path, summary_extra={"incomplete_tasks": [[806, 2]]})
    with pytest.raises(BuildError, match=r"left tasks incomplete .*806, 2"):
        _build(tmp_path, bench)


def _record_benchmark(tmp_path, ident: dict) -> None:
    doc = json.loads(_summary(tmp_path).read_text())
    doc["provenance"]["benchmark"] = ident
    _summary(tmp_path).write_text(json.dumps(doc))


def test_the_run_records_benchmark_is_the_headers(tmp_path):
    _, bench = make_run(tmp_path)
    _record_benchmark(tmp_path, {**bench.identity(), "matches_manifest": True})
    h = _build(tmp_path, bench).header
    assert h.bench["identity_source"] == "run_record" and h.bench["declaration_sha256"] == bench.sha256


def test_a_run_record_naming_another_benchmark_is_refused_though_the_summary_names_this_one(tmp_path):
    """The summary of a --tally-only names the benchmark at tally time; the run record names what ran."""
    _, bench = make_run(tmp_path)
    _record_benchmark(tmp_path, {**bench.identity(), "sha256": "f" * 64})
    with pytest.raises(BuildError, match="declaration ffffffffffff.* by the run record, not fixture_bench"):
        _build(tmp_path, bench)


@pytest.mark.parametrize("key, value, match", [
    ("sha256", "e" * 64, r"declaration [0-9a-f]{16} \(summary\) vs " + "e" * 16 + r" \(run record\)"),
    ("n_seeds", 2, r"n_seeds 3 \(summary\) vs 2 \(run record\)"),
], ids=["declaration", "n_seeds"])
def test_a_summary_and_a_run_record_that_disagree_on_the_benchmark_are_refused(tmp_path, key, value, match):
    compatible = {"compatible_declarations": [{"sha256": "e" * 64, "reason": "an earlier declaration"}]}
    _, bench = make_run(tmp_path, extra=compatible)
    _record_benchmark(tmp_path, {**bench.identity(), key: value})
    with pytest.raises(BuildError, match=match):
        _build(tmp_path, bench)


def test_an_earlier_compatible_declaration_is_stamped_as_evaluated(tmp_path):
    """A summary that predates the package names its declaration `manifest_sha256`, and a run record
    of that era names no benchmark: the header stamps that declaration, read from the summary."""
    compatible = {"compatible_declarations": [{"sha256": "e" * 64, "reason": "an earlier declaration"}]}
    _, bench = make_run(tmp_path, extra=compatible)
    _edit_summary(tmp_path, benchmark={"name": "fixture_bench", "manifest_sha256": "e" * 64, "matches_manifest": True})
    r = _build(tmp_path, bench)
    assert {k: r.header.bench[k] for k in ("declaration_sha256", "version", "version_sha256", "identity_source")} == {
        "declaration_sha256": "e" * 64, "version": None, "version_sha256": None, "identity_source": "summary"}
    assert score(r, bench).rollouts == 6     # the declared lineage scores, with the same units


@pytest.mark.parametrize("stamp", [None, {}, {"name": "fixture_bench", "matches_manifest": True}],
                         ids=["absent", "empty", "no-digest"])
def test_a_summary_that_names_no_benchmark_is_refused(tmp_path, stamp):
    """A summary that names no benchmark has nothing on disk tying its rollouts to one, so it is
    refused by name (for a published run, its published results file is the record)."""
    _, bench = make_run(tmp_path)
    doc = json.loads(_summary(tmp_path).read_text())
    if stamp is None:
        del doc["benchmark"]
    else:
        doc["benchmark"] = stamp
    _summary(tmp_path).write_text(json.dumps(doc))
    with pytest.raises(BuildError, match="^mine: the summary names no benchmark"):
        _build(tmp_path, bench)


def test_another_version_of_the_benchmark_is_refused(tmp_path):
    _, bench = make_run(tmp_path)
    _edit_summary(tmp_path, benchmark={**bench.identity(), "version_sha256": "9" * 64, "matches_manifest": True})
    with pytest.raises(BuildError, match="version 999999999999"):
        _build(tmp_path, bench)


def test_without_a_run_record_a_rollouts_recorded_deviation_is_the_executed_convention(tmp_path):
    dev = {"image_geometry": {"executed": "profile", "declared": "raw"}}
    _, bench = make_run(tmp_path, summary_extra={"provenance": None},
                    meta_extra={t: {"execution_deviations": dev} for t in (806, 1050)})
    h = _build(tmp_path, bench).header
    assert h.image_geometry == "profile" and h.execution["deviations"] == dev and h.execution["off_protocol"]


def test_beta_is_compared_with_a_tolerance():
    g = 0.1 * 7                                   # 0.7000000000000001: 1 + g != 1.7 exactly
    assert 1.0 + g != 1.7
    sc = serve_condition([{"server_metadata": {**TAGGED, "guidance_scale": g, "beta": 1.7}}])
    assert sc.uncond_fields == {} and abs(sc.beta - 1.7) <= 1e-9


def test_an_unreadable_record_exits_3_and_a_missing_one_exits_2(tmp_path, monkeypatch, capsys):
    conf, bench = make_run(tmp_path)
    argv = ["results", "build", "--run-root", str(tmp_path / "r1"), "--run", "mine", "--ft-seed", "0",
            "--pretrain-seed", "none", "--out", str(tmp_path / "m.json"), "--benchmark", "fixture_bench",
            "--bench-conf-dir", str(conf)]
    meta = tmp_path / "logs" / "mine_dualsim" / "task_806" / "meta.json"
    meta.unlink()
    assert main(argv) == 2 and "no meta.json" in capsys.readouterr().out          # missing: not buildable
    meta.mkdir()                                                                     # there, unreadable
    assert main(argv) == 3 and "cannot read" in capsys.readouterr().out
    meta.rmdir()
    make_run(tmp_path / "again")

    def unreadable(root, served):
        try:
            raise PermissionError(13, "Permission denied", str(root))
        except PermissionError as exc:
            raise build.SeedFileError(f"cannot read {root}: {exc}") from exc

    monkeypatch.setattr(build, "read_served_dir", unreadable)
    with pytest.raises(OSError, match="Permission denied"):
        build_results(tmp_path / "again" / "r1", "mine", bench, ft_seed=0, pretrain_seed=None)
