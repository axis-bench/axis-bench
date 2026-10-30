"""The shipped benchmark declaration is complete and its three files agree."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from axis_qvla.bench import spec

ALL24 = (806, 807, 808, 809, 855, 878, 882, 888, 890, 945, 953, 966, 1050, 1223, 1231, 1426,
         1435, 1436, 1438, 1459, 1525, 1527, 1632, 1889)


def test_shipped_benchmark_loads_and_is_consistent():
    b = spec.load_benchmark()
    assert b.name == "franka_heldout24" and b.version == "axis-heldout-v1"
    assert b.task_ids == ALL24 and b.n_seeds == 50
    assert b.action_space == "droid8" and b.image_geometry == "raw" and b.dual_sim
    assert set(b.subsets["usable_12"]) < set(ALL24)
    assert spec.check(b) == []
    assert len(b.sha256) == 64 and b.identity()["n_tasks"] == 24


def test_every_task_has_fifty_seeds_a_prompt_and_calibration():
    b = spec.load_benchmark()
    for tid in ALL24:
        t = b.task(tid)
        assert len(t.seeds) == 50 and t.seeds[0] == tid * 1000
        assert t.prompt and t.open_w > t.closed_w  # closed_w may be slightly negative (calibration noise)


def test_check_reports_every_kind_of_drift(tmp_path: Path):
    """Each file can disagree with the others; the check must name each disagreement."""
    src = spec.CONF_DIR
    for f in ("franka_heldout24.json", "franka_heldout24.tasks.json", "franka_heldout24.layouts.json"):
        (tmp_path / f).write_text((src / f).read_text())
    d = json.loads((tmp_path / "franka_heldout24.json").read_text())
    d["task_ids"] = d["task_ids"] + [999]          # scored but absent everywhere
    d["subsets"]["usable_12"].append(1000)          # subset outside the benchmark
    d["excluded"]["806"] = "oops"                   # excluded AND scored
    (tmp_path / "franka_heldout24.json").write_text(json.dumps(d))
    t = json.loads((tmp_path / "franka_heldout24.tasks.json").read_text())
    t["tasks"]["807"]["test_seeds"] = t["tasks"]["807"]["test_seeds"][:20]
    t["tasks"]["808"]["gripper"]["open_w"] = 0.0    # inverted
    (tmp_path / "franka_heldout24.tasks.json").write_text(json.dumps(t))
    b = spec.load_benchmark(conf_dir=tmp_path)
    problems = "\n".join(spec.check(b))
    for needle in ("999", "1000", "806", "807: 20 seeds", "808: gripper", "layouts file lacks"):
        assert needle in problems, needle


def test_unknown_task_is_a_named_error():
    with pytest.raises(spec.BenchmarkError, match="not in benchmark"):
        spec.load_benchmark().task(811)


def test_informative_20_is_the_24_minus_the_floored_and_saturated_four():
    """The paper's informative view, declared once (was gen.py:18 and page.py:82)."""
    b = spec.load_benchmark()
    inf = b.subsets["informative_20"]
    assert len(inf) == 20 and set(inf) < set(ALL24)
    assert not set(inf) & {1426, 1435, 1436, 1459}
    assert set(ALL24) - set(inf) == {1426, 1435, 1436, 1459}


# The 24 entries: their bytes, and every field but the two descriptive `*_provenance` strings (which no
# code reads) as they stood at 473d5af, the layouts the paper's evaluations read.
TASKS_SHA256 = "5b983dd8baa446d1a551c028c6508e5f00338c20cc612e038f48392cd308fcfb"
EVAL_FIELDS_AT_473D5AF = "ea7656297060c620555eb997a731cab49364d81f3e9d3349276a5661aa842a80"
DESCRIPTIVE = ("layout_box_provenance", "layout_center_provenance")


def test_dev_seeds_are_declared_and_disjoint_from_the_test_seeds():
    """dev_seeds = tid*1000 + 500 + i."""
    b = spec.load_benchmark()
    for tid in b.task_ids:
        dev = b.dev_seeds(tid)
        assert len(dev) == 50 == len(b.task(tid).seeds)
        assert dev == tuple(tid * 1000 + 500 + i for i in range(50))
        assert not set(dev) & set(b.task(tid).seeds)


def test_every_input_is_pinned_as_repo_at_tag():
    d = json.loads((spec.CONF_DIR / "franka_heldout24.json").read_text())
    assert set(d["inputs"]) == {"demos", "task_definitions", "scene_assets", "norm_stats"}
    for key, value in d["inputs"].items():
        assert re.fullmatch(r"axisrobotics/qvla-v1-bench@v1\.0:[A-Za-z0-9_./-]+/", value), (key, value)


def test_layouts_header_describes_the_24_task_set_and_entries_are_unchanged():
    doc = json.loads((spec.CONF_DIR / "franka_heldout24.layouts.json").read_text())
    assert doc["n_tasks"] == 24 == len(doc["tasks"])
    assert "derived224" not in doc["source_dataset"]
    assert "24" in doc["selection"]
    tasks = json.dumps(doc["tasks"], sort_keys=True).encode()
    assert hashlib.sha256(tasks).hexdigest() == TASKS_SHA256
    fields = {t: {k: v for k, v in e.items() if k not in DESCRIPTIVE} for t, e in doc["tasks"].items()}
    assert hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest() == EVAL_FIELDS_AT_473D5AF


R1_DECLARATION_SHA256 = "fcff87cbf738fb15849f15adf91e06e8d3e88434c6aa0b0f8d19d6fe209eb037"


def _edited(tmp_path: Path, **changes) -> Path:
    """A copy of the shipped benchmark whose declaration has `changes` applied (None deletes)."""
    conf = tmp_path / "conf"
    conf.mkdir(parents=True)
    for suffix in (".json", ".tasks.json", ".layouts.json"):
        src = spec.CONF_DIR / f"franka_heldout24{suffix}"
        (conf / src.name).write_bytes(src.read_bytes())
    p = conf / "franka_heldout24.json"
    d = json.loads(p.read_text())
    for k, v in changes.items():
        if v is None:
            d.pop(k, None)
        else:
            d[k] = v
    p.write_text(json.dumps(d, indent=2) + "\n")
    return conf


def test_the_declaration_names_the_r1_declaration_as_compatible():
    """The paper's results are stamped with an earlier declaration, and the declaration itself
    declares that lineage."""
    b = spec.load_benchmark()
    assert R1_DECLARATION_SHA256 in b.compatible
    assert b.compatible[R1_DECLARATION_SHA256].strip()
    assert b.sha256 not in b.compatible
    assert b.accepted_declarations() == frozenset({b.sha256, *b.compatible})


def test_the_declaration_file_keeps_its_byte_format():
    raw = (spec.CONF_DIR / "franka_heldout24.json").read_bytes()
    assert json.dumps(json.loads(raw), indent=2).encode() + b"\n" == raw


@pytest.mark.parametrize("bad", [
    {"sha256": "F" * 64, "reason": "upper-case hex"},
    {"sha256": "a" * 63, "reason": "short"},
    {"sha256": "a" * 64, "reason": "  "},
    {"sha256": "a" * 64},
    "a" * 64,
])
def test_a_malformed_compatible_declaration_is_refused(tmp_path, bad):
    conf = _edited(tmp_path, compatible_declarations=[bad])
    with pytest.raises(spec.BenchmarkError, match="compatible_declarations"):
        spec.load_benchmark(conf_dir=conf)


def test_compatible_declarations_must_be_a_list_without_duplicates(tmp_path):
    e = {"sha256": "a" * 64, "reason": "older edition"}
    with pytest.raises(spec.BenchmarkError, match="compatible_declarations"):
        spec.load_benchmark(conf_dir=_edited(tmp_path / "x", compatible_declarations=e))
    with pytest.raises(spec.BenchmarkError, match="compatible_declarations"):
        spec.load_benchmark(conf_dir=_edited(tmp_path / "y", compatible_declarations=[e, e]))


def test_without_compatible_declarations_only_the_file_itself_is_accepted(tmp_path):
    b = spec.load_benchmark(conf_dir=_edited(tmp_path, compatible_declarations=None))
    assert b.compatible == {} and b.accepted_declarations() == frozenset({b.sha256})


@pytest.mark.parametrize("dev", [
    {"offset": "500", "n": 50}, {"n": 50}, {"offset": 500}, {"offset": 500, "n": 0},
    {"offset": True, "n": 50}, {"offset": -1, "n": 50},
])
def test_malformed_dev_seeds_raise_benchmark_error_not_key_error(tmp_path, dev):
    """A malformed dev_seeds dict (wrong type, a missing key, n == 0, or a negative offset)
    raises BenchmarkError, never a bare KeyError or TypeError."""
    b = spec.load_benchmark(conf_dir=_edited(tmp_path, dev_seeds=dev))
    with pytest.raises(spec.BenchmarkError, match="dev_seeds"):
        b.dev_seeds(806)
    assert any("dev_seeds" in p for p in spec.check(b))


def test_dev_seeds_that_overlap_the_test_seeds_are_refused(tmp_path):
    """offset 0 makes dev_seeds == the test seeds tid*1000+i: a dev run would tune on test."""
    b = spec.load_benchmark(conf_dir=_edited(tmp_path, dev_seeds={"offset": 0, "n": 50}))
    with pytest.raises(spec.BenchmarkError, match="overlap"):
        b.dev_seeds(806)
    assert any("overlap" in p for p in spec.check(b))


def test_the_shipped_dev_seeds_pass_check():
    b = spec.load_benchmark()
    assert spec.check(b) == []


def test_the_r1_reason_names_every_kind_of_change_including_the_harness():
    """The reason must say the `harness` field changed, naming the harness by the public command
    (the declaration names commands, not modules)."""
    b = spec.load_benchmark()
    why = b.compatible[R1_DECLARATION_SHA256]
    for word in ("harness", "`qvla bench eval`", "informative_20", "dev_seeds", "units_sha256"):
        assert word in why, word
    doc = json.loads((spec.CONF_DIR / "franka_heldout24.json").read_text())
    assert doc["harness"] == "qvla bench eval (--dual-sim)"
    assert doc["description"] == ("The 24 held-out Franka pick-and-place tasks, 50 layout seeds each, dual-sim; "
                                  "`qvla bench eval` runs them and the scorer refuses any other task set.")


def test_the_split_keeps_the_public_names():
    """The field validators moved to axis_qvla.bench.spec_checks; callers that import
    BenchmarkError (and the rest) from axis_qvla.bench.spec see the same objects."""
    from axis_qvla.bench import spec_checks

    assert spec.BenchmarkError is spec_checks.BenchmarkError
    assert {"BenchmarkError", "Benchmark", "TaskEntry", "load_benchmark", "check"} <= set(spec.__all__)
