"""A run spec names its inputs by name; `qvla train` resolves each to the content id binding checks."""
from __future__ import annotations

import dataclasses
import json
import pathlib
import re

import numpy as np
import pytest

from axis_qvla.core.canon import dumps_canonical, sha256_bytes
from axis_qvla.signals.bundle import write_bundle
from axis_qvla.train import ids
from axis_qvla.train.armspec import load_armspec, loads_armspec
from axis_qvla.train.names import (PUBLISHED_LABELS, PUBLISHED_NORMS, PUBLISHED_ROWSPACES, NameResolutionError,
                              published_labels_name, published_name, record_labels_name, resolve_labels, resolve_names, resolve_norm, resolve_rowspace)
from axis_qvla.train.paper import paper_arm_files
from tests.unit.train._arms import host
from tests.unit.train._bundles import make_bundle

EXAMPLES = pathlib.Path(__file__).resolve().parents[3] / "experiments" / "examples"


def _labels(root, method: str, steps: int = 2) -> str:
    """Compiled labels whose program is `method`, written under `root`; their 64-hex id."""
    b = make_bundle(plan=np.zeros((steps, 2), dtype=np.int64))
    prog = {**b.program, "method": method}
    root.mkdir(parents=True, exist_ok=True)
    return write_bundle(dataclasses.replace(b, program=prog, program_sha=sha256_bytes(dumps_canonical(prog))), root)


def test_the_published_names_are_the_ids_every_paper_run_spec_names():
    named = {"rowspace": {}, "norm": {}}
    for p in paper_arm_files():
        s = load_armspec(p)
        for kind, ref in (("rowspace", s.rowspace), ("norm", s.norm)):
            if ref and "#sha256:" in ref:
                name, _, digest = ref.partition("#sha256:")
                named[kind].setdefault(name, set()).add(digest)
    assert {k: {v} for k, v in PUBLISHED_ROWSPACES.items()} == {k: named["rowspace"][k] for k in PUBLISHED_ROWSPACES}
    assert {k: {v} for k, v in PUBLISHED_NORMS.items()} == {k: named["norm"][k] for k in PUBLISHED_NORMS}


def test_the_published_labels_are_the_ones_every_seed_0_lerobot_finetune_names(tmp_path):
    seed0 = {load_armspec(p).bundle for p in paper_arm_files()
             if (s := load_armspec(p)).stage == "finetune" and s.backbone != "pi05-openpi" and s.seed == 0}
    assert {f"sha256:{d}" for d in PUBLISHED_LABELS.values()} == seed0
    _, paths = host(tmp_path)
    assert resolve_labels(paths, "name:lerobot.ft_rows_all") == f"sha256:{PUBLISHED_LABELS['lerobot.ft_rows_all']}"
    assert published_labels_name(f"sha256:{PUBLISHED_LABELS['lerobot.ft_rows_all']}") == "name:lerobot.ft_rows_all"
    assert published_labels_name(f"sha256:{'0' * 64}") == f"sha256:{'0' * 64}"


def test_published_names_resolve_and_content_ids_pass_through():
    assert resolve_rowspace("heldout24_ft") == f"heldout24_ft#sha256:{PUBLISHED_ROWSPACES['heldout24_ft']}"
    assert resolve_norm("libero5k_v2_d8") == f"libero5k_v2_d8#sha256:{PUBLISHED_NORMS['libero5k_v2_d8']}"
    content = "mine#sha256:" + "a" * 64
    assert resolve_rowspace(content) == content and published_name(content, PUBLISHED_ROWSPACES) == content
    assert published_name(resolve_rowspace("heldout24_ft"), PUBLISHED_ROWSPACES) == "heldout24_ft"
    with pytest.raises(NameResolutionError, match="not a published one"):
        resolve_rowspace("libero5k_v3")


def test_labels_resolve_by_their_method_name(tmp_path):
    _, paths = host(tmp_path)
    bid = _labels(ids.local_bundles_root(paths), "example.keep_top")
    assert resolve_labels(paths, "name:example.keep_top") == f"sha256:{bid}"
    with pytest.raises(NameResolutionError, match="no compiled labels named 'other'"):
        resolve_labels(paths, "name:other")


def test_two_compiles_of_one_method_resolve_to_the_one_recorded(tmp_path):
    _, paths = host(tmp_path)
    root = ids.local_bundles_root(paths)
    _labels(root, "example.keep_top", steps=2)
    second = _labels(root, "example.keep_top", steps=3)
    with pytest.raises(NameResolutionError, match="none is recorded as the current one"):
        resolve_labels(paths, "name:example.keep_top")
    record_labels_name(root, "example.keep_top", second)
    assert resolve_labels(paths, "name:example.keep_top") == f"sha256:{second}"


def test_a_name_recorded_for_labels_that_are_gone_is_refused(tmp_path):
    _, paths = host(tmp_path)
    record_labels_name(ids.local_bundles_root(paths), "example.keep_top", "b" * 64)
    with pytest.raises(NameResolutionError, match="compile example.keep_top again"):
        resolve_labels(paths, "name:example.keep_top")


@pytest.mark.parametrize("ref", ["name:../x", "name:a/b", "name:", "name:.hidden", "name:Upper"])
def test_a_label_name_that_is_not_a_name_is_refused_before_any_file_is_read(tmp_path, ref):
    _, paths = host(tmp_path)
    outside = ids.local_bundles_root(paths) / "x.json"            # what `name:../x` would open
    outside.parent.mkdir(parents=True)
    outside.write_text(json.dumps({"id": "sha256:" + "b" * 64}))
    with pytest.raises(NameResolutionError, match="is not a compiled-labels name"):
        resolve_labels(paths, ref)


@pytest.mark.parametrize("name", sorted(PUBLISHED_LABELS))
def test_recording_a_released_labels_name_is_refused(tmp_path, name):
    """A run spec naming it trains the release's labels (PUBLISHED_LABELS wins), never the record."""
    root = tmp_path / "bundles"
    with pytest.raises(NameResolutionError, match="names the release's compiled labels"):
        record_labels_name(root, name, "b" * 64)
    assert not root.exists()


@pytest.mark.parametrize("bid", ["b" * 63, "B" * 64, "sha256:" + "b" * 64, "../" + "b" * 61])
def test_recording_a_name_refuses_an_id_that_is_not_64_hex(tmp_path, bid):
    root = tmp_path / "bundles"
    with pytest.raises(NameResolutionError, match="is not a compiled-labels id"):
        record_labels_name(root, "example.keep_top", bid)
    assert not root.exists()


def test_a_spec_by_name_resolves_to_content_and_records_each_name(tmp_path):
    _, paths = host(tmp_path)
    bid = _labels(ids.local_bundles_root(paths), "example.keep_top")
    spec = load_armspec(EXAMPLES / "runs" / "pi05.pretrain.keep_top.toml")
    resolved, names = resolve_names(spec, paths)
    assert resolved.rowspace == resolve_rowspace("libero5k_v2_d8") and resolved.bundle == f"sha256:{bid}"
    assert resolved.norm == resolve_norm("libero5k_v2_d8")
    assert names["bundle"] == {"name": "name:example.keep_top", "id": f"sha256:{bid}"}
    assert resolve_names(resolved, paths) == (resolved, {})


def test_the_example_run_specs_name_no_digest():
    for path in sorted((EXAMPLES / "runs").glob("*.toml")):
        text = path.read_text()
        assert "sha256" not in text and "Devon018" not in text, path.name
        assert loads_armspec(text).bundle == "name:example.keep_top"


def test_keep_top_records_its_name_and_prints_no_id(tmp_path, capsys):
    from experiments.examples import keep_top
    from tests.unit.signals.toy_corpus import WEIGHTS

    scores = tmp_path / "s.json"
    scores.write_text(json.dumps({"version": 1, "reward_id": "toy", "tau": 10.0, "cap": 2.0, "horizon": 10,
                                  "n_episodes": len(WEIGHTS), "n_rows": sum(map(len, WEIGHTS.values())),
                                  "weights": WEIGHTS}))
    out = tmp_path / "bundles"
    assert keep_top.main(["--scores", str(scores), "--toy", "--keep", "0.3", "--steps", "6", "--batch", "2",
                          "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert printed.splitlines()[0] == "OK    example.keep_top  plan (6, 2)" and not re.search("[0-9a-f]{64}", printed)
    assert (out / "names" / "example.keep_top.json").is_file()
    assert (out / "names" / "example.keep_top.control.json").is_file()


def test_keep_top_name_records_a_per_seed_compile_without_touching_the_default_names(tmp_path, capsys):
    """`--seed 43 --name example.keep_top.s43` (the pretrain page's per-seed compile, its CI twin with
    --toy): the seed-43 labels and their control are recorded under the given name, and a seed-42
    compile's example.keep_top / example.keep_top.control records are left as they were."""
    from experiments.examples import keep_top
    from tests.unit.signals.toy_corpus import WEIGHTS

    scores = tmp_path / "s.json"
    scores.write_text(json.dumps({"version": 1, "reward_id": "toy", "tau": 10.0, "cap": 2.0, "horizon": 10,
                                  "n_episodes": len(WEIGHTS), "n_rows": sum(map(len, WEIGHTS.values())),
                                  "weights": WEIGHTS}))
    out = tmp_path / "bundles"
    toy = ["--scores", str(scores), "--toy", "--keep", "0.3", "--steps", "6", "--batch", "2", "--out", str(out)]
    assert keep_top.main(toy) == 0
    names = out / "names"
    before = {p.name: p.read_bytes() for p in names.iterdir()}
    assert keep_top.main([*toy, "--seed", "43", "--name", "example.keep_top.s43"]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed[-2:] == ["OK    example.keep_top.s43  plan (6, 2)", "OK    example.keep_top.s43.control  plan (6, 2)"]
    assert {p.name: p.read_bytes() for p in names.iterdir() if p.name in before} == before
    s43 = json.loads((names / "example.keep_top.s43.json").read_text())["id"]
    assert s43 != json.loads(before["example.keep_top.json"])["id"]
    assert (names / "example.keep_top.s43.control.json").is_file()


@pytest.mark.parametrize("bad", ["Example", "lerobot.ft_rows_all"])
def test_keep_top_refuses_a_name_no_run_spec_could_name_before_compiling(tmp_path, capsys, bad):
    from experiments.examples import keep_top

    assert keep_top.main(["--scores", str(tmp_path / "never-read.json"), "--toy", "--name", bad,
                          "--out", str(tmp_path)]) == 2
    assert capsys.readouterr().out.startswith("FAIL  --name: ")
