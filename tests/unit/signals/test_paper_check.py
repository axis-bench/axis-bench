"""`paper.check` names every way a bundles tree differs from the declaration and the lock, and
`paper.build` stages a tree only once every check has passed -- on a toy release, no pack needed.

The toy release declares three seed replicates of one BC signal, which export the same locked file,
the shape of the paper's seed replicates: only the declaration tells them apart."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import types

import numpy as np
import pytest

from axis_qvla.core.canon import dumps_canonical, sha256_bytes
from axis_qvla.cli import main
from axis_qvla.signals import paper
from axis_qvla.signals.bundle import export_legacy, load_bundle, write_bundle
from axis_qvla.signals.legacy import LegacyTarget
from axis_qvla.signals.methods.bc import BC
from axis_qvla.signals.rowspace import corpus_fingerprint
from axis_qvla.signals.schema import Budget
from axis_qvla.signals.staging import StagingError
from axis_qvla.signals.testing import toy_rowspace

ROWS = "signals/toy/lerobot_rows.npz"
ARMS = {"toy.bc.seed0": ["svla.bc", "xvla.bc"], "toy.bc.seed1": ["svla.bc_sd1", "xvla.bc_sd1"],
        "toy.bc.seed2": ["svla.bc_sd2"]}
TARGET = LegacyTarget(lock_id="toy.rows", path=ROWS, kind="lerobot_rows", sha256="", format="lerobot-legacy",
                      generator="toy", generator_sha256=None, roots_form=None, numerics=None, tolerance=None,
                      template_name=None)


def _conf(edits: dict | None = None) -> str:
    """The toy declaration, with `edits` ({name: {key: value}}) setting or replacing an entry's keys."""
    out = ['schema = "qvla.paper_bundles/1"']
    for seed, (name, arms) in enumerate(ARMS.items()):
        entry = {"name": name, "method": "bc", "format": "lerobot-legacy", "rowspace": "toy", "budget": [4, 2],
                 "seed": seed, "planner": "lerobot-axisrowsampler", "exports": ["toy.rows"], "arms": arms,
                 **(edits or {}).get(name, {})}
        out += ["", "[[bundle]]"] + [f"{k} = {json.dumps(v)}" for k, v in entry.items()]
    return "\n".join(out) + "\n"


@pytest.fixture
def release(tmp_path, monkeypatch):
    """A toy corpus pack (its lock and the one locked file), the declaration, and build/check bound to them."""
    rs = toy_rowspace()
    [rows] = export_legacy(BC("lerobot-axisrowsampler", exports=(TARGET,)).compile(None, rs, Budget(4, 2), 0),
                           "lerobot-legacy", tmp_path / "export")
    pack = tmp_path / "qvla-v1-corpus"
    (pack / ROWS).parent.mkdir(parents=True)
    (pack / ROWS).write_bytes(rows.read_bytes())
    lock = pack / "SIGNALS.lock.json"
    lock.write_text(json.dumps({"files": [{"id": "toy.rows", "path": ROWS, "kind": "lerobot_rows",
                                           "sha256": hashlib.sha256(rows.read_bytes()).hexdigest(),
                                           "trained": [{"arm": a} for arms in ARMS.values() for a in arms]}]}))
    conf = tmp_path / "paper_bundles.toml"
    conf.write_text(_conf())
    monkeypatch.setattr(paper, "paper_rowspaces", lambda _pack, _corpus: {"toy": rs})
    monkeypatch.setattr(paper, "legacy_target", lambda lock_id, fmt: dataclasses.replace(TARGET, lock_id=lock_id))
    (tmp_path / "scratch").mkdir()

    def build(out=pack / "bundles", **kw):
        kw.setdefault("conf", conf)
        return paper.build(pack=pack, corpus=tmp_path, out=out, lock=lock, log=lambda _line: None, **kw)

    def check(tree, declaration=conf):
        return paper.check(tree, scratch=tmp_path / "scratch", conf=declaration, lock=lock)

    return types.SimpleNamespace(pack=pack, lock=lock, conf=conf, out=pack / "bundles", tmp=tmp_path,
                                 build=build, check=check)


def _copy(release, edit=None):
    """A copy of the built tree, outside the pack, with `edit(manifest)` applied."""
    copy = release.tmp / "copy" / "bundles"
    shutil.copytree(release.out, copy)
    if edit is not None:
        man = json.loads((copy / "manifest.json").read_text())
        edit(man)
        (copy / "manifest.json").write_text(json.dumps(man))
    return copy


def _tree_digest(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_a_built_tree_checks_clean_and_the_pack_gains_only_bundles(release):
    man = release.build()
    assert (len(man["bundles"]), len(man["arms"]), list(man["rowspaces"])) == (3, 5, ["toy"])
    assert release.check(release.out) == []
    assert sorted(p.name for p in release.pack.iterdir()) == ["SIGNALS.lock.json", "bundles", "signals"]
    assert paper.staging_dirs(release.out) == []
    before = _tree_digest(release.out)
    release.build()                                       # a rebuild over the tree: the same tree
    assert _tree_digest(release.out) == before and paper.staging_dirs(release.out) == []


def _swap_seed_replicates(man):
    b = man["bundles"]
    b["toy.bc.seed0"]["id"], b["toy.bc.seed1"]["id"] = b["toy.bc.seed1"]["id"], b["toy.bc.seed0"]["id"]


TAMPERS = {
    "swapped seed replicates": (_swap_seed_replicates, [
        "toy.bc.seed0: the seed is 1, the declaration says 0",
        "toy.bc.seed1: the seed is 0, the declaration says 1"]),
    "an arm on another seed's bundle": (lambda m: m["arms"].update({"svla.bc_sd1": "toy.bc.seed0"}), [
        "run svla.bc_sd1: the manifest maps it to toy.bc.seed0, the declaration to toy.bc.seed1"]),
    "another arm on another seed's bundle": (lambda m: m["arms"].update({"xvla.bc": "toy.bc.seed2"}), [
        "run xvla.bc: the manifest maps it to toy.bc.seed2, the declaration to toy.bc.seed0"]),
    "another declaration": (lambda m: m.update(conf_sha256="0" * 64), [
        "the manifest was built from another declaration (conf_sha256 0000000000000000, paper_bundles.toml is "]),
}


@pytest.mark.parametrize("tamper", sorted(TAMPERS))
def test_a_tampered_manifest_is_refused_naming_the_tamper(release, tamper):
    edit, want = TAMPERS[tamper]
    release.build()
    problems = release.check(_copy(release, edit))
    assert len(problems) == len(want) and all(p.startswith(w) for p, w in zip(problems, want)), problems


def test_every_tamper_at_once_is_listed(release):
    release.build()
    problems = release.check(_copy(release, lambda m: [edit(m) for edit, _ in TAMPERS.values()]))
    assert len(problems) == sum(len(w) for _, w in TAMPERS.values()), problems


@pytest.mark.parametrize("edit,want", [
    (lambda m: m["arms"].update({"svla.bc": "toy.bc.gone"}),
     ["run svla.bc: the manifest maps it to toy.bc.gone (not compiled labels of the manifest), the declaration to toy.bc.seed0"]),
    (lambda m: m["bundles"]["toy.bc.seed0"].update(rowspace="nowhere"),
     ["toy.bc.seed0: the manifest's rowspace is 'nowhere', the declaration says 'toy'"]),
    (lambda m: m["bundles"]["toy.bc.seed2"].pop("controls"),
     ["toy.bc.seed2: manifest record fields ['controls'] are missing or mistyped"]),
    (lambda m: m["bundles"]["toy.bc.seed2"].update(exports={"toy.rows": 7}),
     ["toy.bc.seed2: manifest record fields ['exports', 'controls'] are missing or mistyped"]),
    (lambda m: m["bundles"].pop("toy.bc.seed2"),
     ["toy.bc.seed2: declared, but the manifest has no such compiled labels",
      "run svla.bc_sd2: its compiled labels toy.bc.seed2 are not in the manifest",
      "bundles/"]),                                       # its directory is now named by nothing
    (lambda m: m["bundles"].update({"toy.extra": m["bundles"]["toy.bc.seed2"]}),
     ["toy.extra: in the manifest, but no such compiled labels are declared"]),
    (lambda m: m.update(arms=["svla.bc"]), ["manifest arms is not a mapping of names to strings"]),
    (lambda m: m.update(schema="qvla.signal_bundles/0"), ["manifest schema 'qvla.signal_bundles/0' is not "]),
])
def test_a_malformed_manifest_is_reported_by_name_never_raised(release, edit, want):
    release.build()
    problems = release.check(_copy(release, edit))
    assert len(problems) == len(want) and all(p.startswith(w) for p, w in zip(problems, want)), problems


NOT_AN_ID = "is not a compiled-labels id (64 lowercase hex characters)"


@pytest.mark.parametrize("edit,want", [
    # A path-shaped id is never joined onto the tree: `manifest.json/x` names a path below a file.
    (lambda m: m["bundles"]["toy.bc.seed1"].update(controls=["manifest.json/x"]),
     [f"toy.bc.seed1: its control 'manifest.json/x' {NOT_AN_ID}"]),
    (lambda m: m["bundles"]["toy.bc.seed0"].update(id="manifest.json/x"),
     [f"toy.bc.seed0: its id 'manifest.json/x' {NOT_AN_ID}", "bundles/"]),   # its directory is named by nothing
    (lambda m: m["bundles"]["toy.bc.seed2"].update(id=m["bundles"]["toy.bc.seed2"]["id"].upper()),
     ["toy.bc.seed2: its id '", "bundles/"]),
    (lambda m: m["bundles"]["toy.bc.seed2"].update(id="../" + m["bundles"]["toy.bc.seed2"]["id"]),
     ["toy.bc.seed2: its id '../", "bundles/"]),
])
def test_an_id_that_is_not_a_bundle_id_is_reported_by_name(release, edit, want):
    release.build()
    problems = release.check(_copy(release, edit))
    assert len(problems) == len(want) and all(p.startswith(w) for p, w in zip(problems, want)), problems
    assert NOT_AN_ID in problems[0], problems


@pytest.mark.parametrize("bad", ["manifest.json/x", "../rowspaces", "/", "A" * 64])
def test_a_row_space_id_that_is_not_a_content_address_is_reported_by_name(release, bad):
    """Never joined onto the tree: each bundle then records another row space than the manifest's, and
    the real row-space directory is named by nothing."""
    man = release.build()
    real = man["rowspaces"]["toy"]
    problems = release.check(_copy(release, lambda m: m["rowspaces"].update(toy=bad)))
    assert problems == [
        f"frame index toy: its id {bad!r} is not a frame-index id (64 lowercase hex characters)",
        *(f"toy.bc.seed{i}: the frame index is {real!r}, the declaration says {bad!r}" for i in range(3)),
        f"bundles/rowspaces/{real}: not named by the manifest"], problems


def test_a_missing_row_space_is_reported_for_the_declaration_and_each_bundle(release):
    release.build()
    problems = release.check(_copy(release, lambda m: m["rowspaces"].pop("toy")))
    assert problems[0] == "frame index toy: declared, but the manifest has no such frame index"
    assert [p.split(":")[0] for p in problems[1:]] == ["toy.bc.seed0", "toy.bc.seed1", "toy.bc.seed2",
                                                       "bundles/rowspaces/" + json.loads(
                                                           (release.out / "manifest.json").read_text())["rowspaces"]["toy"]]


def test_a_digest_mismatch_names_the_lock_id_and_both_digests(release):
    man = release.build()
    real = man["bundles"]["toy.bc.seed1"]["exports"]["toy.rows"]
    problems = release.check(_copy(release, lambda m: m["bundles"]["toy.bc.seed1"]["exports"].update(
        {"toy.rows": "f" * 64, "toy.other": "e" * 64})))
    assert problems == [
        "toy.bc.seed1: toy.other re-exports as nothing, the manifest records eeeeeeeeeeeeeeee",
        f"toy.bc.seed1: toy.rows re-exports as {real[:16]}, the manifest records ffffffffffffffff",
        "the manifest records exports of files SIGNALS.lock.json does not have: ['toy.other']"], problems


def test_controls_must_be_the_bundles_own(release):
    man = release.build()
    seed0 = man["bundles"]["toy.bc.seed0"]["id"]
    problems = release.check(_copy(release, lambda m: m["bundles"]["toy.bc.seed1"].update(controls=[seed0])))
    assert problems == [f"toy.bc.seed1: the manifest records controls ['{seed0[:16]}'], the compiled labels name []"]


def test_entries_the_manifest_does_not_name_are_reported(release):
    release.build()
    copy = _copy(release)
    (copy / "tmpk3x9").mkdir()
    (copy / "rowspaces" / "stray").write_text("")
    (copy.parent / ".bundles.staging-0123456789ab").mkdir()
    assert release.check(copy) == ["bundles/tmpk3x9: not named by the manifest",
                                   "bundles/rowspaces/stray: not named by the manifest",
                                   ".bundles.staging-0123456789ab: the staging directory of a build that did not finish"]


def test_a_bundle_whose_bytes_changed_is_reported(release):
    man = release.build()
    copy = _copy(release)
    plan = copy / man["bundles"]["toy.bc.seed0"]["id"] / "plan.npy"
    data = bytearray(plan.read_bytes())
    data[-1] ^= 1
    plan.write_bytes(bytes(data))
    [problem] = release.check(copy)
    assert problem.startswith("toy.bc.seed0: ") and "does not match its manifest entry" in problem


def test_the_command_reports_problems_with_exit_1_and_cannot_read_with_exit_2(release, capsys):
    release.build()
    copy = _copy(release, lambda m: m["arms"].update({"svla.bc": "toy.bc.gone"}))
    assert main(["labels", "check", str(copy)]) == 1        # the repository's declaration: many problems
    assert "DIFF  " in capsys.readouterr().out
    (copy / "manifest.json").unlink()
    assert main(["labels", "check", str(copy)]) == 2
    assert capsys.readouterr().out.startswith("FAIL  ")


def _staged_nothing(release):
    assert not release.out.exists() and paper.staging_dirs(release.out) == []


def test_build_refuses_a_pack_with_another_lock(release, tmp_path):
    other = tmp_path / "other.lock.json"
    other.write_text(json.dumps({"files": []}))
    with pytest.raises(paper.PaperBundleError, match="is not this repository's SIGNALS.lock.json"):
        paper.build(pack=release.pack, corpus=tmp_path, out=release.out, conf=release.conf, lock=other)
    _staged_nothing(release)


PUBLISHED_CORPUS = r"--episodes must be the published pretraining episodes \(qvla-v1-corpus/episodes\)"


def _lock_records_the_toy_rowspace(release, **edits):
    """The toy lock with a `rowspaces` record of the toy row space, as SIGNALS.lock.json records the
    published corpus's: its fingerprint, row count and trainable-row count."""
    rs = toy_rowspace()
    doc = json.loads(release.lock.read_text())
    doc["rowspaces"] = {"toy": {"corpus_fingerprint": rs.corpus_fingerprint, "n_rows": rs.n_rows,
                                "n_trainable": rs.n_trainable, **edits}}
    release.lock.write_text(json.dumps(doc))
    return rs


def test_build_accepts_the_corpus_the_lock_records(release):
    _lock_records_the_toy_rowspace(release)
    assert len(release.build()["bundles"]) == 3 and release.check(release.out) == []


def test_build_refuses_a_corpus_that_is_not_the_published_one(release, monkeypatch):
    """A corpus whose last episode is 4 frames longer: another fingerprint and row count, so the
    build is refused before anything is compiled, naming the corpus it must be."""
    rs = _lock_records_the_toy_rowspace(release)
    a = rs.arrays
    other = dataclasses.replace(rs, n_rows=rs.n_rows + 4, corpus_fingerprint=corpus_fingerprint(
        a["episode_task"], a["episode_index"], a["episode_length"] + np.array([0, 0, 0, 4]), a["episode_attempt"]),
        arrays={**a, "episode_length": a["episode_length"] + np.array([0, 0, 0, 4])})
    monkeypatch.setattr(paper, "paper_rowspaces", lambda _pack, _corpus: {"toy": other})
    with pytest.raises(paper.PaperBundleError, match=PUBLISHED_CORPUS) as exc:
        release.build()
    assert f"corpus_fingerprint {other.corpus_fingerprint!r} (the checksum manifest's is {rs.corpus_fingerprint!r})" in str(exc.value)
    assert f"n_rows {rs.n_rows + 4} (the checksum manifest's is {rs.n_rows})" in str(exc.value)
    assert "n_trainable" not in str(exc.value)                 # the same spans: only what differs is named
    _staged_nothing(release)


def test_build_refuses_a_corpus_with_another_trainable_count(release):
    _lock_records_the_toy_rowspace(release, n_trainable=10)
    with pytest.raises(paper.PaperBundleError, match=r"frame index toy has n_trainable 9 \(the checksum manifest's is 10\); "
                                                     + PUBLISHED_CORPUS):
        release.build()
    _staged_nothing(release)


def test_check_corpus_skips_a_flat_row_space(release):
    """heldout24_ft (axis_qvla.signals.rowspace.flat_rowspace) is not cut from --episodes at all, so a
    mismatch in its recorded fields must not be read as --episodes being wrong: check_corpus skips
    every row space whose kind is not "corpus", named or not."""
    from axis_qvla.signals.rowspace import flat_rowspace

    flat = flat_rowspace(name="heldout24_ft", n_rows=86_912, source="the held-out finetune set")
    lock_doc = {"rowspaces": {"heldout24_ft": {"n_rows": 1, "corpus_fingerprint": "sha256:not-the-flat-ones-none"}}}
    paper.check_corpus({"heldout24_ft": flat}, lock_doc, corpus="unused")      # does not raise


def test_build_refuses_when_a_row_space_the_lock_records_is_not_cut(release):
    doc = json.loads(release.lock.read_text())
    doc["rowspaces"] = {"libero5k_v2_d8": {"n_rows": 1}}
    release.lock.write_text(json.dumps(doc))
    with pytest.raises(paper.PaperBundleError, match="records frame index libero5k_v2_d8, which the build does not cut"):
        release.build()
    _staged_nothing(release)


def test_a_genuinely_missing_declared_field_is_reported_as_one(release):
    """A declared entry whose method needs a field REQUIRED does not cover (here, awr_episode's
    `reducer`) is read inside `program_for`, before `compile_program` runs: build() reports it as
    DeclaredFieldMissing, the same KeyError/TypeError boundary cli.py narrows to."""
    release.conf.write_text(_conf(edits={"toy.bc.seed0": {"method": "awr_episode"}}))
    with pytest.raises(paper.DeclaredFieldMissing, match="reducer"):
        release.build()
    _staged_nothing(release)


def test_a_bug_inside_compile_program_is_not_reported_as_a_missing_field(release, monkeypatch):
    """A KeyError from the method's own run() -- real data, not a declared field, a row space or a
    lock record -- is a bug: build() does not catch it (the try around program_for ends before
    compile_program runs), so it propagates as the bare KeyError it was, not DeclaredFieldMissing,
    and cli.py (which only catches DeclaredFieldMissing/RefusedDtype) would let it keep its
    traceback too."""
    from axis_qvla.signals.methods.bc import BC

    def buggy_run(self, labels, rows, budget, seed):
        raise KeyError("observation.not_a_declared_field")

    monkeypatch.setattr(BC, "run", buggy_run)
    with pytest.raises(KeyError, match="observation.not_a_declared_field") as exc:
        release.build()
    assert not isinstance(exc.value, paper.DeclaredFieldMissing)
    _staged_nothing(release)


def test_an_awr_chunk_bundle_records_the_pinned_corpus_and_names_digests(tmp_path):
    """The awr_chunk bundles read the corpus, not a locked file: their inputs must be the corpus
    fingerprint (and, for the approach-aware scorer, the names-table digest) generators.json pins."""
    from axis_qvla.signals import paper_check
    from axis_qvla.signals.bundle import rowspace_id

    fp, names = "sha256:" + "a" * 32, "b" * 64
    generators = tmp_path / "generators.json"
    generators.write_text(json.dumps({"inputs": {"corpus": {"corpus_fingerprint": fp},
                                                 "episode_names": {"reward_input_sha256": names}}}))
    rs = toy_rowspace()
    man = {"rowspaces": {"toy": rowspace_id(rs)}}
    record = {"method": "awr_chunk", "format": "openpi-legacy", "rowspace": "toy", "paper_arm": True}

    def declared(scorer, inputs):
        entry = {"method": "awr_chunk", "format": "openpi-legacy", "rowspace": "toy", "budget": [4, 2], "seed": 0,
                 "planner": "openpi-rowsampler", "scorer": scorer, "exports": []}
        bundle = types.SimpleNamespace(program={"method": "awr_chunk", "params": {"planner": "openpi-rowsampler",
                                                                                 "scorer": scorer}},
                                       seed=0, budget=Budget(4, 2), rowspace=rs, inputs=inputs, legacy=())
        return paper_check._declared("x", entry, record, bundle, man, {}, generators)

    assert declared("phase", {"corpus_fingerprint": fp, "episode_body_names": names}) == []
    assert declared("v2", {"corpus_fingerprint": fp}) == []
    assert declared("phase", {"corpus_fingerprint": "sha256:" + "c" * 32}) == [
        "x: its inputs are {'corpus_fingerprint': 'sha256:ccccccccc'}, generators.json pins "
        "{'corpus_fingerprint': 'sha256:aaaaaaaaa', 'episode_body_names': 'bbbbbbbbbbbbbbbb'}"]
    assert declared("v2", {"corpus_fingerprint": fp, "episode_body_names": names}) == [
        "x: its inputs are {'corpus_fingerprint': 'sha256:aaaaaaaaa', 'episode_body_names': 'bbbbbbbbbbbbbbbb'}, "
        "generators.json pins {'corpus_fingerprint': 'sha256:aaaaaaaaa'}"]


def test_build_writes_only_the_bundles_subtree_and_no_partial_build_in_the_pack(release):
    with pytest.raises(StagingError, match="staging writes only"):
        release.build(out=release.pack / "signals" / "bundles")
    with pytest.raises(StagingError, match="local build"):
        release.build(only={"toy.bc.seed0"})
    _staged_nothing(release)


def test_a_partial_build_outside_the_pack_writes_its_bundles_and_no_manifest(release):
    out = release.tmp / "local" / "bundles"
    man = release.build(out=out, only={"toy.bc.seed1"})
    assert list(man["bundles"]) == ["toy.bc.seed1"] and not (out / "manifest.json").exists()
    assert sorted(p.name for p in out.iterdir()) == sorted([man["bundles"]["toy.bc.seed1"]["id"], "rowspaces"])
    with pytest.raises(paper.PaperBundleError, match="no compiled labels are declared as toy.bc.seed9"):
        release.build(out=out, only={"toy.bc.seed9"})


def test_build_refuses_an_export_that_is_not_the_locked_file(release):
    doc = json.loads(release.lock.read_text())
    doc["files"][0]["sha256"] = "0" * 64
    release.lock.write_text(json.dumps(doc))
    with pytest.raises(paper.PaperBundleError,
                       match=r"toy.rows: the export \(\w+\) is not the file SIGNALS.lock.json lists \(0+\)"):
        release.build()
    _staged_nothing(release)


def test_build_refuses_a_bundle_without_its_declared_control(release):
    release.conf.write_text(_conf({"toy.bc.seed1": {"control": "toy.bc.seed0"}}))
    with pytest.raises(paper.PaperBundleError, match="toy.bc.seed1: its random control is not the declared toy.bc.seed0"):
        release.build()
    _staged_nothing(release)


def test_a_conf_entry_missing_a_field_is_refused_by_name(tmp_path):
    conf = tmp_path / "c.toml"
    conf.write_text('schema = "qvla.paper_bundles/1"\n\n[[bundle]]\nname = "x"\nmethod = "bc"\n')
    with pytest.raises(paper.PaperBundleError, match=r"\(x\) has no format, rowspace, budget, seed, planner, exports, arms"):
        paper.load_conf(conf)


def test_verify_exports_refuses_a_file_the_lock_does_not_have(tmp_path):
    b = BC("lerobot-axisrowsampler", exports=(TARGET,)).compile(None, toy_rowspace(), Budget(4, 2), 0)
    with pytest.raises(paper.PaperBundleError, match="its id 'toy.rows' is not a file of SIGNALS.lock.json"):
        paper.verify_exports(b, "lerobot-legacy", {}, tmp_path)


# Each edit of toy.bc.seed1's declaration, and the one line it must produce, after the conf_sha256 line.
EDITS = {
    "planner": ({"planner": "openpi-rowsampler"},
                "the planner is 'lerobot-axisrowsampler', the declaration says 'openpi-rowsampler'"),
    "granularity": ({"granularity": "row"}, "the granularity is None, the declaration says 'row'"),
    "scorer": ({"scorer": "phase"}, "the scorer is None, the declaration says 'phase'"),
    "source": ({"source": "toy.rows"}, "it read toy.rows as nothing, the checksum manifest's is "),
    "reducer": ({"reducer": "global_ratio"}, "the loss reducer is 'mean', the declaration says 'global_ratio'"),
    "tag row space": ({"tag_rowspace": "toy"}, "the tag frame index is None, the declaration says '"),
    "control": ({"control": "toy.bc.seed0"}, "its controls are [], the declared control toy.bc.seed0 is "),
    "paper_arm": ({"paper_arm": False}, "the manifest's paper_arm is True, the declaration says False"),
    "format": ({"format": "openpi-legacy"},
               "the manifest's format is 'lerobot-legacy', the declaration says 'openpi-legacy'"),
    "method": ({"method": "drop_top50"}, "the manifest's method is 'bc', the declaration says 'drop_top50'"),
}


def _against(release, edits):
    declaration = release.tmp / "edited.toml"
    declaration.write_text(_conf({"toy.bc.seed1": edits}))
    return release.check(release.out, declaration)


@pytest.mark.parametrize("field", sorted(EDITS))
def test_each_declared_field_is_compared_once(release, field):
    """The BC tree checked against a declaration with one field of one entry edited: the declaration's
    digest line, then exactly one line naming that entry and field."""
    release.build()
    edit, want = EDITS[field]
    problems = _against(release, edit)
    assert len(problems) == 2, problems
    assert problems[0].startswith("the manifest was built from another declaration (conf_sha256 ")
    assert problems[1].startswith(f"toy.bc.seed1: {want}"), problems[1]


def test_every_declared_field_at_once_gives_one_line_each(release):
    release.build()
    problems = _against(release, {k: v for edit, _ in EDITS.values() for k, v in edit.items()})
    assert len(problems) == 1 + len(EDITS), problems
    for _edit, want in EDITS.values():
        assert sum(p.startswith(f"toy.bc.seed1: {want}") for p in problems) == 1, (want, problems)


def test_a_swapped_program_is_named_even_where_the_record_agrees_with_the_declaration(release):
    """The record says bc and so does the declaration, but the id points at a bundle compiled from
    another method: the program's own method gives it away."""
    man = release.build()
    copy = _copy(release)
    seed0 = load_bundle(copy / man["bundles"]["toy.bc.seed0"]["id"])
    program = {**seed0.program, "method": "cfg_tags"}
    other = dataclasses.replace(seed0, program=program, program_sha=sha256_bytes(dumps_canonical(program)))
    new_id = write_bundle(other, copy)
    problems = release.check(_copy_edit(copy, lambda m: m["bundles"]["toy.bc.seed0"].update(id=new_id)))
    assert problems == ["toy.bc.seed0: the program's method is 'cfg_tags', and the manifest's 'bc' compiles to 'bc'",
                        f"bundles/{man['bundles']['toy.bc.seed0']['id']}: not named by the manifest"]


def _copy_edit(copy, edit):
    man = json.loads((copy / "manifest.json").read_text())
    edit(man)
    (copy / "manifest.json").write_text(json.dumps(man))
    return copy


ROOT = pytest.mark.skipif(os.geteuid() == 0, reason="root reads a directory whatever its mode, so it cannot be "
                                                   "made unreadable here")


@ROOT
def test_an_unreadable_parent_is_never_read_as_no_staging_directory(release):
    """A parent holding the tree and a killed build's staging directory, readable only by traversal
    (mode 0311): listing it fails, and that failure is raised and reported, never read as "none"."""
    release.build()
    copy = _copy(release)
    (copy.parent / ".bundles.staging-0123456789ab").mkdir()
    os.chmod(copy.parent, 0o311)
    try:
        with pytest.raises(PermissionError):
            paper.staging_dirs(copy)
        assert release.check(copy) == [
            f"{copy.parent}: cannot be listed (Permission denied), so a staging directory a killed build left "
            f"beside bundles/ cannot be ruled out"]
    finally:
        os.chmod(copy.parent, 0o755)
    assert paper.staging_dirs(copy) == [copy.parent / ".bundles.staging-0123456789ab"]


@ROOT
def test_an_untraversable_tree_is_an_io_error_never_an_absent_entry(release, tmp_path):
    release.build()
    copy = _copy(release)
    os.chmod(copy.parent, 0o000)
    try:
        with pytest.raises(PermissionError):
            paper.kind_of(copy / "rowspaces")
        with pytest.raises(PermissionError):
            paper.staging_dirs(copy)
        assert main(["labels", "check", str(copy)]) == 2
    finally:
        os.chmod(copy.parent, 0o755)
    assert paper.kind_of(copy / "rowspaces") == "dir" and paper.kind_of(tmp_path / "nothing") is None


def test_a_missing_parent_holds_no_staging_directory(tmp_path):
    assert paper.staging_dirs(tmp_path / "absent" / "bundles") == []
