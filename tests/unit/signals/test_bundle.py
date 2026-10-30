"""Bundles are content-addressed: the id covers the manifest and every array, and loading verifies it."""
from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import shutil

import numpy as np
import pytest

from axis_qvla.core.canon import dumps_canonical
from axis_qvla.signals import bundle as sb
from axis_qvla.signals import legacy
from axis_qvla.signals.testing import TOY_TRAINABLE, toy_bundle, toy_rowspace


def _forge_array(tmp_path, bundle, key: str, bad: np.ndarray):
    """Write `bundle`, then overwrite the on-disk array named `key` with `bad`, patching the
    manifest's own descriptor and renaming the directory to the new (self-consistent) id. This
    simulates a corrupted-on-disk bundle that still passes _read_record's own consistency checks,
    so it is load_bundle's dtype/shape enforcement that has to catch it."""
    bid = sb.write_bundle(bundle, tmp_path)
    d = tmp_path / bid
    np.save(d / f"{key}.npy", bad, allow_pickle=False)
    man = json.loads((d / "manifest.json").read_bytes())
    man["arrays"][key] = {"dtype": bad.dtype.str, "shape": list(bad.shape), "sha256": sb.array_digest(bad)}
    raw = dumps_canonical(man)
    new_dir = tmp_path / hashlib.sha256(raw).hexdigest()
    (d / "manifest.json").write_bytes(raw)
    d.rename(new_dir)
    return new_dir


def test_write_then_load_returns_every_field(tmp_path):
    b = toy_bundle(weights=True, cond=True)
    bid = sb.write_bundle(b, tmp_path)
    got = sb.load_bundle(tmp_path / bid)
    np.testing.assert_array_equal(got.plan, b.plan)
    np.testing.assert_array_equal(got.weights, b.weights)
    np.testing.assert_array_equal(got.cond, b.cond)
    assert got.program == b.program and got.program_sha == b.program_sha
    assert (got.budget, got.seed, got.reducer) == (b.budget, b.seed, b.reducer)
    assert got.vocab == b.vocab and got.serve == b.serve and got.requires == b.requires
    assert got.diagnostics == b.diagnostics and got.inputs == b.inputs and got.legacy == b.legacy
    assert sb.rowspace_id(got.rowspace) == sb.rowspace_id(b.rowspace)
    assert sb.bundle_id(got) == bid


def test_the_id_is_the_sha256_of_the_manifest_and_does_not_depend_on_where_it_is_written(tmp_path):
    a = sb.write_bundle(toy_bundle(), tmp_path / "a")
    b = sb.write_bundle(toy_bundle(), tmp_path / "b")
    assert a == b == sb.bundle_id(toy_bundle())
    assert hashlib.sha256((tmp_path / "a" / a / "manifest.json").read_bytes()).hexdigest() == a
    assert sb.write_bundle(toy_bundle(), tmp_path / "a") == a          # idempotent


def test_any_array_change_is_a_different_bundle():
    b = toy_bundle(weights=True)
    w = b.weights.copy()
    w[TOY_TRAINABLE[0]] = 1.75
    assert sb.bundle_id(dataclasses.replace(b, weights=w)) != sb.bundle_id(b)


def test_a_tampered_array_manifest_or_directory_is_refused(tmp_path):
    bid = sb.write_bundle(toy_bundle(weights=True), tmp_path)
    d = tmp_path / bid
    w = np.load(d / "weights.npy")
    w[1] = 7.0
    np.save(d / "weights.npy", w)
    with pytest.raises(sb.BundleError, match="weights"):
        sb.load_bundle(d)
    bid = sb.write_bundle(toy_bundle(), tmp_path / "m")
    m = tmp_path / "m" / bid / "manifest.json"
    m.write_bytes(m.read_bytes().replace(b'"seed": 0', b'"seed": 1'))
    with pytest.raises(sb.BundleError, match="sha256 of its manifest"):
        sb.load_bundle(m.parent)
    bid = sb.write_bundle(toy_bundle(), tmp_path / "s")
    (tmp_path / "s" / bid / "extra.npy").write_bytes(b"x")
    with pytest.raises(sb.BundleError, match="does not name"):
        sb.load_bundle(tmp_path / "s" / bid)


def test_a_rowspace_is_stored_and_verified_on_its_own(tmp_path):
    rid = sb.write_rowspace(toy_rowspace(), tmp_path)
    assert rid == sb.rowspace_id(toy_rowspace())
    rs = sb.load_rowspace(tmp_path / rid)
    assert (rs.n_rows, rs.n_trainable, rs.prompts) == (23, 9, ("open the drawer", "pick up the bowl"))
    np.testing.assert_array_equal(rs.arrays["ranges_spans"], toy_rowspace().arrays["ranges_spans"])


def test_export_legacy_writes_the_declared_files_and_refuses_other_formats(tmp_path):
    b = toy_bundle()
    bid = sb.write_bundle(b, tmp_path / "b")
    [p] = sb.export_legacy(sb.load_bundle(tmp_path / "b" / bid), "lerobot-legacy", tmp_path / "out")
    assert p == tmp_path / "out" / "signals" / "toy" / "lerobot_rows.npz"
    with np.load(p) as z:
        np.testing.assert_array_equal(z["rows"], np.array(TOY_TRAINABLE))
    with pytest.raises(sb.BundleError, match="no openpi-legacy files"):
        sb.export_legacy(b, "openpi-legacy", tmp_path / "out")
    with pytest.raises(sb.BundleError):
        sb.export_legacy(b, "tf-legacy", tmp_path / "out")


def test_object_arrays_and_bad_legacy_names_are_refused(tmp_path):
    b = toy_bundle()
    with pytest.raises(sb.BundleError):
        sb.write_bundle(dataclasses.replace(b, legacy_arrays={"rows": np.array([object()])}), tmp_path)
    with pytest.raises(sb.BundleError):
        sb.write_bundle(dataclasses.replace(b, legacy_arrays={"Bad-Name": np.zeros(1)}), tmp_path)


def test_a_bad_rowspace_array_name_is_refused(tmp_path):
    b = toy_bundle()
    bad_rs = dataclasses.replace(b.rowspace, arrays={**b.rowspace.arrays, "Bad-Name": np.zeros(1)})
    with pytest.raises(sb.BundleError, match="frame-index array name"):
        sb.write_bundle(dataclasses.replace(b, rowspace=bad_rs), tmp_path)


def test_a_plan_with_the_wrong_dtype_is_refused_at_write_and_at_load(tmp_path):
    b = toy_bundle()
    bad_plan = b.plan.astype(np.float64)
    with pytest.raises(sb.BundleError, match="plan"):
        sb.write_bundle(dataclasses.replace(b, plan=bad_plan), tmp_path / "write")
    d = _forge_array(tmp_path / "load", b, "plan", bad_plan)
    with pytest.raises(sb.BundleError, match="plan"):
        sb.load_bundle(d)


def test_weights_of_the_wrong_length_are_refused_at_write_and_at_load(tmp_path):
    b = toy_bundle(weights=True)
    bad_weights = b.weights[:-1]                                        # 22, not rowspace.n_rows == 23
    with pytest.raises(sb.BundleError, match="weights"):
        sb.write_bundle(dataclasses.replace(b, weights=bad_weights), tmp_path / "write")
    d = _forge_array(tmp_path / "load", b, "weights", bad_weights)
    with pytest.raises(sb.BundleError, match="weights"):
        sb.load_bundle(d)


def test_cond_whose_field_count_disagrees_with_the_vocab_is_refused_at_write_and_at_load(tmp_path):
    b = toy_bundle(cond=True)
    bad_cond = np.concatenate([b.cond, b.cond], axis=-1)                # F=2, vocab has 1 field
    with pytest.raises(sb.BundleError, match="cond"):
        sb.write_bundle(dataclasses.replace(b, cond=bad_cond), tmp_path / "write")
    d = _forge_array(tmp_path / "load", b, "cond", bad_cond)
    with pytest.raises(sb.BundleError, match="cond"):
        sb.load_bundle(d)


def test_a_legacy_record_referencing_a_missing_array_is_refused_at_write(tmp_path):
    b = toy_bundle()
    bad_doc = {"$o": [["rows", {"$array": "legacy.nope"}]]}
    bad = dataclasses.replace(b, legacy=(dataclasses.replace(b.legacy[0], doc=bad_doc),))
    with pytest.raises(sb.BundleError, match="missing array"):
        sb.write_bundle(bad, tmp_path)


def test_two_legacy_records_sharing_a_path_are_refused_at_write(tmp_path):
    b = toy_bundle()
    dup = dataclasses.replace(b, legacy=b.legacy + b.legacy)
    with pytest.raises(sb.BundleError, match="more than one record"):
        sb.write_bundle(dup, tmp_path)


def test_every_corruption_path_raises_bundle_error_not_a_raw_exception(tmp_path):
    bid = sb.write_bundle(toy_bundle(), tmp_path)
    d = tmp_path / bid
    missing_manifest = tmp_path / "missing-manifest"
    shutil.copytree(d, missing_manifest)
    (missing_manifest / "manifest.json").unlink()
    with pytest.raises(sb.BundleError):
        sb.load_bundle(missing_manifest)                # a BundleError, not a raw FileNotFoundError

    trunc_parent = tmp_path / "trunc_parent"
    truncated = trunc_parent / bid                        # same basename: the manifest itself is untouched
    shutil.copytree(d, truncated)
    (truncated / "plan.npy").write_bytes(b"not an npy file")
    with pytest.raises(sb.BundleError):
        sb.load_bundle(truncated)                        # a BundleError, not a raw ValueError

    missing_key = tmp_path / "missing-key"
    shutil.copytree(d, missing_key)
    man = json.loads((missing_key / "manifest.json").read_bytes())
    del man["budget"]
    raw = dumps_canonical(man)
    new_dir = tmp_path / hashlib.sha256(raw).hexdigest()
    (missing_key / "manifest.json").write_bytes(raw)
    missing_key.rename(new_dir)
    with pytest.raises(sb.BundleError):
        sb.load_bundle(new_dir)                          # a BundleError, not a raw KeyError


def _forge_manifest(d, man: dict):
    """Write `man` as the new canonical manifest of the bundle at `d`, renamed to its new id."""
    raw = dumps_canonical(man)
    new_dir = d.parent / hashlib.sha256(raw).hexdigest()
    (d / "manifest.json").write_bytes(raw)
    d.rename(new_dir)
    return new_dir


def test_a_manifest_whose_arrays_or_rowspace_is_a_list_raises_bundle_error_not_attributeerror(tmp_path):
    b = toy_bundle()
    d = tmp_path / "arrays" / sb.write_bundle(b, tmp_path / "arrays")
    man = json.loads((d / "manifest.json").read_bytes())
    man["arrays"] = []
    with pytest.raises(sb.BundleError):
        sb.load_bundle(_forge_manifest(d, man))

    d = tmp_path / "rowspace" / sb.write_bundle(b, tmp_path / "rowspace")
    man = json.loads((d / "manifest.json").read_bytes())
    man["rowspace"] = []
    with pytest.raises(sb.BundleError):
        sb.load_bundle(_forge_manifest(d, man))

    # vocab null plus a cond array: _check_array_shapes must not reach vocab.fields on None.
    d = tmp_path / "cond" / sb.write_bundle(b, tmp_path / "cond")
    man = json.loads((d / "manifest.json").read_bytes())
    cond = np.zeros((4, 2, 1), dtype=np.uint8)
    np.save(d / "cond.npy", cond, allow_pickle=False)
    man["arrays"]["cond"] = {"dtype": cond.dtype.str, "shape": list(cond.shape), "sha256": sb.array_digest(cond)}
    with pytest.raises(sb.BundleError):
        sb.load_bundle(_forge_manifest(d, man))


def test_a_standalone_rowspace_with_a_bad_array_name_is_refused_at_write(tmp_path):
    rs = dataclasses.replace(toy_rowspace(), arrays={**toy_rowspace().arrays, "Bad-Name": np.zeros(1)})
    with pytest.raises(sb.BundleError, match="frame-index array name"):
        sb.write_rowspace(rs, tmp_path)


def test_a_malformed_legacy_table_node_raises_bundle_error_not_a_raw_exception(tmp_path):
    b = toy_bundle()
    bad_doc = {"$spans": "x"}
    bad = dataclasses.replace(b, legacy=(dataclasses.replace(b.legacy[0], doc=bad_doc),))
    with pytest.raises(sb.BundleError):
        sb.write_bundle(bad, tmp_path)


def test_export_legacy_checks_array_shapes_for_a_bundle_that_was_never_written(tmp_path):
    b = toy_bundle(weights=True)
    bad = dataclasses.replace(b, weights=b.weights[:-1])          # 22, not rowspace.n_rows == 23
    with pytest.raises(sb.BundleError, match="weights"):
        sb.export_legacy(bad, "lerobot-legacy", tmp_path)


def test_the_array_shapes_docstring_does_not_point_at_the_private_contract():
    assert "Contract §" not in inspect.getdoc(sb._check_array_shapes)


def test_toy_rowspace_codecs_agree_with_codec_for_path():
    rs = toy_rowspace()
    assert rs.roots_codec == legacy.codec_for_path(rs.roots_path)
    assert rs.ranges_codec == legacy.codec_for_path(rs.ranges_path)
