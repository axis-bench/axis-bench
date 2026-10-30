"""Staging writes one subtree of a published tree and nothing else; local builds stay outside it."""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import stat

import pytest

from axis_qvla.signals import staging


def _pack(tmp_path):
    pack = tmp_path / "qvla-v1-corpus"
    (pack / "index").mkdir(parents=True)
    (pack / "index" / "ranges.json").write_text("{}")
    lock = {"files": [{"path": "index/ranges.json", "sha256": hashlib.sha256(b"{}").hexdigest()}]}
    (pack / staging.LOCK_NAME).write_text(json.dumps(lock))
    return pack


def test_only_the_named_subtree_of_a_published_tree_is_writable(tmp_path):
    pack = _pack(tmp_path)
    staging.check_out(pack / "bundles", "bundles")
    staging.check_out(tmp_path / "elsewhere", "bundles", partial=True)
    for out in (pack, pack / "index", pack / "signals" / "v1", pack / "traces_sample", pack / "bundles" / "x"):
        with pytest.raises(staging.StagingError, match="staging writes only"):
            staging.check_out(out, "bundles")
    with pytest.raises(staging.StagingError, match="local build"):
        staging.check_out(pack / "bundles", "bundles", partial=True)


ROOT = pytest.mark.skipif(os.geteuid() == 0, reason="root reads a directory whatever its mode, so it cannot be "
                                                   "made unreadable here")


@pytest.fixture
def is_file_swallows_every_oserror(monkeypatch):
    """`Path.is_file` as Python 3.14 has it: False on any OSError, EACCES included."""
    def is_file(self):
        try:
            return stat.S_ISREG(os.stat(self).st_mode)
        except OSError:
            return False

    monkeypatch.setattr(pathlib.Path, "is_file", is_file)


@ROOT
def test_an_untraversable_tree_is_never_read_as_no_published_tree(tmp_path, is_file_swallows_every_oserror):
    """A tree whose lock cannot be stat'ed is not "outside every published tree": a partial build
    pointed into it is refused by an error, never let through."""
    pack = _pack(tmp_path)
    os.chmod(pack, 0o000)
    try:
        with pytest.raises(PermissionError):
            staging.check_out(pack / "bundles", "bundles", partial=True)
        with pytest.raises(PermissionError):
            staging.pack_root(pack / "bundles")
    finally:
        os.chmod(pack, 0o755)
    assert staging.pack_root(pack / "bundles") == pack


def test_a_path_below_a_file_does_not_exist_and_is_still_refused_inside_the_tree(tmp_path):
    pack = _pack(tmp_path)
    assert staging.kind_of(pack / "index" / "ranges.json" / "x") is None
    assert staging.kind_of(pack / "index" / "ranges.json") == "file" and staging.kind_of(pack) == "dir"
    with pytest.raises(staging.StagingError, match="staging writes only"):
        staging.check_out(pack / "index" / "ranges.json" / "bundles", "bundles")
