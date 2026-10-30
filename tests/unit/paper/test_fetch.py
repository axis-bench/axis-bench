"""HfSource's cache-hit path: a cached file whose sha256 matches its checksum-manifest row is
returned without touching the network."""
from __future__ import annotations

import hashlib
from pathlib import Path

from axis_qvla.paper.fetch import HfSource
from axis_qvla.data.hf import HfRepo

REPO = HfRepo("axisrobotics/qvla-fixture", "dataset")
REV = "c" * 40


def _source(tmp_path: Path, *, sha: str, size: int) -> HfSource:
    return HfSource(REPO, REV, tmp_path / "cache", token=None, endpoint="http://127.0.0.1:9",
                    locked=(("cells.json", sha, size),))


def test_a_cache_hit_returns_without_touching_the_network(tmp_path: Path):
    data = b"hello world"
    dest = tmp_path / "cache" / REPO.repo_id.replace("/", "__") / REV / "cells.json"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    src = _source(tmp_path, sha=sha, size=len(data))
    assert src.path("cells.json", sha256=None) == dest

