"""`build-paper` rebuilds the published bundles from the published inputs: compiled from the pack's
locked files (--pretraining-data) and the published corpus (--episodes) into tmp_path, the manifest it
writes -- every arm's bundle id, every bundle's exports and controls, every row-space id -- is the
pack's staged bundles/manifest.json, byte for byte. This is the test that ties the compiler to the
published bundle ids. It takes five to ten minutes, depending on the CPU, and writes the 53
bundles (about 810 MB) under tmp_path; nothing is written into the pack."""
from __future__ import annotations

import json

import pytest

from axis_qvla.core.canon import dumps_canonical
from axis_qvla.signals import paper


@pytest.mark.slow
def test_build_paper_rebuilds_the_staged_manifest_byte_for_byte(signals_pack, corpus, tmp_path):
    staged = (signals_pack / "bundles" / "manifest.json").read_bytes()
    published = json.loads(staged)
    out = tmp_path / "bundles"
    man = paper.build(pack=signals_pack, corpus=corpus, out=out, log=lambda _line: None)
    assert man["rowspaces"] == published["rowspaces"]
    assert sorted(set(man["bundles"]) ^ set(published["bundles"])) == []
    assert [n for n, b in sorted(man["bundles"].items()) if b != published["bundles"][n]] == []
    assert man["arms"] == published["arms"]
    assert dumps_canonical(man) == staged
    assert (out / "manifest.json").read_bytes() == staged
    named = {"manifest.json", "rowspaces"} | {b["id"] for b in published["bundles"].values()}
    named |= {c for b in published["bundles"].values() for c in b["controls"]}
    assert sorted(p.name for p in out.iterdir()) == sorted(named)
