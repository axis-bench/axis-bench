"""axis_qvla.bench.checkpoint_digest: a local checkpoint directory against the pub_digest cells.json records."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from axis_qvla.bench import checkpoint_digest as cd
from axis_qvla.bench.sim.campaign import digest_tree


def _checkpoint(root: Path) -> Path:
    for rel, data in {"_CHECKPOINT_METADATA": b"m", "params/_METADATA": b"p", "params/d/0": b"w0",
                      "assets/a/norm_stats.json": b"{}",
                      "train_state/opt": b"optimizer", ".cache/x": b"download cache",
                      "PROVENANCE.md": b"re-publish note"}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    return root


def _by_hand(root: Path, rels: list[str]) -> str:
    h = hashlib.sha256()
    for rel in rels:
        h.update(rel.encode() + b"\0" + hashlib.sha256((root / rel).read_bytes()).hexdigest().encode() + b"\n")
    return h.hexdigest()


def _cells(cid: str, digest: dict | None) -> bytes:
    return json.dumps({"cells": [{"checkpoint": {"id": cid, "pub_digest": digest}},
                                 {"checkpoint": {"id": "pi05/ft/other", "pub_digest": {"sha256": "0" * 64}}}]}).encode()


def test_the_digest_covers_exactly_the_published_files_in_path_order(tmp_path):
    ck = _checkpoint(tmp_path / "ck")
    got = cd.pub_digest(ck, "pi05")
    rels = ["_CHECKPOINT_METADATA", "assets/a/norm_stats.json", "params/_METADATA", "params/d/0"]
    assert got == {"sha256": _by_hand(ck, rels), "n_files": 4, "bytes": 1 + 2 + 1 + 2}
    assert cd.published_files(ck, "pi05") == rels


def test_a_matching_copy_passes_and_one_changed_byte_is_refused(tmp_path):
    ck = _checkpoint(tmp_path / "ck")
    cells = _cells("pi05/ft/x", cd.pub_digest(ck, "pi05"))
    assert cd.mismatch(ck, "pi05/ft/x", cells) is None
    (ck / "train_state" / "more").write_bytes(b"state beside the published files is not published")
    assert cd.mismatch(ck, "pi05/ft/x", cells) is None
    (ck / "params" / "d" / "0").write_bytes(b"w1")
    msg = cd.mismatch(ck, "pi05/ft/x", cells)
    assert msg.startswith(f"--checkpoint {ck} is not the published pi05/ft/x: its published files digest to ")
    assert msg.endswith("serve it under its own --config and --run")
    # A copy the caller chose is named as such, and the advice to serve it as another run is the caller's.
    labelled = cd.mismatch(ck, "pi05/ft/x", cells, label=f"the local copy {ck}")
    assert labelled.startswith(f"the local copy {ck} is not the published pi05/ft/x") and "--config" not in labelled
    (ck / "params" / "d" / "0").write_bytes(b"w0")
    (ck / "params" / "d" / "1").write_bytes(b"an extra published-tree file")
    assert cd.mismatch(ck, "pi05/ft/x", cells) is not None


@pytest.mark.parametrize("cells, match", [
    (_cells("pi05/ft/x", None), "records no pub_digest for checkpoint pi05/ft/x"),
    (_cells("pi05/ft/y", {"sha256": "1" * 64}), "records no pub_digest for checkpoint pi05/ft/x"),
    (b"not json", "the results map cannot be read"),
])
def test_nothing_recorded_is_a_refusal_not_a_pass(tmp_path, cells, match):
    ck = _checkpoint(tmp_path / "ck")
    with pytest.raises(cd.CheckpointDigestError, match=match):
        cd.mismatch(ck, "pi05/ft/x", cells)


def test_a_missing_directory_or_an_unknown_family_is_refused(tmp_path):
    with pytest.raises(cd.CheckpointDigestError, match="is not a directory"):
        cd.mismatch(tmp_path / "absent", "pi05/ft/x", _cells("pi05/ft/x", {"sha256": "0" * 64}))
    with pytest.raises(cd.CheckpointDigestError, match="names no known family"):
        cd.family("gr00t/ft/x")
    assert {cd.family(f"{f}/ft/x") for f in ("pi05", "smolvla", "xvla")} == {"pi05", "smolvla", "xvla"}


def test_a_family_leaves_out_its_own_training_state(tmp_path):
    ck = tmp_path / "ck"
    for rel in ("pretrained_model/model.safetensors", "training_state/optimizer.safetensors"):
        (ck / rel).parent.mkdir(parents=True, exist_ok=True)
        (ck / rel).write_bytes(b"x")
    assert cd.published_files(ck, "smolvla") == ["pretrained_model/model.safetensors"]


def test_every_paper_checkpoint_id_has_one_recorded_digest():
    """What the flag compares against: the committed cell map records one pub_digest per id."""
    cells = (Path(__file__).resolve().parents[3] / "experiments" / "paper" / "cells.json").read_bytes()
    ids = {c["checkpoint"]["id"] for c in json.loads(cells)["cells"]}
    assert ids and all(len(cd.recorded(i, cells)["sha256"]) == 64 for i in ids)


# --- one walk for every checkpoint digest ----------------------------------------------------------

def _rglob_digest(root: Path) -> dict:
    """The rglob digest of a directory: every regular file under `root`, sorted as Paths,
    (relative path, sha256) pairs. On a directory with no links and nothing unreadable the shared
    walk gives these exact bytes, so every digest a run records of such a directory is this one."""
    files = sorted(p for p in root.rglob("*") if p.is_file())
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(root)).encode() + b"\0" + hashlib.sha256(p.read_bytes()).hexdigest().encode() + b"\n")
    return {"sha256": h.hexdigest(), "n_files": len(files), "bytes": sum(p.stat().st_size for p in files)}


def _write(root: Path, files: dict[str, bytes]) -> Path:
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    return root


PLAIN = {"a/b": b"1", "a.b": b"2", "a/b.c/d": b"3", "z": b"4", "a-b/x": b"5", "params/d/10": b"6",
         "params/d/9": b"7", ".hidden/f": b"8", "sp ace/é.json": b"9", "train_state/opt": b"10"}


def test_on_a_plain_directory_the_shared_walk_is_the_recorded_whole_directory_digest(tmp_path):
    ck = _write(tmp_path / "ck", PLAIN)
    (ck / "empty").mkdir()
    assert cd.tree_digest(ck) == _rglob_digest(ck)
    assert digest_tree(ck) == {"path": str(ck), **_rglob_digest(ck)}


def test_a_linked_directory_digests_as_the_files_it_leads_to(tmp_path):
    """A checkpoint whose params/ is a link: the run digest, the published digest and a copy of the
    same bytes all agree, and the linked weights are counted."""
    files = {"_CHECKPOINT_METADATA": b"m", "params/_METADATA": b"p", "params/d/0": b"w0"}
    copy = _write(tmp_path / "copy", files)
    linked = _write(tmp_path / "linked", {"_CHECKPOINT_METADATA": b"m"})
    _write(tmp_path / "elsewhere", {"_METADATA": b"p", "d/0": b"w0"})
    (linked / "params").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    assert digest_tree(linked)["n_files"] == 3
    assert digest_tree(linked)["sha256"] == digest_tree(copy)["sha256"]
    assert cd.pub_digest(linked, "pi05")["sha256"] == digest_tree(linked)["sha256"]
    assert cd.pub_digest(linked, "pi05") == cd.pub_digest(copy, "pi05")


def test_a_linked_file_digests_as_the_bytes_it_opens(tmp_path):
    copy = _write(tmp_path / "copy", {"params/w": b"weights"})
    linked = tmp_path / "linked"
    (linked / "params").mkdir(parents=True)
    (linked / "params" / "w").symlink_to(copy / "params" / "w")
    assert digest_tree(linked)["sha256"] == digest_tree(copy)["sha256"]


def _refused(root: Path, match: str, *, listing: bool = True) -> None:
    """Every digest refuses; so does the listing alone, unless only a file's bytes are unreadable."""
    for fn in ((cd.tree_files,) if listing else ()) + (cd.tree_digest, digest_tree, lambda r: cd.pub_digest(r, "pi05")):
        with pytest.raises(cd.CheckpointDigestError, match=match):
            fn(root)


def test_an_unreadable_directory_is_refused_never_skipped(tmp_path, monkeypatch):
    """EACCES on a subdirectory (decision-0006 class): the digest fails, naming it; it never reads as
    an empty directory, which would give the checkpoint without its weights the same identity."""
    import os

    ck = _write(tmp_path / "ck", {"_CHECKPOINT_METADATA": b"m", "params/d/0": b"w0"})
    real = os.scandir

    def scandir(path="."):
        if Path(path) == ck / "params":
            raise PermissionError(13, "Permission denied", str(path))
        return real(path)

    monkeypatch.setattr(os, "scandir", scandir)
    _refused(ck, r"params/ cannot be listed \(PermissionError")


def test_an_unreadable_file_is_refused(tmp_path, monkeypatch):
    ck = _write(tmp_path / "ck", {"_CHECKPOINT_METADATA": b"m", "params/d/0": b"w0"})

    def unreadable(p):
        raise PermissionError(13, "Permission denied", str(p))

    monkeypatch.setattr(cd, "sha256_file", unreadable)
    _refused(ck, r"cannot be read \(PermissionError", listing=False)


def test_a_dangling_link_is_refused(tmp_path):
    ck = _write(tmp_path / "ck", {"_CHECKPOINT_METADATA": b"m"})
    (ck / "params").symlink_to(tmp_path / "gone", target_is_directory=True)
    _refused(ck, r"params cannot be read \(FileNotFoundError")


def test_a_link_back_into_the_tree_is_refused(tmp_path):
    ck = _write(tmp_path / "ck", {"params/d/0": b"w0"})
    (ck / "params" / "loop").symlink_to(ck, target_is_directory=True)
    _refused(ck, "params/loop links back to a directory it is inside")


def test_an_entry_that_is_not_a_file_or_directory_is_refused(tmp_path):
    import os

    ck = _write(tmp_path / "ck", {"_CHECKPOINT_METADATA": b"m"})
    os.mkfifo(ck / "pipe")
    _refused(ck, "pipe is neither a regular file nor a directory")


def test_the_run_manifest_refuses_a_checkpoint_it_cannot_digest(tmp_path):
    """The driver records nothing for a checkpoint it cannot identify: a refusal, before
    run_manifest.json, never a record whose digest left the unreadable part out."""
    import types

    from axis_qvla.bench.sim import campaign

    ck = _write(tmp_path / "ck", {"_CHECKPOINT_METADATA": b"m"})
    (ck / "params").symlink_to(tmp_path / "gone", target_is_directory=True)
    a = types.SimpleNamespace(arm="x", config="cfg", checkpoint=str(ck), worker_flags="", policy=[])
    with pytest.raises(SystemExit, match="--checkpoint cannot be digested"):
        campaign.run_manifest(a, {}, [], probe=False)
