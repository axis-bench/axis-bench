"""The one content digest of a local checkpoint directory, and `bench eval --verify-checkpoint`.

Why one: a checkpoint is identified by its bytes in three places that must agree -- the run
manifest's `checkpoint.sha256` (axis_qvla.bench.sim.campaign.digest_tree, which becomes a results
file's `eval_digest` and is what a resumed run and a re-read are compared on), the published
`checkpoint.pub_digest` cells.json records, and `--verify-checkpoint`'s comparison with it. Each is
a sha256 over (relative path, file sha256) pairs in path-part order, computed here from ONE walk
(`tree_files`), so a directory that two walks would list differently cannot get two identities.

The walk reads the directory the way a server loading it does. A symbolic link is followed: a
linked file digests as the bytes it opens and a linked directory as the files under it, so a copy
and a link of the same weights get the same digest. Whatever cannot be read that way is refused
(CheckpointDigestError), never skipped: a directory that cannot be listed, a dangling link, an entry
that is neither a regular file nor a directory, and a directory link back to a directory the walk
is inside (a cycle). Skipping one would give two checkpoints that differ only there one identity.

`pub_digest` covers exactly the files the v1.0 model repository carries for a checkpoint -- never
its optimizer or training state, a download cache, or a PROVENANCE.md a re-publish left -- so a
local copy holding extra files outside those still matches, and a changed, missing or added
published file does not. Hashing reads every byte of the checkpoint, which is why
`--verify-checkpoint` is opt-in.

Core tier: stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from axis_qvla.core.canon import sha256_file

# The training state each family's checkpoint directory holds beside the published files.
_STATE = {"pi05": "train_state", "smolvla": "training_state", "xvla": "training_state"}


class CheckpointDigestError(ValueError):
    """The directory cannot be digested, or the cell map records nothing to compare it with."""


def family(checkpoint_id: str) -> str:
    """The backbone family a checkpoint id names (`pi05/ft/<run>` -> pi05)."""
    head = checkpoint_id.split("/", 1)[0]
    if head not in _STATE:
        raise CheckpointDigestError(f"checkpoint id {checkpoint_id!r} names no known family ({', '.join(_STATE)})")
    return head


def _parts(rel: str) -> tuple[str, ...]:
    return PurePosixPath(rel).parts


def _walk(d: Path, rel: PurePosixPath, inside: frozenset, out: list[str]) -> None:
    try:
        with os.scandir(d) as it:
            names = sorted(e.name for e in it)
    except OSError as exc:
        where = f"{rel.as_posix()}/" if rel.parts else "the directory"
        raise CheckpointDigestError(f"{where} cannot be listed ({type(exc).__name__}: {exc})") from None
    for name in names:
        r = rel / name
        try:
            st = (d / name).stat()          # follows a link: the bytes or the files it leads to
        except OSError as exc:
            raise CheckpointDigestError(f"{r.as_posix()} cannot be read ({type(exc).__name__}: {exc})") from None
        if stat.S_ISDIR(st.st_mode):
            key = (st.st_dev, st.st_ino)
            if key in inside:
                raise CheckpointDigestError(f"{r.as_posix()} links back to a directory it is inside")
            _walk(d / name, r, inside | {key}, out)
        elif stat.S_ISREG(st.st_mode):
            out.append(r.as_posix())
        else:
            raise CheckpointDigestError(f"{r.as_posix()} is neither a regular file nor a directory")


def tree_files(root: Path) -> list[str]:
    """Every file under `root` as a POSIX path relative to it, in digest (path-part) order."""
    root = Path(root)
    try:
        st = root.stat()
    except FileNotFoundError:
        raise CheckpointDigestError(f"{root} is not a directory (it does not exist)") from None
    except OSError as exc:
        raise CheckpointDigestError(f"{root} cannot be read ({type(exc).__name__}: {exc})") from None
    if not stat.S_ISDIR(st.st_mode):
        raise CheckpointDigestError(f"{root} is not a directory")
    out: list[str] = []
    try:
        _walk(root, PurePosixPath(), frozenset({(st.st_dev, st.st_ino)}), out)
    except CheckpointDigestError as exc:
        raise CheckpointDigestError(f"{root}: {exc}") from None
    return sorted(out, key=_parts)


def digest_entries(entries) -> str:
    """sha256 over (relative path, file sha256) pairs, in path-part order."""
    h = hashlib.sha256()
    for rel, sha in sorted(entries, key=lambda e: _parts(e[0])):
        h.update(rel.encode())
        h.update(b"\0")
        h.update(sha.encode())
        h.update(b"\n")
    return h.hexdigest()


def tree_digest(root: Path, keep: Callable[[str], bool] = lambda rel: True) -> dict:
    """{"sha256", "n_files", "bytes"} over the files of `root` that `keep` selects, from one walk."""
    root = Path(root)
    files = [rel for rel in tree_files(root) if keep(rel)]
    try:
        entries = [(rel, sha256_file(root / rel)) for rel in files]
        size = sum((root / rel).stat().st_size for rel in files)
    except OSError as exc:
        raise CheckpointDigestError(f"{root} cannot be read ({type(exc).__name__}: {exc})") from None
    return {"sha256": digest_entries(entries), "n_files": len(files), "bytes": size}


def _published(fam: str) -> Callable[[str], bool]:
    return lambda rel: _parts(rel)[0] not in (_STATE[fam], ".cache") and rel != "PROVENANCE.md"


def published_files(root: Path, fam: str) -> list[str]:
    """The files of `root` a v1.0 model repository carries, as POSIX paths in digest order."""
    return [rel for rel in tree_files(root) if _published(fam)(rel)]


def pub_digest(root: Path, fam: str) -> dict:
    """{"sha256", "n_files", "bytes"} of the published files under `root`."""
    try:
        return tree_digest(root, _published(fam))
    except CheckpointDigestError as exc:
        raise CheckpointDigestError(f"--checkpoint {exc}") from None


def recorded(checkpoint_id: str, cells: bytes) -> dict:
    """The one `pub_digest` the cell map records for `checkpoint_id`."""
    try:
        rows = json.loads(cells)["cells"]
    except (ValueError, KeyError, TypeError) as exc:
        raise CheckpointDigestError(f"the results map cannot be read ({exc})") from None
    found = {json.dumps(c["checkpoint"].get("pub_digest"), sort_keys=True) for c in rows
             if isinstance(c, dict) and isinstance(c.get("checkpoint"), dict) and c["checkpoint"].get("id") == checkpoint_id}
    found.discard("null")
    if len(found) != 1:
        what = "no pub_digest" if not found else f"{len(found)} different pub_digests"
        raise CheckpointDigestError(f"the results map records {what} for checkpoint {checkpoint_id}")
    return json.loads(found.pop())


def mismatch(checkpoint: Path, checkpoint_id: str, cells: bytes, *, label: str | None = None) -> str | None:
    """Why `checkpoint` is not the published `checkpoint_id`, or None when its digest matches.
    `label` names the directory as the user knows it (default `--checkpoint <dir>`, the directory
    they named); a label is the caller's own copy, so the advice to serve it as another run (which
    only fits weights the user chose) is left to the caller."""
    want = recorded(checkpoint_id, cells)
    got = pub_digest(checkpoint, family(checkpoint_id))
    if got["sha256"] == want.get("sha256"):
        return None
    head = (f"{label or f'--checkpoint {checkpoint}'} is not the published {checkpoint_id}: its published files "
            f"digest to {got['sha256'][:12]} ({got['n_files']} files, {got['bytes']} bytes), cells.json records "
            f"{str(want.get('sha256'))[:12]} ({want.get('n_files')} files, {want.get('bytes')} bytes).")
    return head if label else f"{head} Download the checkpoint again, or serve it under its own --config and --run"
