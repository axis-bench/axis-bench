"""`qvla data get`: fetch a published v1.0 data set through the release checksum manifest, verified.

Why: a local copy is only as good as the bytes in it. Every file comes from the commit the release
checksum manifest pins (never a tag that could move), is checked against the checksum manifest's
sha256 and size before it is renamed into place (axis_qvla.data.hf.download), and a file already present
with the bytes the checksum manifest lists is not fetched again, so an interrupted get resumes. A
target the checksum manifest does not cover stops the command; bytes that disagree with the checksum
manifest are refused and never left on disk.

Targets: corpus, bench, results, realworld, traces-sample, checkpoints/<result | run | checkpoint id>
(resolved through experiments/paper/cells.json; --with-init adds the checkpoint it was trained
from). Where the files go is axis_qvla.data.layout's answer, the same one training and evaluation read.
The base weights come from their publishers instead (axis_qvla.data.bases).

Core tier: stdlib only.
"""
from __future__ import annotations

import dataclasses
import fnmatch
from collections.abc import Callable, Mapping
from pathlib import Path

from axis_qvla.core.canon import sha256_file
from axis_qvla.data import layout
from axis_qvla.data.hf import HfRepo, download

TARGETS: dict[str, tuple[str, str]] = {         # target -> (repo name, path prefix)
    "corpus": ("qvla-v1-corpus", ""),
    "traces-sample": ("qvla-v1-corpus", "traces_sample/"),
    "bench": ("qvla-v1-bench", ""),
    "results": ("qvla-v1-results", ""),
    "realworld": ("qvla-v1-realworld", ""),
}
LOCAL_ROUTE = (
    "the release checksum manifest {lock} does not exist, so nothing can be fetched verified. "
    "`qvla paper tables` runs without it from a local snapshot: download axisrobotics/qvla-v1-results "
    "at tag v1.0 (e.g. "
    "`hf download axisrobotics/qvla-v1-results --repo-type dataset --revision v1.0 "
    "--local-dir <dir>`) and run `qvla paper tables --results <dir> --check`; the committed results map "
    "and its frozen expected/ pin every byte it reads (docs/reference/data.md#from-a-local-snapshot)")


class GetError(ValueError):
    """The target is unknown, or the checksum manifest does not cover it."""


@dataclasses.dataclass(frozen=True)
class Selection:
    repo: str      # "<org>/<name>"
    prefix: str    # "" or "a/b/"
    label: str


def _ckpt(ref: str, label: str) -> Selection:
    """A checkpoint ref as the files under it, read by axis_qvla.data.layout.parse_ref: the grammar
    `checkpoint_dest` reads too, so a ref one accepts the other places."""
    try:
        repo, rel = layout.parse_ref(ref)
    except layout.LayoutError as exc:
        raise GetError(f"{label}: {exc}") from None
    return Selection(repo, rel + "/", label)


def resolve(target: str, cells: Mapping, *, with_init: bool, report: Callable[[str], None] = print) -> list[Selection]:
    if target in TARGETS:
        name, prefix = TARGETS[target]
        return [Selection(f"{layout.ORG}/{name}", prefix, target)]
    kind, _, key = target.partition("/")
    if kind != "checkpoints" or not key:
        raise GetError(f"unknown target {target!r}: one of {', '.join(TARGETS)}, "
                       f"checkpoints/<result, run or checkpoint id>")
    hits = [c for c in cells.get("cells", []) if key in (c.get("cell"), c.get("run"), c["checkpoint"].get("id"))]
    if not hits:
        raise GetError(f"no result, run or checkpoint id {key!r} in the results map")
    c = hits[0]
    out = [_ckpt(c["checkpoint"]["ref"], f"checkpoints/{c['checkpoint'].get('id', key)}")]
    init = (c.get("lineage") or {}).get("init")
    if with_init:
        if init:
            out.append(_ckpt(init["ref"], f"checkpoints/{init['id']}"))
        else:
            # A from-scratch result has no initial checkpoint to fetch; say so.
            report(f"checkpoints/{key}: --with-init asked for the checkpoint this result was "
                   f"trained from, but the results map records no lineage.init for it")
    return out


def rows_for(lock: Mapping, sel: Selection, includes: list[str]) -> list[dict]:
    if sel.repo not in lock["repos"]:
        raise GetError(f"{sel.repo} is not in the release checksum manifest")
    rows = [f for f in lock["files"] if f["repo"] == sel.repo and f["path"].startswith(sel.prefix)
            and (not includes or any(fnmatch.fnmatch(f["path"][len(sel.prefix):], g) for g in includes))]
    if not rows:
        raise GetError(f"the release checksum manifest lists no file of {sel.repo} under {sel.prefix or '/'}"
                       + (f" matching {includes}" if includes else ""))
    return rows


def fetch(lock: Mapping, sel: Selection, rows: list[dict], dest: Path, *, token: str | None, endpoint: str,
          report: Callable[[str], None] = print) -> tuple[int, int]:
    """(fetched, already present). A file keeps its repository path under `dest`."""
    meta = lock["repos"][sel.repo]
    repo = HfRepo(sel.repo, meta["type"])
    fetched = present = 0
    for row in rows:
        out = Path(dest) / row["path"]
        if out.is_file() and out.stat().st_size == row["size"] and sha256_file(out) == row["sha256"]:
            present += 1
            continue
        download(repo, meta["commit"], row["path"], out, expected_sha256=row["sha256"],
                 expected_size=row["size"], token=token, endpoint=endpoint)
        fetched += 1
    report(f"{sel.label}: {fetched} fetched, {present} already present -> {dest}")
    return fetched, present
