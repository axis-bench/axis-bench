"""The scene builders' side of the asset contract: every file a scene compiles from was checked.

WHY. axis_qvla.bench.sim.assets makes every store check what it hands out, but two holes sit in the
pinned AXIS-Render materialiser (axis_replay_native.asset_materializer), which the release does
not edit:

* it reads any XML or asset ALREADY in its work dir and asks the store only for what is absent
  (asset_materializer.py:130-135 and :156-162). The eval work dir outlives a run
  (<run_root>/work_<arm>/<worker>/task_<id>), so a stale mesh there would be compiled without
  meeting --asset-dir or ASSETS.lock.json. Verifying the work dir AFTER materialisation does not
  close it: the materialiser rewrites XML file references in place, so a work-dir XML need not equal
  its locked bytes even when it is the right file, and a check of the meshes alone would leave the
  XMLs unchecked. `fresh_scene_dir` empties the target instead, so every file there was written in
  this build from bytes the store returned (and the store checked); the one exception is the entry
  XML, which is the task's own mjcf_xml;
* its XML resolver swallows the loader's exception, so a refused include would surface later as a
  MuJoCo compile error -- an error seed. `raise_if_refused` re-raises it.

`check_asset_inputs` is the rollout CLI's up-front check, so a broken --asset-dir or
--asset-checksums costs no server time and no seed. Stdlib only.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from axis_qvla.bench.sim.assets import AssetFetchRefused, load_assets_lock

# The directory, under a builder's work dir, that one scene is materialised into; emptied first.
SCENE_DIR = "scene"


def fresh_scene_dir(work_dir: str | os.PathLike[str]) -> Path:
    """An EMPTY `<work_dir>/scene`, for one materialise_scene call. Only this fixed subdirectory is
    deleted (the render shell's `<work_dir>/layouts` cache is not a scene input and is kept); a
    symlink there is unlinked, never followed."""
    d = Path(work_dir) / SCENE_DIR
    if d.is_symlink() or d.is_file():
        d.unlink()
    elif d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def raise_if_refused(store: object) -> None:
    """Re-raise a refusal the materialiser swallowed; call it right after materialize_scene."""
    refused = getattr(store, "refused", None)
    if refused is not None:
        raise refused


def check_asset_inputs(asset_dir: str | os.PathLike[str] | None,
                       lock: str | os.PathLike[str] | None) -> None:
    """A named --asset-dir must be a directory and a named --asset-checksums must load. Each failure
    is AssetFetchRefused, which the rollout CLI turns into exit 4 before it contacts the server."""
    if asset_dir is not None:
        try:
            is_dir = Path(asset_dir).is_dir()
        except OSError as exc:   # is_dir() raises EACCES rather than answering False
            raise AssetFetchRefused(f"asset dir {asset_dir} cannot be read: {exc}") from None
        if not is_dir:
            raise AssetFetchRefused(f"asset dir {asset_dir} is not a directory")
    if lock is not None:
        load_assets_lock(lock)
