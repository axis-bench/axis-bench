"""The published replayable source of the real-world line's simulation pretraining demonstrations
(axisrobotics/qvla-v1-realworld) as the renderer reads it.

Layout, relative to its root (T a task id, A an attempt id, I a scene variant, L a mesh layout;
the docs' real-world reference page shows it too):
  manifest.json                                         every task, its instruction, its scene variants and
                                                        mesh layouts, and each episode's attempt id and variant
  robots/franka_emika_panda/                            the Panda meshes the scenes name
  tabletopgen/L.tar                                     one layout directory of object meshes and textures
  tasks/task_T/scenes/variant_I.xml                     a scene the task was recorded in
  tasks/task_T/scenes/attempt_A.json                    the episode as the renderer reads it
  tasks/task_T/traces/task_T_attempt_A.jsonl            the recorded trace the labels are computed from
  tasks/task_T/goals_per_attempt/task_T_attempt_A.json  the task checkers' goal record
  tasks/task_T/data/task_T/checker_config.json          the task's checker configuration
A scene names its assets `../../../robots/...` and `../../../tabletopgen/...`. `prepare` builds a tree
of the same layout beside the downloaded copy, with every needed `.tar` extracted, linking the task files
and the robots rather than copying them (a copy where the two trees sit on different filesystems), so
the downloaded copy keeps exactly the files its checksum manifest lists. The renderer reads the prepared
tree.

Render-tier package, stdlib only.
"""
from __future__ import annotations

import json
import os
import shutil
import tarfile
from pathlib import Path

SCHEMA = "qvla.realworld/1"
MANIFEST = "manifest.json"


class DataError(ValueError):
    """A copy of the qvla-v1-realworld data that lacks what a command needs, or a selection it does not hold."""


def _task(tid: int) -> str:
    return f"tasks/task_{int(tid)}"


def scene_rel(tid: int, variant: int) -> str:
    return f"{_task(tid)}/scenes/variant_{int(variant)}.xml"


def episode_rel(tid: int, aid: int) -> str:
    return f"{_task(tid)}/scenes/attempt_{int(aid)}.json"


def trace_rel(tid: int, aid: int) -> str:
    return f"{_task(tid)}/traces/task_{int(tid)}_attempt_{int(aid)}.jsonl"


def goals_rel(tid: int, aid: int) -> str:
    return f"{_task(tid)}/goals_per_attempt/task_{int(tid)}_attempt_{int(aid)}.json"


def checker_rel(tid: int) -> str:
    return "/".join((_task(tid), "data", f"task_{int(tid)}", "checker_config.json"))


def mesh_tar_rel(layout: str) -> str:
    return f"tabletopgen/{layout}.tar"


def load_manifest(root: Path) -> dict:
    path = Path(root) / MANIFEST
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataError(f"{path}: cannot read ({exc.__class__.__name__}: {exc}); fetch it with "
                        f"`qvla data get realworld --include manifest.json`") from None
    if doc.get("schema") != SCHEMA:
        raise DataError(f"{path}: schema {doc.get('schema')!r}, expected {SCHEMA!r}")
    return doc


def select(manifest: dict, tasks=None) -> list[int]:
    """The task ids of `tasks` (every task when None), refusing one the data does not hold."""
    have = sorted(int(t) for t in manifest["tasks"])
    if tasks is None:
        return have
    missing = sorted(set(int(t) for t in tasks) - set(have))
    if missing:
        raise DataError(f"tasks {missing} are not in the qvla-v1-realworld data ({len(have)} tasks)")
    return sorted(set(int(t) for t in tasks))


def _layouts(manifest: dict, chosen: list[int]) -> list[str]:
    return sorted({lay for t in chosen for lay in manifest["tasks"][str(t)]["layouts"]})


def _task_files(manifest: dict, tid: int) -> list[str]:
    """Every file of task `tid`: its scenes, its checker configuration, and each episode's three files."""
    task = manifest["tasks"][str(tid)]
    files = [scene_rel(tid, v) for v in task["variants"]] + [checker_rel(tid)]
    for e in task["episodes"]:
        files += [episode_rel(tid, e["attempt_id"]), trace_rel(tid, e["attempt_id"]), goals_rel(tid, e["attempt_id"])]
    return files


def task_episode_files(root: Path, manifest: dict, tasks=None) -> dict[int, list[Path]]:
    """{task id: its source episode files} of the selected tasks, as the manifest lists them."""
    return {t: [Path(root) / episode_rel(t, e["attempt_id"])
                for e in sorted(manifest["tasks"][str(t)]["episodes"], key=lambda e: e["attempt_id"])]
            for t in select(manifest, tasks)}


def check_prepared(root: Path, manifest: dict, tasks=None) -> None:
    """Refuse a tree whose selected tasks' mesh layouts are not extracted -- the downloaded copy rather
    than the tree `prepare` builds from it -- whose scenes would name meshes that are not there."""
    for t in select(manifest, tasks):
        for lay in manifest["tasks"][str(t)]["layouts"]:
            if not (Path(root) / "tabletopgen" / lay).is_dir():
                raise DataError(f"{root}: task {t}'s mesh layout tabletopgen/{lay}/ is not extracted; render from "
                                f"the tree `render prepare` builds from the downloaded copy")


def _link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        part = dst.with_name(dst.name + ".partial")
        shutil.copyfile(src, part)
        os.replace(part, dst)


def _extract(tar: Path, tabletopgen: Path, layout: str) -> None:
    """One layout's archive as tabletopgen/<layout>/, whole or not at all: a layout that is there is
    never extracted again, so an interrupted extraction must not leave a part of one there."""
    part = tabletopgen / f".{layout}.partial"
    if part.exists():
        shutil.rmtree(part)
    with tarfile.open(tar) as t:
        t.extractall(part, filter="data")
    (part / layout).rename(tabletopgen / layout)
    part.rmdir()


def prepare(data: Path, out: Path, tasks=None) -> dict:
    """The render input tree of `tasks` at `out`: their files and the robots linked, their layouts extracted."""
    data, out = Path(data), Path(out)
    if out.resolve() == data.resolve() or data.resolve() in out.resolve().parents:
        raise DataError(f"{out} is inside the downloaded copy {data}, whose files the checksum manifest lists; "
                        f"prepare into a directory beside it")
    manifest = load_manifest(data)
    chosen = select(manifest, tasks)
    layouts = _layouts(manifest, chosen)
    files = [rel for t in chosen for rel in _task_files(manifest, t)] + list(manifest["robots"])
    missing = [rel for rel in files + [mesh_tar_rel(lay) for lay in layouts] if not (data / rel).is_file()]
    if missing:
        shown = ", ".join(missing[:3]) + (f" and {len(missing) - 3} more" if len(missing) > 3 else "")
        raise DataError(f"{len(missing)} files the selected tasks need are not in {data}: {shown}; fetch each "
                        f"with `qvla data get realworld --include '<path>'`")
    for rel in files + [MANIFEST]:
        _link(data / rel, out / rel)
    (out / "tabletopgen").mkdir(parents=True, exist_ok=True)
    for lay in layouts:
        if not (out / "tabletopgen" / lay).is_dir():
            _extract(data / mesh_tar_rel(lay), out / "tabletopgen", lay)
    return {"tasks": len(chosen), "episodes": sum(len(manifest["tasks"][str(t)]["episodes"]) for t in chosen),
            "layouts": len(layouts)}
