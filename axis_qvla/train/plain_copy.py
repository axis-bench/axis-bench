"""A plain merged LeRobot copy: what the LeRobot line's merge requires of its inputs, checked first.

A row space whose `lerobot_copy` is "plain" (conf/recipes/rowspaces.toml) trains SmolVLA and X-VLA
on a merge of whole task directories, built on this machine with the LeRobot line's
`python -m lerobot.scripts.axis_merge_datasets --out <dir> <task_<id> dir> ...`. That merge refuses
inputs whose meta/info.json differ in fps, robot type or any feature -- its dtype, shape, names and
`info`, apart from the video encoder settings and the spelling of the depth-map flag -- and says
only `Same features is expected`, followed by both feature tables. Every corpus task names its own
scene objects in `observation.body_pose`, so two tasks of different scenes always differ there.

    qvla train merge-check <task_<id> dir> ...

compares the inputs the way the merge does, before it runs, and names each difference, the tasks it
lies between and what to do about it. A difference in `names` alone is per-task metadata that no
training consumer reads: build the copy from copies of the task directories whose differing names
are cleared (`clear_differing_names`, as the smoke fixture does), never from edited published files.
Any other difference means the tasks are not one dataset, and no plain copy holds them both. A
task whose statistics do not have its features' declared shapes (axis_qvla.train.lerobot_stats) is
refused too: the merge would carry them into the copy.
Binding runs the same check over a plain row space's roots index while its copy is not built yet.

A plain copy's frames say nothing of the roots index it was merged from, so a copy built before the
index changed would train on the old task set. When `qvla train` builds the copy itself it records
the roots index's sha256 in the copy (`axis_plain.json`, PLAIN_RECORD), and binding refuses a copy
whose record names another index than the one on disk. A copy built by hand with the command above
carries no record and is not checked against the index.

Exit codes: 0 the inputs merge, 2 refused.

Core tier: stdlib only.
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib

from axis_qvla.train.lerobot_stats import stats_problems

# The keys under a feature's `info` the merge ignores (the LeRobot line's VIDEO_ENCODER_INFO_KEYS),
# and the depth-map flag's legacy spelling, which it folds into `is_depth_map`.
_ENCODER_INFO = frozenset(f"video.{k}" for k in ("g", "crf", "preset", "fast_decode", "extra_options",
                                                  "video_backend"))
_LEGACY_DEPTH = "video.is_depth_map"
_SHOWN = 6     # names listed per task in a refusal


def read_info(task_dir) -> dict:
    path = pathlib.Path(task_dir) / "meta" / "info.json"
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"{path} is not JSON ({exc})") from None
    if not isinstance(info, dict) or not isinstance(info.get("features"), dict):
        raise ValueError(f"{path} has no `features` table")
    return info


def merge_form(feature: dict, *, names: bool = True) -> dict:
    """A feature as the merge compares it: video encoder settings dropped, the depth flag under its
    one canonical key; without `names` when `names` is false."""
    if not isinstance(feature, dict):
        return {"value": feature}
    f = copy.deepcopy(feature)
    info = f.get("info") or {}
    video_info = f.get("video_info")
    legacy = isinstance(video_info, dict) and video_info.get(_LEGACY_DEPTH)
    depth = bool(info.get("is_depth_map") or info.get(_LEGACY_DEPTH) or legacy)
    f["info"] = {k: v for k, v in info.items() if k not in _ENCODER_INFO and k != _LEGACY_DEPTH}
    f["info"]["is_depth_map"] = depth
    if isinstance(video_info, dict):
        video_info = {k: v for k, v in video_info.items() if k != _LEGACY_DEPTH}
    if video_info:
        f["video_info"] = video_info
    else:
        f.pop("video_info", None)
    if not names:
        f.pop("names", None)
    return f


def _key(value) -> str:
    return json.dumps(value, sort_keys=True)


def _names_shown(tasks: dict[str, dict]) -> str:
    """Each task's names that not every task has, a few each."""
    lists = {t: f.get("names") if isinstance(f.get("names"), list) else [] for t, f in tasks.items()}
    common = set.intersection(*(set(map(_key, v)) for v in lists.values()))
    parts = []
    for t, names in lists.items():
        own = [n for n in names if _key(n) not in common]
        more = f", +{len(own) - _SHOWN}" if len(own) > _SHOWN else ""
        parts.append(f"{t}: {', '.join(map(str, own[:_SHOWN])) or '(none)'}{more}")
    return "; ".join(parts)


def merge_problems(task_dirs) -> list[str]:
    """Every reason the LeRobot line's plain merge would refuse these task directories, or the copy
    it built from them could not train (a task's statistics that do not have its features' declared
    shapes, axis_qvla.train.lerobot_stats); [] when they merge into a copy that trains."""
    dirs = [pathlib.Path(d) for d in task_dirs]
    unique = len({d.name for d in dirs}) == len(dirs)
    infos = {(d.name if unique else str(d)): read_info(d) for d in dirs}
    out = []
    for key in ("fps", "robot_type"):
        values = {t: info.get(key) for t, info in infos.items()}
        if len({_key(v) for v in values.values()}) > 1:
            out.append(f"{key} differs between the tasks ({values}); the tasks are not one dataset, so no "
                       f"plain copy holds them all")
    keys = sorted({k for info in infos.values() for k in info["features"]})
    for k in keys:
        have = {t: info["features"][k] for t, info in infos.items() if k in info["features"]}
        lacking = sorted(set(infos) - set(have))
        if lacking:
            out.append(f"feature {k!r}: {', '.join(lacking)} {'has' if len(lacking) == 1 else 'have'} none and "
                       f"{', '.join(sorted(have))} {'has' if len(have) == 1 else 'have'} it; the tasks are not one "
                       f"dataset, so no plain copy holds them all")
            continue
        if len({_key(merge_form(f)) for f in have.values()}) == 1:
            continue
        if len({_key(merge_form(f, names=False)) for f in have.values()}) == 1:
            out.append(f"feature {k!r}: its names differ between the tasks ({_names_shown(have)}), and the "
                       f"merge requires equal features. Names are per-task metadata no training consumer reads "
                       f"(observation.body_pose names each scene's own objects): merge copies of the task "
                       f"directories (meta/ and data/ copied, videos/ linked) whose differing names are cleared "
                       f"with axis_qvla.train.plain_copy.clear_differing_names, as the smoke fixture does; never edit "
                       f"the published task directories")
        else:
            groups: dict[str, list[str]] = {}
            for t, f in have.items():
                groups.setdefault(_key({x: f.get(x) for x in ("dtype", "shape")}), []).append(t)
            what = "dtype or shape" if len(groups) > 1 else "info"
            out.append(f"feature {k!r}: its {what} differs between the tasks "
                       f"({' | '.join(', '.join(ts) for ts in groups.values())}); the tasks are not one dataset, "
                       f"so no plain copy holds them all")
    for d in dirs:
        out += stats_problems(d)
    return out


def clear_differing_names(task_dirs) -> list[str]:
    """Clear, in each task's meta/info.json, the `names` of every feature whose names differ between
    the tasks; returns those features. It rewrites the files in place, so it is for copies of the
    task directories that this machine owns, never for a mirrored repository's."""
    infos = {pathlib.Path(d): read_info(d) for d in task_dirs}
    keys = sorted({k for info in infos.values() for k in info["features"]})
    differ = [k for k in keys
              if len({_key(info["features"].get(k, {}).get("names")) for info in infos.values()}) > 1]
    for d, info in infos.items():
        for k in differ:
            if k in info["features"]:
                info["features"][k]["names"] = None
        if differ:
            (d / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    return differ


def roots_task_dirs(roots_file, corpus_base) -> list[pathlib.Path]:
    """The task directories a v2 roots index names, in ascending task id (the merged copy's order)."""
    doc = json.loads(pathlib.Path(roots_file).read_text(encoding="utf-8"))
    roots = doc.get("roots") if isinstance(doc, dict) else None
    if not isinstance(roots, dict):
        raise ValueError(f"{roots_file} is not a v2 roots index (no 'roots' mapping)")
    return [pathlib.Path(corpus_base) / rel for _, rel in sorted(roots.items(), key=lambda kv: int(kv[0]))]


def rowspace_problems(dirs) -> list[str]:
    """For binding: why a plain copy of a row space's roots (`dirs`, axis_qvla.train.ids.rowspace_dirs)
    cannot be merged; [] when it can."""
    try:
        problems = merge_problems(roots_task_dirs(dirs["roots"], dirs["corpus_base"]))
    except (OSError, ValueError) as exc:
        return [f"rowspace: the plain copy's inputs cannot be checked: {exc}"]
    return [f"rowspace: the pinned LeRobot fork refuses a plain merge of the tasks in {dirs['roots']}: {p}"
            for p in problems]


PLAIN_RECORD = "axis_plain.json"     # what `qvla train` writes into a plain copy it merged
PLAIN_SCHEMA = "qvla.plain_copy/1"


def write_record(dirs) -> pathlib.Path:
    """Record, in the plain copy `dirs["lerobot_dataset"]` just merged, the sha256 of the roots index
    (`dirs["roots"]`) its task directories were read from."""
    from axis_qvla.core.canon import dumps_canonical, sha256_file

    dest = pathlib.Path(dirs["lerobot_dataset"]) / PLAIN_RECORD
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_bytes(dumps_canonical({"schema": PLAIN_SCHEMA, "roots_sha256": sha256_file(dirs["roots"])}))
    tmp.replace(dest)
    return dest


def record_problems(dirs) -> list[str]:
    """For binding: why the plain copy that is there was not merged from the roots index on disk; []
    when its record names that index's sha256, or when it has no record (built by hand)."""
    from axis_qvla.core.canon import sha256_file

    copy, roots = pathlib.Path(dirs["lerobot_dataset"]), pathlib.Path(dirs["roots"])
    path = copy / PLAIN_RECORD
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, NotADirectoryError):
        return []
    except (OSError, ValueError) as exc:
        return [f"rowspace: cannot read {path} ({exc.__class__.__name__})"]
    want = doc.get("roots_sha256") if isinstance(doc, dict) and doc.get("schema") == PLAIN_SCHEMA else None
    if not isinstance(want, str):
        return [f"rowspace: {path} is not a {PLAIN_SCHEMA} record"]
    try:
        got = sha256_file(roots)
    except OSError as exc:
        return [f"rowspace: cannot read {roots} ({exc.__class__.__name__})"]
    if got == want:
        return []
    return [f"rowspace: {copy} was merged from a roots index with sha256 {want[:12]}, but {roots} is now "
            f"{got[:12]}, so the copy holds another task set; remove {copy} yourself (qvla train never deletes "
            f"a directory) and train again, which merges it from the current index"]


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="qvla train merge-check",
                                description="check that a plain merge accepts these task directories")
    p.add_argument("tasks", nargs="+", type=pathlib.Path, help="the task_<id> directories to merge")
    a = p.parse_args(argv)
    try:
        problems = merge_problems(a.tasks)
    except (OSError, ValueError) as exc:
        print(f"FAIL  {exc}")
        return 2
    for problem in problems:
        print(f"FAIL  {problem}")
    if problems:
        return 2
    print(f"OK    {len(a.tasks)} task directories have equal features; a plain merge accepts them")
    return 0
