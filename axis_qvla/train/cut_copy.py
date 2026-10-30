"""Bind a merged LeRobot copy built on this machine to the inputs its row space names, and check
the files a pi0.5 finetune's plan is compiled from against the repository's pins.

A row space whose `lerobot_copy` is "cut" (conf/recipes/rowspaces.toml) trains SmolVLA and X-VLA
on a copy `axis_merge_datasets --ranges` builds here. Its frame count alone cannot tell it apart
from a copy cut by the same spans from other inputs, or with another action column: both have the
same rows. The copy's cut record (`axis_spans.json`) states what it was cut from, and every field
of it must be what the row space names:

- `ranges_sha256`: the sha256 of the row space's ranges file;
- `action_source`: the row space's declared action column (a copy whose action was read from
  `action_eef` has the same frames, and a SmolVLA finetune pads either width without error);
- `inputs`: the roots index's task directories, each once, in ascending task id (the order the
  copy was merged in), and each `info_sha256` the sha256 of that task's `meta/info.json`.

Every sha256 the record is compared with is the one the repository pins in its file list
(`BENCH_FILES.json`), and the file on disk is the reference only for a file the list pins no
sha256 for: a ranges file or `meta/info.json` edited in the local mirror does not move the
reference. A published repository's mirror without its file list (a partial fetch) is refused,
naming the list and how to fetch it, never read as "nothing pinned"; only a row space this machine
derived itself (`repo = "local"`) has no list. The task directories are read from the roots index on disk, so a pinned index must
have its pinned sha256. Binding never reads the record's `episodes` list: given the spans, the
inputs and the action column the cut is deterministic, and the build checked every episode
against its source rows.

`meta/info.json` fixes a task's counts, features and file layout, not its frame content, so the
input check catches another release, layout or column set, not a re-encode of the same shape;
frame content is what `qvla data verify` checks against the release lock.

A pi0.5 finetune compiled at bind time reads its rows from the same row space's roots index,
ranges file and each task's episode table, and records only the first two by digest, so
`pinned_input_problems` compares every one of those files with the repository's pin before the
compile reads it, by the same rule: a file the repository pins must have its pinned sha256, a
task whose meta/ the repository pins holds no episode file the repository does not list, and a
published repository's mirror must hold its file list.

Core tier: stdlib only.
"""
from __future__ import annotations

import json
import os
import pathlib
from collections.abc import Mapping

from axis_qvla.core.canon import sha256_file

CUT_RECORD = "axis_spans.json"     # what the LeRobot line's `axis_merge_datasets --ranges` writes beside meta/
REPO_FILES = "BENCH_FILES.json"    # a repository's own list of its files with their sha256s


def read_json(path: pathlib.Path) -> tuple[object, str | None]:
    """`(document, None)`, or `(None, why not)`: not there, unreadable (EACCES is named, never
    folded into absence) or not JSON."""
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (FileNotFoundError, NotADirectoryError):
        return None, "is not there"
    except OSError as exc:
        return None, f"cannot be read ({exc.__class__.__name__})"
    except ValueError:
        return None, "is not JSON"


def _digest(path: pathlib.Path) -> tuple[str | None, str | None]:
    try:
        return sha256_file(path), None
    except OSError as exc:
        return None, f"cannot read {path} ({exc.__class__.__name__})"


def _task_dirs(roots_path: pathlib.Path) -> tuple[list[tuple[str, str]], str | None]:
    """The roots index's `(directory name, entry)` pairs in ascending task id, or why not. An entry
    is relative to the row space's corpus base."""
    doc, why = read_json(roots_path)
    if why is not None:
        return [], f"{roots_path} {why}"
    entries = doc.get("roots", doc) if isinstance(doc, dict) else None
    if not isinstance(entries, dict) or not all(isinstance(v, str) for v in entries.values()) or \
            not all(str(k).isdigit() for k in entries):
        return [], f"{roots_path} is not a roots index ({{task id: directory}})"
    return [(pathlib.PurePosixPath(entries[k]).name, entries[k]) for k in sorted(entries, key=int)], None


def _pins(repo_root: pathlib.Path, *, published: bool) -> tuple[dict[str, str], str | None]:
    """The repository's own file list as {path: sha256}. A published repository's mirror must hold
    it (a fetch that left it out cannot be checked against the release); a local one has none."""
    doc, why = read_json(repo_root / REPO_FILES)
    if why == "is not there":
        if not published:
            return {}, None
        return {}, (f"{repo_root / REPO_FILES} is not there, so the files this run reads cannot be checked "
                    f"against the repository's pins (a partial fetch leaves it out); fetch it with them "
                    f"(`qvla data get bench --include {REPO_FILES}`)")
    if why is not None:
        return {}, f"{repo_root / REPO_FILES} {why}"
    rows = doc.get("files") if isinstance(doc, dict) else None
    if not isinstance(rows, list):
        return {}, f"{repo_root / REPO_FILES} lists no files"
    return {r["path"]: r["sha256"] for r in rows if isinstance(r, dict) and "path" in r and "sha256" in r}, None


def _pin(path: pathlib.Path, pins: Mapping[str, str], repo_root: pathlib.Path) -> tuple[str | None, str | None]:
    """`(path in the repository's file list, its pinned sha256)`, or `(None, None)` when the
    repository pins no sha256 for `path`."""
    rel = path.relative_to(repo_root).as_posix() if path.is_relative_to(repo_root) else None
    return (rel, pins[rel]) if rel in pins else (None, None)


def _pinned(path: pathlib.Path, pins: Mapping[str, str], repo_root: pathlib.Path) -> list[str]:
    """[] when the repository pins no sha256 for `path` or `path` has it, else why not."""
    rel, want = _pin(path, pins, repo_root)
    if want is None:
        return []
    got, why = _digest(path)
    if why is not None:
        return [f"rowspace: {why}"]
    if got != want:
        return [f"rowspace: {path} has sha256 {got[:12]}, but {repo_root / REPO_FILES} pins {rel} at {want[:12]}; "
                f"fetch it again"]
    return []


def _episode_files(directory: pathlib.Path) -> tuple[list[pathlib.Path], str | None]:
    """Every `*.parquet` under `directory` at any depth (what the episode table is read from), or why
    not. An unreadable directory is named, never read as empty."""
    out, todo = [], [directory]
    while todo:
        d = todo.pop()
        try:
            with os.scandir(d) as it:
                entries = [e for e in it if not e.name.startswith(".")]
        except FileNotFoundError:
            continue
        except OSError as exc:
            return [], f"cannot read {d} ({exc.__class__.__name__})"
        todo += [pathlib.Path(e.path) for e in entries if e.is_dir()]
        out += [pathlib.Path(e.path) for e in entries if e.name.endswith(".parquet") and not e.is_dir()]
    return sorted(out), None


def pinned_input_problems(rs: Mapping[str, pathlib.Path], repo_root: pathlib.Path, *,
                          published: bool) -> list[str]:
    """Every reason the files a pi0.5 finetune is compiled from are not the repository's (module
    docstring): the roots index and ranges file of `rs` (axis_qvla.train.ids.rowspace_dirs's mapping),
    then each task's pinned meta/ files and its episode files. [] when they are, or when a local
    repository (`published=False`) pins none of them."""
    pins, why = _pins(repo_root, published=published)
    if why is not None:
        return [f"rowspace: {why}"]
    if not pins:
        return []
    out = _pinned(rs["roots"], pins, repo_root) + _pinned(rs["ranges"], pins, repo_root)
    if out:
        return out
    tasks, why = _task_dirs(rs["roots"])
    if why is not None:
        return [f"rowspace: {why}"]
    for _, entry in tasks:
        meta = rs["corpus_base"] / entry / "meta"
        if not meta.is_relative_to(repo_root):
            continue
        prefix = meta.relative_to(repo_root).as_posix() + "/"
        listed = sorted(p for p in pins if p.startswith(prefix))
        for rel in listed:
            out += _pinned(repo_root / rel, pins, repo_root)
        if not listed:
            continue
        files, why = _episode_files(meta / "episodes")
        if why is not None:
            out.append(f"rowspace: {why}")
            continue
        extra = [f for f in files if f.relative_to(repo_root).as_posix() not in pins]
        if extra:
            out.append(f"rowspace: {meta / 'episodes'} holds {len(extra)} file(s) {repo_root / REPO_FILES} does not "
                       f"list (first: {extra[0].name}), and the episode table is every file there; fetch the task again")
    return out


def check_cut_copy(lay, rs: Mapping[str, pathlib.Path], repo_root: pathlib.Path) -> list[str]:
    """Every reason the copy at `rs["lerobot_dataset"]` is not the one row space `lay` (a "cut"
    RowSpaceLayout) names; [] when its cut record matches. `rs` is axis_qvla.train.ids.rowspace_dirs's
    mapping and `repo_root` the row space's repository on this machine."""
    copy = rs["lerobot_dataset"]
    record_path = copy / CUT_RECORD
    record, why = read_json(record_path)
    if why == "is not there":
        return [f"rowspace: {record_path} is not there: {copy} is not a copy cut by `axis_merge_datasets --ranges`; "
                "remove it and run `qvla train` again, which rebuilds it "
                "(docs/reference/data.md#the-merged-lerobot-copy)"]
    if why is not None:
        return [f"rowspace: {record_path} {why}"]
    if not isinstance(record, dict):
        return [f"rowspace: {record_path} is not a cut record"]
    pins, why = _pins(repo_root, published=lay.repo != "local")
    if why is not None:
        return [f"rowspace: {why}"]
    out = []
    cut = record.get("ranges_sha256")
    rel, want = _pin(rs["ranges"], pins, repo_root)
    why, where = None, f"{repo_root / REPO_FILES} pins {rel} at"
    if want is None:
        (want, why), where = _digest(rs["ranges"]), f"{rs['ranges']} has"
    if why is not None:
        out.append(f"rowspace: {why}")
    elif not isinstance(cut, str):
        out.append(f"rowspace: {record_path} records no ranges_sha256")
    elif cut != want:
        out.append(f"rowspace: {copy} was cut from a ranges file with sha256 {cut[:12]}, but {where} "
                   f"{want[:12]}; rebuild the copy from it")
    action = record.get("action_source")
    if action != lay.action_source:
        out.append(f"rowspace: {copy} reads its action from {action!r}, but frame index {lay.name} trains on "
                   f"{lay.action_source!r}; rebuild the copy with --action-source {lay.action_source}")
    out += _check_inputs(record.get("inputs"), record_path, copy, rs, repo_root, pins)
    return out


def _check_inputs(inputs: object, record_path: pathlib.Path, copy: pathlib.Path,
                  rs: Mapping[str, pathlib.Path], repo_root: pathlib.Path, pins: Mapping[str, str]) -> list[str]:
    if not (isinstance(inputs, list) and all(isinstance(i, dict) and isinstance(i.get("root"), str)
                                             and isinstance(i.get("info_sha256"), str) for i in inputs)):
        return [f"rowspace: {record_path} records no inputs [{{root, info_sha256}}, ...]"]
    _, pinned = _pin(rs["roots"], pins, repo_root)
    if pinned is not None:
        got, why = _digest(rs["roots"])
        if why is not None:
            return [f"rowspace: {why}"]
        if got != pinned:
            return [f"rowspace: {rs['roots']} has sha256 {got[:12]}, but {repo_root / REPO_FILES} pins it at "
                    f"{pinned[:12]}, so the task directories it lists are not the repository's; fetch it again"]
    tasks, why = _task_dirs(rs["roots"])
    if why is not None:
        return [f"rowspace: {why}"]
    got, want = [i["root"] for i in inputs], [name for name, _ in tasks]
    if got != want:
        missing, extra = sorted(set(want) - set(got)), sorted(set(got) - set(want))
        detail = (f"missing {missing}" if missing else "") + ("; " if missing and extra else "") + \
                 (f"not in it {extra}" if extra else "")
        return [f"rowspace: {copy} was merged from other task directories than {rs['roots']} lists "
                f"({detail or 'the same ones, in another order or twice'}); rebuild the copy from them"]
    out = []
    for (name, entry), i in zip(tasks, inputs, strict=True):
        info = rs["corpus_base"] / entry / "meta" / "info.json"
        _, expected = _pin(info, pins, repo_root)
        if expected is None:
            expected, why = _digest(info)
            if why is not None:
                out.append(f"rowspace: {why}")
                continue
        if i["info_sha256"] != expected:
            out.append(f"rowspace: {copy} was merged from a {name} whose meta/info.json has sha256 "
                       f"{i['info_sha256'][:12]}, but {info} is {expected[:12]}; rebuild the copy from it")
    return out
