"""The traces sample: the raw recordings behind 21 pretraining tasks, so the label pipeline can run.

The full trace archive is not published; this sample is. `select` groups the pretraining tasks that have
episodes by the first word of their instruction, lower-cased (the verb family: put, rotate,
rearrange, ..., squeegee/wipe), takes one task per family -- the task with the fewest pretraining
episodes, ties to the smaller id -- and then adds every required task not chosen yet (the tasks the
renderer tests read). A required task never displaces a family's pick, and must have episodes.
Every pretraining episode of a chosen task is in the sample, with its recorded trace.

`stage` copies, per chosen task: its episode metadata (episodes, tasks, info) and data parquet (the
published per-frame labels the golden test compares with; no video) from the PUBLISHED copy of the
pretraining data, each file checked against the published tree manifest; one recorded trace per episode; the
task checkers' goal record of each episode's attempt (machine paths removed); and the task's checker
configuration, which both the goal extraction and the reward read (a position-delta goal leaf
without per-axis bounds takes them from it). No scene and no attempt record is staged.

Nothing identifying is staged: the whole tree, `manifest.json` (every file's sha256) included, must
pass `axis_qvla.labels.privacy.check_tree` -- a closed check of every key, column, metadata entry and
string, plus value patterns -- before anything is written. `stage` writes nothing outside `out`,
refuses an `out` that is not empty, and removes `out` when writing fails (a failed removal is
reported with the error); inside a published qvla-v1-corpus tree `out` must be its traces_sample/.

Layout (traces/ beside data/ is what the reward's checker-configuration lookup expects):
    traces/task_<id>_attempt_<aid>.jsonl    goals_per_attempt/task_<id>_attempt_<aid>.json
    data/task_<id>/checker_config.json      corpus/task_<id>/{meta,data}/...    manifest.json

Offline environment: pandas, pyarrow.

    python -m axis_qvla.labels.traces_sample select --episodes <dir> --roots <roots index> [--require <id> ...]
    python -m axis_qvla.labels.traces_sample stage --episodes <dir> --roots <roots index> --survey <dir> \\
        --traces <dir> --published <dir> --expect <tree manifest> --out <dir> [--require <id> ...]
    python -m axis_qvla.labels.traces_sample scan <staged dir>
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pathlib
import shutil
from collections.abc import Mapping, Sequence

from axis_qvla.core.canon import dumps_canonical
from axis_qvla.labels.privacy import RULE, SCHEMA, check_tree, scan, scan_file, scan_tree, verb_family  # noqa: F401
from axis_qvla.signals.staging import check_out

GOAL_KEYS = ("attempt_id", "n_steps", "top_first_pass_frame", "leaves")
_META_FILES = ("info.json", "tasks.parquet")


def families(tasks: Mapping[int, tuple[str, Sequence]]) -> list[str]:
    """The verb families of the tasks that have corpus episodes, sorted."""
    return sorted({verb_family(prompt) for prompt, episodes in tasks.values() if episodes})


def _kept(rel: pathlib.PurePath) -> bool:
    """The corpus files the sample carries: data/, meta/episodes/, meta/info.json, meta/tasks.parquet."""
    parts = rel.parts
    return parts[0] == "data" or (parts[0] == "meta" and len(parts) > 1
                                  and (parts[1] == "episodes" or rel.name in _META_FILES))


def task_episodes(task_dir) -> list[tuple[int, int, int]]:
    """[(episode_index, attempt_id, length), ...] of one task directory, sorted."""
    import pandas as pd

    files = sorted((pathlib.Path(task_dir) / "meta" / "episodes").rglob("*.parquet"))
    # use_threads=False: on pyarrow 25, threaded reads of all 731 episode tables abort the
    # interpreter at exit ("terminate called without an active exception", status 134).
    ep = pd.concat([pd.read_parquet(f, columns=["episode_index", "attempt_id", "length"], use_threads=False)
                    for f in files])
    return sorted((int(e), int(a), int(n)) for e, a, n in zip(ep["episode_index"], ep["attempt_id"], ep["length"]))


def corpus_tasks(corpus, roots_file) -> dict[int, tuple[str, list[tuple[int, int, int]]]]:
    """task id -> (instruction, [(episode_index, attempt_id, length), ...]) for every task of a roots index."""
    from axis_qvla.signals.methods.cfg_tags import task_prompts

    roots = json.loads(pathlib.Path(roots_file).read_text())["roots"]
    out = {}
    for tid, rel in sorted(roots.items(), key=lambda kv: int(kv[0])):
        task_dir = pathlib.Path(corpus) / rel
        prompts = sorted(task_prompts(task_dir))
        if len(prompts) != 1:
            raise ValueError(f"task {tid} has {len(prompts)} instructions; a verb family needs exactly one")
        out[int(tid)] = (prompts[0], task_episodes(task_dir))
    return out


def select(tasks: Mapping[int, tuple[str, Sequence]], *, require: Sequence[int] = ()) -> list[tuple[str, int]]:
    """[(family, task id), ...]: one pick per family in family order, then the required tasks not picked."""
    required = {int(t) for t in require}
    unknown = sorted(required - set(tasks))
    if unknown:
        raise ValueError(f"required tasks {unknown} are not pretraining tasks")
    empty = sorted(t for t in required if not tasks[t][1])
    if empty:
        raise ValueError(f"required tasks {empty} have no pretraining episodes, so no trace of theirs is in the pretraining data")
    by_family: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    for tid, (prompt, episodes) in tasks.items():
        if episodes:
            by_family[verb_family(prompt)].append((len(episodes), int(tid)))
    chosen = [(fam, sorted(by_family[fam])[0][1]) for fam in sorted(by_family)]
    picked = {t for _f, t in chosen}
    chosen += [(verb_family(tasks[t][0]), t) for t in sorted(required - picked)]
    return chosen


def _require_empty(out: pathlib.Path) -> None:
    try:
        if os.listdir(out):
            raise ValueError(f"{out} is not empty: stage into a new or empty directory")
    except FileNotFoundError:
        pass


def _write(out: pathlib.Path, files: Mapping[str, bytes]) -> None:
    """Write every file under `out`; on any failure remove `out` (absent or empty before), and say so
    with the error if that removal fails too."""
    try:
        for rel, data in files.items():
            p = out / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
    except BaseException as exc:
        try:
            shutil.rmtree(out)
        except FileNotFoundError:
            pass
        except OSError as cleanup:
            exc.add_note(f"removing the partly written {out} failed too ({cleanup}); remove it before staging again")
        raise


def stage(*, corpus, roots_file, survey, traces, out, published, expect: Mapping[str, str],
          require: Sequence[int] = ()) -> dict:
    """Select on `corpus` and stage the chosen tasks from `published`, every corpus file of which
    must be listed in `expect` (published path relative to the corpus root -> sha256) with its bytes."""
    check_out(out, "traces_sample")
    out = pathlib.Path(out)
    _require_empty(out)
    tasks = corpus_tasks(corpus, roots_file)
    chosen = select(tasks, require=require)
    roots = json.loads(pathlib.Path(roots_file).read_text())["roots"]
    staged: dict[str, bytes] = {}
    entries = []
    for family, tid in chosen:
        prompt, episodes = tasks[tid]
        rel_root = roots[str(tid)]
        task_dir = pathlib.Path(published) / rel_root
        if task_episodes(task_dir) != episodes:
            raise ValueError(f"task {tid}: the published copy lists other episodes than the selection copy")
        listed = {k for k in expect if k.startswith(f"{rel_root}/") and _kept(pathlib.PurePosixPath(k[len(rel_root) + 1:]))}
        seen = set()
        for f in sorted(task_dir.rglob("*")):
            rel = f.relative_to(task_dir)
            if not f.is_file() or not _kept(rel):
                continue
            key = f"{rel_root}/{rel.as_posix()}"
            data = f.read_bytes()
            if expect.get(key) != hashlib.sha256(data).hexdigest():
                raise ValueError(f"{f} is not the published {key}: stage from the published copy of the pretraining data")
            staged[f"corpus/task_{tid}/{rel.as_posix()}"] = data
            seen.add(key)
        if seen != listed:
            raise ValueError(f"task {tid}: the published copy lacks {sorted(listed - seen)[:5]}")
        staged[f"data/task_{tid}/checker_config.json"] = (
            pathlib.Path(survey) / "data" / f"task_{tid}" / "checker_config.json").read_bytes()
        for _ep, aid, _n in episodes:
            trace = pathlib.Path(traces) / f"task_{tid}_attempt_{aid}.jsonl"
            if not trace.is_file():
                raise FileNotFoundError(f"{trace} is missing: restore task_{tid}'s traces from the trace archive first")
            staged[f"traces/{trace.name}"] = trace.read_bytes()
            goals = json.loads((pathlib.Path(survey) / "goals_per_attempt" / f"task_{tid}_attempt_{aid}.json").read_text())
            clean = {k: goals[k] for k in GOAL_KEYS if k in goals}
            staged[f"goals_per_attempt/task_{tid}_attempt_{aid}.json"] = (json.dumps(clean, indent=2) + "\n").encode()
        entries.append({"task_id": tid, "family": family, "prompt": prompt,
                        "episodes": [{"episode_index": e, "attempt_id": a, "length": n} for e, a, n in episodes]})
    manifest = {"schema": SCHEMA, "rule": RULE, "families": families(tasks), "required": sorted(int(t) for t in require),
                "tasks": entries, "files": {rel: hashlib.sha256(data).hexdigest() for rel, data in sorted(staged.items())}}
    staged["manifest.json"] = dumps_canonical(manifest)
    findings = check_tree(staged)
    if findings:
        raise ValueError(f"refusing to stage: {len(findings)} finding(s), first {findings[:8]}")
    _write(out, staged)
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="axis_qvla.labels.traces_sample", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("select", "stage"):
        p = sub.add_parser(name)
        p.add_argument("--episodes", dest="corpus", metavar="DIR", type=pathlib.Path, required=True,
                       help="the pretraining episodes the selection reads")
        p.add_argument("--roots", type=pathlib.Path, required=True)
        p.add_argument("--require", type=int, action="append", default=[])
    st = sub.choices["stage"]
    st.add_argument("--survey", type=pathlib.Path, required=True, help="goals_per_attempt/ and data/task_*/checker_config.json")
    st.add_argument("--traces", type=pathlib.Path, required=True, help="the restored task_<id>_attempt_<aid>.jsonl traces")
    st.add_argument("--published", type=pathlib.Path, required=True,
                    help="the published copy of (at least) the chosen tasks, laid out as the episodes directory")
    st.add_argument("--expect", type=pathlib.Path, required=True,
                    help="the published tree manifest (qvla.tree_manifest/1) the files must match")
    st.add_argument("--out", type=pathlib.Path, required=True)
    sub.add_parser("scan", help="every finding of the privacy check over a staged tree (exit 1 if any)") \
        .add_argument("root", type=pathlib.Path)
    a = ap.parse_args(argv)
    if a.cmd == "select":
        for family, tid in select(corpus_tasks(a.corpus, a.roots), require=a.require):
            print(f"{family}\t{tid}")
        return 0
    if a.cmd == "scan":
        found = scan_tree(a.root)
        for line in found:
            print(line)
        n = sum(len(f) for _d, _s, f in os.walk(a.root))
        print(f"{len(found)} finding(s) in {n} files under {a.root}")
        return 1 if found else 0
    doc = json.loads(a.expect.read_text())
    if doc.get("schema") != "qvla.tree_manifest/1":
        raise SystemExit(f"{a.expect} is not a qvla.tree_manifest/1 file")
    expect = {rel: entry["sha256"] for rel, entry in doc["files"].items()}
    man = stage(corpus=a.corpus, roots_file=a.roots, survey=a.survey, traces=a.traces, out=a.out,
                published=a.published, expect=expect, require=a.require)
    print(f"staged {len(man['tasks'])} tasks, {len(man['files'])} files -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
