"""SignalBundles for the training tests, built with the signal stream's own types and writer, and
the input files a bound pi0.5 finetune arm opens.

The one place the training tests construct axis_qvla.signals types; if their constructors change, this
file is the only one to follow.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np

from axis_qvla.core.canon import dumps_canonical, sha256_bytes
from axis_qvla.data.layout import BENCH, CORPUS, checkpoint_dest, default_dest, norm_stats_dest
from axis_qvla.signals.bundle import rowspace_id, write_bundle, write_rowspace
from axis_qvla.signals.rowspace import flat_rowspace
from axis_qvla.signals.schema import BARE, Budget, CondVocab, LegacyFile, SignalBundle


def ft_rowspace(n_rows: int = 100, name: str = "heldout24_ft"):
    return flat_rowspace(name=name, n_rows=n_rows, source="tests.unit.train")


def rowspace_ref(rs) -> str:
    """The ArmSpec form of a row space's id, `<name>#sha256:<id>`."""
    return f"{rs.name}#sha256:{rowspace_id(rs)}"


def lerobot_rows_legacy(rows, *, path: str = "signals/test/lerobot_rows.npz", lock_id: str | None = None) -> dict:
    """The legacy record of a LeRobot row plan (`lerobot_rows.npz`, key `rows`)."""
    return {"legacy": (LegacyFile("lerobot-legacy", lock_id, path, "npz",
                                  {"$o": [["rows", {"$array": "legacy.rows"}]]}),),
            "legacy_arrays": {"rows": np.asarray(rows, dtype=np.int64)}}


def make_bundle(*, plan, rowspace=None, weights=None, reducer=None, cond=None, codes=(5,), fields=("quality",),
                serve=(BARE,), requires=None, legacy=(), legacy_arrays=None, inputs=None, params=None) -> SignalBundle:
    """`params` is the program's; a flat row space's bundle records the finetune decode rule by default."""
    from axis_qvla.train.ft_bundle import FLAT_ROW_DECODE

    plan = np.asarray(plan, dtype=np.int64)
    rowspace = rowspace or ft_rowspace()
    vocab = None
    if cond is not None:
        vocab = CondVocab(tuple(fields), {f: {int(c): f"\nQuality: {c}" for c in codes} for f in fields},
                          ("pick up the bowl",))
    if requires is None:
        requires = {"plan"} | ({"loss_weights"} if weights is not None else set()) | (
            {"cond_prompt"} if cond is not None else set())
    if reducer is None:
        reducer = "mean" if weights is None else "global_ratio"
    if params is None:
        params = {"row_decode": FLAT_ROW_DECODE} if rowspace.kind == "flat" else {}
    program = {"method": "tests.unit.train", "version": 1, "params": dict(params)}
    return SignalBundle(program=program, program_sha=sha256_bytes(dumps_canonical(program)),
                        rowspace=rowspace, budget=Budget(*map(int, plan.shape)), seed=0, plan=plan,
                        weights=weights, reducer=reducer, cond=cond, vocab=vocab, serve=tuple(serve),
                        requires=frozenset(requires), legacy=tuple(legacy), legacy_arrays=dict(legacy_arrays or {}),
                        inputs=dict(inputs or {}))


def write_fixture_bundle(root: pathlib.Path, **kw) -> str:
    """Write `make_bundle(**kw)` under `root`; return the ArmSpec form of its id, `sha256:<hex>`."""
    root.mkdir(parents=True, exist_ok=True)
    return "sha256:" + write_bundle(make_bundle(**kw), root)


FT_ROOTS = "ft_demos/splits/qual_v2.roots.json"     # the heldout24_ft layout's roots index and ranges file
FT_RANGES = "ft_demos/splits/qual_v2.ranges.json"


def finetune_inputs(paths, roots_text: bytes = b'{"version": 2, "base": "../..", "roots": {}}',
                    ranges_text: bytes = b"{}") -> str:
    """The files the default finetune arm (tests.unit.train._arms.arm_toml) opens: its published
    init params, the heldout24_qual_v2 norm statistics, and the finetune roots index and ranges
    file. Returns the norm statistics' sha256, which an arm must name for them to bind."""
    (checkpoint_dest("axisrobotics/qvla-v1-pi05:pretrain/example/20604", paths) / "params").mkdir(parents=True)
    norm = norm_stats_dest("heldout24_qual_v2", paths) / "Devon018/Franka-Datasets-v2/norm_stats.json"
    norm.parent.mkdir(parents=True)
    norm.write_text("{}")
    roots = default_dest(BENCH, paths, FT_ROOTS)
    roots.parent.mkdir(parents=True, exist_ok=True)
    roots.write_bytes(roots_text)
    default_dest(BENCH, paths, FT_RANGES).write_bytes(ranges_text)
    return hashlib.sha256(b"{}").hexdigest()


def finetune_digests(paths) -> dict[str, str]:
    """The `inputs` a flat finetune bundle names: the sha256 of this layout's roots index and ranges file."""
    return {k: hashlib.sha256(default_dest(BENCH, paths, rel).read_bytes()).hexdigest()
            for k, rel in (("roots_index", FT_ROOTS), ("ranges", FT_RANGES))}


# Two finetune tasks of 40 and 30 frames under the heldout24_ft layout, cut by their ranges file to
# 28 + 25 = 53 trainable rows: (episode 0 of task 900, frames 2..29), (episode 0 of task 901, frames
# 0..9 then 15..29).
FT_SPANS = {"task_900--0": [[2, 30]], "task_901--0": [[0, 10], [15, 30]]}
FT_EPISODE = [0] * 53
FT_FRAME = list(range(2, 30)) + list(range(0, 10)) + list(range(15, 30))


def pin_bench(paths) -> pathlib.Path:
    """The bench repository's own file list (BENCH_FILES.json) pinning every file now under it at its
    sha256, as a release does; returns the list's path."""
    base = default_dest(BENCH, paths)
    files = sorted(p for p in base.rglob("*") if p.is_file() and p.name != "BENCH_FILES.json")
    rows = [{"path": p.relative_to(base).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in files]
    (base / "BENCH_FILES.json").write_text(json.dumps({"schema": "test", "files": rows}))
    return base / "BENCH_FILES.json"


def finetune_corpus(paths, lengths=(40, 30), spans=FT_SPANS, n_rows=53, *, pin=True):
    """The two finetune tasks, their roots index and ranges file, and the stored flat row space they
    make; with `pin`, the bench's file list pinning every file under it, as a release publishes it."""
    import pandas as pd

    base = default_dest(BENCH, paths)
    roots = {}
    for i, n in enumerate(lengths):
        meta = base / f"ft_task_{900 + i}" / "meta"
        (meta / "episodes").mkdir(parents=True)
        pd.DataFrame({"episode_index": [0], "dataset_from_index": [0], "length": [n]}).to_parquet(
            meta / "episodes" / "file-000.parquet")
        pd.DataFrame({"task_index": [0]}, index=pd.Index([f"put block {i} away"], name="task")).to_parquet(
            meta / "tasks.parquet")
        roots[str(900 + i)] = f"ft_task_{900 + i}"
    index = base / FT_ROOTS
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps({"version": 2, "base": "../..", "roots": roots}))
    (base / FT_RANGES).write_text(json.dumps(spans))
    rs = ft_rowspace(n_rows=n_rows)
    write_rowspace(rs, default_dest(CORPUS, paths, "bundles/rowspaces"))    # the signal compiler's staged row space
    if pin:
        pin_bench(paths)
    return rs


def smoke_shaped_rowspace():
    """A corpus row space shaped like the smoke fixture's: tasks 2013 and 2232, three episodes each,
    609 rows (333 + 276), every episode cut to [5, length - 5): 549 trainable rows, more than one
    global batch of 64, which openpi's row sampler needs to draw a plan at all (it drops an epoch's
    last partial batch)."""
    import hashlib as _hashlib

    from axis_qvla.signals.schema import RowSpace

    task = np.array([2013] * 3 + [2232] * 3, dtype=np.int64)
    length = np.array([111] * 3 + [92] * 3, dtype=np.int64)
    attempt = np.arange(6, dtype=np.int64)
    rows = sorted(zip(task.tolist(), [0, 1, 2] * 2, length.tolist(), attempt.tolist()))
    fp = "sha256:" + _hashlib.sha256("".join("%d:%d:%d:%d\n" % r for r in rows).encode()).hexdigest()[:32]
    return RowSpace(
        name="smoke_2x3", kind="corpus", n_rows=609, n_trainable=549, corpus_fingerprint=fp,
        source="tests.unit.train", roots={"$o": [["version", 2], ["base", "../corpus_d8"],
                                                 ["roots", {"$o": [["2013", "task_2013"], ["2232", "task_2232"]]}]]},
        roots_path="index/roots.json", roots_codec="json-indent1", ranges_path="index/ranges.json",
        ranges_codec="json", prompts=("pick up the bowl", "put the block away"), idle_filter={"function": "test"},
        arrays={"episode_task": task, "episode_index": np.array([0, 1, 2] * 2, dtype=np.int64),
                "episode_from": np.concatenate([[0], np.cumsum(length)[:-1]]).astype(np.int64),
                "episode_length": length, "episode_attempt": attempt,
                "ranges_entry": np.arange(6, dtype=np.int64), "ranges_counts": np.ones(6, dtype=np.int64),
                "ranges_spans": np.array([[5, int(n) - 5] for n in length], dtype=np.int64)})
