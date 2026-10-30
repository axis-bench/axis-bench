"""Row spaces: the flat row index every signal of one corpus addresses, and how it was cut.

A corpus row space concatenates the per-task LeRobot datasets in task-id order, compared as
integers (task_800 before task_1000), and numbers every frame `episode_from + frame`, where
`episode_from` is the length of every earlier task plus the episode's `dataset_from_index`. Its
trainable rows are the frames inside the half-open spans of a ranges file, clamped to the episode;
an episode the ranges file does not name contributes nothing, which is how a subset is cut. A flat
row space is 0..n-1 with every row trainable: a finetuning set addressed by frame.

`build_rowspace` reads only corpus metadata (meta/episodes, meta/tasks) and the two index files,
and refuses index files it cannot write back byte for byte, so a row space always carries the
exact files the paper's loaders read. `episode_rows` is the row-space twin of the schedule
builder's corpus walk (axis_qvla.signals.methods.schedule.episode_rows): the same blocks in the same
order, without a second read of the corpus.

Offline tier: numpy, pandas.
"""
from __future__ import annotations

import dataclasses
import glob
import hashlib
import json
import pathlib
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np

from axis_qvla.data.index.roots import HELDOUT_EXCLUDED
from axis_qvla.labels.idle_filter import IDLE_FILTER
from axis_qvla.signals import legacy, otree
from axis_qvla.signals.blocks.awr import episode_key
from axis_qvla.signals.methods.schedule import _frames_in_spans
from axis_qvla.signals.methods.scaling_subsets import scaling_subsets
from axis_qvla.signals.schema import LegacyFile, RowSpace

PAPER_ROOTS = "index/roots_libero_5k_v2_d8.json"
PAPER_RANGES = "index/ranges_libero_5k_v2.json"
FT_FRAMES = 86_912


class RowSpaceError(ValueError):
    """A row space cannot be built, cut or written back as asked."""


def _task_episodes(task_dir: pathlib.Path) -> tuple[list[tuple[int, int, int, int]], int]:
    """`(episode_index, dataset_from_index, length, attempt_id)` per episode, by episode index, and the
    task's row count -- what the schedule builder reads (methods/schedule.py `_episode_offsets`)."""
    import pandas as pd

    files = sorted(glob.glob(str(task_dir / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no meta/episodes parquet under {task_dir}")
    ep = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    attempt = ep["attempt_id"] if "attempt_id" in ep.columns else [-1] * len(ep)
    rows = sorted((int(e), int(f), int(n), int(a)) for e, f, n, a in
                  zip(ep["episode_index"], ep["dataset_from_index"], ep["length"], attempt, strict=True))
    return rows, int(ep["length"].sum())


def corpus_fingerprint(task, episode, length, attempt) -> str:
    """axis_qvla.signals.provenance.corpus_fingerprint, from a row space's episode table."""
    h = hashlib.sha256()
    for r in sorted(zip(map(int, task), map(int, episode), map(int, length), map(int, attempt))):
        h.update(("%d:%d:%d:%d\n" % r).encode())
    return f"sha256:{h.hexdigest()[:32]}"


def _render(path: str, codec: str, doc, arrays: Mapping) -> bytes:
    return legacy.render(LegacyFile("openpi-legacy", None, path, codec, doc), arrays)


def _fit_codec(data: bytes, path: str, doc, arrays: Mapping, codecs: Sequence[str]) -> str:
    for codec in codecs:
        if _render(path, codec, doc, arrays) == data:
            return codec
    raise RowSpaceError(f"{path}: none of {tuple(codecs)} writes these bytes back; a frame index records "
                        f"only index files it can reproduce exactly")


def _spans_by_episode(arrays: Mapping) -> dict[int, np.ndarray]:
    off = np.concatenate([[0], np.cumsum(arrays["ranges_counts"])])
    spans = np.asarray(arrays["ranges_spans"]).reshape(-1, 2)
    return {int(e): spans[off[i]:off[i + 1]] for i, e in enumerate(np.asarray(arrays["ranges_entry"]).tolist())}


def _count_trainable(arrays: Mapping) -> int:
    spans = _spans_by_episode(arrays)
    return int(sum(_frames_in_spans(spans.get(i, ()), int(n)).size
                   for i, n in enumerate(np.asarray(arrays["episode_length"]).tolist())))


def _ranges_doc(keys: Sequence[str]) -> dict:
    return {"$spans": {"keys": list(keys), "counts": "rowspace.ranges_counts", "spans": "rowspace.ranges_spans"}}


def _prefixed(rs: RowSpace) -> dict:
    return {f"rowspace.{k}": v for k, v in rs.arrays.items()}


def build_rowspace(*, name: str, corpus, roots_file, ranges_file, roots_path: str, ranges_path: str,
                   source: str) -> RowSpace:
    """The row space of a corpus directory under a roots index and a ranges file (see module doc)."""
    from axis_qvla.signals.methods.cfg_tags import task_prompts

    corpus = pathlib.Path(corpus)
    roots_bytes_ = pathlib.Path(roots_file).read_bytes()
    ranges_bytes_ = pathlib.Path(ranges_file).read_bytes()
    roots_doc = json.loads(roots_bytes_)
    if not (isinstance(roots_doc, dict) and isinstance(roots_doc.get("roots"), dict)):
        raise RowSpaceError(f"{roots_file} is not a v2 roots index (no 'roots' mapping)")
    roots_tree = otree.encode(roots_doc)
    roots_codec = _fit_codec(roots_bytes_, roots_path, roots_tree, {}, ("json-indent2-nl", "json-indent1", "json"))
    cols: dict[str, list[int]] = {k: [] for k in ("episode_task", "episode_index", "episode_from",
                                                  "episode_length", "episode_attempt")}
    prompts: set[str] = set()
    base = 0
    for tid, rel in sorted(roots_doc["roots"].items(), key=lambda kv: int(kv[0])):
        task_dir = corpus / rel
        rows, total = _task_episodes(task_dir)
        for ep, dfi, n, attempt in rows:
            cols["episode_task"].append(int(tid))
            cols["episode_index"].append(ep)
            cols["episode_from"].append(base + dfi)
            cols["episode_length"].append(n)
            cols["episode_attempt"].append(attempt)
        prompts |= task_prompts(task_dir)
        base += total
    index = {episode_key(t, e): i for i, (t, e) in enumerate(zip(cols["episode_task"], cols["episode_index"]))}
    ranges = json.loads(ranges_bytes_)
    unknown = [k for k in ranges if k not in index]
    if unknown:
        raise RowSpaceError(f"{ranges_file}: {len(unknown)} keys name no pretraining episode (first: {unknown[0]})")
    keys = list(ranges)
    arrays = {k: np.asarray(v, dtype=np.int64) for k, v in cols.items()}
    arrays["ranges_entry"] = np.asarray([index[k] for k in keys], dtype=np.int64)
    arrays["ranges_counts"] = np.asarray([len(ranges[k]) for k in keys], dtype=np.int64)
    arrays["ranges_spans"] = np.asarray([s for k in keys for s in ranges[k]], dtype=np.int64).reshape(-1, 2)
    ranges_codec = _fit_codec(ranges_bytes_, ranges_path, _ranges_doc(keys),
                              {f"rowspace.{k}": v for k, v in arrays.items()}, ("json", "json-indent1"))
    return RowSpace(
        name=name, kind="corpus", n_rows=base, n_trainable=_count_trainable(arrays),
        corpus_fingerprint=corpus_fingerprint(cols["episode_task"], cols["episode_index"],
                                              cols["episode_length"], cols["episode_attempt"]),
        source=source, roots=roots_tree, roots_path=roots_path, roots_codec=roots_codec,
        ranges_path=ranges_path, ranges_codec=ranges_codec, prompts=tuple(sorted(prompts)),
        idle_filter=dict(IDLE_FILTER), heldout_excluded=tuple(HELDOUT_EXCLUDED), arrays=arrays)


def flat_rowspace(*, name: str, n_rows: int, source: str) -> RowSpace:
    return RowSpace(name=name, kind="flat", n_rows=int(n_rows), n_trainable=int(n_rows),
                    corpus_fingerprint=None, source=source)


def episode_keys(rs: RowSpace) -> list[str]:
    a = rs.arrays
    return [episode_key(t, e) for t, e in zip(np.asarray(a["episode_task"]).tolist(), np.asarray(a["episode_index"]).tolist())]


def ranges_keys(rs: RowSpace) -> list[str]:
    keys = episode_keys(rs)
    return [keys[i] for i in np.asarray(rs.arrays["ranges_entry"]).tolist()]


def ranges_doc(rs: RowSpace) -> dict:
    """The legacy-codec tree of the row space's ranges file (array leaves named `rowspace.*`)."""
    return _ranges_doc(ranges_keys(rs))


def ranges_mapping(rs: RowSpace) -> dict[str, list[list[int]]]:
    off = np.concatenate([[0], np.cumsum(rs.arrays["ranges_counts"])])
    spans = np.asarray(rs.arrays["ranges_spans"]).reshape(-1, 2)
    return {k: spans[off[i]:off[i + 1]].tolist() for i, k in enumerate(ranges_keys(rs))}


def roots_bytes(rs: RowSpace) -> bytes:
    return _render(rs.roots_path, rs.roots_codec, rs.roots, {})


def roots_v2_from_d8(d8: bytes) -> bytes:
    """The pretraining-data-base roots index the d8 index was derived from: the derivation changes
    only `base` and adds two keys (`derived_from`, `action_space`), and the source index was written with indent=1."""
    d = json.loads(d8)
    return json.dumps({"version": d["version"], "base": "../corpus", "roots": d["roots"]}, indent=1).encode()


def roots_v2_bytes(rs: RowSpace) -> bytes:
    """The pretraining-data-base roots index the d8 index was derived from (the bytes the v2 builders hashed)."""
    if "derived_from" not in otree.decode(rs.roots):
        raise RowSpaceError(f"{rs.name}: the roots index is not a d8 index derived from a pretraining-data-base one")
    return roots_v2_from_d8(roots_bytes(rs))


def ranges_bytes(rs: RowSpace) -> bytes:
    return _render(rs.ranges_path, rs.ranges_codec, ranges_doc(rs), _prefixed(rs))


def roots_sha256(rs: RowSpace, form: Literal["d8", "v2"]) -> str:
    return hashlib.sha256(roots_v2_bytes(rs) if form == "v2" else roots_bytes(rs)).hexdigest()


def ranges_sha256(rs: RowSpace) -> str:
    return hashlib.sha256(ranges_bytes(rs)).hexdigest()


def trainable_blocks(rs: RowSpace) -> list[tuple[str, np.ndarray, np.ndarray, int]]:
    """`[(episode_key, global_rows, frames, length), ...]` for every episode with a trainable frame,
    in walk order (task id as an int, then episode index)."""
    if rs.kind != "corpus":
        raise RowSpaceError(f"{rs.name} is a flat frame index: it has no episodes")
    a = rs.arrays
    spans = _spans_by_episode(a)
    out = []
    lengths = np.asarray(a["episode_length"]).tolist()
    starts = np.asarray(a["episode_from"]).tolist()
    keys = episode_keys(rs)
    for i, n in enumerate(lengths):
        frames = _frames_in_spans(spans.get(i, ()), int(n))
        if frames.size:
            out.append((keys[i], int(starts[i]) + frames, frames, int(n)))
    return out


def trainable_rows(rs: RowSpace) -> np.ndarray:
    if rs.kind == "flat":
        return np.arange(rs.n_rows, dtype=np.int64)
    rows = np.concatenate([b[1] for b in trainable_blocks(rs)]).astype(np.int64)
    if rows.size > 1 and not bool((np.diff(rows) > 0).all()):
        raise RowSpaceError(f"{rs.name}: spans overlap or are out of order, so a row would be planned twice")
    return rows


def episode_rows(rs: RowSpace, stats: Mapping, *, drop: bool):
    """`(blocks, progress, unlabelled, n_trainable, n_rows, row_w)`: the contract of
    axis_qvla.signals.methods.schedule.episode_rows, walked over the row space instead of the corpus."""
    blocks, row_w, progress, unlabelled = [], [], [], []
    n_trainable = 0
    for key, rows, frames, length in trainable_blocks(rs):
        st = stats.get(key)
        if st is None:
            raise ValueError(f"{key} has trainable rows but no AWR weights. Every sampleable episode must "
                             f"carry a weight, even the neutral 1.0.")
        if st["n"] != length:
            raise ValueError(f"{key}: weights have {st['n']} entries but the episode has {length} rows. "
                             f"The weights artifact was built against different pretraining data.")
        n_trainable += int(frames.size)
        keep = st["keep_mask"][frames]
        blocks.append(rows[keep] if drop else rows)
        w_ep = st["w"][frames]
        row_w.append(w_ep[keep] if drop else w_ep)
        progress.append(float(st["progress_mask"][frames].mean()))
        unlabelled.append(bool(st["unlabelled"]))
    return (blocks, np.asarray(progress, dtype=np.float64), np.asarray(unlabelled, dtype=bool),
            n_trainable, rs.n_rows, row_w)


def block_task_ids(blocks) -> np.ndarray:
    """The task id of each `trainable_blocks` entry."""
    return np.asarray([int(key.split("--")[0].removeprefix("task_")) for key, *_ in blocks], dtype=np.int64)


def subset(rs: RowSpace, keys, *, name: str, ranges_path: str, ranges_codec: str, source: str) -> RowSpace:
    """The row space restricted to the ranges entries `keys`, in the source file's key order."""
    keep = set(keys)
    all_keys = ranges_keys(rs)
    missing = keep - set(all_keys)
    if missing:
        raise RowSpaceError(f"{len(missing)} subset keys are not ranges entries of {rs.name}")
    a = rs.arrays
    off = np.concatenate([[0], np.cumsum(a["ranges_counts"])])
    sel = [i for i, k in enumerate(all_keys) if k in keep]
    arrays = dict(a)
    arrays["ranges_entry"] = np.asarray(a["ranges_entry"])[sel]
    arrays["ranges_counts"] = np.asarray(a["ranges_counts"])[sel]
    spans = np.asarray(a["ranges_spans"]).reshape(-1, 2)
    arrays["ranges_spans"] = (np.concatenate([spans[off[i]:off[i + 1]] for i in sel])
                              if sel else np.empty((0, 2), dtype=np.int64)).astype(np.int64)
    return dataclasses.replace(rs, name=name, n_trainable=_count_trainable(arrays), ranges_path=ranges_path,
                               ranges_codec=ranges_codec, source=source, arrays=arrays)


def scaling_subset(rs: RowSpace, size: int, *, seed: int = 0) -> RowSpace:
    """The paper's 1K/2.5K scaling cut: a nested prefix of one seeded permutation of the ranges keys
    (axis_qvla.signals.methods.scaling_subsets), written as `index/subsets/scaling_<size>.ranges.json`."""
    keys = scaling_subsets(ranges_keys(rs), [int(size)], int(seed))[int(size)]
    return subset(rs, keys, name=f"{rs.name}.sub{size}", ranges_path=f"index/subsets/scaling_{size}.ranges.json",
                  ranges_codec="json-indent1", source=f"scaling_subsets(size={size}, seed={seed}) of {rs.name}")


def paper_rowspaces(pack, corpus) -> dict[str, RowSpace]:
    """The four row spaces the paper's runs indexed, from the corpus pack's index files and the
    corpus episodes' metadata."""
    pack = pathlib.Path(pack)
    full = build_rowspace(name="libero5k_v2_d8", corpus=corpus, roots_file=pack / PAPER_ROOTS,
                          ranges_file=pack / PAPER_RANGES, roots_path=PAPER_ROOTS, ranges_path=PAPER_RANGES,
                          source="qvla-v1-corpus index/ and the corpus episodes' metadata")
    return {"libero5k_v2_d8": full,
            "libero5k_v2_d8.sub1000": scaling_subset(full, 1000),
            "libero5k_v2_d8.sub2500": scaling_subset(full, 2500),
            "heldout24_ft": flat_rowspace(name="heldout24_ft", n_rows=FT_FRAMES,
                                          source="the held-out finetune set's 86,912 frames, every one trainable")}
