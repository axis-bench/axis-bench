"""Pure-conditioning quality tags: the sim benchmark's episode tags with NO dropout.

OFFLINE TIER: numpy only.

The sim benchmark's tag artifact bakes in pi0.7's two-level dropout (15% whole, 5% component), so the
model also sees the unconditional prompt. The real-world stage-1 pretrain uses PURE conditioning: every
trainable row carries its episode's quintile. Everything that decides WHICH tag a row gets is the pinned
builder's: the P-ratio read off the float32 weights, the global quintile with neutral 3, and the flat row
walk. Only the dropout stream is skipped.

The artifact says so twice. `meta.pure_conditioning` is set, and the online tag reader must honour it,
because there is no NO_TAG row to train the unconditional branch on. And the generator digest covers this
module, so the file can never pass for the pinned builder's output.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from axis_qvla.signals.methods import schedule as bis
from axis_qvla.signals.methods import cfg_tags as bql
from axis_qvla.signals.blocks.awr import DEFAULT_CAP, DEFAULT_HORIZON, DEFAULT_TAU
from axis_qvla.signals.blocks.schedule import N_BINS, quality_bins
from axis_qvla.signals.blocks.tags import NOT_TRAINABLE

BUILDER = "qvla.realrobot.realign.tags/1"


def _generator_digest() -> str:
    """The pinned tag builder's digest, extended by this module's source."""
    h = hashlib.sha256()
    h.update(bql._generator_digest().encode())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()


def _config_hash(weights_path, roots_path, ranges_path) -> str:
    payload = {
        "pure_conditioning": True, "n_bins": N_BINS, "granularity": "episode",
        "generator_digest": _generator_digest(),
        "weights_digest": bis._file_digest(weights_path),
        "roots_digest": bis._file_digest(roots_path),
        "ranges_digest": bis._file_digest(ranges_path),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def build_pure_tags(weights_path: Path, roots_path: Path, ranges_path: Path, out_path: Path) -> dict:
    """Write `quality_<reward_id>.npz` (tag, prompts, meta) and return its meta."""
    stats = bis.episode_stats_from_weights(weights_path, tau=DEFAULT_TAU, cap=DEFAULT_CAP,
                                           horizon=DEFAULT_HORIZON)
    header = bis._weights_header(weights_path)
    reward_id = header.get("reward_id")
    if reward_id is None:
        raise ValueError(f"{weights_path} has no 'reward_id'; the tags could not name their run")
    blocks, progress, unlab, _n_trainable, n_rows, _row_w = bis.episode_rows(
        roots_path, ranges_path, stats, drop=False)
    declared = header.get("n_rows")
    if declared is not None and int(declared) != int(n_rows):
        raise ValueError(
            f"{weights_path} says the corpus has {int(declared)} rows but {roots_path} enumerates "
            f"{int(n_rows)}; the weights and the roots index describe different corpora")
    q_ep = quality_bins(progress, unlab)
    flat = np.concatenate(blocks) if blocks else np.empty(0, dtype=np.int64)
    if flat.size and not np.all(np.diff(flat) > 0):
        raise ValueError("episode row blocks are not strictly ascending across the corpus")
    if flat.size and (flat.min() < 0 or flat.max() >= int(n_rows)):
        raise ValueError(f"row {flat.max()} is outside the corpus of {n_rows} rows")
    tag = np.full(int(n_rows), NOT_TRAINABLE, dtype=np.uint8)
    tag[flat] = np.repeat(np.asarray(q_ep, dtype=np.uint8), [len(b) for b in blocks])
    dropout_meta = {
        "n_trainable": int(flat.size),
        "bin_row_counts": {str(b): int((tag[flat] == b).sum()) for b in range(1, N_BINS + 1)},
        "realized_drop_whole": 0.0, "realized_drop_component": 0.0, "realized_tagged_fraction": 1.0,
    }
    meta = bql.quality_meta(
        tag, bql.corpus_prompts(roots_path), reward_id=reward_id, seed=0,
        horizon=int(header.get("horizon", DEFAULT_HORIZON)), n_rows=int(n_rows),
        config_hash=_config_hash(weights_path, roots_path, ranges_path), episode_rows=blocks,
        q_ep=q_ep, task_ids=bql.episode_task_ids(roots_path, ranges_path, stats,
                                                 block_lengths=[len(b) for b in blocks]),
        dropout_meta=dropout_meta)
    meta.update(pure_conditioning=True, generator_sha256=_generator_digest(), builder=BUILDER)
    bql.check_artifact_filename(out_path, meta["reward_id"], meta["granularity"])
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = bis._unique_tmp_path(out)
    try:
        np.savez(tmp, tag=tag, prompts=np.array([str(s) for s in bql.corpus_prompts(roots_path)]),
                 meta=np.array(json.dumps(meta)))
        tmp.replace(out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return meta
