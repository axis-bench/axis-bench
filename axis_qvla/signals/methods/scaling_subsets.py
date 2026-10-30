"""The 1K / 2.5K pretraining-scaling subsets: episode-filtered ranges files.

OFFLINE TIER: numpy + pandas only (via axis_qvla.signals.methods.schedule's walk over the pretraining
data).

The pi0.5 loader is restricted by the ranges file alone: `pretrain_sampler.plan_pretrain_rows`
keeps a frame iff its episode key has a span covering it, and an episode with no key in `ranges`
contributes nothing -- so deleting an episode's key removes exactly its frames, while the dense
weight/tag artifacts (keyed over the full pretraining data's frame index) stay valid. Subsets are
NESTED prefixes of one seeded permutation of the ranges file's keys: each prefix is still a uniform
random subset, and nesting keeps the scaling curve monotone in data. Step budgets are
1-epoch-matched: trainable frames of the subset // batch (floor).

The scaling runs reuse the 5K run's quality tags and dropout mask (keyed over the full pretraining
data), so inside a subset the five tag bins are no longer equal-frequency.

The released scaling_manifest.json records the digest `cc17570b...` of the generator these
functions come from; a manifest written here records this file's `generator_sha256` instead. The
ranges files themselves carry no digest and rebuild byte for byte.

    python -m axis_qvla.signals.methods.scaling_subsets --roots <roots_d8> --ranges <ranges> \\
        --weights <approach-aware chunk awr_weights.json> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from axis_qvla.signals.methods import schedule as bis
from axis_qvla.signals.blocks.awr import DEFAULT_CAP, DEFAULT_HORIZON, DEFAULT_TAU, episode_key


def _generator_digest() -> str:
    """sha256 over this file AND the walk it delegates row identity to."""
    h = hashlib.sha256()
    h.update(Path(bis.__file__).read_bytes())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()


def _episode_list_digest(keys: list[str]) -> str:
    """Order-insensitive identity of an episode set: sha256 over the sorted keys."""
    return hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()


def _provenance(args) -> dict:
    return {
        "seed": int(args.seed),
        "generator_sha256": _generator_digest(),
        "roots_digest": bis._file_digest(args.roots),
        "ranges_digest": bis._file_digest(args.ranges),
        "weights_digest": bis._file_digest(args.weights),
        "weights_corpus_fingerprint": bis._weights_header(args.weights)
        .get("provenance", {})
        .get("corpus_fingerprint"),
    }


def filter_ranges(ranges: dict, keep: set[str]) -> dict:
    """The ranges mapping restricted to `keep`, preserving the source file's key order and
    each kept episode's span list verbatim -- same schema, fewer keys."""
    return {k: v for k, v in ranges.items() if k in keep}


def scaling_subsets(all_keys: list[str], sizes: list[int], seed: int) -> dict[int, list[str]]:
    """Nested uniform-random subsets: prefixes of one seeded permutation of `all_keys`.

    `all_keys` must be in a canonical order (the ranges file's key order) so the permutation --
    and therefore every subset -- is reproducible from (source file, seed) alone.
    """
    if any(s > len(all_keys) for s in sizes):
        raise ValueError(f"a requested size in {sizes} exceeds the {len(all_keys)} episodes")
    perm = np.random.default_rng(seed).permutation(len(all_keys))
    return {s: [all_keys[i] for i in perm[:s]] for s in sizes}


def corpus_walk(roots_path, ranges_path, stats):
    """`episode_rows` (drop=False) plus the episode key of each block, checked against the
    authoritative enumeration's per-episode row counts."""
    blocks, progress, unlab, n_trainable, n_rows, _row_w = bis.episode_rows(
        roots_path, ranges_path, stats, drop=False
    )
    ranges = json.loads(Path(ranges_path).read_text())
    keys: list[str] = []
    lengths: list[int] = []
    for task_id, root in bis._ordered_roots(roots_path):
        episodes, _total = bis._episode_offsets(root)
        for ep_index in sorted(episodes):
            _start, length = episodes[ep_index]
            key = episode_key(task_id, ep_index)
            frames = bis._frames_in_spans(ranges.get(key, []), length)
            if frames.size == 0:
                continue
            keys.append(key)
            lengths.append(int(frames.size))
    if lengths != [len(b) for b in blocks]:
        raise ValueError(
            "the key walk disagrees with build_index_schedule.episode_rows on per-episode row "
            "counts; the two walks describe different episode sets and every row below would be "
            "attributed to the wrong episode"
        )
    return blocks, keys, progress, unlab, n_trainable, n_rows


def _check_declared_rows(weights_path, n_rows: int) -> None:
    declared = bis._weights_header(weights_path).get("n_rows")
    if declared is not None and int(declared) != int(n_rows):
        raise ValueError(
            f"{weights_path} says the pretraining data has {declared} rows but the roots enumerate "
            f"{n_rows}; the weights and the roots index describe different pretraining data"
        )


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(path)


def run_scaling(args) -> dict:
    stats = bis.episode_stats_from_weights(args.weights, tau=DEFAULT_TAU, cap=DEFAULT_CAP,
                                           horizon=DEFAULT_HORIZON)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ranges = json.loads(Path(args.ranges).read_text())
    subsets = scaling_subsets(list(ranges), args.sizes, args.seed)
    _blocks, _keys, _prog, _unlab, full_trainable, n_rows = corpus_walk(
        args.roots, args.ranges, stats
    )
    _check_declared_rows(args.weights, n_rows)
    manifest = {
        "provenance": _provenance(args),
        "batch": int(args.batch),
        "nested": True,
        "full_corpus": {
            "n_episodes": len(ranges),
            "trainable_rows": int(full_trainable),
            "steps_1epoch": int(full_trainable) // args.batch,
        },
        "subsets": {},
    }
    for size, keys in sorted(subsets.items()):
        out = out_dir / f"scaling_{size}.ranges.json"
        _atomic_write_json(out, filter_ranges(ranges, set(keys)))
        _b, walk_keys, _p, _u, n_trainable, sub_n_rows = corpus_walk(args.roots, out, stats)
        if sub_n_rows != n_rows:
            raise AssertionError("subset walk enumerated a different pretraining-data length")
        if set(walk_keys) - set(keys):
            raise AssertionError("subset ranges kept an episode outside the drawn subset")
        manifest["subsets"][str(size)] = {
            "ranges_file": out.name,
            "seed": int(args.seed),
            "n_episodes": len(keys),
            "n_episodes_with_rows": len(walk_keys),
            "episode_list_sha256": _episode_list_digest(keys),
            "trainable_rows": int(n_trainable),
            "steps_1epoch": int(n_trainable) // args.batch,
        }
    _atomic_write_json(out_dir / "scaling_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="axis_qvla.signals.methods.scaling_subsets", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", required=True)
    ap.add_argument("--ranges", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 2500])
    return ap


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(run_scaling(build_parser().parse_args(argv)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
