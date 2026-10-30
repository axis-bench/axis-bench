"""Write a dense per-row quality-tag artifact for one CFG arm.

OFFLINE TIER: numpy + pandas only.

CFG does NOT consume a schedule -- that would change which rows are drawn, and coverage
neutrality against the plain-BC run is what distinguishes conditioning. So this builder
reuses the schedule builder's corpus walk (`episode_rows`, `episode_stats_from_weights`) and its
quality maths (`quality_bins`) without touching either module's behaviour, and emits an artifact
that only says WHAT EACH ROW IS LABELLED -- never which rows to draw.

The tag is dense over the CONCATENATED CORPUS row space (1,906,469 rows for the released
5,000-episode corpus), not its 1,319,784 trainable rows: the online side is then a pure array
lookup by flat row index with no re-derivation of the index space. About 2 MB.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np

from axis_qvla.signals.methods import schedule as bis
from axis_qvla.signals.blocks import schedule as index_schedule
from axis_qvla.signals.blocks import tags as quality_labels
from axis_qvla.signals.blocks.awr import DEFAULT_CAP, DEFAULT_HORIZON, DEFAULT_TAU, episode_key
from axis_qvla.signals.blocks.schedule import N_BINS, quality_bins
from axis_qvla.signals.blocks.tags import dense_tags, dense_tags_rowwise, q_given_task_entropy_ratio

# The prefix an artifact filename must carry to be making a claim about its reward.
_NAME_PREFIX = "quality_"


def _generator_digest() -> str:
    """sha256 over EVERY source that decides a tag: the quality maths, this driver, and BOTH
    schedule-builder modules whose walk and binning this reuses.

    Four files, not one, because all four move indices or labels. `build_index_schedule` decides
    which rows belong to which episode and in what order (and therefore which dropout draw each
    row gets); `index_schedule.quality_bins` decides q; `quality_labels` decides the dropout;
    this file decides the rest. An edit to any of them stales both artifacts.

    DELIBERATELY conservative -- a comment edit moves the digest even though no tag moves. A
    false "these differ" costs one rebuild and is immediately visible; a false "these match"
    silently corrupts an experiment record, and nothing downstream can detect it.
    """
    h = hashlib.sha256()
    for module in (index_schedule, bis, quality_labels):
        h.update(Path(module.__file__).read_bytes())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()


def _config_hash(args) -> str:
    """Hash the inputs AND the generators, as a sorted-key JSON dict (never concatenation -- it
    has no field delimiter, so `--seed 1 --horizon 0` and `--seed 10 --horizon` collide)."""
    payload = {
        "seed": int(args.seed),
        "n_bins": N_BINS,
        "granularity": str(getattr(args, "granularity", "episode")),
        "plateau": "rank",
        "generator_digest": _generator_digest(),
        "weights_digest": bis._file_digest(args.weights),
        "roots_digest": bis._file_digest(args.roots),
        "ranges_digest": bis._file_digest(args.ranges),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def check_artifact_filename(path, reward_id: str, granularity: str = "episode") -> None:
    """Bind a `quality_<reward_id>.npz` / `quality_<reward_id>_seg.npz` filename to what the
    artifact was built from.

    Only names of that shape are checked. Any other name (a staging path, a dated file) makes no
    claim, so there is nothing to contradict and this is a no-op -- the check must not turn a
    naming convention into a requirement. Same rule, and the same reason, as
    `data_loader._check_schedule_reward_id`.

    The `_seg` suffix claims ROW granularity, and the check runs in BOTH directions: a row-built
    artifact under an unsuffixed name would run the granularity ablation while recording itself
    as the episode arm, which is the identical hole the reward-id half closes for `v2`/`phase`.
    All four artifacts run under ONE config name, so the filename is the only thing that
    distinguishes them.
    """
    stem = Path(path).stem
    if not stem.startswith(_NAME_PREFIX):
        return
    claimed = stem[len(_NAME_PREFIX):]
    claimed_gran = "episode"
    if claimed.endswith("_seg"):
        claimed, claimed_gran = claimed[: -len("_seg")], "row"
    if claimed != str(reward_id):
        raise ValueError(
            f"artifact {path} is named for reward_id={claimed!r} but was built from "
            f"reward_id={reward_id!r}. Both CFG runs run under ONE config name "
            f"(pi05_axis_cfg) and carry identical structure, so this filename is the only "
            f"thing that distinguishes them: this run would record itself as {claimed!r} "
            f"while conditioning on {reward_id!r}."
        )
    if claimed_gran != str(granularity):
        raise ValueError(
            f"artifact {path} is named for granularity={claimed_gran!r} but was built at "
            f"granularity={granularity!r}. The `_seg` suffix is the only thing separating the "
            f"episode-tag run from the row-tag run, so this run would record itself as one "
            f"granularity while conditioning on the other."
        )


def row_quality_bins(row_w, *, plateau: str = "rank") -> list:
    """Per-row quintiles: every row ranked by `delta_quality_bins`, so the tag is exactly the
    row-level quality unit. On the approach-aware weights the global Q1 cutoff is 0.0, so plateau
    frames (a grasp's contact-settle among them) rank bottom-quintile: the tag follows the phase
    reward's own semantics. `plateau` is recorded in the artifact's meta and config hash; "rank"
    is its one value."""
    if plateau != "rank":
        raise ValueError(f"plateau must be 'rank', got {plateau!r}")
    return index_schedule.delta_quality_bins(row_w)


def _check_task_id_walk(task_ids, block_lengths) -> None:
    """The second walk must produce one task id per episode block, in the same order."""
    if len(task_ids) != len(block_lengths):
        raise ValueError(
            f"the task-id walk produced {len(task_ids)} episodes but the authoritative "
            f"enumeration (build_index_schedule.episode_rows) produced {len(block_lengths)}; it "
            f"disagrees with the walk whose row indices this artifact is built on, so every "
            f"quality bin would be attributed to the wrong task"
        )


def episode_task_ids(roots_path, ranges_path, stats: dict, *, block_lengths) -> np.ndarray:
    """The task id of each episode, in `episode_rows`' order.

    `episode_rows` discards the task id (the schedule arms never needed it), so this walks
    the corpus a second time with the SAME loop -- `_ordered_roots` -> `_episode_offsets` ->
    `_frames_in_spans`, skipping episodes with no trainable rows exactly as it does -- and the
    result is checked against the authoritative enumeration's per-episode row counts. That is the
    same technique used and verified separately.

    `stats` is accepted and NOT read: the signature comes from the plan, which paired every corpus
    walk with the AWR stats dict, and this file keeps it rather than diverging from a spec'd
    contract over an unused argument. Do not read it as meaning this function cross-checks
    anything against per-episode weight counts -- it does not; `_frames_in_spans` and the length
    comparison against `block_lengths` are the only checks this walk performs. The weight-length
    agreement (`st["n"] != length`) is `build_index_schedule.episode_rows`'s check, upstream of
    this call, not this function's.
    """
    ranges = json.loads(Path(ranges_path).read_text())
    out: list[int] = []
    lengths: list[int] = []
    for task_id, root in bis._ordered_roots(roots_path):
        episodes, _total = bis._episode_offsets(root)
        for ep_index in sorted(episodes):
            _local_start, length = episodes[ep_index]
            frames = bis._frames_in_spans(ranges.get(episode_key(task_id, ep_index), []), length)
            if frames.size == 0:
                continue
            out.append(int(task_id))
            lengths.append(int(frames.size))
    _check_task_id_walk(out, block_lengths)
    if lengths != list(block_lengths):
        raise ValueError(
            "the task-id walk disagrees with build_index_schedule.episode_rows on per-episode "
            "row counts; the two walks describe different episode sets and the entropy ratio "
            "would attribute quality bins to the wrong tasks"
        )
    return np.asarray(out, dtype=np.int64)


def task_prompts(root) -> set[str]:
    """Every task instruction of one task directory, read from `meta/tasks*.parquet` -- the file
    LeRobot's `dataset_meta.tasks` is built from, because the offline venv cannot load LeRobot.

    RAISES when no task table is found: an empty set would make the token-budget guard vacuous.
    """
    import pandas as pd

    files = sorted(glob.glob(str(Path(root) / "meta" / "tasks*.parquet"))) or sorted(
        glob.glob(str(Path(root) / "meta" / "tasks" / "**" / "*.parquet"), recursive=True)
    )
    if not files:
        raise FileNotFoundError(f"no meta/tasks parquet under {root}")
    prompts: set[str] = set()
    for f in files:
        df = pd.read_parquet(f)
        if "task" in df.columns:
            prompts.update(str(s) for s in df["task"])
        # `is_string_dtype` recognizes both the legacy `object` and pandas 3's `str` index dtype,
        # and still rejects a numeric index (RangeIndex/int64/float64).
        elif df.index.name == "task" or pd.api.types.is_string_dtype(df.index):
            prompts.update(str(s) for s in df.index)
        else:
            raise ValueError(
                f"{f} has neither a 'task' column nor a string index; the prompt strings "
                f"cannot be read and the token-budget guard would have nothing to check"
            )
    return prompts


def corpus_prompts(roots_path) -> list[str]:
    """Every unique task instruction in the corpus, sorted. Stored in the artifact so the online
    token-budget guard tokenizes the REAL longest prompt instead of an estimate."""
    prompts: set[str] = set()
    for _task_id, root in bis._ordered_roots(roots_path):
        prompts |= task_prompts(root)
    return sorted(prompts)


def entropy_units(episode_rows, q_ep, task_ids, *, granularity: str = "episode",
                  row_bins=None) -> tuple[np.ndarray, np.ndarray]:
    """`(q_units, task_units)`: the quality bin and task id of every unit the tag conditions on,
    which is what the `q_given_task_entropy_ratio` diagnostic is computed over.

    Episode granularity: one unit per episode (`q_ep`, `task_ids`). Row granularity: one unit per
    row, each row's own bin and its episode's task id -- an episode-level census would understate
    how much a per-row tag leaks task identity, because rows of one episode share a task. No
    episodes gives two empty arrays, which the diagnostic refuses ("no episodes"); blocks that do
    not pair up with their bins or task ids are refused here, naming the counts.
    """
    if granularity == "row":
        if len(row_bins) != len(episode_rows) or len(task_ids) != len(episode_rows):
            raise ValueError(
                f"{len(episode_rows)} episode row blocks, {len(row_bins)} bin blocks and {len(task_ids)} task "
                f"ids; the three come from one enumeration and a mismatch attributes bins to the wrong tasks"
            )
        lengths = [len(b) for b in episode_rows]
        q_units = np.concatenate([np.asarray(b, dtype=np.int64) for b in row_bins]) \
            if row_bins else np.empty(0, dtype=np.int64)
        return q_units, np.repeat(np.asarray(task_ids, dtype=np.int64), lengths)
    return np.asarray(q_ep), np.asarray(task_ids)


def quality_meta(tag: np.ndarray, prompts, *, reward_id, seed, horizon, n_rows, config_hash,
                 episode_rows, q_ep, task_ids, dropout_meta,
                 granularity: str = "episode", row_bins=None, plateau: str = "rank") -> dict:
    if tag.shape != (int(n_rows),):
        raise ValueError(
            f"tag array has shape {tag.shape} but the pretraining data has n_rows={n_rows}; refusing to "
            f"write an artifact that misdescribes its own index space"
        )
    if not prompts:
        raise ValueError(
            "no prompts: the token-budget guard tokenizes this list, so an empty one makes the "
            "guard vacuous and the one long instruction that overflows goes unnoticed"
        )
    if granularity not in ("episode", "row"):
        raise ValueError(f"granularity must be 'episode' or 'row', got {granularity!r}")
    if (granularity == "row") != (row_bins is not None):
        raise ValueError(
            "row_bins must be given exactly when granularity='row'; a row artifact without its "
            "bins cannot report the granularity diagnostics that justify its existence, and an "
            "episode artifact with them would report diagnostics about bins it never wrote"
        )
    # The task-identity confound is computed at the granularity UNIT the tag actually conditions on: for the
    # row arm, an episode-level census would understate how much a per-row tag leaks task identity
    # (rows of one episode share a task, so within-episode bin variation dilutes H(q|task)).
    q_units, task_units = entropy_units(episode_rows, q_ep, task_ids, granularity=granularity, row_bins=row_bins)
    if granularity == "row":
        distinct = np.array([len(np.unique(np.asarray(b))) for b in row_bins], dtype=np.int64)
        stds = [float(np.std(np.asarray(b, dtype=np.float64))) for b in row_bins if len(b)]
        granularity_diag = {
            "plateau": str(plateau),
            # The ablation's reason to exist, recorded where a reader will look: how much
            # within-episode structure the episode-level tag was throwing away.
            "episodes_multibin_fraction": float((distinct > 1).mean()) if distinct.size else 0.0,
            "within_episode_bin_std_mean": float(np.mean(stds)) if stds else 0.0,
        }
    else:
        granularity_diag = {}
    return {
        "reward_id": str(reward_id),
        "granularity": str(granularity),
        "seed": int(seed),
        "horizon": int(horizon),
        "n_rows": int(n_rows),
        "n_trainable": int(dropout_meta["n_trainable"]),
        "config_hash": str(config_hash),
        "generator_sha256": _generator_digest(),
        "n_prompts": int(len(prompts)),
        "bin_row_counts": dropout_meta["bin_row_counts"],
        "realized_drop_whole": float(dropout_meta["realized_drop_whole"]),
        "realized_drop_component": float(dropout_meta["realized_drop_component"]),
        "realized_tagged_fraction": float(dropout_meta["realized_tagged_fraction"]),
        # The task-identity confound, recorded per artifact rather than referenced from a report:
        # a reader must be able to see it without leaving the file.
        "q_given_task_entropy_ratio": float(q_given_task_entropy_ratio(q_units, task_units)),
        "n_episodes": int(len(episode_rows)),
        **granularity_diag,
    }


def write_quality(path, tag: np.ndarray, prompts, **kw) -> dict:
    """Write `tag` + `prompts` + meta. Returns the meta so a caller can log it."""
    meta = quality_meta(tag, prompts, **kw)
    check_artifact_filename(path, meta["reward_id"], meta["granularity"])
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = bis._unique_tmp_path(p)
    try:
        np.savez(
            tmp,
            tag=tag.astype(np.uint8),
            prompts=np.array([str(s) for s in prompts]),
            meta=np.array(json.dumps(meta)),
        )
        tmp.replace(p)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return meta


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build a quality-tag artifact for one CFG run.")
    ap.add_argument("--weights", required=True, help="AWR weights JSON for this run's reward")
    ap.add_argument("--roots", required=True, help="roots-index JSON for the pretraining data")
    ap.add_argument("--ranges", required=True, help="non-idle sample-ranges JSON")
    ap.add_argument("--out", required=True, help="quality_<reward_id>.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--granularity", choices=("episode", "row"), default="episode",
        help="What one tag describes: 'episode' broadcasts the episode's global quintile "
        "(quality_bins) to its rows; 'row' bins each row's own advantage "
        "(delta_quality_bins), so the tag varies WITHIN an episode. Row artifacts are named "
        "quality_<reward_id>_seg.npz.",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    stats = bis.episode_stats_from_weights(
        a.weights, tau=DEFAULT_TAU, cap=DEFAULT_CAP, horizon=DEFAULT_HORIZON
    )
    header = bis._weights_header(a.weights)
    reward_id = header.get("reward_id")
    if reward_id is None:
        raise ValueError(
            f"{a.weights} has no 'reward_id' in its header; a quality artifact cannot record "
            f"which run it was built from without one"
        )
    # drop=False, always: CFG changes what the model is CONDITIONED ON, never which rows exist.
    # The per-row weights feed the ROW granularity only (`delta_quality_bins`); the episode
    # granularity bins EPISODES (`quality_bins` below) because its tag is a per-episode quality
    # tier.
    blocks, progress, unlab, n_trainable, n_rows, row_w = bis.episode_rows(
        a.roots, a.ranges, stats, drop=False
    )
    declared = header.get("n_rows")
    if declared is not None and int(declared) != int(n_rows):
        raise ValueError(
            f"{a.weights} says the pretraining data has {int(declared)} rows but {a.roots} enumerates "
            f"{int(n_rows)}; the weights and the roots index describe different pretraining data, so the "
            f"tag would be recorded against one index space and applied in the other"
        )
    q_ep = quality_bins(progress, unlab)
    task_ids = episode_task_ids(
        a.roots, a.ranges, stats, block_lengths=[len(b) for b in blocks]
    )
    row_bins = None
    if a.granularity == "row":
        row_bins = row_quality_bins(row_w, plateau="rank")
        tag, dropout_meta = dense_tags_rowwise(
            blocks, row_bins, n_rows=int(n_rows), seed=a.seed
        )
    else:
        tag, dropout_meta = dense_tags(blocks, q_ep, n_rows=int(n_rows), seed=a.seed)
    meta = write_quality(
        a.out, tag, corpus_prompts(a.roots),
        reward_id=reward_id, seed=a.seed,
        horizon=int(header.get("horizon", DEFAULT_HORIZON)),
        n_rows=int(n_rows), config_hash=_config_hash(a),
        episode_rows=blocks, q_ep=q_ep, task_ids=task_ids, dropout_meta=dropout_meta,
        granularity=a.granularity, row_bins=row_bins, plateau="rank",
    )
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
