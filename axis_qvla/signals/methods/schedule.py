"""Build an index-schedule artifact for one arm, and the coverage report that makes it auditable.

OFFLINE TIER: numpy + pandas (the parquet read is function-local). Never jax/torch/lerobot.

Two halves. `write_schedule` / `schedule_meta` serialise a schedule that is already drawn; the
`main` driver below reads the REAL corpus (roots index + non-idle ranges + that arm's AWR weights)
and produces one. The driver is offline because the reward-derived weights are, and because a
schedule must exist as a file before the trainer starts -- see `index_schedule` for why the
schedule is precomputed at all.

The meta block exists so that "what did this arm actually see?" is answerable from the artifact
alone -- samples-seen, unique frames, and the realized quality mix are reporting requirements of
the experiment, not debug output. `generator_sha256` is part of that answer: the schedule is
decided by `axis_qvla.signals.blocks.schedule`, so the artifact records which version of that module
drew it (and `config_hash` digests the same source -- see `_config_hash`).

`realized_draws_per_bin` / `baseline_draws_per_bin` use STRING bin keys, on both the returned
dict and the on-disk JSON, even though the natural key is an int. JSON object keys are always
strings, so an int-keyed dict returned here would silently become string-keyed after
`json.loads(str(np.load(p)["meta"]))` -- a consumer (the online sampler) that indexes by int
after loading from disk would get a silent miss instead of an error. Normalising both sides to
str keeps exactly one shape; do not "simplify" this back to int keys in the returned dict.

`baseline_draws_per_bin` is each bin's share of DRAWABLE rows, scaled to the number of draws.

That share is computed rather than assumed, because a flat `total_draws / N_BINS` only describes
the drawable mix while the bins are quintiles over ROWS (`index_schedule.delta_quality_bins`),
where the drawable shares come out equal by construction and this baseline coincides with the flat
one. `broadcast_episode_bins` bins by EPISODE instead, and episodes have wildly unequal row
counts, so under that binning a flat `total_draws / N_BINS` misdescribes even a `vanilla`
schedule -- against the real corpus it reads +38% in q1 and -26% in q5 for an arm that is drawing
uniformly. Do NOT simplify this to `total_draws / N_BINS` on the grounds that the two currently
coincide: they agree only because row-quintile binning derives equal drawable shares, not by
definition, and the moment a caller bins some other way this computed share is the only thing that
still describes the mix correctly.

"Drawable" is the row set THIS arm can draw -- for `drop` that is the kept partition passed in via
`episode_rows`, not the full non-idle row set, since a drop arm can never draw a row it dropped.
That is exactly the `episode_rows`/`row_bins` pair this function already has; the baseline must be
computed from them rather than re-touching the corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np

from axis_qvla.signals.blocks import schedule as index_schedule
from axis_qvla.signals.blocks.awr import DEFAULT_CAP, DEFAULT_HORIZON, DEFAULT_TAU, episode_key
from axis_qvla.signals.blocks.schedule import (
    LAMBDA_MAX,
    MODES,
    N_BINS,
    RAMP_START_FRAC,
    DEFAULT_KEEP_TOP_FRAC,
    delta_quality_bins,
    draw_schedule,
    quality_bins,
    random_fraction_mask,
    top_fraction_mask,
)


def _generator_source() -> bytes:
    """The source bytes of `axis_qvla.signals.blocks.schedule` -- the module that decides every index.

    Located through the imported module's own `__file__` rather than a path relative to the
    working directory, so it resolves identically whether this is run from the repo root, from a
    git worktree, or from a job script that chdir'd somewhere else. Factored out as its own
    function so a test can substitute a different source without touching the real module on disk.
    """
    return Path(index_schedule.__file__).read_bytes()


def _generator_digest() -> str:
    """sha256 of the generator source. See `_config_hash` for why this is hashed at all."""
    return hashlib.sha256(_generator_source()).hexdigest()


def schedule_meta(
    rows: np.ndarray,
    *,
    mode: str,
    seed: int,
    total_steps: int,
    batch: int,
    row_bins: list[np.ndarray],
    episode_rows: list[np.ndarray],
    keep_fraction: float,
    config_hash: str,
    reward_id: str,
    horizon: int,
    n_rows: int,
) -> dict:
    if rows.shape != (total_steps, batch):
        raise ValueError(
            f"schedule shape {rows.shape} disagrees with meta (total_steps={total_steps}, "
            f"batch={batch}); refusing to write an artifact that misdescribes itself"
        )
    if len(row_bins) != len(episode_rows):
        raise ValueError(
            f"row_bins has {len(row_bins)} episodes but episode_rows has {len(episode_rows)}; the "
            f"realized/expected breakdown would describe different rows than the schedule drew"
        )
    # Row -> its own bin. This MUST be the same per-row binning `draw_schedule` tilted on: reporting
    # realized draws against another binning would misstate the tilt.
    bin_of_row: dict[int, int] = {}
    ep_of_row: dict[int, int] = {}
    for ei, (block, blk_bins) in enumerate(zip(episode_rows, row_bins)):
        for r, b in zip(block.tolist(), np.asarray(blk_bins).tolist()):
            bin_of_row[int(r)] = int(b)
            ep_of_row[int(r)] = ei

    flat = rows.ravel()
    bins = np.array([bin_of_row[int(r)] for r in flat], dtype=np.int64)
    eps = np.array([ep_of_row[int(r)] for r in flat], dtype=np.int64)
    # String keys, not int: see the module docstring -- this is what keeps the returned dict and
    # the on-disk JSON (json.loads always yields string object keys) a single, identical shape.
    realized = {str(b): int((bins == b).sum()) for b in range(1, N_BINS + 1)}
    total_draws = int(flat.size)
    # Drawable-row share, scaled to total_draws -- the null hypothesis a uniform (no-tilt) draw
    # actually satisfies. `episode_rows` is already THIS arm's drawable set (post-drop for `drop`,
    # the full non-idle set otherwise), and `row_bins` is 1:1 with it per ROW, so this reuses both
    # inputs with no second corpus read. See the module docstring for why a flat total/N_BINS is
    # wrong, not just imprecise.
    drawable_bins = (
        np.concatenate([np.asarray(b, dtype=np.int64) for b in row_bins])
        if row_bins else np.empty(0, dtype=np.int64)
    )
    total_drawable = int(drawable_bins.size)
    bin_row_counts = {str(b): int((drawable_bins == b).sum()) for b in range(1, N_BINS + 1)}
    baseline = {
        b: (total_draws * n / total_drawable) if total_drawable else 0.0
        for b, n in bin_row_counts.items()
    }
    n_trainable = int(sum(len(b) for b in episode_rows))
    return {
        "mode": mode,
        "seed": int(seed),
        "total_steps": int(total_steps),
        "batch": int(batch),
        "keep_fraction": float(keep_fraction),
        "config_hash": config_hash,
        # WHICH source drew these rows, readable without re-deriving anything. `config_hash`
        # folds this same digest in, but only as an opaque mixture; a reader comparing two
        # artifacts needs to see the generator digest itself to tell "the corpus moved" from
        # "the sampler moved".
        "generator_sha256": _generator_digest(),
        # arm identity: what reward produced the weights this schedule was drawn from -- so
        # "what did this arm see?" is answerable from the schedule artifact alone, per the module
        # docstring. `n_rows` is the CONCATENATED CORPUS these row indices live in, as enumerated
        # by `episode_rows` (not the weights header's claim about one); the online tier requires
        # it to equal `len(dataset)` exactly.
        "reward_id": reward_id,
        "horizon": int(horizon),
        "n_rows": int(n_rows),
        "unique_episodes": int(len(np.unique(eps))),
        "unique_frames": int(len(np.unique(flat))),
        "trainable_rows": n_trainable,
        "epochs_over_unique_rows": total_draws / max(n_trainable, 1),
        # No surviving mode ramps a quality tilt; these two fields are kept at their neutral
        # values for schema stability (a consumer that already reads them must keep working).
        "lambda_max": 0.0,
        "ramp_start_step": total_steps,
        "realized_draws_per_bin": realized,
        "baseline_draws_per_bin": baseline,
    }


def _unique_tmp_path(p: Path) -> Path:
    """A temp name unique to THIS writer, in the same directory as `p`.

    Two processes writing the same target must not share one temp file: `np.savez` is not
    atomic against a second writer truncating it mid-write, so a name that is a pure function of
    the target path (e.g. "<stem>.tmp.npz") lets both writers' `tmp.replace(p)` land a torn zip
    at the target -- a corrupt artifact that has been cleanly renamed. The pid+random suffix
    makes collision practically impossible; do not "simplify" this back to a name derived only
    from `p`. Staying in `p`'s directory keeps `Path.replace` an atomic rename rather than a
    cross-filesystem copy. The name still ends in ".npz" -- np.savez silently appends ".npz" to
    any filename that doesn't, so a naive "<name>.npz.tmp" gets written as
    "<name>.npz.tmp.npz" and the rename below then fails to find it.
    """
    return p.with_name(f"{p.stem}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}.npz")


def write_schedule(path, rows: np.ndarray, **kw) -> dict:
    """Write `rows` plus its meta block. Returns the meta so a caller can log it."""
    meta = schedule_meta(rows, **kw)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = _unique_tmp_path(p)
    # tmp+rename (same directory, so the rename is atomic): a killed writer must not leave a
    # half-written schedule that loads cleanly and silently trains on the wrong rows. On
    # failure the temp file is removed so a crash does not litter the artifact directory --
    # the original exception always propagates, it is never swallowed here.
    try:
        np.savez(tmp, rows=rows.astype(np.int64), meta=np.array(json.dumps(meta)))
        tmp.replace(p)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return meta


# --- the corpus driver ------------------------------------------------------------------------

CORPUS_ROOT_ENV = "AXIS_PRETRAIN_CORPUS_ROOT"


def episode_stats_from_weights(weights_path, *, tau: float = DEFAULT_TAU,
                               cap: float = DEFAULT_CAP,
                               horizon: int = DEFAULT_HORIZON) -> dict:
    """Per-episode keep mask, progress mask and unlabelled flag, read off the AWR weights.

    `w = min(exp(tau * Delta), cap)` is monotone in `Delta` and equals 1.0 at `Delta == 0`, so the
    sign of the advantage survives the cap. It is stored as FLOAT32, though, and exp(tau * Delta)
    rounds to exactly 1.0f whenever tau * Delta lies within about [-2^-25, 2^-24]: `w >= 1` is
    `Delta >= -2^-25/tau` and `w > 1` is `Delta > 2^-24/tau` (about -3e-9 and 6e-9 at tau 10), NOT
    an exact-zero test. Rows whose Delta is float noise (e.g. 1e-13 left by p_t - p_tc) therefore
    count as no progress. The paper's Eq. (5) writes 1[Delta_t > 0]; on the released
    approach-aware weights 6,805 trainable rows sit in (0, 6e-9) and an exact-zero test would move
    252 of the 5,000 episode quintiles. The locked tags, P-ratio episode weights and filters were
    built with THIS test -- keep it. Both partitions come straight off the artifact an arm already
    has, with no reward recomputation and no second definition of the reward to keep in sync.

    The two masks use DIFFERENT inequalities, deliberately. The drop rule keeps `Delta == 0` (a row
    that holds ground is not a regression, and dropping every plateau would gut the corpus), while
    the progress-time ratio counts only rows that actually move forward -- a long flat plateau must
    not read as quality.

    An episode whose weights are ALL exactly 1.0 is flagged UNLABELLED (it gets NEUTRAL_Q from
    `quality_bins` and is never dropped -- all-1.0 weights are all `>= 1.0`, so the keep mask
    already says so and no special case is needed). Calling that episode "neutralised" would be an
    INFERENCE, not a fact this artifact can support: `axis_qvla.signals.blocks.awr.neutralised_mask`
    documents that `w == 1.0` conflates two different things -- a row masked because its advantage
    window touched an invalid reward row, and a row whose advantage is genuinely zero (a flat
    reward over the horizon gives `exp(tau * 0) == 1.0` on its own). The weights file on disk
    cannot tell these apart; the honest per-episode count of masked rows exists upstream as
    `EpisodeWeights.n_neutral`, but `write_manifest` sums it into a single corpus-wide
    `n_neutral_rows` before the weights ever reach disk, so it is not recoverable from here. `q=3`
    (NEUTRAL_Q) is still the right bin for either case -- an episode that is either untrustworthy
    or genuinely flat throughout is equally uninformative for ranking quality -- so this does not
    change what the code does, only what the label is allowed to claim.

    `tau`/`cap`/`horizon` are checked against the artifact header rather than used in the
    arithmetic. Weights built at a different tau/cap/horizon are a different experiment, and a
    `cap <= 1` would silently collapse `w >= 1` to "keep nothing" without any of this erroring.
    Non-finite weights are refused outright: `w >= 1.0` and `w > 1.0` are both False for NaN, so a
    NaN would silently drop its row from the drop arm and count it as no-progress everywhere else
    -- exactly the failure mode the online tier already refuses to train on
    (`pretrain_dataset.py`'s "refusing to train on a partially-weighted objective"); the offline
    side that produces its input should not be more permissive than that.
    """
    return episode_stats_from_payload(json.loads(Path(weights_path).read_text()), tau=tau, cap=cap,
                                      horizon=horizon, source=str(weights_path))


def episode_stats_from_payload(payload, *, tau: float = DEFAULT_TAU, cap: float = DEFAULT_CAP,
                               horizon: int = DEFAULT_HORIZON, source: str = "the weights") -> dict:
    """`episode_stats_from_weights` for weights already in memory: a parsed weights file, or a
    header mapping with a `weights` mapping of episode key -> per-frame weights. `source` names it in
    errors."""
    if float(cap) <= 1.0:
        raise ValueError(
            f"cap={cap} does not exceed 1.0, so min(exp(tau*Delta), cap) can never be > 1 and the "
            f"sign of Delta is unreadable from the weights"
        )
    header = payload if isinstance(payload, dict) and "weights" in payload else {}
    by_key = header["weights"] if header else payload
    for name, want in (("tau", tau), ("cap", cap), ("horizon", horizon)):
        got = header.get(name)
        if got is not None and float(got) != float(want):
            raise ValueError(
                f"{source} was built with {name}={got} but this schedule assumes {want}; "
                f"the filtering partition and the quality bins would describe a different experiment"
            )
    out: dict[str, dict] = {}
    for key, w_list in by_key.items():
        w = np.asarray(w_list, dtype=np.float64)
        if not np.isfinite(w).all():
            raise ValueError(
                f"{key} in {source} has a non-finite weight; refusing to build a schedule from a "
                f"partially-weighted objective (w >= 1.0 and w > 1.0 are both False for NaN, so this "
                f"would silently drop the row from the filtering run and count it as no-progress everywhere else)"
            )
        out[key] = {
            # The weights themselves, read per row by `delta_quality_bins` for the row-level
            # quality rank. Ranking rows by `w` is ranking them by `Delta`: `w = min(exp(tau*Delta),
            # cap)` is monotone, so the only ties `w` introduces that `Delta` would not are inside
            # its atoms. The cap atom is the top 13.6% of rows, entirely INSIDE the top quintile
            # (top 20%), so it never straddles a bin edge. The w == 1.0 atom (exact plateaus plus
            # |Delta| float noise below ~6e-9, see `episode_stats_from_weights`) does straddle the
            # Q1|Q2 edge, and stable order splits it by corpus position -- the documented
            # chunk-tag tie rule. That is why the row-level rank needs no reward recomputation --
            # it reads straight off the weights already on disk.
            "w": w,
            "keep_mask": w >= 1.0,
            "progress_mask": w > 1.0,
            # `w.size > 0 and`: np.all of an empty array is vacuously True, which would label a
            # zero-length episode "neutralised" rather than "empty".
            "unlabelled": bool(w.size > 0 and np.all(w == 1.0)),
            "n": int(w.size),
        }
    return out


def _weights_header(weights_path) -> dict:
    """The provenance block of a weights artifact -- `reward_id`/`tau`/`cap`/`horizon`/`n_rows` --
    or `{}` for the legacy headerless format. Read separately from `episode_stats_from_weights`
    (which needs the same file) because that function's return value is keyed by episode and has
    no room left to also carry which arm produced it; `main` needs both.
    """
    payload = json.loads(Path(weights_path).read_text())
    return payload if isinstance(payload, dict) and "weights" in payload else {}


def _ordered_roots(roots_path) -> list[tuple[int, str]]:
    """`(task_id, root)` pairs from the roots-index JSON, sorted by task id as an INT.

    Duplicated from `openpi.training.pretrain_dataset._ordered_roots` rather than imported: that
    module is online tier and this one runs under the offline venv. The artifact stores flat row
    indices, so both enumerate task ids as integers in the same order; the two enumerations agree
    over the 5,000-episode corpus (1,318,749 rows, identical as an ordered sequence). As strings,
    "1000" < "800".

    Two formats, as there: v2 `{"version": 2, "base": ..., "roots": {...}}` with roots relative to
    `base` (itself relative to the index file, overridable by $AXIS_PRETRAIN_CORPUS_ROOT), and the
    legacy bare `{task_id: absolute_path}` mapping.
    """
    path = Path(roots_path)
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "roots" in data:
        mapping = data["roots"]
        override = os.environ.get(CORPUS_ROOT_ENV)
        base = Path(override) if override else path.parent / str(data.get("base", "."))
    else:
        mapping = data  # v1: values are absolute and self-sufficient
        base = None

    out: list[tuple[int, str]] = []
    for key, value in mapping.items():
        root = Path(str(value))
        if root.is_absolute():
            resolved = root
        elif base is None:
            raise ValueError(
                f"{path} has a relative root {value!r} but no 'base' key. A bare mapping is read "
                f"as the legacy absolute-path format; add \"version\": 2 and \"base\", or set "
                f"${CORPUS_ROOT_ENV}."
            )
        else:
            resolved = base / root
        out.append((int(key), str(resolved)))
    return sorted(out, key=lambda kv: kv[0])


def _episode_offsets(root) -> tuple[dict[int, tuple[int, int]], int]:
    """`{episode_index: (local_start_row, length)}` plus the sub-dataset's total row count.

    Read from `meta/episodes/**/*.parquet`, the same source (and the same columns) as
    `openpi.training.pretrain_dataset.read_episode_offsets`, so this driver places every episode in
    the concat index space without a LeRobot load -- which the offline venv could not do anyway.
    """
    import glob

    import pandas as pd

    files = sorted(glob.glob(str(Path(root) / "meta" / "episodes" / "**" / "*.parquet"),
                             recursive=True))
    if not files:
        raise FileNotFoundError(f"no meta/episodes parquet under {root}")
    ep = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    episodes = {
        int(e): (int(f), int(n))
        for e, f, n in zip(ep["episode_index"], ep["dataset_from_index"], ep["length"], strict=True)
    }
    return episodes, int(ep["length"].sum())


def _frames_in_spans(spans, length: int) -> np.ndarray:
    """Within-episode frame indices kept by the non-idle ranges, clamped to `[0, length)`.

    Half-open `[s, e)` and clamped exactly as `pretrain_sampler.plan_pretrain_rows` does, spans in
    file order: this is the other half of the flat-index agreement.
    """
    parts = []
    for s, e in spans:
        lo = max(0, int(s))
        hi = min(int(e), int(length))
        if hi > lo:
            parts.append(np.arange(lo, hi, dtype=np.int64))
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def episode_rows(roots_path, ranges_path, stats: dict, *, drop: bool):
    """Per-episode row blocks in the SAME flat index space the loader uses, plus their quality.

    Row identity must match `openpi.training.pretrain_dataset.plan_rows_from_roots` exactly: the
    artifact stores dataset INDICES, so an enumeration that disagrees by one episode trains on the
    wrong frames with no error and no log line. Both walk `_ordered_roots` in task-id order and
    place a row at `global_base + local_start + frame`, where `global_base` is the cumulative
    LENGTH of the earlier sub-datasets -- not the cumulative count of kept rows, which is a
    different (and wrong) space entirely.

    Episodes with no trainable rows are omitted rather than carried as empty blocks: they can never
    be sampled, and including them would push spurious zero-progress entries into the GLOBAL
    quintile and move the bin edges for every other episode. Episodes emptied by the drop rule ARE
    kept, so every arm's quintile reporting bins the same episode population and stays comparable
    across arms.

    Returns `(episode_rows, progress_ratio, unlabelled, n_trainable, corpus_rows, row_w)`, where
    `n_trainable` counts the rows BEFORE the drop rule -- so `kept / n_trainable` measures the drop
    rule alone, and does not silently fold in the non-idle filter that every arm shares.

    `row_w` is the per-ROW AWR weight of each returned block, sliced by the same keep mask as the
    rows so the two are 1:1 by position. This is what lets quality be ranked per ROW
    (`delta_quality_bins`) rather than only per episode (`progress_ratio`) -- the unit the CFG
    quality tags condition on and the schedule's bin reporting uses.

    `corpus_rows` is the final `global_base`: the length of the CONCATENATED corpus this
    enumeration actually walked, i.e. the authoritative size of the index space every returned row
    lives in. It is returned rather than discarded because `meta["n_rows"]` is what the online tier
    binds the artifact to the dataset with (`ScheduleSampler.check_dataset_rows`), and the only
    other candidate -- the weights file's own `n_rows` header -- describes the corpus the WEIGHTS
    were built on, which need not be this one. Build with `--roots roots_OLD --weights weights_NEW`
    and every per-episode check here still passes (episodes absent from `ranges` hit the `continue`
    above before the weights are ever consulted), so a header-sourced `n_rows` would record the new
    corpus's length against indices computed in the old one -- and `check_dataset_rows` would then
    wave through a schedule whose every row means a different frame, which is precisely the case it
    exists to catch.
    """
    ranges = json.loads(Path(ranges_path).read_text())
    episode_rows: list[np.ndarray] = []
    row_w: list[np.ndarray] = []
    progress: list[float] = []
    unlabelled: list[bool] = []
    n_trainable = 0
    global_base = 0
    for task_id, root in _ordered_roots(roots_path):
        episodes, total = _episode_offsets(root)
        for ep_index in sorted(episodes):
            local_start, length = episodes[ep_index]
            key = episode_key(task_id, ep_index)
            frames = _frames_in_spans(ranges.get(key, []), length)
            if frames.size == 0:
                continue
            st = stats.get(key)
            if st is None:
                raise ValueError(
                    f"{key} has trainable rows but no AWR weights. Every sampleable episode must "
                    f"carry a weight, even the neutral 1.0 -- a missing key would otherwise be "
                    f"filled by a reader default and silently reweight training."
                )
            if st["n"] != length:
                raise ValueError(
                    f"{key}: weights have {st['n']} entries but the episode has {length} rows. "
                    f"The weights artifact was built against different pretraining data."
                )
            block = global_base + local_start + frames
            n_trainable += int(frames.size)
            keep = st["keep_mask"][frames]
            episode_rows.append(block[keep] if drop else block)
            # Per-row weights for the DRAWABLE rows only, sliced by exactly the same mask as the
            # rows themselves so the two stay 1:1. `draw_schedule` bins these per row; slicing them
            # differently (or not at all, under `drop`) would tilt toward the wrong rows.
            w_ep = st["w"][frames]
            row_w.append(w_ep[keep] if drop else w_ep)
            # the share of THIS episode's trainable rows that move forward -- an absolute,
            # cross-task-comparable quality, see `quality_bins`.
            progress.append(float(st["progress_mask"][frames].mean()))
            unlabelled.append(bool(st["unlabelled"]))
        global_base += total
    return (
        episode_rows,
        np.asarray(progress, dtype=np.float64),
        np.asarray(unlabelled, dtype=bool),
        n_trainable,
        global_base,
        row_w,
    )


def _file_digest(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _config_hash(args) -> str:
    """Hash the inputs that determine the artifact, so a stale schedule is detectable.

    CONTENTS, not just the paths: the weights and ranges files are regenerated in place as the
    corpus grows, and a hash over filenames alone would call the old and the new schedule the same
    thing -- which is exactly the case the hash exists to catch.

    Hashed as `json.dumps` of an explicit, sorted-key dict, not concatenated fields: concatenation
    has no delimiter between fields, so e.g. `--total-steps 2 --batch 45` and `--total-steps 24
    --batch 5` would hash identically (`"2" + "45"` == `"24" + "5"`).

    `N_BINS`/`LAMBDA_MAX`/`RAMP_START_FRAC` are included even though they are not CLI arguments,
    and even though no current mode reads them in `draw_schedule`: they are part of the schedule
    format's constant surface, and this hash is deliberately conservative about what counts as
    "the same generator" rather than tracking exactly which constants the current code path uses.

    THE GENERATOR SOURCE is hashed for the same reason, one level up. `draw_schedule` and
    `quality_bins` are what actually decide every drawn index, and an edit to either moves every
    row while leaving mode/steps/batch/seed and all three input digests untouched, so the source
    digest is what keeps two different generators from sharing one identity.

    This is DELIBERATELY conservative: the digest is over the whole file, so a comment or
    docstring edit changes `config_hash` even though no index moves. Do not "optimize" this into
    hashing an AST, the function bodies, or a hand-maintained version constant. A false "these
    differ" costs one rebuild and is immediately visible; a false "these match" silently corrupts
    an experiment record, and nothing downstream can detect it.
    """
    payload = {
        "mode": args.mode,
        # Which fraction `drop_top` kept. Two schedules that differ only in this select entirely
        # different corpora, so omitting it would let them claim one identity.
        "keep_top_frac": float(getattr(args, "keep_top_frac", DEFAULT_KEEP_TOP_FRAC)),
        "total_steps": int(args.total_steps),
        "batch": int(args.batch),
        "seed": int(args.seed),
        "n_bins": N_BINS,
        "lambda_max": LAMBDA_MAX,
        "ramp_start_frac": RAMP_START_FRAC,
        "generator_digest": _generator_digest(),
        "weights_digest": _file_digest(args.weights),
        "roots_digest": _file_digest(args.roots),
        "ranges_digest": _file_digest(args.ranges),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build an index-schedule artifact for one run.")
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--weights", required=True, help="AWR weights JSON for this run's reward")
    ap.add_argument("--roots", required=True, help="roots-index JSON for the pretraining data")
    ap.add_argument("--ranges", required=True, help="non-idle sample-ranges JSON")
    ap.add_argument("--out", required=True)
    ap.add_argument("--total-steps", type=int, default=20_605)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    # Read ONLY by `drop_top`; see index_schedule.DEFAULT_KEEP_TOP_FRAC for why the default is 0.50.
    ap.add_argument("--keep-top-frac", type=float, default=DEFAULT_KEEP_TOP_FRAC)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    stats = episode_stats_from_weights(a.weights, tau=DEFAULT_TAU, cap=DEFAULT_CAP,
                                       horizon=DEFAULT_HORIZON)
    header = _weights_header(a.weights)
    reward_id = header.get("reward_id")
    if reward_id is None:
        raise ValueError(
            f"{a.weights} has no 'reward_id' in its header; a schedule cannot record which run "
            f"it was built from without one"
        )
    horizon = int(header.get("horizon", DEFAULT_HORIZON))
    # `drop` filters inside `episode_rows` (its rule is per row and needs no global pass).
    # `drop_top` cannot: its cut is a rank over the WHOLE training set, so every row must be
    # enumerated before any can be judged. Hence enumerate unfiltered, then mask.
    rows_by_ep, progress, unlab, n_trainable, n_rows, row_w = episode_rows(
        a.roots, a.ranges, stats, drop=a.mode == "drop"
    )
    if a.mode in ("drop_top", "drop_random"):
        # Same cardinality, different rule: `drop_random` is `drop_top`'s control, so it must keep
        # the SAME NUMBER of rows and differ only in which. Seeded off the schedule's own seed so
        # the artifact is reproducible from its config_hash.
        keep = (top_fraction_mask(row_w, a.keep_top_frac) if a.mode == "drop_top"
                else random_fraction_mask(row_w, a.keep_top_frac, a.seed))
        rows_by_ep = [b[m] for b, m in zip(rows_by_ep, keep)]
        row_w = [w[m] for w, m in zip(row_w, keep)]
    # `n_rows` is the corpus THIS enumeration walked, never the weights file's claim about some
    # corpus. The header's number is checked against it rather than used: a disagreement means the
    # weights and the roots index describe different corpora (e.g. weights rebuilt after the corpus
    # grew, roots not), and since the rows below index the roots' space, recording the header's
    # length would make `ScheduleSampler.check_dataset_rows` pass against the GROWN dataset while
    # every index points at a different frame. See `episode_rows`.
    declared = header.get("n_rows")
    if declared is not None and int(declared) != int(n_rows):
        raise ValueError(
            f"{a.weights} says the pretraining data has {int(declared)} rows but {a.roots} enumerates "
            f"{int(n_rows)}. The weights and the roots index were built against different "
            f"pretraining data, so the rows drawn here index one space while the run would be bound to "
            f"the other; rebuild whichever is stale rather than trusting either number."
        )
    # `q_ep` stays the reported per-episode quality (the manifest's quintile breakdown reads it),
    # but `row_bins` below is what the schedule's bin reporting and the CFG quality tags actually
    # rank on -- see `delta_quality_bins`: most of the weight variance sits within episodes rather
    # than between them.
    q_ep = quality_bins(progress, unlab)
    row_bins = delta_quality_bins(row_w)
    kept = sum(len(r) for r in rows_by_ep)
    rows = draw_schedule(rows_by_ep, mode=a.mode, total_steps=a.total_steps,
                         batch=a.batch, seed=a.seed)
    meta = write_schedule(a.out, rows, mode=a.mode, seed=a.seed, total_steps=a.total_steps,
                          batch=a.batch, row_bins=row_bins, episode_rows=rows_by_ep,
                          keep_fraction=kept / max(n_trainable, 1),
                          config_hash=_config_hash(a),
                          reward_id=reward_id, horizon=horizon, n_rows=int(n_rows))
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
