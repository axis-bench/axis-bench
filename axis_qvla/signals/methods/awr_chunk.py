"""Build the two AWR weight artifacts that define the two supervision arms.

arm A `v2`    : weights from the stored corpus reward (0.8*progress + 0.2*contact)
arm B `phase` : weights from the contact-conditioned two-phase reward

Everything else about the two runs is identical, so these two files ARE the experiment. They are
written with a provenance header naming the reward, tau, cap and horizon, because a weights file
that cannot be attributed to an arm is worthless after the fact.

WHY THE PHASE REWARD IS RECOMPUTED HERE AND NOT READ. The corpus stores only the v2 reward. The
phase reward needs t0 (first target contact), the object body, and hand-object distance; the
object columns are read through the per-episode `meta/episodes` fields (`object_body`,
`target_body`, `body_names`), since a task's scene variants can differ in body count; the
task-level `info.json` list is not used.

Offline tier. Run under the offline environment.

THE RELEASED FILES (paper Sec. IV.A-C; conf/signals/generators.json):

  object_centric.chunk.awr_weights   --reward v2
  approach_aware.chunk.awr_weights   --reward phase --binding names --phase-balanced \
                                     --names-map conf/signals/episode_body_names.json \
                                     --names-sha256 <its pinned digest>

both with the default tau 10 / cap 2 / horizon 10. Each rebuilds byte for byte. The names binding
reads the object's pose column through per-episode `body_names`/`object_body`, which the pretraining
data's fingerprint does not cover; the shipped table holds the per-episode names the released file
was built with, and `--names-sha256` refuses a build that would bind any others
(`axis_qvla.data.episode_names`). The approach-aware arm is Eq. (3) with windows truncated at the contact
boundary, rescaled by (H-1)/h_t and zeroed below h_t = 3 (`phase_delta_clamped`), PLUS a per-phase
quantile scale applied to Delta before w = min(exp(10 Delta), 2) (`--phase-balanced`; stamped into
the file's `provenance.phase_balance`). Without `--phase-balanced` the phase reward is a different
weighting that no paper arm trained on. The paper's other quality signals derive from these two
files: build_awr_weights_episode (episode weights), build_quality_labels (tags),
build_index_schedule (filtering), build_lerobot_signals (the LeRobot views).

The released approach-aware file was built with this module's reward maths: conf/release/SIGNALS.lock.json
pins its bytes, and conf/signals/generators.json the generator digest its metadata records.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from axis_qvla.data import episode_names
from axis_qvla.signals.blocks.awr import (
    DEFAULT_CAP, DEFAULT_HORIZON, DEFAULT_TAU, EpisodeWeights, build_episode_weights,
    episode_key, neutral_episode, write_manifest,
)
# The phase curve is a label (axis_qvla.labels.phase); this builder, REALALIGN and the paper-rule tests
# keep reading it through this module.
from axis_qvla.labels.phase import (  # noqa: F401
    BONUS, EPS, K, LAMBDA, MIN_RUN, W_CONTACT, W_PROGRESS, _contact_tanh, banded_phase_curve, first_contact,
    phase_reward,
)
from axis_qvla.signals.numerics import tanh_f32  # noqa: F401  (kept on this module's surface)

CORPUS_ROOT_ENV = "AXIS_PRETRAIN_CORPUS_ROOT"
# Windows shorter than this many frames are set to Delta = 0 (too short to extrapolate); the
# per-phase scale fit pools at least PHASE_FIT_MIN_FRAMES trusted frames per phase; the pooled
# capped rate it matches must lie strictly inside (0, MAX_CAPPED_RATE).
MIN_WINDOW_FRAMES = 3
PHASE_FIT_MIN_FRAMES = 10_000
MAX_CAPPED_RATE = 0.5


def _default_index() -> str | None:
    """The corpus root is machine-specific, so it comes from the environment, never a literal.
    Returning None makes --index effectively required when the variable is unset, which fails
    loudly at argparse rather than silently reading some other machine's corpus."""
    root = os.environ.get(CORPUS_ROOT_ENV)
    return str(Path(root) / "roots_relative.json") if root else None


def episode_meta(task_dir: Path) -> pd.DataFrame:
    fs = sorted(glob.glob(str(task_dir / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def _neutralised_mask_clamped(valid: np.ndarray, t0: int, horizon: int) -> np.ndarray:
    """Validity mask matched to the CLAMPED windows: test the end each window actually reads.

    `neutralise_invalid` tests `v[t]` and `v[t+H-1]` -- the unclamped read end. Under clamping
    a phase-1 window near t0 reads `curve[t0-1]`, so testing `v[t+H-1]` (past t0) wrongly
    neutralises trustworthy approach frames whenever the validity clip lands just after
    contact, and would conversely trust a window whose true read end is invalid if validity
    ever had interior holes. Test `v[t]` and `v[min(t+H-1, segment_end)]` per segment.
    """
    v = np.asarray(valid, bool)
    out = np.empty(len(v), dtype=bool)
    for a_, b_ in ((0, t0), (t0, len(v))):
        if b_ <= a_:
            continue
        idx = np.arange(a_, b_)
        end = np.minimum(idx + horizon - 1, b_ - 1)
        out[a_:b_] = ~v[idx] | ~v[end]
    return out


def phase_delta_clamped(curve: np.ndarray, t0: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Forward advantage with windows CLAMPED to their own phase, LENGTH-NORMALISED.

    A horizon window spanning t0 reads the +0.10 contact bonus as Delta >= 0.1 and saturates
    `exp(tau * Delta)` at the cap -- an artifact of the band encoding, not a statement about the
    demonstration. Clamping alone installs the opposite artifact: the
    last horizon-1 frames of each segment get shrinking windows, so Delta tapers deterministically
    to 0 exactly at the pre-grasp moment. Shortened windows are therefore rescaled to the full
    window's per-frame rate (Delta * (H-1)/span); windows shorter than 3 frames carry too little
    signal to extrapolate and are set to 0 (2 boundary frames per segment, honestly neutral).

    Returns `(delta, fit_ok)`: `fit_ok` marks frames whose window is long enough to trust --
    the SCALE FIT must pool only these (and only mask-surviving frames), or the boundary zeros
    and to-be-neutralised frames contaminate the per-phase statistics.
    """
    from axis_qvla.signals.blocks.awr import forward_delta
    parts, oks = [], []
    full = max(horizon - 1, 1)
    for seg in (curve[:t0], curve[t0:]):
        T = len(seg)
        if T == 0:
            parts.append(np.empty(0)); oks.append(np.empty(0, bool)); continue
        d = forward_delta(seg, horizon)
        span = np.minimum(full, (T - 1) - np.arange(T))
        ok = span >= MIN_WINDOW_FRAMES
        d = np.where(ok, d * (full / np.maximum(span, 1)), 0.0)
        parts.append(d); oks.append(ok)
    return np.concatenate(parts), np.concatenate(oks)


def phase_balance(pending, *, tau: float, cap: float, horizon: int) -> tuple[dict, dict]:
    """The per-phase quantile match of the approach-aware weights: `(scales, {slot: EpisodeWeights})`.

    `pending` holds `(slot, task_id, episode_index, curve, valid, t0)` per phase-scored episode. The
    fit pools ONLY frames the weighting will exponentiate: windows long enough to trust (fit_ok from
    the length-normalised clamping) and frames the validity mask will not force to 1.0. Quantile
    matching equalises the capped rate per phase BY CONSTRUCTION at the pooled rate. Scales are
    returned at full precision: a rounded stamp cannot reproduce the artifact.
    """
    from axis_qvla.signals.blocks.awr import AwrWeightError, awr_weight

    thr = float(np.log(cap) / tau)
    deltas, masks = {}, {}
    pool1, pool2 = [], []
    for slot, tid, ep, curve, valid, t0 in pending:
        d, ok = phase_delta_clamped(curve, t0, horizon)
        nm = _neutralised_mask_clamped(valid, t0, horizon)
        deltas[slot], masks[slot] = d, nm
        fit = ok & ~nm
        pool1.append(d[:t0][fit[:t0]])
        pool2.append(d[t0:][fit[t0:]])
    p1, p2 = np.concatenate(pool1), np.concatenate(pool2)
    if min(p1.size, p2.size) < PHASE_FIT_MIN_FRAMES:
        raise AwrWeightError(f"per-phase fit pools too small (p1={p1.size} p2={p2.size}); the branch "
                             f"taxonomy has collapsed one phase and a quantile fit would be noise")
    if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
        raise AwrWeightError("non-finite delta in the scale-fit pool")
    target = float(np.mean(np.concatenate([p1, p2]) >= thr))
    if not 0.0 < target < MAX_CAPPED_RATE:
        raise AwrWeightError(f"degenerate pooled capped rate {target:.4f}")
    q1 = float(np.quantile(p1, 1 - target))
    q2 = float(np.quantile(p2, 1 - target))
    if min(q1, q2) <= 1e-9 or not np.isfinite(q1 + q2):
        raise AwrWeightError(f"degenerate per-phase quantiles q1={q1} q2={q2}")
    sc1, sc2 = thr / q1, thr / q2
    scales = {"mode": "clamped-lennorm+quantile-match", "cap_threshold": thr, "target_capped_rate": target,
              "q1": q1, "q2": q2, "scale1": sc1, "scale2": sc2}
    filled = {}
    for slot, tid, ep, curve, valid, t0 in pending:
        d = deltas[slot].copy()
        d[:t0] *= sc1
        d[t0:] *= sc2
        nm = masks[slot]
        w = np.where(nm, 1.0, awr_weight(d, tau, cap)).astype(np.float32)
        if not np.isfinite(w).all():
            raise AwrWeightError(f"non-finite balanced weight in task_{tid}--{ep}")
        filled[slot] = EpisodeWeights(key=episode_key(tid, ep), weights=w, n_neutral=int(nm.sum()))
    return scales, filled


def chunk_weights(tasks, curve_of, *, names: dict | None, reads_meta: bool, phase_balanced: bool,
                  tau: float, cap: float, horizon: int) -> tuple[list[EpisodeWeights], dict | None, int]:
    """The chunk weights of every episode of `tasks`, in their order: `(episodes, scales, n_neutral)`.

    `tasks` yields `(task_id, frames, meta)`: the task's label columns and its `meta/episodes`
    indexed by `episode_index`. `curve_of(task_id, episode, frames, meta_row)` returns the reward
    curve and its band boundary (`None` for the stored reward, which is not clamped), or None when
    the episode cannot be scored: it then trains at weight 1.0 everywhere and is counted. With
    `reads_meta` each episode's meta row is passed, its `body_names`/`object_body` replaced by
    `names[key]` where the table has the episode. With `phase_balanced` the weights wait for the
    corpus-level per-phase scales (`phase_balance`, returned as `scales`); otherwise each
    episode's weights are `build_episode_weights` of its own curve.
    """
    eps: list[EpisodeWeights | None] = []
    pending: list[tuple] = []         # (slot in eps, tid, ep, curve, valid, t0)
    n_neutral = 0
    for tid, frames, meta in tasks:
        tid = int(tid)
        for ep, g in frames.groupby("episode_index"):
            ep = int(ep)
            g = g.sort_values("frame_index").reset_index(drop=True)
            valid = g["reward_valid"].to_numpy(bool)
            mrow = {}
            if reads_meta:
                mrow = meta.loc[ep].to_dict() if ep in meta.index else {}
                if names is not None and episode_key(tid, ep) in names:
                    bound, obj = names[episode_key(tid, ep)]
                    mrow = {**mrow, "body_names": bound, "object_body": obj}
            res = curve_of(tid, ep, g, mrow)
            if res is None:
                # Never invent a reward: neutral is the honest weight, and it makes the
                # episode behave exactly as it does in the plain-BC baseline.
                eps.append(neutral_episode(tid, ep, len(g)))
                n_neutral += 1
                continue
            # t0 is the band boundary the BRANCH chose (0 = all-manipulation, len =
            # all-approach); every downstream consumer must use it, never a recomputed contact
            # time -- the branch and the recompute can disagree.
            curve, t0 = res
            if phase_balanced:
                eps.append(None)  # slot filled after the corpus-level scales exist
                pending.append((len(eps) - 1, tid, ep, np.asarray(curve, float), valid.copy(), int(t0)))
                continue
            eps.append(build_episode_weights(tid, ep, curve, valid, horizon=horizon, tau=tau, cap=cap))
    scales = None
    if pending:
        scales, filled = phase_balance(pending, tau=tau, cap=cap, horizon=horizon)
        for slot, ew in filled.items():
            eps[slot] = ew
    assert all(e is not None for e in eps)
    return eps, scales, n_neutral


def _corpus_root(a):
    """The corpus these weights describe, for the provenance stamp.

    Taken from the roots index's own `base`, which is stored RELATIVE to the index file, so the
    stamp is computed over the same tree the weights were read from rather than a path guessed
    from the environment.
    """
    import json as _j

    idx = Path(a.index) if getattr(a, "index", None) else None
    if idx is None or not idx.exists():
        return None
    try:
        base = _j.loads(idx.read_text()).get("base")
    except Exception:  # noqa: BLE001 - an unreadable index just means no stamp
        return None
    return (idx.parent / base).resolve() if base else None


def _bound_names(a) -> dict | None:
    """The (body_names, object_body) table the names binding will read, settled BEFORE the data
    pass: the corpus meta, or `--names-map` in its place (which must cover every corpus episode).
    Its content digest is printed, and with `--names-sha256` anything else is refused -- the
    corpus fingerprint does not cover these fields, so without this a stale-names corpus builds
    different weights under the same fingerprint (`axis_qvla.data.episode_names`). Returns the map's
    table when one replaces the meta, else None."""
    table = episode_names.corpus_table(a.index)
    source = "pretraining-data meta"
    override = None
    if a.names_map is not None:
        shipped = episode_names.load_table(a.names_map)
        missing = sorted(set(table) - set(shipped), key=episode_names.key_order)
        if missing:
            sys.exit(f"refusing: {len(missing)} pretraining episodes are not in {a.names_map} "
                     f"(first: {missing[0]}); the names binding would fall back to their meta")
        table = override = {k: shipped[k] for k in table}
        source = str(a.names_map)
    digest = episode_names.table_digest(table)
    print(f"   names table: {len(table)} episodes from {source}, sha256 {digest}", file=sys.stderr)
    if a.names_sha256 is not None and digest != a.names_sha256:
        sys.exit(f"refusing: the names table the binding would read ({source}) digests to "
                 f"{digest}, not the required {a.names_sha256}; pretraining data whose meta "
                 f"carries other per-episode names binds the wrong object -- pass --names-map")
    return override


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", choices=("v2", "phase"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--index",
        default=_default_index(),
        help="roots index JSON; defaults to $AXIS_PRETRAIN_CORPUS_ROOT/roots_relative.json",
    )
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    ap.add_argument("--cap", type=float, default=DEFAULT_CAP)
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    # The object is bound through the per-episode names table (body_names + object_body, see
    # --names-map); `names` is recorded in the weights' formula fingerprint.
    ap.add_argument("--binding", choices=("names",), default="names")
    # The names the binding reads (see _bound_names); phase only.
    ap.add_argument("--names-map", type=Path, default=None,
                    help="per-episode body_names/object_body table to bind through instead of the "
                         "pretraining-data meta (conf/signals/episode_body_names.json)")
    ap.add_argument("--names-sha256", default=None,
                    help="refuse unless the names the binding reads digest to this")
    # Phase balance (phase reward only): windows clamped to their own phase and
    # length-normalised, then Delta multiplied per phase by ln(cap)/tau over that phase's
    # (1 - pooled capped rate) quantile, so both phases cap at the pooled rate (quantile
    # matching, see the scale fit below). Off by
    # default -- turning it on changes what the reward MEANS, and the per-phase scales are
    # stamped into the artifact's provenance.
    ap.add_argument("--phase-balanced", action="store_true")
    # Materialise the per-episode reward CURVES next to the weights (phase only): one npz per
    # task under this directory, `ep_<index>` -> float32 curve, plus t0 in `t0_<index>`. This
    # writes the curves beside the weights without touching the pretraining data's files (whose
    # fingerprint every artifact binds to).
    ap.add_argument("--dump-curves", type=Path, default=None)
    a = ap.parse_args()
    if not a.index:
        ap.error(f"--index is required when ${CORPUS_ROOT_ENV} is unset")
    if a.reward == "phase" and not a.phase_balanced:
        print("note: --phase-balanced not given: these are NOT the paper's approach-aware weights "
              "(approach_aware.chunk.awr_weights); add --phase-balanced to reproduce them",
              file=sys.stderr)
    if a.phase_balanced and a.reward != "phase":
        ap.error("--phase-balanced applies to --reward phase only")
    reads_names = a.reward == "phase"
    if (a.names_map is not None or a.names_sha256 is not None) and not reads_names:
        ap.error("--names-map/--names-sha256 apply where names are read (--reward phase); "
                 "ignoring them would hide which names were bound")

    idx = json.loads(Path(a.index).read_text())
    base = Path(a.index).parent / idx.get("base", ".")
    cols = ["reward", "reward_valid", "contact_target_finger", "contact_target_eef",
            "observation.body_pose", "observation.eef_pose", "episode_index", "frame_index"]
    names_override = _bound_names(a) if reads_names else None

    curve_store: dict[int, dict] = {}

    def tasks():
        for tid, rel in sorted(idx["roots"].items(), key=lambda kv: int(kv[0])):
            td = base / rel
            fs = sorted(glob.glob(str(td / "data" / "**" / "*.parquet"), recursive=True))
            if not fs:
                print(f"WARNING task {tid}: no data parquet", file=sys.stderr)
                continue
            df = pd.concat([pd.read_parquet(f, columns=cols) for f in fs], ignore_index=True)
            yield int(tid), df, episode_meta(td).set_index("episode_index")

    def curve_of(tid, ep, g, mrow):
        if a.reward == "v2":
            return g["reward"].to_numpy(float), None
        res = phase_reward(g, mrow)
        if res is not None and a.dump_curves is not None:
            curve_store.setdefault(tid, {})[ep] = (np.asarray(res[0], np.float32), int(res[1]))
        return res

    eps, scales, n_neutral_eps = chunk_weights(
        tasks(), curve_of, names=names_override, reads_meta=a.reward == "phase", phase_balanced=a.phase_balanced,
        tau=a.tau, cap=a.cap, horizon=a.horizon)
    if scales is not None:
        print(f"   per-phase quantile match: target={scales['target_capped_rate']:.4f} q1={scales['q1']:.5f} "
              f"q2={scales['q2']:.5f} -> scale1={scales['scale1']:.3f} scale2={scales['scale2']:.3f}")

    if a.dump_curves is not None and curve_store:
        a.dump_curves.mkdir(parents=True, exist_ok=True)
        for tid_c, d_c in sorted(curve_store.items()):
            arrs = {}
            for ep_c, (c_c, t0_c) in sorted(d_c.items()):
                arrs[f"ep_{ep_c}"] = c_c
                arrs[f"t0_{ep_c}"] = np.int64(t0_c)
            np.savez_compressed(a.dump_curves / f"task_{tid_c}_phase_curves.npz", **arrs)
        print(f"   dumped curves for {len(curve_store)} tasks -> {a.dump_curves}")

    # binding/balance enter the FORMULA FINGERPRINT for phase (they change the maths); v2's
    # params stay exactly as shipped so its fingerprint is stable.
    extra = ({"binding": a.binding, "phase_balanced": bool(a.phase_balanced)}
             if a.reward == "phase" else {})
    header = write_manifest(Path(a.out), iter(eps), corpus=_corpus_root(a), reward_id=a.reward,
                            tau=a.tau, cap=a.cap, horizon=a.horizon, **extra)
    if scales is not None:
        payload = json.loads(Path(a.out).read_text())
        if "provenance" in payload:
            payload["provenance"]["phase_balance"] = scales
        else:
            # No corpus stamp was written (roots index without a base): do NOT fabricate a
            # half-provenance dict -- a provenance block missing corpus_fingerprint reads as
            # "built against a different corpus" forever. Keep the facts in a sibling key.
            payload["phase_balance"] = scales
        Path(a.out).write_text(json.dumps(payload))
    w = np.concatenate([e.weights for e in eps])
    print(json.dumps(header))
    print(f"   fully-neutral episodes: {n_neutral_eps}/{len(eps)}")
    print(f"   mean w {w.mean():.4f} | at cap {np.mean(w >= a.cap - 1e-9):.1%} | "
          f"w==1 {np.mean(w == 1.0):.1%} | std {w.std():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
