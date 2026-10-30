"""Episode-granularity AWR weights, derived from the row-level artifact.

OFFLINE TIER: numpy only.

The granularity ablation's AWR arm tests whether episode-level scoring keeps
low-task-progress-but-transferable segments -- recoveries, repositioning -- at full weight,
where the row-level arm suppresses them; the cost is that bad segments of good episodes ride
along. One variable changed: each episode's weight is a constant broadcast over its rows, so the
artifact format and the online loss reweighting (per-batch renormalisation included) are
untouched.

The episode score is `pratio_marginal`: score every episode by its P-RATIO (share of rows with
`w > 1` on the float32 weight, i.e. Delta > ~6e-9 rather than Delta > 0 -- float noise counts as
no progress, see build_index_schedule.episode_stats_from_weights -- the same segment-based
statistic the episode CFG tag and the demo selection use, robust to reward-magnitude
fluctuation), then assign each episode a CONSTANT weight by
quantile-matching the row-level arm's weight distribution: walk episodes in P-order and give
each the row-arm weight quantile at its cumulative row-mass midpoint. The episode arm then
carries the IDENTICAL weight histogram over rows as the row arm -- same effective
reweighting mass, same spread -- and the only thing that changes is attachment (constant
within episode, ordered by P-ratio, versus per-row). Zero new knobs. An episode whose rows are
all exactly 1.0 stays all-1.0 (the masked-or-flat conflation the row file cannot resolve).

reward_id is `<source>_ep` (e.g. `phase_ep`): the reward-id string is how every downstream
guard tells arms apart (weights formula fingerprint, schedule filename bind), so the
granularity rides the established discriminator and nothing online changes. The formula
fingerprint is recomputed over the source params plus `granularity="episode"` and
`source_reward_id`, so the two granularities can never share a fingerprint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from axis_qvla.signals.provenance import stamp

_HEADER_FIELDS = ("version", "reward_id", "tau", "cap", "horizon")


def pratio_marginal_weights(weights: dict, *, cap: float, ranges: dict) -> dict[str, float]:
    """Constant per-episode weights: P-ratio rank, row-arm weight marginal.

    The P-ratio here is EXACTLY the episode CFG tag's input: the share of an episode's
    TRAINABLE (non-idle) rows with `w > 1` (float32, so Delta > ~6e-9, not Delta > 0),
    restricted by the same `_frames_in_spans` walk
    `episode_rows` uses -- including idle frames would change the denominator and quietly make
    the episode column two different scores. The weight marginal and the row-mass are likewise
    over trainable rows only, since those are the rows the row-level arm actually reweights.

    Episodes ordered by (P-ratio, key) -- the key tie-break keeps the artifact reproducible.
    Each episode takes the quantile of the SORTED row-arm weights at its cumulative row-mass
    midpoint, so the row-level weight histogram is preserved exactly up to episode-constant
    attachment. All-1.0 (masked-or-flat) episodes and episodes with no trainable rows stay 1.0
    and sit outside both the ordering and the marginal, mirroring the unlabelled handling
    everywhere else.
    """
    from axis_qvla.signals.methods.schedule import _frames_in_spans

    trainable: dict[str, np.ndarray] = {}
    for k, arr in weights.items():
        a = np.asarray(arr, dtype=np.float64)
        frames = _frames_in_spans(ranges.get(k, []), len(a))
        if frames.size == 0 or bool(np.all(a == 1.0)):
            continue
        trainable[k] = a[frames]
    pr = {k: float((v > 1.0).mean()) for k, v in trainable.items()}
    marginal = np.sort(np.concatenate(list(trainable.values())))
    total = marginal.size
    out: dict[str, float] = {k: 1.0 for k in weights if k not in trainable}
    cum = 0
    for k in sorted(trainable, key=lambda k: (pr[k], k)):
        mid = cum + trainable[k].size / 2.0
        q = np.clip(int(mid), 0, total - 1)
        out[k] = float(min(marginal[q], cap))
        cum += trainable[k].size
    return out


def check_source(src, name) -> None:
    """Refuse a row-level weights document that cannot attribute its arm: every header field and a
    full provenance stamp, or an episode artifact derived from it could not say which corpus or
    formula it describes."""
    for f in _HEADER_FIELDS:
        if f not in src:
            raise ValueError(f"{name} has no header field {f!r}; cannot attribute the run")
    prov = src.get("provenance")
    if not prov or "corpus_fingerprint" not in prov or "params" not in prov:
        raise ValueError(
            f"{name} carries no full provenance stamp; an episode artifact derived from it "
            f"could not say which pretraining data or formula it describes"
        )


def episode_document(src, ep_w, *, score: str, source_sha256: str) -> dict:
    """The episode-weights document of a checked source (`check_source`): each episode's constant
    weight `ep_w[key]` broadcast over its rows, the header naming the granularity and the source,
    and a provenance stamp whose formula params are the SOURCE's verbatim plus the granularity
    fields, so the fingerprint never collides with the row-level file's and stays
    self-consistent under pretrain_dataset's recompute-and-compare."""
    prov = src["provenance"]
    tau, cap = float(src["tau"]), float(src["cap"])
    src_reward = str(src["reward_id"])
    reward_id = f"{src_reward}_ep"
    out_w: dict[str, list[float]] = {}
    n_rows = n_neutral = 0
    for key, arr in src["weights"].items():
        w_ep = ep_w[key]
        out_w[key] = [w_ep] * len(arr)
        n_rows += len(arr)
        if w_ep == 1.0:
            n_neutral += len(arr)
    if int(src["n_rows"]) != n_rows:
        raise ValueError(
            f"source header claims {src['n_rows']} rows but its weight arrays hold {n_rows}; "
            f"the file disagrees with itself"
        )
    params = dict(prov["params"])
    params["granularity"] = "episode"
    params["episode_score"] = score
    params["source_reward_id"] = src_reward
    doc = {
        "version": 1,
        "reward_id": reward_id,
        "tau": tau,
        "cap": cap,
        "horizon": int(src["horizon"]),
        "n_episodes": len(out_w),
        "n_rows": n_rows,
        "n_neutral_rows": n_neutral,
        "granularity": "episode",
        "episode_score": score,
        "source_weights_sha256": source_sha256,
        "weights": out_w,
    }
    return stamp(doc, fingerprint=prov["corpus_fingerprint"], reward_id=reward_id, **params)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Broadcast episode-level AWR weights from a row-level weights artifact.")
    ap.add_argument("--weights", required=True, help="row-level AWR weights JSON (the source)")
    ap.add_argument("--out", required=True, help="awr_weights_<reward>_ep.json")
    ap.add_argument(
        "--ranges", help="non-idle sample-ranges JSON; REQUIRED: the P-ratio is over trainable "
        "rows exactly as the episode CFG tag computes it")
    a = ap.parse_args(argv)

    src_path = Path(a.weights)
    src = json.loads(src_path.read_text())
    check_source(src, a.weights)
    cap = float(src["cap"])
    if not a.ranges:
        raise ValueError(
            "--ranges is required for pratio_marginal: without the non-idle spans the "
            "P-ratio's denominator includes idle frames and the episode column stops "
            "being one score"
        )
    ranges = json.loads(Path(a.ranges).read_text())
    ep_w = pratio_marginal_weights(src["weights"], cap=cap, ranges=ranges)
    payload = episode_document(src, ep_w, score="pratio_marginal",
                               source_sha256=hashlib.sha256(src_path.read_bytes()).hexdigest())
    scores = [ep_w[key] for key in src["weights"]]
    reward_id, out_w = payload["reward_id"], payload["weights"]
    n_rows, n_neutral = payload["n_rows"], payload["n_neutral_rows"]
    n_capped_eps = sum(w_ep == cap for w_ep in scores)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(out)
    s = np.asarray(scores)
    print(json.dumps({
        "reward_id": reward_id, "n_episodes": len(out_w), "n_rows": n_rows,
        "n_neutral_rows": n_neutral, "n_capped_episodes": n_capped_eps,
        "w_ep_min": float(s.min()), "w_ep_mean": float(s.mean()), "w_ep_max": float(s.max()),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
