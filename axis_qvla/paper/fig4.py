"""Fig. 4(a) and 4(b) from published inputs: the episode progress ratio P_e of the 5,000 corpus
episodes under both rewards, and its rank correlation with human judgments.

Why: neither figure comes from rollouts, so T0 does not cover them. Both recompute from files
anyone can download (`qvla data get corpus --include 'index/*' --include 'signals/v1/*'`):

  P_e (Fig. 4a)  the share of an episode's trainable rows whose AWR row weight exceeds 1, i.e.
                 whose clipped advantage is positive: an exact count / n over the half-open ranges
                 of index/ranges_libero_5k_v2.json, with the weights of
                 signals/v1/<reward>/chunk/awr_weights.json.
  rho (Fig. 4b)  Spearman's rank correlation (average ranks for ties) between P_e and H, the share
                 of an episode's decidable trainable steps the judges did not mark unproductive
                 (1 - median U over qualifying judges), over the human-study episodes, with a
                 paired-episode percentile bootstrap (numpy default_rng(seed)). The original analysis
                 drew a second resample of `aux_draws_per_resample` indices in the same loop; it is
                 drawn here too, so the published interval reproduces to within one ulp (measured on
                 numpy 1.26.4 and 2.4.4); `--check` holds rho to 1e-12 and the interval ends to 1e-9.
                 experiments/paper/fig4b_human_study.json holds H per episode and the study's rules;
                 the judges' answers are not published.

`qvla paper label-study --pretraining-data <dir> --check` compares with experiments/paper/fig4.expected.json.

Core tier at module scope (Fig. 4a is standard library only); `spearman`, `paired_bootstrap` and
`fig4b` import numpy when called.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

REWARDS = {"object_centric": "signals/v1/object_centric/chunk/awr_weights.json",
           "approach_aware": "signals/v1/approach_aware/chunk/awr_weights.json"}
RANGES = "index/ranges_libero_5k_v2.json"
QUANTILES = ("0.1", "0.25", "0.5", "0.75", "0.9")


def _key(k: str) -> tuple[int, int]:
    t, e = k.removeprefix("task_").split("--")
    return int(t), int(e)


def episode_pe(weights: Mapping[str, list[float]], ranges: Mapping[str, list[list[int]]]) -> dict[tuple[int, int], float]:
    out = {}
    for key, spans in ranges.items():
        w = weights[key]
        rows = [i for a, b in spans for i in range(a, b)]
        k = 0
        for i in rows:              # an exact count (no builtin sum: this module is on T0's import
            k += w[i] > 1.0         # closure, tests/golden/paper/test_t0_version_independence.py)
        out[_key(key)] = k / len(rows)
    return out


def _quantiles(values: list[float]) -> dict[str, float]:
    v = sorted(values)
    return {q: v[round(float(q) * (len(v) - 1))] for q in QUANTILES}


def load_pe(corpus: Path) -> dict[str, dict[tuple[int, int], float]]:
    ranges = json.loads((Path(corpus) / RANGES).read_bytes())
    return {r: episode_pe(json.loads((Path(corpus) / rel).read_bytes())["weights"], ranges)
            for r, rel in REWARDS.items()}


def fig4a(corpus: Path, pe: dict | None = None) -> tuple[bytes, dict]:
    pe = load_pe(corpus) if pe is None else pe
    keys = sorted(pe["object_centric"])
    lines = ["task_id,episode_index,P_object_centric,P_approach_aware"]
    lines += [f"{t},{e},{pe['object_centric'][(t, e)]!r},{pe['approach_aware'][(t, e)]!r}" for t, e in keys]
    data = ("\n".join(lines) + "\n").encode()
    summary = {"sha256": hashlib.sha256(data).hexdigest()}
    summary.update({r: {"n": len(v), "quantiles": _quantiles(list(v.values()))} for r, v in pe.items()})
    return data, summary


def _ranks(x):
    import numpy as np

    order = np.argsort(x, kind="mergesort")
    xs, r, i = x[order], np.empty(len(x)), 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y) -> float:
    import numpy as np

    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rx, ry = _ranks(x), _ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())    # 0 when a resample's values are all tied
    return float((rx * ry).sum() / den) if den else float("nan")


def paired_bootstrap(ps: Mapping[str, list[float]], h: list[float], *, n_boot: int, seed: int,
                     aux_draw: int) -> dict[str, tuple[float, float, int]]:
    """Per reward: (2.5th, 97.5th percentile, resamples with an undefined rho). One index draw per
    resample serves every reward, as in the original analysis."""
    import numpy as np

    rng = np.random.default_rng(seed)
    h = np.asarray(h, float)
    arrays = {r: np.asarray(v, float) for r, v in ps.items()}
    vals: dict[str, list[float]] = {r: [] for r in ps}
    na = dict.fromkeys(ps, 0)
    n = len(h)
    for _ in range(n_boot):
        ic = rng.integers(0, n, n)
        if aux_draw:
            rng.integers(0, aux_draw, aux_draw)
        for r, p in arrays.items():
            v = spearman(p[ic], h[ic])
            if v == v:
                vals[r].append(v)
            else:
                na[r] += 1
    return {r: (float(np.percentile(vals[r], 2.5)), float(np.percentile(vals[r], 97.5)), na[r]) for r in ps}


def fig4b(pe: Mapping[str, Mapping[tuple[int, int], float]], study: Mapping) -> dict:
    if study.get("schema") != "qvla.human_study/1":
        raise ValueError("not a qvla.human_study/1 file")
    eps = [e for e in study["episodes"] if e["in_correlation_set"]]
    keys = [(int(e["task_id"]), int(e["episode_index"])) for e in eps]
    h = [float(e["H"]) for e in eps]
    ps = {r: [pe[r][k] for k in keys] for r in REWARDS}
    prm = study["params"]
    boot = paired_bootstrap(ps, h, n_boot=int(prm["n_boot"]), seed=int(prm["seed"]),
                            aux_draw=int(prm["aux_draws_per_resample"]))
    return {r: {"N": len(keys), "rho": spearman(ps[r], h), "ci95": [boot[r][0], boot[r][1]],
                "bootstrap_na_resamples": boot[r][2]} for r in REWARDS}


def check(summary: Mapping, expected: Mapping) -> list[str]:
    """Fig. 4(a) exactly (standard-library arithmetic); Fig. 4(b) within the expected tolerances
    (numpy's arithmetic may differ in the last bits between versions)."""
    out = []
    if "fig4a" in summary:
        got, want = summary["fig4a"], expected["fig4a"]
        if got["sha256"] != want["sha256"]:
            out.append(f"fig4a.sha256: {got['sha256']} != {want['sha256']}")
        for r in REWARDS:
            if got[r] != want[r]:
                out.append(f"fig4a.{r}: {got[r]} != {want[r]}")
    if "fig4b" in summary:
        tol = expected["tolerance"]
        for r in REWARDS:
            g, w = summary["fig4b"][r], expected["fig4b"][r]
            # `not (... <= tol)` rather than `... > tol`: a NaN rho or interval end (an all-tied
            # resample, say) makes every `>` comparison False, so the naive form reports no
            # difference for a bootstrap that silently failed. `not (nan <= tol)` is True.
            if g["N"] != w["N"] or not (abs(g["rho"] - w["rho"]) <= tol["rho"]):
                out.append(f"fig4b.{r}: N {g['N']} rho {g['rho']!r}, expected N {w['N']} rho {w['rho']!r}")
            if any(not (abs(a - b) <= tol["ci"]) for a, b in zip(g["ci95"], w["ci95"])):
                out.append(f"fig4b.{r}.ci95: {g['ci95']} != {w['ci95']}")
    return out
