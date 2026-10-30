"""The norm check: a norm-statistics file against the training set it should describe.

check_norm holds every statistic in the file to THIS dataset, per dimension, in the model's state order:
- `mean` and `std` within 5% and 10% of the dimension's own standard deviation;
- `q01` and `q99`, the quantiles pi0.5 normalises with ((x - q01) / (q99 - q01): openpi sets quantile
  normalisation for every model but pi0), each within 5% of the dimension's q99 - q01, and q01 never above
  q99. A file openpi computed differs from these every-frame statistics in two ways: its histogram
  estimate of the quantiles (a bin of max - min over 5,000), and the frames it reads (the chunks its
  loader draws: tail-10 anchors, 15-step chunks padded at episode ends). The sampling moves the
  statistics more than the histogram does; both stay within the tolerances on the FR3 finetune set,
  which are still far narrower than a quantile of another dimension, another dataset or another scale.
Malformed content (a statistic that is not a list of numbers of the dataset's width, a table that is a
list) is a named problem, never a traceback.

Offline tier (numpy).
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

__all__ = ["check_norm"]

KEYS = ("mean", "std", "q01", "q99")
TOL_MEAN, TOL_STD, TOL_QUANTILE = 0.05, 0.10, 0.05


def _number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)


def _vectors(name: str, s: object, width: int) -> tuple[dict[str, np.ndarray] | None, list[str]]:
    """The four statistics of one table as arrays, or None and the problems that stop the comparison."""
    if s is None:
        return None, [f"{name}: no statistics"]
    if not isinstance(s, Mapping):
        return None, [f"{name}: the statistics are a {type(s).__name__}, not a table"]
    numeric = {k: s[k] for k in KEYS if isinstance(s.get(k), list) and all(_number(e) for e in s[k])}
    problems = []
    widths = sorted({len(v) for v in numeric.values()} - {width})
    if widths:      # one line: the per-statistic lines below would only repeat the width
        return None, [f"{name}: statistics of width {', '.join(map(str, widths))}, the dataset has {width}"]
    problems += [f"{name}: {k} is not a list of {width} numbers" for k in KEYS
                 if k not in numeric or len(numeric[k]) != width]
    if problems:
        return None, problems
    return {k: np.asarray(numeric[k], np.float64) for k in KEYS}, []


def _against(name: str, v: dict[str, np.ndarray], x: np.ndarray) -> list[str]:
    problems = []
    sd = np.maximum(x.std(0), 1e-6)
    if np.any(np.abs(v["mean"] - x.mean(0)) > TOL_MEAN * sd) or np.any(np.abs(v["std"] - x.std(0)) > TOL_STD * sd):
        problems.append(f"{name}: statistics were not computed on this dataset in the model's order")
    lo, hi = np.quantile(x, 0.01, axis=0), np.quantile(x, 0.99, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    above = np.flatnonzero(v["q01"] > v["q99"])
    if above.size:
        problems.append(f"{name}: q01 above q99 in dims {above.tolist()}")
    for k, want in (("q01", lo), ("q99", hi)):
        off = np.flatnonzero(np.abs(v[k] - want) > TOL_QUANTILE * span)
        if off.size:
            problems.append(f"{name}: {k} is not this dataset's in dims {off.tolist()} (more than "
                            f"{TOL_QUANTILE:.0%} of q99 - q01 from it)")
    return problems


def check_norm(norm_stats_path: Path, dataset_dir: Path, embodiment: str = "fr3") -> list[str]:
    from axis_qvla.realrobot.fr3.norm import columns

    try:
        doc = json.loads(Path(norm_stats_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"{norm_stats_path}: cannot read norm statistics ({exc.__class__.__name__}: {exc})"]
    if not isinstance(doc, Mapping):
        return [f"{norm_stats_path}: cannot read norm statistics (the document is a {type(doc).__name__}, "
                f"not a table)"]
    stats = doc.get("norm_stats")
    if not isinstance(stats, Mapping):
        what = "no norm_stats" if stats is None else f"norm_stats is a {type(stats).__name__}, not a table"
        return [f"{norm_stats_path}: cannot read norm statistics ({what})"]
    try:
        state, actions = columns(dataset_dir, embodiment)
    except ValueError as exc:
        return [str(exc)]
    problems = []
    for name, x in (("state", state), ("actions", actions)):
        v, bad = _vectors(name, stats.get(name), x.shape[1])
        problems += bad if v is None else _against(name, v, x)
    return problems
