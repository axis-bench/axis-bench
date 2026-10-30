"""Small valid row spaces and bundles, for this repository's tests and for a consumer's.

`toy_rowspace()`: two tasks over 23 rows -- task_800 with episodes of 6, 4 and 5 frames, task_1000
with one of 8 -- whose ranges keep 9 trainable rows, `TOY_TRAINABLE`. `toy_bundle()` is a 4 x 2
plan over them, optionally with loss weights and one conditioned field. Every value is fixed, so a
consumer's test can assert on it.

Offline tier: numpy.
"""
from __future__ import annotations

import hashlib

import numpy as np

from axis_qvla.core.canon import dumps_canonical, sha256_bytes
from axis_qvla.core.canonical import canonical_name
from axis_qvla.signals.schema import BARE, Budget, CondVocab, LegacyFile, RowSpace, ServeCondition, SignalBundle

TOY_TRAINABLE = (1, 2, 6, 7, 15, 16, 17, 21, 22)


def toy_rowspace() -> RowSpace:
    task = np.array([800, 800, 800, 1000], dtype=np.int64)
    ep = np.array([0, 1, 2, 0], dtype=np.int64)
    length = np.array([6, 4, 5, 8], dtype=np.int64)
    attempt = np.array([11, 12, 13, 21], dtype=np.int64)
    rows = sorted(zip(task.tolist(), ep.tolist(), length.tolist(), attempt.tolist()))
    fp = "sha256:" + hashlib.sha256("".join("%d:%d:%d:%d\n" % r for r in rows).encode()).hexdigest()[:32]
    return RowSpace(
        name="toy", kind="corpus", n_rows=23, n_trainable=9, corpus_fingerprint=fp,
        source=canonical_name("signals.testing"),
        roots={"$o": [["version", 2], ["base", "../corpus"],
                      ["roots", {"$o": [["1000", "task_1000"], ["800", "task_800"]]}]]},
        roots_path="index/roots_toy.json", roots_codec="json-indent2-nl",
        ranges_path="index/ranges_toy.json", ranges_codec="json",
        prompts=("open the drawer", "pick up the bowl"), idle_filter={"function": "toy"},
        arrays={"episode_task": task, "episode_index": ep,
                "episode_from": np.array([0, 6, 10, 15], dtype=np.int64), "episode_length": length,
                "episode_attempt": attempt,
                "ranges_entry": np.array([3, 0, 1], dtype=np.int64),
                "ranges_counts": np.array([2, 1, 1], dtype=np.int64),
                "ranges_spans": np.array([[0, 3], [6, 100], [1, 3], [0, 2]], dtype=np.int64)})


def toy_bundle(*, weights: bool = False, cond: bool = False) -> SignalBundle:
    rs = toy_rowspace()
    w = None
    if weights:
        w = np.full(23, np.nan, dtype=np.float32)
        w[list(TOY_TRAINABLE)] = np.array([2.0, 1.0, 1.5, 1.5, 1.0, 1.25, 0.75, 1.5, 2.0], dtype=np.float32)
    c, vocab, serve = None, None, (BARE,)
    if cond:
        c = np.array([[[5], [0]], [[3], [1]], [[2], [4]], [[0], [5]]], dtype=np.uint8)
        vocab = CondVocab(("quality",), {"quality": {q: f"\nQuality: {q}" for q in range(1, 6)}}, rs.prompts)
        serve = (BARE, ServeCondition({"quality": "5"}, None, 1.0))
    program = {"method": "toy", "version": 1, "params": {"weights": weights, "cond": cond}}
    requires = {"plan"} | ({"loss_weights"} if weights else set()) | ({"cond_prompt"} if cond else set())
    return SignalBundle(
        program=program, program_sha=sha256_bytes(dumps_canonical(program)), rowspace=rs,
        budget=Budget(4, 2), seed=0, plan=np.array([[1, 2], [6, 7], [15, 16], [17, 21]], dtype=np.int64),
        weights=w, reducer="global_ratio" if weights else "mean", cond=c, vocab=vocab, serve=serve,
        requires=frozenset(requires), diagnostics={"epochs": 8 / 9},
        legacy=(LegacyFile("lerobot-legacy", None, "signals/toy/lerobot_rows.npz", "npz",
                           {"$o": [["rows", {"$array": "legacy.rows"}]]}),),
        legacy_arrays={"rows": np.array(TOY_TRAINABLE, dtype=np.int64)})
