"""Bind a derived artifact to the corpus AND the formula that produced it.

CORE-ish (stdlib + pandas only, offline tier).

THE FAILURE THIS PREVENTS, in three forms, each silent without it:

  1. **The corpus changes.** Episodes are re-rendered, the idle filter moves, an episode is added
     or dropped. `episode_key`-based weights still resolve -- the keys are unchanged -- so the run
     trains with weights computed against different frames and nothing errors.
  2. **The formula parameters change.** `tau`, `cap` or `horizon` is edited in the spec and the
     artifact is not rebuilt. The header still says `tau=10.0` while the experiment believes 5.0.
  3. **The formula CODE changes.** Same path, same filename, different maths. This is the worst of
     the three because nothing about the artifact looks different at all.

None of these is caught by "the file exists" or by the key-based join, and all three produce a run
that completes normally under the run's name.

THE MECHANISM. Two fingerprints, recorded at build time (`stamp`) and compared by whatever reads
the artifact:

  * `corpus_fingerprint` -- a hash of every episode's IDENTITY AND SIZE, `(task_id, episode_index,
    length, attempt_id)`. Length is what makes it catch a re-render: keys alone would not.
  * `formula_fingerprint` -- the reward id, its parameters, and a FORMULA_VERSION the author bumps
    when the maths changes. The version is deliberate rather than a hash of the source file,
    because a comment edit is not a formula change and an automatic hash would cry wolf until
    people learned to ignore it.

A reader refuses a mismatch rather than warning: a warning would be read once and then filtered out
(`axis_qvla.signals.methods.awr` refuses weights whose corpus fingerprint is not its frame index's).
"""

from __future__ import annotations

import hashlib
import json
import pathlib

# BUMP THIS when the weighting maths changes in a way that makes old artifacts wrong. Not for
# comments, refactors or logging -- for anything that would change the numbers.
FORMULA_VERSION = 1


def corpus_fingerprint(corpus: pathlib.Path) -> str:
    """A hash of every episode's identity and size, stable across machines and orderings.

    `(task_id, episode_index, length, attempt_id)` per episode, sorted. LENGTH is the part that
    matters: a re-rendered corpus keeps its keys and changes its frames, and a fingerprint over
    keys alone would call that unchanged.
    """
    import pandas as pd

    rows: list[tuple] = []
    for task_dir in sorted(pathlib.Path(corpus).glob("task_*")):
        try:
            tid = int(task_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        for part in sorted(task_dir.glob("meta/episodes/**/*.parquet")):
            df = pd.read_parquet(part, columns=["episode_index", "length", "attempt_id"])
            rows.extend((tid, int(r.episode_index), int(r.length), int(r.attempt_id))
                        for r in df.itertuples())
    rows.sort()
    h = hashlib.sha256()
    for r in rows:
        h.update(("%d:%d:%d:%d\n" % r).encode())
    return f"sha256:{h.hexdigest()[:32]}"


def formula_fingerprint(reward_id: str, **params) -> str:
    """A hash of the formula identity: its reward, its parameters, and FORMULA_VERSION.

    Sorted-key JSON rather than concatenated fields, so `tau=1, cap=22` and `tau=12, cap=2` cannot
    collide -- the same reasoning as the schedule hash in `build_index_schedule`.
    """
    payload = {"formula_version": FORMULA_VERSION, "reward_id": reward_id,
               "params": {k: params[k] for k in sorted(params)}}
    return "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]


def stamp(artifact: dict, *, corpus: pathlib.Path | None = None, fingerprint: str | None = None,
          reward_id: str, **params) -> dict:
    """Add both fingerprints to an artifact payload, in place. Pass the corpus directory, or its
    fingerprint when the caller already has it (a row space carries it)."""
    if (corpus is None) == (fingerprint is None):
        raise ValueError("pass exactly one of corpus= or fingerprint=")
    artifact["provenance"] = {
        "corpus_fingerprint": fingerprint if fingerprint is not None else corpus_fingerprint(corpus),
        "formula_fingerprint": formula_fingerprint(reward_id, **params),
        "formula_version": FORMULA_VERSION,
        "reward_id": reward_id,
        "params": {k: params[k] for k in sorted(params)},
    }
    return artifact

