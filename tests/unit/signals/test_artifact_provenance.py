"""The two fingerprints a derived artifact is stamped with move when the corpus or the formula it
was built for moves, and only then."""

from __future__ import annotations

import pandas as pd
import pytest

from axis_qvla.signals.provenance import corpus_fingerprint, formula_fingerprint, stamp


def _corpus(root, *, lengths=(10, 20), attempts=(1, 2)):
    d = root / "task_800" / "meta" / "episodes" / "chunk-000"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"episode_index": list(range(len(lengths))),
                  "length": list(lengths),
                  "attempt_id": list(attempts)}).to_parquet(d / "file-000.parquet")
    return root


# ---- the corpus fingerprint ---------------------------------------------------------------------


def test_the_same_corpus_fingerprints_the_same(tmp_path):
    a = corpus_fingerprint(_corpus(tmp_path / "a"))
    b = corpus_fingerprint(_corpus(tmp_path / "b"))
    assert a == b


def test_a_RE_RENDERED_corpus_changes_the_fingerprint(tmp_path):
    """The case keys alone would miss: same episodes, same attempts, different frame counts.

    A weights artifact keyed by episode still resolves against this, which is exactly why the
    fingerprint covers LENGTH and not just identity.
    """
    a = corpus_fingerprint(_corpus(tmp_path / "a", lengths=(10, 20)))
    b = corpus_fingerprint(_corpus(tmp_path / "b", lengths=(10, 21)))
    assert a != b


def test_a_dropped_episode_changes_the_fingerprint(tmp_path):
    a = corpus_fingerprint(_corpus(tmp_path / "a", lengths=(10, 20), attempts=(1, 2)))
    b = corpus_fingerprint(_corpus(tmp_path / "b", lengths=(10,), attempts=(1,)))
    assert a != b


# ---- the formula fingerprint --------------------------------------------------------------------


def test_changing_a_parameter_changes_the_formula_fingerprint():
    assert formula_fingerprint("phase", tau=10.0) != formula_fingerprint("phase", tau=5.0)


def test_changing_the_reward_changes_the_formula_fingerprint():
    assert formula_fingerprint("phase", tau=10.0) != formula_fingerprint("v2", tau=10.0)


def test_parameters_cannot_collide_by_concatenation():
    """`tau=1, cap=22` and `tau=12, cap=2` concatenate identically; sorted-key JSON does not."""
    assert formula_fingerprint("p", tau=1, cap=22) != formula_fingerprint("p", tau=12, cap=2)


# ---- the stamp ----------------------------------------------------------------------------------


def test_stamp_takes_exactly_one_of_a_corpus_or_its_fingerprint(tmp_path):
    corpus = _corpus(tmp_path / "a")
    with pytest.raises(ValueError, match="exactly one of corpus= or fingerprint="):
        stamp({}, reward_id="phase", tau=10.0)
    with pytest.raises(ValueError, match="exactly one of corpus= or fingerprint="):
        stamp({}, corpus=corpus, fingerprint="sha256:" + "0" * 32, reward_id="phase", tau=10.0)
    assert stamp({}, fingerprint=corpus_fingerprint(corpus), reward_id="phase", tau=10.0) == \
        stamp({}, corpus=corpus, reward_id="phase", tau=10.0)
