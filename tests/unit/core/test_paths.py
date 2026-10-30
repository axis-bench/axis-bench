"""`rel`/`resolve` are the portability contract: only the relative half is ever stored.

If `rel()` silently accepted a path outside every root, an absolute path would leak into an
index row or run record and the artifact would be unreachable on any other machine -- the
exact failure that made `roots_relative.json` necessary. So it raises instead.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from axis_qvla.core.hosts import HostProfile
from axis_qvla.core.paths import Paths, PathOutsideRoots


def _host(**over) -> HostProfile:
    roots = {
        "corpus": Path("/srv/corpus"),
        "cache": Path("/srv/cache"),
        "scratch": Path("/srv/scratch"),
        "runs": Path("/srv/runs"),
    }
    roots.update(over.pop("roots", {}))
    return HostProfile(name="t", launcher="local", roots=roots, venvs={}, **over)


def test_root_properties():
    p = Paths(_host())
    assert p.corpus == Path("/srv/corpus")
    assert p.cache == Path("/srv/cache")
    assert p.runs == Path("/srv/runs")
    assert p.catalog == Path("/srv/cache/catalog")


def test_run_dir():
    p = Paths(_host())
    assert p.run_dir("2026-08-09_exp_arm_abc123") == Path("/srv/runs/2026-08-09_exp_arm_abc123")


def test_rel_produces_a_root_qualified_relative_path():
    p = Paths(_host())
    assert p.rel("/srv/corpus/derived224/task_1889") == "corpus/derived224/task_1889"


def test_resolve_is_the_inverse_of_rel():
    p = Paths(_host())
    original = Path("/srv/cache/catalog/base/7.parquet")
    assert p.resolve(p.rel(original)) == original


def test_rel_of_a_root_itself_is_just_the_root_name():
    """`relative_to(root)` on the root returns Path('.'), so the naive form stores
    "corpus/.". It round-trips, but it is a KEY: two spellings of one location is exactly
    the ambiguity `rel` exists to remove, and the next phase stores these."""
    p = Paths(_host())
    assert p.rel("/srv/corpus") == "corpus"
    assert p.resolve(p.rel("/srv/corpus")) == Path("/srv/corpus")


def test_rel_rejects_a_path_under_no_root():
    """This is the guard that stops an absolute path entering a stored record."""
    p = Paths(_host())
    with pytest.raises(PathOutsideRoots, match="elsewhere"):
        p.rel("/elsewhere/thing.parquet")


def test_rel_prefers_the_longest_matching_root():
    """Roots may nest; the most specific must win or the relative half is ambiguous."""
    p = Paths(_host(roots={"inner": Path("/srv/corpus/inner")}))
    assert p.rel("/srv/corpus/inner/a.txt") == "inner/a.txt"


def test_resolve_rejects_an_unknown_root_name():
    p = Paths(_host())
    with pytest.raises(PathOutsideRoots, match="nosuch"):
        p.resolve("nosuch/a.txt")


def test_resolve_rejects_a_tail_that_is_itself_absolute():
    """`Path(root) / "/etc/passwd"` silently DISCARDS root and resets to the absolute path.

    An empty component (`corpus//etc/passwd`) is the way in: it partitions to
    tail='/etc/passwd'. `Location` guards its own stored strings, but the hole is here, in
    the core tier, and `resolve` must not be able to produce a path outside the root it was
    handed a name for -- whatever the caller passes.
    """
    p = Paths(_host())
    with pytest.raises(PathOutsideRoots):
        p.resolve("corpus//etc/passwd")


def test_resolve_is_host_specific():
    """Same relative path, two machines, two absolute results -- the whole point."""
    a = Paths(_host())
    b = Paths(_host(roots={"corpus": Path("/" + "opt/other/corpus")}))
    rel = "corpus/derived224/task_1889"
    assert a.resolve(rel) == Path("/srv/corpus/derived224/task_1889")
    assert b.resolve(rel) == Path("/" + "opt/other/corpus/derived224/task_1889")
