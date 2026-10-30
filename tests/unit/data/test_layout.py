"""axis_qvla.data.layout: one local place for every published file and every file made from them."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from axis_qvla.data.layout import (LayoutError, base_weights_dest, checkpoint_dest, default_dest, local_dest,
                              norm_stats_dest)


@pytest.fixture
def paths(tmp_path):
    return SimpleNamespace(corpus=tmp_path / "corpus", cache=tmp_path / "cache")


def test_dataset_repositories_go_under_the_corpus_root(paths):
    assert default_dest("axisrobotics/qvla-v1-corpus", paths, "episodes") == paths.corpus / "qvla-v1-corpus" / "episodes"
    assert default_dest("axisrobotics/qvla-v1-bench", paths) == paths.corpus / "qvla-v1-bench"
    assert default_dest("qvla-v1-results", paths, "cells.json") == paths.corpus / "qvla-v1-results" / "cells.json"


def test_model_repositories_go_under_cache_checkpoints(paths):
    want = paths.cache / "checkpoints" / "qvla-v1-pi05" / "ft" / "x" / "13579"
    assert default_dest("axisrobotics/qvla-v1-pi05", paths, "ft/x/13579") == want
    assert checkpoint_dest("axisrobotics/qvla-v1-pi05@v1.0:ft/x/13579", paths) == want
    assert checkpoint_dest("axisrobotics/qvla-v1-pi05:ft/x/13579/", paths) == want


def test_the_base_weights_are_a_local_copy_not_a_published_repository(paths):
    """The bases come from their publishers (axis_qvla.data.bases) into one local copy every run reads."""
    assert base_weights_dest(paths) == paths.cache / "checkpoints" / "qvla-v1-base-weights"
    with pytest.raises(LayoutError, match="not a published v1.0 repository"):
        default_dest("axisrobotics/qvla-v1-base-weights", paths)


def test_norm_stats_come_from_the_repository_that_publishes_them(paths):
    assert norm_stats_dest("heldout24_qual_v2", paths) == \
        paths.corpus / "qvla-v1-bench" / "norm_stats" / "heldout24_qual_v2"
    assert norm_stats_dest("libero5k_v2_d8", paths) == \
        paths.corpus / "qvla-v1-corpus" / "norm_stats" / "libero5k_v2_d8"


def test_files_made_here_never_land_in_a_mirrored_repository(paths):
    b = local_dest("bundles", "ab" * 32, paths)
    assert b == paths.cache / "bundles" / ("ab" * 32)
    assert paths.corpus not in b.parents
    assert local_dest("derived", "ft_lerobot_merged", paths) == paths.cache / "derived" / "ft_lerobot_merged"


@pytest.mark.parametrize("call", [
    lambda p: default_dest("axisrobotics/other", p),
    lambda p: default_dest("someone/qvla-v1-bench", p),
    lambda p: default_dest("axisrobotics/qvla-v1-bench", p, "../x"),
    lambda p: default_dest("axisrobotics/qvla-v1-bench", p, "/abs"),
    lambda p: checkpoint_dest("axisrobotics/qvla-v1-pi05@v1.0", p),
    lambda p: checkpoint_dest("axisrobotics/qvla-v1-pi05@v1.0:/", p),
    lambda p: checkpoint_dest("axisrobotics/qvla-v1-pi05@v1.0://", p),
    lambda p: local_dest("elsewhere", "x", p),
    lambda p: norm_stats_dest("droid", p),
])
def test_anything_else_is_refused(paths, call):
    with pytest.raises(LayoutError):
        call(paths)
