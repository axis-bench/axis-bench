"""Every signal-module name the REALALIGN tools call, by name and leading parameters (not bytes).

The REALALIGN tools import these names directly, so a rename or a reordered signature of one fails
here, before a build reaches it."""
from __future__ import annotations

import inspect

import pytest

from axis_qvla.signals import provenance
from axis_qvla.signals.blocks import awr, tags
from axis_qvla.signals.blocks import schedule as bsched
from axis_qvla.signals.methods import awr_chunk, awr_episode, cfg_tags, schedule

SEAMS = [
    (schedule, "main", ["argv"]), (schedule, "_ordered_roots", ["roots_path"]),
    (schedule, "_episode_offsets", ["root"]), (schedule, "_frames_in_spans", ["spans", "length"]),
    (schedule, "_file_digest", ["path"]), (schedule, "_unique_tmp_path", ["p"]),
    (schedule, "_weights_header", ["weights_path"]),
    (schedule, "episode_stats_from_weights", ["weights_path", "tau", "cap", "horizon"]),
    (schedule, "episode_rows", ["roots_path", "ranges_path", "stats", "drop"]),
    (awr_episode, "main", ["argv"]), (cfg_tags, "_generator_digest", []),
    (cfg_tags, "quality_meta", ["tag", "prompts", "reward_id", "seed", "horizon", "n_rows", "config_hash",
                                "episode_rows", "q_ep", "task_ids", "dropout_meta"]),
    (cfg_tags, "corpus_prompts", ["roots_path"]),
    (cfg_tags, "episode_task_ids", ["roots_path", "ranges_path", "stats", "block_lengths"]),
    (cfg_tags, "check_artifact_filename", ["path", "reward_id", "granularity"]),
    (awr_chunk, "phase_delta_clamped", ["curve", "t0", "horizon"]),
    (awr_chunk, "_neutralised_mask_clamped", ["valid", "t0", "horizon"]),
    (awr, "episode_key", ["task_id", "episode_index"]),
    (awr, "build_episode_weights", ["task_id", "episode_index", "reward", "valid", "horizon", "tau", "cap"]),
    (awr, "neutral_episode", ["task_id", "episode_index", "length"]),
    (awr, "write_manifest", ["path", "episodes"]), (awr, "awr_weight", ["delta", "tau", "cap"]),
    (bsched, "quality_bins", ["progress_ratio", "unlabelled"]), (provenance, "corpus_fingerprint", ["corpus"]),
]


@pytest.mark.parametrize("module, name, leading", SEAMS, ids=lambda x: getattr(x, "__name__", str(x)))
def test_the_name_exists_with_its_leading_parameters(module, name, leading):
    params = list(inspect.signature(getattr(module, name)).parameters)
    assert params[:len(leading)] == leading, f"{module.__name__}.{name}{tuple(params)}"


def test_the_constants_keep_their_values():
    assert (awr_chunk.CORPUS_ROOT_ENV, bsched.N_BINS, bsched.NEUTRAL_Q, tags.NO_TAG, tags.NOT_TRAINABLE) == (
        "AXIS_PRETRAIN_CORPUS_ROOT", 5, 3, 0, 255)
    # Directly-imported defaults (weights.py, tags.py): a rename fails at import for the whole suite,
    # which the name/signature check above does not exercise; a silent VALUE change would not fail
    # either, so it is pinned here too.
    assert (awr.DEFAULT_TAU, awr.DEFAULT_CAP, awr.DEFAULT_HORIZON) == (10.0, 2.0, 10)
