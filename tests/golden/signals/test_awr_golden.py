"""The AWR built-ins regenerate their 8 locked files: the episode weights and the LeRobot flats from
the published chunk weights (--pretraining-data), the chunk weights themselves from the corpus (--episodes).

The chunk weights are `min(exp(tau * Delta), cap)` in float64, cast to float32. Delta is host-
independent (correctly rounded float64 arithmetic of host-independent curves), but `np.exp` is not:
numpy dispatches float64 exp to its own AVX-512 kernel where the CPU has one and to the C library
elsewhere. The chunk golden therefore also measures every float64 weight's distance to the nearest
float32 rounding midpoint: with none within EXP_MARGIN_ULPS, any exp whose result is within that
many ulps of this host's writes the same float32 weights, so the locked files do not depend on
which exp a host runs.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import pytest

from axis_qvla.signals.blocks import awr as blocks_awr

from axis_qvla.labels.corpus import CorpusLabels
from axis_qvla.signals.bundle import export_legacy
from axis_qvla.signals.imported import ImportedScores
from axis_qvla.signals.legacy import legacy_target
from axis_qvla.signals.methods.awr import AwrChunk, AwrEpisode, AwrImported
from axis_qvla.signals.schema import Budget

REPO = pathlib.Path(__file__).resolve().parents[3]
NAMES = REPO / "conf" / "signals" / "episode_body_names.json"
EXP_MARGIN_ULPS = 4


def _ulps_to_a_float32_midpoint(x: np.ndarray) -> np.ndarray:
    """For positive finite float64 `x`: the number of float64 steps from each value to the nearest
    float32 rounding midpoint, the point at which its float32 cast changes (both midpoints are exact
    in float64, and for positive doubles the bit patterns count the steps between them)."""
    f = x.astype(np.float32)
    f64 = f.astype(np.float64)
    hi = (f64 + np.nextafter(f, np.float32(np.inf)).astype(np.float64)) / 2
    lo = (f64 + np.nextafter(f, np.float32(0)).astype(np.float64)) / 2
    xi = x.view(np.int64)
    return np.minimum(np.abs(hi.view(np.int64) - xi), np.abs(xi - lo.view(np.int64)))


def test_the_midpoint_distance_counts_float64_steps():
    one = np.float64(1.0)
    mid = (one + np.float64(np.nextafter(np.float32(1.0), np.float32(2.0)))) / 2       # 1 + 2^-24
    x = np.array([mid, np.nextafter(mid, 2.0), np.nextafter(np.nextafter(mid, 0.0), 0.0), one, 2.0])
    assert _ulps_to_a_float32_midpoint(x).tolist() == [0, 1, 2, 2**28, 2**28]


@pytest.fixture
def exp_margin(monkeypatch) -> dict:
    """Records, for every float64 weight the build casts to float32, its distance in float64 ulps to
    the nearest float32 rounding midpoint; the weights returned are the real function's."""
    real, seen = blocks_awr.awr_weight, {"n": 0, "near": [], "min": None}

    def spy(delta, tau=blocks_awr.DEFAULT_TAU, cap=blocks_awr.DEFAULT_CAP):
        pre = np.minimum(np.exp(tau * np.asarray(delta, dtype=np.float64)), cap)
        out = real(delta, tau, cap)
        assert np.array_equal(out, pre.astype(np.float32))
        if pre.size:
            d = _ulps_to_a_float32_midpoint(pre)
            seen["n"] += int(pre.size)
            seen["min"] = int(d.min()) if seen["min"] is None else min(seen["min"], int(d.min()))
            seen["near"].extend(float(v).hex() for v in pre[d <= EXP_MARGIN_ULPS])
        return out

    monkeypatch.setattr(blocks_awr, "awr_weight", spy)
    return seen


def _rows(lock) -> dict:
    return {r["id"]: r for r in lock["files"]}


def _scores(pack, lock, lock_id) -> ImportedScores:
    row = _rows(lock)[lock_id]
    return ImportedScores.from_json(pack / row["path"], name=lock_id, expect_sha256=row["sha256"])


def _exported(bundle, fmt, out) -> dict:
    return {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in export_legacy(bundle, fmt, out)}


@pytest.mark.slow
@pytest.mark.parametrize("family", ["approach_aware", "object_centric"])
def test_episode_weights_regenerate_from_the_locked_chunk_weights(paper_rowspaces, signals_pack, lock, tmp_path, family):
    target = legacy_target(f"{family}.episode.awr_weights", "openpi-legacy")
    b = AwrEpisode("openpi-rowsampler", "global_ratio", exports=(target,)).compile(
        _scores(signals_pack, lock, f"{family}.chunk.awr_weights"), paper_rowspaces["libero5k_v2_d8"],
        Budget(20605, 64), 42)
    assert _exported(b, "openpi-legacy", tmp_path) == {target.path: target.sha256}


@pytest.mark.slow
@pytest.mark.parametrize("flat_id,weights_id", [
    ("lerobot.awr_flat_phase", "approach_aware.chunk.awr_weights"),
    ("lerobot.awr_flat_phase_ep", "approach_aware.episode.awr_weights"),
    ("lerobot.awr_flat_v2", "object_centric.chunk.awr_weights"),
    ("lerobot.awr_flat_v2_ep", "object_centric.episode.awr_weights"),
])
def test_imported_weights_regenerate_the_lerobot_flats(paper_rowspaces, signals_pack, lock, tmp_path, flat_id, weights_id):
    target = legacy_target(flat_id, "lerobot-legacy")
    b = AwrImported("lerobot-axisrowsampler", "per_rank_ratio", exports=(target,)).compile(
        _scores(signals_pack, lock, weights_id), paper_rowspaces["libero5k_v2_d8"], Budget(20621, 64), 0)
    assert _exported(b, "lerobot-legacy", tmp_path) == {target.path: target.sha256}


@pytest.mark.slow
@pytest.mark.parametrize("scorer,lock_id", [("v2", "object_centric.chunk.awr_weights"),
                                            ("phase", "approach_aware.chunk.awr_weights")])
def test_chunk_weights_regenerate_from_the_corpus(paper_rowspaces, corpus, lock, tmp_path, exp_margin, scorer, lock_id):
    spec = json.loads((REPO / "conf" / "signals" / "generators.json").read_text())
    labels = CorpusLabels(corpus, names_table=NAMES if scorer == "phase" else None,
                          names_sha256=spec["inputs"]["episode_names"]["reward_input_sha256"] if scorer == "phase" else None)
    target = legacy_target(lock_id, "openpi-legacy")
    b = AwrChunk(scorer, "openpi-rowsampler", "global_ratio", phase_balanced=scorer == "phase",
                 exports=(target,)).compile(labels, paper_rowspaces["libero5k_v2_d8"], Budget(20605, 64), 42)
    assert _exported(b, "openpi-legacy", tmp_path) == {target.path: target.sha256}
    print(f"{lock_id}: {exp_margin['n']} float64 weights cast to float32; {len(exp_margin['near'])} within "
          f"{EXP_MARGIN_ULPS} ulps of a float32 rounding midpoint; the nearest is {exp_margin['min']} ulps away")
    # every row is cast once, except the rows of the episodes the approach-aware scorer declared unscorable
    assert exp_margin["n"] > 0 and (exp_margin["n"] == b.rowspace.n_rows) == (b.diagnostics["n_unscored_episodes"] == 0)
    assert exp_margin["near"] == [], f"float64 weights within {EXP_MARGIN_ULPS} ulps of a float32 midpoint: " \
                                     f"{exp_margin['near'][:20]}"
