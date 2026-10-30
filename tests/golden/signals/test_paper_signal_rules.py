"""Every quality-signal rule of the paper's Sec. IV, and the details it omits, on synthetic data.

Each test names the sentence it pins. The "UNDOCUMENTED" tests pin behaviour the locked signal
files depend on but the paper text does not state: per-phase scaling, palm contact, tie handling at both label
granularities, the end-of-episode rescale asymmetry, the effective tag-dropout rate, the float32
threshold behind Eq. (5)'s 1[Delta > 0], and the order episode filtering ranks in. They describe
what the code DOES -- the files the models trained on were written this way -- so a change here is
a change to the released signals, not a cleanup.

Offline tier: numpy/pandas only. No private data.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from axis_qvla.signals.methods import awr_chunk as baw
from axis_qvla.signals.methods import schedule as bis
from axis_qvla.signals.blocks.awr import awr_weight, build_episode_weights, forward_delta
from axis_qvla.signals.methods.awr_episode import pratio_marginal_weights
from axis_qvla.signals.blocks.schedule import (
    NEUTRAL_Q, delta_quality_bins, draw_schedule, quality_bins, random_fraction_mask,
    top_fraction_mask,
)
from axis_qvla.signals.blocks.tags import (
    DROP_COMPONENT, DROP_WHOLE, NO_TAG, NOT_TRAINABLE, TAGGED_FRACTION, dense_tags,
    two_level_dropout,
)

H = 10
NAMES = ["plate_0", "cup_0"]
META = {"body_names": NAMES, "object_body": "cup_0"}


def _frames(r, d, finger, eef=None) -> pd.DataFrame:
    """One episode: the object `cup_0` sits at the origin, the hand at distance `d` along x."""
    T = len(r)
    hand = np.zeros((T, 3)); hand[:, 0] = d
    obj = np.tile([0.0, 0, 0, 1, 0, 0, 0], (T, 1))
    other = np.tile([0.5, 0.5, 0, 1, 0, 0, 0], (T, 1))
    bp = np.concatenate([other, obj], axis=1)
    return pd.DataFrame({
        "reward": np.asarray(r, float), "reward_valid": np.ones(T, bool),
        "contact_target_finger": np.asarray(finger, bool),
        "contact_target_eef": np.zeros(T, bool) if eef is None else np.asarray(eef, bool),
        "observation.body_pose": list(bp),
        "observation.eef_pose": [np.r_[hand[t], 1.0, 0, 0, 0] for t in range(T)],
    })


def _two_phase(T=60, tc=30, d0=0.40, dc=0.05):
    d = np.concatenate([np.linspace(d0, dc, tc), np.full(T - tc, dc)])
    p = np.concatenate([np.zeros(tc), np.linspace(0.1, 0.9, T - tc)])
    c = 1.0 - np.tanh(5.0 * d)
    r = 0.8 * p + 0.2 * c                      # Eq. (1)
    tgt = np.zeros(T, bool); tgt[tc:] = True
    return r, d, p, tgt


# --- A. State-derived progress rewards -------------------------------------------------------

def test_eq1_eq3_constants():
    """'r_t = 0.8 p_t + 0.2 c_t, c_t = 1 - tanh(5 d_t)'; 'lambda = 0.5 and b = 0.1'; 'at least
    three consecutive ... contact frames'."""
    assert (baw.W_PROGRESS, baw.W_CONTACT, baw.K) == (0.8, 0.2, 5.0)
    assert (baw.LAMBDA, baw.BONUS, baw.MIN_RUN) == (0.5, 0.10, 3)


def test_eq2_isolates_object_progress_and_eq3_scores_the_two_phases():
    """Eq. (2) p_t = clip((r_t - 0.2 c_t)/0.8, 0, 1); Eq. (3): lambda*clip((d0-dt)/(d0-dtc)) before
    contact, lambda + b + (1-lambda-b) clip(p_t - p_tc) from contact on."""
    r, d, p_true, tgt = _two_phase()
    curve, t0 = baw.banded_phase_curve(r, d, tgt)
    assert t0 == 30
    c = 1.0 - np.tanh(5.0 * d)
    p = np.clip((r - 0.2 * c) / 0.8, 0, 1)
    np.testing.assert_allclose(p, p_true, atol=1e-12)
    np.testing.assert_allclose(curve[:30], 0.5 * np.clip((d[0] - d[:30]) / (d[0] - d[30]), 0, 1))
    np.testing.assert_allclose(curve[30:], 0.6 + 0.4 * np.clip(p[30:] - p[30], 0, 1))


def test_tc_is_the_first_run_of_three_contact_frames():
    """'t_c denote the first frame in a sequence of at least three consecutive ... contact frames'."""
    m = np.zeros(60, bool); m[10:12] = True; m[30:] = True    # a 2-frame blip, then a real run
    assert baw.first_contact(m, baw.MIN_RUN) == 30
    r, d, _, _ = _two_phase()
    assert baw.banded_phase_curve(r, d, m)[1] == 30


def test_UNDOCUMENTED_palm_contact_counts_as_contact():
    """The paper says finger-object contact; the code ORs the finger and the hand/palm (eef) target
    contact columns, so a palm-only grasp still sets t_c."""
    r, d, _, tgt = _two_phase()
    none = np.zeros(len(r), bool)
    palm = baw.phase_reward(_frames(r, d, finger=none, eef=tgt), META)
    finger = baw.phase_reward(_frames(r, d, finger=tgt, eef=none), META)
    assert palm[1] == finger[1] == 30
    np.testing.assert_array_equal(palm[0], finger[0])


def test_contact_at_the_start_uses_the_manipulation_branch_from_frame_zero():
    """'Episodes with contact at the start ... use the manipulation branch from the start'
    (UNDOCUMENTED threshold: t_c < 5)."""
    r, d, p, _ = _two_phase()
    tgt = np.zeros(len(r), bool); tgt[4:] = True
    curve, t0 = baw.banded_phase_curve(r, d, tgt)
    c = 1.0 - np.tanh(5.0 * d); pp = np.clip((r - 0.2 * c) / 0.8, 0, 1)
    assert t0 == 0
    np.testing.assert_allclose(curve, 0.6 + 0.4 * np.clip(pp - pp[0], 0, 1))
    tgt[:] = False; tgt[5:] = True        # t_c == 5 is mid-episode: two bands
    assert baw.banded_phase_curve(r, d, tgt)[1] == 5


def test_no_measurable_approach_uses_the_manipulation_branch():
    """'... or no measurable approach use the manipulation branch' (UNDOCUMENTED: d0 - d_tc < 0.02 m)."""
    r, _, _, tgt = _two_phase()
    d = np.full(len(r), 0.05); d[:30] = np.linspace(0.069, 0.05, 30)    # 1.9 cm of approach
    assert baw.banded_phase_curve(r, d, tgt)[1] == 0


@pytest.mark.parametrize("contact_from", [None, 56])
def test_no_or_tail_contact_scores_approach_by_the_closest_approach(contact_from):
    """'episodes with no contact or contact only at the tail use approach progress normalized by
    the closest approach reached' (UNDOCUMENTED tail threshold: t_c > T - 5)."""
    r, d, _, _ = _two_phase()
    tgt = np.zeros(len(r), bool)
    if contact_from is not None:
        tgt[contact_from:] = True
    curve, t0 = baw.banded_phase_curve(r, d, tgt)
    assert t0 == len(r)
    np.testing.assert_allclose(curve, 0.5 * np.clip((d[0] - d) / (d[0] - d.min()), 0, 1))


def test_unscored_episodes_return_none_and_train_at_weight_one():
    """'Episodes without a valid object assignment remain unscored' (and UNDOCUMENTED: so does an
    episode with neither contact nor 2 cm of approach). The builder writes them all-1.0."""
    r, d, _, _ = _two_phase()
    none = np.zeros(len(r), bool)
    assert baw.banded_phase_curve(r, np.full(len(r), 0.3), none) is None
    g = _frames(r, d, finger=none)
    for meta in ({"body_names": NAMES, "object_body": "bowl_0"}, {"body_names": None, "object_body": "cup_0"},
                 {"body_names": NAMES, "object_body": None}):
        assert baw.phase_reward(g, meta) is None


# --- B. From reward changes to quality labels -------------------------------------------------

def test_eq4_raw_change_over_ten_frames():
    """'Delta_raw_t = z_min(t+H-1, T-1) - z_t, H = 10'."""
    z = np.random.default_rng(0).random(37)
    t = np.arange(37)
    np.testing.assert_array_equal(forward_delta(z, H), z[np.minimum(t + 9, 36)] - z[t])


def test_windows_are_truncated_at_contact_rescaled_and_zeroed_below_three_intervals():
    """'windows that cross the contact boundary are truncated ... rescale the within-phase change
    by (H-1)/h_t; windows with h_t < 3 ... receive a score of zero'. On a curve linear in each
    phase every trusted window therefore reads 9 x the phase slope, and the +b jump is never read."""
    T, tc, a, m = 60, 30, 0.01, 0.004
    curve = np.r_[a * np.arange(tc), 0.6 + m * np.arange(T - tc)]
    delta, ok = baw.phase_delta_clamped(curve, tc, H)
    np.testing.assert_allclose(delta[:tc - 3], 9 * a)
    np.testing.assert_array_equal(delta[tc - 3:tc], 0.0)
    np.testing.assert_allclose(delta[tc:T - 3], 9 * m)
    np.testing.assert_array_equal(delta[T - 3:], 0.0)
    assert not ok[tc - 3:tc].any() and ok[:tc - 3].all()
    assert delta.max() < 0.1     # the contact bonus never enters a window


def test_UNDOCUMENTED_end_of_episode_rescale_is_approach_aware_only():
    """Approach-aware windows are rescaled at the END of the episode too; object-centric windows
    simply shrink there."""
    z = 0.01 * np.arange(40)
    oc = forward_delta(z, H)
    aa, _ = baw.phase_delta_clamped(z, 0, H)          # one all-manipulation segment
    assert oc[35] == pytest.approx(0.04) and aa[35] == pytest.approx(0.09)
    assert oc[30] == pytest.approx(0.09) and aa[30] == pytest.approx(0.09)


def test_invalid_reward_windows_score_zero():
    """'windows with ... invalid reward values receive a score of zero': weight exp(0) = 1."""
    z = 0.02 * np.arange(30.0)
    valid = np.ones(30, bool); valid[20] = False
    w = build_episode_weights(1, 0, z, valid, horizon=H).weights
    t = np.arange(30)
    touched = (t == 20) | (np.minimum(t + 9, 29) == 20)
    assert t[touched].tolist() == [11, 20] and (w[touched] == 1.0).all()
    assert (w[~touched & (t < 29)] > 1.0).all()
    # approach-aware: a window clamped at contact (t0 = 15) reads curve[14], not curve[20]
    nm = baw._neutralised_mask_clamped(valid, 15, H)
    assert nm[20] and not nm[11]


def _corpus(tmp_path, weights: dict[str, list[float]], lengths: dict[int, list[int]], ranges: dict):
    """A toy corpus the schedule walk can enumerate: meta/episodes + tasks, plus its weights file."""
    for tid, lens in lengths.items():
        meta = tmp_path / f"task_{tid}" / "meta"
        (meta / "episodes" / "chunk-000").mkdir(parents=True)
        starts = np.r_[0, np.cumsum(lens)[:-1]]
        pd.DataFrame({"episode_index": range(len(lens)), "dataset_from_index": starts,
                      "length": lens}).to_parquet(meta / "episodes" / "chunk-000" / "file-000.parquet")
        pd.DataFrame({"task": [f"task {tid}"]}).to_parquet(meta / "tasks.parquet")
    roots = tmp_path / "roots.json"
    roots.write_text(json.dumps({"version": 2, "base": ".",
                                 "roots": {str(t): f"task_{t}" for t in lengths}}))
    rng = tmp_path / "ranges.json"; rng.write_text(json.dumps(ranges))
    w = tmp_path / "w.json"
    w.write_text(json.dumps({"reward_id": "v2", "tau": 10.0, "cap": 2.0, "horizon": 10,
                             "n_rows": sum(map(len, weights.values())), "weights": weights}))
    return roots, rng, w


def test_eq5_episode_score_is_the_positive_share_of_retained_rows(tmp_path, monkeypatch):
    """'P_e = 1/|T_e| sum_{t in T_e} 1[Delta_t > 0]' over the RETAINED (non-idle) time steps."""
    weights = {"task_800--0": [2.0, 1.5, 1.0, 0.5, 2.0, 2.0], "task_800--1": [1.2, 0.9, 1.0, 1.1]}
    monkeypatch.delenv(bis.CORPUS_ROOT_ENV, raising=False)
    roots, rng, w = _corpus(tmp_path, weights, {800: [6, 4]},
                            {"task_800--0": [[0, 4]], "task_800--1": [[0, 4]]})
    _blocks, progress, unlab, *_ = bis.episode_rows(roots, rng, bis.episode_stats_from_weights(w), drop=False)
    np.testing.assert_allclose(progress, [2 / 4, 2 / 4])   # frames 4-5 of episode 0 are idle
    assert not unlab.any()


def test_UNDOCUMENTED_progress_test_is_on_the_float32_weight(tmp_path, monkeypatch):
    """Eq. (5)'s 1[Delta_t > 0] is evaluated as `w > 1` on the STORED float32 weight.
    exp(10 Delta) rounds to exactly 1.0f for Delta below 2^-24/10 (about 5.96e-9), so a
    row whose Delta is positive float noise counts as NO progress. On the locked approach-aware
    files 6,805 trainable rows have 0 < Delta < 5.96e-9 (noise in p_t - p_tc); an exact-zero test
    would move P_e on 1,133 episodes and the episode quintile on 252 of 5,000 (object-centric: 75
    rows, 4 quintiles). The P-ratio behind the episode weights reads the same mask."""
    edge = 2.0 ** -24 / 10
    w = awr_weight(np.array([1e-10, 0.99 * edge, 1.01 * edge, 1e-8, -2e-9]), 10.0, 2.0)
    assert w.dtype == np.float32
    assert w.tolist()[:2] == [1.0, 1.0] and (w[2:4] > 1.0).all() and w[4] == 1.0
    noise = float(awr_weight(np.array([1e-10]), 10.0, 2.0)[0])
    weights = {"task_800--0": [noise, 1.6, 0.8, noise], "task_800--1": [noise] * 4}
    monkeypatch.delenv(bis.CORPUS_ROOT_ENV, raising=False)
    roots, rng, wf = _corpus(tmp_path, weights, {800: [4, 4]},
                             {"task_800--0": [[0, 4]], "task_800--1": [[0, 4]]})
    stats = bis.episode_stats_from_weights(wf)
    _blocks, progress, unlab, *_ = bis.episode_rows(roots, rng, stats, drop=False)
    assert progress.tolist() == [0.25, 0.0]              # literal Eq. (5) would give 0.75 and 1.0
    assert unlab.tolist() == [False, True]               # all-noise episode: unscored, neutral tag
    assert stats["task_800--0"]["keep_mask"].tolist() == [True, True, False, True]   # w == 1.0 kept
    both = {**weights, "task_800--2": [1.6, 1.6, 0.8, 0.8]}
    ew = pratio_marginal_weights(both, cap=2.0, ranges={k: [[0, 4]] for k in both})
    assert ew["task_800--1"] == 1.0                      # P-ratio sees the same atom
    assert ew["task_800--0"] <= ew["task_800--2"]         # P = 1/4 ranks below P = 2/4


def test_episode_labels_are_equal_frequency_quintiles_of_pe():
    """'partition them into five equal-frequency bins ... labels q in {1..5}'; an all-neutral
    (unscored) episode gets the neutral tag and takes no rank."""
    pe = np.r_[np.arange(10) / 10.0, 0.0]
    unl = np.r_[np.zeros(10, bool), True]
    np.testing.assert_array_equal(quality_bins(pe, unl), [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, NEUTRAL_Q])


def test_UNDOCUMENTED_chunk_label_ties_break_by_corpus_order():
    """Chunk labels rank Delta corpus-wide; rows with EQUAL Delta (the w == 1 atom: 15.8% of the
    approach-aware rows) are split across bins by flat (task, episode, frame) order -- a stable
    argsort, not a tie-averaging rule. This was considered and kept deliberately, not left
    unexamined."""
    bins = delta_quality_bins([np.zeros(6), np.zeros(4)])
    np.testing.assert_array_equal(np.concatenate(bins), [1, 1, 2, 2, 3, 3, 4, 4, 5, 5])


def test_UNDOCUMENTED_episode_label_ties_break_by_corpus_order():
    """Episode labels rank P_e with a stable sort, so episodes with EQUAL P_e are split across a
    bin edge by (task, episode) corpus order, not averaged. The locked
    object-centric episode tags split 5 episodes at P_e = 0.911111 into 4 x Q4 and 1 x Q5 this way;
    the approach-aware ones have no tie on an edge."""
    pe = np.array([0.9] * 5 + [0.1, 0.2, 0.3, 0.4, 0.5])
    np.testing.assert_array_equal(quality_bins(pe, np.zeros(10, bool)), [3, 4, 4, 5, 5, 1, 1, 2, 2, 3])


def test_episode_labels_are_shared_by_every_retained_row():
    """'episode-level labels are derived from P_e and shared across all retained time steps'."""
    blocks = [np.array([0, 1, 2]), np.array([5, 6])]
    tag, _ = dense_tags(blocks, np.array([2, 5]), n_rows=8, seed=0)
    assert (tag[[3, 4, 7]] == NOT_TRAINABLE).all()
    for rows, q in zip(blocks, (2, 5)):
        assert set(tag[rows].tolist()) <= {q, NO_TAG}


# --- C. Weighting, filtering and conditioning ---------------------------------------------------

def test_awr_weight_is_min_exp_ten_delta_two():
    """'w_t = min(exp(10 Delta_t), 2)'."""
    d = np.array([-0.3, 0.0, 0.03, math.log(2) / 10, 0.5])
    np.testing.assert_array_equal(awr_weight(d, 10.0, 2.0), np.minimum(np.exp(10 * d), 2).astype(np.float32))


def _phase_corpus(tmp_path, n_ep=110, T=240):
    """A one-task corpus with two-phase episodes of varied approach/manipulation speed."""
    rng = np.random.default_rng(7)
    rows, metas = [], []
    for e in range(n_ep):
        tc = int(rng.integers(40, 201))
        d = np.r_[np.linspace(0.4, 0.05, tc) + rng.normal(0, 0.002, tc), np.full(T - tc, 0.05)]
        steps = rng.gamma(0.6, 1.0, T - tc)
        p = np.r_[np.zeros(tc), rng.uniform(0.3, 1) * np.cumsum(steps) / steps.sum()]
        r = 0.8 * p + 0.2 * (1 - np.tanh(5 * d))
        tgt = np.zeros(T, bool); tgt[tc:] = True
        g = _frames(r, d, finger=tgt)
        g["episode_index"] = e; g["frame_index"] = np.arange(T)
        rows.append(g); metas.append({"episode_index": e, "length": T, "attempt_id": 1000 + e,
                                      "body_names": NAMES, "object_body": "cup_0"})
    task = tmp_path / "corpus" / "task_900"
    (task / "data" / "chunk-000").mkdir(parents=True); (task / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    pd.concat(rows, ignore_index=True).to_parquet(task / "data" / "chunk-000" / "file-000.parquet")
    pd.DataFrame(metas).to_parquet(task / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    (tmp_path / "artifacts").mkdir()
    idx = tmp_path / "artifacts" / "roots.json"
    idx.write_text(json.dumps({"version": 2, "base": "../corpus", "roots": {"900": "task_900"}}))
    return idx, rows


def test_UNDOCUMENTED_per_phase_scaling_equalises_the_capped_rate(tmp_path, monkeypatch):
    """The approach-aware weights rescale Delta per phase BEFORE w = min(exp(10 Delta), 2): each
    phase's Delta is multiplied by ln(2)/10 over its own (1 - target) quantile, so both phases cap
    at the pooled rate (the locked file records scale1 0.9587 / scale2 1.0957)."""
    idx, rows = _phase_corpus(tmp_path)
    out = tmp_path / "w.json"
    monkeypatch.setattr("sys.argv", ["x", "--reward", "phase", "--binding", "names", "--phase-balanced",
                                     "--index", str(idx), "--out", str(out)])
    assert baw.main() == 0
    doc = json.loads(out.read_text())
    pb = doc["provenance"]["phase_balance"]
    assert pb["mode"] == "clamped-lennorm+quantile-match"
    assert doc["provenance"]["params"] == {"binding": "names", "cap": 2.0, "horizon": 10,
                                           "phase_balanced": True, "tau": 10.0}
    thr = math.log(2) / 10
    fit1, fit2 = [], []
    for e, g in enumerate(rows):
        curve, t0 = baw.phase_reward(g.drop(columns=["episode_index", "frame_index"]), META)
        d, ok = baw.phase_delta_clamped(curve, t0, H)
        d[:t0] *= pb["scale1"]; d[t0:] *= pb["scale2"]
        want = np.where(baw._neutralised_mask_clamped(np.ones(len(d), bool), t0, H), 1.0,
                        awr_weight(d, 10.0, 2.0)).astype(np.float32)
        np.testing.assert_array_equal(np.asarray(doc["weights"][f"task_900--{e}"], np.float32), want)
        fit1.append(d[:t0][ok[:t0]]); fit2.append(d[t0:][ok[t0:]])
    r1, r2 = (np.mean(np.concatenate(f) >= thr - 1e-12) for f in (fit1, fit2))
    assert pb["scale1"] != pytest.approx(1.0) and pb["scale2"] != pytest.approx(1.0)
    assert r1 == pytest.approx(pb["target_capped_rate"], abs=2e-3)
    assert r2 == pytest.approx(pb["target_capped_rate"], abs=2e-3)


def test_episode_weighting_maps_pe_rank_onto_the_chunk_weight_distribution():
    """'map the time step-weighted percentile rank of P_e to the empirical distribution of
    chunk-level weights': each episode takes the chunk-weight quantile at its row-mass midpoint
    (UNDOCUMENTED: equal P_e break by episode key)."""
    w = {"task_1--0": [2.0, 2.0, 0.5, 1.5], "task_1--1": [0.5, 0.5, 1.2, 0.8],
         "task_1--2": [1.1, 0.9, 1.3, 0.7], "task_1--3": [1.0] * 4}
    rng = {k: [[0, 4]] for k in w}
    out = pratio_marginal_weights(w, cap=2.0, ranges=rng)
    marg = np.sort(np.concatenate([w[k] for k in ("task_1--0", "task_1--1", "task_1--2")]))
    # P_e: ep1 = 1/4, ep2 = 2/4 (tie broken by key: none here), ep0 = 3/4; midpoints 2, 6, 10
    assert out["task_1--1"] == marg[2] and out["task_1--2"] == marg[6] and out["task_1--0"] == marg[10]
    assert out["task_1--3"] == 1.0            # unscored: stays neutral, outside the marginal
    tie = pratio_marginal_weights({"task_1--b": [2.0, 0.5], "task_1--a": [2.0, 0.5]}, cap=2.0,
                                  ranges={"task_1--a": [[0, 2]], "task_1--b": [[0, 2]]})
    assert tie["task_1--a"] <= tie["task_1--b"]


def test_top_half_filtering_by_chunk_ranks_rows_by_delta():
    """'Top-quality-50% filtering retains half ... Chunk-level filtering ranks time steps by Delta_t'."""
    w = [np.array([1.0, 2.0, 0.5]), np.array([1.5, 1.0, 1.9, 0.7])]
    keep = np.concatenate(top_fraction_mask(w, 0.5))
    # 4 of 7 rows: 2.0, 1.9, 1.5 and the FIRST of the two 1.0 rows (ties keep flat order)
    assert keep.tolist() == [True, True, False, True, False, True, False]


def test_episode_filtering_keeps_whole_episodes_and_cuts_the_boundary_one():
    """'episode-level filtering ranks episodes by P_e and retains the highest-quality episodes until
    the same time step budget is reached ... the boundary episode is partially retained'
    (UNDOCUMENTED: the boundary episode keeps its LEADING rows, in frame order)."""
    lens, wep = [4, 3, 5, 2], [1.2, 1.9, 1.5, 0.8]
    keep = top_fraction_mask([np.full(n, v) for n, v in zip(lens, wep)], 0.5)
    assert [int(k.sum()) for k in keep] == [0, 3, 4, 0]        # 7 of 14 rows: ep1 whole, ep2 cut
    assert keep[2].tolist() == [True, True, True, True, False]


def test_UNDOCUMENTED_episode_filtering_ranks_the_mapped_weight_in_corpus_order():
    """Episode filtering ranks the episode-constant MAPPED weight w_ep, not P_e itself. The
    mapping is monotone in P_e, so it never inverts the P_e order, but distinct P_e
    can land on one w_ep atom, and episodes on an atom keep corpus order. Here P(b) = 1/2 < P(c) =
    3/4 map to the same 2.0, so b (first in the corpus) is kept whole and c is the boundary
    episode. On the locked files no atom straddles the 50% edge: the kept sets equal a direct P_e
    rank (3,183 approach-aware, 3,207 object-centric whole episodes, one boundary episode each)."""
    w = {"task_1--0": [2.0, 0.5, 0.5, 0.5], "task_1--1": [2.0, 2.0, 0.5, 0.5],
         "task_1--2": [2.0, 2.0, 2.0, 0.5]}
    wep = pratio_marginal_weights(w, cap=2.0, ranges={k: [[0, 4]] for k in w})
    assert (wep["task_1--0"], wep["task_1--1"], wep["task_1--2"]) == (0.5, 2.0, 2.0)
    order = ["task_1--1", "task_1--2", "task_1--0"]           # corpus order puts b before c
    keep = top_fraction_mask([np.full(4, wep[k]) for k in order], 0.5)
    assert [k.tolist() for k in keep] == [[True] * 4, [True, True, False, False], [False] * 4]


def test_filtered_rows_repeat_to_match_the_plain_bc_updates():
    """'We repeat the selected data to match the number of training updates used by plain BC.'"""
    rows = [np.arange(0, 10), np.arange(20, 30)]
    sched = draw_schedule(rows, mode="drop_top", total_steps=10, batch=4, seed=0)
    counts = np.bincount(sched.ravel(), minlength=30)
    assert set(np.unique(sched)) == set(np.concatenate(rows))
    assert counts[np.concatenate(rows)].tolist() == [2] * 20


def test_random_controls_keep_the_same_number_of_rows():
    """Fig. 4(c): random selection 'matching retained time steps and training updates'."""
    w = [np.random.default_rng(1).random(n) + 0.5 for n in (7, 9, 4)]
    assert sum(m.sum() for m in random_fraction_mask(w, 0.5, 42)) == sum(m.sum() for m in top_fraction_mask(w, 0.5))


def test_UNDOCUMENTED_two_dropout_operations_compose_to_19_25_percent():
    """'two independent dropout operations with probabilities 0.15 and 0.05': with one metadata
    component both remove the same tag, so the untagged rate is 1 - 0.85 x 0.95 = 0.1925
    (the locked tag files realise 0.19239)."""
    assert (DROP_WHOLE, DROP_COMPONENT) == (0.15, 0.05)
    assert 1 - TAGGED_FRACTION == pytest.approx(0.1925)
    whole, comp = two_level_dropout(400_000, seed=0)
    assert whole.mean() == pytest.approx(0.15, abs=3e-3) and comp.mean() == pytest.approx(0.05, abs=2e-3)
    assert (whole | comp).mean() == pytest.approx(0.1925, abs=3e-3)
    _, meta = dense_tags([np.arange(400_000)], np.array([4]), n_rows=400_000, seed=0)
    assert meta["realized_tagged_fraction"] == pytest.approx(1 - (whole | comp).mean())
