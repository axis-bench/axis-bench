"""AWR built-ins: chunk weights from Eq. (1) or the approach-aware phase curve, episode weights
quantile-matched to them, and imported weights for consumers that read the flat view.

`AwrChunk` scores every corpus episode (axis_qvla.labels.corpus.CorpusLabels). The object-centric scorer
reads the stored reward; the approach-aware scorer builds the two-phase curve of the bound object
and raises Unscorable when it cannot (no stable contact or approach anchor, or the bound object's
pose column is missing); `on_unscorable="neutral"` then gives the episode weight 1.0 everywhere, as
the paper did, and the bundle counts it (`n_unscored_episodes`). Delta is a forward difference over
the horizon; for the approach-aware reward the windows are clamped to their phase, length-normalised,
and each phase's Delta is rescaled so both phases cap at the pooled rate
(axis_qvla.signals.methods.awr_chunk.phase_balance) before w = min(exp(tau * Delta), cap); a row whose
window touches an invalid reward row gets 1.0. `AwrEpisode` gives each episode one weight: its
P-ratio rank, quantile-matched to the chunk weights' distribution at the episode's cumulative
row-mass midpoint (axis_qvla.signals.methods.awr_episode.pratio_marginal_weights). `AwrImported` takes
weights someone already computed. Every AWR bundle draws BC's plan.

Offline tier: numpy, pandas.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from axis_qvla.labels.corpus import CorpusLabels
from axis_qvla.signals.blocks.awr import DEFAULT_CAP, DEFAULT_HORIZON, DEFAULT_TAU, episode_key, manifest_payload
from axis_qvla.signals.compile import compile_program
from axis_qvla.signals.exports import legacy_exports
from axis_qvla.signals.imported import ImportedScores
from axis_qvla.signals.methods.awr_chunk import chunk_weights, phase_reward
from axis_qvla.signals.methods.awr_episode import check_source, episode_document, pratio_marginal_weights
from axis_qvla.signals.methods.filtering import check_rows
from axis_qvla.signals.planners import PermutationPlanner
from axis_qvla.signals.program import Compiled, EpisodeValues, RefusedDtype, Unscorable
from axis_qvla.signals.rowspace import ranges_mapping

_POSE_COLUMNS = ("observation.body_pose", "observation.eef_pose")


class StoredReward:
    """Scorer (object-centric): the corpus's stored per-frame reward, Eq. (1)."""

    def curve(self, key, frames, meta_row):
        return frames["reward"].to_numpy(float), None


class PhaseReward:
    """Scorer (approach-aware): the two-phase curve of the bound object; raises Unscorable.

    The pose columns must be float32, as the corpus stores them (list<float>): the hand-object
    distance, and so the contact proxy's tanh, is then float32, computed by the portable kernel
    the locked weights were written with (axis_qvla.signals.methods.awr_chunk). Any other dtype is
    refused, naming the column, rather than scored in another precision."""

    def curve(self, key, frames, meta_row):
        for col in _POSE_COLUMNS:
            dtypes = {np.asarray(v).dtype for v in frames[col].to_numpy()}
            if dtypes - {np.dtype(np.float32)}:
                raise RefusedDtype(f"{key}: column {col!r} holds {sorted(map(str, dtypes))}, not float32; the "
                                   f"approach-aware reward's distances are float32, as the pretraining data stores them")
        res = phase_reward(frames, meta_row)
        if res is None:
            raise Unscorable(key, "no stable target contact or approach anchor, or the bound object's "
                                  "pose column is missing")
        return res


def _weighted(convention, loss_reducer, rows, budget, seed, weights, payload, exports, name, inputs, extra=None):
    planner = PermutationPlanner(convention)
    legacy, arrays = legacy_exports(exports, rows, awr_doc=None if payload is None else (payload, name),
                                    weighted=True)
    w = np.concatenate([np.asarray(v, dtype=np.float64) for v in weights.values.values()])
    cap = float(weights.header.get("cap", DEFAULT_CAP))
    diagnostics = {"mean_weight": float(w.mean()), "frac_at_cap": float(np.mean(w >= cap - 1e-9)),
                   "frac_weight_one": float(np.mean(w == 1.0)), **(extra or {})}
    return Compiled(plan=planner.plan(rows, None, None, budget, seed), planner=planner,
                    produced={"plan": ("PermutationPlanner",), "weights": ("AwrWeight",)}, weights=weights,
                    reducer=loss_reducer, diagnostics=diagnostics, inputs=inputs, legacy=legacy, legacy_arrays=arrays)


def _values(payload) -> EpisodeValues:
    return EpisodeValues({k: np.asarray(v, dtype=np.float32) for k, v in payload["weights"].items()},
                         header={k: v for k, v in payload.items() if k != "weights"})


@dataclasses.dataclass(frozen=True, eq=False)
class AwrChunk:
    scorer: str                  # "v2" (object-centric) or "phase" (approach-aware)
    convention: str
    loss_reducer: str
    binding: str = "names"
    phase_balanced: bool = True
    tau: float = DEFAULT_TAU
    cap: float = DEFAULT_CAP
    horizon: int = DEFAULT_HORIZON
    on_unscorable: str = "neutral"
    exports: tuple = ()

    def __post_init__(self) -> None:
        if self.scorer not in ("v2", "phase"):
            raise ValueError(f"scorer must be 'v2' or 'phase', got {self.scorer!r}")
        if self.on_unscorable not in ("neutral", "raise"):
            raise ValueError(f"on_unscorable must be 'neutral' or 'raise', got {self.on_unscorable!r}")
        if self.scorer == "v2" and self.phase_balanced:
            raise ValueError("phase balancing applies to the approach-aware (phase) scorer only")
        if self.binding != "names":
            raise ValueError("the names binding is the only object binding a paper run used")

    def spec(self) -> dict:
        return {"method": "awr_chunk", "version": 1, "params": {
            "scorer": self.scorer, "binding": self.binding if self.scorer == "phase" else None,
            "phase_balanced": bool(self.phase_balanced), "tau": float(self.tau), "cap": float(self.cap),
            "horizon": int(self.horizon), "on_unscorable": self.on_unscorable, "planner": self.convention,
            "loss_reducer": self.loss_reducer}}

    def payload(self, labels: CorpusLabels, rows) -> tuple[dict, int]:
        """The weights document, key for key what axis_qvla/signals/methods/awr_chunk.py writes, and the
        number of episodes the scorer declared unscorable."""
        phase = self.scorer == "phase"
        scorer = PhaseReward() if phase else StoredReward()

        def curve_of(tid, ep, frames, meta_row):
            try:
                return scorer.curve(episode_key(tid, ep), frames, meta_row)
            except Unscorable:
                if self.on_unscorable == "raise":
                    raise
                return None

        eps, scales, unscored = chunk_weights(
            labels.tasks(rows), curve_of, names=labels.names(rows) if phase else None, reads_meta=phase,
            phase_balanced=self.phase_balanced, tau=float(self.tau), cap=float(self.cap), horizon=int(self.horizon))
        extra = {"binding": self.binding, "phase_balanced": bool(self.phase_balanced)} if phase else {}
        payload = manifest_payload(iter(eps), reward_id=self.scorer, tau=float(self.tau), cap=float(self.cap),
                                   horizon=int(self.horizon), corpus_fingerprint=rows.corpus_fingerprint, **extra)
        if scales is not None:
            payload["provenance"]["phase_balance"] = scales
        return payload, unscored

    def run(self, labels, rows, budget, seed) -> Compiled:
        if not isinstance(labels, CorpusLabels):
            raise TypeError("AwrChunk scores the pretraining data's frames: pass axis_qvla.labels.corpus.CorpusLabels")
        if self.scorer == "v2" and labels.names_table is not None:
            raise ValueError("the object-centric reward binds no object: a names table would be recorded in the "
                             "compiled labels' inputs without being read")
        payload, unscored = self.payload(labels, rows)
        return _weighted(self.convention, self.loss_reducer, rows, budget, seed, _values(payload), payload,
                         self.exports, f"awr_{self.scorer}", labels.digests(rows),
                         {"n_unscored_episodes": float(unscored)})

    def compile(self, labels, rows, budget, seed):
        return compile_program(self, labels, rows, budget, seed)[0]


@dataclasses.dataclass(frozen=True, eq=False)
class AwrEpisode:
    convention: str
    loss_reducer: str
    exports: tuple = ()

    def spec(self) -> dict:
        return {"method": "awr_episode", "version": 1, "params": {
            "score": "pratio_marginal", "planner": self.convention, "loss_reducer": self.loss_reducer}}

    def payload(self, labels: ImportedScores, rows) -> dict:
        """The episode-weights document, key for key what awr_episode.py writes (pratio_marginal)."""
        src = labels.payload
        check_source(src, labels.name)
        fingerprint = src["provenance"]["corpus_fingerprint"]
        if fingerprint != rows.corpus_fingerprint:
            raise ValueError(f"{labels.name} was built on other pretraining data ({fingerprint} != {rows.corpus_fingerprint})")
        ep_w = pratio_marginal_weights(src["weights"], cap=float(src["cap"]), ranges=ranges_mapping(rows))
        return episode_document(src, ep_w, score="pratio_marginal", source_sha256=labels.sha256)

    def run(self, labels, rows, budget, seed) -> Compiled:
        check_rows(labels.header, rows)
        payload = self.payload(labels, rows)
        return _weighted(self.convention, self.loss_reducer, rows, budget, seed, _values(payload), payload,
                         self.exports, "awr_ep", {labels.name: labels.sha256})

    def compile(self, labels, rows, budget, seed):
        return compile_program(self, labels, rows, budget, seed)[0]


@dataclasses.dataclass(frozen=True, eq=False)
class AwrImported:
    convention: str
    loss_reducer: str
    exports: tuple = ()

    def spec(self) -> dict:
        return {"method": "awr_imported", "version": 1, "params": {
            "planner": self.convention, "loss_reducer": self.loss_reducer}}

    def run(self, labels, rows, budget, seed) -> Compiled:
        check_rows(labels.header, rows)
        return _weighted(self.convention, self.loss_reducer, rows, budget, seed, labels.values(), None,
                         self.exports, "awr", {labels.name: labels.sha256})

    def compile(self, labels, rows, budget, seed):
        return compile_program(self, labels, rows, budget, seed)[0]
