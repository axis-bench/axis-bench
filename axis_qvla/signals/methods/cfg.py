"""CFG built-ins: quality tags at episode or chunk granularity, as prompt codes on BC's plan.

Episode tags: the global quintile (1..5) of each episode's P-ratio -- the share of its trainable
rows with w > 1 on the float32 AWR weight -- with episodes whose weights are all 1.0 at the neutral
3 (axis_qvla.signals.blocks.schedule.quality_bins). Chunk tags: the global quintile of each row's own
weight, ties split by corpus position (delta_quality_bins via `plateau="rank"`). Codes are dropped
(0, the bare prompt) by two independent draws, the whole condition at `drop_whole` and the field at
`drop_component` (0.15 and 0.05 in the paper, so 19.25% of rows go untagged). `drop_whole =
drop_component = 0` is pure conditioning: there is no unconditional branch, so no serve condition
guides. A code renders as "\\nQuality: <q>" appended to the instruction, the text both training
forks append. The plan is BC's: CFG changes what a row is conditioned on, never which rows are
drawn. Tags are computed over `tag_rows` (default: the plan's own row space), so a subset arm can
reuse the full corpus's tags, as the paper's scaling arms did.

Offline tier: numpy, pandas.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from axis_qvla.signals.blocks.awr import DEFAULT_HORIZON
from axis_qvla.signals.blocks.schedule import N_BINS, quality_bins
from axis_qvla.signals.blocks.tags import (DROP_COMPONENT, DROP_WHOLE, dense_tags, dense_tags_rowwise,
                                      q_given_task_entropy_ratio)
from axis_qvla.signals.bundle import rowspace_id
from axis_qvla.signals.compile import compile_program
from axis_qvla.signals.exports import legacy_exports
from axis_qvla.signals.methods.cfg_tags import entropy_units, quality_meta, row_quality_bins
from axis_qvla.signals.methods.filtering import check_rows, stats_of
from axis_qvla.signals.planners import PermutationPlanner
from axis_qvla.signals.program import Compiled, Conditioning
from axis_qvla.signals.numerics import log2_extended
from axis_qvla.signals.rowspace import block_task_ids, episode_rows, trainable_blocks
from axis_qvla.signals.schema import BARE, CondVocab, RowSpace, ServeCondition

FIELD = "quality"
PROMPT_MARKER = "\nQuality: "


@dataclasses.dataclass(frozen=True, eq=False)
class CfgTags:
    granularity: str                  # "episode" or "row" (chunk)
    convention: str
    tag_seed: int = 0
    drop_whole: float = DROP_WHOLE
    drop_component: float = DROP_COMPONENT
    plateau: str = "rank"
    serve_tag: int = 5
    tag_rows: RowSpace | None = None
    exports: tuple = ()

    def __post_init__(self) -> None:
        if self.granularity not in ("episode", "row"):
            raise ValueError(f"granularity must be 'episode' or 'row', got {self.granularity!r}")

    def spec(self) -> dict:
        return {"method": "cfg_tags", "version": 1, "params": {
            "granularity": self.granularity, "planner": self.convention, "tag_seed": int(self.tag_seed),
            "drop_whole": float(self.drop_whole), "drop_component": float(self.drop_component),
            "plateau": self.plateau, "serve_tag": int(self.serve_tag),
            "tag_rowspace": None if self.tag_rows is None else rowspace_id(self.tag_rows)}}

    def run(self, labels, rows, budget, seed) -> Compiled:
        tag_rows = self.tag_rows or rows
        if tag_rows.n_rows != rows.n_rows or tag_rows.corpus_fingerprint != rows.corpus_fingerprint:
            raise ValueError(f"tags over {tag_rows.name} ({tag_rows.n_rows} rows) cannot condition rows of "
                             f"{rows.name} ({rows.n_rows} rows): the two frame indexes index different "
                             f"pretraining data")
        header = labels.header
        reward_id = header.get("reward_id")
        if reward_id is None:
            raise ValueError(f"{labels.name} names no reward_id; the tags could not name their run")
        check_rows(header, tag_rows)
        blocks, progress, unlab, _n_trainable, n_rows, row_w = episode_rows(
            tag_rows, stats_of(labels.values(), labels.name), drop=False)
        q_ep = quality_bins(progress, unlab)
        task_ids = block_task_ids(trainable_blocks(tag_rows))
        rates = dict(drop_whole=self.drop_whole, drop_component=self.drop_component)
        row_bins = None
        if self.granularity == "row":
            row_bins = row_quality_bins(row_w, plateau=self.plateau)
            tag, dropout = dense_tags_rowwise(blocks, row_bins, n_rows=n_rows, seed=self.tag_seed, **rates)
        else:
            tag, dropout = dense_tags(blocks, q_ep, n_rows=n_rows, seed=self.tag_seed, **rates)
        q_units, task_units = entropy_units(blocks, q_ep, task_ids, granularity=self.granularity, row_bins=row_bins)
        planner = PermutationPlanner(self.convention)
        plan = planner.plan(rows, None, None, budget, seed)
        vocab = CondVocab((FIELD,), {FIELD: {q: f"{PROMPT_MARKER}{q}" for q in range(1, N_BINS + 1)}}, tag_rows.prompts)
        serve = (BARE, ServeCondition({FIELD: str(int(self.serve_tag))}, None, 1.0))
        meta_live = quality_meta(
            tag, list(tag_rows.prompts), reward_id=reward_id, seed=self.tag_seed,
            horizon=int(header.get("horizon", DEFAULT_HORIZON)), n_rows=n_rows, config_hash="",
            episode_rows=blocks, q_ep=q_ep, task_ids=task_ids, dropout_meta=dropout,
            granularity=self.granularity, row_bins=row_bins, plateau=self.plateau)
        legacy, arrays = legacy_exports(self.exports, rows, tags=(tag, dict(
            prompts=tag_rows.prompts, meta_live=meta_live, weights_sha256=labels.sha256, tag_rows=tag_rows,
            seed=self.tag_seed, granularity=self.granularity, plateau=self.plateau,
            ratio_units=(q_units, task_units))))
        diagnostics = {"tagged_fraction": float(dropout["realized_tagged_fraction"]),
                       "q_given_task_entropy_ratio": float(q_given_task_entropy_ratio(q_units, task_units,
                                                                                      log2=log2_extended))}
        return Compiled(plan=plan, planner=planner,
                        produced={"plan": ("PermutationPlanner",), "cond": ("QualityTags",)},
                        conditioning=Conditioning(tag[:, None].astype(np.uint8), vocab, serve, dropout),
                        diagnostics=diagnostics, inputs={labels.name: labels.sha256}, legacy=legacy,
                        legacy_arrays=arrays)

    def compile(self, labels, rows, budget, seed):
        return compile_program(self, labels, rows, budget, seed)[0]
