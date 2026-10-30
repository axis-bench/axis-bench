"""The LeRobot files of a composed method, built from its compiled labels' own arrays.

SmolVLA and X-VLA train in the pinned LeRobot fork, which reads label files bound to its train
config, one sampling regime per run (src/lerobot/configs/train.py `validate`):
  - a draw schedule, `axis_schedule_path`: npz `rows` int64 [steps, global batch], replayed in file
    order (AxisScheduleSampler), and `meta` JSON whose `n_rows` must be the dataset's frame count and
    whose `mode` and `reward_id` must be the run's `axis_expected_mode` and `axis_expected_reward`.
    It trains alone: no loss weights, no quality tags;
  - a row set, `axis_rows_path`: npz `rows` int64, drawn by AxisRowSampler -- epoch e is
    `rows[torch.randperm(len(rows), seed=run seed * 100_003 + e)]`, the `lerobot-axisrowsampler`
    convention of axis_qvla.signals.planners -- with per-frame loss weights
    (`sample_weighting.extra_params.weights_path`, weighter `axis_flat`: npz `weights` float32 per
    frame, NaN off the plan) or quality tags (`axis_quality_path`: npz `tag` uint8 per frame, 0
    untagged and 255 never drawn, `prompts`, and `meta` with `reward_id`, `n_rows` and
    `bin_row_counts` keyed "1".."5"), not both. A tag q appends "\\nQuality: q" to the prompt.
So a method trains on SmolVLA and X-VLA exactly what it trains on π0.5 in two cases. Its plan is
the row sampler's own draw over every trainable frame for its seed (no Selector, and in practice
PermutationPlanner("lerobot-axisrowsampler")): it exports the row set, with its weights or its
tags, and the run's seed is the seed it was compiled with. Or it neither weighs nor tags: any plan
exports as the schedule. Tags also need one field whose codes are 1..5, each rendered as the fork
renders it. A method outside both cases carries no LeRobot files and trains on π0.5; `refusal`
reads why from its compiled labels. Each file names the method as its `reward_id`, and a schedule
its planner's mode (a filtering schedule's) or convention as its `mode`, which the run spec repeats
as `axis_expected_reward` and `axis_expected_mode`.

Offline tier: numpy.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from axis_qvla.signals import otree
from axis_qvla.signals.blocks.schedule import N_BINS
from axis_qvla.signals.planners import PermutationPlanner, PlanError
from axis_qvla.signals.program import Conditioning, Selection
from axis_qvla.signals.rowspace import trainable_rows
from axis_qvla.signals.schema import Budget, LegacyFile, RowSpace

ROW_SAMPLER = "lerobot-axisrowsampler"
MARKER = "\nQuality: "          # what the fork appends before a tag (lerobot.utils.axis_quality.PROMPT_MARKER)
FORMAT = "lerobot-legacy"


def _npz(path: str, members) -> LegacyFile:
    return LegacyFile(FORMAT, None, path, "npz", {"$o": [list(m) for m in members]})


# Why a method carries no LeRobot files (`refusal`), one sentence per case binding reports.
BOTH = ("loss weights and quality tags together: the pinned LeRobot fork trains one or the other (its "
        "configs/train.py refuses both in one run), so this method trains on pi0.5 only")
OPENPI_ORDER = ("loss weights or quality tags on openpi's draw order, which the LeRobot fork's row sampler does "
                "not draw: compile the method with PermutationPlanner('lerobot-axisrowsampler') and the run's seed")
NOT_DRAWN = ("loss weights or quality tags on a plan the LeRobot fork's row sampler does not draw (a selection, or "
             "another planner's order): the fork replays such a plan only as a draw schedule, which it trains "
             "without weights or tags")
TAG_TEXT = (f"tag text other than the LeRobot fork's: it trains one quality field whose codes 1..{N_BINS} it "
            f"renders as {MARKER!r} + the code")
NO_PROMPTS = "quality tags without the pretraining instructions they are appended to (the vocabulary lists none)"


def tag_problem(vocab, prompts) -> str | None:
    """Why the fork cannot train these codes, or None: one field, codes 1..N_BINS, the fork's text."""
    text = vocab.text[vocab.fields[0]] if len(vocab.fields) == 1 else None
    if text is None or any(not 1 <= c <= N_BINS or t != f"{MARKER}{c}" for c, t in text.items()):
        return TAG_TEXT
    return None if prompts else NO_PROMPTS


def _row_sampler_draw(plan, rows: RowSpace, budget: Budget, seed: int, convention: str) -> bool:
    """`plan` is `convention`'s draw over every trainable frame with `seed`."""
    try:
        return bool(np.array_equal(PermutationPlanner(convention).plan(rows, None, None, budget, seed), plan))
    except PlanError:
        return False


def refusal(bundle) -> str | None:
    """Why the fork cannot train `bundle` as compiled (the cases `lerobot_files` exports nothing for),
    read from the bundle's own arrays; None when nothing in it rules the fork out."""
    weighted, tagged = bundle.weights is not None, bundle.cond is not None
    if weighted and tagged:
        return BOTH
    if not (weighted or tagged):
        return None
    rows, budget, seed = bundle.rowspace, bundle.budget, int(bundle.seed)
    if not _row_sampler_draw(bundle.plan, rows, budget, seed, ROW_SAMPLER):
        return OPENPI_ORDER if _row_sampler_draw(bundle.plan, rows, budget, seed, "openpi-rowsampler") else NOT_DRAWN
    return tag_problem(bundle.vocab, bundle.vocab.prompts or rows.prompts) if tagged else None


def lerobot_files(method: str, rows: RowSpace, budget: Budget, seed: int, *, plan, planner,
                  selection: Selection | None, weighted: bool, conditioning: Conditioning | None
                  ) -> tuple[tuple[LegacyFile, ...], dict]:
    """The lerobot-legacy records of a composed method and the arrays they name (module docstring);
    `((), {})` when the fork cannot train it as compiled (`refusal` says why)."""
    convention = getattr(planner, "convention", None)
    if selection is None and _row_sampler_draw(np.asarray(plan), rows, budget, seed, ROW_SAMPLER):
        if weighted and conditioning is not None:
            return (), {}
        frames = trainable_rows(rows)
        records = [_npz(f"signals/lerobot_rows_{method}.npz", [("rows", {"$array": "legacy.rows"})])]
        arrays = {"rows": frames}
        if weighted:
            records.append(_npz(f"signals/lerobot_awr_flat_{method}.npz", [("weights", {"$array": "weights"})]))
        if conditioning is not None:
            prompts = conditioning.vocab.prompts or rows.prompts
            if tag_problem(conditioning.vocab, prompts) is not None:
                return (), {}
            tag = np.asarray(conditioning.dense)[:, 0].copy()
            meta = {"reward_id": method, "n_rows": int(rows.n_rows),
                    "bin_row_counts": {str(q): int((tag[frames] == q).sum()) for q in range(1, N_BINS + 1)}}
            records.append(_npz(f"signals/quality_{method}.npz", [
                ("tag", {"$array": "legacy.tags"}), ("prompts", {"$strings": [str(p) for p in prompts]}),
                ("meta", {"$json": otree.encode(meta)})]))
            arrays["tags"] = tag
        return tuple(records), arrays
    if weighted or conditioning is not None:
        return (), {}
    kept = rows.n_trainable if selection is None else selection.kept
    meta = {"mode": str(getattr(planner, "mode", None) or convention), "reward_id": method, "seed": int(seed),
            "total_steps": int(budget.steps), "batch": int(budget.global_batch), "n_rows": int(rows.n_rows),
            "keep_fraction": kept / rows.n_trainable}
    return (_npz(f"signals/lerobot_schedule_{method}.npz",
                 [("rows", {"$array": "plan"}), ("meta", {"$json": otree.encode(meta)})]),), {}


def record_meta(record: LegacyFile) -> Mapping | None:
    """The `meta` member of an npz legacy record, decoded, or None when it has none."""
    if record.codec != "npz":
        return None
    node = dict((k, v) for k, v in record.doc["$o"]).get("meta")
    return otree.decode(node["$json"]) if isinstance(node, Mapping) and "$json" in node else None
