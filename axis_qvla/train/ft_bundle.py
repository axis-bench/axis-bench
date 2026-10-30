"""The pi0.5 finetune signal, compiled from the recipe when a finetune arm names no bundle.

The published signal bundles cover every pretraining run and the SmolVLA / X-VLA finetunes; a
pi0.5 finetune's plan and its constant quality tag are recipe fields. `qvla train` compiles them
here, when the arm is bound, into an ordinary SignalBundle, so the openpi line keeps one data path:

  plan  the rows openpi's loader draws: pass e is `rows[torch.randperm(n, manual_seed(seed + e))]`
        over the n trainable rows (axis_qvla.signals.planners, convention `openpi-rowsampler`), and the
        loader batches every pass with drop_last, so a pass whose n is not a multiple of the batch
        loses its last partial batch. The paper's 13,580 x 32 is exactly five passes over 86,912
        rows (86,912 = 2,716 x 32): nothing is dropped there.
  cond  with `data.quality_tag = q` (the paper's tagged finetunes: q = 5), code q on every sample
        except the ones the presentation-keyed dropout drops, which train on the bare prompt
        (code 0). A sample's presentation is the pass it was drawn in. `dropped` is the rule of
        the paper's openpi fork (`LiberoQualityConditioning.dropped`): two independent levels,
        0.15 and 0.05, drawn from a generator keyed on (seed 0, presentation, episode_index,
        frame_index), so the tagged marginal is 0.85 x 0.95 = 0.8075.

The finetune row space is flat: row r is the r-th frame of the fork's ranges walk over the
finetune corpus (FLAT_ROW_DECODE) -- tasks in task-id order (as integers), each task's episodes in
its meta/episodes order, the frames inside each episode's spans in file order, clamped to the
episode -- which is also where each row's episode and frame are read from. Which files those are
is recorded in the bundle (`inputs`: the sha256 of the roots index and of the ranges file), and the
decode rule in its program (`params.row_decode`), so a consumer can check both rather than assume
them.

Core tier at import; numpy, pandas and the signal compiler are imported inside the functions.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
from collections.abc import Mapping

from axis_qvla.core.canon import dumps_canonical, sha256_bytes

DROP_WHOLE = 0.15
DROP_COMPONENT = 0.05
TAG_SEED = 0
N_BINS = 5
PLANNER = "openpi-rowsampler"
PROGRAM_METHOD = "qvla.train.pi05_finetune"
PROGRAM_VERSION = 1
# How a flat finetune row maps onto the corpus, in the fork's terms; `inputs` names the two files.
FLAT_ROW_DECODE = "flat row r = the r-th row of plan_rows_from_roots(roots_index, ranges)"
INPUT_ROOTS = "roots_index"
INPUT_RANGES = "ranges"


class FinetuneBundleError(ValueError):
    """The finetune corpus does not describe the row space the arm names, or the tag is not a bin."""


def dropped(episode_index: int, frame_index: int, presentation: int, *, seed: int = TAG_SEED) -> bool:
    """Whether this presentation of this row trains on the bare prompt (the fork's rule, verbatim).

    Both uniforms are drawn every time, so the second level never shifts the first one's stream."""
    import numpy as np

    digest = hashlib.sha256(f"{int(seed)}:{int(presentation)}:{int(episode_index)}:{int(frame_index)}".encode()).digest()
    u = np.random.default_rng(int.from_bytes(digest[:8], "little")).random(2)
    return bool(u[0] < DROP_WHOLE or u[1] < DROP_COMPONENT)


def _parquet_files(directory: pathlib.Path) -> list[str]:
    """Every `*.parquet` under `directory` at any depth, sorted (what `glob(dir/**/*.parquet)`
    lists), by os.scandir so an unreadable directory is refused by name, never read as empty."""
    out, todo = [], [directory]
    while todo:
        d = todo.pop()
        try:
            with os.scandir(d) as it:
                entries = [e for e in it if not e.name.startswith(".")]
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise FinetuneBundleError(f"cannot read {d} ({exc.__class__.__name__})") from None
        for e in entries:
            if e.is_dir():
                todo.append(pathlib.Path(e.path))
            elif e.name.endswith(".parquet"):
                out.append(e.path)
    return sorted(out)


def _task_episodes(task: pathlib.Path) -> list[tuple[int, int, bool]]:
    """`(episode_index, length, placed)` of one task, in its meta/episodes order. `placed` says the
    episode's `dataset_from_index` is where its rows actually sit (the lengths of the episodes
    before it): only then is the row the loader reads at that offset this episode's frame."""
    import pandas as pd

    files = _parquet_files(task / "meta" / "episodes")
    if not files:
        raise FinetuneBundleError(f"no meta/episodes parquet under {task}")
    ep = pd.concat([pd.read_parquet(f, columns=["episode_index", "dataset_from_index", "length"])
                    for f in files], ignore_index=True)
    out, offset = [], 0
    for e, start, n in zip(ep["episode_index"], ep["dataset_from_index"], ep["length"], strict=True):
        out.append((int(e), int(n), int(start) == offset))
        offset += int(n)
    return out


def _roots_mapping(roots_index: pathlib.Path, corpus_base: pathlib.Path) -> dict:
    """The roots index's task -> directory mapping. A v2 index records its own `base` (relative to
    the index); it must be `corpus_base`, or the index means another corpus than the one read."""
    doc = json.loads(roots_index.read_text(encoding="utf-8"))
    if isinstance(doc, dict) and isinstance(doc.get("roots"), dict):
        if "base" in doc:
            declared = (roots_index.parent / str(doc["base"])).resolve()
            if declared != corpus_base.resolve():
                raise FinetuneBundleError(f"{roots_index} records base {doc['base']!r} ({declared}), not the "
                                          f"episodes directory {corpus_base}")
        doc = doc["roots"]
    if not isinstance(doc, dict) or not doc:
        raise FinetuneBundleError(f"{roots_index} names no task directories")
    return doc


def episode_table(roots_index, corpus_base, ranges=None):
    """`(episode_index, frame_index)` of every row, and the corpus' prompts (module docstring).

    `roots_index` maps task ids to task directories relative to `corpus_base` (a bare mapping, or
    a v2 index whose `roots` it is). With `ranges` (a ranges file: `task_<id>--<episode>` -> half-open
    spans) the rows are the frames inside the spans; without it, every frame of every episode."""
    import numpy as np

    from axis_qvla.signals.blocks.awr import episode_key
    from axis_qvla.signals.methods.cfg_tags import task_prompts

    mapping = _roots_mapping(pathlib.Path(roots_index), pathlib.Path(corpus_base))
    spans = None if ranges is None else json.loads(pathlib.Path(ranges).read_text(encoding="utf-8"))
    eps, frames, prompts, used = [], [], set(), 0
    for tid, rel in sorted(mapping.items(), key=lambda kv: int(kv[0])):
        task = pathlib.Path(corpus_base) / str(rel)
        for e, n, placed in _task_episodes(task):
            if spans is None:
                f = np.arange(n, dtype=np.int64)
            else:
                cut = spans.get(episode_key(int(tid), e))
                used += cut is not None
                f = np.concatenate([np.empty(0, dtype=np.int64)] + [
                    np.arange(max(0, int(s)), min(int(t), n), dtype=np.int64) for s, t in (cut or ())])
            if f.size and not placed:
                # An episode with no row in the table may be misplaced (a finetune task can carry
                # episodes whose recorded offset skips ahead of their rows); one with rows may not.
                raise FinetuneBundleError(f"{task}: episode {e}'s dataset_from_index is not where its rows "
                                          f"sit (the lengths of the episodes before it), so the frame the "
                                          f"loader reads for it is not its own")
            eps.append(np.full(f.size, e, dtype=np.int64))
            frames.append(f)
        prompts |= task_prompts(task)
    if spans is not None and used != len(spans):
        raise FinetuneBundleError(f"{ranges}: {len(spans) - used} of its {len(spans)} episodes are not in the "
                                  f"pretraining data {roots_index} names")
    return np.concatenate(eps), np.concatenate(frames), tuple(sorted(prompts))


def presentations(n_rows: int, steps: int, batch: int):
    """The pass each sample of a steps x batch plan over `n_rows` drawable rows was drawn in. The
    loader drops a pass's last partial batch, so once a plan spans more than one pass each pass
    contributes only its whole batches (axis_qvla.signals.planners.PermutationPlanner.plan, whose
    `DROP_LAST` this reads; the planner returns the plan only, so `compile_finetune` checks each
    pass against the planner's permutation)."""
    import numpy as np

    from axis_qvla.signals.planners import DROP_LAST

    n, total = int(n_rows), int(steps) * int(batch)
    per_pass = n if (total <= n or n % batch == 0 or not DROP_LAST[PLANNER]) else n - n % batch
    return np.arange(total, dtype=np.int64) // per_pass


def _check_passes(flat, pres, rows, seed: int) -> None:
    """Every pass p of the plan is the start of `rows[torch_randperm(n, seed_p)]`: the pass index
    given to the dropout is the pass the planner actually drew the sample in."""
    import numpy as np

    from axis_qvla.signals.planners import CONVENTIONS, torch_randperm

    bounds = np.flatnonzero(np.diff(pres)) + 1
    for p, block in enumerate(np.split(flat, bounds)):
        want = rows[torch_randperm(rows.size, CONVENTIONS[PLANNER](int(seed), p))[:block.size]]
        if not np.array_equal(block, want):
            raise FinetuneBundleError(f"pass {p} of the plan is not the planner's pass {p}: the presentation "
                                      f"keying the dropout would be wrong")


def program(*, quality_tag: int | None, row_decode: str = FLAT_ROW_DECODE) -> dict:
    return {"method": PROGRAM_METHOD, "version": PROGRAM_VERSION,
            "params": {"planner": PLANNER, "quality_tag": quality_tag, "drop_whole": DROP_WHOLE,
                       "drop_component": DROP_COMPONENT, "tag_seed": TAG_SEED, "row_decode": row_decode}}


def compile_finetune(rowspace, *, episode_index, frame_index, prompts, steps: int, batch: int, seed: int,
                     quality_tag: int | None, inputs: Mapping[str, str] | None = None):
    """The SignalBundle of one pi0.5 finetune run (module docstring). `episode_index` and
    `frame_index` hold one entry per row of `rowspace`; `inputs` names the digests of the files
    they were read from (INPUT_ROOTS, INPUT_RANGES)."""
    import numpy as np

    from axis_qvla.signals.planners import PermutationPlanner
    from axis_qvla.signals.rowspace import trainable_rows
    from axis_qvla.signals.schema import BARE, Budget, CondVocab, ServeCondition, SignalBundle

    q = quality_tag
    if q is not None and (isinstance(q, bool) or not isinstance(q, int) or not 1 <= q <= N_BINS):
        raise FinetuneBundleError(f"quality tag {q!r} is not a bin in 1..{N_BINS}; leave data.quality_tag "
                                  f"unset for an untagged finetune")
    if q is not None and not prompts:
        raise FinetuneBundleError("a tagged finetune needs the pretraining data's task prompts: its token "
                                  "budget is checked against them")
    if len(episode_index) != rowspace.n_rows or len(frame_index) != rowspace.n_rows:
        raise FinetuneBundleError(f"the finetune episode table has {len(episode_index)} rows; frame index "
                                  f"{rowspace.name} has {rowspace.n_rows}")
    budget = Budget(int(steps), int(batch))
    plan = PermutationPlanner(PLANNER).plan(rowspace, None, None, budget, int(seed))
    cond = vocab = None
    serve, requires, diagnostics = (BARE,), {"plan"}, {}
    if q is not None:
        rows = trainable_rows(rowspace)
        flat = plan.reshape(-1)
        pres = presentations(int(rows.size), budget.steps, budget.global_batch)
        _check_passes(flat, pres, rows, seed)
        ep = np.asarray(episode_index, dtype=np.int64)[flat].tolist()
        fr = np.asarray(frame_index, dtype=np.int64)[flat].tolist()
        keep = np.fromiter((not dropped(e, f, p) for e, f, p in zip(ep, fr, pres.tolist(), strict=True)),
                           dtype=bool, count=flat.size)
        cond = np.where(keep, q, 0).astype(np.uint8).reshape(budget.steps, budget.global_batch, 1)
        vocab = CondVocab(("quality",), {"quality": {q: f"\nQuality: {q}"}}, tuple(prompts))
        serve, requires = (ServeCondition({"quality": str(q)}, None, 1.0),), {"plan", "cond_prompt"}
        diagnostics = {"tagged_fraction": float(keep.mean())}
    prog = program(quality_tag=q, row_decode=FLAT_ROW_DECODE if rowspace.kind == "flat" else "row space row r")
    return SignalBundle(program=prog, program_sha=sha256_bytes(dumps_canonical(prog)), rowspace=rowspace,
                        budget=budget, seed=int(seed), plan=plan, cond=cond, vocab=vocab, serve=serve,
                        requires=frozenset(requires), diagnostics=diagnostics, inputs=dict(inputs or {}))
